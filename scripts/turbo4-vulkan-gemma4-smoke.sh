#!/usr/bin/env bash
# Opt-in Gemma4 Vulkan smoke test for TurboQuant K/V cache.
#
# This is intentionally not a default CI test: it needs a large Gemma4 GGUF,
# a Vulkan-capable host, and enough VRAM/system memory to launch llama-server.
#
# Required:
#   GEMMA4_MODEL=/path/to/gemma-4-26B-A4B-it-Q4_K_M.gguf
#
# Optional:
#   LLAMA_SERVER=/path/to/llama-server
#   VULKAN_DEVICE=Vulkan0
#   CHAT_TEMPLATE_FILE=/path/to/google-gemma-4-31B-it-interleaved.fixed.jinja
#   HOST=127.0.0.1 PORT=18085 CTX_SIZE=65537 BATCH_SIZE=512 UBATCH_SIZE=512
#   GPU_LAYERS=999 GPU_LAYERS_DRAFT=999 N_CPU_MOE=38 N_CPU_MOE_DRAFT=38
#   NO_MMPROJ=1 LOG_FILE=/tmp/llama-turbo4-vulkan-gemma4-smoke.log
#   PROMPT='Reply with exactly: turbo4v-ok' EXPECT='turbo4v-ok'
#   EXTRA_ARGS='--some-llama-server-flag value'

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_SERVER="${LLAMA_SERVER:-}"
if [[ -z "$LLAMA_SERVER" ]]; then
    for candidate in \
        "$ROOT_DIR/build/bin/llama-server" \
        "$ROOT_DIR/build-vulkan/bin/llama-server" \
        "$ROOT_DIR/build/bin/server" \
        "$ROOT_DIR/build-vulkan/bin/server"; do
        if [[ -x "$candidate" ]]; then
            LLAMA_SERVER="$candidate"
            break
        fi
    done
fi

MODEL="${GEMMA4_MODEL:-${MODEL:-}}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18085}"
CTX_SIZE="${CTX_SIZE:-65537}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
GPU_LAYERS="${GPU_LAYERS:-999}"
GPU_LAYERS_DRAFT="${GPU_LAYERS_DRAFT:-999}"
N_CPU_MOE="${N_CPU_MOE:-38}"
N_CPU_MOE_DRAFT="${N_CPU_MOE_DRAFT:-38}"
NO_MMPROJ="${NO_MMPROJ:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
PROMPT="${PROMPT:-Reply with exactly: turbo4v-ok}"
EXPECT="${EXPECT:-turbo4v-ok}"
LOG_FILE="${LOG_FILE:-${TMPDIR:-/tmp}/llama-turbo4-vulkan-gemma4-smoke.log}"
REQUIRE_LOG_MARKERS="${REQUIRE_LOG_MARKERS:-1}"
VULKAN_DEVICE="${VULKAN_DEVICE:-Vulkan0}"

if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
    echo "SKIP: set LLAMA_SERVER or build llama-server before running this smoke test."
    exit 0
fi

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
    echo "SKIP: set GEMMA4_MODEL to a Gemma4 GGUF before running this smoke test."
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "SKIP: curl is required for the server smoke test."
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 is required to build the JSON request."
    exit 0
fi

DEVICE_LIST="$("$LLAMA_SERVER" --list-devices 2>&1 || true)"
if ! grep -Fq "$VULKAN_DEVICE:" <<< "$DEVICE_LIST"; then
    echo "SKIP: requested Vulkan device '$VULKAN_DEVICE' is unavailable."
    echo "$DEVICE_LIST"
    exit 0
fi

SERVER_PID=""
REQUEST_FILE="$(mktemp "${TMPDIR:-/tmp}/turbo4-vulkan-gemma4-request.XXXXXX.json")"
RESPONSE_FILE="$(mktemp "${TMPDIR:-/tmp}/turbo4-vulkan-gemma4-response.XXXXXX.json")"

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    rm -f "$REQUEST_FILE" "$RESPONSE_FILE"
}
trap cleanup EXIT

server_args=(
    -m "$MODEL"
    -dev "$VULKAN_DEVICE"
    -ngl "$GPU_LAYERS"
    -ngld "$GPU_LAYERS_DRAFT"
    -ncmoe "$N_CPU_MOE"
    -ncmoed "$N_CPU_MOE_DRAFT"
    -c "$CTX_SIZE"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -fa on
    -ctk turbo4
    -ctv turbo4
    --reasoning off
    -np 1
    -lv 4
    --host "$HOST"
    --port "$PORT"
)

if [[ "$NO_MMPROJ" == "1" ]]; then
    server_args+=(--no-mmproj)
fi

if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    server_args+=(--jinja --chat-template-file "$CHAT_TEMPLATE_FILE")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$EXTRA_ARGS"
    server_args+=("${extra_args[@]}")
fi

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

echo "Starting Gemma4 Vulkan TurboQuant smoke:"
echo "  server: $LLAMA_SERVER"
echo "  model:  $MODEL"
echo "  device: $VULKAN_DEVICE"
echo "  cache:  -ctk turbo4 -ctv turbo4"
echo "  log:    $LOG_FILE"

"$LLAMA_SERVER" "${server_args[@]}" > "$LOG_FILE" 2>&1 &
SERVER_PID="$!"

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "FAIL: llama-server exited before becoming healthy."
        tail -n 200 "$LOG_FILE" || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "FAIL: llama-server did not become healthy within ${STARTUP_TIMEOUT}s."
        tail -n 200 "$LOG_FILE" || true
        exit 1
    fi
    sleep 2
done

if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "FAIL: llama-server exited before the healthy endpoint was accepted."
    echo "      Check for a stale process already bound to $HOST:$PORT."
    tail -n 200 "$LOG_FILE" || true
    exit 1
fi

PROMPT="$PROMPT" python3 - "$REQUEST_FILE" <<'PY'
import json
import os
import sys

request = {
    "messages": [
        {"role": "user", "content": os.environ["PROMPT"]},
    ],
    "temperature": 0,
    "max_tokens": 16,
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(request, f)
PY

curl -fsS \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_FILE" \
    "http://$HOST:$PORT/v1/chat/completions" \
    > "$RESPONSE_FILE"

if ! grep -Fq "$EXPECT" "$RESPONSE_FILE"; then
    echo "FAIL: completion did not contain expected marker: $EXPECT"
    cat "$RESPONSE_FILE"
    echo
    tail -n 200 "$LOG_FILE" || true
    exit 1
fi

if [[ "$REQUIRE_LOG_MARKERS" == "1" ]]; then
    if ! grep -Eq "using device Vulkan[0-9]+| - Vulkan[0-9]+ :" "$LOG_FILE"; then
        echo "FAIL: server log does not show Vulkan backend startup."
        tail -n 200 "$LOG_FILE" || true
        exit 1
    fi
    if ! grep -Fq "using device $VULKAN_DEVICE" "$LOG_FILE"; then
        echo "FAIL: server log does not show model placement on $VULKAN_DEVICE."
        tail -n 200 "$LOG_FILE" || true
        exit 1
    fi
    if ! grep -Fq "K (turbo4)" "$LOG_FILE"; then
        echo "FAIL: server log does not show K cache using turbo4."
        tail -n 200 "$LOG_FILE" || true
        exit 1
    fi
    if ! grep -Fq "V (turbo4)" "$LOG_FILE"; then
        echo "FAIL: server log does not show V cache using turbo4."
        tail -n 200 "$LOG_FILE" || true
        exit 1
    fi
fi

echo "PASS: Gemma4 Vulkan server completed with -ctk turbo4 -ctv turbo4."
