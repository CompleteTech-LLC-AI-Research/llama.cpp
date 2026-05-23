/goal
Create a new PR that fixes TurboQuant `turbo4` KV-cache support in llama.cpp Vulkan.

Repository:
`CompleteTech-LLC-AI-Research/llama.cpp`, based on the synced `master` that includes upstream `ggml-org/llama.cpp` through `b0df4c0cf` plus these retained fixes:
- `fix Gemma 4 thinking generation prompt`
- `Add audio transcription fallback for chat input`

Problem:
Gemma4 fails on the B60 Vulkan install when launched with `--cache-type-k q8_0 --cache-type-v turbo4`. The scheduler aborts with:

`pre-allocated tensor (cache_v_l0 (view)) in a buffer (Vulkan0) that cannot run the operation (SET_ROWS)`

Cause to confirm and fix:
The V KV cache is preallocated in Vulkan memory as `GGML_TYPE_TURBO4_0`. Decode updates cache rows through `GGML_OP_SET_ROWS`, but the Vulkan backend does not fully support SET_ROWS into `GGML_TYPE_TURBO4_0`. Existing TurboQuant wiring handles related paths such as turbo3, but turbo4 needs its own Vulkan shader, pipeline registration, support checks, and dispatch/copy behavior.

Implementation target:
Make `GGML_OP_SET_ROWS` work on Vulkan when the destination tensor type is `GGML_TYPE_TURBO4_0`.

Work plan:
1. Create a feature branch from current `master`.
2. Inspect `ggml/src/ggml-vulkan/` for SET_ROWS shader generation, quantized copy/dequant helpers, pipeline registration, and backend op-support checks.
3. Add or port the required Vulkan block/type declarations for TurboQuant turbo4, including dependent turbo2 definitions if turbo4 packing requires them.
4. Extend the Vulkan `copy_to_quant.comp` / generated SET_ROWS shader path for turbo4:
   - support both i32 and i64 row-index variants
   - implement turbo4 transform/quantization correctly
   - pack 4-bit centroid indices into the expected GGML/TurboQuant layout
   - follow neighboring quantized SET_ROWS workgroup and dispatch conventions
5. Register `set_rows_turbo4_0_i32` and `set_rows_turbo4_0_i64` pipelines, populate `pipeline_set_rows_i32/i64[GGML_TYPE_TURBO4_0]`, and update Vulkan SET_ROWS support checks so graph scheduling accepts this op on Vulkan buffers.
6. Keep turbo3 behavior unchanged except for shared helper code that is covered by tests.
7. Build and test locally or on the train/B60 host.
8. Push the branch and open a PR with the code fix.

Required verification:
- Build: `cmake --build build --target test-chat llama-server -j$(nproc)`
- Run: `./build/bin/test-chat`
- On the B60 Vulkan host, launch Gemma4 with production-like settings including:
  `--cache-type-k q8_0 --cache-type-v turbo4 -ngl 999 -ngld 999 -ncmoe 38 -ncmoed 38 -c 65537 -b 512 -ub 512 -fa on --reasoning-budget 0 -np 1`
- Confirm startup no longer hits the Vulkan SET_ROWS scheduler abort, `/health` responds, and a short completion succeeds.
- Smoke compare against `--cache-type-v q8_0` or `turbo3` to catch obvious output corruption.

Constraints:
- Keep the patch narrowly scoped to Vulkan/TurboQuant SET_ROWS support.
- Do not regress CPU, CUDA, Metal, turbo3, the Gemma4 prompt fix, or the audio transcription fallback.
- Do not switch production to turbo4 until runtime validation passes.

Deliverable:
Return the PR URL, changed files, verification results, and any remaining risks.
