#pragma once

#include <stdint.h>

// TurboQuant InnerQ per-channel equalization — cross-TU shared state
// The host-side state lives in turbo-innerq.cu; device-side state is per-TU
// in turbo-quant.cuh (only set-rows.cu needs device access).

#define INNERQ_MAX_CHANNELS 128

#ifdef GGML_BACKEND_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#      define TURBO_IQ_API __declspec(dllexport)
#    else
#      define TURBO_IQ_API __declspec(dllimport)
#    endif
#  else
#    define TURBO_IQ_API __attribute__((visibility("default")))
#  endif
#else
#  define TURBO_IQ_API
#endif

// Host-side shared state (defined in turbo-innerq.cu)
TURBO_IQ_API extern bool  g_innerq_finalized;
TURBO_IQ_API extern float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

// Called from set-rows.cu after InnerQ finalization to publish scale_inv
void turbo_innerq_publish(const float * scale_inv, int group_size);

// Called from llama-kv-cache.cpp (or equivalent) to check if tensor needs update
TURBO_IQ_API bool turbo_innerq_needs_tensor_update(void);

// Monotonic publication generation for per-context tensor updates
TURBO_IQ_API uint64_t turbo_innerq_generation(void);

// Copy a consistent host scale snapshot and its generation.
TURBO_IQ_API bool turbo_innerq_scale_inv_snapshot(float * dst, uint64_t * generation);

// Legacy hook kept for ABI compatibility; tensor updates are generation-based.
TURBO_IQ_API void turbo_innerq_mark_tensor_updated(void);
