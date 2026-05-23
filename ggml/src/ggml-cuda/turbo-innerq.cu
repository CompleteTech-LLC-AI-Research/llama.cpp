#include "turbo-innerq.cuh"
#include <cstdint>
#include <cstring>
#include <mutex>

// Host-side shared state for InnerQ cross-TU communication
TURBO_IQ_API bool  g_innerq_finalized = false;
TURBO_IQ_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

static bool g_innerq_tensor_needs_update = false;
static uint64_t g_innerq_generation = 0;
static std::mutex g_innerq_mutex;

void turbo_innerq_publish(const float * scale_inv, int group_size) {
    std::lock_guard<std::mutex> lock(g_innerq_mutex);
    for (int i = 0; i < group_size && i < INNERQ_MAX_CHANNELS; i++) {
        g_innerq_scale_inv_host[i] = scale_inv[i];
    }
    for (int i = group_size; i < INNERQ_MAX_CHANNELS; i++) {
        g_innerq_scale_inv_host[i] = 1.0f;
    }
    g_innerq_finalized = true;
    g_innerq_tensor_needs_update = true;
    ++g_innerq_generation;
}

TURBO_IQ_API bool turbo_innerq_needs_tensor_update(void) {
    std::lock_guard<std::mutex> lock(g_innerq_mutex);
    return g_innerq_tensor_needs_update;
}

TURBO_IQ_API uint64_t turbo_innerq_generation(void) {
    std::lock_guard<std::mutex> lock(g_innerq_mutex);
    return g_innerq_generation;
}

TURBO_IQ_API bool turbo_innerq_scale_inv_snapshot(float * dst, uint64_t * generation) {
    std::lock_guard<std::mutex> lock(g_innerq_mutex);
    if (!g_innerq_finalized || g_innerq_generation == 0) {
        return false;
    }
    std::memcpy(dst, g_innerq_scale_inv_host, INNERQ_MAX_CHANNELS * sizeof(float));
    *generation = g_innerq_generation;
    return true;
}

TURBO_IQ_API void turbo_innerq_mark_tensor_updated(void) {
    // No-op: each KV cache tracks the published generation it has consumed.
}
