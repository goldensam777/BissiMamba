/* kmamba_cuda_utils.c — CUDA runtime detection and automatic dispatch */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kmamba_cuda_utils.h"

/* Global backend preference - defined in cuda/kmamba_cuda_utils.cu when CUDA available */
#ifndef KMAMBA_BUILD_CUDA
KMambaBackend kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Runtime GPU Detection
 * ═══════════════════════════════════════════════════════════════════════ */

#ifdef KMAMBA_BUILD_CUDA

/* CUDA implementation is in cuda/kmamba_cuda_utils.cu */
/* This prevents gcc from trying to compile CUDA code */

#else /* !KMAMBA_BUILD_CUDA */

int kmamba_cuda_available(void) { return 0; }
int kmamba_cuda_device_count(void) { return 0; }
int kmamba_cuda_current_device(void) { return -1; }

#endif /* KMAMBA_BUILD_CUDA */

/* ═══════════════════════════════════════════════════════════════════════
 * Backend Configuration
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef KMAMBA_BUILD_CUDA
/* When CUDA is available, these functions are implemented in cuda/kmamba_cuda_utils.cu */

void kmamba_backend_init(void) {
    const char *env = getenv("KMAMBA_BACKEND");
    
    if (!env) {
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
        return;
    }
    
    if (strcasecmp(env, "cpu") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_CPU;
        fprintf(stderr, "[k-mamba] Backend forced to CPU via environment\n");
    } else if (strcasecmp(env, "gpu") == 0 || strcasecmp(env, "cuda") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_GPU;
        fprintf(stderr, "[k-mamba] Backend forced to GPU via environment\n");
    } else if (strcasecmp(env, "auto") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
    } else {
        fprintf(stderr, "[k-mamba] Warning: Unknown KMAMBA_BACKEND='%s', using auto\n", env);
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
    }
}

/* kmamba_backend_select is defined above */
KMambaBackend kmamba_backend_select(void) {
    switch (kmamba_backend_preference) {
        case KMAMBA_BACKEND_CPU:
            return KMAMBA_BACKEND_CPU;
            
        case KMAMBA_BACKEND_GPU:
            if (!kmamba_cuda_available()) {
                fprintf(stderr, "[k-mamba] Error: GPU requested but not available\n");
                /* Fall back to CPU with warning */
                return KMAMBA_BACKEND_CPU;
            }
            return KMAMBA_BACKEND_GPU;
            
        case KMAMBA_BACKEND_AUTO:
        default:
            /* GPU if available, else CPU */
            if (kmamba_cuda_available()) {
                return KMAMBA_BACKEND_GPU;
            }
            return KMAMBA_BACKEND_CPU;
    }
}
#endif /* !KMAMBA_BUILD_CUDA */
