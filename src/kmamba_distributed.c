/* kmamba_distributed.c — Multi-GPU support (pipeline parallelism, no NCCL required) */

#include "../include/kmamba_distributed.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#endif

/* ============================================================================
 * Pipeline Parallelism (Single node, no NCCL required)
 * ============================================================================ */

typedef struct {
    KMambaDistributedConfig config;
    int n_gpus;
    int current_gpu;
} KMambaPipelineContext;

KMambaDistributedContext* kmamba_distributed_init_pipeline(int n_gpus) {
    if (n_gpus <= 0) return NULL;
    
    KMambaPipelineContext *ctx = (KMambaPipelineContext*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    
    ctx->n_gpus = n_gpus;
    ctx->current_gpu = 0;
    
    ctx->config.world_size = n_gpus;
    ctx->config.rank = 0;  /* Default to rank 0, caller should set for each GPU */
    ctx->config.use_nccl = 0;
    ctx->config.use_p2p = 0;  /* Will try to enable */
    
    /* Check if P2P is available */
    #ifdef KMAMBA_BUILD_CUDA
    if (n_gpus > 1) {
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, 0, 1);
        ctx->config.use_p2p = can_access;
        
        /* Enable P2P access between all GPUs */
        for (int i = 0; i < n_gpus; i++) {
            for (int j = 0; j < n_gpus; j++) {
                if (i != j) {
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {
                        cudaSetDevice(i);
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }
    }
    #endif
    
    return (KMambaDistributedContext*)ctx;
}

void kmamba_distributed_fini_pipeline(KMambaDistributedContext *ctx) {
    if (!ctx) return;
    
    KMambaPipelineContext *pctx = (KMambaPipelineContext*)ctx;
    
    #ifdef KMAMBA_BUILD_CUDA
    /* Disable P2P access */
    if (pctx->n_gpus > 1) {
        for (int i = 0; i < pctx->n_gpus; i++) {
            for (int j = 0; j < pctx->n_gpus; j++) {
                if (i != j) {
                    cudaSetDevice(i);
                    cudaDeviceDisablePeerAccess(j);
                }
            }
        }
    }
    #endif
    
    free(pctx);
}

int kmamba_pipeline_send_recv(KMambaDistributedContext *ctx,
                               float *d_send_buf,
                               float *d_recv_buf,
                               size_t n_elements,
                               int dst_rank,
                               int src_rank) {
    (void)ctx;
    (void)dst_rank;
    (void)src_rank;

    #ifdef KMAMBA_BUILD_CUDA
    size_t bytes = n_elements * sizeof(float);
    
    /* If P2P is enabled, direct device-to-device copy works */
    /* Otherwise, need to go through host (slower) */
    if (d_send_buf && d_recv_buf) {
        cudaMemcpy(d_recv_buf, d_send_buf, bytes, cudaMemcpyDeviceToDevice);
    } else if (d_send_buf) {
        /* Only send, no recv */
        float *host_buf = (float*)malloc(bytes);
        if (!host_buf) return -1;
        cudaMemcpy(host_buf, d_send_buf, bytes, cudaMemcpyDeviceToHost);
        /* Would need to transfer to other GPU here */
        free(host_buf);
    }
    
    return 0;
    #else
    (void)d_send_buf; (void)d_recv_buf; (void)n_elements;
    (void)dst_rank; (void)src_rank;
    return -1;  /* CUDA not available */
    #endif
}

/* ============================================================================
 * Data Parallelism helpers
 * ============================================================================ */

void kmamba_distributed_split_batch(KMambaDistributedConfig *cfg,
                                     size_t batch_size,
                                     size_t total_gpus) {
    if (!cfg || total_gpus == 0) return;
    
    cfg->world_size = (int)total_gpus;
    
    /* Split batch evenly */
    size_t per_gpu = batch_size / total_gpus;
    size_t remainder = batch_size % total_gpus;
    
    /* Earlier ranks get extra items if not divisible */
    if ((size_t)cfg->rank < remainder) {
        cfg->batch_start = cfg->rank * (per_gpu + 1);
        cfg->batch_end = cfg->batch_start + per_gpu + 1;
    } else {
        cfg->batch_start = cfg->rank * per_gpu + remainder;
        cfg->batch_end = cfg->batch_start + per_gpu;
    }
}

void kmamba_distributed_split_layers(KMambaDistributedConfig *cfg,
                                      size_t n_layers,
                                      size_t total_gpus) {
    if (!cfg || total_gpus == 0) return;
    
    cfg->world_size = (int)total_gpus;
    
    /* Split layers evenly */
    size_t per_gpu = n_layers / total_gpus;
    size_t remainder = n_layers % total_gpus;
    
    /* Earlier ranks get extra layers if not divisible */
    if ((size_t)cfg->rank < remainder) {
        cfg->layer_start = cfg->rank * (per_gpu + 1);
        cfg->layer_end = cfg->layer_start + per_gpu + 1;
    } else {
        cfg->layer_start = cfg->rank * per_gpu + remainder;
        cfg->layer_end = cfg->layer_start + per_gpu;
    }
}
