/* kmamba_distributed.h — Optional NCCL multi-GPU support */

#ifndef KMAMBA_DISTRIBUTED_H
#define KMAMBA_DISTRIBUTED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Multi-GPU Configuration
 * ============================================================================ */

typedef struct {
    int world_size;          /* Total number of GPUs (1 = single GPU) */
    int rank;                /* Current GPU rank (0 to world_size-1) */
    int local_rank;          /* Local GPU ID on this node */
    
    /* Pipeline parallelism: each GPU handles some layers */
    size_t layer_start;      /* First layer on this GPU */
    size_t layer_end;        /* Last layer on this GPU (exclusive) */
    
    /* Data parallelism: each GPU handles some batch items */
    size_t batch_start;
    size_t batch_end;
    
    int use_nccl;            /* 1 if NCCL is available and enabled */
    int use_p2p;             /* 1 if peer-to-peer memory access enabled */
} KMambaDistributedConfig;

/* Default: single GPU, no distribution */
static inline KMambaDistributedConfig kmamba_distributed_default(void) {
    KMambaDistributedConfig cfg = {
        .world_size = 1,
        .rank = 0,
        .local_rank = 0,
        .layer_start = 0,
        .layer_end = 0,  /* Will be set to n_layers */
        .batch_start = 0,
        .batch_end = 0,
        .use_nccl = 0,
        .use_p2p = 0
    };
    return cfg;
}

/* ============================================================================
 * NCCL Support (Optional - requires -DKMAMBA_USE_NCCL at compile time)
 * ============================================================================ */

/* NCCL is optional to maintain zero-dependency philosophy.
 * When NCCL is not available, k-mamba works on single GPU only.
 * To enable NCCL: compile with -DKMAMBA_USE_NCCL and link with -lnccl */

#ifdef KMAMBA_USE_NCCL

/* NCCL includes */
#include <nccl.h>

/* Distributed context with NCCL */
typedef struct {
    KMambaDistributedConfig config;
    ncclComm_t nccl_comm;    /* NCCL communicator */
    cudaStream_t stream;     /* CUDA stream for this rank */
    void *nccl_scratch;      /* Scratch buffer for allreduce */
    size_t scratch_size;
} KMambaDistributedContext;

/* Initialize NCCL for multi-GPU training */
KMambaDistributedContext* kmamba_distributed_init_nccl(int world_size, int rank);

/* Finalize NCCL */
void kmamba_distributed_fini_nccl(KMambaDistributedContext *ctx);

/* All-reduce gradients across GPUs */
int kmamba_allreduce_gradients_nccl(KMambaDistributedContext *ctx,
                                       float *d_gradients,
                                       size_t n_elements);

/* Broadcast parameters from rank 0 to all GPUs */
int kmamba_broadcast_params_nccl(KMambaDistributedContext *ctx,
                                  float *d_params,
                                  size_t n_elements);

#else /* !KMAMBA_USE_NCCL */

/* Stub implementation when NCCL is not available */
typedef struct {
    KMambaDistributedConfig config;
    void *dummy;  /* Placeholder */
} KMambaDistributedContext;

/* Stub functions - print warning and return error */
static inline KMambaDistributedContext* kmamba_distributed_init_nccl(int world_size, int rank) {
    (void)world_size; (void)rank;
    return NULL;  /* NCCL not available */
}

static inline void kmamba_distributed_fini_nccl(KMambaDistributedContext *ctx) {
    (void)ctx;
}

static inline int kmamba_allreduce_gradients_nccl(KMambaDistributedContext *ctx,
                                                    float *d_gradients,
                                                    size_t n_elements) {
    (void)ctx; (void)d_gradients; (void)n_elements;
    return -1;  /* NCCL not available */
}

static inline int kmamba_broadcast_params_nccl(KMambaDistributedContext *ctx,
                                                float *d_params,
                                                size_t n_elements) {
    (void)ctx; (void)d_params; (void)n_elements;
    return -1;  /* NCCL not available */
}

#endif /* KMAMBA_USE_NCCL */

/* ============================================================================
 * Pipeline Parallelism (Single node, multiple GPUs)
 * ============================================================================ */

/* Initialize pipeline parallelism without NCCL (slower, uses cudaMemcpy) */
KMambaDistributedContext* kmamba_distributed_init_pipeline(int n_gpus);

/* Finalize pipeline context */
void kmamba_distributed_fini_pipeline(KMambaDistributedContext *ctx);

/* Transfer activations between GPUs in pipeline */
int kmamba_pipeline_send_recv(KMambaDistributedContext *ctx,
                               float *d_send_buf,
                               float *d_recv_buf,
                               size_t n_elements,
                               int dst_rank,
                               int src_rank);

/* ============================================================================
 * Data Parallelism helpers
 * ============================================================================ */

/* Split batch across GPUs */
void kmamba_distributed_split_batch(KMambaDistributedConfig *cfg,
                                     size_t batch_size,
                                     size_t total_gpus);

/* Split layers across GPUs for pipeline parallelism */
void kmamba_distributed_split_layers(KMambaDistributedConfig *cfg,
                                      size_t n_layers,
                                      size_t total_gpus);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_DISTRIBUTED_H */
