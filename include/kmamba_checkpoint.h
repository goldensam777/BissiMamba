/* kmamba_checkpoint.h — Gradient checkpointing for 1B+ models */

#ifndef KMAMBA_CHECKPOINT_H
#define KMAMBA_CHECKPOINT_H

#include <stddef.h>
#include "kmamba.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Gradient Checkpointing Configuration
 * ============================================================================ */

typedef enum {
    KMAMBA_CHECKPOINT_NONE = 0,      /* Store all activations (default) */
    KMAMBA_CHECKPOINT_LAYER = 1,     /* Checkpoint every N layers */
    KMAMBA_CHECKPOINT_BLOCK = 2      /* Checkpoint every block (recompute during bwd) */
} KMambaCheckpointPolicy;

typedef struct {
    KMambaCheckpointPolicy policy;
    int checkpoint_every_n_layers;   /* For LAYER policy: checkpoint every N layers */
    int recompute_activation;       /* 1 = recompute forward during backward */
} KMambaCheckpointConfig;

/* Default: no checkpointing (store everything) */
static inline KMambaCheckpointConfig kmamba_checkpoint_default(void) {
    KMambaCheckpointConfig cfg = {
        .policy = KMAMBA_CHECKPOINT_NONE,
        .checkpoint_every_n_layers = 1,
        .recompute_activation = 0
    };
    return cfg;
}

/* For 1B+ models: checkpoint every layer, recompute during backward */
static inline KMambaCheckpointConfig kmamba_checkpoint_aggressive(void) {
    KMambaCheckpointConfig cfg = {
        .policy = KMAMBA_CHECKPOINT_BLOCK,
        .checkpoint_every_n_layers = 1,
        .recompute_activation = 1
    };
    return cfg;
}

/* ============================================================================
 * Checkpoint Buffer Management
 * ============================================================================ */

/* Per-layer checkpoint storage (minimal) */
typedef struct {
    /* Only store inputs to layer, not all intermediate activations */
    float *layer_input;      /* [L, D] - input to this layer */
    size_t size;             /* size in bytes */
    int has_checkpoint;      /* 1 if checkpoint is valid */
} MambaLayerCheckpoint;

/* Full model checkpoint storage */
typedef struct {
    MambaLayerCheckpoint *layers;  /* Array per layer */
    size_t n_layers;
    KMambaCheckpointConfig config;
    
    /* GPU buffers for recomputation */
    float *d_recompute_buffer;  /* Temporary buffer for forward recompute */
    size_t recompute_buffer_size;
} KMambaCheckpointState;

/* Create checkpoint state for a model */
KMambaCheckpointState* kmamba_checkpoint_create(const KMamba *m, 
                                               const KMambaCheckpointConfig *cfg);

/* Free checkpoint state */
void kmamba_checkpoint_free(KMambaCheckpointState *ckpt);

/* Save layer input checkpoint (called during forward) */
void kmamba_checkpoint_save_layer(KMambaCheckpointState *ckpt, 
                                   size_t layer_idx,
                                   const float *layer_input,
                                   size_t L, size_t D);

/* Get layer checkpoint for backward pass */
const float* kmamba_checkpoint_get_layer(const KMambaCheckpointState *ckpt,
                                          size_t layer_idx);

/* Clear layer checkpoint (free memory) */
void kmamba_checkpoint_clear_layer(KMambaCheckpointState *ckpt, size_t layer_idx);

#ifdef KMAMBA_BUILD_CUDA
/* ============================================================================
 * GPU Gradient Checkpointing
 * ============================================================================ */

/* GPU checkpoint state */
typedef struct {
    float **d_layer_inputs;     /* Array of device pointers per layer */
    size_t n_layers;
    size_t L, D;                /* Sequence length, model dimension */
    KMambaCheckpointConfig config;
} KMambaCheckpointStateGPU;

/* Create GPU checkpoint state */
KMambaCheckpointStateGPU* kmamba_checkpoint_create_gpu(size_t n_layers, 
                                                        size_t L, size_t D,
                                                        const KMambaCheckpointConfig *cfg);

/* Free GPU checkpoint state */
void kmamba_checkpoint_free_gpu(KMambaCheckpointStateGPU *ckpt);

/* Save layer input to GPU checkpoint */
void kmamba_checkpoint_save_layer_gpu(KMambaCheckpointStateGPU *ckpt,
                                      size_t layer_idx,
                                      const float *d_layer_input);

/* Get layer checkpoint for backward (device pointer) */
const float* kmamba_checkpoint_get_layer_gpu(const KMambaCheckpointStateGPU *ckpt,
                                              size_t layer_idx);

/* Recompute forward pass for a layer during backward
 * This is the key function for gradient checkpointing:
 * - Takes layer input from checkpoint
 * - Recomputes forward pass
 * - Returns activations needed for backward */
int kmamba_checkpoint_recompute_layer_gpu(
    const KMambaCheckpointStateGPU *ckpt,
    const MambaBlock *block,
    size_t layer_idx,
    float *d_output,           /* Output buffer */
    void *d_cublas_handle,     /* cuBLAS handle */
    float *d_workspace,      /* Temporary workspace */
    size_t workspace_size);

#endif /* KMAMBA_BUILD_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_CHECKPOINT_H */
