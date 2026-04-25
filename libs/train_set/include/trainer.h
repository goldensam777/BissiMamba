#ifndef KMAMBA_TRAINER_H
#define KMAMBA_TRAINER_H

#include "kmamba.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Gradient Checkpointing (GC) Configuration
 * ============================================================================ */

typedef enum {
    TRAINER_GC_NONE,    /* No GC: store all layer activations (Fastest, High VRAM) */
    TRAINER_GC_EVERY_N, /* Checkpoint every N layers (Balanced) */
    TRAINER_GC_ALL      /* Recompute everything (Slowest, Lowest VRAM) */
} TrainerGCPolicy;

typedef struct {
    TrainerGCPolicy policy;
    int checkpoint_every_n;
} TrainerGCConfig;

/* Opaque structure for checkpoint storage */
typedef struct KMambaCheckpointState KMambaCheckpointState;

typedef struct {
    KMamba *model;
    KMambaCheckpointState *ckpt;
    TrainerGCConfig gc_config;
    
    /* Workspace for recomputation */
    float *recompute_buffer; 
} Trainer;

/* ============================================================================
 * Trainer API
 * ============================================================================ */

/**
 * Create a new trainer with GC support
 */
Trainer* trainer_create(KMamba *model, const TrainerGCConfig *gc_cfg);

/**
 * Free trainer and associated checkpoints
 */
void trainer_free(Trainer *tr);

/**
 * Forward pass through the model with checkpointing
 */
int trainer_forward(Trainer *tr, const float *input, float *output, size_t batch_size);

/**
 * Backward pass with automatic recomputation
 */
void trainer_backward(Trainer *tr, const float *dY, const float *input, size_t batch_size);

/**
 * Complete training step on a batch
 */
float trainer_train_batch(Trainer *tr, const uint32_t *batch_tokens, size_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_TRAINER_H */
