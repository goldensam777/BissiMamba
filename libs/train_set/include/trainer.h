#ifndef KMAMBA_TRAINER_H
#define KMAMBA_TRAINER_H

#include <stdio.h>
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

/* ============================================================================
 * Logging Configuration
 * ============================================================================ */

typedef struct {
    char *log_dir;           /* Directory for CSV logs (created if not exists) */
    char *run_name;          /* Name prefix for this training run */
    int log_step_every;      /* Log every N steps (0 = disable) */
    int log_epoch_every;     /* Log every N epochs (0 = disable) */
} TrainerLogConfig;

/* ============================================================================
 * Resume/Checkpoint State
 * ============================================================================ */

typedef struct {
    int epoch;               /* Current epoch (0-based) */
    size_t global_step;    /* Total steps across all epochs */
    float best_val_loss;   /* Best validation loss seen */
    char *checkpoint_path; /* Path to save/resume from */
} TrainerResumeState;

/* Opaque structure for checkpoint storage */
typedef struct KMambaCheckpointState KMambaCheckpointState;

typedef struct {
    KMamba *model;
    KMambaCheckpointState *ckpt;
    TrainerGCConfig gc_config;
    TrainerLogConfig log_cfg;
    TrainerResumeState resume;

    /* Logging state */
    FILE *step_log;
    FILE *epoch_log;
    unsigned long long run_id;

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
 * Complete training step on a batch (token-based, for LM)
 */
float trainer_train_batch(Trainer *tr, const uint32_t *batch_tokens, size_t batch_size);

/**
 * Complete training step on a batch (vision/data-based, for vocab_size=0 models)
 * Runs forward on raw data, computes cross-entropy loss against labels, backward, and updates
 */
float trainer_train_batch_vision(Trainer *tr, const float *data, const uint32_t *labels,
                                  size_t batch_size, size_t L, size_t D, int num_classes);

/* ============================================================================
 * Unified Batch API (Generic for all model types)
 * ============================================================================ */

/**
 * Training metrics returned by trainer_train_batch_ex
 */
typedef struct {
    float loss;
    float accuracy;
} TrainerMetrics;

typedef enum {
    TRAINER_BATCH_TOKENS,     /* Language Model: uint32_t* token IDs */
    TRAINER_BATCH_VISION,     /* Vision: float* data + uint32_t* labels */
    TRAINER_BATCH_CUSTOM      /* Extension: user-defined via callbacks */
} TrainerBatchType;

/* Forward declaration for custom batch */
struct TrainerBatch;

/* Custom batch callback signature */
typedef float (*TrainerCustomLossFn)(const struct TrainerBatch *batch, KMamba *m,
                                      float **d_output_out, void *user_ctx);

typedef struct TrainerBatch {
    TrainerBatchType type;
    size_t batch_size;
    union {
        struct {
            const uint32_t *tokens;  /* (batch_size, seq_len+1) token IDs */
        } tokens;
        struct {
            const float *data;       /* (batch_size, L, D) flattened */
            const uint32_t *labels;  /* (batch_size) class labels */
            size_t seq_len;          /* L */
            size_t dim;              /* D */
            int num_classes;
        } vision;
        struct {
            void *user_data;         /* User-provided context */
            void *user_ctx;          /* Additional context pointer */
            TrainerCustomLossFn loss_fn;  /* Custom loss computation */
        } custom;
    };
} TrainerBatch;

/**
 * Unified training step for all batch types
 * Automatically handles logging and metrics
 * Returns loss and accuracy (accuracy > 0 only for VISION batches)
 */
TrainerMetrics trainer_train_batch_ex(Trainer *tr, const TrainerBatch *batch);

/* Helper macros for creating batches */
#define TRAINER_BATCH_TOKENS_INIT(tok, bs) \
    (TrainerBatch){.type = TRAINER_BATCH_TOKENS, .batch_size = (bs), \
                   .tokens = {.tokens = (tok)}}

#define TRAINER_BATCH_VISION_INIT(dat, lbl, bs, L_, D_, nc) \
    (TrainerBatch){.type = TRAINER_BATCH_VISION, .batch_size = (bs), \
                   .vision = {.data = (dat), .labels = (lbl), \
                             .seq_len = (L_), .dim = (D_), .num_classes = (nc)}}

/**
 * Create trainer with logging support
 * log_cfg: if NULL, logging is disabled
 */
Trainer* trainer_create_with_logging(KMamba *model, const TrainerGCConfig *gc_cfg,
                                      const TrainerLogConfig *log_cfg);

/**
 * Resume training from checkpoint
 * Loads model state and resume state from checkpoint_path
 * Returns trainer ready to continue from saved epoch/step
 */
Trainer* trainer_resume(KMamba *model, const TrainerGCConfig *gc_cfg,
                        const TrainerLogConfig *log_cfg,
                        const char *checkpoint_path);

/**
 * Save training checkpoint (model + training state)
 * Call periodically to enable resume capability
 */
int trainer_save_checkpoint(Trainer *tr, const char *path);

/**
 * Load training checkpoint (model + optimizer state)
 * Returns 0 on success, -1 on error
 */
int trainer_load_checkpoint(Trainer *tr, const char *path);

/**
 * Get current training metrics (for display)
 */
typedef struct {
    int epoch;
    size_t global_step;
    float last_loss;
    float last_lr;
    size_t max_rss_kb;
} TrainerResumeMetrics;

void trainer_get_metrics(const Trainer *tr, TrainerResumeMetrics *metrics);

/**
 * Log a step to CSV (called automatically by trainer_train_batch_ex)
 */
void trainer_log_step(Trainer *tr, int epoch, size_t step_in_epoch,
                      float loss, float accuracy, double step_ms);

/**
 * Log an epoch summary to CSV
 */
void trainer_log_epoch(Trainer *tr, int epoch, size_t total_steps,
                       float avg_loss, float avg_accuracy, double epoch_ms);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_TRAINER_H */
