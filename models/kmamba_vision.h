/* ============================================================================
 * kmamba_vision.h - K-Mamba 2D Vision Model for CIFAR-10
 * 
 * Architecture:
 *   Input:  32 × 32 × 3 (H × W × C)
 *   Conv2D Patch Embed: 3×3 kernel, 96 channels
 *   5 × K-Mamba 2D Blocks (96 dim, 192 state)
 *   Global Mean Pooling → 96
 *   Linear Head: 96 → 10
 * 
 * MX450 Optimized: FP16/BF16, batch 32, MUONCLIP optimizer
 * ============================================================================ */

#ifndef KMAMBA_VISION_H
#define KMAMBA_VISION_H

#include <stddef.h>
#include <stdint.h>
#include "kmamba.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */
#define KMAMBA_VISION_IMG_H     32
#define KMAMBA_VISION_IMG_W     32
#define KMAMBA_VISION_IMG_C     3
#define KMAMBA_VISION_NUM_CLASSES 10

/* Model dimensions (MX450 optimized) */
#define KMAMBA_VISION_DIM       96
#define KMAMBA_VISION_STATE     192
#define KMAMBA_VISION_LAYERS    5
#define KMAMBA_VISION_PATCH_K   3
#define KMAMBA_VISION_PATCH_OUT 96

/* Training config */
#define KMAMBA_VISION_BATCH_SIZE    32
#define KMAMBA_VISION_BATCH_FALLBACK 16
#define KMAMBA_VISION_EPOCHS        60
#define KMAMBA_VISION_WARMUP_EPOCHS 5
#define KMAMBA_VISION_BASE_LR       2e-4f
#define KMAMBA_VISION_MIN_LR        1e-5f
#define KMAMBA_VISION_MUON_MU       0.9f
#define KMAMBA_VISION_CLIP_NORM     1.0f
#define KMAMBA_VISION_WEIGHT_DECAY  0.03f

/* CIFAR-10 dataset */
#define CIFAR10_TRAIN_COUNT     50000
#define CIFAR10_TEST_COUNT      10000
#define CIFAR10_RECORD_SIZE     3073  /* 1 label + 32*32*3 bytes */

/* ============================================================================
 * K-Mamba Vision Model Structure
 * ============================================================================ */

typedef struct {
    /* Spatial dimensions after patch embed (no stride, so same) */
    long spatial_h;
    long spatial_w;
    long spatial_ndims;
    long spatial_dims[2];
    
    /* Model dimensions */
    size_t dim;
    size_t state_size;
    size_t n_layers;
    size_t num_classes;
    
    /* Precision mode */
    int use_fp16;
    int use_bf16;
} KMambaVisionConfig;

typedef struct {
    /* Patch Embedding Conv2D: [3, 3, 3] -> [96] */
    float *patch_kernel;    /* [3 * 3 * 3 * 96] = 2592 floats */
    float *patch_bias;      /* [96] */
    
    /* Running statistics for normalization (optional) */
    float *patch_running_mean;
    float *patch_running_var;
} PatchEmbed;

typedef struct {
    /* Mamba Block 2D parameters */
    MambaBlock **blocks;    /* [n_layers] */
    
    /* Shared wavefront plan for all blocks */
    KMWavefrontPlan *wavefront_plan;
} Mamba2DBackbone;

typedef struct {
    /* Classification head: [dim] -> [num_classes] */
    float *head_weight;     /* [num_classes * dim] */
    float *head_bias;       /* [num_classes] */
} ClassificationHead;

typedef struct {
    /* Optimizer state for head */
    float *m_head;          /* Momentum for MUON */
    float *m_head_bias;
} HeadOptimizerState;

typedef struct {
    KMambaVisionConfig cfg;
    
    /* Components */
    PatchEmbed patch_embed;
    Mamba2DBackbone backbone;
    ClassificationHead head;
    
    /* Optimizer */
    MBOptimConfig optim_cfg;
    HeadOptimizerState head_opt;
    size_t train_step;
    
    /* Training state */
    float last_loss;
    float last_grad_norm;
    int nan_detected;
    
    /* Buffers (pre-allocated, reused) */
    float *buf_input;       /* [batch * H * W * C] */
    float *buf_embed;       /* [batch * H * W * dim] */
    float *buf_hidden;      /* [batch * H * W * dim] - inter-layer */
    float *buf_pooled;      /* [batch * dim] */
    float *buf_logits;      /* [batch * num_classes] */
    float *buf_grad_logits; /* [batch * num_classes] */
    float *buf_grad_embed;  /* [batch * H * W * dim] */
    
    size_t batch_size;      /* Actual allocated batch size */
} KMambaVision;

/* ============================================================================
 * CIFAR-10 Dataset
 * ============================================================================ */
typedef struct {
    uint8_t *images;        /* [N * 3072] - RGB pixels */
    uint8_t *labels;        /* [N] - 0-9 */
    size_t n_samples;
    size_t capacity;
    
    /* Current batch indices */
    size_t *indices;
    size_t current_idx;
} CIFAR10Dataset;

/* ============================================================================
 * API Functions
 * ============================================================================ */

/* Create and destroy */
KMambaVision* kmamba_vision_init(size_t dim, size_t state_size, size_t n_layers, 
                                  size_t batch_size, int use_fp16);
void kmamba_vision_free(KMambaVision *model);

/* Random initialization */
void kmamba_vision_init_weights(KMambaVision *model, uint32_t seed);

/* Forward pass: images [batch, 3, 32, 32] -> logits [batch, 10] */
int kmamba_vision_forward(KMambaVision *model, const float *images, 
                          float *logits_out, size_t batch_size);

/* Training step: returns loss, computes gradients internally */
float kmamba_vision_train_step(KMambaVision *model, const float *images, 
                               const uint8_t *labels, size_t batch_size);

/* Optimizer step: apply accumulated gradients */
void kmamba_vision_optimizer_step(KMambaVision *model);

/* Zero gradients */
void kmamba_vision_zero_grad(KMambaVision *model);

/* Detect NaNs in model weights or activations */
int kmamba_vision_check_nan(const KMambaVision *model);

/* ============================================================================
 * CIFAR-10 Dataset API
 * ============================================================================ */

/* Load CIFAR-10 from binary files */
CIFAR10Dataset* cifar10_load_train(const char *data_dir);
CIFAR10Dataset* cifar10_load_test(const char *data_dir);
void cifar10_dataset_free(CIFAR10Dataset *ds);

/* Get next batch (shuffled for train) */
int cifar10_next_batch(CIFAR10Dataset *ds, float *images_out, uint8_t *labels_out, 
                       size_t batch_size);

/* ============================================================================
 * Training Utilities
 * ============================================================================ */

/* LR scheduler: warmup + cosine decay */
float kmamba_vision_get_lr(int epoch, int warmup_epochs, int total_epochs,
                           float base_lr, float min_lr);

/* Cross-entropy loss with softmax */
float cross_entropy_loss(const float *logits, const uint8_t *labels,
                         float *grad_logits, size_t batch_size, size_t num_classes);

/* RMSNorm (preferred) or LayerNorm */
void rmsnorm_f32(const float *x, float *y, size_t n, float eps);
void rmsnorm_f32_batch(const float *x, float *y, size_t batch, size_t dim, float eps);

/* Global mean pooling: [B, H, W, D] -> [B, D] */
void global_mean_pool(const float *input, float *output,
                      size_t batch, size_t h, size_t w, size_t dim);

/* ============================================================================
 * Internal Helpers (exposed for advanced usage)
 * ============================================================================ */

/* Patch embedding: Conv2D 3x3, stride 1, pad 1 */
void patch_embed_forward(const float *kernel, const float *bias,
                         const float *input, float *output,
                         size_t batch, size_t h, size_t w, 
                         size_t in_c, size_t out_c);

void patch_embed_backward(const float *kernel, 
                          const float *input, const float *grad_output,
                          float *grad_kernel, float *grad_bias,
                          size_t batch, size_t h, size_t w,
                          size_t in_c, size_t out_c);

/* Mamba 2D Block forward/backward using scan_nd */
int mamba2d_block_forward(MambaBlock *block, const float *input, float *output,
                          size_t batch, const KMWavefrontPlan *plan);
int mamba2d_block_backward(MambaBlock *block, const float *grad_output,
                          float *grad_input, size_t batch,
                          const KMWavefrontPlan *plan);

/* Update learning rates (for LR scheduler) */
void kmamba_vision_update_lr(KMambaVision *m, float lr_blocks, float lr_embed);

/* ============================================================================
 * Serialization (.ser format via libkser)
 * ============================================================================ */

/* Save model to .ser file */
int kmamba_vision_save(const KMambaVision *model, const char *path);

/* Load model from .ser file */
KMambaVision* kmamba_vision_load(const char *path, size_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_VISION_H */
