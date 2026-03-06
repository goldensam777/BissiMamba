#ifndef MAMBA_LARGE_H
#define MAMBA_LARGE_H

/*
 * mamba_large.h — Public API for the ~1-billion-parameter Mamba LM.
 *
 * Default configuration (ML_CFG_1B):
 *   n_layers  = 38        → Mamba layers stacked
 *   dim       = 2048      → model dimension D
 *   d_inner   = 4096      → expanded inner dim (expand = 2)
 *   d_state   = 16        → SSM state size N
 *   d_conv    = 4         → causal conv1d kernel size
 *   dt_rank   = 128       → Δ projection rank
 *   vocab     = 256       → byte-level tokens (no tokenizer needed)
 *   seq_len   = 2048      → context window
 *
 * Parameter count per layer:
 *   in_proj   D × 2·d_inner        = 2048 × 8192  = 16 777 216
 *   conv1d    d_inner × d_conv     = 4096 × 4     =     16 384
 *   x_proj    d_inner×(dt_r+2·N)  = 4096 × 160   =    655 360
 *   dt_proj   dt_rank × d_inner   = 128  × 4096  =    524 288 (+bias 4096)
 *   A_log     d_inner × d_state   = 4096 × 16    =     65 536
 *   D_param   d_inner             =                     4 096
 *   out_proj  d_inner × D         = 4096 × 2048  =  8 388 608
 *   norm      D                   =                     2 048
 *   ─────────────────────────────────────────────────────────────
 *   per layer                                   ≈ 26 437 636
 *   38 layers                                   ≈ 1 004 629 768
 *   + embedding 256 × 2048                     =    524 288
 *   ─────────────────────────────────────────────────────────────
 *   TOTAL                                       ≈ 1 005 154 056  (~1B) ✓
 *
 * Requires: CUDA ≥ 11.0, cuBLAS, GPU with ≥ 20 GB VRAM (fp32)
 *           or ≥ 10 GB with fp16 (coming in a later pass).
 */

#include <stddef.h>
#include <stdint.h>

/* ── Configuration ───────────────────────────────────────────────── */

typedef struct {
    int   vocab_size;  /* token vocabulary (256 = byte-level)        */
    int   n_layers;    /* number of stacked Mamba layers              */
    int   dim;         /* model dimension D                           */
    int   d_inner;     /* expanded inner dim (= expand_factor * dim)  */
    int   d_state;     /* SSM state size N                            */
    int   d_conv;      /* causal conv1d kernel width                  */
    int   dt_rank;     /* Δ-time projection rank                      */
    int   seq_len;     /* context length (tokens)                     */
    float dt_min;      /* minimum discretisation step                 */
    float dt_max;      /* maximum discretisation step                 */
    int   max_gen_len; /* max new tokens during generation            */
} MLConfig;

/* ~1 B parameter preset */
#define ML_CFG_1B ((MLConfig){ \
    .vocab_size = 256,         \
    .n_layers   = 38,          \
    .dim        = 2048,        \
    .d_inner    = 4096,        \
    .d_state    = 16,          \
    .d_conv     = 4,           \
    .dt_rank    = 128,         \
    .seq_len    = 2048,        \
    .dt_min     = 0.001f,      \
    .dt_max     = 0.1f,        \
    .max_gen_len= 512          \
})

/* Smaller debug / fast-iteration preset (~7 M params) */
#define ML_CFG_SMALL ((MLConfig){ \
    .vocab_size = 256,            \
    .n_layers   = 4,              \
    .dim        = 256,            \
    .d_inner    = 512,            \
    .d_state    = 16,             \
    .d_conv     = 4,              \
    .dt_rank    = 32,             \
    .seq_len    = 512,            \
    .dt_min     = 0.001f,         \
    .dt_max     = 0.1f,           \
    .max_gen_len= 256             \
})

/* ── Opaque model handle ─────────────────────────────────────────── */

typedef struct MLModel MLModel;

/* ── Public API (C-linkage so train_large.c can link) ────────────── */
#ifdef __cplusplus
extern "C" {
#endif

/* ── Lifecycle ───────────────────────────────────────────────────── */

/* Allocate and initialise all weights on GPU.  Returns NULL on failure. */
MLModel *ml_create(const MLConfig *cfg);

/* Free all GPU and host memory owned by the model. */
void     ml_free(MLModel *model);

/* Return approximate parameter count for a configuration. */
long long ml_count_params(const MLConfig *cfg);

/* ── Checkpoint ──────────────────────────────────────────────────── */

/* Write binary snapshot to path. Returns 0 on success, -1 on error. */
int ml_save(const MLModel *model, const char *path);

/* Load binary snapshot. Model must have been created with matching config.
 * Returns 0 on success, -1 on error. */
int ml_load(MLModel *model, const char *path);

/* Checkpoint magic */
#define ML_MAGIC  0x4D4C3142u   /* "ML1B" */

/* ── Inference ───────────────────────────────────────────────────── */

/* Autoregressive generation: feeds prompt then streams up to max_tokens
 * new bytes to stdout (flushed per token).
 * temperature=0 → greedy; 0.8 is a sensible default. */
void ml_generate(MLModel *model,
                 const char *prompt,
                 int         max_tokens,
                 float       temperature);

/* ── Training ────────────────────────────────────────────────────── */

/*
 * MuonClip optimiser config.
 *
 * MuonClip strategy
 * ─────────────────
 * 2-D weight matrices (in_proj, x_proj, dt_proj, out_proj):
 *   1. Global gradient clip (clip_norm).
 *   2. Nesterov-Newton-Schulz orthogonalisation of gradient (ns_steps ≈ 5).
 *   3. Momentum update: M ← beta·M + G_orth
 *   4. Weight update:   W ← W − lr·(M + wd·W)
 *
 * 1-D parameters (biases, RMSNorm scales, A_log, D_param):
 *   AdamW with lr_1d, beta1, beta2, eps, wd.
 */
typedef struct {
    float lr;          /* Muon learning rate  (2-D weights)       */
    float lr_1d;       /* AdamW learning rate (1-D params)        */
    float beta;        /* Muon momentum coefficient               */
    float beta1;       /* AdamW β₁  (1-D params)                  */
    float beta2;       /* AdamW β₂  (1-D params)                  */
    float eps;         /* AdamW ε                                 */
    float weight_decay;/* weight decay applied by both paths      */
    float clip_norm;   /* global gradient clip  (0 = disabled)    */
    int   ns_steps;    /* Newton-Schulz iterations  (5 typical)   */
} MLOptimConfig;

/* Default: Muon lr=0.02, AdamW lr=1e-3, wd=0.01, clip=1.0, NS5 */
#define ML_OPTIM_DEFAULT ((MLOptimConfig){ \
    .lr           = 0.02f,                \
    .lr_1d        = 1e-3f,                \
    .beta         = 0.95f,                \
    .beta1        = 0.9f,                 \
    .beta2        = 0.95f,                \
    .eps          = 1e-8f,                \
    .weight_decay = 0.01f,                \
    .clip_norm    = 1.0f,                 \
    .ns_steps     = 5                     \
})

/* One supervised training step.
 * in_seq[t]  = input  token at position t  (byte value 0-255)
 * tgt_seq[t] = target token at position t  (= in_seq[t+1] for LM)
 * seq_len must equal cfg.seq_len.
 * Returns average cross-entropy loss over the sequence. */
float ml_train_step(MLModel           *model,
                    const int         *in_seq,
                    const int         *tgt_seq,
                    const MLOptimConfig *opt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MAMBA_LARGE_H */
