/*
 * mamba_block.c — MambaBlock SSM architecture with MUONCLIP optimizer
 *
 * Core Mamba implementation: diagonal A, shared B/C vectors,
 * input-dependent delta, W_in/W_out projections, scan1d/scan2d ASM kernels.
 *
 * Part of k-mamba — uses optimatrix for compute kernels.
 */

#include "kmamba.h"
#include "optimatrix.h"
#include "mamba_scan.h"
#ifdef KMAMBA_BUILD_CUDA
#include "mamba_scan_cuda.h"
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Scalar activations (internal — vectorized versions in ASM)
 * ============================================================================ */

static float scalar_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

static float scalar_sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* ============================================================================
 * Forward storage for training (per timestep)
 * ============================================================================ */
typedef struct {
    float *x;         /* seq_len x state_size */
    float *A_diag;    /* seq_len x state_size */
    float *B_bar;     /* seq_len x state_size */
    float *u_seq;     /* seq_len x state_size (controller vectors per timestep) */
} ForwardStore;

/* ============================================================================
 * Global registry mapping MambaBlock -> MBOptimState
 * ============================================================================ */
static MBOptimState *g_opt_states[256];
static MambaBlock   *g_opt_blocks[256];
static size_t        g_opt_n = 0;

/* forward declarations */
static void          _mamba_free_opt_for(MambaBlock *block);
static MBOptimState* _find_opt(MambaBlock *block);
static int           matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols);
static void          project_controller(const MambaBlock *block, const float *x_t,
                                        float *z_buf, float *u_out);
static float         project_delta_value(const MambaBlock *block, const float *x_t,
                                         float *tmp_delta, size_t position,
                                         size_t total_positions);
static void          transpose_row_major(const float *src, float *dst,
                                         size_t rows, size_t cols);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

MBMatrix* mb_matrix_create(size_t rows, size_t cols) {
    MBMatrix *m = (MBMatrix *)malloc(sizeof(MBMatrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = (float *)calloc(rows * cols, sizeof(float));

    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

static int matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols) {
    MBMatrix *m;

    if (!dst) return -1;
    m = mb_matrix_create(rows, cols);
    if (!m) {
        memset(dst, 0, sizeof(*dst));
        return -1;
    }
    *dst = *m;
    free(m);
    return 0;
}

void mb_matrix_free(MBMatrix *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void mb_matrix_zero(MBMatrix *m) {
    if (!m || !m->data) return;
    memset(m->data, 0, m->rows * m->cols * sizeof(float));
}

void mb_matrix_copy(MBMatrix *dst, const MBMatrix *src) {
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->rows != src->rows || dst->cols != src->cols) return;
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

void mb_matrix_print(const MBMatrix *m) {
    if (!m || !m->data) return;

    printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%10.6f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

void mb_matrix_vec_mult(float *out, const MBMatrix *m, const float *v) {
    if (!out || !m || !v || !m->data) return;
    gemv_avx2(m->data, (float *)v, out, (long)m->rows, (long)m->cols);
}

void mb_vec_add(float *y, const float *x, size_t n) {
    if (!y || !x) return;
    for (size_t i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void mb_vec_scale(float *v, float alpha, size_t n) {
    if (!v) return;
    for (size_t i = 0; i < n; i++) {
        v[i] *= alpha;
    }
}

static void project_controller(const MambaBlock *block, const float *x_t,
                                float *z_buf, float *u_out) {
    if (!block || !x_t || !z_buf || !u_out) return;
    mb_matrix_vec_mult(z_buf, &block->W_in, x_t);
    silu_f32(z_buf, u_out, (long)block->config.state_size);
}

static float project_delta_value(const MambaBlock *block, const float *x_t,
                                  float *tmp_delta, size_t position,
                                  size_t total_positions) {
    float dval;

    if (!block || !x_t || !tmp_delta || total_positions == 0) return 0.0f;

    if (block->delta_proj.rows > 0) {
        mb_matrix_vec_mult(tmp_delta, &block->delta_proj, x_t);
        dval = scalar_softplus(tmp_delta[0]);
        if (dval < block->config.dt_min) dval = block->config.dt_min;
        if (dval > block->config.dt_max) dval = block->config.dt_max;
        return dval;
    }

    return block->config.dt_scale *
           ((float)position / (float)total_positions + 1.0f);
}

static void transpose_row_major(const float *src, float *dst,
                                 size_t rows, size_t cols) {
    if (!src || !dst) return;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* ============================================================================
 * Discretization Functions
 * ============================================================================ */

void mb_discretize_A(MBMatrix *A_bar, const MBMatrix *A, float dt) {
    if (!A_bar || !A) return;

    for (size_t i = 0; i < A_bar->rows && i < A->rows; i++) {
        float a_ii = A->data[i * A->cols + i];
        A_bar->data[i * A_bar->cols + i] = expf(dt * a_ii);
    }
}

void mb_discretize_B(float *B_bar, const MBMatrix *A, const float *B,
                     float dt, size_t state_size) {
    if (!B_bar || !A || !B) return;
    (void)A;

    for (size_t i = 0; i < state_size; i++) {
        B_bar[i] = dt * B[i];
    }
}

/* ============================================================================
 * Selective Scan - Core Mamba Operation
 * ============================================================================ */

void mb_selective_scan(float *output, float *state,
                       const float *input, const float *delta,
                       const MBMatrix *A_bar, const float *B_bar,
                       const MBMatrix *C, float D,
                       size_t seq_len, size_t state_size) {

    if (!output || !state || !input || !delta || !A_bar || !B_bar || !C) {
        return;
    }
    (void)C; (void)D;

    memset(state, 0, state_size * sizeof(float));

    float *temp_state = (float *)malloc(state_size * sizeof(float));
    float *A_diag_t = (float *)malloc(state_size * sizeof(float));
    float *B_bar_t = (float *)malloc(state_size * sizeof(float));
    if (!temp_state || !A_diag_t || !B_bar_t) {
        free(temp_state); free(A_diag_t); free(B_bar_t);
        return;
    }

    for (size_t t = 0; t < seq_len; t++) {
        const float *u_t = &input[t * state_size];
        float dt_t = delta[t];

        for (size_t i = 0; i < state_size; i++) {
            float a_val = A_bar->data[i * state_size + i];
            if (a_val > -1e-5f) a_val = -1e-5f;   /* clamp A ≤ -1e-5 */
            float a_diag = expf(dt_t * a_val);
            A_diag_t[i] = a_diag;
            if (fabsf(a_val) < 1e-8f) {
                B_bar_t[i] = dt_t * B_bar[i];
            } else {
                B_bar_t[i] = (a_diag - 1.0f) / a_val * B_bar[i];
            }
        }

        hadamard_avx2(A_diag_t, state, temp_state, (long)state_size);
        hadamard_avx2(B_bar_t, (float *)u_t, state, (long)state_size);
        mb_vec_add(state, temp_state, state_size);

        memcpy(&output[t * state_size], state, state_size * sizeof(float));
    }

    free(A_diag_t); free(B_bar_t); free(temp_state);
}

/* ============================================================================
 * Mamba Block Operations
 * ============================================================================ */

MambaBlock* mamba_block_create(const MBConfig *config) {
    if (!config) return NULL;

    MambaBlock *block = (MambaBlock *)malloc(sizeof(MambaBlock));
    if (!block) return NULL;
    memset(block, 0, sizeof(*block));

    block->config = *config;

    if (matrix_init_owned(&block->W_in, config->state_size, config->dim) != 0 ||
        matrix_init_owned(&block->W_out, config->dim, config->state_size) != 0 ||
        matrix_init_owned(&block->A_log, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->B_mat, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->C_mat, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->delta_proj, 1, config->dim) != 0) {
        mamba_block_free(block);
        return NULL;
    }

    block->hidden = (float *)calloc(config->state_size, sizeof(float));
    block->delta  = (float *)calloc(config->seq_len, sizeof(float));

    size_t LD = config->seq_len * config->state_size;
    block->scan_B     = (float *)malloc(LD * sizeof(float));
    block->scan_C     = (float *)malloc(LD * sizeof(float));
    block->scan_delta = (float *)malloc(LD * sizeof(float));
    block->scan_h     = (float *)calloc(config->state_size, sizeof(float));

    if (!block->W_in.data || !block->W_out.data || !block->A_log.data ||
        !block->B_mat.data || !block->C_mat.data || !block->delta_proj.data ||
        !block->hidden || !block->delta ||
        !block->scan_B || !block->scan_C || !block->scan_delta || !block->scan_h) {
        mamba_block_free(block);
        return NULL;
    }

    /* Allocate ConvND resources if enabled */
    if (config->use_convnd && config->convnd_K > 0 && config->convnd_ndims > 0) {
        long kernel_size = config->convnd_ndims * config->convnd_K * config->dim;
        block->convnd_kernel = (float *)calloc(kernel_size, sizeof(float));
        block->convnd_bias = (float *)calloc(config->dim, sizeof(float));
        
        ConvNDParams p = {
            .dims = NULL,  /* Will be set per forward call */
            .ndims = config->convnd_ndims,
            .D = config->dim,
            .K = config->convnd_K
        };
        block->convnd_ws = convnd_workspace_create(&p);
        
        if (!block->convnd_kernel || !block->convnd_bias || !block->convnd_ws) {
            mamba_block_free(block);
            return NULL;
        }
    }

    return block;
}

void mamba_block_free(MambaBlock *block) {
    if (!block) return;

    if (block->W_in.data) free(block->W_in.data);
    if (block->W_out.data) free(block->W_out.data);
    if (block->A_log.data) free(block->A_log.data);
    if (block->B_mat.data) free(block->B_mat.data);
    if (block->C_mat.data) free(block->C_mat.data);
    if (block->delta_proj.data) free(block->delta_proj.data);
    if (block->hidden) free(block->hidden);
    if (block->delta) free(block->delta);
    if (block->scan_B)     free(block->scan_B);
    if (block->scan_C)     free(block->scan_C);
    if (block->scan_delta) free(block->scan_delta);
    if (block->scan_h)     free(block->scan_h);
    
    /* Free ConvND resources */
    if (block->convnd_kernel) free(block->convnd_kernel);
    if (block->convnd_bias)   free(block->convnd_bias);
    if (block->convnd_ws)     convnd_workspace_free(block->convnd_ws);

    free(block);
}

void mamba_block_init(MambaBlock *block) {
    if (!block) return;

    /* A_log stocke directement A (valeur négative).
     * A = -1 pour toutes les dimensions → decay = exp(-Δ) ∈ (0.37, 0.999).
     * La stabilité est garantie par le clamp A ≤ -1e-5 dans le forward. */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->A_log.data[i] = -1.0f;
    }

    for (size_t i = 0; i < block->config.state_size; i++) {
        block->B_mat.data[i] = 1.0f / sqrtf((float)block->config.state_size);
    }

    for (size_t i = 0; i < block->config.state_size; i++) {
        block->C_mat.data[i] = 1.0f / sqrtf((float)block->config.state_size);
    }

    /* Xavier uniform init for W_in (state_size x dim) */
    {
        float fan_in  = (float)block->W_in.cols;
        float fan_out = (float)block->W_in.rows;
        float scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_in.rows * block->W_in.cols; i++) {
            block->W_in.data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Xavier uniform init for W_out (dim x state_size) */
    {
        float fan_in  = (float)block->W_out.cols;
        float fan_out = (float)block->W_out.rows;
        float scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_out.rows * block->W_out.cols; i++) {
            block->W_out.data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Small uniform init for delta_proj (1 x dim) */
    for (size_t i = 0; i < block->delta_proj.rows * block->delta_proj.cols; i++) {
        block->delta_proj.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
}

void mb_compute_delta(float *delta_out, const MambaBlock *block,
                      const float *delta_in, size_t seq_len) {
    if (!delta_out || !block || !delta_in) return;

    for (size_t i = 0; i < seq_len; i++) {
        float delta_val = delta_in[i];
        delta_val = scalar_softplus(delta_val);

        if (delta_val < block->config.dt_min) {
            delta_val = block->config.dt_min;
        }
        if (delta_val > block->config.dt_max) {
            delta_val = block->config.dt_max;
        }

        delta_out[i] = delta_val;
    }
}

/* ============================================================================
 * Optimizer — MUONCLIP
 * ============================================================================ */

void mamba_attach_optimizer(MambaBlock *block, OptimizerType type, const MBOptimConfig *optconf) {
    if (!block) return;
    MBOptimState *s = (MBOptimState *)malloc(sizeof(MBOptimState));
    size_t dim = block->config.dim;
    size_t state = block->config.state_size;
    size_t size_in = state * dim;
    size_t size_out = dim * state;
    memset(s, 0, sizeof(MBOptimState));
    
    s->type = type;

    s->g_W_in = (float *)calloc(size_in, sizeof(float));
    s->g_W_out = (float *)calloc(size_out, sizeof(float));
    s->g_A_log = (float *)calloc(state, sizeof(float));
    s->g_B_mat = (float *)calloc(state, sizeof(float));
    s->g_C_mat = (float *)calloc(state, sizeof(float));
    s->g_delta_proj = (float *)calloc(dim, sizeof(float));

    /* Allocate first moments for all momentum-based optimizers */
    if (type == OPTIMIZER_ADAM_CLIP || type == OPTIMIZER_ADAMW ||
        type == OPTIMIZER_MUON      || type == OPTIMIZER_SGD) {
        s->m_W_in       = (float *)calloc(size_in,  sizeof(float));
        s->m_W_out      = (float *)calloc(size_out, sizeof(float));
        s->m_A_log      = (float *)calloc(state,    sizeof(float));
        s->m_B_mat      = (float *)calloc(state,    sizeof(float));
        s->m_C_mat      = (float *)calloc(state,    sizeof(float));
        s->m_delta_proj = (float *)calloc(dim,      sizeof(float));
    }
    /* Allocate second moments only for Adam-based optimizers */
    if (type == OPTIMIZER_ADAM_CLIP || type == OPTIMIZER_ADAMW) {
        s->v_W_in       = (float *)calloc(size_in,  sizeof(float));
        s->v_W_out      = (float *)calloc(size_out, sizeof(float));
        s->v_A_log      = (float *)calloc(state,    sizeof(float));
        s->v_B_mat      = (float *)calloc(state,    sizeof(float));
        s->v_C_mat      = (float *)calloc(state,    sizeof(float));
        s->v_delta_proj = (float *)calloc(dim,      sizeof(float));
    }

    s->step = 0;
    if (g_opt_n < 256) { g_opt_blocks[g_opt_n] = block; g_opt_states[g_opt_n] = s; g_opt_n++; }
    else free(s);
    (void)optconf;
}

void mamba_free_optimizer(MambaBlock *block) {
    _mamba_free_opt_for(block);
}

static void _mamba_free_opt_for(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            MBOptimState *s = g_opt_states[i];
            if (!s) return;
            free(s->g_W_in); free(s->g_W_out); free(s->g_A_log); free(s->g_B_mat); free(s->g_C_mat); free(s->g_delta_proj);
            
            /* Free moments only if allocated */
            if (s->m_W_in) { free(s->m_W_in); free(s->v_W_in); }
            if (s->m_W_out) { free(s->m_W_out); free(s->v_W_out); }
            if (s->m_A_log) { free(s->m_A_log); free(s->v_A_log); }
            if (s->m_B_mat) { free(s->m_B_mat); free(s->v_B_mat); }
            if (s->m_C_mat) { free(s->m_C_mat); free(s->v_C_mat); }
            if (s->m_delta_proj) { free(s->m_delta_proj); free(s->v_delta_proj); }
            
            free(s);
            for (size_t j = i; j + 1 < g_opt_n; j++) { g_opt_blocks[j] = g_opt_blocks[j+1]; g_opt_states[j] = g_opt_states[j+1]; }
            g_opt_n--;
            return;
        }
    }
}

void mamba_zero_grads(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            MBOptimState *s = g_opt_states[i];
            size_t dim = block->config.dim; size_t state = block->config.state_size;
            size_t size_in = state * dim; size_t size_out = dim * state;
            memset(s->g_W_in, 0, size_in * sizeof(float)); memset(s->g_W_out, 0, size_out * sizeof(float));
            memset(s->g_A_log, 0, state * sizeof(float)); memset(s->g_B_mat, 0, state * sizeof(float));
            memset(s->g_C_mat, 0, state * sizeof(float)); memset(s->g_delta_proj, 0, dim * sizeof(float));
            return;
        }
    }
}

static MBOptimState* _find_opt(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) if (g_opt_blocks[i] == block) return g_opt_states[i];
    return NULL;
}

/* ============================================================================
 * Optimizer Implementations
 * ============================================================================ */

static void adam_clip_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;

#ifdef KMAMBA_BUILD_CUDA
    adamw_update_cuda(block->W_in.data,       s->g_W_in,       s->m_W_in,       s->v_W_in,       size_in,   conf, s->step);
    adamw_update_cuda(block->W_out.data,      s->g_W_out,      s->m_W_out,      s->v_W_out,      size_out,  conf, s->step);
    adamw_update_cuda(block->A_log.data,      s->g_A_log,      s->m_A_log,      s->v_A_log,      state,     conf, s->step);
    adamw_update_cuda(block->B_mat.data,      s->g_B_mat,      s->m_B_mat,      s->v_B_mat,      state,     conf, s->step);
    adamw_update_cuda(block->C_mat.data,      s->g_C_mat,      s->m_C_mat,      s->v_C_mat,      state,     conf, s->step);
    adamw_update_cuda(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim,       conf, s->step);
#else
    float lr = conf->lr; float mu = conf->mu; float beta2 = conf->beta2;
    float eps = conf->eps; float clip = conf->clip_norm; float wd = conf->weight_decay;

#define ADAM_CLIP_UPDATE(param, grad, m, v, N) do { \
    if (clip > 0.0f) gradient_clip_inplace(grad, (N), clip); \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * g; \
        v[_i] = beta2 * v[_i] + (1.0f - beta2) * (g * g); \
        float m_hat = m[_i] / (1.0f - powf(mu, (float)s->step)); \
        float v_hat = v[_i] / (1.0f - powf(beta2, (float)s->step)); \
        param[_i] -= lr * m_hat / (sqrtf(v_hat) + eps); } \
    } while (0)

    ADAM_CLIP_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, size_in);
    ADAM_CLIP_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, size_out);
    ADAM_CLIP_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, state);
    ADAM_CLIP_UPDATE(block->B_mat.data, s->g_B_mat, s->m_B_mat, s->v_B_mat, state);
    ADAM_CLIP_UPDATE(block->C_mat.data, s->g_C_mat, s->m_C_mat, s->v_C_mat, state);
    ADAM_CLIP_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim);

#undef ADAM_CLIP_UPDATE
#endif
}

static void adamw_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;
    float lr = conf->lr; float mu = conf->mu; float beta2 = conf->beta2; 
    float eps = conf->eps; float wd = conf->weight_decay;

#define ADAMW_UPDATE(param, grad, m, v, N) do { \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * g; \
        v[_i] = beta2 * v[_i] + (1.0f - beta2) * (g * g); \
        float m_hat = m[_i] / (1.0f - powf(mu, (float)s->step)); \
        float v_hat = v[_i] / (1.0f - powf(beta2, (float)s->step)); \
        param[_i] = param[_i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps); } \
    } while (0)

    ADAMW_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, size_in);
    ADAMW_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, size_out);
    ADAMW_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, state);
    ADAMW_UPDATE(block->B_mat.data, s->g_B_mat, s->m_B_mat, s->v_B_mat, state);
    ADAMW_UPDATE(block->C_mat.data, s->g_C_mat, s->m_C_mat, s->v_C_mat, state);
    ADAMW_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim);

#undef ADAMW_UPDATE
}

static void sgd_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;
    float lr = conf->lr; float mu = conf->mu; float wd = conf->weight_decay;

#define SGD_UPDATE(param, grad, m, N) do { \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + g; \
        param[_i] -= lr * m[_i]; } \
    } while (0)

    SGD_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, size_in);
    SGD_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, size_out);
    SGD_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, state);
    SGD_UPDATE(block->B_mat.data, s->g_B_mat, s->m_B_mat, state);
    SGD_UPDATE(block->C_mat.data, s->g_C_mat, s->m_C_mat, state);
    SGD_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, dim);

#undef SGD_UPDATE
}

/* MUON : matrices (W_in, W_out) — Newton-Schulz + momentum + clipping */
static void muon_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;

#ifdef KMAMBA_BUILD_CUDA
    muon_update_mat_cuda(block->W_in.data,  s->g_W_in,  s->m_W_in,  state, dim,   conf);
    muon_update_mat_cuda(block->W_out.data, s->g_W_out, s->m_W_out, dim,   state, conf);
    muon_update_vec_cuda(block->A_log.data,      s->g_A_log,      s->m_A_log,      state, conf);
    muon_update_vec_cuda(block->B_mat.data,      s->g_B_mat,      s->m_B_mat,      state, conf);
    muon_update_vec_cuda(block->C_mat.data,      s->g_C_mat,      s->m_C_mat,      state, conf);
    muon_update_vec_cuda(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, dim,   conf);
#else
    float lr = conf->lr; float mu = conf->mu; float wd = conf->weight_decay;
    float clip = conf->clip_norm;

#define MUON_UPDATE_MAT(param, grad, m, rows, cols) do { \
    size_t _N = (rows) * (cols); \
    newton_schulz5_inplace(grad, rows, cols, 5); \
    for (size_t _i = 0; _i < _N; _i++) { \
        float _g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * _g; \
    } \
    if (clip > 0.0f) gradient_clip_inplace(m, _N, clip); \
    for (size_t _i = 0; _i < _N; _i++) param[_i] -= lr * m[_i]; \
    } while (0)

#define MUON_UPDATE_VEC(param, grad, m, N) do { \
    for (size_t _i = 0; _i < (N); _i++) { \
        float _g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * _g; \
    } \
    if (clip > 0.0f) gradient_clip_inplace(m, (N), clip); \
    for (size_t _i = 0; _i < (N); _i++) param[_i] -= lr * m[_i]; \
    } while (0)

    MUON_UPDATE_MAT(block->W_in.data,  s->g_W_in,  s->m_W_in,  state, dim);
    MUON_UPDATE_MAT(block->W_out.data, s->g_W_out, s->m_W_out, dim,   state);
    MUON_UPDATE_VEC(block->A_log.data,       s->g_A_log,       s->m_A_log,       state);
    MUON_UPDATE_VEC(block->B_mat.data,       s->g_B_mat,       s->m_B_mat,       state);
    MUON_UPDATE_VEC(block->C_mat.data,       s->g_C_mat,       s->m_C_mat,       state);
    MUON_UPDATE_VEC(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  dim);

#undef MUON_UPDATE_MAT
#undef MUON_UPDATE_VEC
#endif
}

void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf) {
    MBOptimState *s = _find_opt(block);
    if (!s) return;
    s->step += 1;

    switch (s->type) {
        case OPTIMIZER_ADAM_CLIP:
            adam_clip_update(block, s, conf);
            break;
        case OPTIMIZER_ADAMW:
            adamw_update(block, s, conf);
            break;
        case OPTIMIZER_SGD:
            sgd_update(block, s, conf);
            break;
        case OPTIMIZER_MUON:
            muon_update(block, s, conf);
            break;
        default:
            /* Default to ADAM_CLIP for safety */
            adam_clip_update(block, s, conf);
            break;
    }
}

/* ============================================================================
 * Forward scan with storage for backward
 * ============================================================================ */

static void selective_scan_forward_store(ForwardStore *store, float *state,
                    const float *input, const float *delta,
                    const MBMatrix *A_bar, const float *B_bar,
                    const MBMatrix *C, float D,
                    size_t seq_len, size_t state_size) {
    if (!store || !state) return;
    (void)C; (void)D;

    store->x = (float *)calloc(seq_len * state_size, sizeof(float));
    store->A_diag = (float *)calloc(seq_len * state_size, sizeof(float));
    store->B_bar = (float *)calloc(seq_len * state_size, sizeof(float));
    store->u_seq = (float *)calloc(seq_len * state_size, sizeof(float));

    memset(state, 0, state_size * sizeof(float));

    float *zero_prev = (float *)calloc(state_size, sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        const float *u_t = &input[t * state_size];
        float dt_t = delta[t];

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < state_size; i++) store->u_seq[t * state_size + i] = u_t[i];

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < state_size; i++) {
            /* Clamp A ≤ -1e-5 : garantit exp(dt*A) < 1 (scan stable)
             * même si AdamW pousse A_log vers des valeurs positives. */
            float a_val = A_bar->data[i * state_size + i];
            if (a_val > -1e-5f) a_val = -1e-5f;
            float a_diag_t = expf(dt_t * a_val);
            store->A_diag[t * state_size + i] = a_diag_t;
            store->B_bar[t * state_size + i] = dt_t * B_bar[i];
        }

        float *x_prev = (t == 0) ? zero_prev : &store->x[(t-1)*state_size];
        float *x_t = &store->x[t * state_size];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < state_size; i++) {
            float a_diag_t = store->A_diag[t * state_size + i];
            float bbar = store->B_bar[t * state_size + i];
            x_t[i] = a_diag_t * x_prev[i] + bbar * u_t[i];
            state[i] = x_t[i];
        }
    }

    free(zero_prev);
}

/* ============================================================================
 * Backward through stored forward trace
 * ============================================================================ */

static void selective_scan_backward(ForwardStore *store, MambaBlock *block,
                                    const float *dY, const float *input_flat,
                                    float *d_input_out,
                                    size_t seq_len, size_t state_size) {
    if (!store || !block) return;
    size_t dim = block->config.dim;

    MBOptimState *s = _find_opt(block);
    if (!s) return;

    float *scan_out = (float *)calloc(seq_len * state_size, sizeof(float));
    float *dY_T = (float *)calloc(dim * seq_len, sizeof(float));
    float *adj_y = (float *)calloc(seq_len * state_size, sizeof(float));
    float *scan_du = (float *)calloc(seq_len * state_size, sizeof(float));
    float *scan_dA = (float *)calloc(state_size, sizeof(float));
    float *scan_dB = (float *)calloc(state_size, sizeof(float));
    float *scan_dC = (float *)calloc(state_size, sizeof(float));
    float *scan_ddelta = (float *)calloc(seq_len, sizeof(float));
    float *contrib_T = (float *)calloc(state_size * seq_len, sizeof(float));
    float *z = (float *)malloc(state_size * sizeof(float));
    if (!scan_out || !dY_T || !adj_y || !scan_du || !scan_dA ||
        !scan_dB || !scan_dC || !scan_ddelta || !contrib_T || !z) {
        free(scan_out); free(dY_T); free(adj_y); free(scan_du); free(scan_dA);
        free(scan_dB); free(scan_dC); free(scan_ddelta); free(contrib_T); free(z);
        return;
    }

    for (size_t t = 0; t < seq_len; t++) {
        hadamard_avx2(store->x + t * state_size, block->C_mat.data,
                      scan_out + t * state_size, (long)state_size);
    }
    transpose_row_major(dY, dY_T, seq_len, dim);

    /* g_W_out += dY^T @ scan_out */
    gemm_avx2(dY_T, scan_out, s->g_W_out,
              (long)dim, (long)seq_len, (long)state_size);

    /* adj_y = dY @ W_out */
    gemm_avx2((float *)dY, block->W_out.data, adj_y,
              (long)seq_len, (long)dim, (long)state_size);

    {
        MambaScan1DBackwardM1Params bp = {
            .x = store->u_seq,
            .A = block->A_log.data,
            .A_diag = store->A_diag,
            .B = block->B_mat.data,
            .C = block->C_mat.data,
            .delta = block->delta,
            .h0 = NULL,
            .h = store->x,
            .dy = adj_y,
            .dx = scan_du,
            .dA = scan_dA,
            .dB = scan_dB,
            .dC = scan_dC,
            .ddelta = scan_ddelta,
            .L = (long)seq_len,
            .D = (long)state_size
        };
        mamba_scan1d_backward_m1_shared_bc(&bp);
    }

    for (size_t i = 0; i < state_size; i++) {
        s->g_A_log[i] += scan_dA[i];
        s->g_B_mat[i] += scan_dB[i];
        s->g_C_mat[i] += scan_dC[i];
    }

    for (size_t t = 0; t < seq_len; t++) {
        const float *x_input_t = &input_flat[t * dim];
        float ddt_t = scan_ddelta[t];

        {
            float raw_t = 0.0f;
            for (size_t k = 0; k < dim; k++) {
                raw_t += block->delta_proj.data[k] * x_input_t[k];
            }
            {
                /* Straight-through estimator pour le clamp :
                 * on propage le gradient même si delta est clampé.
                 * Sans ça, softplus(x) >= ln(2) > dt_max=0.1 toujours,
                 * et delta_proj ne recevrait jamais de gradient. */
                float draw = ddt_t * scalar_sigmoid(raw_t);
                for (size_t k = 0; k < dim; k++) {
                    s->g_delta_proj[k] += draw * x_input_t[k];
                }
            }
        }

        mb_matrix_vec_mult(z, &block->W_in, x_input_t);

        for (size_t j = 0; j < state_size; j++) {
            float sig = scalar_sigmoid(z[j]);
            float dz = sig * (1.0f + z[j] * (1.0f - sig));
            scan_out[t * state_size + j] = scan_du[t * state_size + j] * dz;
        }
    }

    transpose_row_major(scan_out, contrib_T, seq_len, state_size);
    gemm_avx2(contrib_T, (float *)input_flat, s->g_W_in,
              (long)state_size, (long)seq_len, (long)dim);

    /* d_input = d_z @ W_in  (seq_len x state_size) @ (state_size x dim) */
    if (d_input_out) {
        gemm_avx2(scan_out, block->W_in.data, d_input_out,
                  (long)seq_len, (long)state_size, (long)dim);
        /* Residual gradient: d_input += dY (identity path) */
        for (size_t i = 0; i < seq_len * dim; i++)
            d_input_out[i] += dY[i];
    }

    free(scan_out); free(dY_T); free(z); free(adj_y);
    free(scan_du); free(scan_dA); free(scan_dB); free(scan_dC);
    free(scan_ddelta); free(contrib_T);
}

/* ============================================================================
 * Backward entrypoint
 * ============================================================================ */

void mamba_backward(MambaBlock *block, const float *dY, const float *input,
                    float *d_input, size_t batch_index) {
    (void)batch_index;
    size_t seq_len = block->config.seq_len;
    size_t state_size = block->config.state_size;

    MBMatrix *A_bar = mb_matrix_create(state_size, state_size);
    for (size_t i = 0; i < state_size; i++) A_bar->data[i * state_size + i] = block->A_log.data[i];
    float *B_bar = (float *)malloc(state_size * sizeof(float));
    for (size_t i = 0; i < state_size; i++) B_bar[i] = block->B_mat.data[i];

    ForwardStore store;
    memset(&store, 0, sizeof(store));

    size_t dim = block->config.dim;
    float *u_seq = (float *)calloc(seq_len * state_size, sizeof(float));
    if (!u_seq) { mb_matrix_free(A_bar); free(B_bar); return; }

    float *tmp_delta = (float *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                        sizeof(float));
    float *z = (float *)malloc(state_size * sizeof(float));
    if (!tmp_delta || !z) {
        free(z); free(tmp_delta); free(u_seq);
        mb_matrix_free(A_bar); free(B_bar);
        return;
    }

    for (size_t t = 0; t < seq_len; t++) {
        const float *x_t = &input[t * dim];
        project_controller(block, x_t, z, &u_seq[t * state_size]);
        block->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);
    }
    free(z);
    free(tmp_delta);

    selective_scan_forward_store(&store, block->hidden, u_seq, block->delta,
                                A_bar, B_bar, &block->C_mat, 0.0f,
                                seq_len, state_size);

    selective_scan_backward(&store, block, dY, input, d_input, seq_len, state_size);

    free(store.x); free(store.A_diag); free(store.B_bar); free(store.u_seq); free(u_seq);
    mb_matrix_free(A_bar); free(B_bar);
}

/* ============================================================================
 * Forward pass — 1D
 * ============================================================================ */

void mamba_block_forward(MambaBlock *block, float *output, const float *input,
                         size_t batch_size) {
    if (!block || !output || !input) return;

    size_t seq_len = block->config.seq_len;
    size_t dim = block->config.dim;
    size_t state_size = block->config.state_size;

    for (size_t b = 0; b < batch_size; b++) {
        const float *batch_input = &input[b * seq_len * dim];
        float *batch_output = &output[b * seq_len * dim];

        float *u_seq = (float *)calloc(seq_len * state_size, sizeof(float));
        float *z = (float *)malloc(state_size * sizeof(float));
        if (!u_seq || !z) {
            free(z); free(u_seq);
            continue;
        }

        float *tmp_delta = (float *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                            sizeof(float));
        if (!tmp_delta) {
            free(z); free(u_seq);
            continue;
        }

        for (size_t t = 0; t < seq_len; t++) {
            const float *x_t = &batch_input[t * dim];
            project_controller(block, x_t, z, &u_seq[t * state_size]);
            block->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);
        }
        free(z);
        free(tmp_delta);

        float *scan_out = (float *)malloc(seq_len * state_size * sizeof(float));
        if (!scan_out) { free(u_seq); continue; }

        long L = (long)seq_len, D = (long)state_size;

        for (long t = 0; t < L; t++) {
            for (long d = 0; d < D; d++) {
                block->scan_B    [t*D + d] = block->B_mat.data[d];
                block->scan_C    [t*D + d] = block->C_mat.data[d];
                block->scan_delta[t*D + d] = block->delta[t];
            }
        }
        memset(block->scan_h, 0, (size_t)D * sizeof(float));

        MambaScan1DParams sp = {
            .x = u_seq, .A = block->A_log.data,
            .B = block->scan_B, .C = block->scan_C,
            .delta = block->scan_delta, .h = block->scan_h,
            .y = scan_out,
            .L = L, .D = D, .M = 1
        };
        mamba_scan1d_forward(&sp);
        memcpy(block->hidden, block->scan_h, (size_t)D * sizeof(float));

        float *ybuf = (float *)malloc(dim * sizeof(float));
        if (ybuf) {
            for (size_t t = 0; t < seq_len; t++) {
                const float *state_t = &scan_out[t * state_size];
                mb_matrix_vec_mult(ybuf, &block->W_out, state_t);
                /* Residual connection: output = input + mamba(input) */
                for (size_t j = 0; j < dim; j++)
                    batch_output[t * dim + j] = batch_input[t * dim + j] + ybuf[j];
            }
            free(ybuf);
        }

        free(u_seq);
        free(scan_out);
    }
}

