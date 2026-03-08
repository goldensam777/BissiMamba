#include "mamba.h"
#include "optimatrix_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward storage for training (per timestep) */
typedef struct {
    real_t *x;         /* seq_len x state_size */
    real_t *A_diag;    /* seq_len x state_size */
    real_t *B_bar;     /* seq_len x state_size */
    real_t *u_seq;     /* seq_len x state_size (controller vectors per timestep) */
} ForwardStore;

/* Global registry mapping MambaBlock -> OptimState */
static OptimState *g_opt_states[256];
static MambaBlock *g_opt_blocks[256];
static size_t g_opt_n = 0;
/* forward declarations */
static void _mamba_free_opt_for(MambaBlock *block);
static OptimState* _find_opt(MambaBlock *block);
static int matrix_init_owned(Matrix *dst, size_t rows, size_t cols);
static void project_controller(const MambaBlock *block, const real_t *x_t,
                               real_t *z_buf, real_t *u_out);
static real_t project_delta_value(const MambaBlock *block, const real_t *x_t,
                                  real_t *tmp_delta, size_t position,
                                  size_t total_positions);
static void transpose_row_major(const real_t *src, real_t *dst,
                                size_t rows, size_t cols);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

Matrix* matrix_create(size_t rows, size_t cols) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (real_t *)calloc(rows * cols, sizeof(real_t));
    
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    return m;
}

static int matrix_init_owned(Matrix *dst, size_t rows, size_t cols) {
    Matrix *m;

    if (!dst) return -1;
    m = matrix_create(rows, cols);
    if (!m) {
        memset(dst, 0, sizeof(*dst));
        return -1;
    }
    *dst = *m;
    free(m);
    return 0;
}

void matrix_free(Matrix *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void matrix_zero(Matrix *m) {
    if (!m || !m->data) return;
    memset(m->data, 0, m->rows * m->cols * sizeof(real_t));
}

void matrix_copy(Matrix *dst, const Matrix *src) {
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->rows != src->rows || dst->cols != src->cols) return;
    
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(real_t));
}

void matrix_print(const Matrix *m) {
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

void matrix_vec_mult(real_t *out, const Matrix *m, const real_t *v) {
    if (!out || !m || !v || !m->data) return;
    gemv_avx2(m->data, (real_t *)v, out, (long)m->rows, (long)m->cols);
}

void vec_add(real_t *y, const real_t *x, size_t n) {
    if (!y || !x) return;
    for (size_t i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void vec_scale(real_t *v, real_t alpha, size_t n) {
    if (!v) return;
    for (size_t i = 0; i < n; i++) {
        v[i] *= alpha;
    }
}

static void project_controller(const MambaBlock *block, const real_t *x_t,
                               real_t *z_buf, real_t *u_out) {
    if (!block || !x_t || !z_buf || !u_out) return;
    matrix_vec_mult(z_buf, &block->W_in, x_t);
    silu_f32(z_buf, u_out, (long)block->config.state_size);
}

static real_t project_delta_value(const MambaBlock *block, const real_t *x_t,
                                  real_t *tmp_delta, size_t position,
                                  size_t total_positions) {
    real_t dval;

    if (!block || !x_t || !tmp_delta || total_positions == 0) return 0.0f;

    if (block->delta_proj.rows > 0) {
        matrix_vec_mult(tmp_delta, &block->delta_proj, x_t);
        dval = softplus(tmp_delta[0]);
        if (dval < block->config.dt_min) dval = block->config.dt_min;
        if (dval > block->config.dt_max) dval = block->config.dt_max;
        return dval;
    }

    return block->config.dt_scale *
           ((real_t)position / (real_t)total_positions + 1.0f);
}

static void transpose_row_major(const real_t *src, real_t *dst,
                                size_t rows, size_t cols) {
    if (!src || !dst) return;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

real_t softplus(real_t x) {
    if (x > 20.0f) return x;  /* Avoid overflow */
    if (x < -20.0f) return 0.0f;  /* Avoid underflow */
    return logf(1.0f + expf(x));
}

real_t sigmoid(real_t x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

real_t relu(real_t x) {
    return x > 0.0f ? x : 0.0f;
}

/* ============================================================================
 * Discretization Functions
 * ============================================================================ */

/* Compute matrix exponential using Taylor series for small matrices
 * For diagonal matrices: exp(dt*A) = diag(exp(dt*a_ii))
 * Assumes A is provided as diagonal elements
 */
void discretize_A(Matrix *A_bar, const Matrix *A, real_t dt) {
    if (!A_bar || !A) return;
    
    /* For a diagonal state matrix (common in SSMs) */
    for (size_t i = 0; i < A_bar->rows && i < A->rows; i++) {
        real_t a_ii = A->data[i * A->cols + i];
        A_bar->data[i * A_bar->cols + i] = expf(dt * a_ii);
    }
}

/* Compute discretized B using forward Euler for simplicity
 * More accurate: B_bar = integral_0^dt exp(tau*A) d(tau) * B
 * Simplified: B_bar = dt * B (for small dt)
 */
void discretize_B(real_t *B_bar, const Matrix *A, const real_t *B, 
                  real_t dt, size_t state_size) {
    if (!B_bar || !A || !B) return;
    
    /* Simple approximation: scale B by dt */
    for (size_t i = 0; i < state_size; i++) {
        B_bar[i] = dt * B[i];
    }
}

/* ============================================================================
 * Selective Scan - Core Mamba Operation
 * ============================================================================ */

void selective_scan(real_t *output, real_t *state, 
                   const real_t *input, const real_t *delta,
                   const Matrix *A_bar, const real_t *B_bar,
                   const Matrix *C, real_t D,
                   size_t seq_len, size_t state_size) {
    
    if (!output || !state || !input || !delta || !A_bar || !B_bar || !C) {
        return;
    }
    (void)C; (void)D; /* currently unused: readout handled outside */
    
    /* Initialize state to zero */
    memset(state, 0, state_size * sizeof(real_t));

    /* Allocate temporaries once to avoid per-timestep malloc/free */
    real_t *temp_state = (real_t *)malloc(state_size * sizeof(real_t));
    real_t *A_diag_t = (real_t *)malloc(state_size * sizeof(real_t));
    real_t *B_bar_t = (real_t *)malloc(state_size * sizeof(real_t));
    if (!temp_state || !A_diag_t || !B_bar_t) {
        free(temp_state); free(A_diag_t); free(B_bar_t);
        return;
    }

    /* Process each timestep; input is flattened controller vectors (seq_len x state_size) */
    for (size_t t = 0; t < seq_len; t++) {
        const real_t *u_t = &input[t * state_size];
        real_t dt_t = delta[t];

        /* Compute A_diag_t and B_bar_t across state dimensions */
        for (size_t i = 0; i < state_size; i++) {
            real_t a_val = A_bar->data[i * state_size + i];
            real_t a_diag = expf(dt_t * a_val);
            A_diag_t[i] = a_diag;
            if (fabsl(a_val) < 1e-8) {
                B_bar_t[i] = dt_t * B_bar[i];
            } else {
                B_bar_t[i] = (a_diag - 1.0f) / a_val * B_bar[i];
            }
        }

        /* state = A_diag_t ⊙ state  (Hadamard AVX2) */
        hadamard_avx2(A_diag_t, state, temp_state, (long)state_size);

        /* temp_state now holds A_diag_t * prev_state
         * state = temp_state + B_bar_t ⊙ u_t */
        hadamard_avx2(B_bar_t, (real_t *)u_t, state, (long)state_size);
        vec_add(state, temp_state, state_size);

        /* Write state vector into output buffer (flattened) */
        memcpy(&output[t * state_size], state, state_size * sizeof(real_t));
    }

    free(A_diag_t); free(B_bar_t); free(temp_state);
}

/* ============================================================================
 * Mamba Block Operations
 * ============================================================================ */

MambaBlock* mamba_block_create(const MambaConfig *config) {
    if (!config) return NULL;
    
    MambaBlock *block = (MambaBlock *)malloc(sizeof(MambaBlock));
    if (!block) return NULL;
    memset(block, 0, sizeof(*block));
    
    block->config = *config;
    
    /* Allocate matrices */
    /* W_in: state_size x dim  (maps input -> controller vector)
       W_out: dim x state_size (maps state -> output) */
    if (matrix_init_owned(&block->W_in, config->state_size, config->dim) != 0 ||
        matrix_init_owned(&block->W_out, config->dim, config->state_size) != 0 ||
        matrix_init_owned(&block->A_log, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->B_mat, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->C_mat, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->delta_proj, 1, config->dim) != 0) {
        mamba_block_free(block);
        return NULL;
    }
    
    /* Allocate temporary buffers */
    block->hidden = (real_t *)calloc(config->state_size, sizeof(real_t));
    block->delta  = (real_t *)calloc(config->seq_len, sizeof(real_t));

    /* Pre-allocate scan1d adapter buffers */
    size_t LD = config->seq_len * config->state_size;
    block->scan_B     = (real_t *)malloc(LD * sizeof(real_t));
    block->scan_C     = (real_t *)malloc(LD * sizeof(real_t));
    block->scan_delta = (real_t *)malloc(LD * sizeof(real_t));
    block->scan_h     = (real_t *)calloc(config->state_size, sizeof(real_t));

    if (!block->W_in.data || !block->W_out.data || !block->A_log.data ||
        !block->B_mat.data || !block->C_mat.data || !block->delta_proj.data ||
        !block->hidden || !block->delta ||
        !block->scan_B || !block->scan_C || !block->scan_delta || !block->scan_h) {
        mamba_block_free(block);
        return NULL;
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
    /* free optimizer/grad buffers if present */
    /* Those are managed by mamba_free_optimizer through block pointer if used */
    
    free(block);
}

/* Initialize Mamba block with standard parameters */
void mamba_block_init(MambaBlock *block) {
    if (!block) return;
    
    /* Initialize A with stable values (negative real parts for stability) */
    for (size_t i = 0; i < block->config.state_size; i++) {
        /* Logarithmically spaced values for numerical stability */
        real_t spacing = (real_t)(i + 1) / (real_t)block->config.state_size;
        block->A_log.data[i] = -expf(spacing * logf(block->config.dt_scale));
    }
    
    /* Initialize B uniformly */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->B_mat.data[i] = 1.0f / sqrtf((real_t)block->config.state_size);
    }
    
    /* Initialize C uniformly */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->C_mat.data[i] = 1.0f / sqrtf((real_t)block->config.state_size);
    }
    
    /* Xavier uniform init for W_in (state_size x dim) */
    {
        real_t fan_in  = (real_t)block->W_in.cols;  /* dim */
        real_t fan_out = (real_t)block->W_in.rows;  /* state_size */
        real_t scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_in.rows * block->W_in.cols; i++) {
            block->W_in.data[i] = ((real_t)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Xavier uniform init for W_out (dim x state_size) */
    {
        real_t fan_in  = (real_t)block->W_out.cols;  /* state_size */
        real_t fan_out = (real_t)block->W_out.rows;  /* dim */
        real_t scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_out.rows * block->W_out.cols; i++) {
            block->W_out.data[i] = ((real_t)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Small uniform init for delta_proj (1 x dim) */
    for (size_t i = 0; i < block->delta_proj.rows * block->delta_proj.cols; i++) {
        block->delta_proj.data[i] = ((real_t)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
}

void compute_delta(real_t *delta_out, const MambaBlock *block, 
                   const real_t *delta_in, size_t seq_len) {
    if (!delta_out || !block || !delta_in) return;
    
    /* Apply softplus to ensure positive deltas */
    for (size_t i = 0; i < seq_len; i++) {
        real_t delta_val = delta_in[i];
        delta_val = softplus(delta_val);
        
        /* Clamp to valid range */
        if (delta_val < block->config.dt_min) {
            delta_val = block->config.dt_min;
        }
        if (delta_val > block->config.dt_max) {
            delta_val = block->config.dt_max;
        }
        
        delta_out[i] = delta_val;
    }
}

/* Allocate and attach optimizer state to block */
void mamba_attach_optimizer(MambaBlock *block, const OptimConfig *optconf) {
    if (!block) return;
    OptimState *s = (OptimState *)malloc(sizeof(OptimState));
    size_t dim = block->config.dim;
    size_t state = block->config.state_size;
    size_t size_in = state * dim;       /* W_in size */
    size_t size_out = dim * state;      /* W_out size */
    memset(s, 0, sizeof(OptimState));

    /* allocate gradient buffers */
    s->g_W_in = (real_t *)calloc(size_in, sizeof(real_t));
    s->g_W_out = (real_t *)calloc(size_out, sizeof(real_t));
    s->g_A_log = (real_t *)calloc(state, sizeof(real_t));
    s->g_B_mat = (real_t *)calloc(state, sizeof(real_t));
    s->g_C_mat = (real_t *)calloc(state, sizeof(real_t));
    s->g_delta_proj = (real_t *)calloc(dim, sizeof(real_t));

    /* optimizer moments */
    s->m_W_in = (real_t *)calloc(size_in, sizeof(real_t)); s->v_W_in = (real_t *)calloc(size_in, sizeof(real_t));
    s->m_W_out = (real_t *)calloc(size_out, sizeof(real_t)); s->v_W_out = (real_t *)calloc(size_out, sizeof(real_t));
    s->m_A_log = (real_t *)calloc(state, sizeof(real_t)); s->v_A_log = (real_t *)calloc(state, sizeof(real_t));
    s->m_B_mat = (real_t *)calloc(state, sizeof(real_t)); s->v_B_mat = (real_t *)calloc(state, sizeof(real_t));
    s->m_C_mat = (real_t *)calloc(state, sizeof(real_t)); s->v_C_mat = (real_t *)calloc(state, sizeof(real_t));
    s->m_delta_proj = (real_t *)calloc(dim, sizeof(real_t)); s->v_delta_proj = (real_t *)calloc(dim, sizeof(real_t));

    s->step = 0;
    s->step = 0;
    /* register in global registry */
    if (g_opt_n < 256) { g_opt_blocks[g_opt_n] = block; g_opt_states[g_opt_n] = s; g_opt_n++; }
    else free(s);
    (void)optconf; /* unused for now */
}

/* free optimizer state (best-effort) */
void mamba_free_optimizer(MambaBlock *block) {
    /* find in global registry and free */
    _mamba_free_opt_for(block);
}

/* internal helper to free optimizer map entry */
void _mamba_free_opt_for(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            OptimState *s = g_opt_states[i];
            if (!s) return;
            free(s->g_W_in); free(s->g_W_out); free(s->g_A_log); free(s->g_B_mat); free(s->g_C_mat); free(s->g_delta_proj);
            free(s->m_W_in); free(s->v_W_in); free(s->m_W_out); free(s->v_W_out);
            free(s->m_A_log); free(s->v_A_log); free(s->m_B_mat); free(s->v_B_mat);
            free(s->m_C_mat); free(s->v_C_mat); free(s->m_delta_proj); free(s->v_delta_proj);
            free(s);
            /* remove entry */
            for (size_t j = i; j + 1 < g_opt_n; j++) { g_opt_blocks[j] = g_opt_blocks[j+1]; g_opt_states[j] = g_opt_states[j+1]; }
            g_opt_n--;
            return;
        }
    }
}

/* zero gradients (best-effort using global map) */
void mamba_zero_grads(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            OptimState *s = g_opt_states[i];
            size_t dim = block->config.dim; size_t state = block->config.state_size;
            size_t size_in = state * dim; size_t size_out = dim * state;
            memset(s->g_W_in, 0, size_in * sizeof(real_t)); memset(s->g_W_out, 0, size_out * sizeof(real_t));
            memset(s->g_A_log, 0, state * sizeof(real_t)); memset(s->g_B_mat, 0, state * sizeof(real_t));
            memset(s->g_C_mat, 0, state * sizeof(real_t)); memset(s->g_delta_proj, 0, dim * sizeof(real_t));
            return;
        }
    }
}

/* Helper: find optimizer state for block */
OptimState* _find_opt(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) if (g_opt_blocks[i] == block) return g_opt_states[i];
    return NULL;
}

/* Simple MUONCLIP optimizer step operating on each parameter buffer using its stored moments */
void mamba_optimizer_step(MambaBlock *block, const OptimConfig *conf) {
    OptimState *s = _find_opt(block);
    if (!s) return;
    s->step += 1;
    real_t lr = conf->lr; real_t mu = conf->mu; real_t beta2 = conf->beta2; real_t eps = conf->eps; real_t clip = conf->clip_norm; real_t wd = conf->weight_decay;

    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;

    /* helper macro to update a parameter buffer */
#define MUONCLIP_UPDATE(param, grad, m, v, N) do { \
    /* compute global norm and clip */ \
    double sq = 0.0; for (size_t _i=0; _i < (N); _i++) { double g = (double)(grad[_i]); sq += g*g; } \
    double gn = sqrt(sq); double scale = 1.0; if (gn > clip && clip>0.0) scale = clip / gn; \
    for (size_t _i=0; _i < (N); _i++) { real_t g = grad[_i] * (real_t)scale + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * g; \
        v[_i] = beta2 * v[_i] + (1.0f - beta2) * (g * g); \
        real_t m_hat = m[_i] / (1.0f - powf(mu, (real_t)s->step)); \
        real_t v_hat = v[_i] / (1.0f - powf(beta2, (real_t)s->step)); \
        param[_i] -= lr * m_hat / (sqrtf(v_hat) + eps); } \
    } while (0)

    MUONCLIP_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, size_in);
    MUONCLIP_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, size_out);
    MUONCLIP_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, state);
    MUONCLIP_UPDATE(block->B_mat.data, s->g_B_mat, s->m_B_mat, s->v_B_mat, state);
    MUONCLIP_UPDATE(block->C_mat.data, s->g_C_mat, s->m_C_mat, s->v_C_mat, state);
    MUONCLIP_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim);

#undef MUONCLIP_UPDATE
}

/* ------------------------------------------------------------------
 * Forward scan that stores per-timestep values for backward.
 * Math is aligned with scan1d ASM: dA = exp(dt * A), dB = dt * B.
 * ------------------------------------------------------------------ */
void selective_scan_forward_store(ForwardStore *store, real_t *state, 
                   const real_t *input, const real_t *delta,
                   const Matrix *A_bar, const real_t *B_bar,
                   const Matrix *C, real_t D,
                   size_t seq_len, size_t state_size) {
    if (!store || !state) return;
    (void)C; (void)D;
    /* allocate storage arrays */
    store->x = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    store->A_diag = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    store->B_bar = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    store->u_seq = (real_t *)calloc(seq_len * state_size, sizeof(real_t));

    memset(state, 0, state_size * sizeof(real_t));

    /* allocate zero prev buffer to avoid per-step calloc/free */
    real_t *zero_prev = (real_t *)calloc(state_size, sizeof(real_t));

    for (size_t t = 0; t < seq_len; t++) {
        const real_t *u_t = &input[t * state_size];
        real_t dt_t = delta[t];
        /* copy controller vector */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) store->u_seq[t * state_size + i] = u_t[i];

        /* dA and dB per-dim (parallelizable), matching scan1d */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) {
            real_t a_val = A_bar->data[i * state_size + i];
            real_t a_diag_t = expf(dt_t * a_val);
            store->A_diag[t * state_size + i] = a_diag_t;
            store->B_bar[t * state_size + i] = dt_t * B_bar[i];
        }

        /* update state (elementwise) */
        real_t *x_prev = (t == 0) ? zero_prev : &store->x[(t-1)*state_size];
        real_t *x_t = &store->x[t * state_size];
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) {
            real_t a_diag_t = store->A_diag[t * state_size + i];
            real_t bbar = store->B_bar[t * state_size + i];
            x_t[i] = a_diag_t * x_prev[i] + bbar * u_t[i];
            state[i] = x_t[i];
        }
    }

    free(zero_prev);
}

/* Backward through stored forward trace.
 * dY is flattened [seq_len, dim] gradient wrt final block outputs.
 */
void selective_scan_backward(ForwardStore *store, MambaBlock *block, const real_t *dY,
                             const real_t *input_flat, size_t seq_len, size_t state_size) {
    if (!store || !block) return;
    size_t dim = block->config.dim;
    /* find optimizer state */
    OptimState *s = _find_opt(block);
    if (!s) return;

    /* scan_out[t,i] = C_i * h_t_i */
    real_t *scan_out = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    real_t *dY_T = (real_t *)calloc(dim * seq_len, sizeof(real_t));
    /* adj_y: dL/d(scan_out), where scan_out[t,i] = C_i * h_t_i */
    real_t *adj_y = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    real_t *scan_du = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    real_t *scan_dA = (real_t *)calloc(state_size, sizeof(real_t));
    real_t *scan_dB = (real_t *)calloc(state_size, sizeof(real_t));
    real_t *scan_dC = (real_t *)calloc(state_size, sizeof(real_t));
    real_t *scan_ddelta = (real_t *)calloc(seq_len, sizeof(real_t));
    real_t *contrib_T = (real_t *)calloc(state_size * seq_len, sizeof(real_t));
    real_t *z = (real_t *)malloc(state_size * sizeof(real_t));
    if (!scan_out || !dY_T || !adj_y || !scan_du || !scan_dA ||
        !scan_dB || !scan_dC || !scan_ddelta || !contrib_T || !z) {
        free(scan_out);
        free(dY_T);
        free(adj_y);
        free(scan_du);
        free(scan_dA);
        free(scan_dB);
        free(scan_dC);
        free(scan_ddelta);
        free(contrib_T);
        free(z);
        return;
    }

    /* Build scan_out rows and transpose dY once for GEMM-based gradient accumulation. */
    for (size_t t = 0; t < seq_len; t++) {
        hadamard_avx2(store->x + t * state_size, block->C_mat.data,
                      scan_out + t * state_size, (long)state_size);
    }
    transpose_row_major(dY, dY_T, seq_len, dim);

    /* g_W_out += dY^T @ scan_out */
    gemm_avx2(dY_T, scan_out, s->g_W_out,
              (long)dim, (long)seq_len, (long)state_size);

    /* adj_y = dY @ W_out */
    gemm_avx2((real_t *)dY, block->W_out.data, adj_y,
              (long)seq_len, (long)dim, (long)state_size);

    {
        ScanBackwardSharedParams bp = {
            .x = store->u_seq,
            .A = block->A_log.data,
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
        scan1d_backward_m1_shared_bc(&bp);
    }

    for (size_t i = 0; i < state_size; i++) {
        s->g_A_log[i] += scan_dA[i];
        s->g_B_mat[i] += scan_dB[i];
        s->g_C_mat[i] += scan_dC[i];
    }

    for (size_t t = 0; t < seq_len; t++) {
        const real_t *x_input_t = &input_flat[t * dim];
        real_t ddt_t = scan_ddelta[t];

        /* dt_t = clamp(softplus(raw_t)); raw_t = delta_proj @ x_t */
        {
            real_t raw_t = 0.0f;
            for (size_t k = 0; k < dim; k++) {
                raw_t += block->delta_proj.data[k] * x_input_t[k];
            }
            {
                real_t sp = softplus(raw_t);
                if (sp > block->config.dt_min && sp < block->config.dt_max) {
                    real_t draw = ddt_t * sigmoid(raw_t);
                    for (size_t k = 0; k < dim; k++) {
                        s->g_delta_proj[k] += draw * x_input_t[k];
                    }
                }
            }
        }

        /* reconstruct pre-activation z = W_in @ x_input_t (length state_size) */
        matrix_vec_mult(z, &block->W_in, x_input_t);

        for (size_t j = 0; j < state_size; j++) {
            /* silu'(z) */
            real_t sig = sigmoid(z[j]);
            real_t dz = sig * (1.0f + z[j] * (1.0f - sig));
            scan_out[t * state_size + j] = scan_du[t * state_size + j] * dz;
        }
    }

    transpose_row_major(scan_out, contrib_T, seq_len, state_size);
    gemm_avx2(contrib_T, (real_t *)input_flat, s->g_W_in,
              (long)state_size, (long)seq_len, (long)dim);

    free(scan_out);
    free(dY_T);
    free(z);
    free(adj_y);
    free(scan_du);
    free(scan_dA);
    free(scan_dB);
    free(scan_dC);
    free(scan_ddelta);
    free(contrib_T);
}

/* Backward entrypoint: compute gradients for a single batch element (batch_index unused here as we perform batch_size=1 in examples) */
void mamba_backward(MambaBlock *block, const real_t *dY, const real_t *input, size_t batch_index) {
    (void)batch_index;
    size_t seq_len = block->config.seq_len;
    size_t state_size = block->config.state_size;

    /* Re-run forward to capture traces, aligned with scan1d forward math. */
    Matrix *A_bar = matrix_create(state_size, state_size);
    for (size_t i = 0; i < state_size; i++) A_bar->data[i * state_size + i] = block->A_log.data[i];
    real_t *B_bar = (real_t *)malloc(state_size * sizeof(real_t));
    for (size_t i = 0; i < state_size; i++) B_bar[i] = block->B_mat.data[i];

    ForwardStore store;
    memset(&store, 0, sizeof(store));

    /* compute u_seq from inputs: vector controller of size state_size per timestep */
    size_t dim = block->config.dim;
    real_t *u_seq = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    if (!u_seq) { matrix_free(A_bar); free(B_bar); return; }

    /* temporary buffer for delta projection (delta_proj has rows=1) */
    real_t *tmp_delta = (real_t *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                         sizeof(real_t));
    real_t *z = (real_t *)malloc(state_size * sizeof(real_t));
    if (!tmp_delta || !z) {
        free(z);
        free(tmp_delta);
        free(u_seq);
        matrix_free(A_bar);
        free(B_bar);
        return;
    }

    for (size_t t = 0; t < seq_len; t++) {
        const real_t *x_t = &input[t * dim];
        project_controller(block, x_t, z, &u_seq[t * state_size]);
        block->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);
    }
    free(z);
    free(tmp_delta);

    selective_scan_forward_store(&store, block->hidden, u_seq, block->delta, A_bar, B_bar, &block->C_mat, 0.0f, seq_len, state_size);

    /* call backward on stored trace */
    selective_scan_backward(&store, block, dY, input, seq_len, state_size);

    /* free store */
    free(store.x); free(store.A_diag); free(store.B_bar); free(store.u_seq); free(u_seq);
    matrix_free(A_bar); free(B_bar);
}

void mamba_forward(MambaBlock *block, real_t *output, const real_t *input, 
                   size_t batch_size) {
    if (!block || !output || !input) return;
    
    size_t seq_len = block->config.seq_len;
    size_t dim = block->config.dim;
    size_t state_size = block->config.state_size;
    
    /* Process each sequence in batch */
    for (size_t b = 0; b < batch_size; b++) {
        const real_t *batch_input = &input[b * seq_len * dim];
        real_t *batch_output = &output[b * seq_len * dim];
        
        /* Project input: compute vector controller u_seq (seq_len x state_size) */
        real_t *u_seq = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
        real_t *z = (real_t *)malloc(state_size * sizeof(real_t));
        if (!u_seq || !z) {
            free(z);
            free(u_seq);
            continue;
        }

        /* temporary for delta projection */
        real_t *tmp_delta = (real_t *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                             sizeof(real_t));
        if (!tmp_delta) {
            free(z);
            free(u_seq);
            continue;
        }

        for (size_t t = 0; t < seq_len; t++) {
            const real_t *x_t = &batch_input[t * dim];
            project_controller(block, x_t, z, &u_seq[t * state_size]);
            block->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);
        }
        free(z);
        free(tmp_delta);

        /* Selective scan via scan1d ASM kernel — L=seq_len, D=state_size, M=1 */
        real_t *scan_out = (real_t *)malloc(seq_len * state_size * sizeof(real_t));
        if (!scan_out) { free(u_seq); continue; }

        long L = (long)seq_len, D = (long)state_size;

        /* Remplir les buffers pré-alloués */
        for (long t = 0; t < L; t++) {
            for (long d = 0; d < D; d++) {
                block->scan_B    [t*D + d] = block->B_mat.data[d];
                block->scan_C    [t*D + d] = block->C_mat.data[d];
                block->scan_delta[t*D + d] = block->delta[t];
            }
        }
        memset(block->scan_h, 0, (size_t)D * sizeof(real_t));

        ScanParams sp = {
            .x = u_seq, .A = block->A_log.data,
            .B = block->scan_B, .C = block->scan_C,
            .delta = block->scan_delta, .h = block->scan_h,
            .y = scan_out,
            .L = L, .D = D, .M = 1
        };
        scan1d(&sp);
        memcpy(block->hidden, block->scan_h, (size_t)D * sizeof(real_t));
        
        /* Output projection: y_t = W_out @ state_t (scan_out contains state vectors) */
        real_t *ybuf = (real_t *)malloc(dim * sizeof(real_t));
        if (ybuf) {
            for (size_t t = 0; t < seq_len; t++) {
                const real_t *state_t = &scan_out[t * state_size];
                matrix_vec_mult(ybuf, &block->W_out, state_t); /* ybuf length = dim */
                for (size_t j = 0; j < dim; j++) batch_output[t * dim + j] = ybuf[j];
            }
            free(ybuf);
        }
        
        free(u_seq);
        free(scan_out);
    }
}

/* ============================================================================
 * Forward 2D — Mamba sur grille [d1, d2, dim] via scan2d ASM (wavefront)
 * ============================================================================ */
void mamba_forward_2d(MambaBlock *block, real_t *output, const real_t *input,
                      size_t d1, size_t d2) {
    if (!block || !output || !input) return;

    size_t dim  = block->config.dim;
    size_t D    = block->config.state_size;
    size_t P    = d1 * d2;   /* nombre total de positions */
    long   M    = 1;

    /* --- 1. Projection d'entrée : u[p, D] = silu(W_in @ x[p, dim]) --- */
    real_t *u          = (real_t *)calloc(P * D, sizeof(real_t));
    real_t *delta_pos  = (real_t *)calloc(P,     sizeof(real_t));
    real_t *tmp_delta  = (real_t *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                          sizeof(real_t));
    real_t *z          = (real_t *)malloc(D * sizeof(real_t));

    if (!u || !delta_pos || !tmp_delta || !z) {
        free(u); free(delta_pos); free(tmp_delta); free(z);
        return;
    }

    for (size_t p = 0; p < P; p++) {
        const real_t *x_p = &input[p * dim];
        real_t *u_p       = &u[p * D];

        project_controller(block, x_p, z, u_p);
        delta_pos[p] = project_delta_value(block, x_p, tmp_delta, p, P);
    }
    free(z);
    free(tmp_delta);

    /* --- 2. Préparer les tableaux pour scan2d --- */
    real_t *B_s  = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *C_s  = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *d1_s = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *d2_s = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *h_s  = (real_t *)calloc(P * D, sizeof(real_t));  /* tous les états */
    real_t *y_s  = (real_t *)malloc(P * D * sizeof(real_t));

    if (!B_s || !C_s || !d1_s || !d2_s || !h_s || !y_s) {
        free(B_s); free(C_s); free(d1_s); free(d2_s); free(h_s); free(y_s);
        free(u); free(delta_pos);
        return;
    }

    for (size_t p = 0; p < P; p++) {
        for (size_t d = 0; d < D; d++) {
            B_s [p*D + d] = block->B_mat.data[d];
            C_s [p*D + d] = 1.0f;
            d1_s[p*D + d] = delta_pos[p];
            d2_s[p*D + d] = delta_pos[p];
        }
    }
    free(delta_pos);

    /* --- 3. Lancer scan2d --- */
    Scan2DParams sp = {
        .x      = u,
        .A1     = block->A_log.data,
        .A2     = block->A_log.data,
        .B      = B_s,
        .C      = C_s,
        .delta1 = d1_s,
        .delta2 = d2_s,
        .h      = h_s,
        .y      = y_s,
        .d1     = (long)d1,
        .d2     = (long)d2,
        .D      = (long)D,
        .M      = M
    };
    scan2d(&sp);

    /* --- 4. Projection de sortie : output[p, dim] = W_out @ y[p, D] --- */
    real_t *ybuf = (real_t *)malloc(dim * sizeof(real_t));
    if (ybuf) {
        for (size_t p = 0; p < P; p++) {
            matrix_vec_mult(ybuf, &block->W_out, &y_s[p * D]);
            memcpy(&output[p * dim], ybuf, dim * sizeof(real_t));
        }
        free(ybuf);
    }

    free(u); free(B_s); free(C_s); free(d1_s); free(d2_s); free(h_s); free(y_s);
}

/* ============================================================================
 * Backward 2D — Rétropropagation à travers mamba_forward_2d
 *
 * dY    : [d1, d2, dim] — gradient de la loss sur la sortie
 * input : [d1, d2, dim] — entrée originale
 * ============================================================================ */
void mamba_backward_2d(MambaBlock *block, const real_t *dY, const real_t *input,
                       size_t d1, size_t d2) {
    if (!block || !dY || !input) return;

    size_t dim = block->config.dim;
    size_t D   = block->config.state_size;
    size_t P   = d1 * d2;

    OptimState *s = _find_opt(block);
    if (!s) return;

    /* --- Re-run forward pour capturer u, delta_pos, h_all --- */
    real_t *u          = (real_t *)calloc(P * D, sizeof(real_t));
    real_t *delta_pos  = (real_t *)calloc(P,     sizeof(real_t));
    real_t *tmp_delta  = (real_t *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                          sizeof(real_t));
    real_t *B_s        = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *C_s        = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *d1_s       = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *d2_s       = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *h_all      = (real_t *)calloc(P * D, sizeof(real_t));
    real_t *y_scan     = (real_t *)malloc(P * D * sizeof(real_t));
    real_t *z          = (real_t *)malloc(D * sizeof(real_t));

    if (!u || !delta_pos || !tmp_delta || !B_s || !C_s ||
        !d1_s || !d2_s || !h_all || !y_scan || !z) {
        free(u); free(delta_pos); free(tmp_delta);
        free(B_s); free(C_s); free(d1_s); free(d2_s);
        free(h_all); free(y_scan); free(z);
        return;
    }

    /* Projection d'entrée */
    for (size_t p = 0; p < P; p++) {
        const real_t *x_p = &input[p * dim];
        project_controller(block, x_p, z, &u[p * D]);
        delta_pos[p] = project_delta_value(block, x_p, tmp_delta, p, P);
    }
    free(z);
    free(tmp_delta);

    for (size_t p = 0; p < P; p++) {
        for (size_t d = 0; d < D; d++) {
            B_s [p*D + d] = block->B_mat.data[d];
            C_s [p*D + d] = 1.0f;
            d1_s[p*D + d] = delta_pos[p];
            d2_s[p*D + d] = delta_pos[p];
        }
    }

    Scan2DParams sp = {
        .x = u, .A1 = block->A_log.data, .A2 = block->A_log.data,
        .B = B_s, .C = C_s, .delta1 = d1_s, .delta2 = d2_s,
        .h = h_all, .y = y_scan,
        .d1 = (long)d1, .d2 = (long)d2, .D = (long)D, .M = 1
    };
    scan2d(&sp);
    free(B_s); free(C_s); free(d1_s); free(d2_s);

    /* --- Backward W_out + initialiser adj_h --- */
    real_t *adj_h = (real_t *)calloc(P * D, sizeof(real_t));
    real_t *adj_u = (real_t *)calloc(P * D, sizeof(real_t));
    if (!adj_h || !adj_u) {
        free(adj_h); free(adj_u);
        free(u); free(delta_pos); free(h_all); free(y_scan);
        return;
    }

    for (size_t p = 0; p < P; p++) {
        for (size_t j = 0; j < dim; j++) {
            real_t dy = dY[p * dim + j];
            for (size_t d = 0; d < D; d++) {
                s->g_W_out[j * D + d] += dy * y_scan[p * D + d];
                adj_h[p * D + d]      += dy * block->W_out.data[j * D + d];
            }
        }
    }
    free(y_scan);

    /* --- Backward scan2d : wavefront inverse k = (d1+d2-2) → 0 --- */
    for (long k = (long)(d1 + d2 - 2); k >= 0; k--) {
        long i_min = k - (long)d2 + 1; if (i_min < 0) i_min = 0;
        long i_max = k;                 if (i_max > (long)d1 - 1) i_max = (long)d1 - 1;

        for (long i = i_min; i <= i_max; i++) {
            long j = k - i;
            size_t p = (size_t)(i * (long)d2 + j);

            real_t dt = delta_pos[p];

            for (size_t d = 0; d < D; d++) {
                real_t ah  = adj_h[p * D + d];
                if (ah == 0.0f) continue;

                real_t a_val = block->A_log.data[d];
                real_t dA    = expf(dt * a_val);

                real_t h_prev1 = (i > 0) ? h_all[((size_t)(i-1) * d2 + (size_t)j) * D + d] : 0.0f;
                real_t h_prev2 = (j > 0) ? h_all[((size_t)i * d2 + (size_t)(j-1)) * D + d] : 0.0f;

                /* grad A_log — A1 et A2 sont le même buffer */
                s->g_A_log[d] += ah * dt * dA * (h_prev1 + h_prev2);

                /* propagation aux prédécesseurs */
                if (i > 0) adj_h[((size_t)(i-1) * d2 + (size_t)j) * D + d] += ah * dA;
                if (j > 0) adj_h[((size_t)i * d2 + (size_t)(j-1)) * D + d] += ah * dA;

                /* grad B_mat : dB = dt * B[d], d(h)/d(B[d]) = dt * u[p,d] */
                s->g_B_mat[d] += ah * dt * u[p * D + d];

                /* adj_u : d(h)/d(u[p,d]) = dt * B[d] */
                adj_u[p * D + d] += ah * dt * block->B_mat.data[d];
            }
        }
    }
    free(h_all); free(delta_pos);

    /* --- Backward SiLU + W_in --- */
    z = (real_t *)malloc(D * sizeof(real_t));
    if (!z) {
        free(u); free(adj_h); free(adj_u);
        return;
    }

    for (size_t p = 0; p < P; p++) {
        const real_t *x_p = &input[p * dim];
        matrix_vec_mult(z, &block->W_in, x_p);

        for (size_t d = 0; d < D; d++) {
            real_t sig    = sigmoid(z[d]);
            real_t dsilu  = sig * (1.0f + z[d] * (1.0f - sig));
            real_t contrib = adj_u[p * D + d] * dsilu;
            for (size_t kk = 0; kk < dim; kk++)
                s->g_W_in[d * dim + kk] += contrib * x_p[kk];
        }
    }

    free(z);
    free(u); free(adj_h); free(adj_u);
}
