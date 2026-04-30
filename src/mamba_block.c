#include "kmamba.h"
#include "kmamba_kernels.h"
#include "scan_nd.h"
#include "km_memory_pool.h"
#include <stdint.h>

#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "scan_nd.h"

void cuda_block_forward(
    cublasHandle_t cublas,
    const float *d_W_in, const float *d_W_out, const float *d_A_log,
    const float *d_W_B, const float *d_W_C, const float *d_delta_proj,
    const float *d_theta, const float *d_lambda_proj,
    const float *d_x, float *d_y,
    float *d_u_raw, float *d_u, float *d_dt_raw, float *d_dt,
    float *d_B_exp, float *d_C_exp, float *d_dt_exp,
    float *d_h_store, float *d_y_scan, float *d_y_proj,
    float *d_lambda_raw, float *d_lambda,
    int L, int state, int dim, int R);

void cuda_block_backward(
    cublasHandle_t cublas,
    const float *d_W_in, const float *d_W_out, const float *d_A_log,
    const float *d_W_B, const float *d_W_C, const float *d_delta_proj,
    const float *d_theta, const float *d_lambda_proj,
    const float *d_x, const float *d_dy,
    float *d_dW_in, float *d_dW_out, float *d_dA_log,
    float *d_dW_B, float *d_dW_C, float *d_ddelta_proj,
    float *d_dtheta, float *d_dlambda_proj,
    float *d_dx,
    float *d_u_raw, float *d_u, float *d_dt_raw, float *d_dt,
    float *d_dB_scan, float *d_dC_scan, float *d_ddt_scan,
    float *d_dy_scan, float *d_dA_tmp,
    float *d_dlambda, float *d_dlambda_raw,
    int L, int state, int dim, int R);
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline size_t _mimo_R(const MBConfig *cfg) {
    return (cfg->mimo_rank > 1) ? cfg->mimo_rank : 1;
}

static int matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols) {
    if (!dst) return -1;
    dst->rows = rows; dst->cols = cols;
    dst->data = malloc(rows * cols * sizeof(float));
    if (!dst->data) return -1;
    memset(dst->data, 0, rows * cols * sizeof(float));
    return 0;
}

MBMatrix* mb_matrix_create(size_t rows, size_t cols) {
    MBMatrix *m = (MBMatrix *)malloc(sizeof(MBMatrix));
    if (!m) return NULL;
    if (matrix_init_owned(m, rows, cols) != 0) { free(m); return NULL; }
    return m;
}

void mb_matrix_free(MBMatrix *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void mb_matrix_copy(MBMatrix *dst, const MBMatrix *src) {
    if (!dst || !src || !dst->data || !src->data || dst->rows != src->rows || dst->cols != src->cols) return;
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

MambaBlockWorkspace* mamba_block_workspace_create(const MambaBlock *block) {
    if (!block) return NULL;
    MambaBlockWorkspace *ws = (MambaBlockWorkspace *)calloc(1, sizeof(*ws));
    if (!ws) return NULL;
    size_t N = block->config.state_size, L = block->config.seq_len, D = block->config.dim, R = _mimo_R(&block->config), NR = N * R;
    ws->hidden = malloc(N * sizeof(float)); ws->delta = malloc(L * sizeof(float)); 
    ws->scan_B = malloc(L * NR * sizeof(float)); 
    ws->scan_C = malloc(L * NR * sizeof(float)); 
    ws->scan_delta = malloc(L * N * sizeof(float)); ws->scan_h = malloc(N * sizeof(float));
    ws->h_rot = malloc(N * sizeof(float)); ws->Bu_cur = malloc(N * sizeof(float)); ws->prev_Bu = malloc(N * sizeof(float)); ws->z_buf = malloc(D * sizeof(float)); ws->u_seq = malloc(L * R * sizeof(float)); ws->lambda_seq = malloc(L * sizeof(float)); ws->y_rank = malloc(L * R * sizeof(float)); ws->y_proj = malloc(D * sizeof(float)); 
    ws->h_seq = malloc(L * N * sizeof(float)); 
    ws->Bu_seq = malloc(L * N * sizeof(float));
    ws->d_y_rank = malloc(L * R * sizeof(float)); ws->d_scan_B = malloc(L * NR * sizeof(float)); ws->d_scan_C = malloc(L * NR * sizeof(float)); ws->d_delta = malloc(L * sizeof(float)); ws->d_lambda = malloc(L * sizeof(float)); ws->raw_delta = malloc(L * sizeof(float));
    if (!ws->hidden || !ws->delta || !ws->scan_B || !ws->scan_C || !ws->scan_delta || !ws->scan_h || !ws->h_rot || !ws->Bu_cur || !ws->prev_Bu || !ws->z_buf || !ws->u_seq || !ws->lambda_seq || !ws->y_rank || !ws->y_proj || !ws->h_seq || !ws->Bu_seq || !ws->d_y_rank || !ws->d_scan_B || !ws->d_scan_C || !ws->d_delta || !ws->d_lambda || !ws->raw_delta) { mamba_block_workspace_free(ws); return NULL; }
    memset(ws->hidden, 0, N * sizeof(float)); memset(ws->delta, 0, L * sizeof(float)); memset(ws->scan_h, 0, N * sizeof(float));
    return ws;
}

void mamba_block_workspace_free(MambaBlockWorkspace *ws) {
    if (!ws) return;
    free(ws->hidden); free(ws->delta); free(ws->scan_B); free(ws->scan_C); free(ws->scan_delta); free(ws->scan_h); free(ws->h_rot); free(ws->Bu_cur); free(ws->prev_Bu); free(ws->z_buf); free(ws->u_seq); free(ws->lambda_seq); free(ws->y_rank); free(ws->y_proj); free(ws->h_seq); free(ws->Bu_seq); free(ws->d_y_rank); free(ws->d_scan_B); free(ws->d_scan_C); free(ws->d_delta); free(ws->d_lambda); free(ws->raw_delta);
    free(ws);
}

typedef struct { float delta; float raw_delta; float lambda; } Mamba3Control;

static void project_controller_full(const MambaBlock *block, const float *x_t, float *z_buf, float *u_out, Mamba3Control *ctrl) {
    if (!block || !x_t || !z_buf || !u_out || !ctrl) return;
    size_t R = _mimo_R(&block->config), D = block->config.dim;
    gemv_f32(block->W_in.data, x_t, z_buf, (int)R, (int)D); silu_f32(z_buf, u_out, (int)R);
    if (block->delta_proj.rows > 0 && block->delta_proj.data) {
        float dval; gemv_f32(block->delta_proj.data, x_t, &dval, 1, (int)D);
        ctrl->raw_delta = dval;  // save raw before softplus
        softplus_f32(&dval, &dval, 1);
        if (dval < block->config.dt_min) dval = block->config.dt_min;
        if (dval > block->config.dt_max) dval = block->config.dt_max;
        ctrl->delta = dval;
    } else {
        ctrl->raw_delta = block->config.dt_scale;
        ctrl->delta = block->config.dt_scale;
    }
    if (block->lambda_proj.rows > 0 && block->lambda_proj.data) {
        float lambda_raw; gemv_f32(block->lambda_proj.data, x_t, &lambda_raw, 1, (int)D);
        ctrl->lambda = 1.0f / (1.0f + expf(-lambda_raw));
    } else { ctrl->lambda = block->config.default_lambda; }
}

MambaBlock* mamba_block_create(const MBConfig *config) {
    if (!config) return NULL;
    MambaBlock *block = (MambaBlock *)calloc(1, sizeof(MambaBlock));
    if (!block) return NULL;
    block->config = *config;
    if (block->config.max_ndims <= 0 || block->config.max_ndims > KMAMBA_CONFIG_MAX_NDIMS) {
        fprintf(stderr, "mamba_block_create: invalid max_ndims=%ld (1..%d required)\n",
                block->config.max_ndims, KMAMBA_CONFIG_MAX_NDIMS);
        free(block);
        return NULL;
    }
    if (block->config.max_state <= 0) {
        fprintf(stderr, "mamba_block_create: invalid max_state=%ld (>0 required)\n", block->config.max_state);
        free(block);
        return NULL;
    }
    if (block->config.mimo_rank == 0) {
        fprintf(stderr, "mamba_block_create: mimo_rank must be >= 1\n");
        free(block);
        return NULL;
    }
    if (block->config.default_lambda < 0.0f || block->config.default_lambda > 1.0f) {
        fprintf(stderr, "mamba_block_create: invalid default_lambda=%f (expected in [0,1])\n",
                block->config.default_lambda);
        free(block);
        return NULL;
    }
    if (block->config.use_a_log_clamp && block->config.a_log_min == 0.0f) {
        fprintf(stderr, "mamba_block_create: a_log_min must be set when use_a_log_clamp=1\n");
        free(block);
        return NULL;
    }
    if (km_normalize_spatial_topology(&block->config.spatial_ndims, block->config.spatial_dims, block->config.seq_len, block->config.use_convnd, &block->config.convnd_ndims, block->config.convnd_K) != 0) { free(block); return NULL; }
    if (block->config.spatial_ndims > block->config.max_ndims) { free(block); return NULL; }
    size_t R = _mimo_R(&block->config), N = block->config.state_size, D = block->config.dim;
    if (matrix_init_owned(&block->W_in, R, D) != 0 || matrix_init_owned(&block->W_out, D, R) != 0 || matrix_init_owned(&block->A_log, N, 1) != 0 || matrix_init_owned(&block->W_B, N * R, D) != 0 || matrix_init_owned(&block->W_C, N * R, D) != 0 || matrix_init_owned(&block->delta_proj, 1, D) != 0 || matrix_init_owned(&block->lambda_proj, 1, D) != 0) { mamba_block_free(block); return NULL; }
    block->b_B = malloc(N * R * sizeof(float)); block->b_C = malloc(N * R * sizeof(float)); block->theta = malloc((N / 2 > 0 ? N / 2 : 1) * sizeof(float)); block->hidden = malloc(N * sizeof(float)); block->delta = malloc(block->config.seq_len * sizeof(float)); block->scan_B = malloc(block->config.seq_len * N * R * sizeof(float)); block->scan_C = malloc(block->config.seq_len * N * R * sizeof(float)); block->scan_delta = malloc(block->config.seq_len * N * sizeof(float)); block->scan_h = malloc(N * sizeof(float));
    block->wavefront_plan = km_wavefront_plan_create(
        block->config.spatial_dims,
        block->config.spatial_ndims,
        (block->config.max_state > 0) ? block->config.max_state : (long)block->config.state_size);
    if (!block->b_B || !block->b_C || !block->theta || !block->hidden || !block->delta || !block->scan_B || !block->scan_C || !block->scan_delta || !block->scan_h || !block->wavefront_plan) { mamba_block_free(block); return NULL; }
    if (block->config.use_convnd && block->config.convnd_K > 0) {
        long kernel_size = block->config.convnd_ndims * block->config.convnd_K * (long)block->config.dim;
        block->convnd_kernel = malloc(kernel_size * sizeof(float));
        if (!block->convnd_kernel) { mamba_block_free(block); return NULL; }
        if (block->config.use_convnd_bias) {
            block->convnd_bias = malloc(block->config.dim * sizeof(float));
            if (!block->convnd_bias) { mamba_block_free(block); return NULL; }
        }
    }
    return block;
}

void mamba_free_optimizer(MambaBlock *block) {
    if (!block || !block->opt_state) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    free(s->g_W_in); free(s->g_W_out); free(s->g_A_log); free(s->g_W_B); free(s->g_W_C);
    free(s->g_b_B); free(s->g_b_C); free(s->g_delta_proj); free(s->g_lambda_proj); free(s->g_theta);
    if (s->m_W_in) {
        free(s->m_W_in); free(s->m_W_out); free(s->m_A_log); free(s->m_W_B); free(s->m_W_C);
        free(s->m_b_B); free(s->m_b_C); free(s->m_delta_proj); free(s->m_lambda_proj); free(s->m_theta);
    }
    if (s->v_W_in) {
        free(s->v_W_in); free(s->v_W_out); free(s->v_A_log); free(s->v_W_B); free(s->v_W_C);
        free(s->v_b_B); free(s->v_b_C); free(s->v_delta_proj); free(s->v_lambda_proj); free(s->v_theta);
    }
    free(s); block->opt_state = NULL;
}

void mamba_block_free(MambaBlock *block) {
    if (!block) return;
    mamba_free_optimizer(block);
    if (block->W_in.data) free(block->W_in.data);
    if (block->W_out.data) free(block->W_out.data);
    if (block->A_log.data) free(block->A_log.data);
    if (block->W_B.data) free(block->W_B.data);
    if (block->W_C.data) free(block->W_C.data);
    if (block->delta_proj.data) free(block->delta_proj.data);
    if (block->lambda_proj.data) free(block->lambda_proj.data);
    free(block->b_B); free(block->b_C); free(block->theta);
    free(block->hidden); free(block->delta); free(block->scan_B);
    free(block->scan_C); free(block->scan_delta); free(block->scan_h);
    if (block->wavefront_plan) km_wavefront_plan_free(block->wavefront_plan);
    free(block->convnd_kernel); free(block->convnd_bias);
#ifdef KMAMBA_BUILD_CUDA
    if (block->gpu.gpu_ready) {
        cudaFree(block->gpu.d_W_in); cudaFree(block->gpu.d_W_out); cudaFree(block->gpu.d_A_log);
        cudaFree(block->gpu.d_W_B); cudaFree(block->gpu.d_W_C); cudaFree(block->gpu.d_delta_proj);
        cudaFree(block->gpu.d_theta); cudaFree(block->gpu.d_lambda_proj);
        cudaFree(block->gpu.d_b_B); cudaFree(block->gpu.d_b_C);
        cudaFree(block->gpu.d_u_raw); cudaFree(block->gpu.d_u); cudaFree(block->gpu.d_dt_raw);
        cudaFree(block->gpu.d_dt); cudaFree(block->gpu.d_B_exp); cudaFree(block->gpu.d_C_exp);
        cudaFree(block->gpu.d_h_store); cudaFree(block->gpu.d_y_scan); cudaFree(block->gpu.d_y_proj);
        cudaFree(block->gpu.d_lambda_raw); cudaFree(block->gpu.d_lambda);
    }
#endif
    free(block);
}

void mamba_block_init(MambaBlock *block) {
    if (!block) return;
    /*
     * Use a per-call Xorshift64 state seeded from the block's dimensions so
     * initialization is reproducible and thread-safe (no shared srand() state).
     */
    uint64_t rng;
    {
        uint64_t z = (uint64_t)block->config.dim
                   ^ ((uint64_t)block->config.state_size << 16)
                   ^ ((uint64_t)block->config.seq_len    << 32)
                   ^ 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng = z ^ (z >> 31);
    }

    /* Inline xorshift64 step — avoids static helper visibility issues */
#define XS64_NEXT(s) ((s) ^= (s) << 13, (s) ^= (s) >> 7, (s) ^= (s) << 17, (s))
#define XS64_FLOAT(s) ((float)((double)(XS64_NEXT(s) >> 11) * (1.0 / (double)(1ULL << 53))))

    for (size_t i = 0; i < block->config.state_size; i++) block->A_log.data[i] = -1.0f;
    MBMatrix *mats[] = { &block->W_in, &block->W_out, &block->W_B, &block->W_C };
    for (int mi = 0; mi < 4; mi++) {
        MBMatrix *M = mats[mi];
        float scale = sqrtf(6.0f / (float)(M->rows + M->cols));
        for (size_t i = 0; i < M->rows * M->cols; i++)
            M->data[i] = (XS64_FLOAT(rng) * 2.0f - 1.0f) * scale;
    }
    size_t R = _mimo_R(&block->config), N = block->config.state_size;
    memset(block->b_B, 0, N * R * sizeof(float));
    memset(block->b_C, 0, N * R * sizeof(float));
    for (size_t i = 0; i < (N/2 > 0 ? N/2 : 1); i++)
        block->theta[i] = XS64_FLOAT(rng) * (2.0f * 3.14159f / (float)N);
    for (size_t i = 0; i < block->delta_proj.rows * block->delta_proj.cols; i++)
        block->delta_proj.data[i] = (XS64_FLOAT(rng) - 0.5f) * 0.02f;
    for (size_t i = 0; i < block->lambda_proj.rows * block->lambda_proj.cols; i++)
        block->lambda_proj.data[i] = (XS64_FLOAT(rng) - 0.5f) * 0.02f;

#undef XS64_NEXT
#undef XS64_FLOAT
}

void mamba_attach_optimizer(MambaBlock *block, OptimizerType type, const MBOptimConfig *optconf) {
    (void)optconf; if (!block) return;
    mamba_free_optimizer(block);
    MBOptimState *s = (MBOptimState *)calloc(1, sizeof(MBOptimState));
    if (!s) return;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R, TS = N/2 > 0 ? N/2 : 1;
    s->type = type;
    s->g_W_in = (float *)calloc(R * D, sizeof(float)); s->g_W_out = (float *)calloc(D * R, sizeof(float));
    s->g_A_log = (float *)calloc(N, sizeof(float)); s->g_W_B = (float *)calloc(NR * D, sizeof(float));
    s->g_W_C = (float *)calloc(NR * D, sizeof(float)); s->g_b_B = (float *)calloc(NR, sizeof(float));
    s->g_b_C = (float *)calloc(NR, sizeof(float)); s->g_delta_proj = (float *)calloc(D, sizeof(float));
    s->g_lambda_proj = (float *)calloc(D, sizeof(float)); s->g_theta = (float *)calloc(TS, sizeof(float));
    if (type != OPTIMIZER_SGD) {
        s->m_W_in = (float *)calloc(R * D, sizeof(float)); s->m_W_out = (float *)calloc(D * R, sizeof(float));
        s->m_A_log = (float *)calloc(N, sizeof(float)); s->m_W_B = (float *)calloc(NR * D, sizeof(float));
        s->m_W_C = (float *)calloc(NR * D, sizeof(float)); s->m_b_B = (float *)calloc(NR, sizeof(float));
        s->m_b_C = (float *)calloc(NR, sizeof(float)); s->m_delta_proj = (float *)calloc(D, sizeof(float));
        s->m_lambda_proj = (float *)calloc(D, sizeof(float)); s->m_theta = (float *)calloc(TS, sizeof(float));
        if (type == OPTIMIZER_ADAMW || type == OPTIMIZER_ADAM_CLIP) {
            s->v_W_in = (float *)calloc(R * D, sizeof(float)); s->v_W_out = (float *)calloc(D * R, sizeof(float));
            s->v_A_log = (float *)calloc(N, sizeof(float)); s->v_W_B = (float *)calloc(NR * D, sizeof(float));
            s->v_W_C = (float *)calloc(NR * D, sizeof(float)); s->v_b_B = (float *)calloc(NR, sizeof(float));
            s->v_b_C = (float *)calloc(NR, sizeof(float)); s->v_delta_proj = (float *)calloc(D, sizeof(float));
            s->v_lambda_proj = (float *)calloc(D, sizeof(float)); s->v_theta = (float *)calloc(TS, sizeof(float));
        }
    }
    block->opt_state = s;
}

void mamba_zero_grads(MambaBlock *block) {
    if (!block || !block->opt_state) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    memset(s->g_W_in, 0, R * D * sizeof(float)); memset(s->g_W_out, 0, D * R * sizeof(float));
    memset(s->g_A_log, 0, N * sizeof(float)); memset(s->g_W_B, 0, NR * D * sizeof(float));
    memset(s->g_W_C, 0, NR * D * sizeof(float)); memset(s->g_b_B, 0, NR * sizeof(float));
    memset(s->g_b_C, 0, NR * sizeof(float)); memset(s->g_delta_proj, 0, D * sizeof(float));
    memset(s->g_lambda_proj, 0, D * sizeof(float)); memset(s->g_theta, 0, (N/2>0?N/2:1) * sizeof(float));
#ifdef KMAMBA_BUILD_CUDA
    /* Mirror the CPU zero on the GPU gradient buffers so that the GPU
     * backward pass starts accumulating from zero every step. */
    if (block->gpu.gpu_ready) {
        size_t TS = (N / 2 > 0 ? N / 2 : 1);
        cudaMemset(block->gpu.d_g_W_in,        0, R  * D  * sizeof(float));
        cudaMemset(block->gpu.d_g_W_out,        0, D  * R  * sizeof(float));
        cudaMemset(block->gpu.d_g_A_log,        0, N        * sizeof(float));
        cudaMemset(block->gpu.d_g_W_B,          0, NR * D  * sizeof(float));
        cudaMemset(block->gpu.d_g_W_C,          0, NR * D  * sizeof(float));
        cudaMemset(block->gpu.d_g_b_B,          0, NR       * sizeof(float));
        cudaMemset(block->gpu.d_g_b_C,          0, NR       * sizeof(float));
        cudaMemset(block->gpu.d_g_delta_proj,   0, D        * sizeof(float));
        cudaMemset(block->gpu.d_g_lambda_proj,  0, D        * sizeof(float));
        if (block->gpu.d_g_theta)
            cudaMemset(block->gpu.d_g_theta,    0, TS       * sizeof(float));
    }
#endif
}

float mamba_block_grad_sqnorm(const MambaBlock *block) {
    if (!block || !block->opt_state) return 0.0f;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R, TS = N/2 > 0 ? N/2 : 1;
    double acc = 0.0; float n;
    n = gradient_norm_f32(s->g_W_in, (int)(R*D)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_W_out, (int)(D*R)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_A_log, (int)N); acc += (double)n*n;
    n = gradient_norm_f32(s->g_W_B, (int)(NR*D)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_W_C, (int)(NR*D)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_b_B, (int)NR); acc += (double)n*n;
    n = gradient_norm_f32(s->g_b_C, (int)NR); acc += (double)n*n;
    n = gradient_norm_f32(s->g_delta_proj, (int)D); acc += (double)n*n;
    n = gradient_norm_f32(s->g_lambda_proj, (int)D); acc += (double)n*n;
    n = gradient_norm_f32(s->g_theta, (int)TS); acc += (double)n*n;
    return (float)acc;
}

void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf) {
    if (!block || !block->opt_state || !conf) return;
    if (conf->mu <= 0.0f || conf->mu >= 1.0f ||
        conf->beta2 <= 0.0f || conf->beta2 >= 1.0f ||
        conf->eps <= 0.0f) {
        fprintf(stderr, "mamba_optimizer_step: invalid optimizer config (mu,beta2 in (0,1), eps>0 required)\n");
        return;
    }
    MBOptimState *s = (MBOptimState *)block->opt_state; s->step++;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R, TS = N/2 > 0 ? N/2 : 1;
    const float adam_beta1 = conf->mu;
    const float adam_beta2 = conf->beta2;
    const float adam_eps = conf->eps;
    if (s->type == OPTIMIZER_MUON) {
        muon_update_mat_f32(block->W_in.data, s->g_W_in, s->m_W_in, R, D, conf, (int)s->step);
        muon_update_mat_f32(block->W_out.data, s->g_W_out, s->m_W_out, D, R, conf, (int)s->step);
        muon_update_vec_f32(block->A_log.data, s->g_A_log, s->m_A_log, N, conf, (int)s->step);
        muon_update_mat_f32(block->W_B.data, s->g_W_B, s->m_W_B, NR, D, conf, (int)s->step);
        muon_update_mat_f32(block->W_C.data, s->g_W_C, s->m_W_C, NR, D, conf, (int)s->step);
        muon_update_vec_f32(block->b_B, s->g_b_B, s->m_b_B, NR, conf, (int)s->step);
        muon_update_vec_f32(block->b_C, s->g_b_C, s->m_b_C, NR, conf, (int)s->step);
        muon_update_vec_f32(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, D, conf, (int)s->step);
        muon_update_vec_f32(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, D, conf, (int)s->step);
        muon_update_vec_f32(block->theta, s->g_theta, s->m_theta, TS, conf, (int)s->step);
    } else {
        adamw_step_f32(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)(R*D), (int)s->step);
        adamw_step_f32(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)(D*R), (int)s->step);
        adamw_step_f32(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)N, (int)s->step);
        adamw_step_f32(block->W_B.data, s->g_W_B, s->m_W_B, s->v_W_B, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)(NR*D), (int)s->step);
        adamw_step_f32(block->W_C.data, s->g_W_C, s->m_W_C, s->v_W_C, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)(NR*D), (int)s->step);
        adamw_step_f32(block->b_B, s->g_b_B, s->m_b_B, s->v_b_B, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)NR, (int)s->step);
        adamw_step_f32(block->b_C, s->g_b_C, s->m_b_C, s->v_b_C, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)NR, (int)s->step);
        adamw_step_f32(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)D, (int)s->step);
        adamw_step_f32(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, s->v_lambda_proj, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)D, (int)s->step);
        adamw_step_f32(block->theta, s->g_theta, s->m_theta, s->v_theta, conf->lr, adam_beta1, adam_beta2, adam_eps, conf->weight_decay, (int)TS, (int)s->step);
    }
}

static void _ssm_scan_forward(MambaBlock *block, MambaBlockWorkspace *ws, const float *u_seq, const float *lambda_seq, float *y_rank) {
    size_t L = block->config.seq_len, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R; float *h_cur = ws->scan_h, *h_rot = ws->h_rot, *Bu_cur = ws->Bu_cur, *prev_Bu = ws->prev_Bu; memset(h_cur, 0, N * sizeof(float)); memset(prev_Bu, 0, N * sizeof(float));
    for (size_t t = 0; t < L; t++) {
        float dt_t = ws->delta[t], lam_t = lambda_seq ? lambda_seq[t] : block->config.default_lambda; const float *b_t = &ws->scan_B[t * NR], *c_t = &ws->scan_C[t * NR], *u_t = &u_seq[t * R];
        for (size_t i = 0; i + 1 < N; i += 2) { float th = block->theta[i >> 1], cv = cosf(th), sv = sinf(th), h0 = h_cur[i], h1 = h_cur[i+1]; h_rot[i] = cv * h0 - sv * h1; h_rot[i+1] = sv * h0 + cv * h1; } if (N & 1) h_rot[N-1] = h_cur[N-1];
        memset(Bu_cur, 0, N * sizeof(float)); for (size_t r = 0; r < R; r++) { for (size_t n = 0; n < N; n++) Bu_cur[n] += b_t[r * N + n] * u_t[r]; } memcpy(&ws->Bu_seq[t * N], Bu_cur, N * sizeof(float));
        for (size_t n = 0; n < N; n++) { float alpha = km_scan_exp(dt_t * block->A_log.data[n], block->config.use_fast_exp), beta = (1.0f - lam_t) * dt_t * alpha, gamma = lam_t * dt_t; h_cur[n] = alpha * h_rot[n] + beta * prev_Bu[n] + gamma * Bu_cur[n]; } memcpy(&ws->h_seq[t * N], h_cur, N * sizeof(float));
        for (size_t r = 0; r < R; r++) { float yr = 0.0f; for (size_t n = 0; n < N; n++) yr += c_t[r * N + n] * h_cur[n]; y_rank[t * R + r] = yr; } memcpy(prev_Bu, Bu_cur, N * sizeof(float));
    }
}

/* Forward Wavefront Callback Context */
typedef struct {
    MambaBlock *block;
    MambaBlockWorkspace *ws;
    const float *u_seq;
    const float *lambda_seq;
    float *y_rank;
    const long *strides;
} SSMScanForwardCtx;

static void _ssm_scan_forward_callback(long t, long level, void *userdata) {
    SSMScanForwardCtx *ctx = (SSMScanForwardCtx *)userdata;
    MambaBlock *block = ctx->block;
    MambaBlockWorkspace *ws = ctx->ws;
    const size_t N = block->config.state_size;
    const size_t R = (block->config.mimo_rank > 1) ? block->config.mimo_rank : 1;
    const long ndims = block->config.spatial_ndims;
    const long *dims = block->config.spatial_dims;
    const long *strides = ctx->strides;

    (void)level;

    float dt = ws->delta[t];
    float lam = ctx->lambda_seq ? ctx->lambda_seq[t] : block->config.default_lambda;
    const float *b_t = &ws->scan_B[t * N * R];
    const float *c_t = &ws->scan_C[t * N * R];
    const float *u_t = &ctx->u_seq[t * R];

    /* coords for boundary check */
    long coords[KMAMBA_CONFIG_MAX_NDIMS];
    km_unravel_index(t, dims, strides, ndims, coords);

    for (size_t r = 0; r < R; r++) ctx->y_rank[t * R + r] = 0.0f;
    for (size_t n = 0; n < N; n++) {
        float bu_cur_n = 0.0f;
        float h_new_n = 0.0f;
        float bu_prev_acc_n = 0.0f;
        int n_pred = 0;
        float a_val = block->A_log.data[n];
        float alpha = km_scan_exp(dt * a_val, block->config.use_fast_exp);
        for (size_t r = 0; r < R; r++) {
            bu_cur_n += b_t[r * N + n] * u_t[r];
        }
        ws->Bu_seq[t * N + n] = bu_cur_n;

        for (long ax = 0; ax < ndims; ax++) {
            if (coords[ax] > 0) {
                long prev_t = t - strides[ax];
                const float *h_prev = &ws->h_seq[prev_t * N];
                float h_prev_rot_n;
                if ((n & 1U) && n > 0) {
                    float th = block->theta[n >> 1];
                    h_prev_rot_n = sinf(th) * h_prev[n - 1] + cosf(th) * h_prev[n];
                } else if (((n + 1) < N)) {
                    float th = block->theta[n >> 1];
                    h_prev_rot_n = cosf(th) * h_prev[n] - sinf(th) * h_prev[n + 1];
                } else {
                    h_prev_rot_n = h_prev[n];
                }
                h_new_n += alpha * h_prev_rot_n;
                bu_prev_acc_n += ws->Bu_seq[prev_t * N + n];
                n_pred++;
            }
        }
        if (n_pred > 0) {
            float beta_scale = (1.0f - lam) * dt / (float)n_pred;
            h_new_n += beta_scale * alpha * bu_prev_acc_n;
        }
        h_new_n += lam * dt * bu_cur_n;
        ws->h_seq[t * N + n] = h_new_n;

        for (size_t r = 0; r < R; r++) {
            ctx->y_rank[t * R + r] += c_t[r * N + n] * h_new_n;
        }
    }
}

/* Wavefront version for ND spatial processing */
static void _ssm_scan_forward_wavefront(MambaBlock *block,
                                        MambaBlockWorkspace *ws,
                                        const float *u_seq,
                                        const float *lambda_seq,
                                        float *y_rank) {
    const size_t N = block->config.state_size;
    const long total_points = block->wavefront_plan->total_points;
    const long ndims = block->config.spatial_ndims;
    const long *dims = block->config.spatial_dims;

    long *strides = (long *)km_pool_alloc(km_memory_pool_threadlocal(),
                                          (size_t)ndims * sizeof(long));
    if (!strides) return;
    if (!km_make_row_major_strides(dims, ndims, strides)) {
        km_pool_free(km_memory_pool_threadlocal(), strides);
        return;
    }

    /* Zero h_seq and Bu_seq for all positions */
    memset(ws->h_seq,  0, (size_t)total_points * N * sizeof(float));
    memset(ws->Bu_seq, 0, (size_t)total_points * N * sizeof(float));

    SSMScanForwardCtx ctx;
    ctx.block = block;
    ctx.ws = ws;
    ctx.u_seq = u_seq;
    ctx.lambda_seq = lambda_seq;
    ctx.y_rank = y_rank;
    ctx.strides = strides;

    /* Forward through wavefront levels using plan primitive */
    km_wavefront_plan_iter_forward(block->wavefront_plan, _ssm_scan_forward_callback, &ctx);

    km_pool_free(km_memory_pool_threadlocal(), strides);
}

void mamba_block_forward_ws(MambaBlock *block, MambaBlockWorkspace *ws, float *output, const float *input, size_t batch_size) {
    if (!block || !ws || !output || !input) return;
    size_t L = block->config.seq_len, D = block->config.dim, N = block->config.state_size;
    size_t R = _mimo_R(&block->config), NR = N * R;
    /* Buffer size checks */
    size_t scan_B_size = L * D * N;  /* Allocated size */
    size_t scan_B_access_max = (L - 1) * NR + (NR - 1);  /* Max accessed index */
    (void)scan_B_size; (void)scan_B_access_max;
    for (size_t b = 0; b < batch_size; b++) {
        const float *in = &input[b * L * D]; float *out = &output[b * L * D];
        for (size_t t = 0; t < L; t++) { Mamba3Control ctrl = {0.0f, 0.0f, block->config.default_lambda}; project_controller_full(block, &in[t * D], ws->z_buf, &ws->u_seq[t * R], &ctrl); ws->delta[t] = ctrl.delta; ws->raw_delta[t] = ctrl.raw_delta; ws->lambda_seq[t] = ctrl.lambda; gemv_f32(block->W_B.data, &in[t * D], &ws->scan_B[t * NR], (int)NR, (int)D); gemv_f32(block->W_C.data, &in[t * D], &ws->scan_C[t * NR], (int)NR, (int)D); for (size_t i = 0; i < NR; i++) { ws->scan_B[t * NR + i] += block->b_B[i]; ws->scan_C[t * NR + i] += block->b_C[i]; } }
        /* Dispatch: wavefront for ND (>1 spatial dims), sequential for 1D */
        if (block->config.spatial_ndims > 1 && block->wavefront_plan) {
            _ssm_scan_forward_wavefront(block, ws, ws->u_seq, ws->lambda_seq, ws->y_rank);
        } else {
            _ssm_scan_forward(block, ws, ws->u_seq, ws->lambda_seq, ws->y_rank);
        }
        for (size_t t = 0; t < L; t++) { gemv_f32(block->W_out.data, &ws->y_rank[t * R], ws->y_proj, (int)D, (int)R); for (size_t d = 0; d < D; d++) out[t * D + d] = in[t * D + d] + ws->y_proj[d]; }
    }
}

static void _ssm_scan_backward(MambaBlock *block, MambaBlockWorkspace *ws, const float *u_seq, const float *lambda_seq, const float *d_y_rank) {
    size_t L = block->config.seq_len, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R; MBOptimState *grads = block->opt_state;
    float *d_h = (float *)calloc(N, sizeof(float)), *d_Bu_cur = (float *)calloc(N, sizeof(float)), *d_Bu_next = (float *)calloc(N, sizeof(float));
    memset(ws->d_scan_B, 0, L * NR * sizeof(float)); memset(ws->d_scan_C, 0, L * NR * sizeof(float)); memset(ws->d_delta, 0, L * sizeof(float)); memset(ws->d_lambda, 0, L * sizeof(float));
    for (long t = (long)L - 1; t >= 0; t--) {
        float dt_t = ws->delta[t], lam_t = lambda_seq ? lambda_seq[t] : block->config.default_lambda; const float *c_t = &ws->scan_C[t * NR], *h_t = &ws->h_seq[t * N], *h_prev = (t > 0) ? &ws->h_seq[(t - 1) * N] : NULL, *prev_Bu = (t > 0) ? &ws->Bu_seq[(t - 1) * N] : NULL;
        for (size_t r = 0; r < R; r++) { float dy_tr = d_y_rank[t * R + r]; for (size_t n = 0; n < N; n++) { d_h[n] += c_t[r * N + n] * dy_tr; ws->d_scan_C[t * NR + r * N + n] += h_t[n] * dy_tr; } }
        for (size_t n = 0; n < N; n++) {
            float a = block->A_log.data[n];
            int clipped = (block->config.use_a_log_clamp && a > block->config.a_log_min);
            if (clipped) a = block->config.a_log_min;
            float alpha = km_scan_exp(dt_t * a, block->config.use_fast_exp), beta = (1.0f - lam_t) * dt_t * alpha, gamma = lam_t * dt_t, dh_n = d_h[n]; d_Bu_cur[n] += dh_n * gamma; if (t > 0) d_Bu_next[n] = dh_n * beta;
            if (h_prev) {
                float h_rot_n; if (n & 1) { float th = block->theta[n >> 1]; h_rot_n = sinf(th) * h_prev[n-1] + cosf(th) * h_prev[n]; } else { float th = block->theta[n >> 1]; h_rot_n = cosf(th) * h_prev[n] - sinf(th) * h_prev[n+1]; }
                if (!clipped) grads->g_A_log[n] += dh_n * (dt_t * alpha * h_rot_n + (1.0f - lam_t) * dt_t * dt_t * alpha * (prev_Bu ? prev_Bu[n] : 0.0f));
                ws->d_delta[t] += dh_n * (a * alpha * h_rot_n + (1.0f - lam_t) * (alpha + dt_t * a * alpha) * (prev_Bu ? prev_Bu[n] : 0.0f) + lam_t * ws->Bu_seq[t * N + n]);
            } else { ws->d_delta[t] += dh_n * lam_t * ws->Bu_seq[t * N + n]; }
            ws->d_lambda[t] += dh_n * (prev_Bu ? -dt_t * alpha * prev_Bu[n] : 0.0f + dt_t * ws->Bu_seq[t * N + n]); d_h[n] = dh_n * alpha;
        }
        for (size_t i = 0; i + 1 < N; i += 2) {
            float th = block->theta[i >> 1], cv = cosf(th), sv = sinf(th), dh0 = d_h[i], dh1 = d_h[i+1]; float h_prev0 = (t > 0) ? ws->h_seq[(t-1)*N + i] : 0.0f, h_prev1 = (t > 0) ? ws->h_seq[(t-1)*N + i+1] : 0.0f;
            if (t > 0) grads->g_theta[i >> 1] += ((-sv * h_prev0 - cv * h_prev1) * dh0 + (cv * h_prev0 - sv * h_prev1) * dh1);
            d_h[i] = cv * dh0 + sv * dh1; d_h[i+1] = -sv * dh0 + cv * dh1;
        }
        const float *u_t = &u_seq[t * R]; for (size_t r = 0; r < R; r++) { for (size_t n = 0; n < N; n++) ws->d_scan_B[t * NR + r * N + n] += d_Bu_cur[n] * u_t[r]; }
        for (size_t n = 0; n < N; n++) { d_Bu_cur[n] = d_Bu_next[n]; d_Bu_next[n] = 0.0f; }
    }
    free(d_h); free(d_Bu_cur); free(d_Bu_next);
}

/* Backward Wavefront Callback Context */
typedef struct {
    MambaBlock *block;
    MambaBlockWorkspace *ws;
    const float *u_seq;
    const float *lambda_seq;
    const float *d_y_rank;
    float *d_h_all;
    const long *strides;
    MBOptimState *grads;
} SSMScanBackwardCtx;

static void _ssm_scan_backward_callback(long t, long level, void *userdata) {
    SSMScanBackwardCtx *ctx = (SSMScanBackwardCtx *)userdata;
    MambaBlock *block = ctx->block;
    MambaBlockWorkspace *ws = ctx->ws;
    const size_t N = block->config.state_size;
    const size_t R = _mimo_R(&block->config);
    const long ndims = block->config.spatial_ndims;
    const long *dims = block->config.spatial_dims;
    const long *strides = ctx->strides;
    MBOptimState *grads = ctx->grads;

    (void)level;

    float dt = ws->delta[t];
    float lam = ctx->lambda_seq ? ctx->lambda_seq[t] : block->config.default_lambda;
    const float *c_t = &ws->scan_C[t * N * R];
    const float *h_t = &ws->h_seq[t * N];
    const float *Bu_t = &ws->Bu_seq[t * N];
    const float *u_t = &ctx->u_seq[t * R];
    float *d_h = &ctx->d_h_all[t * N];

    /* coords for boundary check */
    long coords[KMAMBA_CONFIG_MAX_NDIMS];
    km_unravel_index(t, dims, strides, ndims, coords);

    /* Gradient from y_rank: d_y_rank[t] -> d_h, d_scan_C */
    for (size_t r = 0; r < R; r++) {
        float dy_tr = ctx->d_y_rank[t * R + r];
        for (size_t n = 0; n < N; n++) {
            d_h[n] += c_t[r * N + n] * dy_tr;
            ws->d_scan_C[t * N * R + r * N + n] += h_t[n] * dy_tr;
        }
    }

    /* 1. Propagate d_h from successors */
    for (size_t n = 0; n < N; n++) {
        float a = block->A_log.data[n];
        if (block->config.use_a_log_clamp && a > block->config.a_log_min) a = block->config.a_log_min;
        float alpha = km_scan_exp(dt * a, block->config.use_fast_exp);
        for (long ax = 0; ax < ndims; ax++) {
            if (coords[ax] < dims[ax] - 1) {
                long next_t = t + strides[ax];
                float *d_h_next = &ctx->d_h_all[next_t * N];
                float dh_rot_n;
                if (n & 1) {
                    float th = block->theta[n >> 1];
                    dh_rot_n = -sinf(th) * d_h_next[n - 1] + cosf(th) * d_h_next[n];
                } else {
                    float th = block->theta[n >> 1];
                    dh_rot_n = cosf(th) * d_h_next[n] + sinf(th) * d_h_next[n + 1];
                }
                d_h[n] += alpha * dh_rot_n;
            }
        }
    }

    /* 2. Gradients at t using predecessors */
    for (size_t n = 0; n < N; n++) {
        float a_raw = block->A_log.data[n];
        int clipped = (block->config.use_a_log_clamp && a_raw > block->config.a_log_min);
        float a = clipped ? block->config.a_log_min : a_raw;
        float alpha = km_scan_exp(dt * a, block->config.use_fast_exp);
        float dh_n = d_h[n];
        
        float h_rot_acc_n = 0.0f;
        float bu_prev_acc_n = 0.0f;
        int n_prev = 0;
        for (long ax = 0; ax < ndims; ax++) {
            if (coords[ax] > 0) {
                long prev_t = t - strides[ax];
                const float *h_prev = &ws->h_seq[prev_t * N];
                float hrp_n;
                if (n & 1) {
                    float th = block->theta[n >> 1];
                    hrp_n = sinf(th) * h_prev[n-1] + cosf(th) * h_prev[n];
                } else {
                    float th = block->theta[n >> 1];
                    hrp_n = cosf(th) * h_prev[n] - sinf(th) * h_prev[n + 1];
                }
                h_rot_acc_n += hrp_n;
                bu_prev_acc_n += ws->Bu_seq[prev_t * N + n];
                n_prev++;
            }
        }
        float h_rot_n = (n_prev > 0) ? (h_rot_acc_n / (float)n_prev) : 0.0f;
        float prev_bu_n = (n_prev > 0) ? (bu_prev_acc_n / (float)n_prev) : 0.0f;

        /* A_log gradient */
        if (!clipped) grads->g_A_log[n] += dh_n * (dt * alpha * h_rot_n + (1.0f - lam) * dt * dt * alpha * prev_bu_n);

        /* delta gradient */
        ws->d_delta[t] += dh_n * (a * alpha * h_rot_n + (1.0f - lam) * (alpha + dt * a * alpha) * prev_bu_n + lam * Bu_t[n]);

        /* lambda gradient */
        ws->d_lambda[t] += dh_n * (-dt * alpha * prev_bu_n + dt * Bu_t[n]);

        /* scan_B gradient */
        float d_Bu_cur_n = dh_n * lam * dt;
        for (size_t r = 0; r < R; r++) {
            ws->d_scan_B[t * N * R + r * N + n] += d_Bu_cur_n * u_t[r];
        }
    }

    /* 3. Gradients w.r.t. theta */
    for (size_t i = 0; i + 1 < N; i += 2) {
        float th = block->theta[i >> 1];
        float cv = cosf(th), sv = sinf(th);
        float dh0 = d_h[i], dh1 = d_h[i+1];
        float h0_acc = 0.0f, h1_acc = 0.0f;
        int n_prev = 0;
        for (long ax = 0; ax < ndims; ax++) {
            if (coords[ax] > 0) {
                long prev_t = t - strides[ax];
                h0_acc += ws->h_seq[prev_t * N + i];
                h1_acc += ws->h_seq[prev_t * N + i + 1];
                n_prev++;
            }
        }
        if (n_prev > 0) {
            float h0 = h0_acc / (float)n_prev;
            float h1 = h1_acc / (float)n_prev;
            grads->g_theta[i >> 1] += ((-sv * h0 - cv * h1) * dh0 + (cv * h0 - sv * h1) * dh1);
        }
    }
}

/* Wavefront version for ND spatial backward pass */
static void _ssm_scan_backward_wavefront(MambaBlock *block, MambaBlockWorkspace *ws,
                                         const float *u_seq, const float *lambda_seq,
                                         const float *d_y_rank) {
    const size_t N = block->config.state_size;
    const size_t R = _mimo_R(&block->config);
    const long total_points = block->wavefront_plan->total_points;
    const long ndims = block->config.spatial_ndims;
    const long *dims = block->config.spatial_dims;

    long *strides = (long *)km_pool_alloc(km_memory_pool_threadlocal(),
                                          (size_t)ndims * sizeof(long));
    if (!strides) return;
    if (!km_make_row_major_strides(dims, ndims, strides)) {
        km_pool_free(km_memory_pool_threadlocal(), strides);
        return;
    }

    /* Zero gradient accumulators */
    memset(ws->d_scan_B, 0, (size_t)total_points * N * R * sizeof(float));
    memset(ws->d_scan_C, 0, (size_t)total_points * N * R * sizeof(float));
    memset(ws->d_delta, 0, (size_t)total_points * sizeof(float));
    memset(ws->d_lambda, 0, (size_t)total_points * sizeof(float));

    /* d_h for each position - need to accumulate from successors */
    float *d_h_all = (float *)calloc((size_t)total_points * N, sizeof(float));
    if (!d_h_all) {
        km_pool_free(km_memory_pool_threadlocal(), strides);
        return;
    }

    SSMScanBackwardCtx ctx;
    ctx.block = block;
    ctx.ws = ws;
    ctx.u_seq = u_seq;
    ctx.lambda_seq = lambda_seq;
    ctx.d_y_rank = d_y_rank;
    ctx.d_h_all = d_h_all;
    ctx.strides = strides;
    ctx.grads = block->opt_state;

    /* Backward through wavefront levels using plan primitive */
    km_wavefront_plan_iter_reverse(block->wavefront_plan, _ssm_scan_backward_callback, &ctx);

    free(d_h_all);
    km_pool_free(km_memory_pool_threadlocal(), strides);
}

void mamba_backward_ws(MambaBlock *block, MambaBlockWorkspace *ws, const float *dY, const float *input, float *d_input, size_t batch_index) {
    (void)batch_index; if (!block || !ws || !dY || !input || !d_input || !block->opt_state) return;
    size_t L = block->config.seq_len, D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R; MBOptimState *g = block->opt_state;
    gemm_f32_ABt(dY, block->W_out.data, ws->d_y_rank, (int)L, (int)D, (int)R); gemm_f32_AtB(dY, ws->y_rank, g->g_W_out, (int)D, (int)L, (int)R);
    /* Dispatch: wavefront for ND (>1 spatial dims), sequential for 1D */
    if (block->config.spatial_ndims > 1 && block->wavefront_plan) {
        _ssm_scan_backward_wavefront(block, ws, ws->u_seq, ws->lambda_seq, ws->d_y_rank);
    } else {
        _ssm_scan_backward(block, ws, ws->u_seq, ws->lambda_seq, ws->d_y_rank);
    }
    for (size_t t = 0; t < L; t++) {
        const float *in_t = &input[t * D];
        for (size_t i = 0; i < NR; i++) { float dsb = ws->d_scan_B[t * NR + i], dsc = ws->d_scan_C[t * NR + i]; g->g_b_B[i] += dsb; g->g_b_C[i] += dsc; for (size_t d = 0; d < D; d++) { g->g_W_B[i * D + d] += dsb * in_t[d]; g->g_W_C[i * D + d] += dsc * in_t[d]; } }
        {
            float dd = ws->d_delta[t];
            /* Compute gradient w.r.t. raw delta (before softplus and clamp) */
            float sigmoid = 1.0f / (1.0f + expf(-ws->raw_delta[t]));
            float d_raw = dd * sigmoid;  /* gradient of softplus */
            /* Only propagate if unrestricted; otherwise gradient is zero */
            if (ws->delta[t] > block->config.dt_min && ws->delta[t] < block->config.dt_max) {
                for (size_t d = 0; d < D; d++)
                    g->g_delta_proj[d] += d_raw * in_t[d];
            }
        }
        float dl_raw = ws->d_lambda[t] * ws->lambda_seq[t] * (1.0f - ws->lambda_seq[t]); for (size_t d = 0; d < D; d++) g->g_lambda_proj[d] += dl_raw * in_t[d];
    }
    for (size_t i = 0; i < L * D; i++) d_input[i] = dY[i];
}

void mamba_backward(MambaBlock *block, const float *dY, const float *input, float *d_input, size_t batch_index) { MambaBlockWorkspace *ws = mamba_block_workspace_create(block); mamba_backward_ws(block, ws, dY, input, d_input, batch_index); mamba_block_workspace_free(ws); }

#ifdef KMAMBA_BUILD_CUDA
static int mamba_block_ensure_gpu(MambaBlock *block) {
    if (!block || block->gpu.gpu_ready) return 0;
    size_t L = block->config.seq_len, D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    size_t bytes_L_D = L * D * sizeof(float), bytes_L_R = L * R * sizeof(float), bytes_L_NR = L * NR * sizeof(float), bytes_L_N = L * N * sizeof(float);
#define CUDA_CHECK(err) if (err != cudaSuccess) return -1
    CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_W_in, R * D * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_W_out, D * R * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_A_log, N * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_W_B, NR * D * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_W_C, NR * D * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_delta_proj, D * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_theta, (N/2 > 0 ? N/2 : 1) * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_lambda_proj, D * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_b_B, NR * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_b_C, NR * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_u_raw, bytes_L_R)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_u, bytes_L_R)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_dt_raw, L * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_dt, L * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_B_exp, bytes_L_NR)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_C_exp, bytes_L_NR)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_h_store, bytes_L_N)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_y_scan, bytes_L_R)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_y_proj, bytes_L_D)); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_lambda_raw, L * sizeof(float))); CUDA_CHECK(cudaMalloc((void **)&block->gpu.d_lambda, L * sizeof(float)));
#undef CUDA_CHECK
    block->gpu.gpu_ready = 1; return 0;
}

static int _mamba_block_forward_gpu(MambaBlock *block, float *output, const float *input, size_t batch_size) {
    if (!block || !output || !input || batch_size == 0) return -1;
    if (mamba_block_ensure_gpu(block) != 0) return -1;
    static cublasHandle_t cublas_handle = NULL; static int cublas_init_done = 0; if (!cublas_init_done) { if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) return -1; cublas_init_done = 1; }
    size_t L = block->config.seq_len, D = block->config.dim, R = _mimo_R(&block->config), NR = block->config.state_size * R; size_t bytes_L_D = L * D * sizeof(float);
    float *d_input, *d_output; if (cudaMalloc((void**)&d_input, bytes_L_D) != cudaSuccess) return -1; if (cudaMalloc((void**)&d_output, bytes_L_D) != cudaSuccess) { cudaFree(d_input); return -1; }
    cudaMemcpy(block->gpu.d_W_in, block->W_in.data, R * D * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_W_out, block->W_out.data, D * R * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_A_log, block->A_log.data, block->config.state_size * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_W_B, block->W_B.data, NR * D * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_W_C, block->W_C.data, NR * D * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_delta_proj, block->delta_proj.data, D * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_theta, block->theta, (block->config.state_size/2 > 0 ? block->config.state_size/2 : 1) * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_lambda_proj, block->lambda_proj.data, D * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_b_B, block->b_B, NR * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(block->gpu.d_b_C, block->b_C, NR * sizeof(float), cudaMemcpyHostToDevice);
    for (size_t b = 0; b < batch_size; b++) { cudaMemcpy(d_input, &input[b * L * D], bytes_L_D, cudaMemcpyHostToDevice); cuda_block_forward(cublas_handle, block->gpu.d_W_in, block->gpu.d_W_out, block->gpu.d_A_log, block->gpu.d_W_B, block->gpu.d_W_C, block->gpu.d_delta_proj, block->gpu.d_theta, block->gpu.d_lambda_proj, d_input, d_output, block->gpu.d_u_raw, block->gpu.d_u, block->gpu.d_dt_raw, block->gpu.d_dt, block->gpu.d_B_exp, block->gpu.d_C_exp, NULL, block->gpu.d_h_store, block->gpu.d_y_scan, block->gpu.d_y_proj, block->gpu.d_lambda_raw, block->gpu.d_lambda, (int)L, (int)block->config.state_size, (int)D, (int)R); cudaMemcpy(&output[b * L * D], d_output, bytes_L_D, cudaMemcpyDeviceToHost); }
    cudaFree(d_input); cudaFree(d_output); return 0;
}
#endif

void mamba_block_forward(MambaBlock *block, float *output, const float *input, size_t batch_size) { KMAMBA_AUTO_BACKEND();
#ifdef KMAMBA_BUILD_CUDA
    if (backend == KMAMBA_BACKEND_GPU) { if (_mamba_block_forward_gpu(block, output, input, batch_size) == 0) return; }
#endif
    MambaBlockWorkspace *ws = mamba_block_workspace_create(block); mamba_block_forward_ws(block, ws, output, input, batch_size); mamba_block_workspace_free(ws);
}

MBOptimState* mamba_local_grad_alloc(const MambaBlock *block) {
    if (!block) return NULL;
    MBOptimState *s = (MBOptimState *)calloc(1, sizeof(MBOptimState));
    if (!s) return NULL;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R, TS = N/2 > 0 ? N/2 : 1;
    s->g_W_in = (float *)calloc(R * D, sizeof(float)); s->g_W_out = (float *)calloc(D * R, sizeof(float)); s->g_A_log = (float *)calloc(N, sizeof(float)); s->g_W_B = (float *)calloc(NR * D, sizeof(float)); s->g_W_C = (float *)calloc(NR * D, sizeof(float)); s->g_b_B = (float *)calloc(NR, sizeof(float)); s->g_b_C = (float *)calloc(NR, sizeof(float)); s->g_delta_proj = (float *)calloc(D, sizeof(float)); s->g_lambda_proj = (float *)calloc(D, sizeof(float)); s->g_theta = (float *)calloc(TS, sizeof(float));
    return s;
}

void mamba_local_grad_reduce(MambaBlock *block, const MBOptimState *local) {
    if (!block || !block->opt_state || !local) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R, TS = N/2 > 0 ? N/2 : 1;
    for (size_t i=0; i<R*D; i++) s->g_W_in[i] += local->g_W_in[i];
    for (size_t i=0; i<D*R; i++) s->g_W_out[i] += local->g_W_out[i];
    for (size_t i=0; i<N; i++) s->g_A_log[i] += local->g_A_log[i];
    for (size_t i=0; i<NR*D; i++) { s->g_W_B[i] += local->g_W_B[i]; s->g_W_C[i] += local->g_W_C[i]; }
    for (size_t i=0; i<NR; i++) { s->g_b_B[i] += local->g_b_B[i]; s->g_b_C[i] += local->g_b_C[i]; }
    for (size_t i=0; i<D; i++) { s->g_delta_proj[i] += local->g_delta_proj[i]; s->g_lambda_proj[i] += local->g_lambda_proj[i]; }
    for (size_t i=0; i<TS; i++) s->g_theta[i] += local->g_theta[i];
}

void mamba_local_grad_free(MBOptimState *local) {
    if (!local) return;
    free(local->g_W_in);
    free(local->g_W_out);
    free(local->g_A_log);
    free(local->g_W_B);
    free(local->g_W_C);
    free(local->g_b_B);
    free(local->g_b_C);
    free(local->g_delta_proj);
    free(local->g_lambda_proj);
    free(local->g_theta);
    free(local);
}

void mamba_backward_ws_local(MambaBlock *block, MambaBlockWorkspace *ws, const float *dY, const float *input, float *d_input, size_t batch_index, MBOptimState *local_grad) {
    MBOptimState *orig = block->opt_state; block->opt_state = local_grad; mamba_backward_ws(block, ws, dY, input, d_input, batch_index); block->opt_state = orig;
}
