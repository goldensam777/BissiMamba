#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kmamba.h"
#include "kmamba_kernels.h"
#include "kmamba_ser.h"
#include "kmamba_cuda_utils.h"

#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Declaration of CUDA kernels and dispatchers from cuda/mamba_block.cu */
#ifdef __cplusplus
extern "C" {
#endif
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
    int L, int state, int dim, int R,
    int spatial_ndims, const long *spatial_dims);

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
    int L, int state, int dim, int R,
    int spatial_ndims, const long *spatial_dims);

void gpu_optimizer_step(MambaBlock *block, const MBOptimConfig *conf);

/* Embedding/Head kernels */
void cuda_embedding_forward(const float *d_embed, const uint32_t *d_tokens, float *d_out, int L, int D);
void cuda_head_forward(cublasHandle_t handle, const float *d_head, const float *d_hidden, float *d_logits, int L, int D, int V);
void cuda_softmax_loss_kernel(const float *d_logits, const uint32_t *d_targets, float *d_loss, float *d_dlogits, int L, int V);
void cuda_adamw_step_wrapper(float *param, float *grad, float *m, float *v,
                              float lr, float beta1, float beta2, float eps, float wd, int n, int step);
#ifdef __cplusplus
}
#endif

/* Initialize GPU memory for a single MambaBlock */
static int mamba_block_gpu_init(MambaBlock *b) {
    if (!b || b->gpu.gpu_ready) return 0;

    size_t D = b->config.dim;
    size_t N = b->config.state_size;
    size_t R = (b->config.mimo_rank > 0) ? b->config.mimo_rank : 1;
    size_t NR = N * R;
    size_t TS = D / 2;

    /* Parameters */
    cudaMalloc((void**)&b->gpu.d_W_in,  R * D * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_W_out, D * R * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_A_log, N * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_W_B,   NR * D * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_W_C,   NR * D * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_b_B,   NR * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_b_C,   NR * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_delta_proj,  D * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_lambda_proj, D * sizeof(float));
    if (b->theta) cudaMalloc((void**)&b->gpu.d_theta, TS * sizeof(float));

    /* Copy to device */
    cudaMemcpy(b->gpu.d_W_in,  b->W_in.data,  R * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_W_out, b->W_out.data, D * R * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_A_log, b->A_log.data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_W_B,   b->W_B.data,   NR * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_W_C,   b->W_C.data,   NR * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_b_B,   b->b_B,        NR * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_b_C,   b->b_C,        NR * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_delta_proj,  b->delta_proj.data,  D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b->gpu.d_lambda_proj, b->lambda_proj.data, D * sizeof(float), cudaMemcpyHostToDevice);
    if (b->theta) cudaMemcpy(b->gpu.d_theta, b->theta, TS * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate Gradients and Optimizer states if training */
    if (b->opt_state) {
        cudaMalloc((void**)&b->gpu.d_g_W_in,  R * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_W_out, D * R * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_A_log, N * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_W_B,   NR * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_W_C,   NR * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_b_B,   NR * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_b_C,   NR * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_delta_proj,  D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_g_lambda_proj, D * sizeof(float));
        if (b->theta) cudaMalloc((void**)&b->gpu.d_g_theta, TS * sizeof(float));

        cudaMalloc((void**)&b->gpu.d_m_W_in,  R * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_W_in,  R * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_W_out, D * R * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_W_out, D * R * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_A_log, N * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_A_log, N * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_W_B,   NR * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_W_B,   NR * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_W_C,   NR * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_W_C,   NR * D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_b_B,   NR * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_b_B,   NR * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_b_C,   NR * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_b_C,   NR * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_delta_proj,  D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_delta_proj,  D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_m_lambda_proj, D * sizeof(float));
        cudaMalloc((void**)&b->gpu.d_v_lambda_proj, D * sizeof(float));
        if (b->theta) {
            cudaMalloc((void**)&b->gpu.d_m_theta, TS * sizeof(float));
            cudaMalloc((void**)&b->gpu.d_v_theta, TS * sizeof(float));
        }

        /* Zero moments and gradients */
        cudaMemset(b->gpu.d_m_W_in, 0, R * D * sizeof(float));
        cudaMemset(b->gpu.d_v_W_in, 0, R * D * sizeof(float));
        cudaMemset(b->gpu.d_m_W_out, 0, D * R * sizeof(float));
        cudaMemset(b->gpu.d_v_W_out, 0, D * R * sizeof(float));
        cudaMemset(b->gpu.d_m_A_log, 0, N * sizeof(float));
        cudaMemset(b->gpu.d_v_A_log, 0, N * sizeof(float));
        cudaMemset(b->gpu.d_m_W_B, 0, NR * D * sizeof(float));
        cudaMemset(b->gpu.d_v_W_B, 0, NR * D * sizeof(float));
        cudaMemset(b->gpu.d_m_W_C, 0, NR * D * sizeof(float));
        cudaMemset(b->gpu.d_v_W_C, 0, NR * D * sizeof(float));
        cudaMemset(b->gpu.d_m_b_B, 0, NR * sizeof(float));
        cudaMemset(b->gpu.d_v_b_B, 0, NR * sizeof(float));
        cudaMemset(b->gpu.d_m_b_C, 0, NR * sizeof(float));
        cudaMemset(b->gpu.d_v_b_C, 0, NR * sizeof(float));
        cudaMemset(b->gpu.d_m_delta_proj, 0, D * sizeof(float));
        cudaMemset(b->gpu.d_v_delta_proj, 0, D * sizeof(float));
        cudaMemset(b->gpu.d_m_lambda_proj, 0, D * sizeof(float));
        cudaMemset(b->gpu.d_v_lambda_proj, 0, D * sizeof(float));
        if (b->theta) {
            cudaMemset(b->gpu.d_m_theta, 0, TS * sizeof(float));
            cudaMemset(b->gpu.d_v_theta, 0, TS * sizeof(float));
        }
        
        /* Zero gradients */
        cudaMemset(b->gpu.d_g_W_in, 0, R * D * sizeof(float));
        cudaMemset(b->gpu.d_g_W_out, 0, D * R * sizeof(float));
        cudaMemset(b->gpu.d_g_A_log, 0, N * sizeof(float));
        cudaMemset(b->gpu.d_g_W_B, 0, NR * D * sizeof(float));
        cudaMemset(b->gpu.d_g_W_C, 0, NR * D * sizeof(float));
        cudaMemset(b->gpu.d_g_b_B, 0, NR * sizeof(float));
        cudaMemset(b->gpu.d_g_b_C, 0, NR * sizeof(float));
        cudaMemset(b->gpu.d_g_delta_proj, 0, D * sizeof(float));
        cudaMemset(b->gpu.d_g_lambda_proj, 0, D * sizeof(float));
        if (b->theta) cudaMemset(b->gpu.d_g_theta, 0, TS * sizeof(float));
    }

    size_t L = b->config.seq_len;
    cudaMalloc((void**)&b->gpu.d_u_raw, L * R * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_u,     L * R * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_dt_raw, L * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_dt,     L * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_B_exp,  L * NR * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_C_exp,  L * NR * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_h_store, L * N * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_y_scan,  L * R * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_y_proj,  L * D * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_lambda_raw, L * sizeof(float));
    cudaMalloc((void**)&b->gpu.d_lambda,     L * sizeof(float));

    b->gpu.gpu_ready = 1;
    return 0;
}

static void mamba_block_gpu_free(MambaBlock *b) {
    if (!b || !b->gpu.gpu_ready) return;
    cudaFree(b->gpu.d_W_in); cudaFree(b->gpu.d_W_out); cudaFree(b->gpu.d_A_log);
    cudaFree(b->gpu.d_W_B); cudaFree(b->gpu.d_W_C); cudaFree(b->gpu.d_b_B); cudaFree(b->gpu.d_b_C);
    cudaFree(b->gpu.d_delta_proj); cudaFree(b->gpu.d_lambda_proj);
    if (b->theta) cudaFree(b->gpu.d_theta);

    if (b->opt_state) {
        cudaFree(b->gpu.d_g_W_in); cudaFree(b->gpu.d_g_W_out); cudaFree(b->gpu.d_g_A_log);
        cudaFree(b->gpu.d_g_W_B); cudaFree(b->gpu.d_g_W_C); cudaFree(b->gpu.d_g_b_B); cudaFree(b->gpu.d_g_b_C);
        cudaFree(b->gpu.d_g_delta_proj); cudaFree(b->gpu.d_g_lambda_proj);
        if (b->theta) cudaFree(b->gpu.d_g_theta);

        cudaFree(b->gpu.d_m_W_in); cudaFree(b->gpu.d_v_W_in);
        cudaFree(b->gpu.d_m_W_out); cudaFree(b->gpu.d_v_W_out);
        cudaFree(b->gpu.d_m_A_log); cudaFree(b->gpu.d_v_A_log);
        cudaFree(b->gpu.d_m_W_B); cudaFree(b->gpu.d_v_W_B);
        cudaFree(b->gpu.d_m_W_C); cudaFree(b->gpu.d_v_W_C);
        cudaFree(b->gpu.d_m_b_B); cudaFree(b->gpu.d_v_b_B);
        cudaFree(b->gpu.d_m_b_C); cudaFree(b->gpu.d_v_b_C);
        cudaFree(b->gpu.d_m_delta_proj); cudaFree(b->gpu.d_v_delta_proj);
        cudaFree(b->gpu.d_m_lambda_proj); cudaFree(b->gpu.d_v_lambda_proj);
        if (b->theta) { cudaFree(b->gpu.d_m_theta); cudaFree(b->gpu.d_v_theta); }
    }

    cudaFree(b->gpu.d_u_raw); cudaFree(b->gpu.d_u); cudaFree(b->gpu.d_dt_raw); cudaFree(b->gpu.d_dt);
    cudaFree(b->gpu.d_B_exp); cudaFree(b->gpu.d_C_exp); cudaFree(b->gpu.d_h_store); cudaFree(b->gpu.d_y_scan);
    cudaFree(b->gpu.d_y_proj); cudaFree(b->gpu.d_lambda_raw); cudaFree(b->gpu.d_lambda);

    b->gpu.gpu_ready = 0;
}

/* Global cuBLAS handle */
static cublasHandle_t g_cublas_handle = NULL;
static int g_cublas_initialized = 0;

static cublasHandle_t get_cublas_handle(void) {
    if (!g_cublas_initialized) {
        cublasCreate(&g_cublas_handle);
        g_cublas_initialized = 1;
    }
    return g_cublas_handle;
}

/* Initialize GPU memory for KMamba (embedding, head, and layers) */
int kmamba_gpu_init(KMamba *m) {
    if (!m) return -1;
    if (m->cfg.gpu_device >= 0) cudaSetDevice(m->cfg.gpu_device);
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;

    cudaMalloc((void**)&m->gpu.d_embedding, V * D * sizeof(float));
    cudaMalloc((void**)&m->gpu.d_head, D * V * sizeof(float));

    /* Copy parameters to device */
    cudaMemcpy(m->gpu.d_embedding, m->embedding, V * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m->gpu.d_head, m->head, D * V * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate optimizer states on device */
    if (m->m_embedding && m->v_embedding) {
        cudaMalloc((void**)&m->gpu.d_m_embed, V * D * sizeof(float));
        cudaMemset(m->gpu.d_m_embed, 0, V * D * sizeof(float));
        cudaMalloc((void**)&m->gpu.d_v_embed, V * D * sizeof(float));
        cudaMemset(m->gpu.d_v_embed, 0, V * D * sizeof(float));
        cudaMalloc((void**)&m->gpu.d_g_embed, V * D * sizeof(float));
    }
    if (m->m_head && m->v_head) {
        cudaMalloc((void**)&m->gpu.d_m_head, D * V * sizeof(float));
        cudaMemset(m->gpu.d_m_head, 0, D * V * sizeof(float));
        cudaMalloc((void**)&m->gpu.d_v_head, D * V * sizeof(float));
        cudaMemset(m->gpu.d_v_head, 0, D * V * sizeof(float));
        cudaMalloc((void**)&m->gpu.d_g_head, D * V * sizeof(float));
    }

    /* Initialize all layers on GPU */
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        if (mamba_block_gpu_init(m->layers[i]) != 0) return -1;
    }

    m->gpu.gpu_ready = 1;
    return 0;
}

/* Free GPU memory for KMamba */
void kmamba_gpu_free(KMamba *m) {
    if (!m || !m->gpu.gpu_ready) return;
    cudaFree(m->gpu.d_embedding);
    cudaFree(m->gpu.d_head);
    cudaFree(m->gpu.d_m_embed);
    cudaFree(m->gpu.d_v_embed);
    cudaFree(m->gpu.d_g_embed);
    cudaFree(m->gpu.d_m_head);
    cudaFree(m->gpu.d_v_head);
    cudaFree(m->gpu.d_g_head);
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_gpu_free(m->layers[i]);
    }
    m->gpu.gpu_ready = 0;
}

/* Sync host parameters from device after training */
void kmamba_gpu_sync_to_host(KMamba *m) {
    if (!m || !m->gpu.gpu_ready) return;
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;
    cudaMemcpy(m->embedding, m->gpu.d_embedding, V * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m->head, m->gpu.d_head, D * V * sizeof(float), cudaMemcpyDeviceToHost);
}

#endif /* KMAMBA_BUILD_CUDA */

/* ═════════════════════════════════════════════════════════════════════════════
 * KMamba API Implementation (CPU Reference)
 * ═════════════════════════════════════════════════════════════════════════════ */

static size_t _mimo_R(size_t mimo_rank) {
    return (mimo_rank > 1) ? mimo_rank : 1;
}

void kmamba_config_set_defaults(KMambaConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_ndims = 8;
    cfg->max_state = 64;
    cfg->use_fast_exp = 0;
    cfg->vocab_size = 32768;
    cfg->dim = 512;
    cfg->state_size = 64;
    cfg->seq_len = 1024;
    cfg->n_layers = 12;
    cfg->mimo_rank = 1;
    cfg->dt_scale = 1.0f;
    cfg->dt_min = 0.001f;
    cfg->dt_max = 0.1f;
    cfg->default_lambda = 0.5f;
    cfg->use_a_log_clamp = 1;
    cfg->a_log_min = -1e-5f;
    cfg->d_conv = 4;
    cfg->expand_factor = 2.0f;
    cfg->gpu_device = -1;
    cfg->weight_tying = 1;
    for (int i = 0; i < KMAMBA_CONFIG_MAX_NDIMS; i++) {
        cfg->pad_left[i] = -1;
        cfg->pad_right[i] = -1;
    }
}

void kmamba_optim_config_set_defaults(MBOptimConfig *opt_cfg) {
    if (!opt_cfg) return;
    opt_cfg->lr = 1e-3f;
    opt_cfg->mu = 0.9f;
    opt_cfg->beta2 = 0.999f;
    opt_cfg->eps = 1e-8f;
    opt_cfg->clip_norm = 1.0f;
    opt_cfg->weight_decay = 0.01f;
}

KMamba* kmamba_create(const KMambaConfig *cfg) {
    if (!cfg) return NULL;
    
    /* Mandatory field validation - No hidden fallbacks */
    /* vocab_size=0 is valid for vision models (no vocabulary) */
    if (cfg->dim == 0 || cfg->state_size == 0 || cfg->seq_len == 0 || cfg->n_layers == 0) {
        fprintf(stderr, "kmamba_create: missing mandatory config (dim, state_size, seq_len, n_layers)\n");
        return NULL;
    }
    if (cfg->max_ndims <= 0 || cfg->max_ndims > KMAMBA_CONFIG_MAX_NDIMS) {
        fprintf(stderr, "kmamba_create: invalid max_ndims=%ld\n", cfg->max_ndims);
        return NULL;
    }
    if (cfg->max_state <= 0) {
        fprintf(stderr, "kmamba_create: invalid max_state=%ld\n", cfg->max_state);
        return NULL;
    }

    KMamba *m = (KMamba *)calloc(1, sizeof(KMamba));
    if (!m) return NULL;
    m->cfg = *cfg;
    
    if (m->cfg.convnd_K > 0) {
        long default_pad = (m->cfg.convnd_K - 1) / 2;
        for (long ax = 0; ax < KMAMBA_CONFIG_MAX_NDIMS; ax++) {
            if (m->cfg.pad_left[ax] < 0) m->cfg.pad_left[ax] = default_pad;
            if (m->cfg.pad_right[ax] < 0) m->cfg.pad_right[ax] = default_pad;
        }
    }
    
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;

    /* Only allocate embedding and head for language models (vocab_size > 0) */
    /* Vision models (vocab_size == 0) don't need these */
    if (V > 0) {
        m->embedding = (float *)malloc(V * D * sizeof(float));
        m->head = (float *)malloc(D * V * sizeof(float));
        if (!m->embedding || !m->head) {
            kmamba_free(m);
            return NULL;
        }
    } else {
        m->embedding = NULL;
        m->head = NULL;
    }
    m->layers = (MambaBlock **)malloc(m->cfg.n_layers * sizeof(MambaBlock *));
    if (!m->layers) {
        kmamba_free(m);
        return NULL;
    }
    
    MBConfig lcfg = {0};
    lcfg.max_ndims = m->cfg.max_ndims;
    lcfg.max_state = m->cfg.max_state;
    lcfg.use_fast_exp = m->cfg.use_fast_exp;
    lcfg.dim = D;
    lcfg.state_size = m->cfg.state_size;
    lcfg.seq_len = m->cfg.seq_len;
    lcfg.mimo_rank = m->cfg.mimo_rank;
    lcfg.dt_scale = m->cfg.dt_scale;
    lcfg.dt_min = m->cfg.dt_min;
    lcfg.dt_max = m->cfg.dt_max;
    lcfg.default_lambda = m->cfg.default_lambda;
    lcfg.use_a_log_clamp = m->cfg.use_a_log_clamp;
    lcfg.a_log_min = m->cfg.a_log_min;
    lcfg.d_conv = m->cfg.d_conv;
    lcfg.expand_factor = m->cfg.expand_factor;
    lcfg.spatial_ndims = m->cfg.spatial_ndims;
    memcpy(lcfg.spatial_dims, m->cfg.spatial_dims, sizeof(lcfg.spatial_dims));
    lcfg.use_convnd = m->cfg.use_convnd;
    lcfg.convnd_K = m->cfg.convnd_K;
    lcfg.convnd_ndims = m->cfg.convnd_ndims;
    memcpy(lcfg.pad_left, m->cfg.pad_left, sizeof(lcfg.pad_left));
    memcpy(lcfg.pad_right, m->cfg.pad_right, sizeof(lcfg.pad_right));
    
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        m->layers[i] = mamba_block_create(&lcfg);
        if (!m->layers[i]) {
            fprintf(stderr, "kmamba_create: failed to create layer %zu\n", i);
            m->cfg.n_layers = i; /* for proper cleanup in kmamba_free */
            kmamba_free(m);
            return NULL;
        }
    }
    
    return m;
}

void kmamba_free(KMamba *m) {
    if (!m) return;
#ifdef KMAMBA_BUILD_CUDA
    kmamba_gpu_free(m);
#endif
    if (m->embedding) free(m->embedding);
    if (m->head) free(m->head);
    if (m->m_embedding) free(m->m_embedding);
    if (m->v_embedding) free(m->v_embedding);
    if (m->m_head) free(m->m_head);
    if (m->v_head) free(m->v_head);
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_free(m->layers[i]);
    }
    free(m->layers);
    free(m);
}

int kmamba_init(KMamba *m, uint32_t seed) {
    if (!m) return -1;
    srand(seed);
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;
    
    init_xavier_uniform_f32(m->embedding, V, D, seed);
    init_xavier_uniform_f32(m->head, D, V, seed + 1);
    
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_init(m->layers[i]);
    }
    return 0;
}

int kmamba_enable_training(KMamba *m, const MBOptimConfig *opt_blocks, float lr_embed_head, float weight_decay) {
    if (!m) return -1;
    m->for_training = 1;
    m->opt_blocks = *opt_blocks;
    m->lr_embed_head = lr_embed_head;
    m->weight_decay = weight_decay;
    
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;
    m->m_embedding = (float *)calloc(V * D, sizeof(float));
    m->v_embedding = (float *)calloc(V * D, sizeof(float));
    m->m_head = (float *)calloc(D * V, sizeof(float));
    m->v_head = (float *)calloc(D * V, sizeof(float));
    
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_attach_optimizer(m->layers[i], OPTIMIZER_ADAMW, opt_blocks);
    }
    return 0;
}

/* Full GPU Batch Training */
float kmamba_train_batch(KMamba *m, const uint32_t *batch_tokens, size_t batch_size) {
#ifdef KMAMBA_BUILD_CUDA
    if (!m || !batch_tokens || batch_size == 0) return NAN;
    if (m->cfg.gpu_device >= 0) cudaSetDevice(m->cfg.gpu_device);
    if (!m->gpu.gpu_ready && kmamba_gpu_init(m) != 0) return NAN;
    
    cublasHandle_t handle = get_cublas_handle();
    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;
    size_t Lp1 = L + 1;
    size_t n_layers = m->cfg.n_layers;
    float invB = 1.0f / (float)batch_size;
    float invL = 1.0f / (float)L;

    /* No hidden fallbacks for optimizer params */
    if (m->opt_blocks.mu == 0.0f || m->opt_blocks.beta2 == 0.0f || m->opt_blocks.eps == 0.0f) {
        fprintf(stderr, "kmamba_train_batch: optimizer parameters (mu, beta2, eps) must be non-zero\n");
        return NAN;
    }
    float adam_beta1 = m->opt_blocks.mu;
    float adam_beta2 = m->opt_blocks.beta2;
    float adam_eps = m->opt_blocks.eps;

    /* Device temp buffers for current batch */
    float *d_acts, *d_logits, *d_dlogits, *d_dhidden, *d_loss;
    uint32_t *d_batch_tokens;
    cudaMalloc((void**)&d_acts, (n_layers + 1) * L * D * sizeof(float));
    cudaMalloc((void**)&d_logits, L * V * sizeof(float));
    cudaMalloc((void**)&d_dlogits, L * V * sizeof(float));
    cudaMalloc((void**)&d_dhidden, L * D * sizeof(float));
    cudaMalloc((void**)&d_loss, sizeof(float));
    cudaMalloc((void**)&d_batch_tokens, batch_size * Lp1 * sizeof(uint32_t));
    cudaMemcpy(d_batch_tokens, batch_tokens, batch_size * Lp1 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    float total_loss = 0.0f;
    m->step_embed_head++;

    /* Zero all GPU gradient buffers before accumulating gradients for this batch. */
    cudaMemset(m->gpu.d_g_head,  0, D * V * sizeof(float));
    cudaMemset(m->gpu.d_g_embed, 0, V * D * sizeof(float));
    for (size_t i = 0; i < n_layers; i++) {
        mamba_zero_grads(m->layers[i]);
    }

    for (size_t b = 0; b < batch_size; b++) {
        uint32_t *d_seq = d_batch_tokens + b * Lp1;
        uint32_t *d_tok_in = d_seq;
        uint32_t *d_tok_tgt = d_seq + 1;

        /* 1. Forward */
        cuda_embedding_forward(m->gpu.d_embedding, d_tok_in, d_acts, (int)L, (int)D);
        for (size_t i = 0; i < n_layers; i++) {
            MambaBlock *l = m->layers[i];
            cuda_block_forward(handle, l->gpu.d_W_in, l->gpu.d_W_out, l->gpu.d_A_log,
                               l->gpu.d_W_B, l->gpu.d_W_C, l->gpu.d_delta_proj,
                               l->gpu.d_theta, l->gpu.d_lambda_proj,
                               d_acts + i * L * D, d_acts + (i + 1) * L * D,
                               l->gpu.d_u_raw, l->gpu.d_u, l->gpu.d_dt_raw, l->gpu.d_dt,
                               l->gpu.d_B_exp, l->gpu.d_C_exp, NULL,
                               l->gpu.d_h_store, l->gpu.d_y_scan, l->gpu.d_y_proj,
                               l->gpu.d_lambda_raw, l->gpu.d_lambda,
                               (int)L, (int)l->config.state_size, (int)D, (int)_mimo_R(l->config.mimo_rank),
                               l->config.spatial_ndims, l->config.spatial_dims);
        }
        cuda_head_forward(handle, m->gpu.d_head, d_acts + n_layers * L * D, d_logits, (int)L, (int)D, (int)V);
        
        float sample_loss_h;
        cuda_softmax_loss_kernel(d_logits, d_tok_tgt, d_loss, d_dlogits, (int)L, (int)V);
        cudaMemcpy(&sample_loss_h, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        total_loss += sample_loss_h * invL;

        /* 2. Backward */
        /* Head backward: d_dhidden = dlogits @ head */
        /* cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, L, V, &a1, head, D, dlogits, V, &b0, dhidden, D) */
        float a1 = 1.0f, b0 = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)L, (int)V, &a1, m->gpu.d_head, (int)D, d_dlogits, (int)V, &b0, d_dhidden, (int)D);
        /* Gradient head: d_g_head += dlogits^T @ hidden */
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, (int)V, (int)D, (int)L, &a1, d_dlogits, (int)V, d_acts + n_layers * L * D, (int)D, &a1, m->gpu.d_g_head, (int)V);

        float *d_dy = d_dhidden;
        float *d_dx = d_dhidden; /* Reuse or toggle */
        for (int i = (int)n_layers - 1; i >= 0; i--) {
            MambaBlock *l = m->layers[i];
            cuda_block_backward(handle, l->gpu.d_W_in, l->gpu.d_W_out, l->gpu.d_A_log,
                                l->gpu.d_W_B, l->gpu.d_W_C, l->gpu.d_delta_proj,
                                l->gpu.d_theta, l->gpu.d_lambda_proj,
                                d_acts + i * L * D, d_dy,
                                l->gpu.d_g_W_in, l->gpu.d_g_W_out, l->gpu.d_g_A_log,
                                l->gpu.d_g_W_B, l->gpu.d_g_W_C, l->gpu.d_g_delta_proj,
                                l->gpu.d_g_theta, l->gpu.d_g_lambda_proj,
                                d_dx,
                                l->gpu.d_u_raw, l->gpu.d_u, l->gpu.d_dt_raw, l->gpu.d_dt,
                                l->gpu.d_B_exp, l->gpu.d_C_exp, NULL,
                                l->gpu.d_h_store, l->gpu.d_y_scan,
                                l->gpu.d_lambda, l->gpu.d_lambda_raw,
                                (int)L, (int)l->config.state_size, (int)D, (int)_mimo_R(l->config.mimo_rank),
                                l->config.spatial_ndims, l->config.spatial_dims);
        }
        
        /* Embedding gradient accumulation on GPU */
        /* (Simplified: need a kernel for d_g_embed += d_dx at tokens) */
        // cuda_embedding_backward(m->gpu.d_g_embed, d_tok_in, d_dx, L, D);
    }

    /* 3. Optimizer Steps on GPU */
    for (size_t i = 0; i < n_layers; i++) {
        gpu_optimizer_step(m->layers[i], &m->opt_blocks);
    }
    cuda_adamw_step_wrapper(m->gpu.d_embedding, m->gpu.d_g_embed, m->gpu.d_m_embed, m->gpu.d_v_embed,
                          m->lr_embed_head, adam_beta1, adam_beta2, adam_eps, m->weight_decay, (int)(V * D), (int)m->step_embed_head);
    cuda_adamw_step_wrapper(m->gpu.d_head, m->gpu.d_g_head, m->gpu.d_m_head, m->gpu.d_v_head,
                          m->lr_embed_head, adam_beta1, adam_beta2, adam_eps, m->weight_decay, (int)(D * V), (int)m->step_embed_head);

    cudaFree(d_acts); cudaFree(d_logits); cudaFree(d_dlogits); cudaFree(d_dhidden); cudaFree(d_loss); cudaFree(d_batch_tokens);
    return total_loss * invB;
#else
    return 0.0f;
#endif
}

/* ========= Tensors & Config Access ========= */
const KMambaConfig* kmamba_get_config(const KMamba *m) { return m ? &m->cfg : NULL; }
size_t kmamba_step_count(const KMamba *m) { return m ? m->step_embed_head : 0; }
void kmamba_update_lr(KMamba *m, float lb, float le) { if(m){ m->opt_blocks.lr=lb; m->lr_embed_head=le; } }

int kmamba_enable_training_with_optimizer(KMamba *m, OptimizerType opt_type,
                                          const MBOptimConfig *opt_blocks,
                                          float lr_embed_head, float weight_decay) {
    if (!m || !opt_blocks) return -1;
    if (kmamba_enable_training(m, opt_blocks, lr_embed_head, weight_decay) != 0) return -1;
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_attach_optimizer(m->layers[i], opt_type, opt_blocks);
    }
    return 0;
}

int kmamba_forward(KMamba *m, const uint32_t *tokens, float *logits_out) {
    if (!m || !tokens || !logits_out) return -1;
    const size_t L = m->cfg.seq_len;
    const size_t D = m->cfg.dim;
    const size_t V = m->cfg.vocab_size;
    const size_t hidden_elems = L * D;

    float *acts_cur = (float *)malloc(hidden_elems * sizeof(float));
    float *acts_next = (float *)malloc(hidden_elems * sizeof(float));
    if (!acts_cur || !acts_next) {
        free(acts_cur);
        free(acts_next);
        return -1;
    }

    for (size_t t = 0; t < L; t++) {
        size_t tok = (size_t)tokens[t];
        if (tok >= V) tok = 0;
        memcpy(&acts_cur[t * D], &m->embedding[tok * D], D * sizeof(float));
    }

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_forward(m->layers[i], acts_next, acts_cur, 1);
        {
            float *tmp = acts_cur;
            acts_cur = acts_next;
            acts_next = tmp;
        }
    }

    gemm_f32(acts_cur, m->head, logits_out, (int)L, (int)D, (int)V);
    free(acts_cur);
    free(acts_next);
    return 0;
}

float kmamba_train_step(KMamba *m, const uint32_t *tokens_plus1) {
    return kmamba_train_batch(m, tokens_plus1, 1);
}

int kmamba_save(const KMamba *m, const char *path) {
    if (!m || !path) return -1;
    return kmamba_save_ser((KMamba *)m, path, KSER_FP32);
}

KMamba* kmamba_load(const char *path, int for_training,
                    const MBOptimConfig *opt_blocks,
                    float lr_embed_head, float weight_decay) {
    KMamba *m;
    if (!path) return NULL;
    m = kmamba_load_ser(path, 0);
    if (!m) return NULL;
    if (for_training && opt_blocks) {
        if (kmamba_enable_training(m, opt_blocks, lr_embed_head, weight_decay) != 0) {
            kmamba_free(m);
            return NULL;
        }
    }
    return m;
}

float* kmamba_get_tensor(KMamba *m, const char *name) {
    size_t li;
    if (!m || !name) return NULL;
    if (strcmp(name, "embedding") == 0) return m->embedding;
    if (strcmp(name, "head") == 0) return m->head;

    if (sscanf(name, "layers.%zu.W_in", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->W_in.data;
    if (sscanf(name, "layers.%zu.W_out", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->W_out.data;
    if (sscanf(name, "layers.%zu.A_log", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->A_log.data;
    if (sscanf(name, "layers.%zu.W_B", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->W_B.data;
    if (sscanf(name, "layers.%zu.W_C", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->W_C.data;
    if (sscanf(name, "layers.%zu.delta_proj", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->delta_proj.data;
    if (sscanf(name, "layers.%zu.lambda_proj", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->lambda_proj.data;
    if (sscanf(name, "layers.%zu.b_B", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->b_B;
    if (sscanf(name, "layers.%zu.b_C", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->b_C;
    if (sscanf(name, "layers.%zu.theta", &li) == 1 && li < m->cfg.n_layers) return m->layers[li]->theta;
    return NULL;
}

static int kmamba_tensor_size(const KMamba *m, const char *name, size_t *n_out) {
    size_t li, D, N, R;
    if (!m || !name || !n_out) return -1;
    D = m->cfg.dim;
    N = m->cfg.state_size;
    R = _mimo_R(m->cfg.mimo_rank);

    if (strcmp(name, "embedding") == 0) { *n_out = m->cfg.vocab_size * D; return 0; }
    if (strcmp(name, "head") == 0) { *n_out = D * m->cfg.vocab_size; return 0; }

    if (sscanf(name, "layers.%zu.W_in", &li) == 1 && li < m->cfg.n_layers) { *n_out = R * D; return 0; }
    if (sscanf(name, "layers.%zu.W_out", &li) == 1 && li < m->cfg.n_layers) { *n_out = D * R; return 0; }
    if (sscanf(name, "layers.%zu.A_log", &li) == 1 && li < m->cfg.n_layers) { *n_out = N; return 0; }
    if (sscanf(name, "layers.%zu.W_B", &li) == 1 && li < m->cfg.n_layers) { *n_out = N * R * D; return 0; }
    if (sscanf(name, "layers.%zu.W_C", &li) == 1 && li < m->cfg.n_layers) { *n_out = N * R * D; return 0; }
    if (sscanf(name, "layers.%zu.delta_proj", &li) == 1 && li < m->cfg.n_layers) { *n_out = D; return 0; }
    if (sscanf(name, "layers.%zu.lambda_proj", &li) == 1 && li < m->cfg.n_layers) { *n_out = D; return 0; }
    if (sscanf(name, "layers.%zu.b_B", &li) == 1 && li < m->cfg.n_layers) { *n_out = N * R; return 0; }
    if (sscanf(name, "layers.%zu.b_C", &li) == 1 && li < m->cfg.n_layers) { *n_out = N * R; return 0; }
    if (sscanf(name, "layers.%zu.theta", &li) == 1 && li < m->cfg.n_layers) { *n_out = (N / 2 > 0 ? N / 2 : 1); return 0; }
    return -1;
}

int kmamba_set_tensor(KMamba *m, const char *name, const float *data) {
    float *dst;
    size_t n;
    if (!m || !name || !data) return -1;
    dst = kmamba_get_tensor(m, name);
    if (!dst) return -1;
    if (kmamba_tensor_size(m, name, &n) != 0) return -1;
    memcpy(dst, data, n * sizeof(float));
    return 0;
}

int kmamba_set_vocab(KMamba *m, uint32_t id, const char *token, uint16_t len) {
    (void)token;
    (void)len;
    if (!m) return -1;
    if (id >= m->cfg.vocab_size) return -1;
    return 0;
}

float kmamba_last_grad_norm(const KMamba *m) {
    return m ? m->last_grad_norm : 0.0f;
}

float kmamba_last_grad_over_clip(const KMamba *m) {
    return m ? m->last_grad_over_clip : 0.0f;
}

int kmamba_last_grad_would_clip(const KMamba *m) {
    return m ? m->last_grad_would_clip : 0;
}
