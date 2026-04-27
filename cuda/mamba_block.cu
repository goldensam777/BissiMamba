/*
 * mamba_block.cu — Forward + backward GPU pour un bloc Mamba
 *
 * Pipeline forward :
 *   x [L, dim]
 *     -> W_in^T GEMM  -> u_raw [L, state]  -> SiLU      -> u [L, state]
 *     -> delta_proj   -> dt_raw [L]         -> softplus  -> dt [L]
 *     -> broadcast    -> B_exp, C_exp, dt_exp [L, state]
 *     -> scan1d (Blelloch GPU)              -> h_store [L, state], y_scan [L, state]
 *     -> W_out^T GEMM -> y_proj [L, dim]
 *     -> residual     -> y = y_proj + x
 *
 * Toutes les opérations sont sur GPU.
 * cuBLAS pour les GEMM, kernels custom pour elementwise + reductions.
 *
 * Convention GEMM row-major via cuBLAS (col-major interne) :
 *   C[M,N] = A[M,K] @ B[K,N]  :  cublasSgemm(h,N,N, N,M,K, a, B,N, A,K, b, C,N)
 *   C[M,N] = A[M,K] @ B^T     :  cublasSgemm(h,T,N, N,M,K, a, B,K, A,K, b, C,N)  (B=[N,K])
 *   C[M,N] = A^T   @ B[K,N]   :  cublasSgemm(h,N,T, N,M,K, a, B,N, A,M, b, C,N)  (A=[K,M])
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "scan_nd.h"

/* ── Macro de vérification ────────────────────────────────────── */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d — %d\n", __FILE__, __LINE__, _s); \
        exit(1); \
    } \
} while(0)

/* ── Helpers GEMM row-major ───────────────────────────────────── */

/* C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N] */
static void gemm(cublasHandle_t h, int M, int N, int K,
                 float alpha, const float *A, const float *B,
                 float beta,  float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

/* C[M,N] = alpha * A[M,K] @ B^T + beta * C  (B est [N,K]) */
static void gemm_bt(cublasHandle_t h, int M, int N, int K,
                    float alpha, const float *A, const float *B,
                    float beta,  float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha, B, K, A, K, &beta, C, N));
}

/* C[M,N] = alpha * A^T @ B + beta * C  (A est [K,M], B est [K,N]) */
static void gemm_at(cublasHandle_t h, int M, int N, int K,
                    float alpha, const float *A, const float *B,
                    float beta,  float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

/* ── Kernels elementwise ──────────────────────────────────────── */

__global__ void cuda_silu_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
}

/* dy_dx = silu'(x_raw) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
 * dx = du * dy_dx  */
__global__ void cuda_silu_bwd_kernel(const float *du, const float *x_raw,
                                float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v   = x_raw[i];
    float sig = 1.0f / (1.0f + expf(-v));
    dx[i] = du[i] * sig * (1.0f + v * (1.0f - sig));
}

/* softplus avec clamp : dt = clamp(log(1+exp(x)), dt_min, dt_max) */
#define DT_MIN 1e-3f
#define DT_MAX 0.1f

__global__ void cuda_softplus_clamp_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    y[i] = fmaxf(DT_MIN, fminf(DT_MAX, sp));
}

/* Backward du softplus clampé : ddt_raw = ddt * sigmoid(x) si dans [min,max] */
__global__ void cuda_softplus_clamp_bwd_kernel(const float *ddt, const float *x_raw,
                                          const float *dt, float *ddt_raw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float clamped = dt[i];
    if (clamped <= DT_MIN || clamped >= DT_MAX) {
        ddt_raw[i] = 0.0f;
    } else {
        float sig = 1.0f / (1.0f + expf(-x_raw[i]));
        ddt_raw[i] = ddt[i] * sig;
    }
}

/* Broadcast vec [D] -> out [L, D] : out[t, d] = vec[d] */
__global__ void cuda_broadcast_d_to_ld(const float *vec, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = vec[idx % D];
}

/* Broadcast scalar_per_pos [L] -> out [L, D] : out[t, d] = scalar[t] */
__global__ void cuda_broadcast_l_to_ld(const float *scalar, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = scalar[idx / D];
}

/* Réduction [L, D] -> [D] : out[d] = sum_t in[t, d] (accumule avec +=) */
__global__ void cuda_reduce_sum_L(const float *in, float *out, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < L; t++) acc += in[t * D + d];
    out[d] += acc;  /* += pour accumuler sur le batch */
}

/* Réduction [L, D] -> [L] : out[t] = sum_d in[t, d] */
__global__ void cuda_reduce_sum_D(const float *in, float *out, int L, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) acc += in[t * D + d];
    out[t] = acc;   /* écrit (utilisé comme temporaire) */
}

/* y += x (accumulation résiduelle) */
__global__ void cuda_add_inplace_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

/* Sigmoid : y[i] = 1 / (1 + exp(-x[i])) */
__global__ void cuda_sigmoid_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = (v > 20.0f) ? 1.0f : (v < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-v));
}

/* Sigmoid backward : dx = dy * sigma * (1 - sigma) */
__global__ void cuda_sigmoid_bwd_kernel(const float *dy, const float *y, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}

/* dx[L, dim] += ddt[L] outer delta_proj[dim] : dx[t,d] += ddt[t]*dproj[d] */
__global__ void cuda_outer_add_kernel(float *dx, const float *ddt,
                                 const float *dproj, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int t = idx / D;
    int d = idx % D;
    dx[idx] += ddt[t] * dproj[d];
}

/* AdamW GPU kernel */
__global__ void cuda_adamw_step_kernel(float *param, const float *grad,
                                      float *m, float *v,
                                      float lr, float beta1, float beta2,
                                      float eps, float wd,
                                      int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float p = param[i];

    /* Weight decay */
    p -= lr * wd * p;

    /* Update moments */
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    /* Bias correction */
    float m_hat = mi / (1.0f - powf(beta1, (float)t));
    float v_hat = vi / (1.0f - powf(beta2, (float)t));

    /* Update parameter */
    param[i] = p - lr * m_hat / (sqrtf(v_hat) + eps);
}

/* ── Forward d'un bloc ────────────────────────────────────────── */
/*
 * Tous les pointeurs sont des device pointers (VRAM).
 * Les buffers workspace (u_raw, u, dt_raw, dt, B_exp, C_exp, dt_exp,
 * h_store, y_scan, y_proj) sont pré-alloués par l'appelant.
 */
/* ── Forward d'un bloc ────────────────────────────────────────── */
/*
 * Tous les pointeurs sont des device pointers (VRAM).
 * Les buffers workspace sont pré-alloués par l'appelant.
 */
extern "C" void cuda_block_forward(
    cublasHandle_t cublas,
    /* Paramètres du bloc [VRAM] */
    const float *d_W_in,        /* [R, dim]      — MIMO: R=mimo_rank, R=1 for SISO */
    const float *d_W_out,       /* [dim, R]      */
    const float *d_A_log,       /* [state]       */
    const float *d_W_B,         /* [N*R, dim]    data-dependent B projection */
    const float *d_W_C,         /* [N*R, dim]    data-dependent C projection */
    const float *d_delta_proj,  /* [dim]         */
    const float *d_theta,       /* [state/2]     rotation angles (may be NULL) */
    const float *d_lambda_proj, /* [dim]         exp-trapezoidal lambda projection */
    /* Entrée / sortie */
    const float *d_x,           /* [L, dim]  input  */
    float       *d_y,           /* [L, dim]  output */
    /* Workspace */
    float *d_u_raw,   float *d_u,    /* [L, R]        */
    float *d_dt_raw,  float *d_dt,   /* [L]           */
    float *d_B_exp,   float *d_C_exp, float *d_dt_exp, /* [L, N*R] / [L, N*R] */
    float *d_h_store, float *d_y_scan, float *d_y_proj, /* [L,N] / [L,R] / [L,dim] */
    float *d_lambda_raw, float *d_lambda,  /* [L] workspace for lambda */
    int L, int state, int dim, int R,  /* R = mimo_rank (1 = SISO) */
    int spatial_ndims, const long *spatial_dims)
{
    if (!cublas) return;
    const float a1 = 1.0f, b0 = 0.0f;
    int NR = state * R;
    int blk;

    /* 1. in_proj : u_raw [L, R] = x [L, dim] @ W_in^T  (W_in=[R,dim]) */
    gemm_bt(cublas, L, R, dim, a1, d_x, d_W_in, b0, d_u_raw);

    /* 2. SiLU */
    blk = (L * R + 255) / 256;
    cuda_silu_fwd_kernel<<<blk, 256>>>(d_u_raw, d_u, L * R);

    /* 3. delta : dt_raw [L] = x [L, dim] @ delta_proj [dim] */
    gemm(cublas, L, 1, dim, a1, d_x, d_delta_proj, b0, d_dt_raw);
    blk = (L + 255) / 256;
    cuda_softplus_clamp_fwd_kernel<<<blk, 256>>>(d_dt_raw, d_dt, L);

    /* 4. Data-dependent B [L, N*R], C [L, N*R], lambda [L] */
    gemm_bt(cublas, L, NR, dim, a1, d_x, d_W_B, b0, d_B_exp);
    gemm_bt(cublas, L, NR, dim, a1, d_x, d_W_C, b0, d_C_exp);
    gemm(cublas, L, 1, dim, a1, d_x, d_lambda_proj, b0, d_lambda_raw);
    { int blk_l = (L + 255) / 256;
      cuda_sigmoid_fwd_kernel<<<blk_l, 256>>>(d_lambda_raw, d_lambda, L); }

    /* 5. SSM scan via unified ScanND backend */
    ScanNDParams p;
    p.max_ndims = 8;
    p.max_state = 64;
    p.use_fast_exp = 0;
    p.dims = spatial_dims;
    p.ndims = spatial_ndims;
    p.D = (long)state;
    p.M = 1;
    p.x = d_u;
    p.A = d_A_log;
    p.B = d_B_exp;
    p.C = d_C_exp;
    p.delta = d_dt;
    p.h = d_h_store;
    p.y = d_y_scan;
    p.theta = d_theta;
    p.lambda = d_lambda;
    p.default_lambda = 0.5f;
    p.use_a_log_clamp = 0;
    p.a_log_min = -1e-5f;

    om_scannd_forward(&p);

    /* 6. out_proj : y_proj [L, dim] = y_scan [L, R] @ W_out^T  (W_out=[dim,R]) */
    gemm_bt(cublas, L, dim, R, a1, d_y_scan, d_W_out, b0, d_y_proj);

    /* 7. Résiduel : y = y_proj + x */
    CUDA_CHECK(cudaMemcpy(d_y, d_y_proj, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    blk = (L * dim + 255) / 256;
    cuda_add_inplace_kernel<<<blk, 256>>>(d_y, d_x, L * dim);
    (void)d_dt_exp;
}

/* ── Backward d'un bloc ───────────────────────────────────────── */
extern "C" void cuda_block_backward(
    cublasHandle_t cublas,
    /* Paramètres (lecture seule) */
    const float *d_W_in, const float *d_W_out,   /* [R,dim] / [dim,R] */
    const float *d_A_log,
    const float *d_W_B, const float *d_W_C,       /* [N*R, dim] */
    const float *d_delta_proj,
    const float *d_theta,         /* [state/2] rotation angles */
    const float *d_lambda_proj,   /* [dim] lambda projection */
    /* Activations sauvées au forward */
    const float *d_x,
    const float *d_u_raw, const float *d_u,       /* [L, R] */
    const float *d_dt_raw, const float *d_dt,
    const float *d_B_exp, const float *d_C_exp, const float *d_dt_exp, /* [L, N*R] */
    const float *d_h_store, const float *d_y_scan, /* [L,N] / [L,R] */
    const float *d_lambda,        /* [L] sigmoid output saved at forward */
    /* Gradient entrant */
    const float *d_dy,            /* [L, dim] upstream gradient */
    /* Gradients des paramètres (accumulés, +=) */
    float *d_dW_in, float *d_dW_out,
    float *d_dA_log,
    float *d_dW_B, float *d_dW_C,
    float *d_ddelta_proj,
    float *d_g_theta,             /* [state/2] grad for theta */
    float *d_g_lambda_proj,       /* [dim] grad for lambda_proj */
    /* Gradient de sortie */
    float *d_dx,                  /* [L, dim] downstream gradient */
    /* Workspace temporaire */
    float *d_dy_scan,           /* [L, R] */
    float *d_du,                /* [L, R] */
    float *d_du_raw,            /* [L, R] */
    float *d_ddt,               /* [L] */
    float *d_ddt_raw,           /* [L] */
    float *d_dB_scan,           /* [L, N*R] scan gradient de B */
    float *d_dC_scan,           /* [L, N*R] scan gradient de C */
    float *d_ddt_scan,          /* [L, state] scan gradient de dt */
    float *d_dA_tmp,            /* [state]   scan gradient de A (tmp) */
    float *d_dlambda,           /* [L]       scan gradient of lambda */
    float *d_dlambda_raw,       /* [L]       grad through sigmoid */
    int L, int state, int dim, int R,   /* R = mimo_rank */
    int spatial_ndims, const long *spatial_dims)
{
    const float a1 = 1.0f, b0 = 0.0f;
    int NR = state * R;
    int blk;

    /* ── Résiduel : dy passe aussi vers dx (on initialise dx = dy) ── */
    CUDA_CHECK(cudaMemcpy(d_dx, d_dy, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    /* ── Backward out_proj ──────────────────────────────────────── */
    gemm_at(cublas, dim, R, L, a1, d_dy, d_y_scan, a1, d_dW_out);
    gemm(cublas, L, R, dim, a1, d_dy, d_W_out, b0, d_dy_scan);

    /* ── Backward complex SSM (via ScanND) ─────────────────────── */
    ScanNDParams p;
    p.max_ndims = 8;
    p.max_state = 64;
    p.use_fast_exp = 0;
    p.dims = spatial_dims;
    p.ndims = spatial_ndims;
    p.D = (long)state;
    p.M = 1;
    p.x = d_u;
    p.A = d_A_log;
    p.B = d_B_exp;
    p.C = d_C_exp;
    p.delta = d_dt;
    p.h = (float*)d_h_store;
    p.y = (float*)d_dy_scan;
    p.theta = d_theta;
    p.lambda = d_lambda;
    p.default_lambda = 0.5f;
    p.use_a_log_clamp = 0;
    p.a_log_min = -1e-5f;

    /* Grad outputs */
    p.dA = d_dA_tmp;
    p.dB = d_dB_scan;
    p.dC = d_dC_scan;
    p.ddelta = d_ddt_scan;
    p.dlambda = d_dlambda;
    p.dtheta = d_g_theta;

    om_scannd_backward(&p);

    /* ── Backward softplus ──────────────────────────────────────── */
    blk = (L + 255) / 256;
    cuda_softplus_clamp_bwd_kernel<<<blk, 256>>>(d_ddt, d_dt_raw, d_dt, d_ddt_raw, L);

    /* ── Backward delta_proj ────────────────────────────────────── */
    gemm_at(cublas, 1, dim, L, a1, d_ddt_raw, d_x, a1, d_ddelta_proj);
    blk = (L * dim + 255) / 256;
    cuda_outer_add_kernel<<<blk, 256>>>(d_dx, d_ddt_raw, d_delta_proj, L, dim);

    /* ── Backward SiLU + in_proj ──────────────────────────────── */
    blk = (L * R + 255) / 256;
    cuda_silu_bwd_kernel<<<blk, 256>>>(d_du, d_u_raw, d_du_raw, L * R);
    gemm_at(cublas, R, dim, L, a1, d_du_raw, d_x, a1, d_dW_in);
    gemm(cublas, L, dim, R, a1, d_du_raw, d_W_in, a1, d_dx);

    /* ── Backward lambda_proj ───────────────────────────────────── */
    blk = (L + 255) / 256;
    cuda_sigmoid_bwd_kernel<<<blk, 256>>>(d_dlambda, d_lambda, d_dlambda_raw, L);
    gemm_at(cublas, 1, dim, L, a1, d_dlambda_raw, d_x, a1, d_g_lambda_proj);
    cuda_outer_add_kernel<<<(L * dim + 255) / 256, 256>>>(
        d_dx, d_dlambda_raw, d_lambda_proj, L, dim);

    (void)d_dt_exp;
}

/* ============================================================
 * GPU Optimizer Step (Full GPU)
 * Applies AdamW step entirely on GPU.
 * ============================================================ */
#include "kmamba.h"
#include "kmamba_kernels.h"

extern "C" void gpu_optimizer_step(MambaBlock *block, const MBOptimConfig *conf) {
    if (!block || !block->opt_state || !block->gpu.gpu_ready) return;
    
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t R = block->config.mimo_rank > 0 ? block->config.mimo_rank : 1;
    size_t NR = N * R;
    size_t TS = D / 2;
    
    MBOptimState *s = (MBOptimState *)block->opt_state;
    s->step++;
    
    /* Helper to launch AdamW kernel on GPU */
    auto step_gpu = [&](float *d_param, float *d_grad, float *d_m, float *d_v, size_t n) {
        if (!d_param || !d_grad || !d_m || !d_v || n == 0) return;
        int blk = (int)((n + 255) / 256);
        cuda_adamw_step_kernel<<<blk, 256>>>(d_param, d_grad, d_m, d_v,
                                            conf->lr, 0.9f, 0.999f, conf->eps, 
                                            conf->weight_decay, (int)n, (int)s->step);
    };
    
    /* Apply to all parameters using the persistent GPU buffers in block->gpu */
    step_gpu(block->gpu.d_W_in,  block->gpu.d_g_W_in,  block->gpu.d_m_W_in,  block->gpu.d_v_W_in,  R * D);
    step_gpu(block->gpu.d_W_out, block->gpu.d_g_W_out, block->gpu.d_m_W_out, block->gpu.d_v_W_out, D * R);
    step_gpu(block->gpu.d_A_log, block->gpu.d_g_A_log, block->gpu.d_m_A_log, block->gpu.d_v_A_log, N);
    step_gpu(block->gpu.d_W_B,   block->gpu.d_g_W_B,   block->gpu.d_m_W_B,   block->gpu.d_v_W_B,   NR * D);
    step_gpu(block->gpu.d_W_C,   block->gpu.d_g_W_C,   block->gpu.d_m_W_C,   block->gpu.d_v_W_C,   NR * D);
    step_gpu(block->gpu.d_b_B,   block->gpu.d_g_b_B,   block->gpu.d_m_b_B,   block->gpu.d_v_b_B,   NR);
    step_gpu(block->gpu.d_b_C,   block->gpu.d_g_b_C,   block->gpu.d_m_b_C,   block->gpu.d_v_b_C,   NR);
    step_gpu(block->gpu.d_delta_proj,  block->gpu.d_g_delta_proj,  block->gpu.d_m_delta_proj,  block->gpu.d_v_delta_proj, D);
    step_gpu(block->gpu.d_lambda_proj, block->gpu.d_g_lambda_proj, block->gpu.d_m_lambda_proj, block->gpu.d_v_lambda_proj, D);
    if (block->theta)
        step_gpu(block->gpu.d_theta, block->gpu.d_g_theta, block->gpu.d_m_theta, block->gpu.d_v_theta, TS);
}

/* ── Wrapper C pour l'optimizer AdamW ─────────────────────────── */
extern "C" void cuda_adamw_step_wrapper(float *param, float *grad, float *m, float *v,
                                          float lr, float beta1, float beta2, float eps,
                                          float wd, int n, int step) {
    int blk = (n + 255) / 256;
    cuda_adamw_step_kernel<<<blk, 256>>>(param, grad, m, v, lr, beta1, beta2, eps, wd, n, step);
}
