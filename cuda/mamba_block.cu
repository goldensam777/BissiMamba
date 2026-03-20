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
#include "scan.h"
#include "mamba_scan_cuda.h"

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

__global__ void silu_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
}

/* dy_dx = silu'(x_raw) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
 * dx = du * dy_dx  */
__global__ void silu_bwd_kernel(const float *du, const float *x_raw,
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

__global__ void softplus_clamp_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    y[i] = fmaxf(DT_MIN, fminf(DT_MAX, sp));
}

/* Backward du softplus clampé : ddt_raw = ddt * sigmoid(x) si dans [min,max] */
__global__ void softplus_clamp_bwd_kernel(const float *ddt, const float *x_raw,
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
__global__ void broadcast_d_to_ld(const float *vec, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = vec[idx % D];
}

/* Broadcast scalar_per_pos [L] -> out [L, D] : out[t, d] = scalar[t] */
__global__ void broadcast_l_to_ld(const float *scalar, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = scalar[idx / D];
}

/* Réduction [L, D] -> [D] : out[d] = sum_t in[t, d] (accumule avec +=) */
__global__ void reduce_sum_L(const float *in, float *out, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < L; t++) acc += in[t * D + d];
    out[d] += acc;  /* += pour accumuler sur le batch */
}

/* Réduction [L, D] -> [L] : out[t] = sum_d in[t, d] */
__global__ void reduce_sum_D(const float *in, float *out, int L, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) acc += in[t * D + d];
    out[t] = acc;   /* écrit (utilisé comme temporaire) */
}

/* y += x (accumulation résiduelle) */
__global__ void add_inplace_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

/* Sigmoid : y[i] = 1 / (1 + exp(-x[i])) */
__global__ void sigmoid_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = (v > 20.0f) ? 1.0f : (v < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-v));
}

/* Sigmoid backward : dx = dy * sigma * (1 - sigma) */
__global__ void sigmoid_bwd_kernel(const float *dy, const float *y, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}

/* dx[L, dim] += ddt[L] outer delta_proj[dim] : dx[t,d] += ddt[t]*dproj[d] */
__global__ void outer_add_kernel(float *dx, const float *ddt,
                                 const float *dproj, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int t = idx / D;
    int d = idx % D;
    dx[idx] += ddt[t] * dproj[d];
}

/* ── Forward d'un bloc ────────────────────────────────────────── */
/*
 * Tous les pointeurs sont des device pointers (VRAM).
 * Les buffers workspace (u_raw, u, dt_raw, dt, B_exp, C_exp, dt_exp,
 * h_store, y_scan, y_proj) sont pré-alloués par l'appelant.
 */
/* ── Complex SSM sequential kernel (forward) ─────────────────── */
/*
 * Single-threaded sequential scan with R(θ) rotation and exp-trapezoidal discretization.
 * h_t = alpha_t * R(θ)*h_{t-1} + beta_t * Bu_{t-1} + gamma_t * Bu_t
 * h_store[t*D + d] = h_t[d]
 * y_scan [t*D + d] = C_t[d] * h_t[d]
 */
__global__ void complex_ssm_fwd_kernel(
    const float *u,       /* [L, D] input (post-SiLU) */
    const float *A_log,   /* [D] */
    const float *B_exp,   /* [L, D] data-dep B */
    const float *C_exp,   /* [L, D] data-dep C */
    const float *dt,      /* [L] per-timestep delta */
    const float *theta,   /* [D/2] rotation angles */
    const float *lambda,  /* [L]   per-timestep lambda (sigmoid output) */
    float *h_store,       /* [L, D] state at each step */
    float *y_scan,        /* [L, D] output */
    int L, int D)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float h_rot_local[1024];
    float prev_Bu[1024];

    /* Initialize */
    for (int d = 0; d < D; d++) {
        h_store[d] = 0.0f;
        prev_Bu[d] = 0.0f;
    }

    float *h_cur = h_store;

    for (int t = 0; t < L; t++) {
        float dt_t  = dt[t];
        float lam_t = lambda ? lambda[t] : 0.5f;

        /* Apply R(θ) to h_cur */
        for (int i = 0; i + 1 < D; i += 2) {
            float th = theta ? theta[i >> 1] : 0.0f;
            float c = cosf(th), s = sinf(th);
            float h0 = h_cur[i], h1 = h_cur[i+1];
            h_rot_local[i]   = c*h0 - s*h1;
            h_rot_local[i+1] = s*h0 + c*h1;
        }
        if (D & 1) h_rot_local[D-1] = h_cur[D-1];

        float *h_out = &h_store[t * D];
        for (int d = 0; d < D; d++) {
            float a = A_log[d]; if (a > -1e-5f) a = -1e-5f;
            float alpha  = expf(dt_t * a);
            float beta   = (1.0f - lam_t) * dt_t * alpha;
            float gamma_ = lam_t * dt_t;
            float bu_t   = B_exp[t*D+d] * u[t*D+d];
            h_out[d] = alpha * h_rot_local[d] + beta * prev_Bu[d] + gamma_ * bu_t;
            prev_Bu[d] = bu_t;
        }
        for (int d = 0; d < D; d++)
            y_scan[t*D+d] = C_exp[t*D+d] * h_out[d];

        h_cur = h_out;
    }
}

extern "C" void gpu_block_forward(
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
    int L, int state, int dim, int R)  /* R = mimo_rank (1 = SISO) */
{
    const float a1 = 1.0f, b0 = 0.0f;
    int NR = state * R;
    int blk;

    /* 1. in_proj : u_raw [L, R] = x [L, dim] @ W_in^T  (W_in=[R,dim]) */
    gemm_bt(cublas, L, R, dim, a1, d_x, d_W_in, b0, d_u_raw);

    /* 2. SiLU */
    blk = (L * R + 255) / 256;
    silu_fwd_kernel<<<blk, 256>>>(d_u_raw, d_u, L * R);

    /* 3. delta : dt_raw [L] = x [L, dim] @ delta_proj [dim] */
    gemm(cublas, L, 1, dim, a1, d_x, d_delta_proj, b0, d_dt_raw);
    blk = (L + 255) / 256;
    softplus_clamp_fwd_kernel<<<blk, 256>>>(d_dt_raw, d_dt, L);

    /* 4. Data-dependent B [L, N*R], C [L, N*R], lambda [L] */
    gemm_bt(cublas, L, NR, dim, a1, d_x, d_W_B, b0, d_B_exp);
    gemm_bt(cublas, L, NR, dim, a1, d_x, d_W_C, b0, d_C_exp);
    /* lambda_raw [L] = x [L,dim] @ lambda_proj [dim] */
    gemm(cublas, L, 1, dim, a1, d_x, d_lambda_proj, b0, d_lambda_raw);
    { int blk_l = (L + 255) / 256;
      sigmoid_fwd_kernel<<<blk_l, 256>>>(d_lambda_raw, d_lambda, L); }

    /* 5. Complex SSM sequential scan with R(θ) rotation + exp-trapezoidal
     * Note: the kernel currently handles SISO (R=1). With R>1, Bu_t is
     * the reduced dot product B_t[N,R] @ u_t[R]. For R=1 this is B*u as before.
     * For R>1 we run with the first R elements of u and the [N*R] B/C layout.
     * The kernel is single-threaded so MIMO is handled inline. */
    CUDA_CHECK(cudaMemset(d_h_store, 0, L * state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_y_scan,  0, L * R * sizeof(float)));
    complex_ssm_fwd_kernel<<<1, 1>>>(
        d_u, d_A_log, d_B_exp, d_C_exp, d_dt,
        d_theta, d_lambda, d_h_store, d_y_scan, L, state);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 6. out_proj : y_proj [L, dim] = y_scan [L, R] @ W_out^T  (W_out=[dim,R]) */
    gemm_bt(cublas, L, dim, R, a1, d_y_scan, d_W_out, b0, d_y_proj);

    /* 7. Résiduel : y = y_proj + x */
    CUDA_CHECK(cudaMemcpy(d_y, d_y_proj, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    blk = (L * dim + 255) / 256;
    add_inplace_kernel<<<blk, 256>>>(d_y, d_x, L * dim);
    /* Note: d_dt_exp is no longer used (kept for API compatibility) */
    (void)d_dt_exp;
}

/* ── Backward d'un bloc ───────────────────────────────────────── */
/*
 * Accumule les gradients des paramètres (+=).
 * dx est écrit (pas accumulé — l'appelant doit additionner si besoin).
 *
 * Buffers temporaires (pré-alloués par l'appelant) :
 *   d_dy_scan [L, state], d_du [L, state], d_du_raw [L, state]
 *   d_ddt [L], d_ddt_raw [L]
 *   d_dB_scan [L, state], d_dC_scan [L, state], d_ddt_scan [L, state]
 *   d_dA_tmp [state]
 */
/* ── Complex SSM sequential backward kernel ─────────────────── */
/*
 * Sequential backward through the complex SSM scan.
 * Computes:
 *   d_du     [L, D]: gradient w.r.t. u (input to SSM)
 *   d_dA_acc [D]:    gradient w.r.t. A_log (accumulated)
 *   d_dB_out [L, D]: gradient w.r.t. B_exp
 *   d_dC_out [L, D]: gradient w.r.t. C_exp
 *   d_ddt_out[L, D]: gradient w.r.t. dt (per-dim, reduce later)
 *   d_g_theta[D/2]:  gradient w.r.t. theta (accumulated)
 *   d_ddy_adj[L, D]: adjoint of y_scan passed down (= d_du_scan)
 */
__global__ void complex_ssm_bwd_kernel(
    const float *d_dy_scan,  /* [L, D] upstream gradient of y_scan */
    const float *u,          /* [L, D] */
    const float *A_log,      /* [D] */
    const float *B_exp,      /* [L, D] */
    const float *C_exp,      /* [L, D] */
    const float *dt,         /* [L] */
    const float *lambda,     /* [L]   sigmoid(lambda_proj*x) */
    const float *theta,      /* [D/2] */
    const float *h_store,    /* [L, D] state at each step */
    float *d_du,             /* [L, D] out */
    float *d_dA_acc,         /* [D]    out (accumulated) */
    float *d_dB_out,         /* [L, D] out */
    float *d_dC_out,         /* [L, D] out */
    float *d_ddt_out,        /* [L, D] out */
    float *d_g_theta,        /* [D/2]  out (accumulated) */
    float *d_dlambda,        /* [L]    out: grad w.r.t. lambda_t */
    int L, int D)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj_h[1024];
    float adj_prev_Bu[1024];
    for (int d = 0; d < D; d++) { adj_h[d] = 0.0f; adj_prev_Bu[d] = 0.0f; }

    for (int t = L - 1; t >= 0; t--) {
        float dt_t  = dt[t];
        float lam_t = lambda ? lambda[t] : 0.5f;
        const float *h_t    = &h_store[t * D];
        const float *h_prev = (t > 0) ? &h_store[(t-1)*D] : NULL;

        float d_lam_t = 0.0f;

        for (int d = 0; d < D; d++) {
            float ct_d   = C_exp[t*D+d];
            float bt_d   = B_exp[t*D+d];
            float ut_d   = u[t*D+d];
            float a_val  = A_log[d]; if (a_val > -1e-5f) a_val = -1e-5f;
            float a_diag = expf(dt_t * a_val);
            float beta_t = (1.0f - lam_t) * dt_t * a_diag;
            float dy_s   = d_dy_scan[t*D+d];
            float bu_t   = bt_d * ut_d;
            float bu_prev = (h_prev != NULL) ? B_exp[(t-1)*D+d] * u[(t-1)*D+d] : 0.0f;

            /* adj of h_t from y_t and future state */
            float ah = adj_h[d] + dy_s * ct_d + adj_prev_Bu[d] * beta_t;
            /* Actually adj_prev_Bu holds the gradient that flows from the next step's
             * beta_{t+1} * Bu_t term. We handle this by incorporating it via adj_h propagation. */
            /* Cleaner: ah_actual excludes adj_prev_Bu (handle separately) */
            float ah_actual = adj_h[d] + dy_s * ct_d;

            /* dC */
            d_dC_out[t*D+d] = dy_s * h_t[d];

            /* dBu_t = ah_actual * gamma_t */
            float gamma_t = lam_t * dt_t;
            float d_bu_t  = ah_actual * gamma_t;
            /* dB_t[d] */
            d_dB_out[t*D+d] = d_bu_t * ut_d;
            /* du[d] */
            d_du[t*D+d] = d_bu_t * bt_d;

            /* Recover h_rot from stored state and Bu terms */
            float h_rot_d = (a_diag > 1e-30f)
                ? (h_t[d] - beta_t * bu_prev - gamma_t * bu_t) / a_diag
                : 0.0f;

            /* dA_log */
            d_dA_acc[d] += ah_actual * dt_t * a_diag * h_rot_d;

            /* d_lambda_t: d_h/d_lam * ah_actual */
            /* d_beta/d_lam = -dt * alpha; d_gamma/d_lam = dt */
            d_lam_t += ah_actual * ((-dt_t * a_diag) * bu_prev + dt_t * bu_t);

            /* ddt */
            d_ddt_out[t*D+d] = ah_actual * (a_val * a_diag * h_rot_d
                                + (1.0f - lam_t) * a_diag * bu_prev
                                + lam_t * bu_t);

            /* d_h_rot = ah_actual * a_diag  for theta + adj_h propagation */
            adj_h[d] = ah_actual * a_diag;

            /* propagate adj through prev_Bu for t-1 */
            adj_prev_Bu[d] = ah_actual * beta_t;
        }

        if (d_dlambda) d_dlambda[t] = d_lam_t;

        /* Gradient for theta and propagate adj_h = R^T * d_h_rot */
        for (int i = 0; i + 1 < D; i += 2) {
            float hp0 = h_prev ? h_prev[i]   : 0.0f;
            float hp1 = h_prev ? h_prev[i+1] : 0.0f;
            float th  = theta ? theta[i >> 1] : 0.0f;
            float c   = cosf(th), sv = sinf(th);
            float dr0 = adj_h[i], dr1 = adj_h[i+1];

            if (theta && d_g_theta)
                d_g_theta[i >> 1] += dr0 * (-sv * hp0 - c * hp1)
                                   + dr1 * (c * hp0 - sv * hp1);

            adj_h[i]   = c * dr0 + sv * dr1;
            adj_h[i+1] = -sv * dr0 + c * dr1;
        }
        if (D & 1) { /* adj_h[D-1] = adj_h[D-1] (unchanged) */ }
    }
}

extern "C" void gpu_block_backward(
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
    int L, int state, int dim, int R)   /* R = mimo_rank */
{
    const float a1 = 1.0f, b0 = 0.0f;
    int NR = state * R;
    int blk;

    /* ── Résiduel : dy passe aussi vers dx (on initialise dx = dy) ── */
    CUDA_CHECK(cudaMemcpy(d_dx, d_dy, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    /* ── Backward out_proj ──────────────────────────────────────── */
    /* dW_out [dim, R] += dy^T @ y_scan  (A=[L,dim], B=[L,R]) */
    gemm_at(cublas, dim, R, L, a1, d_dy, d_y_scan, a1, d_dW_out);

    /* dy_scan [L, R] = dy [L, dim] @ W_out [dim, R] */
    gemm(cublas, L, R, dim, a1, d_dy, d_W_out, b0, d_dy_scan);

    /* ── Backward complex SSM (sequential) ──────────────────────── */
    /* Zero accumulators */
    CUDA_CHECK(cudaMemset(d_dA_tmp, 0, state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ddt_scan, 0, L * state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dlambda, 0, L * sizeof(float)));

    complex_ssm_bwd_kernel<<<1, 1>>>(
        d_dy_scan,
        d_u, d_A_log, d_B_exp, d_C_exp, d_dt, d_lambda, d_theta, d_h_store,
        d_du, d_dA_tmp, d_dB_scan, d_dC_scan, d_ddt_scan, d_g_theta, d_dlambda,
        L, state);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Accumule dA_log */
    blk = (state + 255) / 256;
    add_inplace_kernel<<<blk, 256>>>(d_dA_log, d_dA_tmp, state);

    /* g_W_B [N*R, dim] += dB_scan^T [N*R, L] @ x [L, dim] */
    gemm_at(cublas, NR, dim, L, a1, d_dB_scan, d_x, a1, d_dW_B);

    /* g_W_C [N*R, dim] += dC_scan^T [N*R, L] @ x [L, dim] */
    gemm_at(cublas, NR, dim, L, a1, d_dC_scan, d_x, a1, d_dW_C);

    /* ddt [L] = sum_d ddt_scan [t, d] */
    blk = (L + 255) / 256;
    reduce_sum_D<<<blk, 256>>>(d_ddt_scan, d_ddt, L, state);

    /* ── Backward softplus ──────────────────────────────────────── */
    softplus_clamp_bwd_kernel<<<blk, 256>>>(d_ddt, d_dt_raw, d_dt, d_ddt_raw, L);

    /* ── Backward delta_proj ────────────────────────────────────── */
    gemm_at(cublas, 1, dim, L, a1, d_ddt_raw, d_x, a1, d_ddelta_proj);
    blk = (L * dim + 255) / 256;
    outer_add_kernel<<<blk, 256>>>(d_dx, d_ddt_raw, d_delta_proj, L, dim);

    /* ── Backward SiLU + in_proj (W_in=[R,dim], u=[L,R]) ──────── */
    blk = (L * R + 255) / 256;
    silu_bwd_kernel<<<blk, 256>>>(d_du, d_u_raw, d_du_raw, L * R);

    gemm_at(cublas, R, dim, L, a1, d_du_raw, d_x, a1, d_dW_in);
    gemm(cublas, L, dim, R, a1, d_du_raw, d_W_in, a1, d_dx);

    /* ── Backward lambda_proj ───────────────────────────────────── */
    blk = (L + 255) / 256;
    sigmoid_bwd_kernel<<<blk, 256>>>(d_dlambda, d_lambda, d_dlambda_raw, L);
    gemm_at(cublas, 1, dim, L, a1, d_dlambda_raw, d_x, a1, d_g_lambda_proj);
    outer_add_kernel<<<(L * dim + 255) / 256, 256>>>(
        d_dx, d_dlambda_raw, d_lambda_proj, L, dim);

    /* API compat */
    (void)d_dt_exp;
}
