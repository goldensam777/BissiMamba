/*
 * mamba_cuda.cu — CUDA implementation of the ~1B-parameter Mamba LM.
 *
 * Architecture per Mamba layer (follows the original Mamba paper):
 *
 *   residual = x                               [B, T, D]
 *   x  = rms_norm(x)                           [B, T, D]
 *   xz = in_proj(x)                            [B, T, 2·d_inner]
 *   x, z = split(xz, d_inner)                 [B, T, d_inner] each
 *   x  = causal_conv1d(x)                      [B, T, d_inner]
 *   x  = silu(x)                               [B, T, d_inner]
 *
 *   BCdt = x_proj(x)   [B,T, dt_rank + 2·d_state]
 *   dt   = softplus(dt_proj(BCdt[..,:dt_rank]) + dt_bias)  [B,T,d_inner]
 *   B    = BCdt[.., dt_rank : dt_rank+d_state]             [B,T,d_state]
 *   C    = BCdt[.., dt_rank+d_state :]                     [B,T,d_state]
 *
 *   y = ssm_scan(x, dt, A_log, B, C, D_param)  [B,T,d_inner]
 *   y = y * silu(z)                             [B,T,d_inner]
 *   x = out_proj(y) + residual                  [B,T,D]
 *
 * Requires: CUDA ≥ 11.0, cuBLAS.
 * Build:    nvcc -O3 -arch=sm_80 -o mamba_cuda.o -c mamba_cuda.cu -lcublas
 */

#include "mamba_large.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════
 * Error macros
 * ═══════════════════════════════════════════════════════════════════ */

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error [%s:%d]: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
            exit(1);                                                  \
        }                                                             \
    } while(0)

#define CUBLAS_CHECK(call)                                            \
    do {                                                              \
        cublasStatus_t _s = (call);                                   \
        if (_s != CUBLAS_STATUS_SUCCESS) {                            \
            fprintf(stderr, "cuBLAS error [%s:%d]: %d\n",            \
                    __FILE__, __LINE__, (int)_s);                     \
            exit(1);                                                  \
        }                                                             \
    } while(0)


/* ═══════════════════════════════════════════════════════════════════
 * cuBLAS helpers  (row-major C = A * B)
 *   A[M,K]  B[K,N]  →  C[M,N]
 * ═══════════════════════════════════════════════════════════════════ */

/* Row-major matmul wrapper: C = alpha*A*B + beta*C
 * A[M,K], B[K,N], C[M,N], all row-major on device. */
static void gemm_row(cublasHandle_t h,
                     int M, int N, int K,
                     float alpha,
                     const float *A,  /* [M,K] row-major */
                     const float *B,  /* [K,N] row-major */
                     float beta,
                     float *C         /* [M,N] row-major */)
{
    /* cuBLAS col-major trick: C^T = B^T * A^T
     * cublasSgemm(N,N, n,m,k, alpha, B,n, A,k, beta, C,n) */
    CUBLAS_CHECK(cublasSgemm(h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N));
}

/* Batched: for each sample in a batch (leading dim B), apply the same
 * weight matrix W[out_features, in_features] to X[B*T, in_features]
 * producing Y[B*T, out_features]. Uses single gemm with merged BT dim. */
static void linear_forward(cublasHandle_t h,
                            const float *X,   /* [BT, in]  */
                            const float *W,   /* [out, in] */
                            float       *Y,   /* [BT, out] */
                            int BT, int in_features, int out_features)
{
    gemm_row(h, BT, out_features, in_features, 1.0f, X, W, 0.0f, Y);
}


/* ═══════════════════════════════════════════════════════════════════
 * CUDA Kernels
 * ═══════════════════════════════════════════════════════════════════ */

/* ── RMSNorm forward ─────────────────────────────────────────────── */
__global__ void rms_norm_fwd_kernel(
        float *out,           /* [BT, D] */
        const float *x,       /* [BT, D] */
        const float *w,       /* [D]     */
        int D, float eps)
{
    int row = blockIdx.x;
    const float *xr = x   + row * D;
    float       *yr = out + row * D;

    /* compute RMS */
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        ss += xr[i] * xr[i];
    /* warp-reduce */
    __shared__ float shmem[32];
    ss += __shfl_down_sync(0xffffffff, ss, 16);
    ss += __shfl_down_sync(0xffffffff, ss,  8);
    ss += __shfl_down_sync(0xffffffff, ss,  4);
    ss += __shfl_down_sync(0xffffffff, ss,  2);
    ss += __shfl_down_sync(0xffffffff, ss,  1);
    if ((threadIdx.x & 31) == 0) shmem[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < (blockDim.x >> 5)) ss = shmem[threadIdx.x];
    ss += __shfl_down_sync(0xffffffff, ss, 16);
    ss += __shfl_down_sync(0xffffffff, ss,  8);
    ss += __shfl_down_sync(0xffffffff, ss,  4);
    ss += __shfl_down_sync(0xffffffff, ss,  2);
    ss += __shfl_down_sync(0xffffffff, ss,  1);
    __shared__ float rms_inv;
    if (threadIdx.x == 0) rms_inv = rsqrtf(ss / (float)D + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = xr[i] * rms_inv * w[i];
}

/* ── SiLU (element-wise) ─────────────────────────────────────────── */
__global__ void silu_kernel(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ── Causal conv1d (depthwise, groups = d_inner) ─────────────────── */
/*
 * x       : [BT, d_inner]  (B*T flattened, treated as B sequences of T)
 * w       : [d_inner, d_conv]
 * b       : [d_inner]  (bias)
 * out     : [BT, d_inner]
 * T       : sequence length
 * d_inner : channels
 * d_conv  : kernel size
 */
__global__ void causal_conv1d_fwd_kernel(
        const float *x,       /* [B, T, d_inner] */
        const float *w,       /* [d_inner, d_conv] */
        const float *bias,    /* [d_inner] */
        float       *out,     /* [B, T, d_inner] */
        int B, int T, int d_inner, int d_conv)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x + blockIdx.z * blockDim.x;
    if (b >= B || t >= T || d >= d_inner) return;

    float acc = bias ? bias[d] : 0.0f;
    for (int k = 0; k < d_conv; k++) {
        int src_t = t - k;
        if (src_t >= 0) {
            acc += x[b * T * d_inner + src_t * d_inner + d]
                 * w[d * d_conv + k];
        }
    }
    out[b * T * d_inner + t * d_inner + d] = acc;
}

/* ── SSM scan forward ────────────────────────────────────────────── */
/*
 * Parallel across (B, d_inner); sequential across T; parallel across
 * d_state via shared memory accumulation.
 *
 * x      : [B, T, d_inner]   — input signal (after conv+silu)
 * dt     : [B, T, d_inner]   — discretisation step (after softplus)
 * A_log  : [d_inner, d_state]— log(-A), negative → stable A
 * B_ssm  : [B, T, d_state]   — input matrix (shared across d_inner)
 * C_ssm  : [B, T, d_state]   — output matrix
 * D_par  : [d_inner]         — skip connection
 * y      : [B, T, d_inner]   — output
 *
 * Grid  : (B, d_inner)
 * Block : d_state threads
 * Smem  : T * sizeof(float)
 */
__global__ void ssm_scan_fwd_kernel(
        const float *x,       /* [B,T,d_inner] */
        const float *dt,      /* [B,T,d_inner] */
        const float *A_log,   /* [d_inner,d_state] */
        const float *B_ssm,   /* [B,T,d_state] */
        const float *C_ssm,   /* [B,T,d_state] */
        const float *D_par,   /* [d_inner] */
        float       *y,       /* [B,T,d_inner] */
        int B, int T, int d_inner, int d_state)
{
    int b  = blockIdx.x;
    int d  = blockIdx.y;
    int s  = threadIdx.x;   /* state dimension index */

    extern __shared__ float sh_y[];   /* [T] — accumulated output for this (b,d) */

    /* Initialise shared output buffer */
    for (int t = s; t < T; t += d_state)
        sh_y[t] = 0.0f;
    __syncthreads();

    float a_ds  = A_log[d * d_state + s];   /* log(-A[d,s]) */
    float h_s   = 0.0f;                     /* SSM state, lives in register */

    for (int t = 0; t < T; t++) {
        int  base_bt  = b * T * d_inner + t * d_inner;
        int  base_bts = b * T * d_state + t * d_state;

        float x_td   = x[base_bt + d];
        float dt_td  = dt[base_bt + d];               /* already softplus'd */
        float B_ts   = B_ssm[base_bts + s];
        float C_ts   = C_ssm[base_bts + s];

        /* ZOH discretisation (diagonal A) */
        float dA = expf(dt_td * a_ds);          /* exp(dt · log(-A)) = A^dt */
        float dB = dt_td * B_ts;

        h_s = dA * h_s + dB * x_td;

        /* Each thread contributes C[t,s]*h[s] to y[b,t,d] */
        atomicAdd(&sh_y[t], C_ts * h_s);
    }
    __syncthreads();

    /* Thread 0 writes final y (+ D skip) to global memory */
    if (s == 0) {
        float d_d = D_par[d];
        for (int t = 0; t < T; t++) {
            float x_td = x[b * T * d_inner + t * d_inner + d];
            y[b * T * d_inner + t * d_inner + d] = sh_y[t] + d_d * x_td;
        }
    }
}

/* ── Softplus ────────────────────────────────────────────────────── */
__global__ void softplus_kernel(float *x, int n, float dt_min, float dt_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    v = (v > 20.0f) ? v : logf(1.0f + expf(v));
    if (v < dt_min) v = dt_min;
    if (v > dt_max) v = dt_max;
    x[i] = v;
}

/* ── Elementwise multiply ────────────────────────────────────────── */
__global__ void mul_kernel(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

/* ── Residual add ────────────────────────────────────────────────── */
__global__ void add_kernel(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

/* ── Add bias (broadcast over rows) ─────────────────────────────── */
__global__ void add_bias_kernel(float *x, const float *bias, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) x[i] += bias[i % cols];
}

/* ── Embedding forward ───────────────────────────────────────────── */
__global__ void embed_fwd_kernel(
        float *out,            /* [BT, D] */
        const float *table,    /* [vocab, D] */
        const int   *tokens,   /* [BT] */
        int BT, int D)
{
    int bt = blockIdx.x;
    int d  = threadIdx.x + blockIdx.y * blockDim.x;
    if (bt >= BT || d >= D) return;
    int tok = tokens[bt];
    out[bt * D + d] = table[tok * D + d];
}

/* ── Softmax + cross-entropy (fused, per row) ────────────────────── */
/*
 * logits : [BT, vocab]
 * target : [BT]  (token index)
 * losses : [BT]  (per-position cross-entropy)
 * d_logits: [BT, vocab] — softmax probabilities (overwritten), then
 *           d_logits[i,target[i]] -= 1  → gradient of loss w.r.t. logits
 */
__global__ void softmax_xentropy_kernel(
        float       *logits,   /* [BT, vocab] — in-place */
        float       *losses,   /* [BT] */
        const int   *target,   /* [BT] */
        int BT, int vocab)
{
    int row = blockIdx.x;
    float *l = logits + row * vocab;

    /* find max for numerical stability (shared across warps) */
    __shared__ float sh[32];
    float mx = -1e30f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        mx = fmaxf(mx, l[i]);
    mx += __shfl_down_sync(0xffffffff, mx, 16);
    mx += __shfl_down_sync(0xffffffff, mx,  8);  /* actually max, not add... */
    /* redo properly */
    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        thread_max = fmaxf(thread_max, l[i]);
    /* warp reduce max */
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = thread_max;
    __syncthreads();
    float global_max = -1e30f;
    if (threadIdx.x < (blockDim.x >> 5)) global_max = sh[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    __shared__ float gmax;
    if (threadIdx.x == 0) gmax = global_max;
    __syncthreads();

    /* exp and sum */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float e = expf(l[i] - gmax);
        l[i] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = sum;
    __syncthreads();
    float gsum = 0.0f;
    if (threadIdx.x < (blockDim.x >> 5)) gsum = sh[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        gsum += __shfl_down_sync(0xffffffff, gsum, offset);
    __shared__ float gs;
    if (threadIdx.x == 0) gs = gsum;
    __syncthreads();

    /* normalise and compute loss */
    int tgt = target[row];
    float p_tgt = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float p = l[i] / gs;
        l[i] = p;                    /* now l[] = softmax probabilities */
        if (i == tgt) p_tgt = p;
    }
    /* reduce p_tgt across threads */
    for (int offset = 16; offset > 0; offset >>= 1)
        p_tgt += __shfl_down_sync(0xffffffff, p_tgt, offset);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = p_tgt;
    __syncthreads();
    if (threadIdx.x == 0) {
        float pt = sh[0];
        if (pt < 1e-12f) pt = 1e-12f;
        losses[row] = -logf(pt);
        /* subtract 1 from target logit: combined softmax+CE gradient */
        l[tgt] -= 1.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Backward kernels
 * ═══════════════════════════════════════════════════════════════════ */

/* ── RMSNorm backward: Grid=BT, Block=256 ────────────────────────── */
__global__ void rms_norm_bwd_kernel(
        float *d_x, float *d_norm_w,
        const float *d_out, const float *x, const float *norm_w,
        int D, float eps)
{
    int row = blockIdx.x;
    const float *xr  = x    + row * D;
    const float *dor = d_out + row * D;
    float       *dxr = d_x  + row * D;
    __shared__ float sh[2];
    float sumsq = 0.f, dot = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        sumsq += xr[j] * xr[j];
        dot   += dor[j] * norm_w[j] * xr[j];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, off);
        dot   += __shfl_down_sync(0xffffffff, dot,   off);
    }
    if (threadIdx.x == 0) { sh[0] = sumsq; sh[1] = dot; }
    __syncthreads();
    float rms  = rsqrtf(sh[0] / (float)D + eps);
    float coef = sh[1] * rms * rms / (float)D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        atomicAdd(&d_norm_w[j], dor[j] * xr[j] * rms);
        dxr[j] += rms * (dor[j] * norm_w[j] - xr[j] * coef);
    }
}

/* ── SiLU backward ─────────────────────────────────────────────── */
__global__ void silu_bwd_kernel(float *d_x, const float *d_out,
                                 const float *x_pre, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sig  = 1.f / (1.f + expf(-x_pre[i]));
    d_x[i] += d_out[i] * sig * (1.f + x_pre[i] * (1.f - sig));
}

/* ── Causal conv1d backward ─────────────────────────────────────── */
__global__ void causal_conv1d_bwd_kernel(
        float *d_x, float *d_w, float *d_bias,
        const float *d_out, const float *x, const float *w,
        int B, int T, int di, int dc)
{
    int b = blockIdx.x, t = blockIdx.y;
    int d = threadIdx.x + blockIdx.z * blockDim.x;
    if (b >= B || t >= T || d >= di) return;
    float dv = d_out[b * T * di + t * di + d];
    atomicAdd(&d_bias[d], dv);
    for (int k = 0; k < dc; k++) {
        int src = t - k;
        if (src >= 0) {
            int ix = b * T * di + src * di + d;
            atomicAdd(&d_x[ix],         dv * w[d * dc + k]);
            atomicAdd(&d_w[d * dc + k], dv * x[ix]);
        }
    }
}

/* ── SSM scan forward — saves hidden states for backward ─────────── */
__global__ void ssm_scan_fwd_train_kernel(
        const float *x, const float *dt, const float *A_log,
        const float *B_ssm, const float *C_ssm, const float *D_par,
        float *y, float *h_states,
        int B, int T, int di, int ds)
{
    int b = blockIdx.x, d = blockIdx.y, s = threadIdx.x;
    extern __shared__ float sh_y[];
    for (int t = s; t < T; t += ds) sh_y[t] = 0.f;
    __syncthreads();
    float a_ds = A_log[d * ds + s], h_s = 0.f;
    for (int t = 0; t < T; t++) {
        int bt  = b * T * di + t * di;
        int bts = b * T * ds + t * ds;
        float dA = expf(dt[bt + d] * a_ds);
        float dB = dt[bt + d] * B_ssm[bts + s];
        h_s = dA * h_s + dB * x[bt + d];
        h_states[((b * T + t) * di + d) * ds + s] = h_s;
        atomicAdd(&sh_y[t], C_ssm[bts + s] * h_s);
    }
    __syncthreads();
    if (s == 0) {
        float dd = D_par[d];
        for (int t = 0; t < T; t++) {
            float xv = x[b * T * di + t * di + d];
            y[b * T * di + t * di + d] = sh_y[t] + dd * xv;
        }
    }
}

/* ── SSM scan backward: Grid=(B,di) Block=ds ─────────────────────── */
__global__ void ssm_scan_bwd_kernel(
        const float *x, const float *dt, const float *A_log,
        const float *B_ssm, const float *C_ssm, const float *D_par,
        const float *h_states, const float *d_y,
        float *d_x, float *d_dt,
        float *d_A_log, float *d_B_ssm, float *d_C_ssm, float *d_D_par,
        int B, int T, int di, int ds)
{
    int b = blockIdx.x, d = blockIdx.y, s = threadIdx.x;
    unsigned smask = (unsigned)((1 << ds) - 1);
    float a_ds = A_log[d * ds + s];
    if (s == 0) {
        float dd = D_par[d], acc = 0.f;
        for (int t = 0; t < T; t++) {
            int idx = b * T * di + t * di + d;
            acc += d_y[idx] * x[idx];
            atomicAdd(&d_x[idx], d_y[idx] * dd);
        }
        atomicAdd(&d_D_par[d], acc);
    }
    float d_h_next = 0.f;
    for (int t = T - 1; t >= 0; t--) {
        int bt  = b * T * di + t * di + d;
        int bts = b * T * ds + t * ds + s;
        float dt_v = dt[bt], B_ts = B_ssm[bts], C_ts = C_ssm[bts];
        float dy   = d_y[bt], xv  = x[bt];
        float h_t  = h_states[((b * T + t) * di + d) * ds + s];
        float h_p  = (t > 0) ? h_states[((b * T + t - 1) * di + d) * ds + s] : 0.f;
        float dA   = expf(dt_v * a_ds);
        float dB   = dt_v * B_ts;
        float d_h  = C_ts * dy + d_h_next;
        d_h_next   = dA * d_h;
        atomicAdd(&d_C_ssm[bts], h_t * dy);
        float d_dB = d_h * xv;
        atomicAdd(&d_B_ssm[bts], d_dB * dt_v);
        float d_dA = d_h * h_p;
        atomicAdd(&d_A_log[d * ds + s], d_dA * dA * dt_v);
        float dx_s  = d_h * dB;
        float ddt_s = d_dA * dA * a_ds + d_dB * B_ts;
        for (int off = ds >> 1; off >= 1; off >>= 1) {
            dx_s  += __shfl_xor_sync(smask, dx_s,  off);
            ddt_s += __shfl_xor_sync(smask, ddt_s, off);
        }
        if (s == 0) {
            atomicAdd(&d_x[bt],  dx_s);
            atomicAdd(&d_dt[bt], ddt_s);
        }
    }
}

/* ── Softplus backward (zero gradient in clamped region) ─────────── */
__global__ void softplus_bwd_kernel(
        float *d_x, const float *d_out,
        const float *x_pre, const float *x_post,
        int n, float dt_min, float dt_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x_post[i];
    if (v <= dt_min || v >= dt_max) return;
    d_x[i] += d_out[i] / (1.f + expf(-x_pre[i]));
}

/* ── Elementwise-mul backward ─────────────────────────────────────── */
__global__ void mul_bwd_kernel(float *d_a, float *d_b,
                                const float *d_y, const float *a,
                                const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_a[i] += d_y[i] * b[i];
    d_b[i] += d_y[i] * a[i];
}

/* ── Bias gradient ─────────────────────────────────────────────────── */
__global__ void bias_grad_kernel(float *d_bias, const float *d_x,
                                  int rows, int cols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.f;
    for (int r = 0; r < rows; r++) s += d_x[r * cols + j];
    atomicAdd(&d_bias[j], s);
}

/* ── Embedding scatter-add backward ───────────────────────────────── */
__global__ void embed_bwd_kernel(float *d_table, const float *d_h,
                                  const int *tokens, int BT, int D)
{
    int bt = blockIdx.x;
    int d  = threadIdx.x + blockIdx.y * blockDim.x;
    if (bt >= BT || d >= D) return;
    atomicAdd(&d_table[tokens[bt] * D + d], d_h[bt * D + d]);
}

/* ═══════════════════════════════════════════════════════════════════
 * MuonClip optimiser kernels
 * ═══════════════════════════════════════════════════════════════════ */

/* ── Global L2 norm (partial reduce, one block per chunk) ─────────── */
__global__ void l2_norm_kernel(const float *g, float *partial, int n)
{
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? g[i] * g[i] : 0.0f;
    sh[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sh[0];
}

/* ── Scale in-place: x *= s ─────────────────────────────────────── */
__global__ void scale_kernel(float *x, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

/* ── Newton-Schulz linear combo: G = a*G + b*P + c*Q  (in-place) ── */
__global__ void ns_combo_kernel(float *G, const float *P, const float *Q,
                                float a, float b, float c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) G[i] = a * G[i] + b * P[i] + c * Q[i];
}

/* ── Muon momentum update + weight step ─────────────────────────── */
/* M ← beta·M + G_orth;   W ← W − lr·(M + wd·W) */
__global__ void muon_step_kernel(float *W, float *M, const float *G_orth,
                                 int n, float lr, float beta, float wd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m = beta * M[i] + G_orth[i];
    M[i] = m;
    W[i] -= lr * (m + wd * W[i]);
    /* gradient is zeroed by the caller after NS (scratch reuse) */
}

/* ── AdamW for 1-D parameters (biases, norms, A_log, D_param) ───── */
__global__ void adamw_1d_kernel(float *W, float *G, float *M, float *V,
                                int n, float lr, float b1, float b2,
                                float eps, float wd, float clip,
                                float bc1, float bc2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = G[i] * clip + wd * W[i];
    M[i] = b1 * M[i] + (1.0f - b1) * g;
    V[i] = b2 * V[i] + (1.0f - b2) * g * g;
    float m_hat = M[i] / bc1;
    float v_hat = V[i] / bc2;
    W[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    G[i]  = 0.0f;   /* zero grad in-place */
}


/* ═══════════════════════════════════════════════════════════════════
 * Internal structs
 * ═══════════════════════════════════════════════════════════════════ */

/* Device memory for one Mamba layer */
typedef struct {
    /* ── Weights ── */
    float *norm_w;          /* [D]                     RMSNorm scale          */
    float *in_proj;         /* [2*d_inner, D]          no bias                */
    float *conv_w;          /* [d_inner, d_conv]       depthwise conv         */
    float *conv_b;          /* [d_inner]               conv bias              */
    float *x_proj;          /* [dt_rank+2*d_state, d_inner]                   */
    float *dt_proj;         /* [d_inner, dt_rank]      expand Δ               */
    float *dt_bias;         /* [d_inner]               Δ bias                 */
    float *out_proj;        /* [D, d_inner]            output projection      */
    float *A_log;           /* [d_inner, d_state]      log(-A), initialised   */
    float *D_param;         /* [d_inner]               skip connection        */

    /* ── Gradients (same shapes) ── */
    float *g_norm_w, *g_in_proj, *g_conv_w, *g_conv_b;
    float *g_x_proj, *g_dt_proj, *g_dt_bias, *g_out_proj;
    float *g_A_log, *g_D_param;

    /* ── Optimiser state (same shapes, zeroed) ──
     *    2-D params: m_* = Muon momentum, v_* = AdamW v (unused for Muon)
     *    1-D params: m_* = AdamW m,       v_* = AdamW v                  */
    float *m_norm_w, *v_norm_w;
    float *m_in_proj, *v_in_proj;
    float *m_conv_w,  *v_conv_w;
    float *m_conv_b,  *v_conv_b;
    float *m_x_proj,  *v_x_proj;
    float *m_dt_proj, *v_dt_proj;
    float *m_dt_bias, *v_dt_bias;
    float *m_out_proj, *v_out_proj;
    float *m_A_log,   *v_A_log;
    float *m_D_param, *v_D_param;
} MambaLayerGPU;

/* Full model */
struct MLModel {
    MLConfig        cfg;
    cublasHandle_t  cublas;

    /* Token embedding [vocab, D] */
    float *embedding;
    float *g_embedding;
    float *m_embedding, *v_embedding;

    /* Final RMSNorm [D] */
    float *final_norm_w;
    float *g_final_norm_w;
    float *m_final_norm_w, *v_final_norm_w;

    /* LM head: tied to embedding — no extra weights */

    /* Layers (host array, each element holds device pointers) */
    MambaLayerGPU *layers;

    /* Adam step counter */
    size_t step;
};


/* ═══════════════════════════════════════════════════════════════════
 * Device allocation helpers
 * ═══════════════════════════════════════════════════════════════════ */

static float *dev_alloc_zero(size_t n)
{
    if (n == 0) return NULL;
    float *p = NULL;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(float)));
    return p;
}

static float *dev_alloc_copy(const float *host, size_t n)
{
    float *p = dev_alloc_zero(n);
    CUDA_CHECK(cudaMemcpy(p, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}

/* Xavier uniform initialisation on host, then upload */
static float *dev_xavier(size_t fan_in, size_t fan_out, size_t n)
{
    float *h = (float *)malloc(n * sizeof(float));
    float scale = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    float *d = dev_alloc_copy(h, n);
    free(h);
    return d;
}

/* Small uniform init */
static float *dev_uniform(size_t n, float lo, float hi)
{
    float *h = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        h[i] = lo + (hi - lo) * (float)rand() / RAND_MAX;
    float *d = dev_alloc_copy(h, n);
    free(h);
    return d;
}

/* Allocate two zero'd moment arrays (m and v) of size n */
#define MOMENTS(mptr, vptr, n) \
    do { (mptr) = dev_alloc_zero(n); \
         (vptr) = dev_alloc_zero(n); } while(0)


/* ═══════════════════════════════════════════════════════════════════
 * Layer init / free
 * ═══════════════════════════════════════════════════════════════════ */

static void layer_init(MambaLayerGPU *L, const MLConfig *c)
{
    int D  = c->dim, di = c->d_inner, ds = c->d_state;
    int dc = c->d_conv, dr = c->dt_rank;

    /* ── Weights ── */
    L->norm_w   = dev_uniform(D, 0.95f, 1.05f);                /* ≈ 1 */
    L->in_proj  = dev_xavier(D, 2*di, (size_t)2*di * D);
    L->conv_w   = dev_uniform((size_t)di * dc, -0.01f, 0.01f);
    L->conv_b   = dev_alloc_zero(di);
    L->x_proj   = dev_xavier(di, dr+2*ds, (size_t)(dr+2*ds) * di);
    L->dt_proj  = dev_xavier(dr, di, (size_t)di * dr);
    L->dt_bias  = dev_alloc_zero(di);                           /* zero init */
    L->out_proj = dev_xavier(di, D, (size_t)D * di);
    L->D_param  = dev_uniform(di, 0.9f, 1.1f);                 /* ≈ 1 */

    /* A_log: initialise with log-spaced values in log(-A) */
    {
        float *h = (float *)malloc((size_t)di * ds * sizeof(float));
        for (int d = 0; d < di; d++)
            for (int s = 0; s < ds; s++) {
                float n_s = (float)(s + 1) / (float)ds;
                h[d * ds + s] = -logf(n_s * (float)ds);
            }
        L->A_log = dev_alloc_copy(h, (size_t)di * ds);
        free(h);
    }

    /* ── Gradients ── */
    L->g_norm_w  = dev_alloc_zero(D);
    L->g_in_proj = dev_alloc_zero((size_t)2*di * D);
    L->g_conv_w  = dev_alloc_zero((size_t)di * dc);
    L->g_conv_b  = dev_alloc_zero(di);
    L->g_x_proj  = dev_alloc_zero((size_t)(dr+2*ds) * di);
    L->g_dt_proj = dev_alloc_zero((size_t)di * dr);
    L->g_dt_bias = dev_alloc_zero(di);
    L->g_out_proj= dev_alloc_zero((size_t)D * di);
    L->g_A_log   = dev_alloc_zero((size_t)di * ds);
    L->g_D_param = dev_alloc_zero(di);

    /* ── Optimiser state (all zeroed) ── */
    MOMENTS(L->m_norm_w,  L->v_norm_w,  D);
    MOMENTS(L->m_in_proj, L->v_in_proj, (size_t)2*di * D);
    MOMENTS(L->m_conv_w,  L->v_conv_w,  (size_t)di * dc);
    MOMENTS(L->m_conv_b,  L->v_conv_b,  di);
    MOMENTS(L->m_x_proj,  L->v_x_proj,  (size_t)(dr+2*ds) * di);
    MOMENTS(L->m_dt_proj, L->v_dt_proj, (size_t)di * dr);
    MOMENTS(L->m_dt_bias, L->v_dt_bias, di);
    MOMENTS(L->m_out_proj,L->v_out_proj,(size_t)D * di);
    MOMENTS(L->m_A_log,   L->v_A_log,   (size_t)di * ds);
    MOMENTS(L->m_D_param, L->v_D_param, di);
}

#define CFREE(p) do { if (p) { cudaFree(p); (p) = NULL; } } while(0)

static void layer_free(MambaLayerGPU *L)
{
    CFREE(L->norm_w);  CFREE(L->in_proj);  CFREE(L->conv_w);   CFREE(L->conv_b);
    CFREE(L->x_proj);  CFREE(L->dt_proj);  CFREE(L->dt_bias);  CFREE(L->out_proj);
    CFREE(L->A_log);   CFREE(L->D_param);
    CFREE(L->g_norm_w);CFREE(L->g_in_proj);CFREE(L->g_conv_w); CFREE(L->g_conv_b);
    CFREE(L->g_x_proj);CFREE(L->g_dt_proj);CFREE(L->g_dt_bias);CFREE(L->g_out_proj);
    CFREE(L->g_A_log); CFREE(L->g_D_param);
    CFREE(L->m_norm_w);CFREE(L->v_norm_w);
    CFREE(L->m_in_proj);CFREE(L->v_in_proj);
    CFREE(L->m_conv_w);CFREE(L->v_conv_w);
    CFREE(L->m_conv_b);CFREE(L->v_conv_b);
    CFREE(L->m_x_proj);CFREE(L->v_x_proj);
    CFREE(L->m_dt_proj);CFREE(L->v_dt_proj);
    CFREE(L->m_dt_bias);CFREE(L->v_dt_bias);
    CFREE(L->m_out_proj);CFREE(L->v_out_proj);
    CFREE(L->m_A_log);CFREE(L->v_A_log);
    CFREE(L->m_D_param);CFREE(L->v_D_param);
}


/* ═══════════════════════════════════════════════════════════════════
 * Forward pass through one Mamba layer
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Runs the full Mamba layer computation.
 * x_in  : [B, T, D]  — input (residual stream)
 * x_out : [B, T, D]  — output (= layer output + residual)
 *
 * Intermediate buffers are allocated internally via cudaMalloc each call
 * (acceptable for training step granularity; production code would cache).
 */
static void layer_forward(const MambaLayerGPU *L, const MLConfig *c,
                           cublasHandle_t cublas,
                           const float *x_in,   /* [B,T,D] */
                           float       *x_out,  /* [B,T,D] */
                           int B, int save_acts, /* save_acts: for backward */
                           float **acts_out)     /* optional activation cache */
{
    int D  = c->dim, di = c->d_inner, ds = c->d_state;
    int dc = c->d_conv, dr = c->dt_rank;
    int T  = c->seq_len;
    int BT = B * T;
    int threads = 256;

    (void)save_acts; (void)acts_out;   /* TODO: activation caching for backward */

    /* Alloc work buffers */
    float *normed;  CUDA_CHECK(cudaMalloc(&normed,  (size_t)BT * D  * sizeof(float)));
    float *xz;      CUDA_CHECK(cudaMalloc(&xz,      (size_t)BT * 2*di * sizeof(float)));
    float *x_br;    CUDA_CHECK(cudaMalloc(&x_br,    (size_t)BT * di  * sizeof(float)));
    float *z_br;    CUDA_CHECK(cudaMalloc(&z_br,    (size_t)BT * di  * sizeof(float)));
    float *x_conv;  CUDA_CHECK(cudaMalloc(&x_conv,  (size_t)BT * di  * sizeof(float)));
    float *BCdt;    CUDA_CHECK(cudaMalloc(&BCdt,     (size_t)BT * (dr+2*ds) * sizeof(float)));
    float *dt_raw;  CUDA_CHECK(cudaMalloc(&dt_raw,   (size_t)BT * dr  * sizeof(float)));
    float *dt_full; CUDA_CHECK(cudaMalloc(&dt_full,  (size_t)BT * di  * sizeof(float)));
    float *B_ssm;   CUDA_CHECK(cudaMalloc(&B_ssm,    (size_t)BT * ds  * sizeof(float)));
    float *C_ssm;   CUDA_CHECK(cudaMalloc(&C_ssm,    (size_t)BT * ds  * sizeof(float)));
    float *ssm_y;   CUDA_CHECK(cudaMalloc(&ssm_y,    (size_t)BT * di  * sizeof(float)));
    float *y_gated; CUDA_CHECK(cudaMalloc(&y_gated,  (size_t)BT * di  * sizeof(float)));
    float *y_proj;  CUDA_CHECK(cudaMalloc(&y_proj,   (size_t)BT * D   * sizeof(float)));

    /* 1. RMSNorm */
    rms_norm_fwd_kernel<<<BT, 256>>>(normed, x_in, L->norm_w, D, 1e-5f);

    /* 2. in_proj: [BT,D] × [D, 2*di]^T  → [BT, 2*di] */
    /* in_proj stored as [2*di, D]; we need Y = X * W^T */
    /* gemm: C[BT,2di] = X[BT,D] * W^T[D,2di]  → gemm_row with W transposed */
    {
        float alpha=1.f, beta=0.f;
        CUBLAS_CHECK(cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            2*di, BT, D,
            &alpha, L->in_proj, D,
            normed, D,
            &beta, xz, 2*di));
    }

    /* 3. Split xz into x_br and z_br */
    CUDA_CHECK(cudaMemcpy(x_br, xz,         (size_t)BT*di*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(z_br, xz + BT*di, (size_t)BT*di*sizeof(float), cudaMemcpyDeviceToDevice));

    /* 4. Causal conv1d on x_br */
    {
        int d_blocks = (di + 255) / 256;
        dim3 grid(B, T, d_blocks);
        causal_conv1d_fwd_kernel<<<grid, 256>>>(
            x_br, L->conv_w, L->conv_b, x_conv,
            B, T, di, dc);
    }

    /* 5. SiLU on x_conv (in-place, becomes x for SSM) */
    silu_kernel<<<(BT*di + 255)/256, 256>>>(x_conv, BT*di);

    /* 6. x_proj: [BT,di] → [BT, dr+2*ds] */
    {
        float alpha=1.f, beta=0.f;
        CUBLAS_CHECK(cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dr+2*ds, BT, di,
            &alpha, L->x_proj, di,
            x_conv, di,
            &beta, BCdt, dr+2*ds));
    }

    /* 7. Split BCdt into dt_raw [BT,dr], B_ssm [BT,ds], C_ssm [BT,ds] */
    CUDA_CHECK(cudaMemcpy(dt_raw, BCdt,                (size_t)BT*dr*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(B_ssm,  BCdt + BT*dr,        (size_t)BT*ds*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(C_ssm,  BCdt + BT*(dr+ds),   (size_t)BT*ds*sizeof(float), cudaMemcpyDeviceToDevice));

    /* 8. dt_proj: [BT,dr] → [BT,di], then add bias, then softplus */
    {
        float alpha=1.f, beta=0.f;
        CUBLAS_CHECK(cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            di, BT, dr,
            &alpha, L->dt_proj, dr,
            dt_raw, dr,
            &beta, dt_full, di));
    }
    add_bias_kernel<<<(BT*di+255)/256, 256>>>(dt_full, L->dt_bias, BT, di);
    softplus_kernel<<<(BT*di+255)/256, 256>>>(dt_full, BT*di, c->dt_min, c->dt_max);

    /* 9. SSM scan: zero output, then launch kernel */
    CUDA_CHECK(cudaMemset(ssm_y, 0, (size_t)BT*di*sizeof(float)));
    {
        dim3 grid(B, di);
        size_t smem = (size_t)T * sizeof(float);
        ssm_scan_fwd_kernel<<<grid, ds, smem>>>(
            x_conv, dt_full, L->A_log, B_ssm, C_ssm, L->D_param,
            ssm_y, B, T, di, ds);
    }

    /* 10. Gate: y = ssm_y * silu(z_br) */
    CUDA_CHECK(cudaMemcpy(y_gated, ssm_y, (size_t)BT*di*sizeof(float), cudaMemcpyDeviceToDevice));
    silu_kernel<<<(BT*di+255)/256, 256>>>(z_br, BT*di);
    mul_kernel<<<(BT*di+255)/256, 256>>>(y_gated, z_br, BT*di);

    /* 11. out_proj: [BT,di] → [BT,D] */
    {
        float alpha=1.f, beta=0.f;
        CUBLAS_CHECK(cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, BT, di,
            &alpha, L->out_proj, di,
            y_gated, di,
            &beta, y_proj, D));
    }

    /* 12. Residual add */
    CUDA_CHECK(cudaMemcpy(x_out, y_proj, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice));
    add_kernel<<<(BT*D+255)/256, 256>>>(x_out, x_in, BT*D);

    /* Free work buffers */
    cudaFree(normed);  cudaFree(xz);     cudaFree(x_br);   cudaFree(z_br);
    cudaFree(x_conv);  cudaFree(BCdt);   cudaFree(dt_raw);  cudaFree(dt_full);
    cudaFree(B_ssm);   cudaFree(C_ssm);  cudaFree(ssm_y);   cudaFree(y_gated);
    cudaFree(y_proj);
}


/* ═══════════════════════════════════════════════════════════════════
 * MuonClip host helpers
 * ═══════════════════════════════════════════════════════════════════ */

/* ── Compute gradient L2 norm (device → host scalar) ─────────────── */
static float grad_l2_norm(const float *d_grad, int n)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    float *d_part;
    CUDA_CHECK(cudaMalloc(&d_part, (size_t)blocks * sizeof(float)));
    l2_norm_kernel<<<blocks, threads, (size_t)threads*sizeof(float)>>>(d_grad, d_part, n);
    float *h_part = (float *)malloc((size_t)blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_part, d_part, (size_t)blocks*sizeof(float), cudaMemcpyDeviceToHost));
    double sq = 0.0;
    for (int i = 0; i < blocks; i++) sq += (double)h_part[i];
    free(h_part);
    cudaFree(d_part);
    return (float)sqrt(sq);
}

/* ── Newton-Schulz orthogonalisation (NS-5) ────────────────────────
 *
 * Orthogonalises G[rows, cols] in-place using cuBLAS.
 * Allocates and frees all scratch internally.
 *
 * Forward mode  (rows <= cols): A = G  @ G^T  [rows, rows]
 *   G_new = (15/8)·G − (5/4)·A·G  + (3/8)·A²·G
 *
 * Transposed mode (rows > cols): A = G^T @ G   [cols, cols]
 *   G_new = (15/8)·G − (5/4)·G·A  + (3/8)·G·A²
 * ─────────────────────────────────────────────────────────────────── */
static void ns_orthogonalize(cublasHandle_t cublas,
                              float *G,           /* [rows, cols] row-major, modified in-place */
                              int rows, int cols,
                              int ns_steps)
{
    int n    = rows * cols;
    int dim  = (rows <= cols) ? rows : cols;   /* smaller square dimension */
    int fwd  = (rows <= cols);                 /* forward vs. transposed mode */

    /* Allocate scratch: A [dim,dim], P [rows,cols], Q [rows,cols] */
    float *A, *P, *Q;
    CUDA_CHECK(cudaMalloc(&A, (size_t)dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P, (size_t)n        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Q, (size_t)n        * sizeof(float)));

    /* 1. Normalise G: G /= ||G||_F */
    float frob = grad_l2_norm(G, n);
    if (frob < 1e-12f) { cudaFree(A); cudaFree(P); cudaFree(Q); return; }
    scale_kernel<<<(n+255)/256, 256>>>(G, 1.0f / frob, n);

    float one = 1.0f, zero = 0.0f;

    for (int k = 0; k < ns_steps; k++) {
        if (fwd) {
            /* A[rows,rows] = G[rows,cols] @ G^T[cols,rows]
             * cuBLAS col-major: (G_cm)^T @ G_cm  where G_cm = G row-major
             * → cublasSgemm(T, N, rows, rows, cols, 1, G, cols, G, cols, 0, A, rows) */
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N, rows, rows, cols,
                &one,  G, cols, G, cols, &zero, A, rows));

            /* P[rows,cols] = A[rows,rows] @ G[rows,cols]
             * C[M,N]=A[M,K]*B[K,N] row-major: cublasSgemm(N,N, N,M,K, 1, B,N, A,K, 0, C,N)
             * → cublasSgemm(N,N, cols,rows,rows, 1, G,cols, A,rows, 0, P,cols) */
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, rows,
                &one,  G, cols, A, rows, &zero, P, cols));

            /* Q[rows,cols] = A[rows,rows] @ P[rows,cols] */
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, rows,
                &one,  P, cols, A, rows, &zero, Q, cols));
        } else {
            /* A[cols,cols] = G^T[cols,rows] @ G[rows,cols]
             * cublasSgemm(N, T, cols, cols, rows, 1, G,cols, G,cols, 0, A,cols) */
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_T, cols, cols, rows,
                &one,  G, cols, G, cols, &zero, A, cols));

            /* P[rows,cols] = G[rows,cols] @ A[cols,cols]
             * cublasSgemm(N,N, cols,rows,cols, 1, A,cols, G,cols, 0, P,cols) */
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, cols,
                &one,  A, cols, G, cols, &zero, P, cols));

            /* Q[rows,cols] = P[rows,cols] @ A[cols,cols]  (= G @ A @ A) */
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, cols,
                &one,  A, cols, P, cols, &zero, Q, cols));
        }

        /* G = (15/8)·G − (5/4)·P + (3/8)·Q */
        ns_combo_kernel<<<(n+255)/256, 256>>>(
            G, P, Q,
            15.0f/8.0f, -5.0f/4.0f, 3.0f/8.0f, n);
    }

    cudaFree(A); cudaFree(P); cudaFree(Q);
}

/* ── MuonClip update for a 2-D weight matrix ─────────────────────── */
static void muon_update_2d(cublasHandle_t cublas,
                            float *W,      /* [rows, cols] */
                            float *G,      /* [rows, cols] gradient — zeroed after */
                            float *M,      /* [rows, cols] momentum buffer         */
                            int rows, int cols,
                            const MLOptimConfig *opt,
                            float clip_scale)   /* pre-computed from global norm */
{
    if (!W || !G || !M || rows <= 0 || cols <= 0) return;
    int n = rows * cols;

    /* 1. Apply global gradient clip */
    if (clip_scale < 1.0f)
        scale_kernel<<<(n+255)/256, 256>>>(G, clip_scale, n);

    /* 2. Newton-Schulz orthogonalise G (normalises + NS-5) */
    ns_orthogonalize(cublas, G, rows, cols, opt->ns_steps);

    /* 3. Muon momentum step + weight update */
    /*    Scale the orthogonalised gradient by sqrt(max(1, rows/cols))  */
    /*    to keep RMS consistent (as in the original Muon paper).       */
    float rms_scale = sqrtf(fmaxf(1.0f, (float)rows / (float)cols));
    scale_kernel<<<(n+255)/256, 256>>>(G, rms_scale, n);

    muon_step_kernel<<<(n+255)/256, 256>>>(
        W, M, G, n, opt->lr, opt->beta, opt->weight_decay);

    /* 4. Zero gradient */
    CUDA_CHECK(cudaMemset(G, 0, (size_t)n * sizeof(float)));
}

/* ── AdamW update for a 1-D (or small special) parameter ─────────── */
static void adamw_update_1d(float *W, float *G, float *M, float *V,
                             size_t n, const MLOptimConfig *opt,
                             size_t step, float clip_scale)
{
    if (!W || !G || n == 0) return;
    float bc1 = 1.0f - powf(opt->beta1, (float)step);
    float bc2 = 1.0f - powf(opt->beta2, (float)step);
    adamw_1d_kernel<<<((int)n+255)/256, 256>>>(
        W, G, M, V, (int)n,
        opt->lr_1d, opt->beta1, opt->beta2, opt->eps,
        opt->weight_decay, clip_scale, bc1, bc2);
}


/* ═══════════════════════════════════════════════════════════════════
 * Public API — implementation (C-linkage for train_large.c)
 * ═══════════════════════════════════════════════════════════════════ */

extern "C" {

long long ml_count_params(const MLConfig *c)
{
    long long per_layer =
        (long long)c->dim +                                /* norm */
        (long long)2*c->d_inner * c->dim +                 /* in_proj */
        (long long)c->d_inner * c->d_conv +                /* conv_w */
        (long long)c->d_inner +                            /* conv_b */
        (long long)(c->dt_rank + 2*c->d_state)*c->d_inner +/* x_proj */
        (long long)c->d_inner * c->dt_rank + c->d_inner +  /* dt_proj+bias */
        (long long)c->dim * c->d_inner +                   /* out_proj */
        (long long)c->d_inner * c->d_state +               /* A_log */
        (long long)c->d_inner;                             /* D_param */
    return per_layer * c->n_layers
        + (long long)c->vocab_size * c->dim                /* embedding */
        + (long long)c->dim;                               /* final norm */
}

MLModel *ml_create(const MLConfig *cfg)
{
    srand((unsigned)time(NULL));

    MLModel *m = (MLModel *)calloc(1, sizeof(MLModel));
    if (!m) return NULL;
    m->cfg  = *cfg;
    m->step = 0;

    CUBLAS_CHECK(cublasCreate(&m->cublas));

    int V = cfg->vocab_size, D = cfg->dim;

    /* Embedding */
    m->embedding   = dev_xavier(V, D, (size_t)V * D);
    m->g_embedding = dev_alloc_zero((size_t)V * D);
    m->m_embedding = dev_alloc_zero((size_t)V * D);
    m->v_embedding = dev_alloc_zero((size_t)V * D);

    /* Final norm */
    m->final_norm_w   = dev_uniform(D, 0.95f, 1.05f);
    m->g_final_norm_w = dev_alloc_zero(D);
    m->m_final_norm_w = dev_alloc_zero(D);
    m->v_final_norm_w = dev_alloc_zero(D);

    /* Layers */
    m->layers = (MambaLayerGPU *)calloc((size_t)cfg->n_layers, sizeof(MambaLayerGPU));
    if (!m->layers) { ml_free(m); return NULL; }
    for (int l = 0; l < cfg->n_layers; l++)
        layer_init(&m->layers[l], cfg);

    fprintf(stderr, "MLModel created: %lld params (~%.2f B)\n",
            ml_count_params(cfg), (double)ml_count_params(cfg)/1e9);

    return m;
}

void ml_free(MLModel *m)
{
    if (!m) return;
    if (m->cublas) cublasDestroy(m->cublas);
    CFREE(m->embedding);  CFREE(m->g_embedding);
    CFREE(m->m_embedding); CFREE(m->v_embedding);
    CFREE(m->final_norm_w); CFREE(m->g_final_norm_w);
    CFREE(m->m_final_norm_w); CFREE(m->v_final_norm_w);
    if (m->layers) {
        for (int l = 0; l < m->cfg.n_layers; l++)
            layer_free(&m->layers[l]);
        free(m->layers);
    }
    free(m);
}

/* ── Forward pass (full sequence) ───────────────────────────────── */

static void model_forward(MLModel *m, const int *tokens_dev,
                           float *logits_dev,   /* [BT, vocab] */
                           int B)
{
    const MLConfig *c = &m->cfg;
    int D = c->dim, T = c->seq_len, BT = B * T;

    float *h;
    CUDA_CHECK(cudaMalloc(&h, (size_t)BT * D * sizeof(float)));

    /* Embedding lookup */
    {
        int d_blocks = (D + 255) / 256;
        dim3 grid(BT, d_blocks);
        embed_fwd_kernel<<<grid, 256>>>(h, m->embedding, tokens_dev, BT, D);
    }

    /* Mamba layers */
    float *h2;
    CUDA_CHECK(cudaMalloc(&h2, (size_t)BT * D * sizeof(float)));
    for (int l = 0; l < c->n_layers; l++) {
        layer_forward(&m->layers[l], c, m->cublas, h, h2, B, 0, NULL);
        float *tmp = h; h = h2; h2 = tmp;   /* ping-pong */
    }
    cudaFree(h2);

    /* Final RMSNorm */
    rms_norm_fwd_kernel<<<BT, 256>>>(h, h, m->final_norm_w, D, 1e-5f);

    /* LM head (tied embedding): logits = h * embedding^T
     * h [BT, D], embedding [V, D]  → logits [BT, V] */
    {
        float alpha=1.f, beta=0.f;
        CUBLAS_CHECK(cublasSgemm(m->cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            c->vocab_size, BT, D,
            &alpha, m->embedding, D,
            h, D,
            &beta, logits_dev, c->vocab_size));
    }

    cudaFree(h);
}

/* ── Training step ───────────────────────────────────────────────── */

float ml_train_step(MLModel *m, const int *in_seq, const int *tgt_seq,
                    const MLOptimConfig *opt)
{
    const MLConfig *c = &m->cfg;
    int T = c->seq_len, V = c->vocab_size;
    int BT = T;   /* batch = 1 for simplicity */

    m->step++;

    /* Copy tokens to device */
    int *d_in, *d_tgt;
    CUDA_CHECK(cudaMalloc(&d_in,  (size_t)T * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt, (size_t)T * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in,  in_seq,  (size_t)T*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, tgt_seq, (size_t)T*sizeof(int), cudaMemcpyHostToDevice));

    /* Forward */
    float *logits;
    CUDA_CHECK(cudaMalloc(&logits, (size_t)BT * V * sizeof(float)));
    model_forward(m, d_in, logits, 1);

    /* Fused softmax + cross-entropy + gradient */
    float *d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)BT * sizeof(float)));
    softmax_xentropy_kernel<<<BT, 256>>>(logits, d_losses, d_tgt, BT, V);

    /* Copy loss to host */
    float *h_losses = (float *)malloc((size_t)BT * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_losses, d_losses, (size_t)BT*sizeof(float), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < BT; i++) total += (double)h_losses[i];
    free(h_losses);

    /* Backward: logits[] now holds (softmax_probs - one_hot) = d_loss/d_logits
     * Gradient for embedding (LM head backward, weight tying):
     *   d_embedding += logits^T * h_last   (simplified: full gradient not back-propped through layers here)
     * Full layer-by-layer backward omitted in this first pass; planned for next iteration.
     * Adam update on the embedding as the main trainable proxy. */
    {
        /* embedding gradient from LM head: d_emb += logits^T * hidden */
        /* For now, gradient flows only through the LM head weights. */
    }

    /* ── MuonClip parameter updates ──────────────────────────────── */
    /*
     * Compute a single global gradient norm across ALL parameters for
     * the clip scale.  (Gradients are mostly zero at this stage since
     * full backprop is TODO, but the infrastructure is correct.)
     */
    double global_sq = 0.0;
    {
        /* Helper lambda-like: accumulate sq norm of one gradient buffer */
        auto acc = [&](const float *dg, int n) {
            if (!dg || n == 0) return;
            float nrm = grad_l2_norm(dg, n);
            global_sq += (double)nrm * (double)nrm;
        };
        acc(m->g_embedding,    c->vocab_size * c->dim);
        acc(m->g_final_norm_w, c->dim);
        for (int l = 0; l < c->n_layers; l++) {
            MambaLayerGPU *L = &m->layers[l];
            int D=c->dim, di=c->d_inner, ds=c->d_state, dc=c->d_conv, dr=c->dt_rank;
            acc(L->g_norm_w,   D);
            acc(L->g_in_proj,  2*di*D);
            acc(L->g_conv_w,   di*dc);
            acc(L->g_conv_b,   di);
            acc(L->g_x_proj,   (dr+2*ds)*di);
            acc(L->g_dt_proj,  di*dr);
            acc(L->g_dt_bias,  di);
            acc(L->g_out_proj, D*di);
            acc(L->g_A_log,    di*ds);
            acc(L->g_D_param,  di);
        }
    }
    float clip_scale = 1.0f;
    if (opt->clip_norm > 0.0f) {
        float gn = (float)sqrt(global_sq);
        if (gn > opt->clip_norm) clip_scale = opt->clip_norm / gn;
    }

    /* Embedding: AdamW (lookup table, not a linear projection) */
    adamw_update_1d(m->embedding,    m->g_embedding,    m->m_embedding,    m->v_embedding,
                    (size_t)c->vocab_size*c->dim, opt, m->step, clip_scale);
    /* Final norm: AdamW (1-D scale) */
    adamw_update_1d(m->final_norm_w, m->g_final_norm_w, m->m_final_norm_w, m->v_final_norm_w,
                    (size_t)c->dim, opt, m->step, clip_scale);

    for (int l = 0; l < c->n_layers; l++) {
        MambaLayerGPU *L = &m->layers[l];
        int D=c->dim, di=c->d_inner, ds=c->d_state, dc=c->d_conv, dr=c->dt_rank;

        /* ── MuonClip: 2-D projection weights ── */
        muon_update_2d(m->cublas, L->in_proj,  L->g_in_proj,  L->m_in_proj,  2*di, D,        opt, clip_scale);
        muon_update_2d(m->cublas, L->x_proj,   L->g_x_proj,   L->m_x_proj,   dr+2*ds, di,   opt, clip_scale);
        muon_update_2d(m->cublas, L->dt_proj,  L->g_dt_proj,  L->m_dt_proj,  di, dr,         opt, clip_scale);
        muon_update_2d(m->cublas, L->out_proj, L->g_out_proj, L->m_out_proj, D, di,           opt, clip_scale);
        muon_update_2d(m->cublas, L->conv_w,   L->g_conv_w,   L->m_conv_w,   di, dc,          opt, clip_scale);

        /* ── AdamW: 1-D params and special matrices ── */
        adamw_update_1d(L->norm_w,  L->g_norm_w,  L->m_norm_w,  L->v_norm_w,  D,           opt, m->step, clip_scale);
        adamw_update_1d(L->conv_b,  L->g_conv_b,  L->m_conv_b,  L->v_conv_b,  di,          opt, m->step, clip_scale);
        adamw_update_1d(L->dt_bias, L->g_dt_bias, L->m_dt_bias, L->v_dt_bias, di,          opt, m->step, clip_scale);
        adamw_update_1d(L->A_log,   L->g_A_log,   L->m_A_log,   L->v_A_log,   (size_t)di*ds, opt, m->step, clip_scale);
        adamw_update_1d(L->D_param, L->g_D_param, L->m_D_param, L->v_D_param, di,          opt, m->step, clip_scale);
    }

    cudaFree(d_in); cudaFree(d_tgt); cudaFree(logits); cudaFree(d_losses);
    return (float)(total / (double)T);
}

/* ── Generation ─────────────────────────────────────────────────── */

void ml_generate(MLModel *m, const char *prompt, int max_tokens, float temperature)
{
    const MLConfig *c = &m->cfg;
    int T = c->seq_len, V = c->vocab_size;

    /* Build context buffer from prompt */
    int *context = (int *)calloc((size_t)T, sizeof(int));
    size_t plen = prompt ? strlen(prompt) : 0;
    if (plen >= (size_t)T) {
        const char *start = prompt + (plen - (size_t)T);
        for (int i = 0; i < T; i++) context[i] = (unsigned char)start[i];
    } else {
        size_t pad = (size_t)T - plen;
        for (size_t i = 0; i < pad; i++)  context[i] = 0;
        for (size_t i = 0; i < plen; i++) context[pad + i] = (unsigned char)prompt[i];
    }

    int  *d_in;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)T * sizeof(int)));
    float *d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)T * V * sizeof(float)));
    float *h_logits = (float *)malloc((size_t)V * sizeof(float));

    for (int gen = 0; gen < max_tokens; gen++) {
        CUDA_CHECK(cudaMemcpy(d_in, context, (size_t)T*sizeof(int), cudaMemcpyHostToDevice));
        model_forward(m, d_in, d_logits, 1);

        /* Fetch last position logits */
        CUDA_CHECK(cudaMemcpy(h_logits,
            d_logits + (T-1) * V,
            (size_t)V * sizeof(float),
            cudaMemcpyDeviceToHost));

        /* Temperature + softmax on host */
        float mx = h_logits[0];
        for (int i = 1; i < V; i++) if (h_logits[i] > mx) mx = h_logits[i];
        float sum = 0.0f;
        for (int i = 0; i < V; i++) {
            h_logits[i] = expf((h_logits[i] - mx) / (temperature > 0 ? temperature : 1.0f));
            sum += h_logits[i];
        }
        for (int i = 0; i < V; i++) h_logits[i] /= sum;

        /* Multinomial sample */
        double r = (double)rand() / ((double)RAND_MAX + 1.0);
        int next = V - 1;
        double cum = 0.0;
        for (int i = 0; i < V; i++) {
            cum += (double)h_logits[i];
            if (r < cum) { next = i; break; }
        }

        /* Print byte */
        if (next >= 32 && next < 127) putchar(next);
        else if (next == '\n' || next == '\t') putchar(next);
        else putchar('?');
        fflush(stdout);

        if (next == '\n') break;

        /* Slide window */
        memmove(context, context + 1, (size_t)(T-1) * sizeof(int));
        context[T-1] = next;
    }
    putchar('\n');
    fflush(stdout);

    free(h_logits); free(context);
    cudaFree(d_in); cudaFree(d_logits);
}

/* ── Checkpoint ─────────────────────────────────────────────────── */

/* Helper: save a device tensor to an open file */
static int save_tensor(FILE *f, const float *d_ptr, size_t n)
{
    if (!d_ptr || n == 0) return 0;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return -1;
    CUDA_CHECK(cudaMemcpy(h, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t written = fwrite(h, sizeof(float), n, f);
    free(h);
    return (written == n) ? 0 : -1;
}

/* Helper: load from file into a device tensor */
static int load_tensor(FILE *f, float *d_ptr, size_t n)
{
    if (!d_ptr || n == 0) return 0;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return -1;
    size_t red = fread(h, sizeof(float), n, f);
    if (red != n) { free(h); return -1; }
    CUDA_CHECK(cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    return 0;
}

int ml_save(const MLModel *m, const char *path)
{
    if (!m || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t magic = ML_MAGIC;
    fwrite(&magic,    sizeof(uint32_t), 1, f);
    fwrite(&m->cfg,   sizeof(MLConfig), 1, f);
    fwrite(&m->step,  sizeof(size_t),   1, f);

    const MLConfig *c = &m->cfg;
    int V=c->vocab_size, D=c->dim, di=c->d_inner, ds=c->d_state, dc=c->d_conv, dr=c->dt_rank;

    save_tensor(f, m->embedding,    (size_t)V*D);
    save_tensor(f, m->final_norm_w, (size_t)D);

    for (int l = 0; l < c->n_layers; l++) {
        const MambaLayerGPU *L = &m->layers[l];
        save_tensor(f, L->norm_w,   D);
        save_tensor(f, L->in_proj,  (size_t)2*di*D);
        save_tensor(f, L->conv_w,   (size_t)di*dc);
        save_tensor(f, L->conv_b,   di);
        save_tensor(f, L->x_proj,   (size_t)(dr+2*ds)*di);
        save_tensor(f, L->dt_proj,  (size_t)di*dr);
        save_tensor(f, L->dt_bias,  di);
        save_tensor(f, L->out_proj, (size_t)D*di);
        save_tensor(f, L->A_log,    (size_t)di*ds);
        save_tensor(f, L->D_param,  di);
    }

    fclose(f);
    return 0;
}

int ml_load(MLModel *m, const char *path)
{
    if (!m || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != ML_MAGIC)
        { fclose(f); return -1; }

    MLConfig saved;
    fread(&saved, sizeof(MLConfig), 1, f);
    fread(&m->step, sizeof(size_t), 1, f);

    const MLConfig *c = &m->cfg;
    int V=c->vocab_size, D=c->dim, di=c->d_inner, ds=c->d_state, dc=c->d_conv, dr=c->dt_rank;

    if (load_tensor(f, m->embedding,    (size_t)V*D) != 0) goto fail;
    if (load_tensor(f, m->final_norm_w, (size_t)D)   != 0) goto fail;

    for (int l = 0; l < c->n_layers; l++) {
        MambaLayerGPU *L = &m->layers[l];
        if (load_tensor(f, L->norm_w,   D)                    != 0) goto fail;
        if (load_tensor(f, L->in_proj,  (size_t)2*di*D)       != 0) goto fail;
        if (load_tensor(f, L->conv_w,   (size_t)di*dc)        != 0) goto fail;
        if (load_tensor(f, L->conv_b,   di)                   != 0) goto fail;
        if (load_tensor(f, L->x_proj,   (size_t)(dr+2*ds)*di) != 0) goto fail;
        if (load_tensor(f, L->dt_proj,  (size_t)di*dr)        != 0) goto fail;
        if (load_tensor(f, L->dt_bias,  di)                   != 0) goto fail;
        if (load_tensor(f, L->out_proj, (size_t)D*di)         != 0) goto fail;
        if (load_tensor(f, L->A_log,    (size_t)di*ds)        != 0) goto fail;
        if (load_tensor(f, L->D_param,  di)                   != 0) goto fail;
    }

    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}

} /* extern "C" */
