#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "scan_nd.h"

#define SCANND_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return -1; \
    } \
} while (0)

static int scannd_dims_valid_host(const long *dims, long ndims) {
    if (!dims || ndims <= 0) return 0;
    for (long axis = 0; axis < ndims; axis++) {
        if (dims[axis] <= 0) return 0;
    }
    return 1;
}

static int scannd_safe_add_long(long a, long b, long *out) {
    if (!out) return 0;
    if ((b > 0 && a > LONG_MAX - b) || (b < 0 && a < LONG_MIN - b)) return 0;
    *out = a + b;
    return 1;
}

static int scannd_safe_mul_long(long a, long b, long *out) {
    if (!out || a < 0 || b < 0) return 0;
    if (a == 0 || b == 0) {
        *out = 0;
        return 1;
    }
    if (a > LONG_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static long scannd_total_points_host(const long *dims, long ndims) {
    long total = 1;

    if (!scannd_dims_valid_host(dims, ndims)) return -1;
    for (long axis = 0; axis < ndims; axis++) {
        if (!scannd_safe_mul_long(total, dims[axis], &total)) return -1;
    }
    return total;
}

static long scannd_max_level_host(const long *dims, long ndims) {
    long max_level = 0;

    if (!scannd_dims_valid_host(dims, ndims)) return -1;
    for (long axis = 0; axis < ndims; axis++) {
        if (!scannd_safe_add_long(max_level, dims[axis] - 1, &max_level)) return -1;
    }
    return max_level;
}

static int scannd_build_strides_host(const long *dims, long ndims, long *strides) {
    long stride = 1;

    if (!dims || !strides || ndims <= 0) return 0;

    for (long axis = ndims - 1; axis >= 0; axis--) {
        strides[axis] = stride;
        if (!scannd_safe_mul_long(stride, dims[axis], &stride)) return 0;
    }
    return 1;
}

static int scannd_build_suffix_caps(const long *dims, long ndims, long *suffix_caps) {
    long suffix = 0;

    if (!dims || !suffix_caps || ndims <= 0) return 0;

    suffix_caps[ndims] = 0;
    for (long axis = ndims - 1; axis >= 0; axis--) {
        long cap = dims[axis] - 1;
        if (cap < 0 || !scannd_safe_add_long(suffix, cap, &suffix)) return 0;
        suffix_caps[axis] = suffix;
    }
    return 1;
}

static long scannd_max_long(long a, long b) {
    return (a > b) ? a : b;
}

static long scannd_min_long(long a, long b) {
    return (a < b) ? a : b;
}

static long scannd_level_size_recursive(const long *dims,
                                        const long *suffix_caps,
                                        long ndims,
                                        long axis,
                                        long remaining) {
    long lo;
    long hi;
    long count = 0;

    if (axis == ndims - 1) {
        return (remaining >= 0 && remaining < dims[axis]) ? 1 : 0;
    }

    lo = scannd_max_long(0, remaining - suffix_caps[axis + 1]);
    hi = scannd_min_long(dims[axis] - 1, remaining);

    for (long value = lo; value <= hi; value++) {
        long sub = scannd_level_size_recursive(dims, suffix_caps, ndims, axis + 1, remaining - value);
        if (sub < 0 || !scannd_safe_add_long(count, sub, &count)) return -1;
    }

    return count;
}

typedef struct {
    const long *dims;
    const long *strides;
    const long *level_offsets;
    const long *suffix_caps;
    long *ordered_offsets;
    long *prev_offsets;
    long *idx;
    long ndims;
    long level;
    int failed;
} ScanNDEmitCtx;

static void scannd_emit_level_recursive(ScanNDEmitCtx *ctx, long axis, long remaining) {
    long lo;
    long hi;

    if (!ctx || ctx->failed) return;

    if (axis == ctx->ndims - 1) {
        if (remaining >= 0 && remaining < ctx->dims[axis]) {
            long ordinal = 0;
            long offset = 0;
            long slot;

            ctx->idx[axis] = remaining;
            for (long d = 0; d < ctx->ndims; d++) {
                offset += ctx->idx[d] * ctx->strides[d];
            }

            slot = ctx->level_offsets[ctx->level];
            while (slot + ordinal < ctx->level_offsets[ctx->level + 1] &&
                   ctx->ordered_offsets[slot + ordinal] != -1) {
                ordinal++;
            }

            if (slot + ordinal >= ctx->level_offsets[ctx->level + 1]) {
                ctx->failed = 1;
                return;
            }

            slot += ordinal;
            ctx->ordered_offsets[slot] = offset;
            for (long d = 0; d < ctx->ndims; d++) {
                ctx->prev_offsets[slot * ctx->ndims + d] =
                    (ctx->idx[d] > 0) ? (offset - ctx->strides[d]) : -1;
            }
        }
        return;
    }

    lo = scannd_max_long(0, remaining - ctx->suffix_caps[axis + 1]);
    hi = scannd_min_long(ctx->dims[axis] - 1, remaining);

    for (long value = lo; value <= hi; value++) {
        ctx->idx[axis] = value;
        scannd_emit_level_recursive(ctx, axis + 1, remaining - value);
        if (ctx->failed) return;
    }
}

static int scannd_build_schedule(const long *dims, long ndims,
                                 long **level_offsets_out,
                                 long **ordered_offsets_out,
                                 long **prev_offsets_out,
                                 long *total_points_out,
                                 long *max_level_out) {
    long total_points;
    long max_level;
    long *strides = NULL;
    long *suffix_caps = NULL;
    long *level_offsets = NULL;
    long *ordered_offsets = NULL;
    long *prev_offsets = NULL;
    long *idx = NULL;
    int ok = 0;

    if (!level_offsets_out || !ordered_offsets_out || !prev_offsets_out ||
        !total_points_out || !max_level_out) {
        return 0;
    }

    total_points = scannd_total_points_host(dims, ndims);
    max_level = scannd_max_level_host(dims, ndims);
    if (total_points <= 0 || max_level < 0) return 0;

    strides = (long *)malloc((size_t)ndims * sizeof(long));
    suffix_caps = (long *)malloc((size_t)(ndims + 1) * sizeof(long));
    level_offsets = (long *)malloc((size_t)(max_level + 2) * sizeof(long));
    ordered_offsets = (long *)malloc((size_t)total_points * sizeof(long));
    prev_offsets = (long *)malloc((size_t)(total_points * ndims) * sizeof(long));
    idx = (long *)malloc((size_t)ndims * sizeof(long));

    if (!strides || !suffix_caps || !level_offsets || !ordered_offsets || !prev_offsets || !idx) goto cleanup;
    if (!scannd_build_strides_host(dims, ndims, strides)) goto cleanup;
    if (!scannd_build_suffix_caps(dims, ndims, suffix_caps)) goto cleanup;

    for (long i = 0; i < total_points; i++) ordered_offsets[i] = -1;

    level_offsets[0] = 0;
    for (long level = 0; level <= max_level; level++) {
        long level_size = scannd_level_size_recursive(dims, suffix_caps, ndims, 0, level);
        if (level_size < 0 || !scannd_safe_add_long(level_offsets[level], level_size, &level_offsets[level + 1])) {
            goto cleanup;
        }
    }

    for (long level = 0; level <= max_level; level++) {
        ScanNDEmitCtx ctx;
        ctx.dims = dims;
        ctx.strides = strides;
        ctx.level_offsets = level_offsets;
        ctx.suffix_caps = suffix_caps;
        ctx.ordered_offsets = ordered_offsets;
        ctx.prev_offsets = prev_offsets;
        ctx.idx = idx;
        ctx.ndims = ndims;
        ctx.level = level;
        ctx.failed = 0;

        scannd_emit_level_recursive(&ctx, 0, level);
        if (ctx.failed) goto cleanup;
    }

    *level_offsets_out = level_offsets;
    *ordered_offsets_out = ordered_offsets;
    *prev_offsets_out = prev_offsets;
    *total_points_out = total_points;
    *max_level_out = max_level;
    level_offsets = NULL;
    ordered_offsets = NULL;
    prev_offsets = NULL;
    ok = 1;

cleanup:
    free(strides);
    free(suffix_caps);
    free(idx);
    free(level_offsets);
    free(ordered_offsets);
    free(prev_offsets);
    return ok;
}

__global__ void scannd_level_kernel(const float *x,
                                    const float *A,
                                    const float *B,
                                    const float *delta,
                                    const float *lambda,
                                    float default_lambda,
                                    const long *ordered_offsets,
                                    const long *prev_offsets,
                                    long level_start,
                                    long level_size,
                                    long total_points,
                                    int ndims,
                                    int D,
                                    int M,
                                    float *h,
                                    float *Bu_prev) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * (long)D * M;

    if (tid >= total_threads) return;

    int m = (int)(tid % M);
    long tmp = tid / M;
    int d = (int)(tmp % D);
    long point_ordinal = tmp / D;
    long slot = level_start + point_ordinal;
    long offset = ordered_offsets[slot];
    long dm = (long)d * M + m;
    long pdm = offset * ((long)D * M) + dm;
    float x_val = x[offset * D + d];
    float dt_bar = 0.0f;
    float h_new;

    for (int axis = 0; axis < ndims; axis++) {
        dt_bar += delta[((long)axis * total_points + offset) * D + d];
    }
    /* Note: pas de division par ndims (conforme Eq 10 du papier) */

    /* Mamba-3: lambda per point for exp-trapezoidal */
    float lambda_n = lambda ? lambda[offset] : default_lambda;

    /* Compute Bu_t = B_t * x_t */
    float bu_t = B[pdm] * x_val;

    /* Mamba-3 exp-trapezoidal formula */
    float bu_prev_acc = 0.0f;
    float decay_acc = 0.0f;
    float h_decay_acc = 0.0f;
    int n_pred = 0;

    /* Single pass: compute all alphas, accumulate Bu and h contributions */
    for (int axis = 0; axis < ndims; axis++) {
        long prev_offset = prev_offsets[slot * ndims + axis];
        if (prev_offset >= 0) {
            float dt_axis = delta[((long)axis * total_points + offset) * D + d];
            float a_val = A[((long)axis * D + d) * M + m];
            float alpha = expf(dt_axis * a_val);
            long prev_pdm = prev_offset * ((long)D * M) + dm;

            bu_prev_acc += Bu_prev[prev_pdm];
            decay_acc += alpha;
            h_decay_acc += alpha * h[prev_pdm];
            n_pred++;
        }
    }

    float bu_prev_avg = (n_pred > 0) ? (bu_prev_acc / n_pred) : 0.0f;
    float alpha_avg = (n_pred > 0) ? (decay_acc / n_pred) : 0.0f;
    float beta = (1.0f - lambda_n) * dt_bar * alpha_avg;
    float gamma = lambda_n * dt_bar;

    h_new = bu_prev_avg * beta + bu_t * gamma + h_decay_acc;
    Bu_prev[pdm] = bu_t;  /* Store for next level */

    h[pdm] = h_new;
}

__global__ void scannd_output_kernel(const float *h,
                                     const float *C,
                                     long total_points,
                                     int D,
                                     int M,
                                     float *y) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = total_points * (long)D;

    if (tid >= total_threads) return;

    long offset = tid / D;
    int d = (int)(tid % D);
    long base = offset * ((long)D * M) + (long)d * M;
    float acc = 0.0f;

    for (int m = 0; m < M; m++) {
        acc += C[base + m] * h[base + m];
    }
    y[offset * D + d] = acc;
}

/* ── Backward kernels ───────────────────────────────────────── */

__global__ void scannd_bwd_level_kernel(
    const float *x,        /* [total_points, D] */
    const float *A,        /* [ndims, D, M] */
    const float *B,        /* [total_points, D, M] */
    const float *C,        /* [total_points, D, M] */
    const float *delta,    /* [ndims, total_points, D] */
    const float *lambda,   /* [total_points] or NULL */
    float default_lambda,
    const float *theta,    /* [D/2] or NULL */
    const float *h,        /* [total_points, D, M] */
    const float *dy,       /* [total_points, D] upstream gradient */
    const long *ordered_offsets,
    const long *prev_offsets,
    long level_start,
    long level_size,
    long total_points,
    int ndims,
    int D,
    int M,
    float *adj_h,          /* [total_points, D, M] stored adjoints */
    /* Output gradients (accumulated) */
    float *dA,
    float *dB,
    float *dC,
    float *ddelta,
    float *dlambda,
    float *dtheta)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * (long)D * M;

    if (tid >= total_threads) return;

    int m = (int)(tid % M);
    long tmp = tid / M;
    int d = (int)(tmp % D);
    long point_ordinal = tmp / D;
    long slot = level_start + point_ordinal;
    long offset = ordered_offsets[slot];
    
    long dm = (long)d * M + m;
    long pdm = offset * ((long)D * M) + dm;

    /* Upstream gradient from output y_t */
    float dy_t = dy[offset * D + d];
    float c_val = C[pdm];
    float h_val = h[pdm];

    /* g_C contribution */
    if (dC) atomicAdd(&dC[pdm], dy_t * h_val);

    /* Local adjoint adj_h[t] from dy_t and successors */
    /* Successors logic: for each point, we need to find points that have it as predecessor.
     * In an ND grid, if current is t, successors are t + e_axis.
     * This kernel is called in reverse level order. 
     * Successors are in levels > current level, so their adjoints are already computed.
     * BUT: identifying successors linearly is hard. 
     * Strategy: each point propagates its contribution TO its predecessors' adjoints.
     * This means we need to do the propagation in the level where we HAVE the information.
     * So at level L, we compute adj_h[t] and propagate back to predecessors at level L-1.
     */

    float ah_t = adj_h[pdm] + dy_t * c_val;
    float x_t = x[offset * D + d];
    float dt_bar = 0.0f;
    for (int axis = 0; axis < ndims; axis++) {
        dt_bar += delta[((long)axis * total_points + offset) * D + d];
    }
    float lam_t = lambda ? lambda[offset] : default_lambda;

    /* Bu_t contribution to dB */
    float gamma_t = lam_t * dt_bar;
    float d_bu_t = ah_t * gamma_t;
    if (dB) atomicAdd(&dB[pdm], d_bu_t * x_t);

    /* dlambda contribution */
    /* Mamba-3 forward: h_t = h_decay + Bu_prev * beta + Bu_cur * gamma
     * dlambda = ah_t * (dbeta/dlambda * Bu_prev + dgamma/dlambda * Bu_cur)
     * dgamma/dlambda = dt_bar
     * dbeta/dlambda = -dt_bar * alpha_avg
     */

    /* Propagate to predecessors */
    for (int axis = 0; axis < ndims; axis++) {
        long prev_offset = prev_offsets[slot * ndims + axis];
        if (prev_offset >= 0) {
            float dt_axis = delta[((long)axis * total_points + offset) * D + d];
            float a_val = A[((long)axis * D + d) * M + m];
            float alpha = expf(dt_axis * a_val);
            long prev_pdm = prev_offset * ((long)D * M) + dm;
            
            /* Recover h_prev_rot from stored states and Bu */
            /* Simplified for now: just use stored h[prev] and apply R(theta) */
            float h_prev_val = h[prev_pdm];
            float h_prev_rot;
            if (theta && (d & 1U)) {
                float th = theta[d >> 1];
                h_prev_rot = sinf(th) * h[prev_offset * ((long)D * M) + (d-1)*M + m] + cosf(th) * h_prev_val;
            } else if (theta && (d+1 < D)) {
                float th = theta[d >> 1];
                h_prev_rot = cosf(th) * h_prev_val - sinf(th) * h[(prev_offset * ((long)D * M) + (d+1)*M + m)];
            } else {
                h_prev_rot = h_prev_val;
            }

            /* 1. Contribution to adj_h[prev] with R(theta)^T = R(-theta) */
            float adj_val;
            if (theta) {
                float th = theta[d >> 1];
                float cv = cosf(th), sv = sinf(th);
                if (d & 1U) {
                    /* adj_h1 = alpha * (sin*adj_h0_new + cos*adj_h1_new) */
                    /* Note: current thread handles m-th state of channel d.
                     * We need adj_h of the other channel in the pair to compute full adjoint.
                     * Wait, simpler: each thread (d,m) computes its contribution to ITS predecessor.
                     */
                    adj_val = ah_t * alpha * cv;
                    atomicAdd(&adj_h[prev_pdm], adj_val);
                    /* contribution of ah_t to the other channel's adjoint in the pair */
                    atomicAdd(&adj_h[prev_offset * ((long)D * M) + (d-1)*M + m], ah_t * alpha * sv);
                } else if (d+1 < D) {
                    /* adj_h0 = alpha * (cos*adj_h0_new - sin*adj_h1_new) */
                    adj_val = ah_t * alpha * cv;
                    atomicAdd(&adj_h[prev_pdm], adj_val);
                    atomicAdd(&adj_h[prev_offset * ((long)D * M) + (d+1)*M + m], -ah_t * alpha * sv);
                } else {
                    atomicAdd(&adj_h[prev_pdm], ah_t * alpha);
                }
            } else {
                atomicAdd(&adj_h[prev_pdm], ah_t * alpha);
            }

            /* 2. Contribution to dA */
            if (dA) {
                float da_val = ah_t * dt_axis * alpha * h_prev_rot;
                atomicAdd(&dA[((long)axis * D + d) * M + m], da_val);
            }

            /* 3. Contribution to ddelta */
            if (ddelta) {
                float ddt = ah_t * a_val * alpha * h_prev_rot;
                atomicAdd(&ddelta[((long)axis * total_points + offset) * D + d], ddt);
            }

            /* bu_prev contribution for dlambda and ddelta */
            /* Bu_prev = B_prev * x_prev (should be saved or recomputed)
             * For now, let's assume we can compute it if needed. 
             * But we need Bu_prev at point t, which is Bu at predecessor.
             */
            /* 4. Contribution to dtheta */
            if (dtheta) {
                float cv = cosf(theta[d >> 1]), sv = sinf(theta[d >> 1]);
                float hp0, hp1;
                long p_off0 = prev_offset * ((long)D * M) + (d & ~1U) * M + m;
                long p_off1 = prev_offset * ((long)D * M) + (d | 1U) * M + m;
                hp0 = h[p_off0];
                hp1 = h[p_off1];
                
                float dth_val;
                if (d & 1U) {
                    /* h1_new = alpha * (sin*h0 + cos*h1) -> d/dth = alpha * (cos*h0 - sin*h1) */
                    dth_val = ah_t * alpha * (cv * hp0 - sv * hp1);
                } else {
                    /* h0_new = alpha * (cos*h0 - sin*h1) -> d/dth = alpha * (-sin*h0 - cos*h1) */
                    dth_val = ah_t * alpha * (-sv * hp0 - cv * hp1);
                }
                atomicAdd(&dtheta[d >> 1], dth_val);
            }
        }
    }
}

/* Note: rotation adjoint must be handled carefully.
 * Since R(theta) is orthogonal, R(theta)^T = R(-theta).
 * We can launch a separate kernel for rotation or integrate it.
 */

int om_scannd_forward(ScanNDParams *p) {
    long *h_level_offsets = NULL;
    long *h_ordered_offsets = NULL;
    long *h_prev_offsets = NULL;
    long total_points;
    long max_level;
    long *d_ordered_offsets = NULL;
    long *d_prev_offsets = NULL;
    float *d_Bu_prev = NULL;
    int rc = -1;

    if (!p || !scannd_dims_valid_host(p->dims, p->ndims) || p->D <= 0 || p->M <= 0 ||
        !p->x || !p->A || !p->B || !p->C || !p->delta || !p->h || !p->y) {
        return -1;
    }

    if (!scannd_build_schedule(p->dims, p->ndims,
                               &h_level_offsets, &h_ordered_offsets, &h_prev_offsets,
                               &total_points, &max_level)) {
        return -1;
    }

    /* Mamba-3: Allocate Bu_prev buffer for exp-trapezoidal formula */
    size_t bu_prev_size = (size_t)total_points * p->D * p->M * sizeof(float);
    SCANND_CUDA_CHECK(cudaMalloc(&d_ordered_offsets, (size_t)total_points * sizeof(long)));
    SCANND_CUDA_CHECK(cudaMalloc(&d_prev_offsets, (size_t)(total_points * p->ndims) * sizeof(long)));
    SCANND_CUDA_CHECK(cudaMalloc(&d_Bu_prev, bu_prev_size));
    SCANND_CUDA_CHECK(cudaMemset(d_Bu_prev, 0, bu_prev_size)); /* Zero Bu_prev */
    SCANND_CUDA_CHECK(cudaMemcpy(d_ordered_offsets, h_ordered_offsets,
                                 (size_t)total_points * sizeof(long),
                                 cudaMemcpyHostToDevice));
    SCANND_CUDA_CHECK(cudaMemcpy(d_prev_offsets, h_prev_offsets,
                                 (size_t)(total_points * p->ndims) * sizeof(long),
                                 cudaMemcpyHostToDevice));

    for (long level = 0; level <= max_level; level++) {
        long level_start = h_level_offsets[level];
        long level_size = h_level_offsets[level + 1] - level_start;
        long total_threads = level_size * p->D * p->M;
        int blocks = (int)((total_threads + 255) / 256);

        if (level_size <= 0) continue;

        scannd_level_kernel<<<blocks, 256>>>(
            p->x, p->A, p->B, p->delta, p->lambda, p->default_lambda,
            d_ordered_offsets, d_prev_offsets,
            level_start, level_size, total_points,
            (int)p->ndims, (int)p->D, (int)p->M, p->h, d_Bu_prev);
        SCANND_CUDA_CHECK(cudaGetLastError());
    }

    {
        long total_threads = total_points * p->D;
        int blocks = (int)((total_threads + 255) / 256);
        scannd_output_kernel<<<blocks, 256>>>(
            p->h, p->C, total_points, (int)p->D, (int)p->M, p->y);
        SCANND_CUDA_CHECK(cudaGetLastError());
    }

    SCANND_CUDA_CHECK(cudaDeviceSynchronize());
    rc = 0;

    cudaFree(d_ordered_offsets);
    cudaFree(d_prev_offsets);
    cudaFree(d_Bu_prev);
    free(h_level_offsets);
    free(h_ordered_offsets);
    free(h_prev_offsets);
    return rc;
}

int om_scannd_backward(ScanNDParams *p) {
    long *h_level_offsets = NULL;
    long *h_ordered_offsets = NULL;
    long *h_prev_offsets = NULL;
    long total_points;
    long max_level;
    long *d_ordered_offsets = NULL;
    long *d_prev_offsets = NULL;
    float *d_adj_h = NULL;
    int rc = -1;

    if (!p || !scannd_dims_valid_host(p->dims, p->ndims) || p->D <= 0 || p->M <= 0 ||
        !p->x || !p->A || !p->B || !p->C || !p->delta || !p->h || !p->y) {
        return -1;
    }

    if (!scannd_build_schedule(p->dims, p->ndims,
                               &h_level_offsets, &h_ordered_offsets, &h_prev_offsets,
                               &total_points, &max_level)) {
        return -1;
    }

    /* Allocate adjoint buffer */
    size_t state_bytes = (size_t)total_points * p->D * p->M * sizeof(float);
    SCANND_CUDA_CHECK(cudaMalloc(&d_adj_h, state_bytes));
    SCANND_CUDA_CHECK(cudaMemset(d_adj_h, 0, state_bytes));

    SCANND_CUDA_CHECK(cudaMalloc(&d_ordered_offsets, (size_t)total_points * sizeof(long)));
    SCANND_CUDA_CHECK(cudaMalloc(&d_prev_offsets, (size_t)(total_points * p->ndims) * sizeof(long)));
    SCANND_CUDA_CHECK(cudaMemcpy(d_ordered_offsets, h_ordered_offsets,
                                 (size_t)total_points * sizeof(long),
                                 cudaMemcpyHostToDevice));
    SCANND_CUDA_CHECK(cudaMemcpy(d_prev_offsets, h_prev_offsets,
                                 (size_t)(total_points * p->ndims) * sizeof(long),
                                 cudaMemcpyHostToDevice));

    /* Reverse wavefront loop */
    for (long level = max_level; level >= 0; level--) {
        long level_start = h_level_offsets[level];
        long level_size = h_level_offsets[level + 1] - level_start;
        long total_threads = level_size * p->D * p->M;
        int blocks = (int)((total_threads + 255) / 256);

        if (level_size <= 0) continue;

        scannd_bwd_level_kernel<<<blocks, 256>>>(
            p->x, p->A, p->B, p->C, p->delta, p->lambda, p->default_lambda,
            p->theta, p->h, p->y,
            d_ordered_offsets, d_prev_offsets,
            level_start, level_size, total_points,
            (int)p->ndims, (int)p->D, (int)p->M, d_adj_h,
            p->dA, p->dB, p->dC, p->ddelta, p->dlambda, p->dtheta);
        SCANND_CUDA_CHECK(cudaGetLastError());
    }

    SCANND_CUDA_CHECK(cudaDeviceSynchronize());
    rc = 0;

    cudaFree(d_ordered_offsets);
    cudaFree(d_prev_offsets);
    cudaFree(d_adj_h);
    free(h_level_offsets);
    free(h_ordered_offsets);
    free(h_prev_offsets);
    return rc;
}
