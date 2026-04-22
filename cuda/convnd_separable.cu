/*
 * convnd_separable.cu — Convolution ND Séparable CUDA
 *
 * Cascade de convolutions 1D par axe avec ping-pong buffers GPU.
 * Chaque axe est traité séquentiellement, avec parallélisme intra-niveau
 * wavefront sur chaque étape 1D.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "convnd.h"
#include "wavefront_plan.h"
#include "km_topology.h"
#include "convnd_cuda_common.cuh"

/* ── 1D convolution kernel ─────────────────────────────────── */
__global__ void separable_1d_kernel(
    const float *input,
    float *output,
    const float *kernel_1d,
    const long *ordered_offsets,
    long level_start,
    long level_size,
    const long *dims,
    long ndims,
    const long *strides,
    long axis,
    long D,
    long K)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * D;

    if (tid >= total_threads) return;

    int d = (int)(tid % D);
    long point_ordinal = tid / D;
    long slot = level_start + point_ordinal;
    long linear = ordered_offsets[slot];

    /* Unravel to get coordinates */
    long coords[KMAMBA_MAX_NDIMS];
    convnd_d_unravel_index(linear, dims, ndims, strides, coords);

    long axis_stride = strides[axis];
    long kernel_radius = K / 2;
    long coord_axis = coords[axis];

    /* Handle boundaries: copy input */
    if (coord_axis < kernel_radius || coord_axis >= dims[axis] - kernel_radius) {
        output[linear * D + d] = input[linear * D + d];
        return;
    }

    /* 1D convolution along axis */
    float sum = 0.0f;
    for (long k = 0; k < K; k++) {
        long offset_k = (k - kernel_radius) * axis_stride;
        long src_idx = linear + offset_k;
        sum += input[src_idx * D + d] * kernel_1d[k];
    }
    output[linear * D + d] = sum;
}


/* ── API: Separable Forward ─────────────────────────────────── */
int om_convnd_separable_forward(ConvNDSeparableParams *p) {
    KMWavefrontPlan **plans = NULL;
    float *d_temp = NULL;
    float *current_input = NULL;
    float *current_output = NULL;
    long spatial_total;
    int rc = -1;

    if (!p || !p->input || !p->output || !p->kernel_axes ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return -1;
    }

    /* Compute total spatial size */
    spatial_total = 1;
    for (long i = 0; i < p->ndims; i++) spatial_total *= p->dims[i];

    /* Allocate temporary GPU buffer for ping-pong */
    CONVND_CUDA_CHECK(cudaMalloc(&d_temp, (size_t)spatial_total * p->D * sizeof(float)));

    /* Create plans for each axis */
    plans = (KMWavefrontPlan **)malloc((size_t)p->ndims * sizeof(KMWavefrontPlan *));
    if (!plans) goto cleanup;

    for (long axis = 0; axis < p->ndims; axis++) {
        plans[axis] = km_wavefront_plan_create(p->dims, p->ndims);
        if (!plans[axis]) goto cleanup;
    }

    /* Cascade: 1D conv per axis */
    current_input = p->input;
    current_output = d_temp;

    for (long axis = 0; axis < p->ndims; axis++) {
        KMWavefrontPlan *plan = plans[axis];
        long *d_ordered_offsets = NULL;
        long *d_dims = NULL;
        long *d_strides = NULL;
        long *h_strides = NULL;
        float *d_kernel_1d = NULL;

        /* Upload kernel for this axis */
        CONVND_CUDA_CHECK(cudaMalloc(&d_kernel_1d, (size_t)p->K * sizeof(float)));
        CONVND_CUDA_CHECK(cudaMemcpy(d_kernel_1d, p->kernel_axes[axis],
                                         (size_t)p->K * sizeof(float),
                                         cudaMemcpyHostToDevice));

        /* Compute strides */
        h_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
        if (!h_strides) { cudaFree(d_kernel_1d); goto cleanup; }
        convnd_h_make_strides(p->dims, p->ndims, h_strides);

        /* Upload plan data */
        CONVND_CUDA_CHECK(cudaMalloc(&d_ordered_offsets,
                                         (size_t)plan->total_points * sizeof(long)));
        CONVND_CUDA_CHECK(cudaMalloc(&d_dims, (size_t)p->ndims * sizeof(long)));
        CONVND_CUDA_CHECK(cudaMalloc(&d_strides, (size_t)p->ndims * sizeof(long)));

        CONVND_CUDA_CHECK(cudaMemcpy(d_ordered_offsets, plan->level_offsets,
                                         (size_t)plan->total_points * sizeof(long),
                                         cudaMemcpyHostToDevice));
        CONVND_CUDA_CHECK(cudaMemcpy(d_dims, p->dims,
                                         (size_t)p->ndims * sizeof(long),
                                         cudaMemcpyHostToDevice));
        CONVND_CUDA_CHECK(cudaMemcpy(d_strides, h_strides,
                                         (size_t)p->ndims * sizeof(long),
                                         cudaMemcpyHostToDevice));

        /* Process each level */
        for (long level = 0; level <= plan->max_level; level++) {
            long level_size = km_wavefront_plan_level_size(plan, level);
            if (level_size <= 0) continue;

            long level_start = plan->level_starts[level];
            long total_threads = level_size * p->D;
            int blocks = (int)((total_threads + 255) / 256);

            separable_1d_kernel<<<blocks, 256>>>(
                current_input, current_output, d_kernel_1d,
                d_ordered_offsets, level_start, level_size,
                d_dims, p->ndims, d_strides, axis,
                p->D, p->K);

            CONVND_CUDA_CHECK(cudaGetLastError());
        }

        CONVND_CUDA_CHECK(cudaDeviceSynchronize());

        /* Cleanup per-axis allocations */
        cudaFree(d_kernel_1d);
        cudaFree(d_ordered_offsets);
        cudaFree(d_dims);
        cudaFree(d_strides);
        free(h_strides);

        /* Ping-pong swap */
        if (axis == 0) {
            current_input = d_temp;
            current_output = (p->ndims > 1) ? p->output : d_temp;
        } else if (axis == p->ndims - 1) {
            /* Last axis: ensure output goes to p->output */
            if (current_output != p->output) {
                CONVND_CUDA_CHECK(cudaMemcpy(p->output, current_output,
                                                 (size_t)spatial_total * p->D * sizeof(float),
                                                 cudaMemcpyDeviceToDevice));
            }
        } else {
            /* Intermediate: swap buffers */
            float *tmp = current_input;
            current_input = current_output;
            current_output = (tmp == d_temp) ? p->output : d_temp;
        }
    }

    /* Add bias if present */
    if (p->bias) {
        /* TODO: implement bias addition kernel */
    }

    rc = 0;

cleanup:
    if (plans) {
        for (long i = 0; i < p->ndims; i++) {
            if (plans[i]) km_wavefront_plan_free(plans[i]);
        }
        free(plans);
    }
    cudaFree(d_temp);
    return rc;
}
