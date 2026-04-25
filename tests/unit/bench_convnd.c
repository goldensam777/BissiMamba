/*
 * bench_convnd.c — Benchmark comparatif: convND dense vs séparable
 *
 * Compare les deux implémentations sur les mêmes données:
 * - convnd_forward_wavefront (dense K^N)
 * - convnd_separable_forward_wavefront (cascade 1D)
 *
 * Usage: ./bench_convnd [dims...] [D] [K]
 *   Example: ./bench_convnd 128 128 64 3  (2D 128x128, 64 canaux, K=3)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "convnd.h"
#include "km_topology.h"
#include "wavefront_plan.h"

/* Timer utils - avoid conflict with POSIX timer_t */
typedef struct {
    struct timespec start;
    struct timespec end;
} bench_timer_t;

static void timer_start(bench_timer_t *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static double timer_elapsed_ms(bench_timer_t *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    double ms = (t->end.tv_sec - t->start.tv_sec) * 1000.0;
    ms += (t->end.tv_nsec - t->start.tv_nsec) / 1e6;
    return ms;
}

static long product(const long *arr, long n) {
    long p = 1;
    for (long i = 0; i < n; i++) p *= arr[i];
    return p;
}

static float frand(void) {
    return (float)rand() / RAND_MAX - 0.5f;
}

int main(int argc, char *argv[]) {
    long dims[4] = {64, 64, 1, 1};  /* Default 2D */
    long ndims = 2;
    long D = 32;  /* Canaux */
    long K = 3;   /* Kernel size */
    long spatial_total;
    float *input, *output_dense, *output_sep, *kernel_dense, **kernel_sep, *bias;
    ConvNDParams p_dense;
    ConvNDSeparableParams p_sep;
    KMWavefrontPlan *plan;
    bench_timer_t timer;
    double t_dense, t_sep;
    int i, j;
    float max_diff, diff;
    (void)product;  /* Suppress unused warning */

    /* Parse arguments */
    if (argc >= 3) {
        ndims = argc - 3;
        if (ndims > 4) ndims = 4;
        for (i = 0; i < ndims; i++) {
            dims[i] = atoi(argv[1 + i]);
        }
        D = atoi(argv[argc - 2]);
        K = atoi(argv[argc - 1]);
    }

    printf("=== Benchmark convND Dense vs Séparable ===\n");
    printf("Shape: [");
    spatial_total = 1;
    for (i = 0; i < ndims; i++) {
        printf("%ld%s", dims[i], (i < ndims - 1) ? ", " : "");
        spatial_total *= dims[i];
    }
    printf("], D=%ld, K=%ld\n", D, K);
    printf("Spatial total: %ld, Volume: %ld\n\n", spatial_total, spatial_total * D);

    /* Allocate memory */
    input = (float *)malloc((size_t)spatial_total * D * sizeof(float));
    output_dense = (float *)malloc((size_t)spatial_total * D * sizeof(float));
    output_sep = (float *)malloc((size_t)spatial_total * D * sizeof(float));
    kernel_dense = (float *)malloc((size_t)((long)pow(K, ndims) * D) * sizeof(float));
    kernel_sep = (float **)malloc((size_t)ndims * sizeof(float *));
    bias = (float *)malloc((size_t)D * sizeof(float));

    if (!input || !output_dense || !output_sep || !kernel_dense || !kernel_sep || !bias) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (i = 0; i < ndims; i++) {
        kernel_sep[i] = (float *)malloc((size_t)K * sizeof(float));
        if (!kernel_sep[i]) {
            fprintf(stderr, "Allocation failed\n");
            return 1;
        }
    }

    /* Initialize data */
    srand(42);
    for (i = 0; i < spatial_total * D; i++) input[i] = frand();
    for (i = 0; i < (long)pow(K, ndims) * D; i++) kernel_dense[i] = frand() / K;
    for (i = 0; i < ndims; i++) {
        for (j = 0; j < K; j++) {
            /* Separable kernel = product of 1D kernels = same total weight */
            kernel_sep[i][j] = frand() / K;
        }
    }
    for (i = 0; i < D; i++) bias[i] = frand() * 0.1f;

    /* Create wavefront plan */
    plan = km_wavefront_plan_create(dims, ndims, 0);
    if (!plan) {
        fprintf(stderr, "Failed to create wavefront plan\n");
        return 1;
    }

    /* === DENSE FORWARD === */
    memset(&p_dense, 0, sizeof(p_dense));
    p_dense.input = input;
    p_dense.kernel = kernel_dense;
    p_dense.bias = bias;
    p_dense.output = output_dense;
    p_dense.dims = dims;
    p_dense.ndims = ndims;
    p_dense.D = D;
    p_dense.K = K;

    /* Warmup */
    convnd_forward_wavefront(&p_dense, plan);

    /* Benchmark dense */
    timer_start(&timer);
    for (i = 0; i < 10; i++) {
        convnd_forward_wavefront(&p_dense, plan);
    }
    t_dense = timer_elapsed_ms(&timer) / 10.0;

    /* === SEPARABLE FORWARD === */
    memset(&p_sep, 0, sizeof(p_sep));
    p_sep.input = input;
    p_sep.kernel_axes = kernel_sep;
    p_sep.bias = bias;
    p_sep.output = output_sep;
    p_sep.dims = dims;
    p_sep.ndims = ndims;
    p_sep.D = D;
    p_sep.K = K;

    /* Warmup */
    convnd_separable_forward_wavefront(&p_sep, NULL);

    /* Benchmark separable */
    timer_start(&timer);
    for (i = 0; i < 10; i++) {
        convnd_separable_forward_wavefront(&p_sep, NULL);
    }
    t_sep = timer_elapsed_ms(&timer) / 10.0;

    /* === RESULTS === */
    printf("Dense (K^N=%.0f^%ld):     %.3f ms\n", pow(K, ndims), ndims, t_dense);
    printf("Séparable (ndims×K=%ld×%ld): %.3f ms\n", ndims, K, t_sep);
    printf("Speedup: %.2fx\n\n", t_dense / t_sep);

    /* === CORRECTNESS CHECK === */
    max_diff = 0.0f;
    for (i = 0; i < spatial_total * D; i++) {
        diff = fabsf(output_dense[i] - output_sep[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max |dense - sep|: %.6f\n", max_diff);
    printf("Note: Differences expected due to different kernel initialization.\n");
    printf("      Both implementations should be correct independently.\n");

    /* === MEMORY BANDWIDTH ESTIMATE === */
    double bytes_dense = (spatial_total * D * 2 + pow(K, ndims) * D) * sizeof(float);
    double bytes_sep = spatial_total * D * (2 + ndims) * sizeof(float) + ndims * K * sizeof(float);
    printf("\n=== Bandwidth ===\n");
    printf("Dense:    %.1f MB, %.1f GB/s\n", bytes_dense / 1e6, bytes_dense / 1e9 / (t_dense / 1000.0));
    printf("Séparable: %.1f MB, %.1f GB/s\n", bytes_sep / 1e6, bytes_sep / 1e9 / (t_sep / 1000.0));

    /* Cleanup */
    free(input);
    free(output_dense);
    free(output_sep);
    free(kernel_dense);
    for (i = 0; i < ndims; i++) free(kernel_sep[i]);
    free(kernel_sep);
    free(bias);
    km_wavefront_plan_free(plan);

    return 0;
}
