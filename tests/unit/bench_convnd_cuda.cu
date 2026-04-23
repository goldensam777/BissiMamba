/*
 * bench_convnd_cuda.cu — Benchmark ConvND CUDA: Dense vs Séparable
 *
 * Compare les performances des convolutions dense K^N et séparable
 * cascade 1D sur GPU avec différentes tailles de grille 2D/3D.
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

/* External CUDA APIs - declared in convnd.h */
int om_convnd_forward(ConvNDParams *p);
int om_convnd_separable_forward(ConvNDSeparableParams *p);

/* Timing helper */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Fill with random pattern */
static void fill_random(float *data, long n) {
    for (long i = 0; i < n; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }
}

/* Benchmark 2D grid */
static void bench_2d(long N, long D, long K, int warmup, int runs) {
    long dims[2] = {N, N};
    long ndims = 2;
    long spatial = N * N;
    long total = spatial * D;
    long kernel_volume = K * K;

    /* Host allocations */
    float *h_input = (float *)malloc(total * sizeof(float));
    float *h_output_dense = (float *)malloc(total * sizeof(float));
    float *h_output_sep = (float *)malloc(total * sizeof(float));
    float *h_kernel_dense = (float *)malloc(kernel_volume * D * sizeof(float));
    float h_kernel_1d[3] = {0.333f, 0.333f, 0.333f};

    fill_random(h_input, total);
    fill_random(h_kernel_dense, kernel_volume * D);

    /* Device allocations */
    float *d_input, *d_output_dense, *d_output_sep, *d_kernel_dense;
    float *d_kernel_axis0, *d_kernel_axis1;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output_dense, total * sizeof(float));
    cudaMalloc(&d_output_sep, total * sizeof(float));
    cudaMalloc(&d_kernel_dense, kernel_volume * D * sizeof(float));
    cudaMalloc(&d_kernel_axis0, K * sizeof(float));
    cudaMalloc(&d_kernel_axis1, K * sizeof(float));

    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_dense, h_kernel_dense, kernel_volume * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_axis0, h_kernel_1d, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_axis1, h_kernel_1d, K * sizeof(float), cudaMemcpyHostToDevice);

    /* Dense params */
    ConvNDParams p_dense;
    p_dense.input = d_input;
    p_dense.output = d_output_dense;
    p_dense.kernel = d_kernel_dense;
    p_dense.bias = NULL;
    p_dense.dy = NULL;
    p_dense.dinput = NULL;
    p_dense.dkernel = NULL;
    p_dense.dbias = NULL;
    p_dense.dims = dims;
    p_dense.ndims = ndims;
    p_dense.D = D;
    p_dense.K = K;

    /* Separable params */
    float *kernel_axes[2] = {d_kernel_axis0, d_kernel_axis1};
    ConvNDSeparableParams p_sep;
    p_sep.input = d_input;
    p_sep.output = d_output_sep;
    p_sep.kernel_axes = kernel_axes;
    p_sep.bias = NULL;
    p_sep.dims = dims;
    p_sep.ndims = ndims;
    p_sep.D = D;
    p_sep.K = K;

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        om_convnd_forward(&p_dense);
        om_convnd_separable_forward(&p_sep);
    }
    cudaDeviceSynchronize();

    /* Benchmark Dense */
    double t_start = get_time_ms();
    for (int i = 0; i < runs; i++) {
        om_convnd_forward(&p_dense);
    }
    cudaDeviceSynchronize();
    double t_dense = (get_time_ms() - t_start) / runs;

    /* Benchmark Separable */
    t_start = get_time_ms();
    for (int i = 0; i < runs; i++) {
        om_convnd_separable_forward(&p_sep);
    }
    cudaDeviceSynchronize();
    double t_sep = (get_time_ms() - t_start) / runs;

    /* Compute bandwidth */
    double data_mb = (double)total * sizeof(float) / (1024.0 * 1024.0);
    double bw_dense = data_mb / (t_dense / 1000.0);
    double bw_sep = data_mb / (t_sep / 1000.0);

    printf("  Dense:   %8.3f ms  (%6.2f GB/s)\n", t_dense, bw_dense);
    printf("  Sep:     %8.3f ms  (%6.2f GB/s)\n", t_sep, bw_sep);
    printf("  Speedup: %.2fx\n", t_dense / t_sep);

    /* Cleanup */
    free(h_input); free(h_output_dense); free(h_output_sep); free(h_kernel_dense);
    cudaFree(d_input); cudaFree(d_output_dense); cudaFree(d_output_sep);
    cudaFree(d_kernel_dense); cudaFree(d_kernel_axis0); cudaFree(d_kernel_axis1);
}

/* Benchmark 3D grid */
static void bench_3d(long N, long D, long K, int warmup, int runs) {
    long dims[3] = {N, N, N};
    long ndims = 3;
    long spatial = N * N * N;
    long total = spatial * D;
    long kernel_volume = K * K * K;

    float *h_input = (float *)malloc(total * sizeof(float));
    float *h_kernel_dense = (float *)malloc(kernel_volume * D * sizeof(float));
    float h_kernel_1d[3] = {0.333f, 0.333f, 0.333f};

    fill_random(h_input, total);
    fill_random(h_kernel_dense, kernel_volume * D);

    float *d_input, *d_output_dense, *d_output_sep, *d_kernel_dense;
    float *d_kernel_axis0, *d_kernel_axis1, *d_kernel_axis2;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output_dense, total * sizeof(float));
    cudaMalloc(&d_output_sep, total * sizeof(float));
    cudaMalloc(&d_kernel_dense, kernel_volume * D * sizeof(float));
    cudaMalloc(&d_kernel_axis0, K * sizeof(float));
    cudaMalloc(&d_kernel_axis1, K * sizeof(float));
    cudaMalloc(&d_kernel_axis2, K * sizeof(float));

    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_dense, h_kernel_dense, kernel_volume * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_axis0, h_kernel_1d, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_axis1, h_kernel_1d, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_axis2, h_kernel_1d, K * sizeof(float), cudaMemcpyHostToDevice);

    ConvNDParams p_dense;
    p_dense.input = d_input;
    p_dense.output = d_output_dense;
    p_dense.kernel = d_kernel_dense;
    p_dense.bias = NULL;
    p_dense.dy = NULL;
    p_dense.dinput = NULL;
    p_dense.dkernel = NULL;
    p_dense.dbias = NULL;
    p_dense.dims = dims;
    p_dense.ndims = ndims;
    p_dense.D = D;
    p_dense.K = K;

    float *kernel_axes[3] = {d_kernel_axis0, d_kernel_axis1, d_kernel_axis2};
    ConvNDSeparableParams p_sep;
    p_sep.input = d_input;
    p_sep.output = d_output_sep;
    p_sep.kernel_axes = kernel_axes;
    p_sep.bias = NULL;
    p_sep.dims = dims;
    p_sep.ndims = ndims;
    p_sep.D = D;
    p_sep.K = K;

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        om_convnd_forward(&p_dense);
        om_convnd_separable_forward(&p_sep);
    }
    cudaDeviceSynchronize();

    /* Benchmark */
    double t_start = get_time_ms();
    for (int i = 0; i < runs; i++) om_convnd_forward(&p_dense);
    cudaDeviceSynchronize();
    double t_dense = (get_time_ms() - t_start) / runs;

    t_start = get_time_ms();
    for (int i = 0; i < runs; i++) om_convnd_separable_forward(&p_sep);
    cudaDeviceSynchronize();
    double t_sep = (get_time_ms() - t_start) / runs;

    double data_mb = (double)total * sizeof(float) / (1024.0 * 1024.0);
    double bw_dense = data_mb / (t_dense / 1000.0);
    double bw_sep = data_mb / (t_sep / 1000.0);

    printf("  Dense:   %8.3f ms  (%6.2f GB/s)\n", t_dense, bw_dense);
    printf("  Sep:     %8.3f ms  (%6.2f GB/s)\n", t_sep, bw_sep);
    printf("  Speedup: %.2fx\n", t_dense / t_sep);

    free(h_input); free(h_kernel_dense);
    cudaFree(d_input); cudaFree(d_output_dense); cudaFree(d_output_sep);
    cudaFree(d_kernel_dense);
    cudaFree(d_kernel_axis0); cudaFree(d_kernel_axis1); cudaFree(d_kernel_axis2);
}

int main(int argc, char **argv) {
    int warmup = 2;
    int runs = 10;
    long D = 64;
    long K = 3;

    if (argc > 1) runs = atoi(argv[1]);

    printf("================================================\n");
    printf("Benchmark ConvND CUDA: Dense K^N vs Séparable\n");
    printf("================================================\n");
    printf("Config: D=%ld channels, K=%ld kernel size\n", D, K);
    printf("Warmup: %d, Runs: %d\n\n", warmup, runs);

    /* Check CUDA */
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA device found!\n");
        return 1;
    }
    printf("CUDA devices: %d\n", device_count);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    /* 2D benchmarks */
    printf("2D Grids (D=%ld):\n", D);
    long sizes_2d[] = {64, 128, 256, 512};
    for (int i = 0; i < 4; i++) {
        printf("\n%ld×%ld:\n", sizes_2d[i], sizes_2d[i]);
        bench_2d(sizes_2d[i], D, K, warmup, runs);
    }

    /* 3D benchmarks */
    printf("\n\n3D Grids (D=%ld):\n", D);
    long sizes_3d[] = {32, 64, 128};
    for (int i = 0; i < 3; i++) {
        printf("\n%ld×%ld×%ld:\n", sizes_3d[i], sizes_3d[i], sizes_3d[i]);
        bench_3d(sizes_3d[i], D, K, warmup, runs);
    }

    printf("\n================================================\n");
    printf("Benchmark complete\n");
    printf("================================================\n");

    return 0;
}
