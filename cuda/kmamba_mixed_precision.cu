/* kmamba_mixed_precision.cu — CUDA kernels for mixed precision FP16/BF16 */

#include "kmamba_mixed_precision.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/* ============================================================================
 * FP16 <-> FP32 conversion kernels
 * ============================================================================ */

__global__ void kmamba_cuda_fp32_to_fp16_kernel(const float *src, __half *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

__global__ void kmamba_cuda_fp16_to_fp32_kernel(const __half *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

/* ============================================================================
 * BF16 <-> FP32 conversion kernels
 * ============================================================================ */

__global__ void kmamba_cuda_fp32_to_bf16_kernel(const float *src, __nv_bfloat16 *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

__global__ void kmamba_cuda_bf16_to_fp32_kernel(const __nv_bfloat16 *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

/* ============================================================================
 * Host wrappers for conversions
 * ============================================================================ */

void kmamba_cuda_fp32_to_fp16(const float *d_src, void *d_dst, int n) {
    int blocks = (n + 255) / 256;
    kmamba_cuda_fp32_to_fp16_kernel<<<blocks, 256>>>(d_src, (__half*)d_dst, n);
    cudaDeviceSynchronize();
}

void kmamba_cuda_fp16_to_fp32(const void *d_src, float *d_dst, int n) {
    int blocks = (n + 255) / 256;
    kmamba_cuda_fp16_to_fp32_kernel<<<blocks, 256>>>((__half*)d_src, d_dst, n);
    cudaDeviceSynchronize();
}

void kmamba_cuda_fp32_to_bf16(const float *d_src, void *d_dst, int n) {
    int blocks = (n + 255) / 256;
    kmamba_cuda_fp32_to_bf16_kernel<<<blocks, 256>>>(d_src, (__nv_bfloat16*)d_dst, n);
    cudaDeviceSynchronize();
}

void kmamba_cuda_bf16_to_fp32(const void *d_src, float *d_dst, int n) {
    int blocks = (n + 255) / 256;
    kmamba_cuda_bf16_to_fp32_kernel<<<blocks, 256>>>((__nv_bfloat16*)d_src, d_dst, n);
    cudaDeviceSynchronize();
}

/* ============================================================================
 * Tensor Cores GEMM wrapper (FP16)
 * ============================================================================ */

#include <cublas_v2.h>

/* C = A @ B in FP16 using Tensor Cores
 * A: [M, K], B: [K, N], C: [M, N]
 * All pointers are device pointers in FP16 */
extern "C" cublasStatus_t kmamba_cublas_gemm_fp16(
    cublasHandle_t handle,
    cublasOperation_t transA, cublasOperation_t transB,
    int M, int N, int K,
    const __half *d_A, int lda,
    const __half *d_B, int ldb,
    __half *d_C, int ldc,
    const float *alpha, const float *beta)
{
    /* Use Tensor Cores when possible */
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    return cublasGemmEx(handle, transA, transB, M, N, K,
                        alpha,
                        d_A, CUDA_R_16F, lda,
                        d_B, CUDA_R_16F, ldb,
                        beta,
                        d_C, CUDA_R_16F, ldc,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

/* ============================================================================
 * Gradient scaling for FP16 stability
 * ============================================================================ */

__global__ void kmamba_scale_gradients_fp16_kernel(float *gradients, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gradients[i] *= scale;
}

void kmamba_cuda_scale_gradients(float *d_gradients, int n, float scale) {
    int blocks = (n + 255) / 256;
    kmamba_scale_gradients_fp16_kernel<<<blocks, 256>>>(d_gradients, n, scale);
    cudaDeviceSynchronize();
}

/* ============================================================================
 * FP16 overflow detection
 * ============================================================================ */

__global__ void kmamba_check_overflow_kernel(const float *gradients, int n, 
                                              float threshold, int *overflow_flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float g = gradients[i];
    if (g != g || g > threshold || g < -threshold) {
        atomicOr(overflow_flag, 1);
    }
}

int kmamba_cuda_check_overflow(const float *d_gradients, int n, float threshold) {
    int *d_overflow_flag;
    cudaMalloc(&d_overflow_flag, sizeof(int));
    cudaMemset(d_overflow_flag, 0, sizeof(int));
    
    int blocks = (n + 255) / 256;
    kmamba_check_overflow_kernel<<<blocks, 256>>>(d_gradients, n, threshold, d_overflow_flag);
    cudaDeviceSynchronize();
    
    int overflow_flag = 0;
    cudaMemcpy(&overflow_flag, d_overflow_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_overflow_flag);
    
    return overflow_flag;
}
