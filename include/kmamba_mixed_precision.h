/* kmamba_mixed_precision.h — Mixed precision FP16/BF16 support for 1B+ models */

#ifndef KMAMBA_MIXED_PRECISION_H
#define KMAMBA_MIXED_PRECISION_H

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * Mixed Precision Types
 * ============================================================================ */

typedef enum {
    KMAMBA_PREC_FP32 = 0,  /* Default: full precision */
    KMAMBA_PREC_FP16 = 1,  /* Half precision: Tensor Cores, needs loss scaling */
    KMAMBA_PREC_BF16 = 2   /* BFloat16: better range than FP16, no loss scaling */
} KMambaPrecision;

/* ============================================================================
 * FP16/BF16 Conversions (host-side)
 * ============================================================================ */

/* Convert float to half-precision (FP16) */
static inline uint16_t float_to_fp16(float f) {
    /* IEEE 754 conversion: float32 -> float16
     * Extract sign, exponent, mantissa and requantize */
    union { float f; uint32_t u; } u = { .f = f };
    uint32_t sign = (u.u >> 31) & 0x1;
    uint32_t exp = (u.u >> 23) & 0xFF;
    uint32_t mant = u.u & 0x7FFFFF;
    
    /* Handle special cases */
    if (exp == 0xFF) {
        /* Inf/NaN */
        uint16_t h = (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
        return h;
    }
    
    if (exp == 0) {
        /* Zero/subnormal -> zero in FP16 */
        return sign << 15;
    }
    
    /* Normalized: requantize exponent from 8-bit (bias 127) to 5-bit (bias 15) */
    int32_t new_exp = (int32_t)exp - 127 + 15;
    
    if (new_exp >= 31) {
        /* Overflow -> Inf */
        return (sign << 15) | 0x7C00;
    }
    if (new_exp <= 0) {
        /* Underflow -> zero */
        return sign << 15;
    }
    
    /* Round mantissa from 23-bit to 10-bit */
    uint32_t new_mant = mant >> 13;
    /* Round to nearest even */
    if ((mant & 0x1FFF) > 0x1000 || ((mant & 0x1FFF) == 0x1000 && (new_mant & 1))) {
        new_mant++;
    }
    
    return (sign << 15) | ((uint16_t)new_exp << 10) | (uint16_t)new_mant;
}

/* Convert half-precision (FP16) to float */
static inline float fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 0x1;
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x3FF;
    
    if (exp == 0) {
        /* Zero/subnormal */
        if (mant == 0) {
            union { uint32_t u; float f; } u = { .u = (uint32_t)sign << 31 };
            return u.f;
        }
        /* Subnormal: denormalized */
        /* For simplicity, convert subnormal to normalized approximation */
        uint32_t new_exp = 1 + 127 - 15; /* Smallest normal exponent in FP32 */
        uint32_t new_mant = mant << 13;
        while ((new_mant & 0x800000) == 0 && new_exp > 0) {
            new_mant <<= 1;
            new_exp--;
        }
        union { uint32_t u; float f; } u = { .u = ((uint32_t)sign << 31) | (new_exp << 23) | (new_mant & 0x7FFFFF) };
        return u.f;
    }
    
    if (exp == 0x1F) {
        /* Inf/NaN */
        uint32_t new_mant = mant ? 0x7FFFFF : 0;
        union { uint32_t u; float f; } u = { .u = ((uint32_t)sign << 31) | (0xFF << 23) | new_mant };
        return u.f;
    }
    
    /* Normalized: requantize exponent from 5-bit (bias 15) to 8-bit (bias 127) */
    uint32_t new_exp = (uint32_t)exp - 15 + 127;
    uint32_t new_mant = ((uint32_t)mant) << 13;
    
    union { uint32_t u; float f; } u = { .u = ((uint32_t)sign << 31) | (new_exp << 23) | new_mant };
    return u.f;
}

/* Convert float to BFloat16 */
static inline uint16_t float_to_bf16(float f) {
    /* BF16 is like FP32 but truncated mantissa: 1 sign, 8 exp, 7 mantissa */
    union { float f; uint32_t u; } u = { .f = f };
    /* Simply truncate the lower 16 bits of the float32 */
    uint16_t bf = (uint16_t)(u.u >> 16);
    return bf;
}

/* Convert BFloat16 to float */
static inline float bf16_to_float(uint16_t h) {
    /* BF16 -> FP32: shift left by 16, lower bits are zero */
    union { uint32_t u; float f; } u = { .u = ((uint32_t)h) << 16 };
    return u.f;
}

/* ============================================================================
 * Batch conversions
 * ============================================================================ */

/* Convert FP32 array to FP16 array */
void kmamba_fp32_to_fp16(const float *src, uint16_t *dst, size_t n);

/* Convert FP16 array to FP32 array */
void kmamba_fp16_to_fp32(const uint16_t *src, float *dst, size_t n);

/* Convert FP32 array to BF16 array */
void kmamba_fp32_to_bf16(const float *src, uint16_t *dst, size_t n);

/* Convert BF16 array to FP32 array */
void kmamba_bf16_to_fp32(const uint16_t *src, float *dst, size_t n);

#ifdef KMAMBA_BUILD_CUDA
/* ============================================================================
 * CUDA Mixed Precision Functions
 * ============================================================================ */

/* Note: CUDA kernels are defined in cuda/kmamba_mixed_precision.cu
 * The following are C linkage wrappers for host code to call */

#ifdef __cplusplus
extern "C" {
#endif

/* Convert FP32 to FP16 on device */
void kmamba_cuda_fp32_to_fp16(const float *d_src, void *d_dst, int n);

/* Convert FP16 to FP32 on device */
void kmamba_cuda_fp16_to_fp32(const void *d_src, float *d_dst, int n);

/* Convert FP32 to BF16 on device */
void kmamba_cuda_fp32_to_bf16(const float *d_src, void *d_dst, int n);

/* Convert BF16 to FP32 on device */
void kmamba_cuda_bf16_to_fp32(const void *d_src, float *d_dst, int n);

/* Scale gradients on device */
void kmamba_cuda_scale_gradients(float *d_gradients, int n, float scale);

/* Check for FP16 overflow in gradients (returns 1 if overflow detected) */
int kmamba_cuda_check_overflow(const float *d_gradients, int n, float threshold);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * Loss Scaling for FP16 Training Stability
 * ============================================================================ */

/* Initialize loss scale (default: 65536.0f for FP16, 1.0f for BF16) */
static inline float kmamba_get_default_loss_scale(KMambaPrecision prec) {
    switch (prec) {
        case KMAMBA_PREC_FP16: return 65536.0f;
        case KMAMBA_PREC_BF16: return 1.0f;  /* BF16 has same range as FP32 */
        default: return 1.0f;
    }
}

/* Check for FP16 overflow/underflow in gradients
 * Returns 1 if any Inf/NaN detected, 0 otherwise */
int kmamba_fp16_check_overflow(const float *gradients, size_t n, float threshold);

/* Adjust loss scale based on gradient status
 * If overflow detected: scale /= 2
 * If underflow (no overflow for N steps): scale *= 2
 * Returns new loss scale */
float kmamba_adjust_loss_scale(float current_scale, int overflow_detected, 
                                int *consecutive_no_overflow, int increase_every_n);

#endif /* KMAMBA_BUILD_CUDA */

#endif /* KMAMBA_MIXED_PRECISION_H */
