/* kmamba_mixed_precision.c — Mixed precision FP16/BF16 CPU implementation */

#include "../include/kmamba_mixed_precision.h"
#include <string.h>

/* ============================================================================
 * Batch conversions (CPU)
 * ============================================================================ */

void kmamba_fp32_to_fp16(const float *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_fp16(src[i]);
    }
}

void kmamba_fp16_to_fp32(const uint16_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp16_to_float(src[i]);
    }
}

void kmamba_fp32_to_bf16(const float *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_bf16(src[i]);
    }
}

void kmamba_bf16_to_fp32(const uint16_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = bf16_to_float(src[i]);
    }
}

/* ============================================================================
 * Loss Scaling utilities
 * ============================================================================ */

int kmamba_fp16_check_overflow(const float *gradients, size_t n, float threshold) {
    for (size_t i = 0; i < n; i++) {
        float g = gradients[i];
        /* Check for Inf or NaN */
        if (g != g || g > threshold || g < -threshold) {
            return 1;  /* Overflow detected */
        }
    }
    return 0;  /* No overflow */
}

float kmamba_adjust_loss_scale(float current_scale, int overflow_detected,
                                int *consecutive_no_overflow, int increase_every_n) {
    if (overflow_detected) {
        /* Overflow: reduce scale and reset counter */
        *consecutive_no_overflow = 0;
        return current_scale / 2.0f;
    } else {
        /* No overflow: increment counter and possibly increase scale */
        (*consecutive_no_overflow)++;
        if (*consecutive_no_overflow >= increase_every_n) {
            *consecutive_no_overflow = 0;
            /* Increase scale but cap at reasonable max */
            float new_scale = current_scale * 2.0f;
            return (new_scale > 16777216.0f) ? 16777216.0f : new_scale;
        }
    }
    return current_scale;
}
