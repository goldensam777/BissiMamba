/* ============================================================================
 * kser_quantize.c — Quantization / dequantization utilities
 * FP32 ↔ FP16 / BF16 / INT8 (zero-dependency, C99)
 * ============================================================================ */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "kser.h"

/* ── FP32 ↔ FP16 (IEEE 754 half-precision) ─────────────────────────────── */

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t s  = (x >> 31) & 1u;
    uint32_t e  = (x >> 23) & 0xFFu;
    uint32_t m  = x & 0x7FFFFFu;

    if (e == 0)         return (uint16_t)(s << 15);            /* ±zero    */
    if (e == 0xFF)      return (uint16_t)((s<<15)|0x7C00|(m?0x200:0)); /* inf/NaN */

    int ne = (int)e - 127 + 15;
    if (ne >= 31)       return (uint16_t)((s<<15)|0x7C00);     /* overflow */
    if (ne <= 0) {
        if (ne < -10)   return (uint16_t)(s << 15);
        m = (m | 0x800000u) >> (1 - ne);
        return (uint16_t)((s<<15) | (m>>13));
    }
    return (uint16_t)((s<<15) | ((uint16_t)ne<<10) | (m>>13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t x;

    if (e == 0) {
        if (m == 0) { x = s << 31; }
        else {
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            x = (s<<31)|((e+112)<<23)|(m<<13);
        }
    } else if (e == 0x1F) {
        x = (s<<31)|0x7F800000u|(m<<13);
    } else {
        x = (s<<31)|((e+112)<<23)|(m<<13);
    }
    float f; memcpy(&f, &x, 4); return f;
}

/* ── FP32 ↔ BF16 (brain float, truncate mantissa) ──────────────────────── */

static uint16_t f32_to_bf16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1u;
    x = (x + 0x7FFFu + lsb) >> 16;
    return (uint16_t)x;
}

static float bf16_to_f32(uint16_t h) {
    uint32_t x = (uint32_t)h << 16;
    float f; memcpy(&f, &x, 4); return f;
}

/* ── FP32 → INT8 (per-tensor min-max symmetric) ─────────────────────────── */

/* INT8 header stored at the start of the quantized buffer:
 *   float scale (4 bytes)
 *   float zero_point (4 bytes)
 * followed by n uint8 values.
 * Total buffer size = 8 + n bytes.
 */
#define INT8_HDR 8  /* bytes for scale + zero_point */

static void int8_params(const float* data, uint64_t n,
                         float* scale, float* zp) {
    float lo = data[0], hi = data[0];
    for (uint64_t i = 1; i < n; i++) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }
    float range = hi - lo;
    if (range < 1e-8f) { *scale = 1.0f; *zp = lo; return; }
    *scale = range / 255.0f;
    *zp    = lo;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

size_t kser_dtype_size(KSerDtype dtype) {
    switch (dtype) {
        case KSER_FP32: return 4;
        case KSER_FP16: return 2;
        case KSER_BF16: return 2;
        case KSER_INT8: return 1;  /* NOTE: caller must add INT8_HDR separately */
        default:        return 4;
    }
}

/*
 * kser_quantize — convert n FP32 values to target dtype.
 * dst must be large enough:
 *   FP32/FP16/BF16: n * kser_dtype_size(dtype)
 *   INT8:           INT8_HDR + n bytes
 */
int kser_quantize(const float* src, void* dst, uint64_t n, KSerDtype dtype) {
    if (!src || !dst || n == 0) return KSER_ERR_IO;
    uint64_t i;
    switch (dtype) {
        case KSER_FP32:
            memcpy(dst, src, n * 4);
            return KSER_OK;
        case KSER_FP16: {
            uint16_t* out = (uint16_t*)dst;
            for (i=0; i<n; i++) out[i] = f32_to_f16(src[i]);
            return KSER_OK;
        }
        case KSER_BF16: {
            uint16_t* out = (uint16_t*)dst;
            for (i=0; i<n; i++) out[i] = f32_to_bf16(src[i]);
            return KSER_OK;
        }
        case KSER_INT8: {
            float scale, zp;
            int8_params(src, n, &scale, &zp);
            uint8_t* out = (uint8_t*)dst;
            memcpy(out,   &scale, 4);
            memcpy(out+4, &zp,    4);
            out += INT8_HDR;
            for (i=0; i<n; i++) {
                float v = (src[i] - zp) / scale;
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                out[i] = (uint8_t)(v + 0.5f);
            }
            return KSER_OK;
        }
        default: return KSER_ERR_FORMAT;
    }
}

/*
 * kser_dequantize — convert back to FP32.
 * For INT8, src must start with the 8-byte header written by kser_quantize.
 */
int kser_dequantize(const void* src, float* dst, uint64_t n, KSerDtype dtype) {
    if (!src || !dst || n == 0) return KSER_ERR_IO;
    uint64_t i;
    switch (dtype) {
        case KSER_FP32:
            memcpy(dst, src, n * 4);
            return KSER_OK;
        case KSER_FP16: {
            const uint16_t* in = (const uint16_t*)src;
            for (i=0; i<n; i++) dst[i] = f16_to_f32(in[i]);
            return KSER_OK;
        }
        case KSER_BF16: {
            const uint16_t* in = (const uint16_t*)src;
            for (i=0; i<n; i++) dst[i] = bf16_to_f32(in[i]);
            return KSER_OK;
        }
        case KSER_INT8: {
            const uint8_t* raw = (const uint8_t*)src;
            float scale, zp;
            memcpy(&scale, raw,   4);
            memcpy(&zp,    raw+4, 4);
            raw += INT8_HDR;
            for (i=0; i<n; i++) dst[i] = zp + raw[i] * scale;
            return KSER_OK;
        }
        default: return KSER_ERR_FORMAT;
    }
}

/*
 * kser_quantize_size — buffer size needed for kser_quantize output.
 */
size_t kser_quantize_size(uint64_t n, KSerDtype dtype) {
    if (dtype == KSER_INT8) return INT8_HDR + n;
    return n * kser_dtype_size(dtype);
}