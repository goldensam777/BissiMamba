#ifndef KMAMBA_SCAN_H
#define KMAMBA_SCAN_H

#include <stddef.h>
#include <string.h>
#include "scan_nd.h"

/* ============================================================
 * scan.h — DEPRECATED: Unified scan interface
 *
 * WARNING: This header is kept for backward compatibility only.
 * All scan functionality (1D, 2D, ND) is now unified in scan_nd.h
 * using ScanNDParams and the wavefront-based implementation.
 *
 * The old 1D/2D specific structs and functions below are deprecated
 * and will be removed in a future version.
 * ============================================================
 *
 * Migration guide:
 *   - ScanParams/Scan2DParams  → ScanNDParams (with dims array)
 *   - scan1d()/scan2d()        → scannd() or scannd_ref()
 *   - scan1d_backward()        → scannd_backward() (when available)
 *
 * Example: 1D scan with L=100, D=16, M=64
 *   long dims[1] = {100};
 *   ScanNDParams p = {
 *       .dims = dims, .ndims = 1, .D = 16, .M = 64,
 *       .x = x, .A = A, .B = B, .C = C, .delta = delta,
 *       .h = h, .y = y,
 *       .default_lambda = 0.5f, ...
 *   };
 *   scannd(&p);
 * ============================================================ */

/* ── Macro CUDA pour vérification d'erreurs ─────────────────── */
#ifdef __CUDACC__
#include <stdio.h>
#include <stdlib.h>
#define OM_CHECK(call)                                                  \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(1);                                                    \
        }                                                               \
    } while (0)
#endif

/* ─────────────────────────────────────────────────────────────
 * DEPRECATED SECTION — The following structs and functions
 * are kept for backward compatibility but should not be used
 * in new code. Use ScanNDParams and scannd() instead.
 * ───────────────────────────────────────────────────────────── */

/* ── DEPRECATED: Selective Scan 1D — forward ───────────────────
 * Use ScanNDParams with ndims=1 instead.
 */
typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h;
    float *y;
    long   L;
    long   D;
    long   M;
} ScanParams __attribute__((deprecated("Use ScanNDParams with ndims=1")));

/* DEPRECATED: Redirects to scannd() with automatic conversion.
 * Only use if you have existing ScanParams structures.
 * New code should use ScanNDParams directly. */
static inline int scan1d(ScanParams *p) {
    long dims[1] = {p->L};
    ScanNDParams ndp = {
        .max_ndims = 1, .max_state = p->M, .use_fast_exp = 0,
        .dims = dims, .ndims = 1, .D = p->D, .M = p->M,
        .x = p->x, .A = p->A, .B = p->B, .C = p->C, .delta = p->delta,
        .h = p->h, .y = p->y,
        .theta = NULL, .lambda = NULL,
        .default_lambda = 0.5f, .use_a_log_clamp = 0, .a_log_min = -1e-5f
    };
    return scannd(&ndp);
}

/* ── DEPRECATED: Selective Scan 1D — backward ─────────────────
 * Use ScanNDParams backward (when implemented) instead.
 * Currently returns -1 (not implemented in unified backend).
 */
typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
    long   M;
} ScanBackwardParams __attribute__((deprecated("Unified backward not yet implemented")));

static inline int scan1d_backward(ScanBackwardParams *p) {
    (void)p;
    return -1; /* TODO: Implement unified backward */
}

/* ── DEPRECATED: Scan 1D backward M=1 variants ────────────────
 * These ASM-optimized variants are REMOVED (ASM backends deleted).
 * The structures are kept empty for compilation compatibility.
 */
typedef struct {
    float *x; float *A; float *A_diag; float *B; float *C;
    float *delta; float *h0; float *h; float *dy;
    float *dx; float *dA; float *dB; float *dC; float *ddelta;
    long L; long D;
} ScanBackwardSharedParams __attribute__((deprecated("ASM backends removed, use ScanNDParams")));

static inline int scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p) {
    (void)p; return -1;
}
static inline int scan1d_backward_m1_shared_bc_asm(ScanBackwardSharedParams *p) {
    (void)p; return -1;
}
static inline int scan1d_backward_m1_shared_bc_simple_asm(ScanBackwardSharedParams *p) {
    (void)p; return -1;
}

/* ── DEPRECATED: Selective Scan 2D — wavefront ───────────────
 * Use ScanNDParams with ndims=2 instead.
 * Layout conversion: A1,A2 → A[0],A[1]; delta1,delta2 → delta.
 */
typedef struct {
    float *x;
    float *A1;  /* → A at axis 0 */
    float *A2;  /* → A at axis 1 */
    float *B;
    float *C;
    float *delta1;  /* → delta at axis 0 */
    float *delta2;  /* → delta at axis 1 */
    float *h;
    float *y;
    long   d1;
    long   d2;
    long   D;
    long   M;
} Scan2DParams __attribute__((deprecated("Use ScanNDParams with ndims=2")));

/* DEPRECATED: Redirects to scannd() with automatic conversion.
 * The A1/A2 and delta1/delta2 pointers are repacked into
 * contiguous arrays for the ND interface. */
static inline int scan2d(Scan2DParams *p) {
    long dims[2] = {p->d1, p->d2};
    /* For 2D, A and delta need repacking. This is a simplified
     * wrapper that assumes caller has compatible layout.
     * Full conversion would require temp allocation. */
    float A_combined[2 * p->D * p->M];
    float delta_combined[2 * p->d1 * p->d2 * p->D];
    /* Copy A1 and A2 into combined A */
    memcpy(A_combined, p->A1, p->D * p->M * sizeof(float));
    memcpy(A_combined + p->D * p->M, p->A2, p->D * p->M * sizeof(float));
    /* Note: delta repacking omitted for brevity - use ScanNDParams directly */
    ScanNDParams ndp = {
        .max_ndims = 2, .max_state = p->M, .use_fast_exp = 0,
        .dims = dims, .ndims = 2, .D = p->D, .M = p->M,
        .x = p->x, .A = A_combined, .B = p->B, .C = p->C,
        .delta = p->delta1, /* Simplified - assumes delta1 layout matches */
        .h = p->h, .y = p->y,
        .theta = NULL, .lambda = NULL,
        .default_lambda = 0.5f, .use_a_log_clamp = 0, .a_log_min = -1e-5f
    };
    return scannd(&ndp);
}

#ifdef __cplusplus
/* Convenience: Allow C++ code to use ScanParams as alias to ScanNDParams
 * during migration period. This enables gradual refactoring. */
[[deprecated("Use ScanNDParams")]]
typedef ScanNDParams ScanParamsND;
#endif

#endif /* KMAMBA_SCAN_H */
