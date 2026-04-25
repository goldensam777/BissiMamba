#ifndef KMAMBA_SCAN_H
#define KMAMBA_SCAN_H

#include <stddef.h>

/* ============================================================
 * scan.h — Types et fonctions pour les scans sélectifs Mamba
 *
 * Scan 1D (séquences) et Scan 2D (grilles wavefront).
 * Ces opérations sont spécifiques à la logique Mamba —
 * elles ne font pas partie d'optimatrix.
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

/* ── Selective Scan 1D — forward ────────────────────────────── */
/*
 * Layout mémoire :
 *   x     : [L, D]
 *   A     : [D, M]       (partagé sur L)
 *   B     : [L, D, M]    (sélectif)
 *   C     : [L, D, M]    (sélectif)
 *   delta : [L, D]
 *   h     : [L, D, M]    (états cachés, stockés pour backward)
 *   y     : [L, D]
 *
 * Récurrence :
 *   h_t[d,m] = exp(dt_t[d] * A[d,m]) * h_{t-1}[d,m]
 *            + dt_t[d] * B_t[d,m] * x_t[d]
 *   y_t[d]   = sum_m C_t[d,m] * h_t[d,m]
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
} ScanParams;

void scan1d(ScanParams *p);

/* ── Selective Scan 1D — backward générique [L, D, M] ───────── */
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
} ScanBackwardParams;

void scan1d_backward(ScanBackwardParams *p);

/* ── Selective Scan 1D — backward M=1 (B/C partagés) ────────── */
typedef struct {
    float *x;
    float *A;
    float *A_diag;   /* exp(dt*A) précompté, ou NULL */
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
} ScanBackwardSharedParams;

void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p);
void scan1d_backward_m1_shared_bc_asm(ScanBackwardSharedParams *p);
void scan1d_backward_m1_shared_bc_simple_asm(ScanBackwardSharedParams *p);

/* ── Selective Scan 2D — wavefront ──────────────────────────── */
/*
 * Récurrence 2D sur grille (d1 x d2) :
 *   h(i,j,d,m) = dA1 * h(i-1,j,d,m)
 *              + dA2 * h(i,j-1,d,m)
 *              + dB  * x(i,j,d)
 *   y(i,j,d)   = sum_m C(i,j,d,m) * h(i,j,d,m)
 *
 * Ordonnancement wavefront : diagonale k = i+j.
 * Les positions sur la même diagonale sont indépendantes.
 */
typedef struct {
    float *x;
    float *A1;
    float *A2;
    float *B;
    float *C;
    float *delta1;
    float *delta2;
    float *h;
    float *y;
    long   d1;
    long   d2;
    long   D;
    long   M;
} Scan2DParams;

void scan2d(Scan2DParams *p);

/* ── CUDA API (scan ND GPU) ───────────────────────────────────
 * Note: 1D scans now use unified om_scannd_forward with ndims=1.
 * The separate 1D CUDA kernels have been removed in favor of
 * the wavefront-based ND implementation using Mamba-3 formula.
 */

#endif /* KMAMBA_SCAN_H */
