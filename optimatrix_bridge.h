/*
 * optimatrix_bridge.h — Interface C vers les kernels ASM x86-64 d'optimatrix
 *
 * Toutes les fonctions opèrent sur float32 (real_t = float).
 * Conventions : tableaux row-major, indices long.
 */

#ifndef OPTIMATRIX_BRIDGE_H
#define OPTIMATRIX_BRIDGE_H

/* ---- BLAS Niveau 2 : GEMV ----------------------------------------- */

/* y = A · x   (scalaire)
 * A : m×n float32 row-major, x : n, y : m
 */
void gemv(float *A, float *x, float *y, long m, long n);

/* y = A · x   (AVX2 — 8 floats/cycle)
 * Même interface que gemv, 8–12× plus rapide sur grandes matrices.
 */
void gemv_avx2(float *A, float *x, float *y, long m, long n);

/* ---- BLAS Niveau 3 : GEMM ----------------------------------------- */

/* C = A · B   (scalaire)
 * A : m×k, B : k×n, C : m×n  — tous float32 row-major
 */
void gemm(float *A, float *B, float *C, long m, long k, long n);

/* C = A · B   (AVX2)
 * Même interface, vectorisé sur n (8 floats/iter).
 */
void gemm_avx2(float *A, float *B, float *C, long m, long k, long n);

/* ---- Produit de Hadamard ------------------------------------------ */

/* z[i] = x[i] * y[i]   (scalaire) */
void hadamard(float *x, float *y, float *z, long n);

/* z[i] = x[i] * y[i]   (AVX2) */
void hadamard_avx2(float *x, float *y, float *z, long n);

/* ---- Fonctions d'activation --------------------------------------- */

/* y[i] = max(0, x[i]) */
void relu_f32(float *x, float *y, long n);

/* y[i] = 1 / (1 + expf(-x[i])) */
void sigmoid_f32(float *x, float *y, long n);

/* y[i] = x[i] * sigmoid(x[i])  (SiLU / Swish) */
void silu_f32(float *x, float *y, long n);

/* y[i] = logf(1 + expf(x[i])) */
void softplus_f32(float *x, float *y, long n);

/* ---- Selective Scan 1D -------------------------------------------- */

typedef struct {
    float *x;       /* [L, D]       — entrée */
    float *A;       /* [D, M]       — log diagonal de A (valeurs négatives) */
    float *B;       /* [L, D, M]    — matrice d'entrée sélective */
    float *C;       /* [L, D, M]    — matrice de sortie sélective */
    float *delta;   /* [L, D]       — pas de discrétisation (positif) */
    float *h;       /* [D, M]       — état caché (entrée/sortie) */
    float *y;       /* [L, D]       — sortie */
    long   L;
    long   D;
    long   M;
} ScanParams;

/* Scan sélectif 1D — interface principale */
void scan1d(ScanParams *p);

typedef struct {
    float *x;       /* [L, D]       — entrée du scan */
    float *A;       /* [D, M]       — paramètres de transition */
    float *B;       /* [L, D, M]    — matrice d'entrée sélective */
    float *C;       /* [L, D, M]    — matrice de sortie sélective */
    float *delta;   /* [L, D]       — pas de discrétisation */
    float *h0;      /* [D, M]       — état initial optionnel (NULL => zéro) */
    float *h;       /* [L, D, M]    — états forward après mise à jour */
    float *dy;      /* [L, D]       — gradient amont sur y */
    float *dx;      /* [L, D]       — gradient sur x */
    float *dA;      /* [D, M]       — gradient sur A */
    float *dB;      /* [L, D, M]    — gradient sur B */
    float *dC;      /* [L, D, M]    — gradient sur C */
    float *ddelta;  /* [L, D]       — gradient sur delta */
    long   L;
    long   D;
    long   M;
} ScanBackwardParams;

/* Backward du scan sélectif 1D.
 * Les buffers de sortie sont remis à zéro puis écrits.
 */
void scan1d_backward(ScanBackwardParams *p);

typedef struct {
    float *x;       /* [L, D]       — entrée du scan */
    float *A;       /* [D]          — paramètres de transition */
    float *B;       /* [D]          — matrice B partagée par canal */
    float *C;       /* [D]          — matrice C partagée par canal */
    float *delta;   /* [L]          — pas scalaire par timestep */
    float *h0;      /* [D]          — état initial optionnel (NULL => zéro) */
    float *h;       /* [L, D]       — états forward après mise à jour */
    float *dy;      /* [L, D]       — gradient amont sur y */
    float *dx;      /* [L, D]       — gradient sur x */
    float *dA;      /* [D]          — gradient sur A */
    float *dB;      /* [D]          — gradient sur B */
    float *dC;      /* [D]          — gradient sur C */
    float *ddelta;  /* [L]          — gradient sur delta */
    long   L;
    long   D;
} ScanBackwardSharedParams;

/* Backward spécialisé pour M=1 avec B/C partagés par canal
 * et delta[t] scalaire. Les buffers de sortie sont remis à zéro puis écrits.
 */
void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p);

/* ---- Selective Scan 2D (Wavefront) -------------------------------- */

typedef struct {
    float *x;       /* [d1, d2, D]     — entrée */
    float *A1;      /* [D, M]          — transition axe 1 */
    float *A2;      /* [D, M]          — transition axe 2 */
    float *B;       /* [d1, d2, D, M]  — matrice d entrée sélective */
    float *C;       /* [d1, d2, D, M]  — matrice de sortie sélective */
    float *delta1;  /* [d1, d2, D]     — pas axe 1 */
    float *delta2;  /* [d1, d2, D]     — pas axe 2 */
    float *h;       /* [d1, d2, D, M]  — tous les états cachés */
    float *y;       /* [d1, d2, D]     — sortie */
    long   d1;
    long   d2;
    long   D;
    long   M;
} Scan2DParams;

/* Scan sélectif 2D avec ordonnancement wavefront */
void scan2d(Scan2DParams *p);

#endif /* OPTIMATRIX_BRIDGE_H */
