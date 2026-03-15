#ifndef MAMBA_SCAN_H
#define MAMBA_SCAN_H

#include <stddef.h>

/* ============================================================
 * Mamba-specific Scan 1D (Selective State Space Model)
 * ============================================================ */

typedef struct {
    float *x;       /* Input sequence [L * D] */
    float *A;       /* Diagonal matrix [D * M] */
    float *B;       /* Input matrix [L * D * M] */
    float *C;       /* Output matrix [L * D * M] */
    float *dt;      /* Delta [L * D] */
    float *h;       /* Hidden states [L * D * M] */
    float *y;       /* Output [L * D] */
    long   L;       /* Sequence length */
    long   D;       /* Feature dimension */
    long   M;       /* State dimension */
} MambaScan1DParams;

/* Forward scan */
void mamba_scan1d_forward(MambaScan1DParams *p);

/* Backward scan variants */
typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *dt;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddt;
    long   L;
    long   D;
    long   M;
} MambaScan1DBackwardParams;

void mamba_scan1d_backward(MambaScan1DBackwardParams *p);

/* M=1 optimized backward */
typedef struct {
    float *x;
    float *A_diag;
    float *B;
    float *C;
    float *dt;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddt;
    long   L;
    long   D;
} MambaScan1DBackwardM1Params;

void mamba_scan1d_backward_m1_shared_bc(MambaScan1DBackwardM1Params *p);

#endif
