#ifndef MAMBA_SCAN_2D_H
#define MAMBA_SCAN_2D_H

#include <stddef.h>

/* ============================================================
 * Mamba-specific Scan 2D (Wavefront)
 * ============================================================ */

typedef struct {
    float *x;        /* Input [d1 * d2 * D] */
    float *A1;       /* First diagonal matrix [D * M] */
    float *A2;       /* Second diagonal matrix [D * M] */
    float *B;        /* Input matrix [d1 * d2 * D * M] */
    float *C;        /* Output matrix [d1 * d2 * D * M] */
    float *dt1;      /* Delta for first dimension [d1 * d2 * D] */
    float *dt2;      /* Delta for second dimension [d1 * d2 * D] */
    float *h;        /* Hidden states [d1 * d2 * D * M] */
    float *y;        /* Output [d1 * d2 * D] */
    long   d1;       /* First spatial dimension */
    long   d2;       /* Second spatial dimension */
    long   D;        /* Feature dimension */
    long   M;        /* State dimension */
} MambaScan2DParams;

/* Wavefront scan 2D */
void mamba_scan2d_forward(MambaScan2DParams *p);

#endif
