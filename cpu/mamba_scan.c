#include "scan.h"
#include "mamba_scan.h"

/* ============================================================
 * Mamba Scan CPU — wrappers vers les kernels ASM/C
 * ============================================================ */

void mamba_scan1d_forward(ScanParams *p) {
    scan1d(p);
}

void mamba_scan1d_backward(ScanBackwardParams *p) {
    scan1d_backward(p);
}

void mamba_scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p) {
    scan1d_backward_m1_shared_bc(p);
}

void mamba_scan2d_forward(Scan2DParams *p) {
    scan2d(p);
}
