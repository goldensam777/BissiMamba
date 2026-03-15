#include "mamba_scan.h"
#include <string.h>

/* ============================================================
 * Mamba Scan 1D CPU Implementation
 * ============================================================ */

void mamba_scan1d_forward(MambaScan1DParams *p) {
    // Externally defined in scan1d.asm
    extern void scan1d(void *params);
    scan1d(p);
}

void mamba_scan1d_backward(MambaScan1DBackwardParams *p) {
    // Externally defined in scan1d_backward_m1_shared_bc.asm
    extern void scan1d_backward_m1_shared_bc(void *params);
    scan1d_backward_m1_shared_bc(p);
}

void mamba_scan1d_backward_m1_shared_bc(MambaScan1DBackwardM1Params *p) {
    // Externally defined in scan1d_backward_m1_shared_bc.asm
    extern void scan1d_backward_m1_shared_bc(void *params);
    scan1d_backward_m1_shared_bc(p);
}
