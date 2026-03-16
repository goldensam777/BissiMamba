#ifndef MAMBA_SCAN_H
#define MAMBA_SCAN_H

/* Types canoniques définis dans scan.h */
#include "scan.h"
#include "mamba_scan_2d.h"

/* Aliases pour l'API publique k-mamba */
typedef ScanParams              MambaScan1DParams;
typedef ScanBackwardParams      MambaScan1DBackwardParams;
typedef ScanBackwardSharedParams MambaScan1DBackwardM1Params;

/* CPU forward/backward */
void mamba_scan1d_forward(MambaScan1DParams *p);
void mamba_scan1d_backward(MambaScan1DBackwardParams *p);
void mamba_scan1d_backward_m1_shared_bc(MambaScan1DBackwardM1Params *p);

#endif
