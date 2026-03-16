#ifndef MAMBA_SCAN_2D_H
#define MAMBA_SCAN_2D_H

/* Types canoniques définis dans scan.h */
#include "scan.h"

/* Alias pour l'API publique k-mamba 2D */
typedef Scan2DParams MambaScan2DParams;

void mamba_scan2d_forward(MambaScan2DParams *p);

#endif
