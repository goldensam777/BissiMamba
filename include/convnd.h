/* convnd.h — Convolution ND native wavefront */

#ifndef CONVND_H
#define CONVND_H

#include <stddef.h>
#include "wavefront_plan.h"

/* Convolution ND params - version wavefront unifiée */
typedef struct {
    float *input;           /* Input tensor [prod(dims), D] */
    const float *kernel;    /* Full kernel [K^ndims, D] */
    const float *bias;      /* Bias [D] or NULL */
    float *output;          /* Output tensor [prod(dims), D] */
    float *dy;              /* Gradient w.r.t. output (for backward) */
    float *dinput;          /* Gradient w.r.t. input */
    float *dkernel;         /* Gradient w.r.t. kernel */
    float *dbias;           /* Gradient w.r.t. bias */
    const long *dims;       /* Spatial shape [ndims] */
    long ndims;             /* Number of spatial dimensions */
    long D;                 /* Depth/channels */
    long K;                 /* Kernel size along every axis */
} ConvNDParams;

typedef enum {
    CONVND_FORWARD = 1,     /* Forward pass only */
    CONVND_BACKWARD = 2,    /* Backward pass only */
    CONVND_COMPLETE = 3     /* Forward + Backward */
} ConvNDMode;

/* Kernel volume K^ndims */
long convnd_kernel_volume(long ndims, long K);

/* Forward pass with wavefront - parallélisé intra-niveau */
void convnd_forward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan);

/* Backward pass with wavefront */
void convnd_backward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan);

/* Unified entry point */
void convnd(ConvNDParams *p, ConvNDMode mode);

/* ============================================================================
 * SEPARABLE CONVOLUTION ND — Mamba Classic Style
 * ============================================================================ */

typedef struct {
    float *input;           /* Input tensor [prod(dims), D] */
    float *output;          /* Output tensor [prod(dims), D] */
    float **kernel_axes;    /* [ndims] pointers to 1D kernels [K] each */
    const float *bias;      /* Bias [D] or NULL */
    long *dims;             /* Spatial shape [ndims] */
    long ndims;             /* Number of spatial dimensions */
    long D;                 /* Depth/channels */
    long K;                 /* Kernel size along every axis */
} ConvNDSeparableParams;

typedef struct {
    float *dinput;          /* Grad w.r.t input [prod(dims), D] */
    float **dkernel_axes;   /* [ndims] grads w.r.t kernels [K] each */
    float *dbias;           /* Grad w.r.t bias [D] or NULL */
    long *dims;             /* Spatial shape [ndims] */
    long ndims;             /* Number of spatial dimensions */
    long D;                 /* Depth/channels */
    long K;                 /* Kernel size along every axis */
} ConvNDSeparableBackwardParams;

typedef enum {
    CONVND_SEPARABLE_FORWARD = 1,
    CONVND_SEPARABLE_BACKWARD = 2,
    CONVND_SEPARABLE_COMPLETE = 3
} ConvNDSeparableMode;

/* Forward pass — cascade of 1D convolutions per axis with wavefront */
void convnd_separable_forward_wavefront(ConvNDSeparableParams *p, KMWavefrontPlan **plans_per_axis);

/* Backward pass — reverse cascade for gradients */
void convnd_separable_backward_wavefront(
    ConvNDSeparableParams *forward_p,
    ConvNDSeparableBackwardParams *grad_p,
    float *dy,
    KMWavefrontPlan **plans_per_axis);

/* Unified entry point for separable convolution */
void convnd_separable(ConvNDSeparableParams *p, ConvNDSeparableMode mode, KMWavefrontPlan **plans_per_axis);

/* ============================================================================
 * CUDA Backend (optional — compile with nvcc)
 * ============================================================================ */
#ifdef __CUDACC__
/* Forward pass on GPU. Returns 0 on success, -1 on error. */
int om_convnd_forward(ConvNDParams *p);

/* Backward pass on GPU. Returns 0 on success, -1 on error. */
int om_convnd_backward(ConvNDParams *p);

/* Separable forward pass on GPU. Returns 0 on success, -1 on error. */
int om_convnd_separable_forward(ConvNDSeparableParams *p);
#endif

#endif
