/* ============================================================
 * convnd.c — Convolution ND native wavefront parallèle
 *
 * Architecture :
 *   Convolution ND dense avec ordonnancement wavefront.
 *   Noyau complet K^N (pas de séparabilité).
 *   Parallélisme intra-niveau via OpenMP (optionnel).
 *
 * Forward  : wavefront niveau par niveau, parallèle intra-niveau
 * Backward : wavefront inverse, accumulation gradients
 *
 * Unification :
 *   - Plus de distinction séparable/full
 *   - Une seule API convnd() avec wavefront natif
 *   - Parallélisme OpenMP optionnel (#ifdef _OPENMP)
 * ============================================================ */

#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "wavefront_plan.h"
#include "km_topology.h"
#include "convnd.h"
#include "kmamba_cuda_utils.h"

/* ============================================================
 * Forward declarations for separable convolution
 * ============================================================ */

void convnd_separable_forward_wavefront(ConvNDSeparableParams *p, KMWavefrontPlan **plans_per_axis);
void convnd_separable_backward_wavefront(
    ConvNDSeparableParams *forward_p,
    ConvNDSeparableBackwardParams *grad_p,
    float *dy,
    KMWavefrontPlan **plans_per_axis);

/* ============================================================
 * Helpers
 * ============================================================ */

static long product(const long *arr, long n) {
    long p = 1;
    for (long i = 0; i < n; i++) p *= arr[i];
    return p;
}

static long convnd_power_long(long base, long exp) {
    long out = 1;
    for (long i = 0; i < exp; i++) out *= base;
    return out;
}

static void convnd_unravel_index(long linear, const long *dims, long ndims, long *coords) {
    for (long axis = ndims; axis-- > 0;) {
        coords[axis] = linear % dims[axis];
        linear /= dims[axis];
    }
}

static void convnd_make_row_major_strides(const long *dims, long ndims, long *strides) {
    long stride = 1;
    for (long axis = ndims; axis-- > 0;) {
        strides[axis] = stride;
        stride *= dims[axis];
    }
}

long convnd_kernel_volume(long ndims, long K) {
    if (ndims <= 0 || K <= 0) return 0;
    return convnd_power_long(K, ndims);
}

/* ============================================================
 * CONVND SÉPARABLE MAMBA-CLASSIC
 * ============================================================
 * Architecture: Cascade de convolutions 1D par axe avec wavefront
 * Chaque étape séparable utilise son propre ordonnancement wavefront
 * pour parallélisme intra-niveau maximal.
 *
 * kernel_axes[axis] = noyau 1D de taille K pour l'axe (stride entre canaux = 1)
 * On applique: input -> conv_axis0 -> temp -> conv_axis1 -> ... -> output
 */

/* Structures ConvNDSeparableParams et ConvNDSeparableBackwardParams 
 * sont définies dans convnd.h */

/* Conv 1D le long d'un axe spécifique avec wavefront */
static void convnd_separable_1d_wavefront(
    float *input,
    float *output,
    const float *kernel_1d,  /* [K] */
    long axis,               /* axe de convolution (0..ndims-1) */
    const long *pad_left,
    const long *pad_right,
    long *dims,
    long ndims,
    long D,
    long K,
    KMWavefrontPlan *plan)
{
    long spatial_total = product(dims, ndims);
    long *strides = (long *)malloc((size_t)ndims * sizeof(long));
    if (!strides) return;
    convnd_make_row_major_strides(dims, ndims, strides);

    long axis_stride = strides[axis];
    long default_pad = (K - 1) / 2;
    long pad_left_axis = pad_left ? pad_left[axis] : default_pad;
    long pad_right_axis = pad_right ? pad_right[axis] : default_pad;
    (void)pad_right_axis; /* reserved for asymmetry checks; left shift drives index mapping */

    /* Wavefront: chaque niveau = positions indépendantes le long de l'axe */
    for (long level = 0; level <= plan->max_level; level++) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long point = 0; point < level_size; point++) {
            long linear = level_offsets[point];
            int tid = 0;
#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
            tid = omp_get_thread_num();
#endif
            long *coords = plan->coords_thread + (size_t)tid * (size_t)ndims;

            /* Unravel index pour trouver coordonnées */
            long tmp = linear;
            for (long d = ndims; d-- > 0;) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            long coord_axis = coords[axis];
            for (long c = 0; c < D; c++) {
                float sum = 0.0f;
                for (long k = 0; k < K; k++) {
                    long src_coord_axis = coord_axis + k - pad_left_axis;
                    if (src_coord_axis < 0 || src_coord_axis >= dims[axis]) continue;
                    {
                        long offset_k = (src_coord_axis - coord_axis) * axis_stride;
                        long src_idx = linear + offset_k;
                        sum += input[src_idx * D + c] * kernel_1d[k];
                    }
                }
                output[linear * D + c] = sum;
            }
        }
    }

    free(strides);
}

/* Forward séparable: cascade de conv 1D avec wavefront par étape */
void convnd_separable_forward_wavefront(ConvNDSeparableParams *p, KMWavefrontPlan **plans_per_axis) {
    long spatial_total;
    float *temp_buffer = NULL;
    float *current_input, *current_output;

    if (!p || !p->input || !p->output || !p->kernel_axes ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return;
    }

    spatial_total = product(p->dims, p->ndims);

    /* Buffer temporaire pour ping-pong entre étapes */
    temp_buffer = (float *)malloc((size_t)spatial_total * p->D * sizeof(float));
    if (!temp_buffer) return;

    current_input = p->input;
    current_output = temp_buffer;

    /* Cascade: appliquer conv 1D axe par axe */
    for (long axis = 0; axis < p->ndims; axis++) {
        KMWavefrontPlan *plan = plans_per_axis ? plans_per_axis[axis] : NULL;
        if (!plan) {
            /* Fallback: créer plan si non fourni */
            plan = km_wavefront_plan_create(p->dims, p->ndims, 0);
            if (!plan) { free(temp_buffer); return; }
        }

        convnd_separable_1d_wavefront(
            current_input,
            current_output,
            p->kernel_axes[axis],
            axis,
            p->pad_left,
            p->pad_right,
            p->dims,
            p->ndims,
            p->D,
            p->K,
            plan
        );

        /* Swap pour étape suivante */
        if (axis == 0) {
            current_input = temp_buffer;
            current_output = (p->ndims > 1) ? p->output : temp_buffer;
        } else if (axis == p->ndims - 1) {
            /* Dernière étape: vers output final */
            if (current_output != p->output) {
                memcpy(p->output, current_output, (size_t)spatial_total * p->D * sizeof(float));
            }
        } else {
            /* Étapes intermédiaires: ping-pong */
            float *tmp = current_input;
            current_input = current_output;
            current_output = (tmp == temp_buffer) ? p->output : temp_buffer;
        }

        if (!plans_per_axis || !plans_per_axis[axis]) {
            km_wavefront_plan_free(plan);
        }
    }

    /* Ajouter bias si présent */
    if (p->bias) {
        for (long i = 0; i < spatial_total; i++) {
            for (long c = 0; c < p->D; c++) {
                p->output[i * p->D + c] += p->bias[c];
            }
        }
    }

    free(temp_buffer);
}

/* ============================================================
 * Backward séparable: gradient dinput et dkernel par axe
 * ============================================================ */

/* Backward 1D le long d'un axe: calcule dinput et dkernel */
static void convnd_separable_1d_backward_wavefront(
    float *input,          /* forward input [spatial*D] */
    float *dy,             /* gradient from output [spatial*D] */
    float *kernel_1d,      /* [K] */
    float *dinput_out,     /* grad w.r.t input [spatial*D] (accumulate) */
    float *dkernel_out,    /* grad w.r.t kernel [K] (accumulate) */
    long axis,
    const long *pad_left,
    const long *pad_right,
    long *dims,
    long ndims,
    long D,
    long K,
    KMWavefrontPlan *plan)
{
    long spatial_total = product(dims, ndims);
    long *strides = (long *)malloc((size_t)ndims * sizeof(long));
    if (!strides) return;
    convnd_make_row_major_strides(dims, ndims, strides);

    long axis_stride = strides[axis];
    long default_pad = (K - 1) / 2;
    long pad_left_axis = pad_left ? pad_left[axis] : default_pad;
    long pad_right_axis = pad_right ? pad_right[axis] : default_pad;
    (void)pad_right_axis;

    /* Init dkernel accumulation */
    memset(dkernel_out, 0, (size_t)K * sizeof(float));

    /* Wavefront backward: iterate in reverse level order */
    for (long level = plan->max_level; level >= 0; level--) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long point = 0; point < level_size; point++) {
            long linear = level_offsets[point];
            int tid = 0;
#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
            tid = omp_get_thread_num();
#endif
            long *coords = plan->coords_thread + (size_t)tid * (size_t)ndims;

            long tmp = linear;
            for (long d = ndims; d-- > 0;) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            long coord_axis = coords[axis];

            for (long c = 0; c < D; c++) {
                float grad = dy[linear * D + c];

                /* Accumulate dkernel */
                for (long k = 0; k < K; k++) {
                    long src_coord_axis = coord_axis + k - pad_left_axis;
                    if (src_coord_axis < 0 || src_coord_axis >= dims[axis]) continue;
                    {
                        long offset_k = (src_coord_axis - coord_axis) * axis_stride;
                        long src_idx = linear + offset_k;
                        dkernel_out[k] += grad * input[src_idx * D + c];
                        dinput_out[src_idx * D + c] += grad * kernel_1d[k];
                    }
                }
            }
        }
    }

    free(strides);
}

/* Backward séparable: cascade inverse pour gradients */
void convnd_separable_backward_wavefront(
    ConvNDSeparableParams *forward_p,  /* params from forward pass */
    ConvNDSeparableBackwardParams *grad_p,
    float *dy,                         /* gradient w.r.t output [spatial*D] */
    KMWavefrontPlan **plans_per_axis)
{
    long spatial_total;
    float *temp_grad = NULL;  /* ping-pong buffer for gradient flow */

    if (!forward_p || !grad_p || !dy) return;
    if (!forward_p->dims || forward_p->ndims <= 0) return;

    spatial_total = product(forward_p->dims, forward_p->ndims);

    /* Init dinput accumulator */
    memset(grad_p->dinput, 0, (size_t)spatial_total * grad_p->D * sizeof(float));

    temp_grad = (float *)malloc((size_t)spatial_total * grad_p->D * sizeof(float));
    if (!temp_grad) return;
    memcpy(temp_grad, dy, (size_t)spatial_total * grad_p->D * sizeof(float));

    /* Backward cascade: reverse order of axes */
    for (long axis = forward_p->ndims - 1; axis >= 0; axis--) {
        KMWavefrontPlan *plan = plans_per_axis ? plans_per_axis[axis] : NULL;
        if (!plan) {
            plan = km_wavefront_plan_create(forward_p->dims, forward_p->ndims, 0);
            if (!plan) { free(temp_grad); return; }
        }

        /* Need intermediate dinput accumulator for this axis */
        float *dinput_axis = (axis == 0) ? grad_p->dinput :
            (float *)calloc((size_t)spatial_total * grad_p->D, sizeof(float));
        if (!dinput_axis) { km_wavefront_plan_free(plan); free(temp_grad); return; }

        convnd_separable_1d_backward_wavefront(
            (axis == forward_p->ndims - 1) ? forward_p->input : forward_p->output,
            temp_grad,
            forward_p->kernel_axes[axis],
            dinput_axis,
            grad_p->dkernel_axes[axis],
            axis,
            forward_p->pad_left,
            forward_p->pad_right,
            forward_p->dims,
            forward_p->ndims,
            grad_p->D,
            grad_p->K,
            plan
        );

        /* Pass gradient backward to next axis */
        if (axis > 0) {
            memcpy(temp_grad, dinput_axis, (size_t)spatial_total * grad_p->D * sizeof(float));
            if (dinput_axis != grad_p->dinput) free(dinput_axis);
        }

        if (!plans_per_axis || !plans_per_axis[axis]) {
            km_wavefront_plan_free(plan);
        }
    }

    /* Gradient of bias */
    if (grad_p->dbias) {
        for (long i = 0; i < spatial_total; i++) {
            for (long c = 0; c < grad_p->D; c++) {
                grad_p->dbias[c] += dy[i * grad_p->D + c];
            }
        }
    }

    free(temp_grad);
}

/* ============================================================
 * Forward Wavefront - Parallélisé intra-niveau (DENSE ORIGINAL)
 * ============================================================ */

void convnd_forward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan) {
    long kernel_volume;
    long *spatial_strides;

    if (!p || !p->input || !p->kernel || !p->output ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return;
    }
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return;

    kernel_volume = convnd_kernel_volume(p->ndims, p->K);
    spatial_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!spatial_strides) return;

    convnd_make_row_major_strides(p->dims, p->ndims, spatial_strides);

    for (long level = 0; level <= plan->max_level; level++) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (long point = 0; point < level_size; point++) {
            long out_linear = level_offsets[point];
            long out_coords[p->ndims];

            convnd_unravel_index(out_linear, p->dims, p->ndims, out_coords);

            for (long d = 0; d < p->D; d++) {
                float sum = p->bias ? p->bias[d] : 0.0f;

                for (long kernel_linear = 0; kernel_linear < kernel_volume; kernel_linear++) {
                    long tmp = kernel_linear;
                    long src_linear = 0;
                    int valid = 1;

                    for (long axis = p->ndims; axis-- > 0;) {
                        long k_axis = tmp % p->K;
                        long src_coord;

                        tmp /= p->K;
                        src_coord = out_coords[axis] - (p->K - 1 - k_axis);
                        if (src_coord < 0 || src_coord >= p->dims[axis]) {
                            valid = 0;
                            break;
                        }
                        src_linear += src_coord * spatial_strides[axis];
                    }

                    if (!valid) continue;
                    sum += p->input[src_linear * p->D + d] *
                           p->kernel[kernel_linear * p->D + d];
                }

                p->output[out_linear * p->D + d] = sum;
            }
        }
    }

    free(spatial_strides);
}

/* ============================================================
 * Backward Wavefront
 * ============================================================ */

void convnd_backward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan) {
    long total_spatial;
    long kernel_volume;
    long *spatial_strides;

    if (!p || !p->input || !p->kernel || !p->dy ||
        !p->dinput || !p->dkernel ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return;
    }
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return;

    total_spatial = product(p->dims, p->ndims);
    kernel_volume = convnd_kernel_volume(p->ndims, p->K);
    spatial_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!spatial_strides) return;

    convnd_make_row_major_strides(p->dims, p->ndims, spatial_strides);
    memset(p->dinput, 0, (size_t)(total_spatial * p->D) * sizeof(float));
    memset(p->dkernel, 0, (size_t)(kernel_volume * p->D) * sizeof(float));
    if (p->dbias) memset(p->dbias, 0, (size_t)p->D * sizeof(float));

    for (long level = plan->max_level; level >= 0; level--) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

        for (long point = 0; point < level_size; point++) {
            long out_linear = level_offsets[point];
            long out_coords[p->ndims];

            convnd_unravel_index(out_linear, p->dims, p->ndims, out_coords);

            for (long d = 0; d < p->D; d++) {
                float grad = p->dy[out_linear * p->D + d];

                if (p->dbias) p->dbias[d] += grad;

                for (long kernel_linear = 0; kernel_linear < kernel_volume; kernel_linear++) {
                    long tmp = kernel_linear;
                    long src_linear = 0;
                    int valid = 1;

                    for (long axis = p->ndims; axis-- > 0;) {
                        long k_axis = tmp % p->K;
                        long src_coord;

                        tmp /= p->K;
                        src_coord = out_coords[axis] - (p->K - 1 - k_axis);
                        if (src_coord < 0 || src_coord >= p->dims[axis]) {
                            valid = 0;
                            break;
                        }
                        src_linear += src_coord * spatial_strides[axis];
                    }

                    if (!valid) continue;
                    p->dkernel[kernel_linear * p->D + d] +=
                        grad * p->input[src_linear * p->D + d];
                    p->dinput[src_linear * p->D + d] +=
                        grad * p->kernel[kernel_linear * p->D + d];
                }
            }
        }
    }

    free(spatial_strides);
}

/* ============================================================
 * Entry point unifié — avec dispatch automatique GPU
 * ============================================================ */

void convnd(ConvNDParams *p, ConvNDMode mode) {
    KMWavefrontPlan *plan;

    if (!p || !p->dims || p->ndims <= 0) return;

    /* Automatic GPU dispatch if available */
    KMAMBA_AUTO_BACKEND();

#ifdef KMAMBA_BUILD_CUDA
    if (backend == KMAMBA_BACKEND_GPU) {
        /* Try GPU first */
        int gpu_ok = 1;
        if (mode & CONVND_FORWARD) {
            if (om_convnd_forward(p) != 0) gpu_ok = 0;
        }
        if (gpu_ok && (mode & CONVND_BACKWARD)) {
            if (om_convnd_backward(p) != 0) gpu_ok = 0;
        }
        if (gpu_ok) return; /* GPU success */
        /* Fall back to CPU on GPU failure */
    }
#endif

    /* CPU implementation */
    plan = km_wavefront_plan_create(p->dims, p->ndims, 0);
    if (!plan) return;

    if (mode & CONVND_FORWARD) {
        convnd_forward_wavefront(p, plan);
    }

    if (mode & CONVND_BACKWARD) {
        convnd_backward_wavefront(p, plan);
    }

    km_wavefront_plan_free(plan);
}
