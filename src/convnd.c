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

#include "wavefront_plan.h"
#include "km_topology.h"
#include "convnd.h"
#include "kmamba_cuda_utils.h"

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

/* Structure paramètres pour convolution séparable */
typedef struct {
    float *input;           /* Input [prod(dims), D] */
    float *output;          /* Output [prod(dims), D] */
    float **kernel_axes;    /* [ndims] pointeurs vers noyaux 1D [K] chacun */
    const float *bias;      /* Bias [D] ou NULL */
    long *dims;             /* Shape spatiale [ndims] */
    long ndims;             /* Nombre d'axes */
    long D;                 /* Canaux */
    long K;                 /* Taille noyau par axe */
} ConvNDSeparableParams;

/* Conv 1D le long d'un axe spécifique avec wavefront */
static void convnd_separable_1d_wavefront(
    float *input,
    float *output,
    const float *kernel_1d,  /* [K] */
    long axis,               /* axe de convolution (0..ndims-1) */
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
    long kernel_radius = K / 2;  /* K impair assumé pour causalité centrée */

    /* Wavefront: chaque niveau = positions indépendantes le long de l'axe */
    for (long level = 0; level <= plan->max_level; level++) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long point = 0; point < level_size; point++) {
            long linear = level_offsets[point];
            long coords[KMAMBA_MAX_NDIMS];

            /* Unravel index pour trouver coordonnées */
            long tmp = linear;
            for (long d = ndims; d-- > 0;) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            /* Vérifier que la coordonnée sur l'axe actuel est valide pour conv */
            long coord_axis = coords[axis];
            if (coord_axis < kernel_radius || coord_axis >= dims[axis] - kernel_radius) {
                /* Bord: copie directe ou zero padding selon politique */
                for (long c = 0; c < D; c++) {
                    output[linear * D + c] = input[linear * D + c];
                }
                continue;
            }

            /* Convolution 1D le long de l'axe */
            for (long c = 0; c < D; c++) {
                float sum = 0.0f;
                for (long k = 0; k < K; k++) {
                    long offset_k = (k - kernel_radius) * axis_stride;
                    long src_idx = linear + offset_k;
                    sum += input[src_idx * D + c] * kernel_1d[k];
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
            plan = km_wavefront_plan_create(p->dims, p->ndims);
            if (!plan) { free(temp_buffer); return; }
        }

        convnd_separable_1d_wavefront(
            current_input,
            current_output,
            p->kernel_axes[axis],
            axis,
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
    plan = km_wavefront_plan_create(p->dims, p->ndims);
    if (!plan) return;

    if (mode & CONVND_FORWARD) {
        convnd_forward_wavefront(p, plan);
    }

    if (mode & CONVND_BACKWARD) {
        convnd_backward_wavefront(p, plan);
    }

    km_wavefront_plan_free(plan);
}
