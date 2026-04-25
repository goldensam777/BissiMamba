#include "scan_nd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef KMAMBA_DEBUG_SCAN
#define DBG_SCAN(...) do {} while(0)
#define DBG_SCAN_FIRST_POINT(...) do {} while(0)
#define DBG_SCAN_FIRST_POINT_D0(...) do {} while(0)
#define DBG_SCAN_FIRST_POINT_D0_M0(...) do {} while(0)
#else
#define DBG_SCAN(...) fprintf(stderr, __VA_ARGS__)
#define DBG_SCAN_FIRST_POINT(...) do { if ((level == 0 || level == 1) && point == 0) fprintf(stderr, __VA_ARGS__); } while(0)
#define DBG_SCAN_FIRST_POINT_D0(...) do { if ((level == 0 || level == 1) && point == 0 && d == 0) fprintf(stderr, __VA_ARGS__); } while(0)
#define DBG_SCAN_FIRST_POINT_D0_M0(...) do { if ((level == 0 || level == 1) && point == 0 && d == 0 && m == 0) fprintf(stderr, __VA_ARGS__); } while(0)
#endif

#include "km_memory_pool.h"
#include "scan.h"
#include "wavefront_plan.h"
#include "km_topology.h"
#include "kmamba_cuda_utils.h"

int scannd_validate(const ScanNDParams *p) {
    long total_points;

    if (!p || !p->dims) return 0;
    if (p->ndims <= 0 || p->D <= 0 || p->M <= 0) return 0;
    if (!p->x || !p->A || !p->B || !p->C || !p->delta || !p->h || !p->y) return 0;
    if (!wavefront_nd_validate_dims(p->dims, p->ndims)) return 0;

    total_points = wavefront_nd_total_points(p->dims, p->ndims);
    if (total_points <= 0) return 0;

    return 1;
}

int scannd_ref_with_plan(ScanNDParams *p, const KMWavefrontPlan *plan) {
    long total_points;
    long *strides;
    const float default_lambda = p->default_lambda;
    const float a_log_min = p->a_log_min;

    if (!scannd_validate(p)) return -1;
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return -1;

    total_points = plan->total_points;
    if (total_points <= 0) return -1;

    /* Utilise memory pool thread-local pour éviter malloc répété */
    KMMemoryPool *pool = km_memory_pool_threadlocal();
    strides = (long *)km_pool_alloc(pool, (size_t)p->ndims * sizeof(long));
    if (!strides) {
        km_pool_free(pool, strides);
        return -1;
    }
    if (!km_make_row_major_strides(p->dims, p->ndims, strides)) {
        km_pool_free(pool, strides);
        return -1;
    }

    /* Mamba-3: Allocate buffers for exp-trapezoidal state via pool */
    float *h_rot = NULL, *Bu_prev = NULL, *Bu_cur = NULL;
    /* Per-state-elem buffers for R(θ) rotation and Bu tracking */
    size_t h_rot_size = (size_t)p->D * p->M * sizeof(float);
    size_t bu_prev_size = (size_t)total_points * p->D * p->M * sizeof(float);
    size_t bu_cur_size = (size_t)p->D * p->M * sizeof(float);

    /* Allocate h_rot only if theta rotation is needed */
    if (p->theta) {
        h_rot = (float *)km_pool_alloc(pool, h_rot_size);
    }
    /* Note: calloc pas supporté par pool, on alloue puis memset */
    Bu_prev = (float *)km_pool_alloc(pool, bu_prev_size);
    Bu_cur = (float *)km_pool_alloc(pool, bu_cur_size);
    if (!Bu_prev || !Bu_cur) {
        km_pool_free(pool, strides);
        km_pool_free(pool, h_rot); km_pool_free(pool, Bu_prev); km_pool_free(pool, Bu_cur);
        return -1;
    }
    /* Zero Bu_prev */
    memset(Bu_prev, 0, bu_prev_size);

    for (long level = 0; level <= plan->max_level; level++) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) {
            km_pool_free(pool, strides);
            km_pool_free(pool, h_rot); km_pool_free(pool, Bu_prev); km_pool_free(pool, Bu_cur);
            return -1;
        }

        /* Wavefront: tous les points d'un niveau sont indépendants → parallélisme */
        int level_error = 0;
#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
#pragma omp parallel for schedule(static) reduction(||:level_error)
#endif
        for (long point = 0; point < level_size; point++) {
            int tid = 0;
#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
            tid = omp_get_thread_num();
#endif
            long offset = level_offsets[point];
            if (offset < 0 || offset >= total_points) {
                level_error = 1;
                continue;
            }

            long *idx_local = plan->coords_thread + (size_t)tid * (size_t)p->ndims;

            km_unravel_index(offset, p->dims, strides, p->ndims, idx_local);

            /* Mamba-3: lambda per point for exp-trapezoidal */
            float lambda_n = p->lambda ? p->lambda[offset] : default_lambda;

            for (long d = 0; d < p->D; d++) {
                float x_val = p->x[offset * p->D + d];
                float dt_bar = 0.0f;
                float y_acc = 0.0f;

                for (long axis = 0; axis < p->ndims; axis++) {
                    dt_bar += p->delta[(axis * total_points + offset) * p->D + d];
                }
                /* Note: pas de division par ndims (conforme Eq 10 du papier) */

                for (long m = 0; m < p->M; m++) {
                    long dm = d * p->M + m;
                    long pdm = offset * p->D * p->M + dm;

                    /* Compute Bu_t = B_t * x_t */
                    float bu_t = p->B[pdm] * x_val;
                    if (Bu_cur) Bu_cur[dm] = bu_t;

                    /* Mamba-3 exp-trapezoidal formula */
                    float h_new;
                    float bu_prev_acc = 0.0f;
                    float decay_acc = 0.0f;
                    float h_decay_acc = 0.0f;
                    int n_pred = 0;

                    /* Single pass: compute all alphas, accumulate Bu and h contributions */
                    for (long axis = 0; axis < p->ndims; axis++) {
                        if (idx_local[axis] > 0) {
                            long prev_offset = offset - strides[axis];
                            float dt_axis = p->delta[(axis * total_points + offset) * p->D + d];
                            float a_val = p->A[(axis * p->D + d) * p->M + m];
                            if (p->use_a_log_clamp && a_val > a_log_min) a_val = a_log_min;
                            float alpha = km_scan_exp(dt_axis * a_val, p->use_fast_exp);
                            long prev_pdm = prev_offset * p->D * p->M + dm;

                            bu_prev_acc += Bu_prev[prev_pdm];
                            decay_acc += alpha;
                            /* Use h_rot if available, else use h directly */
                            float h_prev = h_rot ? h_rot[prev_pdm] : p->h[prev_pdm];
                            h_decay_acc += alpha * h_prev;
                            n_pred++;
                        }
                    }

                    float bu_prev_avg = (n_pred > 0) ? (bu_prev_acc / n_pred) : 0.0f;
                    float alpha_avg = (n_pred > 0) ? (decay_acc / n_pred) : 0.0f;
                    float beta = (1.0f - lambda_n) * dt_bar * alpha_avg;
                    float gamma = lambda_n * dt_bar;

                    h_new = bu_prev_avg * beta + bu_t * gamma + h_decay_acc;
                    Bu_prev[pdm] = bu_t;  /* Store for next level */

                    p->h[pdm] = h_new;
                    y_acc += p->C[pdm] * h_new;
                }

                p->y[offset * p->D + d] = y_acc;
            }

            /* Mamba-3: Apply R(θ) rotation to h states for next level (per channel pairs) */
            if (p->theta && h_rot) {
                for (long d = 0; d + 1 < p->D; d += 2) {
                    float th = p->theta[d >> 1];
                    float cv = cosf(th), sv = sinf(th);
                    for (long m = 0; m < p->M; m++) {
                        long pdm0 = offset * p->D * p->M + d * p->M + m;
                        long pdm1 = offset * p->D * p->M + (d + 1) * p->M + m;
                        float h0 = p->h[pdm0];
                        float h1 = p->h[pdm1];
                        h_rot[pdm0] = cv * h0 - sv * h1;
                        h_rot[pdm1] = sv * h0 + cv * h1;
                    }
                }
                /* Handle odd D */
                if (p->D & 1) {
                    for (long m = 0; m < p->M; m++) {
                        long pdm = offset * p->D * p->M + (p->D - 1) * p->M + m;
                        h_rot[pdm] = p->h[pdm];
                    }
                }
            }
        }
        /* Vérification erreur après la boucle parallèle */
        if (level_error) {
            km_pool_free(pool, strides);
            km_pool_free(pool, h_rot); km_pool_free(pool, Bu_prev); km_pool_free(pool, Bu_cur);
            return -1;
        }
    }

    /* Libère vers le pool pour réutilisation (pas de free système) */
    km_pool_free(pool, strides);
    km_pool_free(pool, h_rot);
    km_pool_free(pool, Bu_prev);
    km_pool_free(pool, Bu_cur);
    return 0;
}

int scannd_ref(ScanNDParams *p) {
    KMWavefrontPlan *plan;
    int rc;

    if (!scannd_validate(p)) return -1;

    plan = km_wavefront_plan_create(p->dims, p->ndims,
                                    (p->max_state > 0) ? p->max_state : p->M);
    if (!plan) return -1;

    rc = scannd_ref_with_plan(p, plan);
    km_wavefront_plan_free(plan);
    return rc;
}

int scannd(ScanNDParams *p) {
    if (!scannd_validate(p)) return -1;

    /* Automatic GPU dispatch if available */
    KMAMBA_AUTO_BACKEND();

#ifdef KMAMBA_BUILD_CUDA
    if (backend == KMAMBA_BACKEND_GPU) {
        /* Try GPU implementation first */
        if (om_scannd_forward(p) == 0) {
            return 0; /* GPU success */
        }
        /* Fall back to CPU on GPU failure */
    }
#endif

    /* CPU implementation - unified wavefront C implementation for all dimensions */
    return scannd_ref(p);
}
