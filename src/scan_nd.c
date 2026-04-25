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

static inline float km_scan_exp(float x) {
#ifdef KMAMBA_FAST_EXP_APPROX
    /* Fast 3rd-order approximation around 0, clamped for stability. */
    if (x > 8.0f) x = 8.0f;
    if (x < -8.0f) x = -8.0f;
    float x2 = x * x;
    float x3 = x2 * x;
    return 1.0f + x + 0.5f * x2 + (1.0f / 6.0f) * x3;
#else
    return expf(x);
#endif
}

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

    DBG_SCAN("DEBUG scannd: entering scannd_ref_with_plan\n");
    
    if (!scannd_validate(p)) return -1;
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return -1;

    total_points = plan->total_points;
    DBG_SCAN("DEBUG scannd: total_points=%ld\n", total_points);
    if (total_points <= 0) return -1;

    /* Utilise memory pool thread-local pour éviter malloc répété */
    KMMemoryPool *pool = km_memory_pool_threadlocal();
    DBG_SCAN("DEBUG scannd: got memory pool\n");
    strides = (long *)km_pool_alloc(pool, (size_t)p->ndims * sizeof(long));
    DBG_SCAN("DEBUG scannd: allocated strides=%p\n", (void*)strides);
    if (!strides) {
        km_pool_free(pool, strides);
        return -1;
    }
    if (!km_make_row_major_strides(p->dims, p->ndims, strides)) {
        km_pool_free(pool, strides);
        return -1;
    }
    DBG_SCAN("DEBUG scannd: strides created\n");

    /* Mamba-3: Allocate buffers for exp-trapezoidal state via pool */
    float *h_rot = NULL, *Bu_prev = NULL, *Bu_cur = NULL;
    DBG_SCAN("DEBUG scannd: p->lambda=%p\n", (void*)p->lambda);
    if (p->lambda) {
        DBG_SCAN("DEBUG scannd: allocating Mamba-3 buffers, D=%ld, M=%ld\n", p->D, p->M);
        /* Per-state-elem buffers for R(θ) rotation and Bu tracking */
        size_t h_rot_size = (size_t)p->D * p->M * sizeof(float);
        size_t bu_prev_size = (size_t)total_points * p->D * p->M * sizeof(float);
        size_t bu_cur_size = (size_t)p->D * p->M * sizeof(float);
        DBG_SCAN("DEBUG scannd: h_rot_size=%zu, bu_prev_size=%zu, bu_cur_size=%zu\n",
                h_rot_size, bu_prev_size, bu_cur_size);
        
        h_rot = (float *)km_pool_alloc(pool, h_rot_size);
        DBG_SCAN("DEBUG scannd: h_rot=%p\n", (void*)h_rot);
        /* Note: calloc pas supporté par pool, on alloue puis memset */
        Bu_prev = (float *)km_pool_alloc(pool, bu_prev_size);
        DBG_SCAN("DEBUG scannd: Bu_prev=%p\n", (void*)Bu_prev);
        Bu_cur = (float *)km_pool_alloc(pool, bu_cur_size);
        DBG_SCAN("DEBUG scannd: Bu_cur=%p\n", (void*)Bu_cur);
        if (!h_rot || !Bu_prev || !Bu_cur) {
            DBG_SCAN("DEBUG scannd: buffer allocation failed!\n");
            km_pool_free(pool, strides);
            km_pool_free(pool, h_rot); km_pool_free(pool, Bu_prev); km_pool_free(pool, Bu_cur);
            return -1;
        }
        DBG_SCAN("DEBUG scannd: zeroing Bu_prev...\n");
        /* Zero Bu_prev */
        memset(Bu_prev, 0, bu_prev_size);
        DBG_SCAN("DEBUG scannd: buffers allocated successfully\n");
    }

    DBG_SCAN("DEBUG scannd: starting wavefront loop, max_level=%ld\n", plan->max_level);
    for (long level = 0; level <= plan->max_level; level++) {
        DBG_SCAN("DEBUG scannd: level=%ld\n", level);
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        DBG_SCAN("DEBUG scannd: level_offsets=%p, level_size=%ld\n", (void*)level_offsets, level_size);
        if (!level_offsets || level_size < 0) {
            km_pool_free(pool, strides);
            km_pool_free(pool, h_rot); km_pool_free(pool, Bu_prev); km_pool_free(pool, Bu_cur);
            return -1;
        }

        /* Wavefront: tous les points d'un niveau sont indépendants → parallélisme */
        int level_error = 0;
        DBG_SCAN("DEBUG scannd: starting loop (OpenMP %s)\n",
#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
                "enabled"
#else
                "disabled"
#endif
                );
#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
#pragma omp parallel for schedule(static) reduction(||:level_error)
#endif
        for (long point = 0; point < level_size; point++) {
            DBG_SCAN_FIRST_POINT("DEBUG scannd: level=%ld, first iteration, point=0\n", level);
            
            long offset = level_offsets[point];
            DBG_SCAN_FIRST_POINT("DEBUG scannd: offset=%ld\n", offset);
            
            if (offset < 0 || offset >= total_points) {
                DBG_SCAN_FIRST_POINT("DEBUG scannd: invalid offset!\n");
                level_error = 1;
                continue;
            }

            long idx_local[KMAMBA_MAX_NDIMS];

            km_unravel_index(offset, p->dims, strides, p->ndims, idx_local);

            DBG_SCAN_FIRST_POINT("DEBUG scannd: idx_local=[%ld,%ld]\n", idx_local[0], idx_local[1]);

            /* Mamba-3: lambda per point for exp-trapezoidal */
            DBG_SCAN_FIRST_POINT("DEBUG scannd: accessing lambda[%ld]...\n", offset);
            float lambda_n = p->lambda ? p->lambda[offset] : 0.5f;
            DBG_SCAN_FIRST_POINT("DEBUG scannd: lambda_n=%f\n", lambda_n);

            for (long d = 0; d < p->D; d++) {
                DBG_SCAN_FIRST_POINT_D0("DEBUG scannd: accessing x[%ld * %ld + %ld]...\n", offset, p->D, d);
                float x_val = p->x[offset * p->D + d];
                DBG_SCAN_FIRST_POINT_D0("DEBUG scannd: x_val=%f\n", x_val);
                float dt_bar = 0.0f;
                float y_acc = 0.0f;

                DBG_SCAN_FIRST_POINT_D0("DEBUG scannd: computing dt_bar...\n");
                for (long axis = 0; axis < p->ndims; axis++) {
                    dt_bar += p->delta[(axis * total_points + offset) * p->D + d];
                }
                /* Note: pas de division par ndims (conforme Eq 10 du papier) */
                DBG_SCAN_FIRST_POINT_D0("DEBUG scannd: dt_bar=%f\n", dt_bar);
                DBG_SCAN_FIRST_POINT_D0("DEBUG scannd: starting m loop, M=%ld\n", p->M);

                for (long m = 0; m < p->M; m++) {
                    long dm = d * p->M + m;
                    long pdm = offset * p->D * p->M + dm;

                    /* Compute Bu_t = B_t * x_t */
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: accessing p->B[%ld], pdm=%ld\n", pdm, pdm);
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: p->B=%p\n", (void*)p->B);
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: p->B[0]=%f, p->B[%ld]=%f\n",
                            p->B[0], pdm, p->B[pdm]);
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: computing bu_t = p->B[%ld] * x_val...\n", pdm);
                    float bu_t = p->B[pdm] * x_val;
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: bu_t = %f * %f = %f\n", p->B[pdm], x_val, bu_t);
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: accessing Bu_cur[%ld]...\n", dm);
                    if (Bu_cur) Bu_cur[dm] = bu_t;
                    DBG_SCAN_FIRST_POINT_D0_M0("DEBUG scannd: Bu_cur updated, checking Mamba-3...\n");

                    /* Mamba-3 exp-trapezoidal: compute alphas once, reuse for Bu and h decay */
                    float h_new;
                    if (p->lambda && Bu_prev && Bu_cur) {
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
                                if (a_val > -1e-5f) a_val = -1e-5f;
                                float alpha = km_scan_exp(dt_axis * a_val);
                                long prev_pdm = prev_offset * p->D * p->M + dm;

                                bu_prev_acc += Bu_prev[prev_pdm];
                                decay_acc += alpha;
                                /* Use h_rot if available, else use h directly */
                                float h_prev = h_rot ? h_rot[prev_pdm] : p->h[prev_pdm];
                                h_decay_acc += alpha * h_prev;
                                n_pred++;
                            }
                        }

                        /* Mamba-3 exp-trapezoidal formula */
                        float bu_prev_avg = (n_pred > 0) ? (bu_prev_acc / n_pred) : 0.0f;
                        float alpha_avg = (n_pred > 0) ? (decay_acc / n_pred) : 0.0f;
                        float beta = (1.0f - lambda_n) * dt_bar * alpha_avg;
                        float gamma = lambda_n * dt_bar;

                        h_new = bu_prev_avg * beta + bu_t * gamma + h_decay_acc;
                        Bu_prev[pdm] = bu_t;  /* Store for next level */
                    } else {
                        /* Original scan_nd formula (backward compatible) */
                        h_new = dt_bar * bu_t;
                        for (long axis = 0; axis < p->ndims; axis++) {
                            if (idx_local[axis] > 0) {
                                long prev_offset = offset - strides[axis];
                                float dt_axis = p->delta[(axis * total_points + offset) * p->D + d];
                                float a_val = p->A[(axis * p->D + d) * p->M + m];
                                float decay = km_scan_exp(dt_axis * a_val);
                                long prev_pdm = prev_offset * p->D * p->M + dm;
                                h_new += decay * p->h[prev_pdm];
                            }
                        }
                    }

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

    plan = km_wavefront_plan_create(p->dims, p->ndims);
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

    /* CPU implementation - optimized paths for common dimensions */
    if (p->ndims == 1) {
        ScanParams p1 = {
            .x = (float *)p->x,
            .A = (float *)p->A,
            .B = (float *)p->B,
            .C = (float *)p->C,
            .delta = (float *)p->delta,
            .h = p->h,
            .y = p->y,
            .L = p->dims[0],
            .D = p->D,
            .M = p->M
        };
        scan1d(&p1);
        return 0;
    }

    if (p->ndims == 2) {
        long total_points = wavefront_nd_total_points(p->dims, p->ndims);
        Scan2DParams p2;

        if (total_points <= 0) return -1;

        p2.x = (float *)p->x;
        p2.A1 = (float *)p->A;
        p2.A2 = (float *)(p->A + p->D * p->M);
        p2.B = (float *)p->B;
        p2.C = (float *)p->C;
        p2.delta1 = (float *)p->delta;
        p2.delta2 = (float *)(p->delta + total_points * p->D);
        p2.h = p->h;
        p2.y = p->y;
        p2.d1 = p->dims[0];
        p2.d2 = p->dims[1];
        p2.D = p->D;
        p2.M = p->M;

        scan2d(&p2);
        return 0;
    }

    return scannd_ref(p);
}
