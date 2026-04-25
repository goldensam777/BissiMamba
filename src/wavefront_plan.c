#include "wavefront_plan.h"

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    const long *dims;
    long ndims;
    long *dst;
    long capacity;
    long count;
} WavefrontPlanBuildCtx;

static int wavefront_plan_collect_offsets(const long *idx,
                                          long ndims,
                                          long level,
                                          long ordinal_in_level,
                                          void *user) {
    WavefrontPlanBuildCtx *ctx = (WavefrontPlanBuildCtx *)user;
    long offset;

    (void)level;
    (void)ordinal_in_level;

    if (!ctx || !idx || ndims != ctx->ndims) return -1;
    if (ctx->count >= ctx->capacity) return -1;

    offset = wavefront_nd_row_major_offset(ctx->dims, idx, ndims);
    if (offset < 0) return -1;

    ctx->dst[ctx->count++] = offset;
    return 0;
}

KMWavefrontPlan *km_wavefront_plan_create(const long *dims, long ndims, long max_state) {
    KMWavefrontPlan *plan;
    long total_points;
    long max_level;
    long cursor = 0;

    if (!wavefront_nd_validate_dims(dims, ndims)) return NULL;

    total_points = wavefront_nd_total_points(dims, ndims);
    max_level = wavefront_nd_max_level(dims, ndims);
    if (total_points <= 0 || max_level < 0) return NULL;

    plan = (KMWavefrontPlan *)calloc(1, sizeof(*plan));
    if (!plan) return NULL;

    plan->dims = (long *)malloc((size_t)ndims * sizeof(long));
    plan->level_starts = (long *)malloc((size_t)(max_level + 2) * sizeof(long));
    plan->level_offsets = (long *)malloc((size_t)total_points * sizeof(long));

    if (!plan->dims || !plan->level_starts || !plan->level_offsets) {
        km_wavefront_plan_free(plan);
        return NULL;
    }

    memcpy(plan->dims, dims, (size_t)ndims * sizeof(long));
    plan->ndims = ndims;
    plan->max_state = max_state;
    plan->total_points = total_points;
    plan->max_level = max_level;
    plan->max_level_size = 0;

    for (long level = 0; level <= max_level; level++) {
        long level_size = wavefront_nd_level_size(dims, ndims, level);
        WavefrontPlanBuildCtx ctx;
        int rc;

        if (level_size < 0) {
            km_wavefront_plan_free(plan);
            return NULL;
        }

        if (level_size > plan->max_level_size) plan->max_level_size = level_size;
        plan->level_starts[level] = cursor;

        ctx.dims = dims;
        ctx.ndims = ndims;
        ctx.dst = plan->level_offsets + cursor;
        ctx.capacity = level_size;
        ctx.count = 0;

        rc = wavefront_nd_for_level(dims, ndims, level, NULL,
                                    wavefront_plan_collect_offsets, &ctx);
        if (rc != 0 || ctx.count != level_size) {
            km_wavefront_plan_free(plan);
            return NULL;
        }

        cursor += level_size;
    }

    if (cursor != total_points) {
        km_wavefront_plan_free(plan);
        return NULL;
    }

    plan->level_starts[max_level + 1] = total_points;

#if defined(_OPENMP) && !defined(KMAMBA_NO_OPENMP)
    plan->n_threads = omp_get_max_threads();
#else
    plan->n_threads = 1;
#endif
    if (plan->n_threads <= 0) plan->n_threads = 1;

    if (plan->max_state > 0) {
        plan->scratch_thread = (float *)calloc((size_t)plan->n_threads * (size_t)plan->max_state, sizeof(float));
        if (!plan->scratch_thread) {
            km_wavefront_plan_free(plan);
            return NULL;
        }
    }
    plan->coords_thread = (long *)calloc((size_t)plan->n_threads * (size_t)plan->ndims, sizeof(long));
    if (!plan->coords_thread) {
        km_wavefront_plan_free(plan);
        return NULL;
    }
    return plan;
}

void km_wavefront_plan_free(KMWavefrontPlan *plan) {
    if (!plan) return;
    free(plan->dims);
    free(plan->level_starts);
    free(plan->level_offsets);
    free(plan->scratch_thread);
    free(plan->coords_thread);
    free(plan);
}

int km_wavefront_plan_matches_dims(const KMWavefrontPlan *plan,
                                   const long *dims,
                                   long ndims) {
    if (!plan || !dims || ndims <= 0) return 0;
    if (plan->ndims != ndims) return 0;

    for (long axis = 0; axis < ndims; axis++) {
        if (plan->dims[axis] != dims[axis]) return 0;
    }

    return 1;
}

long km_wavefront_plan_level_size(const KMWavefrontPlan *plan, long level) {
    if (!plan || level < 0 || level > plan->max_level) return -1;
    return plan->level_starts[level + 1] - plan->level_starts[level];
}

const long *km_wavefront_plan_level_offsets(const KMWavefrontPlan *plan, long level) {
    if (!plan || level < 0 || level > plan->max_level) return NULL;
    return plan->level_offsets + plan->level_starts[level];
}

int km_wavefront_plan_iter_forward(const KMWavefrontPlan *plan,
                                   KMWavefrontPlanIterCallback callback,
                                   void *userdata) {
    long level;
    if (!plan || !callback) return -1;
    for (level = 0; level <= plan->max_level; level++) {
        const long *offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        long i;
        if (!offsets || level_size < 0) return -1;
        for (i = 0; i < level_size; i++) {
            callback(offsets[i], level, userdata);
        }
    }
    return 0;
}

int km_wavefront_plan_iter_reverse(const KMWavefrontPlan *plan,
                                   KMWavefrontPlanIterCallback callback,
                                   void *userdata) {
    long level;
    if (!plan || !callback) return -1;
    for (level = plan->max_level; level >= 0; level--) {
        const long *offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        long i;
        if (!offsets || level_size < 0) return -1;
        for (i = 0; i < level_size; i++) {
            callback(offsets[i], level, userdata);
        }
    }
    return 0;
}
