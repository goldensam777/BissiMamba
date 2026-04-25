#ifndef KMAMBA_WAVEFRONT_PLAN_H
#define KMAMBA_WAVEFRONT_PLAN_H

#include "wavefront_nd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * wavefront_plan.h — Plan exécutable réutilisable pour opérateurs ND
 *
 * Le plan ne connaît que la topologie de la grille :
 *   - dims / ndims
 *   - niveaux wavefront
 *   - offsets row-major groupés par niveau
 *
 * Il est indépendant de :
 *   - la taille du noyau de convolution
 *   - la taille d'état du scan
 *   - la sémantique de l'opérateur
 *
 * Les opérateurs ND (scan, conv, etc.) partagent donc le même plan, tout en
 * conservant leurs propres paramètres métier.
 * ============================================================================ */

typedef struct {
    long *dims;           /* [ndims] */
    long ndims;
    long max_state;
    long total_points;
    long max_level;
    long max_level_size;
    long *level_starts;   /* [max_level + 2] */
    long *level_offsets;  /* [total_points] offsets row-major groupés par niveau */
    int n_threads;
    float *scratch_thread; /* [n_threads * max_state] reusable thread-local scratch */
    long  *coords_thread;  /* [n_threads * ndims] reusable thread-local coords */
} KMWavefrontPlan;

typedef void (*KMWavefrontPlanIterCallback)(long offset, long level, void *userdata);

/* Construit un plan pour une grille ND bornée. Retourne NULL si invalide. */
KMWavefrontPlan *km_wavefront_plan_create(const long *dims, long ndims, long max_state);

/* Libère le plan et tous ses buffers. */
void km_wavefront_plan_free(KMWavefrontPlan *plan);

/* Vérifie que le plan correspond exactement à dims/ndims. */
int km_wavefront_plan_matches_dims(const KMWavefrontPlan *plan,
                                   const long *dims,
                                   long ndims);

/* Taille du niveau donné, ou -1 si invalide. */
long km_wavefront_plan_level_size(const KMWavefrontPlan *plan, long level);

/* Pointeur vers les offsets row-major du niveau donné, ou NULL si invalide. */
const long *km_wavefront_plan_level_offsets(const KMWavefrontPlan *plan, long level);

/* Iterate level-by-level from 0 to max_level. */
int km_wavefront_plan_iter_forward(const KMWavefrontPlan *plan,
                                   KMWavefrontPlanIterCallback callback,
                                   void *userdata);

/* Iterate level-by-level from max_level down to 0. */
int km_wavefront_plan_iter_reverse(const KMWavefrontPlan *plan,
                                   KMWavefrontPlanIterCallback callback,
                                   void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_WAVEFRONT_PLAN_H */
