#ifndef KMAMBA_TOPOLOGY_H
#define KMAMBA_TOPOLOGY_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Max ND slots stored in public config fixed arrays.
 * Execution ND/state limits are runtime-configurable (max_ndims/max_state). */
#define KMAMBA_CONFIG_MAX_NDIMS 32

/* Produit des dimensions avec vérification overflow */
int km_spatial_dims_product(const long *dims, long ndims, size_t *product_out);

/* Calcule les strides row-major pour une forme ND
 * strides[axis] = produit des dims[axis+1 .. ndims-1]
 * Retourne 0 si paramètres invalides */
int km_make_row_major_strides(const long *dims, long ndims, long *strides_out);

/* Convertit un index linéaire (row-major) en coordonnées ND
 * Inverse de km_ravel_index / wavefront_nd_row_major_offset
 * dims et strides doivent avoir ndims éléments */
void km_unravel_index(long linear,
                      const long *dims,
                      const long *strides,
                      long ndims,
                      long *coords_out);

/* Convertit des coordonnées ND en index linéaire row-major
 * Équivalent à wavefront_nd_row_major_offset, fourni ici pour complétude */
long km_ravel_index(const long *coords,
                    const long *dims,
                    const long *strides,
                    long ndims);

/* Puissance entière long : base^exp, exp >= 0
 * Retourne 0 si exp < 0 ou base == 0 && exp == 0 */
long km_power_long(long base, long exp);

/* Normalise la topologie spatiale pour création modèle/bloc */
int km_normalize_spatial_topology(long *spatial_ndims,
                                  long *spatial_dims,
                                  size_t seq_len,
                                  int use_convnd,
                                  long *convnd_ndims,
                                  long convnd_K);

/* Runtime-selectable fast exp approximation (3rd-order polynomial). */
static inline float km_fast_expf(float x) {
    if (x > 8.0f) x = 8.0f;
    if (x < -8.0f) x = -8.0f;
    return 1.0f + x + 0.5f * x * x + (1.0f / 6.0f) * x * x * x;
}

static inline float km_scan_exp(float x, int use_fast_exp) {
    return use_fast_exp ? km_fast_expf(x) : expf(x);
}

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_TOPOLOGY_H */
