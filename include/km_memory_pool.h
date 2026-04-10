/*
 * km_memory_pool.h — Simple memory pool for k-mamba
 *
 * Évite les malloc/free répétés dans la boucle d'entraînement.
 * Thread-safe avec OpenMP (chaque thread a son propre pool).
 */

#ifndef KM_MEMORY_POOL_H
#define KM_MEMORY_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory Pool
 * ============================================================================ */
typedef struct KMMemoryPool KMMemoryPool;

/* Create/destroy */
KMMemoryPool* km_memory_pool_create(void);
void km_memory_pool_destroy(KMMemoryPool *pool);

/* Allocation - retourne un buffer réutilisable ou alloue si nécessaire */
void* km_pool_alloc(KMMemoryPool *pool, size_t size);
void km_pool_free(KMMemoryPool *pool, void *ptr);

/* Clear all allocations (garde la capacité) */
void km_pool_clear(KMMemoryPool *pool);

/* Get thread-local pool (crée si nécessaire) */
KMMemoryPool* km_memory_pool_threadlocal(void);

/* Destroy all thread-local pools */
void km_memory_pool_destroy_all_threadlocal(void);

#ifdef __cplusplus
}
#endif

#endif /* KM_MEMORY_POOL_H */
