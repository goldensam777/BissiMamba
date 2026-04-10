/*
 * km_memory_pool.c — Simple bump allocator with reuse
 */

#include "km_memory_pool.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Chunk de mémoire réutilisable */
typedef struct PoolChunk {
    void *ptr;
    size_t size;
    int in_use;
} PoolChunk;

struct KMMemoryPool {
    PoolChunk *chunks;
    size_t n_chunks;
    size_t capacity;
    size_t total_allocated;
    size_t total_reused;
};

/* ============================================================================
 * Pool operations
 * ============================================================================ */
KMMemoryPool* km_memory_pool_create(void) {
    KMMemoryPool *pool = (KMMemoryPool*)calloc(1, sizeof(KMMemoryPool));
    if (!pool) return NULL;
    
    pool->capacity = 16;
    pool->chunks = (PoolChunk*)calloc(pool->capacity, sizeof(PoolChunk));
    if (!pool->chunks) {
        free(pool);
        return NULL;
    }
    return pool;
}

void km_memory_pool_destroy(KMMemoryPool *pool) {
    if (!pool) return;
    
    for (size_t i = 0; i < pool->n_chunks; i++) {
        free(pool->chunks[i].ptr);
    }
    free(pool->chunks);
    free(pool);
}

void* km_pool_alloc(KMMemoryPool *pool, size_t size) {
    if (!pool || size == 0) return NULL;
    
    /* Recherche d'un chunk libre de taille suffisante */
    for (size_t i = 0; i < pool->n_chunks; i++) {
        if (!pool->chunks[i].in_use && pool->chunks[i].size >= size) {
            pool->chunks[i].in_use = 1;
            pool->total_reused += size;
            return pool->chunks[i].ptr;
        }
    }
    
    /* Pas de chunk réutilisable → allocation */
    void *ptr = malloc(size);
    if (!ptr) return NULL;
    
    /* Ajout au pool */
    if (pool->n_chunks >= pool->capacity) {
        size_t new_cap = pool->capacity * 2;
        PoolChunk *new_chunks = (PoolChunk*)realloc(pool->chunks, new_cap * sizeof(PoolChunk));
        if (!new_chunks) {
            free(ptr);
            return NULL;
        }
        pool->chunks = new_chunks;
        pool->capacity = new_cap;
    }
    
    pool->chunks[pool->n_chunks].ptr = ptr;
    pool->chunks[pool->n_chunks].size = size;
    pool->chunks[pool->n_chunks].in_use = 1;
    pool->n_chunks++;
    pool->total_allocated += size;
    
    return ptr;
}

void km_pool_free(KMMemoryPool *pool, void *ptr) {
    if (!pool || !ptr) return;
    
    for (size_t i = 0; i < pool->n_chunks; i++) {
        if (pool->chunks[i].ptr == ptr) {
            pool->chunks[i].in_use = 0;
            return;
        }
    }
}

void km_pool_clear(KMMemoryPool *pool) {
    if (!pool) return;
    
    for (size_t i = 0; i < pool->n_chunks; i++) {
        pool->chunks[i].in_use = 0;
    }
}

/* ============================================================================
 * Thread-local pools
 * ============================================================================ */
#ifdef _OPENMP
static KMMemoryPool **thread_pools = NULL;
static int max_threads = 0;

KMMemoryPool* km_memory_pool_threadlocal(void) {
    int tid = omp_get_thread_num();
    
    #pragma omp critical
    {
        if (!thread_pools || tid >= max_threads) {
            int new_max = tid + 1;
            KMMemoryPool **new_pools = (KMMemoryPool**)realloc(thread_pools, 
                                                               new_max * sizeof(KMMemoryPool*));
            if (new_pools) {
                for (int i = max_threads; i < new_max; i++) {
                    new_pools[i] = NULL;
                }
                thread_pools = new_pools;
                max_threads = new_max;
            }
        }
    }
    
    if (!thread_pools || tid >= max_threads) return NULL;
    
    if (!thread_pools[tid]) {
        #pragma omp critical
        {
            if (!thread_pools[tid]) {
                thread_pools[tid] = km_memory_pool_create();
            }
        }
    }
    
    return thread_pools[tid];
}

void km_memory_pool_destroy_all_threadlocal(void) {
    if (!thread_pools) return;
    
    for (int i = 0; i < max_threads; i++) {
        if (thread_pools[i]) {
            km_memory_pool_destroy(thread_pools[i]);
            thread_pools[i] = NULL;
        }
    }
    free(thread_pools);
    thread_pools = NULL;
    max_threads = 0;
}
#else
/* Sans OpenMP: pool global unique */
static KMMemoryPool *global_pool = NULL;

KMMemoryPool* km_memory_pool_threadlocal(void) {
    if (!global_pool) {
        global_pool = km_memory_pool_create();
    }
    return global_pool;
}

void km_memory_pool_destroy_all_threadlocal(void) {
    km_memory_pool_destroy(global_pool);
    global_pool = NULL;
}
#endif
