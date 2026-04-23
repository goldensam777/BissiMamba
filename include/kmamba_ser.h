#ifndef KMAMBA_SER_H
#define KMAMBA_SER_H

#include "kmamba.h"
#include "kser.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * k-mamba .ser Serialization Integration
 * ============================================================================ */

/* Flags for loading */
#define KM_SER_LAZY     0x01  /* Memory-mapped lazy loading */
#define KM_SER_VERIFY   0x02  /* Verify checksum */

/**
 * Save k-mamba model to .ser format
 * 
 * @param m         Model to save
 * @param path      Output file path
 * @param dtype     Storage dtype (FP32/FP16/BF16/INT8)
 * @return          KSER_OK on success, error code on failure
 */
int kmamba_save_ser(KMamba* m, const char* path, KSerDtype dtype);

/**
 * Load k-mamba model from .ser format
 * 
 * @param path      Input file path
 * @param flags     Loading flags (KM_SER_LAZY, etc.)
 * @return          Loaded model or NULL on failure
 */
KMamba* kmamba_load_ser(const char* path, int flags);

/**
 * Get file info without loading
 * 
 * @param path      File path
 * @return          Info structure
 */
KMSerInfo kmamba_ser_info(const char* path);

/* Extended info structure */
typedef struct {
    KSerInfo base;
    uint32_t n_layers;
    uint32_t state_size;
    uint32_t seq_len;
} KMSerInfo;

/**
 * Save checkpoint with optimizer state
 * 
 * @param m         Model
 * @param opt       Optimizer state
 * @param path      Output path
 * @param dtype     Storage dtype
 * @return          KSER_OK on success
 */
int kmamba_save_checkpoint_ser(KMamba* m, void* opt, const char* path, KSerDtype dtype);

/**
 * Load checkpoint with optimizer state
 * 
 * @param path      Checkpoint path
 * @param m_out     Output model pointer
 * @param opt_out   Output optimizer pointer
 * @param flags     Loading flags
 * @return          KSER_OK on success
 */
int kmamba_load_checkpoint_ser(const char* path, KMamba** m_out, 
                                  void** opt_out, int flags);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_SER_H */
