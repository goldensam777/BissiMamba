#ifndef KMAMBA_CONFIGS_H
#define KMAMBA_CONFIGS_H

#include "kmamba.h"           /* KMambaConfig */
#include "kmamba_kernels.h"    /* MBOptimConfig */
#include "kmamba_cuda_utils.h" /* KMambaBackend */

#ifdef __cplusplus
extern "C" {
#endif

/* Unified configuration for a K-Mamba model instance.
 * Training parameters (batch_size, epochs, data_path, GC policy)
 * belong to the Trainer, NOT here. */
typedef struct {
    KMambaConfig model;        /* Architecture (dim, state_size, spatial_ndims...) */
    MBOptimConfig optim;       /* Optimizer (lr, mu, beta2, weight_decay...) */
    int backend;              /* KMambaBackend: 0=AUTO, 1=CPU, 2=GPU */
    int gpu_device;           /* CUDA device ID (-1 = default) */
    char data_path[256];      /* Path to dataset */
    char checkpoint_path[256]; /* Path to checkpoint file */
} KMambaFullConfig;

/* Fill with safe defaults (calls kmamba_config_set_defaults + optim defaults) */
void kmamba_configs_default(KMambaFullConfig *cfg);

/* Load configuration from a JSON file. Returns 0 on success, -1 on error. */
int  kmamba_configs_load_json(KMambaFullConfig *cfg, const char *path);

/* Save current configuration to a JSON file (for reproducibility). */
int  kmamba_configs_save_json(const KMambaFullConfig *cfg, const char *path);

/* Create a KMamba model from a full config. Returns NULL on failure. */
KMamba* kmamba_configs_create_model(const KMambaFullConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_CONFIGS_H */
