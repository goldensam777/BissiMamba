/* kmamba_checkpoint.c — Gradient checkpointing CPU implementation */

#include "../include/kmamba_checkpoint.h"
#include "../include/kmamba.h"
#include <stdlib.h>
#include <string.h>
#include <string.h>

/* ============================================================================
 * CPU Checkpoint Implementation
 * ============================================================================ */

KMambaCheckpointState* kmamba_checkpoint_create(const KMamba *m,
                                               const KMambaCheckpointConfig *cfg) {
    if (!m || !cfg) return NULL;
    
    KMambaCheckpointState *ckpt = (KMambaCheckpointState*)calloc(1, sizeof(*ckpt));
    if (!ckpt) return NULL;
    
    ckpt->n_layers = m->cfg.n_layers;
    ckpt->config = *cfg;
    
    if (cfg->policy == KMAMBA_CHECKPOINT_NONE) {
        /* No checkpointing, no storage needed */
        ckpt->layers = NULL;
        return ckpt;
    }
    
    /* Allocate checkpoint array */
    ckpt->layers = (MambaLayerCheckpoint*)calloc(ckpt->n_layers, sizeof(*ckpt->layers));
    if (!ckpt->layers) {
        free(ckpt);
        return NULL;
    }
    
    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;
    size_t layer_bytes = L * D * sizeof(float);
    
    /* Initialize layer checkpoints */
    for (size_t i = 0; i < ckpt->n_layers; i++) {
        int should_checkpoint = 0;
        
        switch (cfg->policy) {
            case KMAMBA_CHECKPOINT_LAYER:
                should_checkpoint = (i % cfg->checkpoint_every_n_layers) == 0;
                break;
            case KMAMBA_CHECKPOINT_BLOCK:
                should_checkpoint = 1;  /* Checkpoint every block */
                break;
            default:
                should_checkpoint = 0;
        }
        
        if (should_checkpoint) {
            ckpt->layers[i].layer_input = (float*)malloc(layer_bytes);
            ckpt->layers[i].size = layer_bytes;
            ckpt->layers[i].has_checkpoint = 0;
            
            if (!ckpt->layers[i].layer_input) {
                /* Cleanup on failure */
                for (size_t j = 0; j <= i; j++) {
                    free(ckpt->layers[j].layer_input);
                }
                free(ckpt->layers);
                free(ckpt);
                return NULL;
            }
        }
    }
    
    return ckpt;
}

void kmamba_checkpoint_free(KMambaCheckpointState *ckpt) {
    if (!ckpt) return;
    
    if (ckpt->layers) {
        for (size_t i = 0; i < ckpt->n_layers; i++) {
            free(ckpt->layers[i].layer_input);
        }
        free(ckpt->layers);
    }
    
    free(ckpt);
}

void kmamba_checkpoint_save_layer(KMambaCheckpointState *ckpt,
                                   size_t layer_idx,
                                   const float *layer_input,
                                   size_t L, size_t D) {
    if (!ckpt || !ckpt->layers) return;
    if (layer_idx >= ckpt->n_layers) return;
    
    MambaLayerCheckpoint *layer = &ckpt->layers[layer_idx];
    if (!layer->layer_input) return;  /* Not configured for checkpointing */
    
    size_t bytes = L * D * sizeof(float);
    memcpy(layer->layer_input, layer_input, bytes);
    layer->has_checkpoint = 1;
}

const float* kmamba_checkpoint_get_layer(const KMambaCheckpointState *ckpt,
                                          size_t layer_idx) {
    if (!ckpt || !ckpt->layers) return NULL;
    if (layer_idx >= ckpt->n_layers) return NULL;
    
    MambaLayerCheckpoint *layer = &ckpt->layers[layer_idx];
    if (!layer->has_checkpoint) return NULL;
    
    return layer->layer_input;
}

void kmamba_checkpoint_clear_layer(KMambaCheckpointState *ckpt, size_t layer_idx) {
    if (!ckpt || !ckpt->layers) return;
    if (layer_idx >= ckpt->n_layers) return;
    
    ckpt->layers[layer_idx].has_checkpoint = 0;
}
