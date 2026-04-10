/* kmamba_checkpoint.cu — Gradient checkpointing CUDA implementation */

#include "kmamba_checkpoint.h"
#include "kmamba.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ============================================================================
 * GPU Checkpoint Implementation
 * ============================================================================ */

KMambaCheckpointStateGPU* kmamba_checkpoint_create_gpu(size_t n_layers,
                                                        size_t L, size_t D,
                                                        const KMambaCheckpointConfig *cfg) {
    if (!cfg) return NULL;
    
    KMambaCheckpointStateGPU *ckpt = (KMambaCheckpointStateGPU*)calloc(1, sizeof(*ckpt));
    if (!ckpt) return NULL;
    
    ckpt->n_layers = n_layers;
    ckpt->L = L;
    ckpt->D = D;
    ckpt->config = *cfg;
    
    if (cfg->policy == KMAMBA_CHECKPOINT_NONE) {
        ckpt->d_layer_inputs = NULL;
        return ckpt;
    }
    
    /* Allocate array of device pointers */
    ckpt->d_layer_inputs = (float**)calloc(n_layers, sizeof(float*));
    if (!ckpt->d_layer_inputs) {
        free(ckpt);
        return NULL;
    }
    
    size_t layer_bytes = L * D * sizeof(float);
    
    /* Allocate device memory for checkpoints */
    for (size_t i = 0; i < n_layers; i++) {
        int should_checkpoint = 0;
        
        switch (cfg->policy) {
            case KMAMBA_CHECKPOINT_LAYER:
                should_checkpoint = (i % cfg->checkpoint_every_n_layers) == 0;
                break;
            case KMAMBA_CHECKPOINT_BLOCK:
                should_checkpoint = 1;
                break;
            default:
                should_checkpoint = 0;
        }
        
        if (should_checkpoint) {
            cudaError_t err = cudaMalloc(&ckpt->d_layer_inputs[i], layer_bytes);
            if (err != cudaSuccess) {
                /* Cleanup on failure */
                for (size_t j = 0; j < i; j++) {
                    cudaFree(ckpt->d_layer_inputs[j]);
                }
                free(ckpt->d_layer_inputs);
                free(ckpt);
                return NULL;
            }
            /* Initialize to zero */
            cudaMemset(ckpt->d_layer_inputs[i], 0, layer_bytes);
        }
    }
    
    return ckpt;
}

void kmamba_checkpoint_free_gpu(KMambaCheckpointStateGPU *ckpt) {
    if (!ckpt) return;
    
    if (ckpt->d_layer_inputs) {
        for (size_t i = 0; i < ckpt->n_layers; i++) {
            cudaFree(ckpt->d_layer_inputs[i]);
        }
        free(ckpt->d_layer_inputs);
    }
    
    free(ckpt);
}

void kmamba_checkpoint_save_layer_gpu(KMambaCheckpointStateGPU *ckpt,
                                      size_t layer_idx,
                                      const float *d_layer_input) {
    if (!ckpt || !ckpt->d_layer_inputs) return;
    if (layer_idx >= ckpt->n_layers) return;
    if (!ckpt->d_layer_inputs[layer_idx]) return;  /* Not checkpointing this layer */
    
    size_t bytes = ckpt->L * ckpt->D * sizeof(float);
    cudaMemcpy(ckpt->d_layer_inputs[layer_idx], d_layer_input, bytes,
               cudaMemcpyDeviceToDevice);
}

const float* kmamba_checkpoint_get_layer_gpu(const KMambaCheckpointStateGPU *ckpt,
                                              size_t layer_idx) {
    if (!ckpt || !ckpt->d_layer_inputs) return NULL;
    if (layer_idx >= ckpt->n_layers) return NULL;
    return ckpt->d_layer_inputs[layer_idx];
}

/* Recompute forward pass for a layer during backward */
int kmamba_checkpoint_recompute_layer_gpu(
    const KMambaCheckpointStateGPU *ckpt,
    const MambaBlock *block,
    size_t layer_idx,
    float *d_output,
    void *d_cublas_handle,
    float *d_workspace,
    size_t workspace_size) {
    
    if (!ckpt || !block || !d_output) return -1;
    
    cublasHandle_t cublas = (cublasHandle_t)d_cublas_handle;
    
    /* Get checkpointed input */
    const float *d_input = kmamba_checkpoint_get_layer_gpu(ckpt, layer_idx);
    if (!d_input) return -1;  /* No checkpoint available */
    
    /* Cast block parameters for GPU (they should already be on device) */
    /* Note: In real implementation, block->W_in.data etc should be device pointers
     * when using GPU. This requires the model to maintain separate device copies. */
    
    /* Simplified: just copy input to output for now
     * Full implementation would call gpu_block_forward with proper device buffers */
    size_t bytes = ckpt->L * ckpt->D * sizeof(float);
    cudaMemcpy(d_output, d_input, bytes, cudaMemcpyDeviceToDevice);
    
    return 0;
}
