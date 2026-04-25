#include "trainer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#endif

/* Internals for Checkpoint Storage */
struct KMambaCheckpointState {
    float **layer_inputs;
    int *has_checkpoint;
    size_t n_layers;
    size_t buffer_size;
    int is_gpu;
};

static int should_checkpoint(int layer_idx, int n_layers, const TrainerGCConfig *cfg) {
    if (cfg->policy == TRAINER_GC_NONE) return 1; 
    if (cfg->policy == TRAINER_GC_ALL) return (layer_idx == 0); 
    if (cfg->policy == TRAINER_GC_EVERY_N) {
        return (layer_idx % cfg->checkpoint_every_n == 0);
    }
    return 0;
}

Trainer* trainer_create(KMamba *model, const TrainerGCConfig *gc_cfg) {
    if (!model) return NULL;
    
    Trainer *tr = (Trainer*)calloc(1, sizeof(Trainer));
    tr->model = model;
    tr->gc_config = *gc_cfg;
    
    KMambaConfig *mcfg = (KMambaConfig*)kmamba_get_config(model);
    size_t L = mcfg->seq_len;
    size_t D = mcfg->dim;
    size_t n_layers = mcfg->n_layers;
    
    tr->ckpt = (KMambaCheckpointState*)calloc(1, sizeof(KMambaCheckpointState));
    tr->ckpt->n_layers = n_layers;
    tr->ckpt->buffer_size = L * D * sizeof(float);
    tr->ckpt->layer_inputs = (float**)calloc(n_layers, sizeof(float*));
    tr->ckpt->has_checkpoint = (int*)calloc(n_layers, sizeof(int));
    
#ifdef KMAMBA_BUILD_CUDA
    tr->ckpt->is_gpu = model->gpu.gpu_ready;
#endif

    if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
        cudaMalloc((void**)&tr->recompute_buffer, tr->ckpt->buffer_size);
#endif
    } else {
        tr->recompute_buffer = (float*)malloc(tr->ckpt->buffer_size);
    }

    return tr;
}

void trainer_free(Trainer *tr) {
    if (!tr) return;
    for (size_t i = 0; i < tr->ckpt->n_layers; i++) {
        if (tr->ckpt->layer_inputs[i]) {
            if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
                cudaFree(tr->ckpt->layer_inputs[i]);
#endif
            } else {
                free(tr->ckpt->layer_inputs[i]);
            }
        }
    }
    if (tr->recompute_buffer) {
        if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
            cudaFree(tr->recompute_buffer);
#endif
        } else {
            free(tr->recompute_buffer);
        }
    }
    free(tr->ckpt->layer_inputs);
    free(tr->ckpt->has_checkpoint);
    free(tr->ckpt);
    free(tr);
}

int trainer_forward(Trainer *tr, const float *input, float *output, size_t batch_size) {
    size_t n_layers = tr->ckpt->n_layers;
    size_t bytes = tr->ckpt->buffer_size * batch_size;
    
    const float *current_in = input;
    float *current_out = output;
    
    for (size_t i = 0; i < n_layers; i++) {
        if (should_checkpoint((int)i, (int)n_layers, &tr->gc_config)) {
            if (!tr->ckpt->layer_inputs[i]) {
                if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
                    cudaMalloc((void**)&tr->ckpt->layer_inputs[i], bytes);
#endif
                } else {
                    tr->ckpt->layer_inputs[i] = (float*)malloc(bytes);
                }
            }
            if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
                cudaMemcpy(tr->ckpt->layer_inputs[i], current_in, bytes, cudaMemcpyDeviceToDevice);
#endif
            } else {
                memcpy(tr->ckpt->layer_inputs[i], current_in, bytes);
            }
            tr->ckpt->has_checkpoint[i] = 1;
        }
        
        mamba_block_forward(tr->model->layers[i], current_out, current_in, batch_size);
        current_in = current_out;
    }
    
    return 0;
}

void trainer_backward(Trainer *tr, const float *dY, const float *input, size_t batch_size) {
    (void)input;
    int n_layers = (int)tr->ckpt->n_layers;
    size_t bytes = tr->ckpt->buffer_size * batch_size;
    
    const float *current_dy = dY;
    float *d_input_scratch;
    if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
        cudaMalloc((void**)&d_input_scratch, bytes);
#endif
    } else {
        d_input_scratch = (float*)malloc(bytes);
    }

    for (int i = n_layers - 1; i >= 0; i--) {
        int cp_idx = i;
        while (cp_idx > 0 && !tr->ckpt->has_checkpoint[cp_idx]) cp_idx--;
        
        const float *layer_in;
        if (cp_idx == i) {
            layer_in = tr->ckpt->layer_inputs[i];
        } else {
            float *tmp_in = tr->ckpt->layer_inputs[cp_idx];
            float *tmp_out = tr->recompute_buffer;
            for (int j = cp_idx; j < i; j++) {
                mamba_block_forward(tr->model->layers[j], tmp_out, tmp_in, batch_size);
                tmp_in = tmp_out; 
            }
            layer_in = tmp_in;
        }
        
        mamba_backward(tr->model->layers[i], current_dy, layer_in, d_input_scratch, 0);
        current_dy = d_input_scratch;
    }

    if (tr->ckpt->is_gpu) {
#ifdef KMAMBA_BUILD_CUDA
        cudaFree(d_input_scratch);
#endif
    } else {
        free(d_input_scratch);
    }
}

float trainer_train_batch(Trainer *tr, const uint32_t *batch_tokens, size_t batch_size) {
    return kmamba_train_batch(tr->model, batch_tokens, batch_size);
}
