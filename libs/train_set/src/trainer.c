#include "trainer.h"
#include "kmamba_kernels.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <sys/resource.h>
#include <locale.h>

#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#endif

/* ============================================================================
 * Utilities
 * ============================================================================ */

void trainer_print_experiment_header(
    const char *model_name,
    size_t dim, size_t n_layers, size_t seq_len,
    long spatial_ndims,
    const char *optimizer_name,
    float lr, float weight_decay,
    const char *backend_str,
    const char *gc_policy_str,
    size_t batch_size, size_t epochs)
{
    setlocale(LC_ALL, "");
    printf("\n");
    printf("\u250C"); for (int i=0; i<50; ++i) printf("-"); printf("\u2510\n");
    printf("\u2502 %-48s\u2502\n", "K-MAMBA EXPERIMENT DETAILS");
    printf("\u251C"); for (int i=0; i<50; ++i) printf("-"); printf("\u2524\n");
    printf("\u2502 %-16s: %-29s\u2502\n", "Model", model_name);
    printf("\u2502 %-16s: %-29zu\u2502\n", "Dim", dim);
    printf("\u2502 %-16s: %-29zu\u2502\n", "Layers", n_layers);
    printf("\u2502 %-16s: %-29zu\u2502\n", "Seq Len", seq_len);
    printf("\u2502 %-16s: %-29ld\u2502\n", "Spatial Dims", spatial_ndims);
    printf("\u251C"); for (int i=0; i<50; ++i) printf("-"); printf("\u2524\n");
    printf("\u2502 %-16s: %-29s\u2502\n", "Optimizer", optimizer_name);
    printf("\u2502 %-16s: %-29.4g\u2502\n", "LR", lr);
    printf("\u2502 %-16s: %-29.4g\u2502\n", "Weight Decay", weight_decay);
    printf("\u2502 %-16s: %-29s\u2502\n", "Backend", backend_str);
    printf("\u2502 %-16s: %-29s\u2502\n", "GC Policy", gc_policy_str);
    printf("\u251C"); for (int i=0; i<50; ++i) printf("-"); printf("\u2524\n");
    printf("\u2502 %-16s: %-29zu\u2502\n", "Batch Size", batch_size);
    printf("\u2502 %-16s: %-29zu\u2502\n", "Epochs", epochs);
    printf("\u2514"); for (int i=0; i<50; ++i) printf("-"); printf("\u2518\n");
    printf("\n");
}


static size_t current_rss_kb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
    return (size_t)usage.ru_maxrss;
}

static int ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        return -1; /* exists but not a directory */
    }
    if (mkdir(path, 0755) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static char* join_path(const char *dir, const char *file) {
    size_t len = strlen(dir) + strlen(file) + 2;
    char *path = (char*)malloc(len);
    if (!path) return NULL;
    snprintf(path, len, "%s/%s", dir, file);
    return path;
}

static FILE* open_csv_log(const char *path, const char *header) {
    FILE *f = fopen(path, "a+");
    if (!f) return NULL;
    /* Check if file is empty to write header */
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "%s\n", header);
        fflush(f);
    }
    return f;
}

/* Internals for Checkpoint Storage */
struct KMambaCheckpointState {
    float **layer_inputs;
    int *has_checkpoint;
    size_t n_layers;
    size_t buffer_size;
    int is_gpu;
};

static int should_checkpoint(int layer_idx, int n_layers, const TrainerGCConfig *cfg) {
    (void)n_layers;
    if (cfg->policy == TRAINER_GC_NONE) return 1;
    if (cfg->policy == TRAINER_GC_ALL) return (layer_idx == 0);
    if (cfg->policy == TRAINER_GC_EVERY_N) {
        return (layer_idx % cfg->checkpoint_every_n == 0);
    }
    return 0;
}

/* ============================================================================
 * Learning Rate Scheduler (Linear Warmup + Cosine Decay)
 * ============================================================================ */

void trainer_init_lr_scheduler(Trainer *tr, float initial_lr, float min_lr,
                               int warmup_steps, int max_steps) {
    if (!tr) return;
    tr->lr_scheduler.initial_lr = initial_lr;
    tr->lr_scheduler.min_lr = min_lr;
    tr->lr_scheduler.warmup_steps = warmup_steps;
    tr->lr_scheduler.max_steps = max_steps;
    tr->lr_scheduler.current_lr = initial_lr;
}

void trainer_update_lr(Trainer *tr, int current_step) {
    if (!tr) return;

    LRScheduler *sched = &tr->lr_scheduler;
    float new_lr;

    if (current_step < sched->warmup_steps && sched->warmup_steps > 0) {
        /* Linear warmup: lr goes from 0 to initial_lr */
        new_lr = sched->initial_lr * ((float)current_step / (float)sched->warmup_steps);
    } else if (current_step >= sched->max_steps) {
        /* End of training: use min_lr */
        new_lr = sched->min_lr;
    } else {
        /* Cosine decay: lr decays from initial_lr to min_lr */
        int decay_steps = sched->max_steps - sched->warmup_steps;
        int step_in_decay = current_step - sched->warmup_steps;
        float progress = (float)step_in_decay / (float)decay_steps;
        float cosine = 0.5f * (1.0f + cosf(3.14159265f * progress));
        new_lr = sched->min_lr + (sched->initial_lr - sched->min_lr) * cosine;
    }

    sched->current_lr = new_lr;

    /* Update model learning rates */
    if (tr->model) {
        tr->model->opt_blocks.lr = new_lr;
        /* Also update embedding head LR if different */
        tr->model->lr_embed_head = new_lr;
    }
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
    float *d_input_scratch = NULL;
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
        
        if (d_input_scratch) {
            mamba_backward(tr->model->layers[i], current_dy, layer_in, d_input_scratch, 0);
            current_dy = d_input_scratch;
        }
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

/* Cross-entropy loss for classification */
static float cross_entropy_loss(const float *logits, const uint32_t *labels,
                                 size_t batch_size, int num_classes, size_t seq_len) {
    float loss = 0.0f;
    for (size_t b = 0; b < batch_size; b++) {
        /* Use last position for classification */
        const float *sample_logits = logits + (b * seq_len + seq_len - 1) * num_classes;
        uint32_t label = labels[b];
        /* Softmax + negative log likelihood */
        float max_logit = sample_logits[0];
        for (int c = 1; c < num_classes; c++) {
            if (sample_logits[c] > max_logit) max_logit = sample_logits[c];
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(sample_logits[c] - max_logit);
        }
        float log_prob = sample_logits[label] - max_logit - logf(sum_exp);
        loss -= log_prob;
    }
    return loss / (float)batch_size;
}

/* Compute gradients for cross-entropy loss */
static void cross_entropy_grad(const float *logits, const uint32_t *labels,
                                float *d_logits, size_t batch_size,
                                int num_classes, size_t seq_len) {
    memset(d_logits, 0, batch_size * seq_len * num_classes * sizeof(float));
    for (size_t b = 0; b < batch_size; b++) {
        /* Gradient only at last position */
        float *sample_d_logits = d_logits + (b * seq_len + seq_len - 1) * num_classes;
        const float *sample_logits = logits + (b * seq_len + seq_len - 1) * num_classes;
        uint32_t label = labels[b];
        /* Softmax */
        float max_logit = sample_logits[0];
        for (int c = 1; c < num_classes; c++) {
            if (sample_logits[c] > max_logit) max_logit = sample_logits[c];
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(sample_logits[c] - max_logit);
        }
        for (int c = 0; c < num_classes; c++) {
            float prob = expf(sample_logits[c] - max_logit) / sum_exp;
            sample_d_logits[c] = prob;
        }
        sample_d_logits[label] -= 1.0f;
        /* Scale by batch size for mean gradient */
        for (int c = 0; c < num_classes; c++) {
            sample_d_logits[c] /= (float)batch_size;
        }
    }
}

float trainer_train_batch_vision(Trainer *tr, const float *data, const uint32_t *labels,
                                  size_t batch_size, size_t L, size_t D, int num_classes) {
    if (!tr || !data || !labels) return -1.0f;

    KMamba *m = tr->model;
    KMambaConfig *cfg = (KMambaConfig*)kmamba_get_config(m);
    size_t n_layers = cfg->n_layers;

    /* Allocate temporary buffers */
    size_t layer_bytes = L * D * batch_size * sizeof(float);
    float *current = (float*)malloc(layer_bytes);
    float *next = (float*)malloc(layer_bytes);
    float *output_logits = (float*)malloc(batch_size * L * num_classes * sizeof(float));
    float *d_output = (float*)malloc(batch_size * L * num_classes * sizeof(float));

    if (!current || !next || !output_logits || !d_output) {
        free(current); free(next); free(output_logits); free(d_output);
        return -1.0f;
    }

    /* Copy input data (flattened: batch, L, D) */
    memcpy(current, data, batch_size * L * D * sizeof(float));

    /* Forward pass through all layers */
    for (size_t i = 0; i < n_layers; i++) {
        /* Checkpoint if needed */
        if (should_checkpoint((int)i, (int)n_layers, &tr->gc_config)) {
            if (!tr->ckpt->layer_inputs[i]) {
                tr->ckpt->layer_inputs[i] = (float*)malloc(layer_bytes);
            }
            memcpy(tr->ckpt->layer_inputs[i], current, layer_bytes);
            tr->ckpt->has_checkpoint[i] = 1;
        }
        mamba_block_forward(m->layers[i], next, current, batch_size);
        /* Swap buffers */
        float *tmp = current; current = next; next = tmp;
    }

    /* Project to logits using head (simple linear projection) */
    /* For vision models, head is [dim x num_classes] */
    if (m->head) {
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *out_logits = output_logits + (b * L + t) * num_classes;
                const float *hidden = current + (b * L + t) * D;
                /* Simple linear projection: logits = hidden @ head */
                for (int c = 0; c < num_classes; c++) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < D; d++) {
                        sum += hidden[d] * m->head[d * num_classes + c];
                    }
                    out_logits[c] = sum;
                }
            }
        }
    } else {
        /* No head - use hidden directly (must match num_classes) */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *out_logits = output_logits + (b * L + t) * num_classes;
                const float *hidden = current + (b * L + t) * D;
                for (int c = 0; c < num_classes && c < (int)D; c++) {
                    out_logits[c] = hidden[c];
                }
            }
        }
    }

    /* Compute loss */
    float loss = 0.0f;
    if (output_logits) {
        loss = cross_entropy_loss(output_logits, labels, batch_size, num_classes, L);
    }

    /* Compute output gradients */
    if (output_logits) {
        cross_entropy_grad(output_logits, labels, d_output, batch_size, num_classes, L);
    }

    /* Backward pass through head */
    float *d_hidden = (float*)malloc(layer_bytes);
    memset(d_hidden, 0, layer_bytes);

    if (m->head) {
        /* Gradient w.r.t hidden and head */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *d_h = d_hidden + (b * L + t) * D;
                const float *d_logit = d_output + (b * L + t) * num_classes;
                for (size_t d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_classes; c++) {
                        sum += d_logit[c] * m->head[d * num_classes + c];
                    }
                    d_h[d] = sum;
                }
            }
        }
    } else {
        /* Direct gradient pass-through */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *d_h = d_hidden + (b * L + t) * D;
                const float *d_logit = d_output + (b * L + t) * num_classes;
                for (int c = 0; c < num_classes && c < (int)D; c++) {
                    d_h[c] = d_logit[c];
                }
            }
        }
    }

    /* Backward through layers */
    float *d_in = (float*)malloc(layer_bytes);
    /* layer_in not needed, using current directly */
    float *d_out = d_hidden;

    for (int i = (int)n_layers - 1; i >= 0; i--) {
        /* Find checkpoint */
        int cp_idx = i;
        while (cp_idx > 0 && !tr->ckpt->has_checkpoint[cp_idx]) cp_idx--;

        const float *layer_input;
        if (cp_idx == i) {
            layer_input = tr->ckpt->layer_inputs[i];
        } else {
            /* Recompute from checkpoint */
            memcpy(tr->recompute_buffer, tr->ckpt->layer_inputs[cp_idx], layer_bytes);
            for (int j = cp_idx; j < i; j++) {
                mamba_block_forward(m->layers[j], next, tr->recompute_buffer, batch_size);
                memcpy(tr->recompute_buffer, next, layer_bytes);
            }
            layer_input = tr->recompute_buffer;
        }

        /* Backward this layer */
        mamba_backward(m->layers[i], d_out, layer_input, d_in, 0);
        /* Swap for next iteration */
        float *tmp = d_out; d_out = d_in; d_in = tmp;
    }

    /* Optimizer step */
    for (size_t i = 0; i < n_layers; i++) {
        MambaBlock *block = m->layers[i];
        if (block->opt_state) {
            /* Simple SGD step for now */
            /* TODO: Use proper AdamW step */
        }
    }

    free(current);
    free(next);
    free(output_logits);
    free(d_output);
    free(d_hidden);
    free(d_in);

    return loss;
}

/* ============================================================================
 * Logging and Resume Functions
 * ============================================================================ */

extern void print_progress_table_header(void);
extern void print_progress_table_row(size_t epoch, float loss, float acc, float samples_per_s, float ms, float lr);
extern void print_progress_table_footer(void);

TrainerMetrics trainer_run(
    Trainer *trainer,
    const float *data,
    const uint32_t *labels,
    size_t num_samples,
    size_t L, size_t D, int num_classes,
    size_t batch_size,
    size_t epochs,
    const char *checkpoint_path __attribute__((unused)),
    int verbose)
{
    TrainerMetrics final_metrics = {0};
    if (!trainer || !data || !labels || batch_size == 0 || epochs == 0) return final_metrics;

    size_t steps_per_epoch = num_samples / batch_size;
    if (steps_per_epoch == 0) steps_per_epoch = 1;
    float *batch_data = (float*)malloc(batch_size * L * D * sizeof(float));
    uint32_t *batch_labels = (uint32_t*)malloc(batch_size * sizeof(uint32_t));

    /* Initialize LR scheduler if not already done */
    int total_steps = (int)(epochs * steps_per_epoch);
    if (trainer->lr_scheduler.max_steps == 0) {
        /* Auto-init: 10% warmup, cosine decay to 1% of initial LR */
        int warmup_steps = total_steps / 10;
        float initial_lr = trainer->model ? trainer->model->opt_blocks.lr : 0.001f;
        float min_lr = initial_lr * 0.01f;
        trainer_init_lr_scheduler(trainer, initial_lr, min_lr, warmup_steps, total_steps);
    }

    if (verbose) {
        trainer_print_experiment_header(
            "?", D, trainer->ckpt->n_layers, L, (long)D, "AdamW", trainer->lr_scheduler.initial_lr, 0.0f, "auto", "none", batch_size, epochs);
        print_progress_table_header();
    }

    int global_step = 0;
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f, epoch_acc = 0.0f;
        double epoch_start = (double)clock() / CLOCKS_PER_SEC;
        for (size_t step = 0; step < steps_per_epoch; ++step) {
            /* Update learning rate at each step */
            trainer_update_lr(trainer, global_step);

            size_t offset = step * batch_size * L * D;
            memcpy(batch_data, data + offset, batch_size * L * D * sizeof(float));
            memcpy(batch_labels, labels + step * batch_size, batch_size * sizeof(uint32_t));
            float loss = trainer_train_batch_vision(trainer, batch_data, batch_labels, batch_size, L, D, num_classes);
            epoch_loss += loss;
            // Dummy accuracy for now (implement as needed)
            epoch_acc += 0.0f;

            global_step++;
        }
        double epoch_end = (double)clock() / CLOCKS_PER_SEC;
        double ms = (epoch_end - epoch_start) * 1000.0;
        float avg_loss = epoch_loss / steps_per_epoch;
        float avg_acc = epoch_acc / steps_per_epoch;
        float samples_per_s = (float)(steps_per_epoch * batch_size) / ((epoch_end - epoch_start) > 0 ? (epoch_end - epoch_start) : 1);
        float lr = trainer->lr_scheduler.current_lr;
        if (verbose) print_progress_table_row(epoch, avg_loss, avg_acc, samples_per_s, (float)ms, lr);
        final_metrics.loss = avg_loss;
        final_metrics.accuracy = avg_acc;
    }
    if (verbose) print_progress_table_footer();
    free(batch_data);
    free(batch_labels);
    return final_metrics;
}

void print_progress_table_header(void) {
    setlocale(LC_ALL, "");
    printf("\u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n");
    printf("| Epoch |  Loss   | Acc (%%) | Samples/s | Time (ms) |    LR     |\n");
    printf("\u251C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n");
}

void print_progress_table_row(size_t epoch, float loss, float acc, float samples_per_s, float ms, float lr) {
    printf("\u2502 %-5zu \u2502 %7.4f \u2502 %7.2f \u2502 %9.1f \u2502 %9.1f \u2502 %9.2e \u2502\n",
        epoch, loss, acc, samples_per_s, ms, lr);
}

void print_progress_table_footer(void) {
    printf("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n");
}


Trainer* trainer_create_with_logging(KMamba *model, const TrainerGCConfig *gc_cfg,
                                      const TrainerLogConfig *log_cfg) {
    Trainer *tr = trainer_create(model, gc_cfg);
    if (!tr) return NULL;

    if (log_cfg) {
        tr->log_cfg = *log_cfg;
        tr->log_cfg.log_dir = log_cfg->log_dir ? strdup(log_cfg->log_dir) : NULL;
        tr->log_cfg.run_name = log_cfg->run_name ? strdup(log_cfg->run_name) : NULL;

        if (tr->log_cfg.log_dir) {
            if (ensure_dir(tr->log_cfg.log_dir) != 0) {
                fprintf(stderr, "[trainer] Failed to create log dir: %s\n", tr->log_cfg.log_dir);
            } else {
                const char *run_name = tr->log_cfg.run_name ? tr->log_cfg.run_name : "run";
                char step_filename[256];
                char epoch_filename[256];
                snprintf(step_filename, sizeof(step_filename), "%s.step.csv", run_name);
                snprintf(epoch_filename, sizeof(epoch_filename), "%s.epoch.csv", run_name);

                char *step_path = join_path(tr->log_cfg.log_dir, step_filename);
                char *epoch_path = join_path(tr->log_cfg.log_dir, epoch_filename);

                tr->step_log = open_csv_log(step_path,
                    "run_id,epoch,step_in_epoch,global_step,loss,accuracy,grad_norm,step_ms,tokens_per_sec,rss_kb,lr");
                tr->epoch_log = open_csv_log(epoch_path,
                    "run_id,epoch,total_steps,avg_loss,avg_accuracy,epoch_ms,tokens_per_sec,rss_kb");

                free(step_path);
                free(epoch_path);
            }
        }
        tr->run_id = (unsigned long long)time(NULL);
    }

    tr->resume.epoch = 0;
    tr->resume.global_step = 0;
    tr->resume.best_val_loss = 1e30f;
    tr->resume.checkpoint_path = NULL;

    return tr;
}

static int write_resume_state(const char *path, const TrainerResumeState *state) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "epoch=%d\n", state->epoch);
    fprintf(f, "global_step=%zu\n", state->global_step);
    fprintf(f, "best_val_loss=%.6f\n", state->best_val_loss);
    fclose(f);
    return 0;
}

static int read_resume_state(const char *path, TrainerResumeState *state) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "epoch=", 6) == 0) {
            state->epoch = atoi(line + 6);
        } else if (strncmp(line, "global_step=", 12) == 0) {
            state->global_step = (size_t)atoll(line + 12);
        } else if (strncmp(line, "best_val_loss=", 14) == 0) {
            state->best_val_loss = atof(line + 14);
        }
    }
    fclose(f);
    return 0;
}

/**
 * Save optimizer state for a single layer
 */
static int save_layer_opt_state(FILE *fp, MambaBlock *block) {
    if (!block || !block->opt_state) return 0;

    MBConfig *cfg = &block->config;
    size_t d_state = cfg->state_size;
    size_t N = d_state;        /* N = state_size for scan dimensions */
    size_t R = (cfg->mimo_rank > 1) ? cfg->mimo_rank : 1;  /* Use mimo_rank from config */
    size_t d_inner = cfg->dim * cfg->expand_factor;

    /* Helper to write a float array */
    #define WRITE_ARRAY(ptr, count) \
        if ((ptr) && (count) > 0) { \
            size_t n = fwrite((ptr), sizeof(float), (count), fp); \
            if (n != (count)) return -1; \
        }

    /* Write dimensions first */
    size_t dims[6] = {cfg->dim, d_state, N, R, d_inner, cfg->seq_len};
    if (fwrite(dims, sizeof(size_t), 6, fp) != 6) return -1;

    /* Gradients */
    WRITE_ARRAY(block->opt_state->g_W_in, cfg->dim * d_state);
    WRITE_ARRAY(block->opt_state->g_W_out, d_state * cfg->dim);
    WRITE_ARRAY(block->opt_state->g_A_log, d_state);
    WRITE_ARRAY(block->opt_state->g_W_B, d_state);
    WRITE_ARRAY(block->opt_state->g_W_C, d_state);
    WRITE_ARRAY(block->opt_state->g_b_B, d_state);
    WRITE_ARRAY(block->opt_state->g_b_C, d_state);
    WRITE_ARRAY(block->opt_state->g_delta_proj, cfg->dim);
    WRITE_ARRAY(block->opt_state->g_theta, d_state / 2);
    WRITE_ARRAY(block->opt_state->g_lambda_proj, cfg->dim);

    /* Adam moments: m and v */
    WRITE_ARRAY(block->opt_state->m_W_in, cfg->dim * d_state);
    WRITE_ARRAY(block->opt_state->v_W_in, cfg->dim * d_state);
    WRITE_ARRAY(block->opt_state->m_W_out, d_state * cfg->dim);
    WRITE_ARRAY(block->opt_state->v_W_out, d_state * cfg->dim);
    WRITE_ARRAY(block->opt_state->m_A_log, d_state);
    WRITE_ARRAY(block->opt_state->v_A_log, d_state);
    WRITE_ARRAY(block->opt_state->m_W_B, d_state);
    WRITE_ARRAY(block->opt_state->v_W_B, d_state);
    WRITE_ARRAY(block->opt_state->m_W_C, d_state);
    WRITE_ARRAY(block->opt_state->v_W_C, d_state);
    WRITE_ARRAY(block->opt_state->m_b_B, d_state);
    WRITE_ARRAY(block->opt_state->v_b_B, d_state);
    WRITE_ARRAY(block->opt_state->m_b_C, d_state);
    WRITE_ARRAY(block->opt_state->v_b_C, d_state);
    WRITE_ARRAY(block->opt_state->m_delta_proj, cfg->dim);
    WRITE_ARRAY(block->opt_state->v_delta_proj, cfg->dim);
    WRITE_ARRAY(block->opt_state->m_theta, d_state / 2);
    WRITE_ARRAY(block->opt_state->v_theta, d_state / 2);
    WRITE_ARRAY(block->opt_state->m_lambda_proj, cfg->dim);
    WRITE_ARRAY(block->opt_state->v_lambda_proj, cfg->dim);

    #undef WRITE_ARRAY
    return 0;
}

int trainer_save_checkpoint(Trainer *tr, const char *path) {
    if (!tr || !path) return -1;

    /* Save model weights */
    if (kmamba_save(tr->model, path) != 0) {
        fprintf(stderr, "[trainer] Failed to save model to %s\n", path);
        return -1;
    }

    /* Save training state alongside model */
    char state_path[512];
    snprintf(state_path, sizeof(state_path), "%s.state", path);
    if (write_resume_state(state_path, &tr->resume) != 0) {
        fprintf(stderr, "[trainer] Failed to save state to %s\n", state_path);
        return -1;
    }

    /* Save optimizer state */
    char opt_path[512];
    snprintf(opt_path, sizeof(opt_path), "%s.opt", path);
    FILE *fp = fopen(opt_path, "wb");
    if (!fp) {
        fprintf(stderr, "[trainer] Failed to open %s for writing\n", opt_path);
        return -1;
    }

    /* Write header */
    size_t n_layers = tr->model->cfg.n_layers;
    if (fwrite(&n_layers, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* Write optimizer state for each layer */
    for (size_t i = 0; i < n_layers; i++) {
        if (save_layer_opt_state(fp, tr->model->layers[i]) != 0) {
            fprintf(stderr, "[trainer] Failed to save optimizer state for layer %zu\n", i);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);

    printf("[trainer] Checkpoint saved: epoch=%d, step=%zu, path=%s\n",
           tr->resume.epoch, tr->resume.global_step, path);
    return 0;
}

Trainer* trainer_resume(KMamba *model, const TrainerGCConfig *gc_cfg,
                        const TrainerLogConfig *log_cfg,
                        const char *checkpoint_path) {
    (void)model; /* We load the model from checkpoint, not use the passed one */
    if (!checkpoint_path) return NULL;

    /* Try to load model first */
    KMamba *loaded_model = kmamba_load(checkpoint_path, 1, NULL, 0.0f, 0.0f);
    if (!loaded_model) {
        fprintf(stderr, "[trainer] Failed to load checkpoint: %s\n", checkpoint_path);
        return NULL;
    }

    /* Create trainer with loaded model */
    Trainer *tr = trainer_create_with_logging(loaded_model, gc_cfg, log_cfg);
    if (!tr) {
        kmamba_free(loaded_model);
        return NULL;
    }

    /* Load resume state if exists */
    char state_path[512];
    snprintf(state_path, sizeof(state_path), "%s.state", checkpoint_path);
    if (read_resume_state(state_path, &tr->resume) == 0) {
        printf("[trainer] Resumed from epoch=%d, global_step=%zu\n",
               tr->resume.epoch, tr->resume.global_step);
    } else {
        printf("[trainer] No state file found, starting from epoch 0\n");
        tr->resume.epoch = 0;
        tr->resume.global_step = 0;
        tr->resume.best_val_loss = 1e30f;
    }

    tr->resume.checkpoint_path = strdup(checkpoint_path);
    return tr;
}

/**
 * Load optimizer state for a single layer
 */
static int load_layer_opt_state(FILE *fp, MambaBlock *block) {
    if (!block) return 0;

    MBConfig *cfg = &block->config;
    size_t d_state = cfg->state_size;

    /* Read dimensions */
    size_t dims[6];
    if (fread(dims, sizeof(size_t), 6, fp) != 6) return -1;

    /* Ensure optimizer state is allocated */
    if (!block->opt_state) {
        /* Allocate optimizer state - use calloc to zero-initialize */
        block->opt_state = (MBOptimState*)calloc(1, sizeof(MBOptimState));
        if (!block->opt_state) return -1;
    }

    /* Helper to read a float array */
    #define READ_ARRAY(ptr, count) \
        if ((count) > 0) { \
            if (!(ptr)) { (ptr) = (float*)malloc((count) * sizeof(float)); } \
            if (!(ptr)) return -1; \
            size_t n = fread((ptr), sizeof(float), (count), fp); \
            if (n != (count)) return -1; \
        }

    /* Read gradients */
    READ_ARRAY(block->opt_state->g_W_in, cfg->dim * d_state);
    READ_ARRAY(block->opt_state->g_W_out, d_state * cfg->dim);
    READ_ARRAY(block->opt_state->g_A_log, d_state);
    READ_ARRAY(block->opt_state->g_W_B, d_state);
    READ_ARRAY(block->opt_state->g_W_C, d_state);
    READ_ARRAY(block->opt_state->g_b_B, d_state);
    READ_ARRAY(block->opt_state->g_b_C, d_state);
    READ_ARRAY(block->opt_state->g_delta_proj, cfg->dim);
    READ_ARRAY(block->opt_state->g_theta, d_state / 2);
    READ_ARRAY(block->opt_state->g_lambda_proj, cfg->dim);

    /* Read Adam moments */
    READ_ARRAY(block->opt_state->m_W_in, cfg->dim * d_state);
    READ_ARRAY(block->opt_state->v_W_in, cfg->dim * d_state);
    READ_ARRAY(block->opt_state->m_W_out, d_state * cfg->dim);
    READ_ARRAY(block->opt_state->v_W_out, d_state * cfg->dim);
    READ_ARRAY(block->opt_state->m_A_log, d_state);
    READ_ARRAY(block->opt_state->v_A_log, d_state);
    READ_ARRAY(block->opt_state->m_W_B, d_state);
    READ_ARRAY(block->opt_state->v_W_B, d_state);
    READ_ARRAY(block->opt_state->m_W_C, d_state);
    READ_ARRAY(block->opt_state->v_W_C, d_state);
    READ_ARRAY(block->opt_state->m_b_B, d_state);
    READ_ARRAY(block->opt_state->v_b_B, d_state);
    READ_ARRAY(block->opt_state->m_b_C, d_state);
    READ_ARRAY(block->opt_state->v_b_C, d_state);
    READ_ARRAY(block->opt_state->m_delta_proj, cfg->dim);
    READ_ARRAY(block->opt_state->v_delta_proj, cfg->dim);
    READ_ARRAY(block->opt_state->m_theta, d_state / 2);
    READ_ARRAY(block->opt_state->v_theta, d_state / 2);
    READ_ARRAY(block->opt_state->m_lambda_proj, cfg->dim);
    READ_ARRAY(block->opt_state->v_lambda_proj, cfg->dim);

    #undef READ_ARRAY
    return 0;
}

int trainer_load_checkpoint(Trainer *tr, const char *path) {
    if (!tr || !path) return -1;

    /* Load model weights */
    KMamba *loaded = kmamba_load(path, 1, NULL, 0.0f, 0.0f);
    if (!loaded) {
        fprintf(stderr, "[trainer] Failed to load model from %s\n", path);
        return -1;
    }

    /* Replace model in trainer */
    if (tr->model) {
        kmamba_free(tr->model);
    }
    tr->model = loaded;

    /* Load training state */
    char state_path[512];
    snprintf(state_path, sizeof(state_path), "%s.state", path);
    if (read_resume_state(state_path, &tr->resume) == 0) {
        printf("[trainer] Loaded resume state: epoch=%d, step=%zu\n",
               tr->resume.epoch, tr->resume.global_step);
    } else {
        printf("[trainer] No state file found, using defaults\n");
        tr->resume.epoch = 0;
        tr->resume.global_step = 0;
        tr->resume.best_val_loss = 1e30f;
    }

    /* Load optimizer state */
    char opt_path[512];
    snprintf(opt_path, sizeof(opt_path), "%s.opt", path);
    FILE *fp = fopen(opt_path, "rb");
    if (!fp) {
        fprintf(stderr, "[trainer] Warning: No optimizer state found at %s\n", opt_path);
        /* Not fatal - model can train from scratch */
        return 0;
    }

    /* Read header */
    size_t n_layers;
    if (fread(&n_layers, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        fprintf(stderr, "[trainer] Failed to read optimizer header\n");
        return -1;
    }

    if (n_layers != tr->model->cfg.n_layers) {
        fprintf(stderr, "[trainer] Warning: Layer count mismatch (%zu vs %zu)\n",
                n_layers, tr->model->cfg.n_layers);
    }

    /* Read optimizer state for each layer */
    size_t layers_to_load = n_layers < tr->model->cfg.n_layers ? n_layers : tr->model->cfg.n_layers;
    for (size_t i = 0; i < layers_to_load; i++) {
        if (load_layer_opt_state(fp, tr->model->layers[i]) != 0) {
            fprintf(stderr, "[trainer] Failed to load optimizer state for layer %zu\n", i);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    printf("[trainer] Optimizer state loaded for %zu layers\n", layers_to_load);

    return 0;
}

void trainer_get_metrics(const Trainer *tr, TrainerResumeMetrics *metrics) {
    if (!tr || !metrics) return;
    metrics->epoch = tr->resume.epoch;
    metrics->global_step = tr->resume.global_step;
    metrics->last_loss = 0.0f;  /* Updated by training loop */
    metrics->last_lr = 0.0f;    /* Updated by training loop */
    metrics->max_rss_kb = current_rss_kb();
}

/* ============================================================================
 * Unified Batch API Implementation
 * ============================================================================ */

/**
 * Compute accuracy from logits and labels for vision models.
 * logits: [batch_size x num_classes]
 * labels: [batch_size] class indices
 * Returns accuracy in [0.0, 1.0]
 */
static float compute_vision_accuracy(const float *logits, const uint32_t *labels,
                                     size_t batch_size, int num_classes) {
    int correct = 0;
    for (size_t b = 0; b < batch_size; b++) {
        /* Find argmax of logits for this sample */
        int pred_class = 0;
        float max_logit = logits[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            float logit = logits[b * num_classes + c];
            if (logit > max_logit) {
                max_logit = logit;
                pred_class = c;
            }
        }
        if ((uint32_t)pred_class == labels[b]) {
            correct++;
        }
    }
    return (float)correct / (float)batch_size;
}

/**
 * Vision training step with accuracy computation.
 * Full implementation with forward, backward, loss, accuracy, and AdamW optimizer.
 * Returns loss and accuracy.
 */
static TrainerMetrics trainer_train_batch_vision_with_acc(Trainer *tr,
                                                          const float *data,
                                                          const uint32_t *labels,
                                                          size_t batch_size,
                                                          size_t L,
                                                          size_t D,
                                                          int num_classes) {
    TrainerMetrics result = {-1.0f, 0.0f};
    if (!tr || !data || !labels) return result;

    KMamba *m = tr->model;
    KMambaConfig *cfg = (KMambaConfig*)kmamba_get_config(m);
    size_t n_layers = cfg->n_layers;

    /* Allocate temporary buffers */
    size_t layer_bytes = L * D * batch_size * sizeof(float);
    float *current = (float*)malloc(layer_bytes);
    float *next = (float*)malloc(layer_bytes);
    float *output_logits = NULL;
    float *d_output = (float*)malloc(batch_size * L * num_classes * sizeof(float));

    if (!current || !next || !output_logits || !d_output) {
        free(current); free(next); free(output_logits); free(d_output);
        return result;
    }

    /* Copy input data (flattened: batch, L, D) */
    memcpy(current, data, batch_size * L * D * sizeof(float));

    /* Forward pass through all layers */
    for (size_t i = 0; i < n_layers; i++) {
        /* Checkpoint if needed */
        if (should_checkpoint((int)i, (int)n_layers, &tr->gc_config)) {
            if (!tr->ckpt->layer_inputs[i]) {
                tr->ckpt->layer_inputs[i] = (float*)malloc(layer_bytes);
            }
            memcpy(tr->ckpt->layer_inputs[i], current, layer_bytes);
            tr->ckpt->has_checkpoint[i] = 1;
        }
        mamba_block_forward(m->layers[i], next, current, batch_size);

        /* Swap buffers */
        float *tmp = current; current = next; next = tmp;
    }

    /* Project to logits using head (simple linear projection) */
    /* For vision models, head is [dim x num_classes] */
    if (m->head) {
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *out_logits = output_logits + (b * L + t) * num_classes;
                const float *hidden = current + (b * L + t) * D;
                /* Simple linear projection: logits = hidden @ head */
                for (int c = 0; c < num_classes; c++) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < D; d++) {
                        sum += hidden[d] * m->head[d * num_classes + c];
                    }
                    out_logits[c] = sum;
                }
            }
        }
    } else {
        /* No head - use hidden directly (must match num_classes) */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *out_logits = output_logits + (b * L + t) * num_classes;
                const float *hidden = current + (b * L + t) * D;
                for (int c = 0; c < num_classes && c < (int)D; c++) {
                    out_logits[c] = hidden[c];
                }
            }
        }
    }

    /* Compute loss and accuracy (using last position for classification) */
    float loss = 0.0f;
    if (output_logits) {
        loss = cross_entropy_loss(output_logits, labels, batch_size, num_classes, L);
        result.accuracy = compute_vision_accuracy(output_logits, labels, batch_size, num_classes);
    }
    result.loss = loss;

    /* Compute output gradients */
    if (output_logits) {
        cross_entropy_grad(output_logits, labels, d_output, batch_size, num_classes, L);
    }

    /* Backward pass through head */
    float *d_hidden = (float*)malloc(layer_bytes);
    memset(d_hidden, 0, layer_bytes);

    /* Allocate head gradients */
    float *head_grad = (float*)calloc(D * num_classes, sizeof(float));
    const MBOptimConfig *conf = &m->opt_blocks;
    const float beta1 = 0.9f, beta2_adam = 0.999f, eps = 1e-8f;
    int t_step = (int)tr->resume.global_step + 1;

    if (m->head && head_grad) {
        /* Gradient w.r.t hidden and head */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *d_h = d_hidden + (b * L + t) * D;
                const float *d_logit = d_output + (b * L + t) * num_classes;
                const float *hidden = current + (b * L + t) * D;
                for (size_t d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_classes; c++) {
                        sum += d_logit[c] * m->head[d * num_classes + c];
                        head_grad[d * num_classes + c] += hidden[d] * d_logit[c];
                    }
                    d_h[d] = sum;
                }
            }
        }
        /* Update head weights with AdamW */
        if (m->m_head && m->v_head) {
            adamw_step_f32(m->head, head_grad, m->m_head, m->v_head,
                          conf->lr, beta1, beta2_adam, eps, conf->weight_decay,
                          D * num_classes, t_step);
        }
    } else {
        /* Direct gradient pass-through */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < L; t++) {
                float *d_h = d_hidden + (b * L + t) * D;
                const float *d_logit = d_output + (b * L + t) * num_classes;
                for (int c = 0; c < num_classes && c < (int)D; c++) {
                    d_h[c] = d_logit[c];
                }
            }
        }
    }

    free(head_grad);

    /* Backward through layers */
    float *d_in = (float*)malloc(layer_bytes);
    float *d_out = d_hidden;

    /* Zero gradients before backward pass */
    for (size_t i = 0; i < n_layers; i++) {
        mamba_zero_grads(m->layers[i]);
    }

    for (int i = (int)n_layers - 1; i >= 0; i--) {
        /* Find checkpoint */
        int cp_idx = i;
        while (cp_idx > 0 && !tr->ckpt->has_checkpoint[cp_idx]) cp_idx--;

        const float *layer_input;
        if (cp_idx == i) {
            layer_input = tr->ckpt->layer_inputs[i];
        } else {
            /* Recompute from checkpoint */
            memcpy(tr->recompute_buffer, tr->ckpt->layer_inputs[cp_idx], layer_bytes);
            for (int j = cp_idx; j < i; j++) {
                mamba_block_forward(m->layers[j], next, tr->recompute_buffer, batch_size);
                memcpy(tr->recompute_buffer, next, layer_bytes);
            }
            layer_input = tr->recompute_buffer;
        }

        /* Backward this layer - compute gradients */
        mamba_backward(m->layers[i], d_out, layer_input, d_in, 0);

        /* Swap for next iteration */
        float *tmp = d_out; d_out = d_in; d_in = tmp;
    }

    free(d_in);
    free(d_hidden);

    /* Gradient clipping: compute total norm across all layers */
    float total_norm_sq = 0.0f;
    for (size_t i = 0; i < n_layers; i++) {
        total_norm_sq += mamba_block_grad_sqnorm(m->layers[i]);
    }
    float total_norm = sqrtf(total_norm_sq);

    if (conf->clip_norm > 0.0f && total_norm > conf->clip_norm) {
        float scale = conf->clip_norm / total_norm;
        /* Scale all layer gradients */
        for (size_t i = 0; i < n_layers; i++) {
            MambaBlock *block = m->layers[i];
            if (!block->opt_state) continue;
            MBOptimState *opt = block->opt_state;
            size_t d_state = cfg->state_size;
            size_t n;
            n = block->W_in.rows * block->W_in.cols;
            for (size_t j = 0; j < n; j++) opt->g_W_in[j] *= scale;
            n = block->W_out.rows * block->W_out.cols;
            for (size_t j = 0; j < n; j++) opt->g_W_out[j] *= scale;
            n = block->A_log.rows * block->A_log.cols;
            for (size_t j = 0; j < n; j++) opt->g_A_log[j] *= scale;
            n = block->W_B.rows * block->W_B.cols;
            for (size_t j = 0; j < n; j++) opt->g_W_B[j] *= scale;
            n = block->W_C.rows * block->W_C.cols;
            for (size_t j = 0; j < n; j++) opt->g_W_C[j] *= scale;
            for (size_t j = 0; j < d_state; j++) opt->g_b_B[j] *= scale;
            for (size_t j = 0; j < d_state; j++) opt->g_b_C[j] *= scale;
            n = block->delta_proj.rows * block->delta_proj.cols;
            for (size_t j = 0; j < n; j++) opt->g_delta_proj[j] *= scale;
            n = block->lambda_proj.rows * block->lambda_proj.cols;
            for (size_t j = 0; j < n; j++) opt->g_lambda_proj[j] *= scale;
            if (opt->g_theta) {
                for (size_t j = 0; j < d_state / 2; j++) opt->g_theta[j] *= scale;
            }
        }
    }

    /* AdamW optimizer step for each layer */
    for (size_t i = 0; i < n_layers; i++) {
        MambaBlock *block = m->layers[i];
        if (block->opt_state) {
            MBOptimState *opt = block->opt_state;
            /* Optimizer config is stored in KMamba struct */
            const MBOptimConfig *conf = &m->opt_blocks;
            size_t d_state = cfg->state_size;
            int t = (int)tr->resume.global_step + 1; /* timestep for bias correction */

            /* AdamW hyperparameters (defaults when not in config) */
            const float beta1 = 0.9f;
            const float beta2_adam = 0.999f;
            const float eps = conf->eps > 0 ? conf->eps : 1e-8f;

            /* W_in: [R x dim] -> [dim x d_state] total elements */
            adamw_step_f32(block->W_in.data, opt->g_W_in,
                          opt->m_W_in, opt->v_W_in,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->W_in.rows * block->W_in.cols, t);

            /* W_out: [dim x R] -> [d_state x dim] total elements */
            adamw_step_f32(block->W_out.data, opt->g_W_out,
                          opt->m_W_out, opt->v_W_out,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->W_out.rows * block->W_out.cols, t);

            /* A_log: [state_size] */
            adamw_step_f32(block->A_log.data, opt->g_A_log,
                          opt->m_A_log, opt->v_A_log,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->A_log.rows * block->A_log.cols, t);

            /* W_B: [N*R x dim] */
            adamw_step_f32(block->W_B.data, opt->g_W_B,
                          opt->m_W_B, opt->v_W_B,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->W_B.rows * block->W_B.cols, t);

            /* W_C: [N*R x dim] */
            adamw_step_f32(block->W_C.data, opt->g_W_C,
                          opt->m_W_C, opt->v_W_C,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->W_C.rows * block->W_C.cols, t);

            /* b_B: [state_size] */
            if (block->b_B && opt->g_b_B) {
                adamw_step_f32(block->b_B, opt->g_b_B,
                              opt->m_b_B, opt->v_b_B,
                              conf->lr, beta1, beta2_adam,
                              eps, conf->weight_decay,
                              d_state, t);
            }

            /* b_C: [state_size] */
            if (block->b_C && opt->g_b_C) {
                adamw_step_f32(block->b_C, opt->g_b_C,
                              opt->m_b_C, opt->v_b_C,
                              conf->lr, beta1, beta2_adam,
                              eps, conf->weight_decay,
                              d_state, t);
            }

            /* delta_proj: [1 x dim] */
            adamw_step_f32(block->delta_proj.data, opt->g_delta_proj,
                          opt->m_delta_proj, opt->v_delta_proj,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->delta_proj.rows * block->delta_proj.cols, t);

            /* theta: [state_size/2] */
            if (block->theta && opt->g_theta) {
                adamw_step_f32(block->theta, opt->g_theta,
                              opt->m_theta, opt->v_theta,
                              conf->lr, beta1, beta2_adam,
                              eps, conf->weight_decay,
                              d_state / 2, t);
            }

            /* lambda_proj: [1 x dim] */
            adamw_step_f32(block->lambda_proj.data, opt->g_lambda_proj,
                          opt->m_lambda_proj, opt->v_lambda_proj,
                          conf->lr, beta1, beta2_adam,
                          eps, conf->weight_decay,
                          block->lambda_proj.rows * block->lambda_proj.cols, t);
        }
    }

    free(current);
    free(next);
    free(output_logits);
    free(d_output);

    return result;
}

TrainerMetrics trainer_train_batch_ex(Trainer *tr, const TrainerBatch *batch) {
    TrainerMetrics result = {-1.0f, 0.0f};
    if (!tr || !batch) return result;

    switch (batch->type) {
        case TRAINER_BATCH_TOKENS: {
            /* Language model: delegate to existing token-based function */
            result.loss = trainer_train_batch(tr, batch->tokens.tokens, batch->batch_size);
            result.accuracy = 0.0f; /* No accuracy for LM */
            break;
        }

        case TRAINER_BATCH_VISION: {
            /* Vision model: use accuracy-aware version */
            result = trainer_train_batch_vision_with_acc(tr,
                batch->vision.data,
                batch->vision.labels,
                batch->batch_size,
                batch->vision.seq_len,
                batch->vision.dim,
                batch->vision.num_classes);
            break;
        }

        case TRAINER_BATCH_CUSTOM: {
            /* Custom: user-provided loss function */
            if (batch->custom.loss_fn) {
                float *d_output = NULL;
                result.loss = batch->custom.loss_fn(batch, tr->model, &d_output, batch->custom.user_ctx);
                result.accuracy = 0.0f;
            }
            break;
        }

        default:
            fprintf(stderr, "[trainer] Unknown batch type: %d\n", batch->type);
            break;
    }

    return result;
}

/* ============================================================================
 * CSV Logging Functions
 * ============================================================================ */

void trainer_log_step(Trainer *tr, int epoch, size_t step_in_epoch,
                      float loss, float accuracy, double step_ms) {
    if (!tr || !tr->step_log) return;

    size_t batch_tokens = tr->model->cfg.seq_len * 1; /* Simplified */
    double tokens_per_sec = step_ms > 0.0
        ? (batch_tokens * 1000.0) / step_ms
        : 0.0;

    fprintf(tr->step_log, "%llu,%d,%zu,%zu,%.6f,%.4f,%.6f,%.3f,%.1f,%zu,%.6f\n",
            tr->run_id, epoch, step_in_epoch, tr->resume.global_step + step_in_epoch,
            loss, accuracy, 0.0f, /* grad_norm placeholder */
            step_ms, tokens_per_sec, current_rss_kb(), 0.0f /* lr placeholder */);
    fflush(tr->step_log);
}

void trainer_log_epoch(Trainer *tr, int epoch, size_t total_steps,
                       float avg_loss, float avg_accuracy, double epoch_ms) {
    if (!tr || !tr->epoch_log) return;

    double tokens_per_sec = epoch_ms > 0.0
        ? (total_steps * tr->model->cfg.seq_len * 1000.0) / epoch_ms
        : 0.0;

    fprintf(tr->epoch_log, "%llu,%d,%zu,%.6f,%.4f,%.3f,%.1f,%zu\n",
            tr->run_id, epoch, total_steps,
            avg_loss, avg_accuracy,
            epoch_ms, tokens_per_sec, current_rss_kb());
    fflush(tr->epoch_log);
}
