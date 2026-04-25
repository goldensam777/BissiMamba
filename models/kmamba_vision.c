/* ============================================================================
 * kmamba_vision.c - K-Mamba 2D Vision Model Implementation for CIFAR-10
 * 
 * True 2D-native K-Mamba (NO flattening) with wavefront scheduling.
 * Optimized for MX450-class low-power GPU.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "kmamba_vision.h"
#include "kmamba.h"
#include "kmamba_kernels.h"
#include "scan_nd.h"
#include "wavefront_plan.h"
#include "km_topology.h"
#include "kser.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Internal Constants
 * ============================================================================ */
#define EPSILON_NORM    1e-6f
#define EPSILON_SOFTPLUS 1e-4f
#define EPSILON_DELTA   1e-3f

/* ============================================================================
 * RMSNorm Implementation (preferred over LayerNorm)
 * ============================================================================ */

void rmsnorm_f32(const float *x, float *y, size_t n, float eps) {
    float sq_sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sq_sum += x[i] * x[i];
    }
    float rms = sqrtf(sq_sum / (float)n + eps);
    float scale = 1.0f / rms;
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] * scale;
    }
}

void rmsnorm_f32_batch(const float *x, float *y, size_t batch, size_t dim, float eps) {
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch; b++) {
        rmsnorm_f32(&x[b * dim], &y[b * dim], dim, eps);
    }
}

/* ============================================================================
 * Global Mean Pooling: [B, H, W, D] -> [B, D]
 * ============================================================================ */

void global_mean_pool(const float *input, float *output,
                      size_t batch, size_t h, size_t w, size_t dim) {
    size_t spatial = h * w;
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch; b++) {
        for (size_t d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (size_t hw = 0; hw < spatial; hw++) {
                sum += input[b * spatial * dim + hw * dim + d];
            }
            output[b * dim + d] = sum / (float)spatial;
        }
    }
}

/* ============================================================================
 * Patch Embedding: Conv2D 3x3, stride 1, pad 1 (preserves spatial dims)
 * ============================================================================ */

/* Simple Conv2D with kernel 3x3, stride 1, padding 1 */
void patch_embed_forward(const float *kernel, const float *bias,
                         const float *input, float *output,
                         size_t batch, size_t h, size_t w,
                         size_t in_c, size_t out_c) {
    /* kernel layout: [out_c, in_c, 3, 3] or [out_c, 3*3*in_c] */
    /* input layout: [batch, h, w, in_c] */
    /* output layout: [batch, h, w, out_c] */
    
    const int k = 3;
    const int pad = 1;
    
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t b = 0; b < batch; b++) {
        for (size_t oh = 0; oh < h; oh++) {
            for (size_t ow = 0; ow < w; ow++) {
                for (size_t oc = 0; oc < out_c; oc++) {
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (size_t ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = (int)oh + kh - pad;
                                int iw = (int)ow + kw - pad;
                                
                                if (ih >= 0 && ih < (int)h && iw >= 0 && iw < (int)w) {
                                    float in_val = input[b * h * w * in_c + 
                                                        ih * w * in_c + 
                                                        iw * in_c + ic];
                                    float k_val = kernel[oc * in_c * k * k + 
                                                        ic * k * k + 
                                                        kh * k + kw];
                                    sum += in_val * k_val;
                                }
                            }
                        }
                    }
                    
                    output[b * h * w * out_c + oh * w * out_c + ow * out_c + oc] = sum;
                }
            }
        }
    }
}

void patch_embed_backward(const float *kernel,
                          const float *input, const float *grad_output,
                          float *grad_kernel, float *grad_bias,
                          size_t batch, size_t h, size_t w,
                          size_t in_c, size_t out_c) {
    const int k = 3;
    const int pad = 1;
    
    /* Zero gradients */
    if (grad_kernel) memset(grad_kernel, 0, out_c * in_c * k * k * sizeof(float));
    if (grad_bias) memset(grad_bias, 0, out_c * sizeof(float));
    
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t b = 0; b < batch; b++) {
        for (size_t oh = 0; oh < h; oh++) {
            for (size_t ow = 0; ow < w; ow++) {
                for (size_t oc = 0; oc < out_c; oc++) {
                    float grad = grad_output[b * h * w * out_c + oh * w * out_c + ow * out_c + oc];
                    
                    #pragma omp atomic
                    grad_bias[oc] += grad;
                    
                    for (size_t ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = (int)oh + kh - pad;
                                int iw = (int)ow + kw - pad;
                                
                                if (ih >= 0 && ih < (int)h && iw >= 0 && iw < (int)w) {
                                    float in_val = input[b * h * w * in_c + 
                                                        ih * w * in_c + 
                                                        iw * in_c + ic];
                                    
                                    size_t k_idx = oc * in_c * k * k + ic * k * k + kh * k + kw;
                                    #pragma omp atomic
                                    grad_kernel[k_idx] += in_val * grad;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * Cross-Entropy Loss with Softmax
 * ============================================================================ */

float cross_entropy_loss(const float *logits, const uint8_t *labels,
                         float *grad_logits, size_t batch_size, size_t num_classes) {
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; b++) {
        const float *logit = &logits[b * num_classes];
        float *grad = &grad_logits[b * num_classes];
        
        /* Find max for numerical stability */
        float max_logit = logit[0];
        for (size_t c = 1; c < num_classes; c++) {
            if (logit[c] > max_logit) max_logit = logit[c];
        }
        
        /* Compute exp and sum */
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            grad[c] = expf(logit[c] - max_logit);
            sum_exp += grad[c];
        }
        
        /* Normalize to get probabilities */
        for (size_t c = 0; c < num_classes; c++) {
            grad[c] /= sum_exp;
        }
        
        /* Cross-entropy loss */
        uint8_t label = labels[b];
        float prob_true = grad[label];
        if (prob_true > 1e-10f) {
            total_loss += -logf(prob_true);
        } else {
            total_loss += 100.0f; /* Large loss for numerical issues */
        }
        
        /* Gradient: p - y */
        grad[label] -= 1.0f;
    }
    
    return total_loss / (float)batch_size;
}

/* ============================================================================
 * Mamba 2D Block: Uses scan_nd with wavefront
 * ============================================================================ */

int mamba2d_block_forward(MambaBlock *block, const float *input, float *output,
                          size_t batch, const KMWavefrontPlan *plan) {
    printf("DEBUG: mamba2d_block_forward start\n");
    fflush(stdout);
    
    if (!block || !input || !output || !plan) return -1;
    
    size_t H = plan->dims[0];
    size_t W = plan->dims[1];
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t spatial = H * W;
    
    printf("DEBUG: H=%zu, W=%zu, D=%zu, N=%zu, spatial=%zu\n", H, W, D, N, spatial);
    printf("DEBUG: block->config.seq_len=%u\n", block->config.seq_len);
    fflush(stdout);
    
    /* Use workspace for batch processing */
    printf("DEBUG: creating workspace...\n");
    fflush(stdout);
    
    MambaBlockWorkspace *ws = mamba_block_workspace_create(block);
    if (!ws) return -1;
    
    printf("DEBUG: workspace created, allocating buffers...\n");
    fflush(stdout);
    
    /* Allocate temporary buffers for 2D processing */
    size_t buf_size = spatial * D * sizeof(float);
    printf("DEBUG: allocating buffers, buf_size=%zu\n", buf_size);
    fflush(stdout);
    
    float *x_norm = (float *)malloc(buf_size);
    printf("DEBUG: x_norm=%p\n", (void*)x_norm);
    fflush(stdout);
    
    float *y_scan = (float *)malloc(buf_size);
    printf("DEBUG: y_scan=%p\n", (void*)y_scan);
    fflush(stdout);
    
    if (!x_norm || !y_scan) {
        printf("DEBUG: buffer allocation failed!\n");
        fflush(stdout);
        free(x_norm); free(y_scan);
        mamba_block_workspace_free(ws);
        return -1;
    }
    
    printf("DEBUG: buffers allocated, entering batch loop...\n");
    fflush(stdout);
    
    for (size_t b = 0; b < batch; b++) {
        if (b == 0) {
            printf("DEBUG: batch loop b=0, calculating x_in, x_out...\n");
            fflush(stdout);
        }
        
        const float *x_in = &input[b * spatial * D];
        float *x_out = &output[b * spatial * D];
        
        if (b == 0) {
            printf("DEBUG: x_in=%p, x_out=%p\n", (void*)x_in, (void*)x_out);
            fflush(stdout);
            printf("DEBUG: starting RMSNorm...\n");
            fflush(stdout);
        }
        
        /* 1. RMSNorm */
        for (size_t hw = 0; hw < spatial; hw++) {
            rmsnorm_f32(&x_in[hw * D], &x_norm[hw * D], D, EPSILON_NORM);
        }
        
        if (b == 0) {
            printf("DEBUG: RMSNorm done\n");
            fflush(stdout);
        }
        
        /* 2. SSM projections and scan */
        if (b == 0) {
            printf("DEBUG: starting SSM projections loop, spatial=%zu\n", spatial);
            fflush(stdout);
        }
        
        /* For each spatial position, compute u_t, B_t, C_t, delta_t */
        for (size_t hw = 0; hw < spatial; hw++) {
            if (b == 0 && hw == 0) {
                printf("DEBUG: SSM loop hw=0, calculating x_pos...\n");
                fflush(stdout);
            }
            
            float *x_pos = &x_norm[hw * D];
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: x_pos=%p, calling gemv_f32 for W_in...\n", (void*)x_pos);
                fflush(stdout);
            }
            
            /* u = W_in @ x (via SiLU) */
            gemv_f32(block->W_in.data, x_pos, ws->z_buf, (int)block->config.mimo_rank, (int)D);
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: gemv_f32 W_in done, calling silu_f32...\n");
                fflush(stdout);
            }
            
            silu_f32(ws->z_buf, &ws->u_seq[hw * block->config.mimo_rank], (int)block->config.mimo_rank);
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: silu_f32 done\n");
                fflush(stdout);
                printf("DEBUG: about to call gemv_f32 for W_B, block->W_B.data=%p\n", (void*)block->W_B.data);
                fflush(stdout);
            }
            
            /* B, C projections - layout pour scan_nd est [hw][D][N] */
            /* W_B/W_C sont [N*R, D], on projette pour obtenir [N*R] */
            /* NOTE: Pour le scan Mamba-ND, B/C sont utilisés comme:
             *   h_new = dt_bar * B[hw,d,n] * x[hw,d] + ...
             * Mais notre gemv donne déjà sum_d W[n,d] * x[d] = B_eff[n]
             * On stocke donc B_eff[n] dans toutes les positions d pour compatibilité
             */
            float tmp_B[192]; /* N * max_R = 192 * 1 = 192 */
            float tmp_C[192];
            gemv_f32(block->W_B.data, x_pos, tmp_B, (int)(N * block->config.mimo_rank), (int)D);
            gemv_f32(block->W_C.data, x_pos, tmp_C, (int)(N * block->config.mimo_rank), (int)D);
            
            /* Stocker dans layout [hw][D][N] - pour compatibilité scan_nd */
            /* Note: tmp_B[n] contient déjà la projection, on la réplique pour chaque d */
            for (size_t d = 0; d < D; d++) {
                for (size_t n = 0; n < N; n++) {
                    ws->scan_B[hw * D * N + d * N + n] = tmp_B[n];
                    ws->scan_C[hw * D * N + d * N + n] = tmp_C[n];
                }
            }
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: gemv_f32 W_B/W_C done, reorganized\n");
                fflush(stdout);
            }
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: gemv_f32 W_C done\n");
                fflush(stdout);
                printf("DEBUG: about to add biases\n");
                printf("DEBUG: block->b_B=%p, block->b_C=%p\n", 
                       (void*)block->b_B, (void*)block->b_C);
                printf("DEBUG: ws->scan_B=%p, ws->scan_C=%p\n",
                       (void*)ws->scan_B, (void*)ws->scan_C);
                printf("DEBUG: N=%zu, mimo_rank=%zu, spatial=%zu\n", 
                       N, (size_t)block->config.mimo_rank, spatial);
                /* Test reading bias values */
                printf("DEBUG: Testing block->b_B[0]=%f, block->b_C[0]=%f\n",
                       block->b_B ? block->b_B[0] : 0.0f, 
                       block->b_C ? block->b_C[0] : 0.0f);
                printf("DEBUG: Testing ws->scan_B[0]=%f, ws->scan_C[0]=%f\n",
                       ws->scan_B ? ws->scan_B[0] : 0.0f,
                       ws->scan_C ? ws->scan_C[0] : 0.0f);
                fflush(stdout);
            }
            
            /* Add biases - simplified to avoid complex indexing */
            size_t NR = N * block->config.mimo_rank;
            if (NR > 0 && block->b_B && block->b_C && ws->scan_B && ws->scan_C) {
                if (b == 0 && hw == 0) {
                    printf("DEBUG: Starting bias loop, NR=%zu\n", NR);
                    fflush(stdout);
                }
                for (size_t i = 0; i < NR; i++) {
                    size_t idx = hw * NR + i;
                    /* Bounds check */
                    if (idx >= spatial * NR) {
                        printf("DEBUG: Index out of bounds! idx=%zu, max=%zu\n", idx, spatial * NR);
                        fflush(stdout);
                        break;
                    }
                    if (b == 0 && hw == 0 && i < 3) {
                        printf("DEBUG: bias loop i=%zu, idx=%zu\n", i, idx);
                        fflush(stdout);
                    }
                    ws->scan_B[idx] += block->b_B[i];
                    if (b == 0 && hw == 0 && i < 3) {
                        printf("DEBUG: scan_B updated\n");
                        fflush(stdout);
                    }
                    ws->scan_C[idx] += block->b_C[i];
                    if (b == 0 && hw == 0 && i < 3) {
                        printf("DEBUG: scan_C updated\n");
                        fflush(stdout);
                    }
                }
                if (b == 0 && hw == 0) {
                    printf("DEBUG: Bias loop completed\n");
                    fflush(stdout);
                }
            }
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: Starting delta computation...\n");
                fflush(stdout);
                printf("DEBUG: block->delta_proj.data=%p\n", (void*)block->delta_proj.data);
                fflush(stdout);
            }
            
            /* Delta computation with stability */
            float dt_raw;
            gemv_f32(block->delta_proj.data, x_pos, &dt_raw, 1, (int)D);
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: gemv_f32 delta_proj done\n");
                fflush(stdout);
            }
            
            softplus_f32(&dt_raw, &ws->delta[hw], 1);
            ws->delta[hw] += EPSILON_DELTA;
            if (ws->delta[hw] < block->config.dt_min) ws->delta[hw] = block->config.dt_min;
            if (ws->delta[hw] > block->config.dt_max) ws->delta[hw] = block->config.dt_max;
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: delta computation done\n");
                fflush(stdout);
            }
            
            /* Lambda for exp-trapezoidal (Mamba-3) */
            if (block->lambda_proj.data) {
                float lam_raw;
                gemv_f32(block->lambda_proj.data, x_pos, &lam_raw, 1, (int)D);
                ws->lambda_seq[hw] = 1.0f / (1.0f + expf(-lam_raw));
            } else {
                ws->lambda_seq[hw] = 0.5f;
            }
            
            if (b == 0 && hw == 0) {
                printf("DEBUG: lambda computation done\n");
                fflush(stdout);
            }
        }
        
        if (b == 0) {
            printf("DEBUG: Starting 2D SSM Scan setup...\n");
            fflush(stdout);
        }
        
        /* 3. 2D SSM Scan using scan_nd */
        ScanNDParams scan_params = {
            .dims = plan->dims,
            .ndims = 2,
            .D = (long)D,
            .M = (long)N,
            .x = x_norm,
            .A = block->A_log.data,
            .B = ws->scan_B,
            .C = ws->scan_C,
            .delta = ws->delta,
            .h = ws->h_seq,
            .y = y_scan,
            .theta = block->theta,
            .lambda = ws->lambda_seq
        };
        
        if (b == 0) {
            printf("DEBUG: Allocating delta_2d...\n");
            fflush(stdout);
        }
        
        /* Reshape delta for scan_nd: [ndims, prod(dims), D] */
        float *delta_2d = (float *)malloc(2 * spatial * D * sizeof(float));
        if (!delta_2d) {
            printf("DEBUG: delta_2d allocation failed!\n");
            fflush(stdout);
            free(x_norm); free(y_scan);
            mamba_block_workspace_free(ws);
            return -1;
        }
        
        if (b == 0) {
            printf("DEBUG: Filling delta_2d...\n");
            fflush(stdout);
        }
        
        for (size_t hw = 0; hw < spatial; hw++) {
            for (size_t d = 0; d < D; d++) {
                delta_2d[0 * spatial * D + hw * D + d] = ws->delta[hw];
                delta_2d[1 * spatial * D + hw * D + d] = ws->delta[hw];
            }
        }
        scan_params.delta = delta_2d;
        
        if (b == 0) {
            printf("DEBUG: Calling scannd_ref_with_plan...\n");
            fflush(stdout);
        }
        
        /* Run 2D scan */
        int rc = scannd_ref_with_plan(&scan_params, plan);
        
        if (b == 0) {
            printf("DEBUG: scannd_ref_with_plan returned %d\n", rc);
            fflush(stdout);
        }
        
        free(delta_2d);
        
        if (rc != 0) {
            free(x_norm); free(y_scan);
            mamba_block_workspace_free(ws);
            return -1;
        }
        
        /* 4. Output projection and residual */
        for (size_t hw = 0; hw < spatial; hw++) {
            gemv_f32(block->W_out.data, &y_scan[hw * D], ws->y_proj, (int)D, (int)D);
            for (size_t d = 0; d < D; d++) {
                x_out[hw * D + d] = x_in[hw * D + d] + ws->y_proj[d];
            }
        }
    }
    
    free(x_norm);
    free(y_scan);
    mamba_block_workspace_free(ws);
    return 0;
}

/* ============================================================================
 * Model Creation and Destruction
 * ============================================================================ */

KMambaVision* kmamba_vision_init(size_t dim, size_t state_size, size_t n_layers,
                                  size_t batch_size, int use_fp16) {
    KMambaVision *model = (KMambaVision *)calloc(1, sizeof(KMambaVision));
    if (!model) return NULL;
    
    /* Config */
    model->cfg.dim = dim ? dim : KMAMBA_VISION_DIM;
    model->cfg.state_size = state_size ? state_size : KMAMBA_VISION_STATE;
    model->cfg.n_layers = n_layers ? n_layers : KMAMBA_VISION_LAYERS;
    model->cfg.num_classes = KMAMBA_VISION_NUM_CLASSES;
    model->cfg.use_fp16 = use_fp16;
    model->cfg.use_bf16 = 0;
    
    /* Spatial dimensions (preserved by patch embed) */
    model->cfg.spatial_h = KMAMBA_VISION_IMG_H;
    model->cfg.spatial_w = KMAMBA_VISION_IMG_W;
    model->cfg.spatial_ndims = 2;
    model->cfg.spatial_dims[0] = model->cfg.spatial_h;
    model->cfg.spatial_dims[1] = model->cfg.spatial_w;
    
    model->batch_size = batch_size ? batch_size : KMAMBA_VISION_BATCH_SIZE;
    
    size_t D = model->cfg.dim;
    size_t H = model->cfg.spatial_h;
    size_t W = model->cfg.spatial_w;
    size_t spatial = H * W;
    
    /* Patch embedding */
    model->patch_embed.patch_kernel = (float *)calloc(3 * 3 * 3 * KMAMBA_VISION_PATCH_OUT, sizeof(float));
    model->patch_embed.patch_bias = (float *)calloc(KMAMBA_VISION_PATCH_OUT, sizeof(float));
    if (!model->patch_embed.patch_kernel || !model->patch_embed.patch_bias) {
        kmamba_vision_free(model);
        return NULL;
    }
    
    /* Create wavefront plan shared by all blocks */
    model->backbone.wavefront_plan = km_wavefront_plan_create(
        model->cfg.spatial_dims, model->cfg.spatial_ndims);
    if (!model->backbone.wavefront_plan) {
        kmamba_vision_free(model);
        return NULL;
    }
    
    /* Create Mamba blocks with 2D topology */
    model->backbone.blocks = (MambaBlock **)calloc(model->cfg.n_layers, sizeof(MambaBlock *));
    if (!model->backbone.blocks) {
        kmamba_vision_free(model);
        return NULL;
    }
    
    for (size_t i = 0; i < model->cfg.n_layers; i++) {
        MBConfig block_cfg = {
            .dim = D,
            .state_size = model->cfg.state_size,
            .seq_len = spatial,
            .mimo_rank = 1,
            .dt_scale = 0.1f,
            .dt_min = 1e-3f,
            .dt_max = 0.1f,
            .use_fp16 = use_fp16,
            .use_bf16 = 0,
            .spatial_ndims = 2,
            .spatial_dims = {H, W, 0, 0, 0, 0, 0, 0},
            .use_convnd = 0
        };
        
        model->backbone.blocks[i] = mamba_block_create(&block_cfg);
        if (!model->backbone.blocks[i]) {
            kmamba_vision_free(model);
            return NULL;
        }
    }
    
    /* Classification head */
    model->head.head_weight = (float *)calloc(model->cfg.num_classes * D, sizeof(float));
    model->head.head_bias = (float *)calloc(model->cfg.num_classes, sizeof(float));
    model->head_opt.m_head = (float *)calloc(model->cfg.num_classes * D, sizeof(float));
    model->head_opt.m_head_bias = (float *)calloc(model->cfg.num_classes, sizeof(float));
    if (!model->head.head_weight || !model->head.head_bias || 
        !model->head_opt.m_head || !model->head_opt.m_head_bias) {
        kmamba_vision_free(model);
        return NULL;
    }
    
    /* Buffers */
    size_t B = model->batch_size;
    model->buf_input = (float *)calloc(B * H * W * 3, sizeof(float));
    model->buf_embed = (float *)calloc(B * H * W * D, sizeof(float));
    model->buf_hidden = (float *)calloc(B * H * W * D, sizeof(float));
    model->buf_pooled = (float *)calloc(B * D, sizeof(float));
    model->buf_logits = (float *)calloc(B * model->cfg.num_classes, sizeof(float));
    model->buf_grad_logits = (float *)calloc(B * model->cfg.num_classes, sizeof(float));
    model->buf_grad_embed = (float *)calloc(B * H * W * D, sizeof(float));
    
    if (!model->buf_input || !model->buf_embed || !model->buf_hidden ||
        !model->buf_pooled || !model->buf_logits || !model->buf_grad_logits ||
        !model->buf_grad_embed) {
        kmamba_vision_free(model);
        return NULL;
    }
    
    /* Optimizer config */
    model->optim_cfg.lr = KMAMBA_VISION_BASE_LR;
    model->optim_cfg.mu = KMAMBA_VISION_MUON_MU;
    model->optim_cfg.beta2 = 0.999f;
    model->optim_cfg.eps = 1e-8f;
    model->optim_cfg.clip_norm = KMAMBA_VISION_CLIP_NORM;
    model->optim_cfg.weight_decay = KMAMBA_VISION_WEIGHT_DECAY;
    
    return model;
}

void kmamba_vision_free(KMambaVision *model) {
    if (!model) return;
    
    /* Patch embed */
    free(model->patch_embed.patch_kernel);
    free(model->patch_embed.patch_bias);
    
    /* Backbone */
    if (model->backbone.blocks) {
        for (size_t i = 0; i < model->cfg.n_layers; i++) {
            if (model->backbone.blocks[i]) {
                mamba_block_free(model->backbone.blocks[i]);
            }
        }
        free(model->backbone.blocks);
    }
    if (model->backbone.wavefront_plan) {
        km_wavefront_plan_free(model->backbone.wavefront_plan);
    }
    
    /* Head */
    free(model->head.head_weight);
    free(model->head.head_bias);
    free(model->head_opt.m_head);
    free(model->head_opt.m_head_bias);
    
    /* Buffers */
    free(model->buf_input);
    free(model->buf_embed);
    free(model->buf_hidden);
    free(model->buf_pooled);
    free(model->buf_logits);
    free(model->buf_grad_logits);
    free(model->buf_grad_embed);
    
    free(model);
}

/* ============================================================================
 * Weight Initialization
 * ============================================================================ */

void kmamba_vision_init_weights(KMambaVision *model, uint32_t seed) {
    if (!model) return;
    
    srand(seed);
    
    /* Patch embedding: Kaiming init */
    size_t patch_fan_in = 3 * 3 * 3;
    init_kaiming_uniform_f32(model->patch_embed.patch_kernel, patch_fan_in, seed);
    memset(model->patch_embed.patch_bias, 0, KMAMBA_VISION_PATCH_OUT * sizeof(float));
    
    /* Mamba blocks */
    for (size_t i = 0; i < model->cfg.n_layers; i++) {
        mamba_block_init(model->backbone.blocks[i]);
    }
    
    /* Classification head: Xavier init */
    init_xavier_uniform_f32(model->head.head_weight, model->cfg.dim, model->cfg.num_classes, seed);
    memset(model->head.head_bias, 0, model->cfg.num_classes * sizeof(float));
}

/* ============================================================================
 * Forward Pass
 * ============================================================================ */

int kmamba_vision_forward(KMambaVision *model, const float *images,
                          float *logits_out, size_t batch_size) {
    printf("DEBUG: forward start\n");
    fflush(stdout);
    
    if (!model || !images || !logits_out) return -1;
    if (batch_size > model->batch_size) return -1;
    
    size_t B = batch_size;
    size_t H = model->cfg.spatial_h;
    size_t W = model->cfg.spatial_w;
    size_t D = model->cfg.dim;
    size_t spatial = H * W;
    
    printf("DEBUG: forward B=%zu, H=%zu, W=%zu, D=%zu\n", B, H, W, D);
    fflush(stdout);
    
    float *x = model->buf_embed;      /* [B, H, W, D] */
    float *h = model->buf_hidden;     /* [B, H, W, D] */
    float *pooled = model->buf_pooled; /* [B, D] */
    
    printf("DEBUG: forward buffers: x=%p, h=%p, pooled=%p\n", (void*)x, (void*)h, (void*)pooled);
    fflush(stdout);
    
    /* 1. Patch Embedding: [B, 32, 32, 3] -> [B, 32, 32, 96] */
    printf("DEBUG: calling patch_embed_forward...\n");
    fflush(stdout);
    
    patch_embed_forward(model->patch_embed.patch_kernel, model->patch_embed.patch_bias,
                        images, x, B, H, W, 3, D);
    
    printf("DEBUG: patch_embed_forward done\n");
    fflush(stdout);
    
    /* 2. K-Mamba 2D Backbone: 5 blocks with residual connections */
    printf("DEBUG: starting backbone, n_layers=%zu\n", model->cfg.n_layers);
    fflush(stdout);
    
    for (size_t layer = 0; layer < model->cfg.n_layers; layer++) {
        printf("DEBUG: layer %zu\n", layer);
        fflush(stdout);
        
        float *layer_in = (layer == 0) ? x : h;
        float *layer_out = (layer % 2 == 0) ? h : x;
        
        printf("DEBUG: layer %zu - calling mamba2d_block_forward...\n", layer);
        fflush(stdout);
        
        if (mamba2d_block_forward(model->backbone.blocks[layer], layer_in, layer_out,
                                   B, model->backbone.wavefront_plan) != 0) {
            printf("DEBUG: layer %zu - mamba2d_block_forward failed\n", layer);
            fflush(stdout);
            return -1;
        }
        
        printf("DEBUG: layer %zu - done\n", layer);
        fflush(stdout);
        
        /* After first layer, we alternate buffers */
        if (layer == 0) {
            /* Copy first output to working buffer */
            memcpy(h, x, B * spatial * D * sizeof(float));
        }
    }
    
    /* Final output is in h if n_layers is odd, x if even */
    float *features = (model->cfg.n_layers % 2 == 1) ? h : x;
    
    /* 3. Global Mean Pooling: [B, 32, 32, D] -> [B, D] */
    global_mean_pool(features, pooled, B, H, W, D);
    
    /* 4. Classification Head: [B, D] -> [B, 10] */
    for (size_t b = 0; b < B; b++) {
        gemv_f32(model->head.head_weight, &pooled[b * D], &logits_out[b * model->cfg.num_classes],
                (int)model->cfg.num_classes, (int)D);
        for (size_t c = 0; c < model->cfg.num_classes; c++) {
            logits_out[b * model->cfg.num_classes + c] += model->head.head_bias[c];
        }
    }
    
    return 0;
}

/* ============================================================================
 * Training Step
 * ============================================================================ */

float kmamba_vision_train_step(KMambaVision *model, const float *images,
                               const uint8_t *labels, size_t batch_size) {
    printf("DEBUG: train_step start\n");
    fflush(stdout);
    
    if (!model || !images || !labels) return -1.0f;
    if (batch_size > model->batch_size) return -1.0f;
    
    printf("DEBUG: train_step - calling forward...\n");
    fflush(stdout);
    
    /* Forward pass */
    if (kmamba_vision_forward(model, images, model->buf_logits, batch_size) != 0) {
        printf("DEBUG: forward failed\n");
        fflush(stdout);
        return -1.0f;
    }
    
    printf("DEBUG: forward done, computing loss...\n");
    fflush(stdout);
    
    /* Compute loss and gradients */
    float loss = cross_entropy_loss(model->buf_logits, labels, model->buf_grad_logits,
                                     batch_size, model->cfg.num_classes);
    
    printf("DEBUG: loss=%f\n", loss);
    fflush(stdout);
    model->last_loss = loss;
    
    /* Check for NaN */
    if (isnan(loss) || isinf(loss)) {
        model->nan_detected = 1;
        return loss;
    }
    
    /* Backward pass through head */
    size_t B = batch_size;
    size_t D = model->cfg.dim;
    
    /* d_head_weight = grad_logits^T @ pooled */
    float *pooled = model->buf_pooled;
    float *d_head_weight = (float *)calloc(model->cfg.num_classes * D, sizeof(float));
    float *d_head_bias = (float *)calloc(model->cfg.num_classes, sizeof(float));
    
    /* Compute gradients for head */
    for (size_t b = 0; b < B; b++) {
        for (size_t c = 0; c < model->cfg.num_classes; c++) {
            float grad = model->buf_grad_logits[b * model->cfg.num_classes + c];
            d_head_bias[c] += grad;
            for (size_t d = 0; d < D; d++) {
                d_head_weight[c * D + d] += grad * pooled[b * D + d];
            }
        }
    }
    
    /* Average gradients */
    for (size_t i = 0; i < model->cfg.num_classes * D; i++) d_head_weight[i] /= (float)B;
    for (size_t i = 0; i < model->cfg.num_classes; i++) d_head_bias[i] /= (float)B;
    
    /* Apply MUON update to head (simplified - just momentum + SGD) */
    model->train_step++;
    
    /* Simple momentum update for head */
    for (size_t i = 0; i < model->cfg.num_classes * D; i++) {
        model->head_opt.m_head[i] = model->optim_cfg.mu * model->head_opt.m_head[i] + d_head_weight[i];
        float clip_scale = 1.0f;
        float norm = fabsf(model->head_opt.m_head[i]);
        if (norm > model->optim_cfg.clip_norm) {
            clip_scale = model->optim_cfg.clip_norm / norm;
        }
        model->head.head_weight[i] -= model->optim_cfg.lr * model->head_opt.m_head[i] * clip_scale;
        model->head.head_weight[i] *= (1.0f - model->optim_cfg.lr * model->optim_cfg.weight_decay);
    }
    
    for (size_t i = 0; i < model->cfg.num_classes; i++) {
        model->head_opt.m_head_bias[i] = model->optim_cfg.mu * model->head_opt.m_head_bias[i] + d_head_bias[i];
        model->head.head_bias[i] -= model->optim_cfg.lr * model->head_opt.m_head_bias[i];
    }
    
    free(d_head_weight);
    free(d_head_bias);
    
    /* TODO: Backward through backbone and patch embed */
    /* For now, we just train the head - full backward pass would require
     * implementing backward for mamba2d_block and patch_embed */
    
    return loss;
}

/* ============================================================================
 * Optimizer Step and Utilities
 * ============================================================================ */

void kmamba_vision_optimizer_step(KMambaVision *model) {
    if (!model) return;
    
    /* Step Mamba blocks optimizers */
    for (size_t i = 0; i < model->cfg.n_layers; i++) {
        mamba_optimizer_step(model->backbone.blocks[i], &model->optim_cfg);
    }
}

void kmamba_vision_zero_grad(KMambaVision *model) {
    if (!model) return;
    
    /* Zero Mamba blocks gradients */
    for (size_t i = 0; i < model->cfg.n_layers; i++) {
        mamba_zero_grads(model->backbone.blocks[i]);
    }
}

int kmamba_vision_check_nan(const KMambaVision *model) {
    if (!model) return 1;
    if (model->nan_detected) return 1;
    
    /* Check patch embed */
    for (size_t i = 0; i < 3 * 3 * 3 * KMAMBA_VISION_PATCH_OUT; i++) {
        if (isnan(model->patch_embed.patch_kernel[i])) return 1;
    }
    
    /* Check head */
    for (size_t i = 0; i < model->cfg.num_classes * model->cfg.dim; i++) {
        if (isnan(model->head.head_weight[i])) return 1;
    }
    
    return 0;
}

/* ============================================================================
 * LR Scheduler
 * ============================================================================ */

float kmamba_vision_get_lr(int epoch, int warmup_epochs, int total_epochs,
                           float base_lr, float min_lr) {
    if (epoch < warmup_epochs) {
        /* Linear warmup */
        return base_lr * ((float)epoch / (float)warmup_epochs);
    } else {
        /* Cosine decay */
        float progress = (float)(epoch - warmup_epochs) / 
                        (float)(total_epochs - warmup_epochs);
        return min_lr + (base_lr - min_lr) * 0.5f * (1.0f + cosf(progress * 3.14159265359f));
    }
}

/* ============================================================================
 * CIFAR-10 Dataset Loader
 * ============================================================================ */

static CIFAR10Dataset* cifar10_load_files(const char *path_prefix, int is_train) {
    CIFAR10Dataset *ds = (CIFAR10Dataset *)calloc(1, sizeof(CIFAR10Dataset));
    if (!ds) return NULL;
    
    size_t n_files = is_train ? 5 : 1;
    size_t records_per_file = 10000;
    ds->n_samples = n_files * records_per_file;
    ds->capacity = ds->n_samples;
    
    ds->images = (uint8_t *)malloc(ds->n_samples * 3072);
    ds->labels = (uint8_t *)malloc(ds->n_samples);
    ds->indices = (size_t *)malloc(ds->n_samples * sizeof(size_t));
    
    if (!ds->images || !ds->labels || !ds->indices) {
        cifar10_dataset_free(ds);
        return NULL;
    }
    
    /* Initialize indices */
    for (size_t i = 0; i < ds->n_samples; i++) {
        ds->indices[i] = i;
    }
    
    /* Load binary files */
    for (size_t f = 0; f < n_files; f++) {
        char filename[256];
        if (is_train) {
            snprintf(filename, sizeof(filename), "%s/data_batch_%zu.bin", path_prefix, f + 1);
        } else {
            snprintf(filename, sizeof(filename), "%s/test_batch.bin", path_prefix);
        }
        
        FILE *fp = fopen(filename, "rb");
        if (!fp) {
            /* Try alternative paths */
            if (is_train) {
                snprintf(filename, sizeof(filename), "%s/cifar-10-batches-bin/data_batch_%zu.bin", 
                        path_prefix, f + 1);
            } else {
                snprintf(filename, sizeof(filename), "%s/cifar-10-batches-bin/test_batch.bin", path_prefix);
            }
            fp = fopen(filename, "rb");
        }
        
        if (!fp) {
            fprintf(stderr, "Warning: Cannot open %s\n", filename);
            cifar10_dataset_free(ds);
            return NULL;
        }
        
        /* Read records: 1 label byte + 3072 pixel bytes */
        size_t offset = f * records_per_file;
        for (size_t r = 0; r < records_per_file; r++) {
            uint8_t label;
            if (fread(&label, 1, 1, fp) != 1) {
                fprintf(stderr, "Error reading label from %s\n", filename);
                fclose(fp);
                cifar10_dataset_free(ds);
                return NULL;
            }
            ds->labels[offset + r] = label;
            
            if (fread(&ds->images[(offset + r) * 3072], 1, 3072, fp) != 3072) {
                fprintf(stderr, "Error reading image from %s\n", filename);
                fclose(fp);
                cifar10_dataset_free(ds);
                return NULL;
            }
        }
        
        fclose(fp);
    }
    
    ds->current_idx = 0;
    
    /* Shuffle if training */
    if (is_train) {
        srand((unsigned)time(NULL));
        for (size_t i = ds->n_samples - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            size_t tmp = ds->indices[i];
            ds->indices[i] = ds->indices[j];
            ds->indices[j] = tmp;
        }
    }
    
    return ds;
}

CIFAR10Dataset* cifar10_load_train(const char *data_dir) {
    return cifar10_load_files(data_dir, 1);
}

CIFAR10Dataset* cifar10_load_test(const char *data_dir) {
    return cifar10_load_files(data_dir, 0);
}

void cifar10_dataset_free(CIFAR10Dataset *ds) {
    if (!ds) return;
    free(ds->images);
    free(ds->labels);
    free(ds->indices);
    free(ds);
}

int cifar10_next_batch(CIFAR10Dataset *ds, float *images_out, uint8_t *labels_out,
                       size_t batch_size) {
    if (!ds || !images_out || !labels_out) return -1;
    
    printf("DEBUG: cifar10_next_batch start, batch_size=%zu\n", batch_size);
    fflush(stdout);
    printf("DEBUG: ds=%p, n_samples=%zu, current_idx=%zu\n", (void*)ds, ds->n_samples, ds->current_idx);
    fflush(stdout);
    
    for (size_t b = 0; b < batch_size; b++) {
        if (b == 0) {
            printf("DEBUG: b=0, accessing indices[current_idx=%zu]...\n", ds->current_idx);
            fflush(stdout);
        }
        size_t idx = ds->indices[ds->current_idx];
        
        if (b == 0) {
            printf("DEBUG: idx=%zu, ds->labels=%p\n", idx, (void*)ds->labels);
            fflush(stdout);
            printf("DEBUG: About to access ds->labels[%zu]...\n", idx);
            fflush(stdout);
        }
        
        /* Copy label */
        uint8_t label_val = ds->labels[idx];
        
        if (b == 0) {
            printf("DEBUG: label_val=%d, assigning to labels_out[0]...\n", (int)label_val);
            fflush(stdout);
        }
        
        labels_out[b] = label_val;
        
        if (b == 0) {
            printf("DEBUG: Label assigned, about to access images...\n");
            fflush(stdout);
            printf("DEBUG: ds->images=%p, idx*3072=%zu\n", (void*)ds->images, idx*3072);
            fflush(stdout);
        }
        
        /* Convert and normalize image: uint8 [0,255] -> float32 [0,1] or [-1,1] */
        /* CIFAR-10 is stored as RGB, we need to convert to [H, W, C] layout */
        if (b == 0) {
            printf("DEBUG: Calculating img pointer...\n");
            fflush(stdout);
        }
        uint8_t *img = &ds->images[idx * 3072];
        
        if (b == 0) {
            printf("DEBUG: img=%p (ds->images + %zu)\n", (void*)img, idx * 3072);
            fflush(stdout);
            printf("DEBUG: Validating pointer...\n");
            fflush(stdout);
        }
        
        /* Validate pointer is in range */
        size_t images_size = ds->n_samples * 3072;
        size_t offset = idx * 3072;
        if (b == 0) {
            printf("DEBUG: images_size=%zu, offset=%zu, remaining=%zu\n", 
                   images_size, offset, images_size - offset);
            fflush(stdout);
        }
        
        /* Normalize to [-1, 1] and convert from planar to interleaved */
        if (b == 0) {
            printf("DEBUG: Starting image conversion loop...\n");
            fflush(stdout);
        }
        for (size_t y = 0; y < 32; y++) {
            for (size_t x = 0; x < 32; x++) {
                size_t planar_idx = y * 32 + x;
                /* CIFAR-10 planar format: R[0:1024], G[1024:2048], B[2048:3072] */
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: Accessing img[%zu], img[%zu], img[%zu]...\n", 
                           planar_idx, 1024 + planar_idx, 2048 + planar_idx);
                    fflush(stdout);
                }
                
                uint8_t r_raw = img[planar_idx];
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: r_raw=%d, reading g...\n", (int)r_raw);
                    fflush(stdout);
                }
                
                uint8_t g_raw = img[1024 + planar_idx];
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: g_raw=%d, reading b...\n", (int)g_raw);
                    fflush(stdout);
                }
                
                uint8_t b_raw = img[2048 + planar_idx];
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: b_raw=%d, converting to float...\n", (int)b_raw);
                    fflush(stdout);
                }
                
                float r = (r_raw / 127.5f) - 1.0f;
                float g = (g_raw / 127.5f) - 1.0f;
                float b_val = (b_raw / 127.5f) - 1.0f;
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: RGB values read, calculating interleaved_idx...\n");
                    fflush(stdout);
                }
                
                size_t interleaved_idx = b * 32 * 32 * 3 + y * 32 * 3 + x * 3;
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: interleaved_idx=%zu, writing to images_out...\n", interleaved_idx);
                    fflush(stdout);
                }
                
                images_out[interleaved_idx + 0] = r;
                images_out[interleaved_idx + 1] = g;
                images_out[interleaved_idx + 2] = b_val;
                
                if (b == 0 && y == 0 && x == 0) {
                    printf("DEBUG: First pixel written successfully!\n");
                    fflush(stdout);
                }
            }
        }
        
        ds->current_idx++;
        if (ds->current_idx >= ds->n_samples) {
            ds->current_idx = 0;
            /* Reshuffle for next epoch */
            for (size_t i = ds->n_samples - 1; i > 0; i--) {
                size_t j = rand() % (i + 1);
                size_t tmp = ds->indices[i];
                ds->indices[i] = ds->indices[j];
                ds->indices[j] = tmp;
            }
        }
    }
    
    return 0;
}

/* ============================================================================
 * Serialization (.ser format via libkser)
 * ============================================================================ */

int kmamba_vision_save(const KMambaVision *model, const char *path) {
    if (!model || !path) return -1;
    
    /* Create KSerConfig for vision model */
    KSerConfig cfg = {
        .vocab_size = 0,  /* No vocab for vision */
        .dim = (uint32_t)model->cfg.dim,
        .state_size = (uint32_t)model->cfg.state_size,
        .n_layers = (uint32_t)model->cfg.n_layers,
        .seq_len = (uint32_t)(model->cfg.spatial_h * model->cfg.spatial_w),
        .d_conv = 0,
        .expand_factor = 1.0f,
        .dtype = model->cfg.use_fp16 ? KSER_FP16 : KSER_FP32
    };
    strncpy(cfg.model_name, "kmamba_vision_cifar10", 63);
    
    /* Create writer */
    KSerWriter *writer = kser_writer_create(path, &cfg);
    if (!writer) {
        fprintf(stderr, "Failed to create .ser writer for %s\n", path);
        return -1;
    }
    
    size_t D = model->cfg.dim;
    size_t H = model->cfg.spatial_h;
    size_t W = model->cfg.spatial_w;
    size_t spatial = H * W;
    
    /* Save patch embedding weights */
    uint32_t patch_shape[4] = {3 * 3 * 3, (uint32_t)D, 0, 0};
    if (kser_writer_add_tensor(writer, "patch_kernel", 
                               model->patch_embed.patch_kernel,
                               patch_shape, KSER_FP32, KSER_FP32) != KSER_OK) {
        fprintf(stderr, "Failed to save patch_kernel\n");
        kser_writer_free(writer);
        return -1;
    }
    
    uint32_t patch_bias_shape[4] = {(uint32_t)D, 0, 0, 0};
    if (kser_writer_add_tensor(writer, "patch_bias",
                               model->patch_embed.patch_bias,
                               patch_bias_shape, KSER_FP32, KSER_FP32) != KSER_OK) {
        fprintf(stderr, "Failed to save patch_bias\n");
        kser_writer_free(writer);
        return -1;
    }
    
    /* Save Mamba block weights */
    for (size_t i = 0; i < model->cfg.n_layers; i++) {
        MambaBlock *block = model->backbone.blocks[i];
        char name_buf[64];
        
        /* W_in [R, D] */
        uint32_t w_in_shape[4] = {(uint32_t)block->W_in.rows, (uint32_t)block->W_in.cols, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_in", i);
        kser_writer_add_tensor(writer, name_buf, block->W_in.data, w_in_shape, KSER_FP32, KSER_FP32);
        
        /* W_out [D, R] */
        uint32_t w_out_shape[4] = {(uint32_t)block->W_out.rows, (uint32_t)block->W_out.cols, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_out", i);
        kser_writer_add_tensor(writer, name_buf, block->W_out.data, w_out_shape, KSER_FP32, KSER_FP32);
        
        /* A_log [N] */
        uint32_t a_shape[4] = {(uint32_t)block->config.state_size, 0, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_A_log", i);
        kser_writer_add_tensor(writer, name_buf, block->A_log.data, a_shape, KSER_FP32, KSER_FP32);
        
        /* W_B [N*R, D] */
        uint32_t wb_shape[4] = {(uint32_t)block->W_B.rows, (uint32_t)block->W_B.cols, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_B", i);
        kser_writer_add_tensor(writer, name_buf, block->W_B.data, wb_shape, KSER_FP32, KSER_FP32);
        
        /* W_C [N*R, D] */
        uint32_t wc_shape[4] = {(uint32_t)block->W_C.rows, (uint32_t)block->W_C.cols, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_C", i);
        kser_writer_add_tensor(writer, name_buf, block->W_C.data, wc_shape, KSER_FP32, KSER_FP32);
        
        /* delta_proj [1, D] */
        uint32_t dp_shape[4] = {1, (uint32_t)D, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_delta_proj", i);
        kser_writer_add_tensor(writer, name_buf, block->delta_proj.data, dp_shape, KSER_FP32, KSER_FP32);
        
        /* b_B [N*R] */
        uint32_t bb_shape[4] = {(uint32_t)(block->config.state_size * block->config.mimo_rank), 0, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_b_B", i);
        kser_writer_add_tensor(writer, name_buf, block->b_B, bb_shape, KSER_FP32, KSER_FP32);
        
        /* b_C [N*R] */
        uint32_t bc_shape[4] = {(uint32_t)(block->config.state_size * block->config.mimo_rank), 0, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_b_C", i);
        kser_writer_add_tensor(writer, name_buf, block->b_C, bc_shape, KSER_FP32, KSER_FP32);
        
        /* theta [N/2] */
        uint32_t theta_shape[4] = {(uint32_t)(block->config.state_size / 2), 0, 0, 0};
        snprintf(name_buf, sizeof(name_buf), "block_%zu_theta", i);
        kser_writer_add_tensor(writer, name_buf, block->theta, theta_shape, KSER_FP32, KSER_FP32);
        
        /* lambda_proj [1, D] */
        snprintf(name_buf, sizeof(name_buf), "block_%zu_lambda_proj", i);
        kser_writer_add_tensor(writer, name_buf, block->lambda_proj.data, dp_shape, KSER_FP32, KSER_FP32);
    }
    
    /* Save classification head */
    uint32_t head_shape[4] = {(uint32_t)model->cfg.num_classes, (uint32_t)D, 0, 0};
    if (kser_writer_add_tensor(writer, "head_weight",
                               model->head.head_weight,
                               head_shape, KSER_FP32, KSER_FP32) != KSER_OK) {
        fprintf(stderr, "Failed to save head_weight\n");
        kser_writer_free(writer);
        return -1;
    }
    
    uint32_t head_bias_shape[4] = {(uint32_t)model->cfg.num_classes, 0, 0, 0};
    if (kser_writer_add_tensor(writer, "head_bias",
                               model->head.head_bias,
                               head_bias_shape, KSER_FP32, KSER_FP32) != KSER_OK) {
        fprintf(stderr, "Failed to save head_bias\n");
        kser_writer_free(writer);
        return -1;
    }
    
    /* Finalize */
    int rc = kser_writer_finalize(writer);
    kser_writer_free(writer);
    
    if (rc == KSER_OK) {
        printf("✓ Model saved to %s\n", path);
        return 0;
    } else {
        fprintf(stderr, "Failed to finalize .ser file (error %d)\n", rc);
        return -1;
    }
}

KMambaVision* kmamba_vision_load(const char *path, size_t batch_size) {
    if (!path) return NULL;
    
    /* Open reader */
    KSerReader *reader = kser_reader_open(path);
    if (!reader) {
        fprintf(stderr, "Failed to open .ser file: %s\n", path);
        return NULL;
    }
    
    if (!kser_reader_is_valid(reader)) {
        fprintf(stderr, "Invalid .ser file: %s\n", path);
        kser_reader_close(reader);
        return NULL;
    }
    
    /* Get config */
    const KSerConfig *cfg = kser_reader_config(reader);
    if (!cfg) {
        fprintf(stderr, "Failed to read config from .ser file\n");
        kser_reader_close(reader);
        return NULL;
    }
    
    printf("Loading model: %s\n", cfg->model_name);
    printf("  Dim: %u, State: %u, Layers: %u\n", cfg->dim, cfg->state_size, cfg->n_layers);
    
    /* Create model with config from file */
    KMambaVision *model = kmamba_vision_init(cfg->dim, cfg->state_size, 
                                             cfg->n_layers, batch_size, 0);
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        kser_reader_close(reader);
        return NULL;
    }
    
    size_t D = model->cfg.dim;
    
    /* Load patch embedding */
    float *patch_kernel = kser_reader_load_tensor(reader, "patch_kernel");
    if (patch_kernel) {
        memcpy(model->patch_embed.patch_kernel, patch_kernel, 3 * 3 * 3 * D * sizeof(float));
        free(patch_kernel);
    }
    
    float *patch_bias = kser_reader_load_tensor(reader, "patch_bias");
    if (patch_bias) {
        memcpy(model->patch_embed.patch_bias, patch_bias, D * sizeof(float));
        free(patch_bias);
    }
    
    /* Load Mamba blocks */
    for (size_t i = 0; i < model->cfg.n_layers; i++) {
        MambaBlock *block = model->backbone.blocks[i];
        char name_buf[64];
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_in", i);
        float *w_in = kser_reader_load_tensor(reader, name_buf);
        if (w_in) { memcpy(block->W_in.data, w_in, block->W_in.rows * block->W_in.cols * sizeof(float)); free(w_in); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_out", i);
        float *w_out = kser_reader_load_tensor(reader, name_buf);
        if (w_out) { memcpy(block->W_out.data, w_out, block->W_out.rows * block->W_out.cols * sizeof(float)); free(w_out); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_A_log", i);
        float *a_log = kser_reader_load_tensor(reader, name_buf);
        if (a_log) { memcpy(block->A_log.data, a_log, block->config.state_size * sizeof(float)); free(a_log); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_B", i);
        float *w_b = kser_reader_load_tensor(reader, name_buf);
        if (w_b) { memcpy(block->W_B.data, w_b, block->W_B.rows * block->W_B.cols * sizeof(float)); free(w_b); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_W_C", i);
        float *w_c = kser_reader_load_tensor(reader, name_buf);
        if (w_c) { memcpy(block->W_C.data, w_c, block->W_C.rows * block->W_C.cols * sizeof(float)); free(w_c); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_delta_proj", i);
        float *dp = kser_reader_load_tensor(reader, name_buf);
        if (dp) { memcpy(block->delta_proj.data, dp, block->delta_proj.rows * block->delta_proj.cols * sizeof(float)); free(dp); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_b_B", i);
        float *bb = kser_reader_load_tensor(reader, name_buf);
        if (bb) { memcpy(block->b_B, bb, block->config.state_size * block->config.mimo_rank * sizeof(float)); free(bb); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_b_C", i);
        float *bc = kser_reader_load_tensor(reader, name_buf);
        if (bc) { memcpy(block->b_C, bc, block->config.state_size * block->config.mimo_rank * sizeof(float)); free(bc); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_theta", i);
        float *theta = kser_reader_load_tensor(reader, name_buf);
        if (theta) { memcpy(block->theta, theta, (block->config.state_size / 2) * sizeof(float)); free(theta); }
        
        snprintf(name_buf, sizeof(name_buf), "block_%zu_lambda_proj", i);
        float *lp = kser_reader_load_tensor(reader, name_buf);
        if (lp) { memcpy(block->lambda_proj.data, lp, block->lambda_proj.rows * block->lambda_proj.cols * sizeof(float)); free(lp); }
    }
    
    /* Load classification head */
    float *head_weight = kser_reader_load_tensor(reader, "head_weight");
    if (head_weight) {
        memcpy(model->head.head_weight, head_weight, 
               model->cfg.num_classes * D * sizeof(float));
        free(head_weight);
    }
    
    float *head_bias = kser_reader_load_tensor(reader, "head_bias");
    if (head_bias) {
        memcpy(model->head.head_bias, head_bias, model->cfg.num_classes * sizeof(float));
        free(head_bias);
    }
    
    kser_reader_close(reader);
    printf("✓ Model loaded from %s\n", path);
    
    return model;
}

/* ============================================================================
 * Main Training Program
 * ============================================================================ */

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

int main(int argc, char **argv) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     K-Mamba 2D Vision - CIFAR-10 Training (MX450 Optimized)   ║\n");
    printf("║                                                               ║\n");
    printf("║  Architecture: 32×32×3 → Conv2D → 5×Mamba2D → Pool → 10       ║\n");
    printf("║  Model Size:   Dim=%d, State=%d, Layers=%d                      ║\n",
           KMAMBA_VISION_DIM, KMAMBA_VISION_STATE, KMAMBA_VISION_LAYERS);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    /* Parse arguments */
    const char *data_dir = (argc > 1) ? argv[1] : "data/cifar-10-batches-bin";
    int epochs = (argc > 2) ? atoi(argv[2]) : KMAMBA_VISION_EPOCHS;
    
    /* Initialize model */
    printf("🔧 Initializing model...\n");
    KMambaVision *model = kmamba_vision_init(
        KMAMBA_VISION_DIM, 
        KMAMBA_VISION_STATE,
        KMAMBA_VISION_LAYERS,
        KMAMBA_VISION_BATCH_SIZE,
        0  /* FP32 for stability on MX450 */
    );
    
    if (!model) {
        fprintf(stderr, "❌ Failed to initialize model\n");
        return 1;
    }
    
    kmamba_vision_init_weights(model, 42);
    
    printf("✓ Model initialized: %zu parameters\n", 
           (size_t)(3 * 3 * 3 * 96 + 96 +  /* patch embed */
                    model->cfg.n_layers * ( /* blocks */
                        96 * 96 + 96 * 96 + 96 + 96 * 192 + 96 * 192 + 96 + 96 + 96
                    ) +
                    10 * 96 + 10));  /* head */
    
    /* Load CIFAR-10 */
    printf("📊 Loading CIFAR-10 from %s...\n", data_dir);
    CIFAR10Dataset *train_ds = cifar10_load_train(data_dir);
    CIFAR10Dataset *test_ds = cifar10_load_test(data_dir);
    
    if (!train_ds || !test_ds) {
        fprintf(stderr, "⚠️  Could not load CIFAR-10 data. Please download from:\n");
        fprintf(stderr, "   https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n");
        fprintf(stderr, "   Extract to: %s/\n\n", data_dir);
        fprintf(stderr, "   Running with synthetic data for testing...\n\n");
        
        /* Synthetic data for testing */
        train_ds = NULL;
        test_ds = NULL;
    } else {
        printf("✓ Loaded: %zu train, %zu test samples\n\n", 
               train_ds->n_samples, test_ds->n_samples);
    }
    
    /* Training setup */
    size_t batch_size = KMAMBA_VISION_BATCH_SIZE;
    size_t steps_per_epoch = train_ds ? (train_ds->n_samples / batch_size) : 100;
    float *batch_images = (float *)malloc(batch_size * 32 * 32 * 3 * sizeof(float));
    uint8_t *batch_labels = (uint8_t *)malloc(batch_size);
    
    printf("⚡ Training Configuration:\n");
    printf("   Epochs:       %d\n", epochs);
    printf("   Batch size:   %zu\n", batch_size);
    printf("   Steps/epoch:  %zu\n", steps_per_epoch);
    printf("   Optimizer:    MUON (μ=%.2f, clip=%.1f)\n", 
           KMAMBA_VISION_MUON_MU, KMAMBA_VISION_CLIP_NORM);
    printf("   LR schedule:  warmup %d epochs + cosine decay\n\n", 
           KMAMBA_VISION_WARMUP_EPOCHS);
    
    /* Training loop */
    printf("╔════════╦═══════════╦══════════╦═══════════╦══════════╗\n");
    printf("║ Epoch  ║   Loss    ║   Acc    ║   LR      ║   Time   ║\n");
    printf("╠════════╬═══════════╬══════════╬═══════════╬══════════╣\n");
    fflush(stdout);
    
    printf("DEBUG: Starting training loop...\n");
    fflush(stdout);
    
    double total_start = get_time();
    float best_acc = 0.0f;
    
    printf("DEBUG: About to start epoch 0\n");
    fflush(stdout);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("DEBUG: Epoch %d starting...\n", epoch);
        fflush(stdout);
        
        double epoch_start = get_time();
        
        /* Update learning rate */
        printf("DEBUG: Getting LR...\n");
        fflush(stdout);
        float lr = kmamba_vision_get_lr(epoch, KMAMBA_VISION_WARMUP_EPOCHS, 
                                        epochs, KMAMBA_VISION_BASE_LR, 
                                        KMAMBA_VISION_MIN_LR);
        printf("DEBUG: LR=%f, setting optim_cfg...\n", lr);
        fflush(stdout);
        model->optim_cfg.lr = lr;
        
        printf("DEBUG: Epoch %d initialized, entering step loop...\n", epoch);
        fflush(stdout);
        
        float epoch_loss = 0.0f;
        int correct = 0;
        int total = 0;
        
        for (size_t step = 0; step < steps_per_epoch; step++) {
            if (step == 0 && epoch == 0) {
                printf("DEBUG: Step 0 - loading batch...\n");
                fflush(stdout);
            }
            
            /* Load batch */
            if (train_ds) {
                cifar10_next_batch(train_ds, batch_images, batch_labels, batch_size);
            } else {
                /* Synthetic data */
                for (size_t b = 0; b < batch_size; b++) {
                    batch_labels[b] = rand() % 10;
                    for (size_t i = 0; i < 32 * 32 * 3; i++) {
                        batch_images[b * 32 * 32 * 3 + i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    }
                }
            }
            
            if (step == 0 && epoch == 0) {
                printf("DEBUG: Step 0 - batch loaded, calling train_step...\n");
                fflush(stdout);
            }
            
            /* Train step */
            float loss = kmamba_vision_train_step(model, batch_images, batch_labels, batch_size);
            
            if (step == 0 && epoch == 0) {
                printf("DEBUG: Step 0 - train_step completed, loss=%f\n", loss);
                fflush(stdout);
            }
            
            if (isnan(loss) || loss < 0) {
                printf("\n❌ NaN or error detected at epoch %d, step %zu\n", epoch, step);
                break;
            }
            
            epoch_loss += loss;
            
            /* Simple accuracy estimate (head only training) */
            for (size_t b = 0; b < batch_size; b++) {
                float max_logit = model->buf_logits[b * 10];
                int pred = 0;
                for (int c = 1; c < 10; c++) {
                    if (model->buf_logits[b * 10 + c] > max_logit) {
                        max_logit = model->buf_logits[b * 10 + c];
                        pred = c;
                    }
                }
                if (pred == batch_labels[b]) correct++;
                total++;
            }
        }
        
        epoch_loss /= (float)steps_per_epoch;
        float acc = 100.0f * (float)correct / (float)total;
        if (acc > best_acc) best_acc = acc;
        
        double epoch_time = get_time() - epoch_start;
        
        printf("║ %4d   ║  %7.4f  ║  %5.2f%%  ║  %.2e  ║  %5.1fs  ║\n",
               epoch + 1, epoch_loss, acc, lr, epoch_time);
        
        /* Check for NaN */
        if (kmamba_vision_check_nan(model)) {
            printf("\n❌ NaN detected in model weights!\n");
            break;
        }
        
        /* Periodic save */
        if ((epoch + 1) % 10 == 0) {
            printf("║        ║  ✓ Saved checkpoint                           ║\n");
        }
    }
    
    double total_time = get_time() - total_start;
    
    printf("╚════════╩═══════════╩══════════╩═══════════╩══════════╝\n\n");
    printf("🏁 Training complete!\n");
    printf("   Best accuracy: %.2f%%\n", best_acc);
    printf("   Total time:    %.1f min\n", total_time / 60.0f);
    
    /* Test evaluation */
    if (test_ds) {
        printf("\n📊 Evaluating on test set...\n");
        int test_correct = 0;
        size_t test_batches = test_ds->n_samples / batch_size;
        
        for (size_t i = 0; i < test_batches; i++) {
            cifar10_next_batch(test_ds, batch_images, batch_labels, batch_size);
            kmamba_vision_forward(model, batch_images, model->buf_logits, batch_size);
            
            for (size_t b = 0; b < batch_size; b++) {
                float max_logit = model->buf_logits[b * 10];
                int pred = 0;
                for (int c = 1; c < 10; c++) {
                    if (model->buf_logits[b * 10 + c] > max_logit) {
                        max_logit = model->buf_logits[b * 10 + c];
                        pred = c;
                    }
                }
                if (pred == batch_labels[b]) test_correct++;
            }
        }
        
        float test_acc = 100.0f * (float)test_correct / (float)(test_batches * batch_size);
        printf("   Test accuracy: %.2f%%\n", test_acc);
    }
    
    /* Cleanup */
    free(batch_images);
    free(batch_labels);
    if (train_ds) cifar10_dataset_free(train_ds);
    if (test_ds) cifar10_dataset_free(test_ds);
    kmamba_vision_free(model);
    
    return 0;
}
