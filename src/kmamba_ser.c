/* ============================================================================
 * kmamba_ser.c - Integration libkser avec k-mamba
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmamba.h"
#include "kmamba_ser.h"
#include "kser.h"

/* ============================================================================
 * Save k-mamba model to .ser format
 * ============================================================================ */
int kmamba_save_ser(KMamba* m, const char* path, KSerDtype dtype) {
    if (!m || !path) return KSER_ERR_IO;
    
    /* Get model config */
    const KMambaConfig* cfg = kmamba_get_config(m);
    if (!cfg) return KSER_ERR_IO;
    
    /* Build kser config */
    KSerConfig ser_cfg = {
        .vocab_size = cfg->vocab_size,
        .dim = cfg->dim,
        .state_size = cfg->state_size,
        .n_layers = cfg->n_layers,
        .seq_len = cfg->seq_len,
        .d_conv = cfg->d_conv,
        .expand_factor = cfg->expand_factor,
        .dtype = dtype,
        .model_name = {0}
    };
    strncpy(ser_cfg.model_name, cfg->model_name, 63);
    
    /* Create writer */
    KSerWriter* w = kser_writer_create(path, &ser_cfg);
    if (!w) return KSER_ERR_IO;
    
    int ret = KSER_OK;
    
    /* Add vocabulary if using cl100k_base */
    if (cfg->vocab_size > 256 && cfg->vocab_data) {
        for (uint32_t i = 0; i < cfg->vocab_size && i < 100000; i++) {
            const char* token = cfg->vocab_data[i];
            if (token) {
                kser_writer_add_vocab(w, i, token, (uint16_t)strlen(token));
            }
        }
    }
    
    /* Add embedding weights */
    float* embed = kmamba_get_tensor(m, "embedding");
    if (embed) {
        uint32_t shape[4] = {cfg->vocab_size, cfg->dim, 0, 0};
        ret = kser_writer_add_tensor(w, "embedding", embed, shape, 
                                      KSER_FP32, dtype);
        if (ret != KSER_OK) goto cleanup;
    }
    
    /* Add layer weights */
    for (uint32_t layer = 0; layer < cfg->n_layers; layer++) {
        char name[64];
        
        /* SSM weights */
        snprintf(name, sizeof(name), "layers.%d.ssm.A", layer);
        float* A = kmamba_get_tensor(m, name);
        if (A) {
            uint32_t shape[4] = {cfg->dim, cfg->state_size, 0, 0};
            ret = kser_writer_add_tensor(w, name, A, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        snprintf(name, sizeof(name), "layers.%d.ssm.B", layer);
        float* B = kmamba_get_tensor(m, name);
        if (B) {
            uint32_t shape[4] = {cfg->dim, cfg->state_size, 0, 0};
            ret = kser_writer_add_tensor(w, name, B, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        snprintf(name, sizeof(name), "layers.%d.ssm.C", layer);
        float* C = kmamba_get_tensor(m, name);
        if (C) {
            uint32_t shape[4] = {cfg->dim, cfg->state_size, 0, 0};
            ret = kser_writer_add_tensor(w, name, C, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        snprintf(name, sizeof(name), "layers.%d.ssm.D", layer);
        float* D = kmamba_get_tensor(m, name);
        if (D) {
            uint32_t shape[4] = {cfg->dim, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, D, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        /* Projections */
        snprintf(name, sizeof(name), "layers.%d.in_proj", layer);
        float* in_proj = kmamba_get_tensor(m, name);
        if (in_proj) {
            uint32_t shape[4] = {cfg->dim, cfg->dim * 2, 0, 0};
            ret = kser_writer_add_tensor(w, name, in_proj, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        snprintf(name, sizeof(name), "layers.%d.out_proj", layer);
        float* out_proj = kmamba_get_tensor(m, name);
        if (out_proj) {
            uint32_t shape[4] = {cfg->dim, cfg->dim, 0, 0};
            ret = kser_writer_add_tensor(w, name, out_proj, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        /* MLP */
        snprintf(name, sizeof(name), "layers.%d.mlp.up", layer);
        float* mlp_up = kmamba_get_tensor(m, name);
        if (mlp_up) {
            uint32_t hidden = (uint32_t)(cfg->dim * cfg->expand_factor);
            uint32_t shape[4] = {cfg->dim, hidden, 0, 0};
            ret = kser_writer_add_tensor(w, name, mlp_up, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        snprintf(name, sizeof(name), "layers.%d.mlp.down", layer);
        float* mlp_down = kmamba_get_tensor(m, name);
        if (mlp_down) {
            uint32_t hidden = (uint32_t)(cfg->dim * cfg->expand_factor);
            uint32_t shape[4] = {hidden, cfg->dim, 0, 0};
            ret = kser_writer_add_tensor(w, name, mlp_down, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        /* Conv1d */
        snprintf(name, sizeof(name), "layers.%d.conv.weight", layer);
        float* conv_w = kmamba_get_tensor(m, name);
        if (conv_w) {
            uint32_t shape[4] = {cfg->dim, cfg->d_conv, 0, 0};
            ret = kser_writer_add_tensor(w, name, conv_w, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
        
        snprintf(name, sizeof(name), "layers.%d.conv.bias", layer);
        float* conv_b = kmamba_get_tensor(m, name);
        if (conv_b) {
            uint32_t shape[4] = {cfg->dim, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, conv_b, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }
    }
    
    /* Add output head */
    float* head = kmamba_get_tensor(m, "head");
    if (head) {
        uint32_t shape[4] = {cfg->dim, cfg->vocab_size, 0, 0};
        ret = kser_writer_add_tensor(w, "head", head, shape, KSER_FP32, dtype);
        if (ret != KSER_OK) goto cleanup;
    }
    
    /* Finalize */
    ret = kser_writer_finalize(w);
    
cleanup:
    kser_writer_free(w);
    return ret;
}

static int load_vocab_cb(uint32_t id, const char* token, uint16_t len, void* userdata) {
    KMamba* m = (KMamba*)userdata;
    return kmamba_set_vocab(m, id, token, len);
}

/* ============================================================================
 * Load k-mamba model from .ser format
 * ============================================================================ */
KMamba* kmamba_load_ser(const char* path, int flags) {
    if (!path) return NULL;
    
    /* Open reader */
    KSerReader* r = kser_reader_open(path);
    if (!r) return NULL;
    
    /* Verify checksum if requested */
    if (flags & KM_SER_VERIFY) {
        if (!kser_reader_is_valid(r)) {
            fprintf(stderr, "Error: Checksum verification failed for %s\n", path);
            kser_reader_close(r);
            return NULL;
        }
    }
    
    /* Get config */
    const KSerConfig* scfg = kser_reader_config(r);
    if (!scfg) {
        kser_reader_close(r);
        return NULL;
    }
    
    /* Build k-mamba config */
    KMambaConfig cfg = {
        .vocab_size = scfg->vocab_size,
        .dim = scfg->dim,
        .state_size = scfg->state_size,
        .n_layers = scfg->n_layers,
        .seq_len = scfg->seq_len,
        .d_conv = scfg->d_conv,
        .expand_factor = scfg->expand_factor,
        .model_name = {0}
    };
    strncpy(cfg.model_name, scfg->model_name, 63);
    
    /* Create model */
    KMamba* m = kmamba_create(&cfg);
    if (!m) {
        kser_reader_close(r);
        return NULL;
    }
    
    /* Load tensors */
    uint64_t nt = kser_reader_count_tensors(r);
    for (uint64_t i = 0; i < nt; i++) {
        const KSerTensorEntry* info = kser_reader_get_tensor_info(r, i);
        if (info) {
            float* data = kser_reader_load_tensor(r, info->name);
            if (data) {
                kmamba_set_tensor(m, info->name, data);
                free(data);
            }
        }
    }
    
    /* Load vocabulary if present */
    kser_reader_load_vocab(r, load_vocab_cb, m);
    
    kser_reader_close(r);
    return m;
}

/* ============================================================================
 * Get file info without loading
 * ============================================================================ */
KMSerInfo kmamba_ser_info(const char* path) {
    KMSerInfo info = {0};
    
    KSerInfo base = kser_file_info(path);
    memcpy(&info.base, &base, sizeof(KSerInfo));
    
    info.n_layers = base.n_layers;
    /* These would need to be read from extended config */
    info.state_size = 0;  /* Not available in base KSerInfo */
    info.seq_len = 0;
    
    return info;
}

/* ============================================================================
 * Save checkpoint with optimizer state
 * ============================================================================ */
int kmamba_save_checkpoint_ser(KMamba* m, void* opt, const char* path, KSerDtype dtype) {
    /* For now, just save model weights */
    /* Full checkpoint would include: */
    /* - Model weights */
    /* - Optimizer state (momentum buffers, etc.) */
    /* - Training step count */
    /* - RNG state */
    
    (void)opt; /* Unused for now */
    return kmamba_save_ser(m, path, dtype);
}

/* ============================================================================
 * Load checkpoint with optimizer state
 * ============================================================================ */
int kmamba_load_checkpoint_ser(const char* path, KMamba** m_out, 
                                void** opt_out, int flags) {
    if (!path || !m_out) return KSER_ERR_IO;
    
    *m_out = kmamba_load_ser(path, flags);
    if (!*m_out) return KSER_ERR_IO;
    
    /* Optimizer state loading - TODO */
    if (opt_out) *opt_out = NULL;
    
    return KSER_OK;
}
