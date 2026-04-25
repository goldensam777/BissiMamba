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
    snprintf(ser_cfg.model_name, sizeof(ser_cfg.model_name), "%s", cfg->model_name);
    
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
    
    /* Add layer weights (single canonical naming) */
    for (uint32_t layer = 0; layer < cfg->n_layers; layer++) {
        char name[64];
        uint32_t R = (cfg->mimo_rank > 1) ? (uint32_t)cfg->mimo_rank : 1;
        uint32_t N = (uint32_t)cfg->state_size;
        uint32_t D = (uint32_t)cfg->dim;
        
        snprintf(name, sizeof(name), "layers.%u.W_in", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {R, D, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.W_out", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {D, R, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.A_log", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {N, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.W_B", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {N * R, D, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.W_C", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {N * R, D, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.delta_proj", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {D, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.lambda_proj", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {D, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.b_B", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {N * R, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.b_C", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {N * R, 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
            if (ret != KSER_OK) goto cleanup;
        }

        snprintf(name, sizeof(name), "layers.%u.theta", layer);
        {
            float* t = kmamba_get_tensor(m, name);
            uint32_t shape[4] = {(N / 2 > 0 ? N / 2 : 1), 0, 0, 0};
            ret = kser_writer_add_tensor(w, name, t, shape, KSER_FP32, dtype);
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
    
    /* Build k-mamba config - start with defaults for new runtime fields */
    KMambaConfig cfg;
    kmamba_config_set_defaults(&cfg);

    /* Override with values from file (legacy fields) */
    cfg.vocab_size = scfg->vocab_size;
    cfg.dim = scfg->dim;
    cfg.state_size = scfg->state_size;
    cfg.n_layers = scfg->n_layers;
    cfg.seq_len = scfg->seq_len;
    cfg.d_conv = scfg->d_conv;
    cfg.expand_factor = scfg->expand_factor;
    snprintf(cfg.model_name, sizeof(cfg.model_name), "%s", scfg->model_name);
    
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
