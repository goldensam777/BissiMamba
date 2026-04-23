/* ============================================================================
 * kser_write.c — Writer for .ser files
 *
 * Write order (strict):
 *   1. Header (16 bytes) — written in create()
 *   2. Config (96 bytes) — written in create()
 *   3. vocab_count placeholder (4 bytes) — written in create(), updated in finalize()
 *   4. Vocab entries — written as add_vocab() is called
 *   5. Tensor data — written as add_tensor() is called (after first tensor, no more vocab)
 *   6. tensor_count (4 bytes) — written in finalize()
 *   7. KSerTensorEntry[] — written in finalize()
 *   8. SHA256 (32 bytes) — written in finalize()
 *
 * The file is written to a .tmp path and renamed atomically on finalize.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "kser.h"

/* Forward declarations from kser_checksum.c */
int kser_sha256_file(FILE* fp, long end_pos, uint8_t hash[32]);
size_t kser_quantize_size(uint64_t n, KSerDtype dtype);

/* ── Internal state ─────────────────────────────────────────────────────── */

struct KSerWriter {
    FILE*    fp;
    char     tmp_path[512];
    char     final_path[512];

    int      finalized;
    int      in_tensor_section; /* set to 1 after first add_tensor call */

    KSerConfig cfg;

    long     vocab_count_pos; /* file offset of vocab_count field */
    uint32_t vocab_count;

    long     data_start;      /* file offset where tensor data begins */

    KSerTensorEntry* tensors;
    uint32_t         tensor_count;
    uint32_t         tensor_capacity;
};

/* ── Helpers ────────────────────────────────────────────────────────────── */

static int writer_grow_tensors(KSerWriter* w) {
    uint32_t new_cap = w->tensor_capacity ? w->tensor_capacity * 2 : 64;
    KSerTensorEntry* t = realloc(w->tensors, new_cap * sizeof(KSerTensorEntry));
    if (!t) return KSER_ERR_MEMORY;
    w->tensors = t;
    w->tensor_capacity = new_cap;
    return KSER_OK;
}

/* ── Writer API ─────────────────────────────────────────────────────────── */

KSerWriter* kser_writer_create(const char* path, const KSerConfig* cfg) {
    if (!path || !cfg) return NULL;

    KSerWriter* w = calloc(1, sizeof(KSerWriter));
    if (!w) return NULL;

    /* Store paths */
    snprintf(w->final_path, sizeof(w->final_path), "%s", path);
    snprintf(w->tmp_path,   sizeof(w->tmp_path),   "%s.tmp", path);

    w->fp = fopen(w->tmp_path, "w+b");
    if (!w->fp) { free(w); return NULL; }

    memcpy(&w->cfg, cfg, sizeof(KSerConfig));

    /* 1. Write magic header (16 bytes) */
    if (fwrite(KSER_MAGIC_BYTES, 1, KSER_HEADER_SIZE, w->fp) != KSER_HEADER_SIZE)
        goto fail;

    /* 2. Write config block (96 bytes) */
    if (fwrite(&w->cfg, sizeof(KSerConfig), 1, w->fp) != 1)
        goto fail;

    /* 3. Reserve vocab_count placeholder (4 bytes, = 0 for now) */
    w->vocab_count_pos = ftell(w->fp);
    uint32_t zero = 0;
    if (fwrite(&zero, sizeof(uint32_t), 1, w->fp) != 1)
        goto fail;

    /* Vocab entries follow immediately. data_start will be set on first tensor. */
    w->data_start = -1;

    return w;

fail:
    fclose(w->fp);
    remove(w->tmp_path);
    free(w);
    return NULL;
}

int kser_writer_add_vocab(KSerWriter* w, uint32_t id,
                           const char* token, uint16_t len) {
    if (!w || w->finalized)         return KSER_ERR_IO;
    if (w->in_tensor_section)       return KSER_ERR_FORMAT; /* too late */
    if (!token && len > 0)          return KSER_ERR_IO;

    if (fwrite(&id,    sizeof(uint32_t), 1, w->fp) != 1) return KSER_ERR_IO;
    if (fwrite(&len,   sizeof(uint16_t), 1, w->fp) != 1) return KSER_ERR_IO;
    if (len > 0 && fwrite(token, 1, len, w->fp) != len)  return KSER_ERR_IO;

    w->vocab_count++;
    return KSER_OK;
}

int kser_writer_add_tensor(KSerWriter* w, const char* name,
                            const void* data, const uint32_t shape[4],
                            KSerDtype src_dtype, KSerDtype storage_dtype) {
    if (!w || w->finalized)  return KSER_ERR_IO;
    if (!name || !data)      return KSER_ERR_IO;

    /* First tensor: mark start of data section */
    if (!w->in_tensor_section) {
        w->in_tensor_section = 1;
        w->data_start = ftell(w->fp);
    }

    /* Grow tensor array if needed */
    if (w->tensor_count >= w->tensor_capacity) {
        int r = writer_grow_tensors(w);
        if (r != KSER_OK) return r;
    }

    /* Compute element count */
    uint64_t n = 1;
    for (int i = 0; i < 4; i++) if (shape[i] > 0) n *= shape[i];

    /* Fill tensor entry */
    KSerTensorEntry* e = &w->tensors[w->tensor_count];
    memset(e, 0, sizeof(KSerTensorEntry));
    strncpy(e->name, name, 31);
    memcpy(e->shape, shape, 4 * sizeof(uint32_t));
    e->dtype      = (uint8_t)storage_dtype;
    e->offset     = (uint64_t)ftell(w->fp);
    e->size_bytes = kser_quantize_size(n, storage_dtype);

    /* Write (quantized) data */
    if (src_dtype == storage_dtype) {
        /* No conversion needed */
        size_t raw = n * kser_dtype_size(src_dtype);
        if (fwrite(data, 1, raw, w->fp) != raw) return KSER_ERR_IO;
    } else if (src_dtype == KSER_FP32) {
        void* buf = malloc(e->size_bytes);
        if (!buf) return KSER_ERR_MEMORY;
        if (kser_quantize((const float*)data, buf, n, storage_dtype) != KSER_OK) {
            free(buf); return KSER_ERR_FORMAT;
        }
        size_t wrote = fwrite(buf, 1, e->size_bytes, w->fp);
        free(buf);
        if (wrote != e->size_bytes) return KSER_ERR_IO;
    } else {
        return KSER_ERR_FORMAT; /* unsupported conversion */
    }

    w->tensor_count++;
    return KSER_OK;
}

int kser_writer_finalize(KSerWriter* w) {
    if (!w || w->finalized) return KSER_ERR_IO;

    /* If no tensors were added, data_start = current pos */
    if (w->data_start < 0)
        w->data_start = ftell(w->fp);

    /* 6+7. Write KSerTensorEntry[] followed by tensor_count */
    for (uint32_t i = 0; i < w->tensor_count; i++) {
        if (fwrite(&w->tensors[i], sizeof(KSerTensorEntry), 1, w->fp) != 1)
            return KSER_ERR_IO;
    }
    if (fwrite(&w->tensor_count, sizeof(uint32_t), 1, w->fp) != 1)
        return KSER_ERR_IO;

    /* Update vocab_count at its reserved position */
    long pre_checksum_pos = ftell(w->fp);
    if (fseek(w->fp, w->vocab_count_pos, SEEK_SET) != 0) return KSER_ERR_IO;
    if (fwrite(&w->vocab_count, sizeof(uint32_t), 1, w->fp) != 1) return KSER_ERR_IO;
    if (fseek(w->fp, pre_checksum_pos, SEEK_SET) != 0) return KSER_ERR_IO;

    /* 8. Compute SHA256 of everything written so far */
    fflush(w->fp);
    uint8_t hash[32];
    if (kser_sha256_file(w->fp, pre_checksum_pos, hash) != 0)
        return KSER_ERR_IO;

    /* Write checksum */
    if (fseek(w->fp, pre_checksum_pos, SEEK_SET) != 0) return KSER_ERR_IO;
    if (fwrite(hash, 1, 32, w->fp) != 32) return KSER_ERR_IO;

    fflush(w->fp);
    w->finalized = 1;
    return KSER_OK;
}

void kser_writer_free(KSerWriter* w) {
    if (!w) return;

    if (!w->finalized && w->fp)
        kser_writer_finalize(w);

    if (w->fp) {
        fclose(w->fp);
        if (w->finalized) {
            rename(w->tmp_path, w->final_path);
        } else {
            remove(w->tmp_path);
        }
    }

    free(w->tensors);
    free(w);
}