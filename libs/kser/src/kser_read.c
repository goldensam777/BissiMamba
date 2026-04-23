/* ============================================================================
 * kser_read.c — Reader for .ser files
 *
 * Reads the file layout written by kser_write.c:
 *   [0]       16  Header
 *   [16]      96  KSerConfig
 *   [112]      4  vocab_count
 *   [116]      V  Vocab entries
 *   [116+V]    D  Tensor data
 *   [116+V+D]  4  tensor_count
 *   [+4]       T  KSerTensorEntry[]
 *   [end-32]  32  SHA256
 *
 * Cross-platform: mmap on POSIX, MapViewOfFile on Windows.
 * Falls back to fread if mmap fails.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include "kser.h"

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  define FSEEKO _fseeki64
#  define FTELLO _ftelli64
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  define FSEEKO fseeko
#  define FTELLO ftello
#endif

/* Forward from kser_checksum.c */
int kser_sha256_file(FILE* fp, long end_pos, uint8_t hash[32]);

/* ── Platform mmap abstraction ──────────────────────────────────────────── */

typedef struct {
    uint8_t* ptr;
    size_t   size;
#ifdef _WIN32
    HANDLE   file_handle;
    HANDLE   map_handle;
#else
    int      fd;
#endif
} MmapView;

static int mmap_open(const char* path, MmapView* v) {
    memset(v, 0, sizeof(MmapView));

#ifdef _WIN32
    v->file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                                  NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (v->file_handle == INVALID_HANDLE_VALUE) return -1;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(v->file_handle, &sz)) {
        CloseHandle(v->file_handle); return -1;
    }
    v->size = (size_t)sz.QuadPart;

    v->map_handle = CreateFileMappingA(v->file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!v->map_handle) {
        CloseHandle(v->file_handle); return -1;
    }

    v->ptr = (uint8_t*)MapViewOfFile(v->map_handle, FILE_MAP_READ, 0, 0, 0);
    if (!v->ptr) {
        CloseHandle(v->map_handle);
        CloseHandle(v->file_handle);
        return -1;
    }
    return 0;

#else
    v->fd = open(path, O_RDONLY);
    if (v->fd < 0) return -1;

    struct stat st;
    if (fstat(v->fd, &st) < 0) { close(v->fd); return -1; }
    v->size = (size_t)st.st_size;

    v->ptr = (uint8_t*)mmap(NULL, v->size, PROT_READ, MAP_PRIVATE, v->fd, 0);
    if (v->ptr == MAP_FAILED) { v->ptr = NULL; close(v->fd); return -1; }
    return 0;
#endif
}

static void mmap_close(MmapView* v) {
    if (!v->ptr) return;
#ifdef _WIN32
    UnmapViewOfFile(v->ptr);
    CloseHandle(v->map_handle);
    CloseHandle(v->file_handle);
#else
    munmap(v->ptr, v->size);
    close(v->fd);
#endif
    memset(v, 0, sizeof(MmapView));
}

/* ── Reader internals ───────────────────────────────────────────────────── */

struct KSerReader {
    /* Memory view (primary access) */
    MmapView mmap;
    int      has_mmap;

    /* Fallback FILE* when mmap fails */
    FILE*    fp;

    /* Parsed layout */
    KSerConfig       cfg;
    uint32_t         vocab_count;
    long             vocab_data_pos; /* file offset of first vocab entry */

    KSerTensorEntry* tensors;
    uint32_t         tensor_count;

    /* Checksum */
    uint8_t          stored_hash[32];
    int              hash_valid;
};

/* Read n bytes at offset from mmap or FILE* */
static int rread(KSerReader* r, long offset, void* dst, size_t n) {
    if (r->has_mmap) {
        if ((size_t)offset + n > r->mmap.size) return -1;
        memcpy(dst, r->mmap.ptr + offset, n);
        return 0;
    }
    if (FSEEKO(r->fp, offset, SEEK_SET) != 0) return -1;
    if (fread(dst, 1, n, r->fp) != n) return -1;
    return 0;
}

static const uint8_t* rptr(KSerReader* r, long offset) {
    if (!r->has_mmap) return NULL;
    if ((size_t)offset >= r->mmap.size) return NULL;
    return r->mmap.ptr + offset;
}

/* ── Header validation ──────────────────────────────────────────────────── */

static int validate_header(const uint8_t hdr[KSER_HEADER_SIZE]) {
    if (memcmp(hdr, KSER_MAGIC_BYTES, KSER_MAGIC_SIZE) != 0) return 0;
    if (hdr[8] != 0xCE || hdr[9] != 0xB7)   return 0; /* η UTF-8 */
    if (hdr[10] != KSER_VERSION)              return 0;
    return 1;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

KSerReader* kser_reader_open(const char* path) {
    if (!path) return NULL;

    KSerReader* r = calloc(1, sizeof(KSerReader));
    if (!r) return NULL;

    /* Try mmap first */
    if (mmap_open(path, &r->mmap) == 0) {
        r->has_mmap = 1;
    } else {
        r->fp = fopen(path, "rb");
        if (!r->fp) { free(r); return NULL; }
        r->has_mmap = 0;
    }

    /* Need file size for fallback path */
    size_t file_size;
    if (r->has_mmap) {
        file_size = r->mmap.size;
    } else {
        if (FSEEKO(r->fp, 0, SEEK_END) != 0) goto fail;
        file_size = (size_t)FTELLO(r->fp);
    }

    /* Minimum viable file: header(16)+config(96)+vocab_count(4)+tensor_count(4)+sha256(32) = 152 */
    if (file_size < 152) goto fail;

    /* 1. Validate header */
    uint8_t hdr[KSER_HEADER_SIZE];
    if (rread(r, 0, hdr, KSER_HEADER_SIZE) != 0) goto fail;
    if (!validate_header(hdr)) goto fail;

    /* 2. Read config */
    if (rread(r, KSER_HEADER_SIZE, &r->cfg, sizeof(KSerConfig)) != 0) goto fail;

    /* 3. Read vocab_count */
    long vocab_count_pos = KSER_HEADER_SIZE + (long)sizeof(KSerConfig);
    if (rread(r, vocab_count_pos, &r->vocab_count, sizeof(uint32_t)) != 0) goto fail;

    /* 4. Walk vocab to find data_start */
    r->vocab_data_pos = vocab_count_pos + (long)sizeof(uint32_t);
    long pos = r->vocab_data_pos;
    for (uint32_t i = 0; i < r->vocab_count; i++) {
        uint32_t id;  uint16_t len;
        if (rread(r, pos,                  &id,  sizeof(uint32_t)) != 0) goto fail;
        if (rread(r, pos+sizeof(uint32_t), &len, sizeof(uint16_t)) != 0) goto fail;
        pos += (long)sizeof(uint32_t) + (long)sizeof(uint16_t) + len;
    }
    /* pos now points to start of tensor data */

    /* 5. SHA256 is the last 32 bytes */
    long checksum_pos = (long)file_size - 32;
    if (rread(r, checksum_pos, r->stored_hash, 32) != 0) goto fail;

    /* 6. Tensor index is just before the checksum */
    /*    Layout: ... | tensor_count(4) | KSerTensorEntry[tensor_count] | sha256(32) */
    /*    We don't know tensor_count yet, so read it from just before the index end  */
    /*    Walk backwards: checksum at end, before it is the index, before that data  */
    /*    We know: index_end = checksum_pos                                           */
    /*    Read tensor_count from (checksum_pos - tensor_count*sizeof(Entry) - 4)     */
    /*    But we don't know tensor_count — scan forward from data_start instead      */

    /* Simpler: tensor_count is written first, then entries, then sha256.
     * So: [data...][tensor_count:4][entries:N*72][sha256:32]
     * We scan the tensor_count from checksum_pos - 4 backwards:
     *   checksum_pos = end_of_entries
     *   tensor_count_pos = ?
     * Use binary search: try reading tensor_count at various positions
     * Actually: scan forward from pos (data_start) — tensor data is opaque,
     * so we can't skip it. The only anchor is checksum_pos.
     *
     * Layout guarantee from kser_write.c finalize():
     *   fwrite(&tensor_count, 4, ...)
     *   fwrite(tensors, 72*tensor_count, ...)
     *   <sha256 pos>
     *
     * So: tensor_count is at offset (checksum_pos - 4 - tensor_count*72)
     * This is a chicken-and-egg problem. Resolve by reading uint32 from
     * successive candidate positions starting from checksum_pos-4-0*72 = checksum_pos-4.
     */
    {
        long index_end = checksum_pos; /* = checksum_pos */

        /* Try tensor_count = 0 first, then increase */
        int found = 0;
        uint32_t tc = 0;
        long tc_pos;

        /* Maximum plausible tensor count for a 3B model ≈ 24*8 = 192 */
        for (tc = 0; tc <= 4096; tc++) {
            tc_pos = index_end - (long)(4 + tc * sizeof(KSerTensorEntry));
            if (tc_pos < pos) break; /* before data start — impossible */
            uint32_t candidate;
            if (rread(r, tc_pos, &candidate, sizeof(uint32_t)) != 0) break;
            if (candidate == tc) {
                found = 1;
                break;
            }
        }

        if (!found) goto fail;

        r->tensor_count = tc;
        if (tc > 0) {
            r->tensors = calloc(tc, sizeof(KSerTensorEntry));
            if (!r->tensors) goto fail;
            long entries_pos = tc_pos + (long)sizeof(uint32_t);
            if (rread(r, entries_pos, r->tensors,
                      tc * sizeof(KSerTensorEntry)) != 0) goto fail;
        }
    }

    /* 7. Verify checksum (optional — only if FILE* path; mmap can verify too) */
    if (!r->has_mmap && r->fp) {
        uint8_t computed[32];
        if (kser_sha256_file(r->fp, checksum_pos, computed) == 0) {
            r->hash_valid = (memcmp(computed, r->stored_hash, 32) == 0);
        }
    }

    return r;

fail:
    if (r->has_mmap) mmap_close(&r->mmap);
    if (r->fp) fclose(r->fp);
    free(r->tensors);
    free(r);
    return NULL;
}

const KSerConfig* kser_reader_config(KSerReader* r) {
    return r ? &r->cfg : NULL;
}

int kser_reader_load_vocab(KSerReader* r, KSerVocabCallback cb, void* userdata) {
    if (!r || !cb) return KSER_OK;

    long pos = r->vocab_data_pos;
    for (uint32_t i = 0; i < r->vocab_count; i++) {
        uint32_t id;
        uint16_t len;

        if (rread(r, pos, &id, sizeof(uint32_t)) != 0) return KSER_ERR_IO;
        pos += sizeof(uint32_t);
        if (rread(r, pos, &len, sizeof(uint16_t)) != 0) return KSER_ERR_IO;
        pos += sizeof(uint16_t);

        int ret;
        if (r->has_mmap) {
            const char* token = (const char*)rptr(r, pos);
            if (!token) return KSER_ERR_IO;
            ret = cb(id, token, len, userdata);
        } else {
            char* buf = malloc(len + 1);
            if (!buf) return KSER_ERR_MEMORY;
            if (rread(r, pos, buf, len) != 0) { free(buf); return KSER_ERR_IO; }
            buf[len] = '\0';
            ret = cb(id, buf, len, userdata);
            free(buf);
        }

        if (ret != 0) return ret;
        pos += len;
    }
    return KSER_OK;
}

float* kser_reader_load_tensor(KSerReader* r, const char* name) {
    if (!r || !name) return NULL;

    for (uint32_t i = 0; i < r->tensor_count; i++) {
        if (strcmp(r->tensors[i].name, name) != 0) continue;

        KSerTensorEntry* e = &r->tensors[i];
        uint64_t n = 1;
        for (int j = 0; j < 4; j++) if (e->shape[j] > 0) n *= e->shape[j];

        float* out = calloc(n, sizeof(float));
        if (!out) return NULL;

        if (r->has_mmap) {
            const uint8_t* src = rptr(r, (long)e->offset);
            if (!src) { free(out); return NULL; }
            kser_dequantize(src, out, n, (KSerDtype)e->dtype);
        } else {
            void* tmp = malloc(e->size_bytes);
            if (!tmp) { free(out); return NULL; }
            if (rread(r, (long)e->offset, tmp, e->size_bytes) != 0) {
                free(tmp); free(out); return NULL;
            }
            kser_dequantize(tmp, out, n, (KSerDtype)e->dtype);
            free(tmp);
        }
        return out;
    }
    return NULL;
}

uint64_t kser_reader_count_tensors(KSerReader* r) {
    return r ? r->tensor_count : 0;
}

const KSerTensorEntry* kser_reader_get_tensor_info(KSerReader* r, uint64_t idx) {
    if (!r || idx >= r->tensor_count) return NULL;
    return &r->tensors[idx];
}

void kser_reader_close(KSerReader* r) {
    if (!r) return;
    if (r->has_mmap) mmap_close(&r->mmap);
    if (r->fp) fclose(r->fp);
    free(r->tensors);
    free(r);
}

KSerInfo kser_file_info(const char* path) {
    KSerInfo info;
    memset(&info, 0, sizeof(info));

    FILE* fp = fopen(path, "rb");
    if (!fp) return info;

    struct stat st;
    if (stat(path, &st) == 0) info.file_size = (uint64_t)st.st_size;

    uint8_t hdr[KSER_HEADER_SIZE];
    if (fread(hdr, 1, KSER_HEADER_SIZE, fp) != KSER_HEADER_SIZE) goto done;
    if (!validate_header(hdr)) goto done;

    KSerConfig cfg;
    if (fread(&cfg, sizeof(KSerConfig), 1, fp) != 1) goto done;

    info.valid      = 1;
    info.vocab_size = cfg.vocab_size;
    info.dim        = cfg.dim;
    info.n_layers   = cfg.n_layers;
    info.dtype      = cfg.dtype;
    memcpy(info.model_name, cfg.model_name, 64);

    /* Rough parameter estimate */
    info.n_params  = (uint64_t)cfg.vocab_size * cfg.dim * 2;
    info.n_params += (uint64_t)cfg.dim * cfg.state_size * 4 * cfg.n_layers;
    info.n_params += (uint64_t)cfg.dim * cfg.dim * 4 * cfg.n_layers;

done:
    fclose(fp);
    return info;
}