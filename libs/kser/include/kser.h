#ifndef KSER_H
#define KSER_H

/* ============================================================================
 * kser.h — Public API for libkser
 * Serialization format for K-Mamba models (.ser)
 *
 * File layout (strict sequential):
 *   [0]       16 bytes  Magic header (SERENITY + η + version + reserved)
 *   [16]      96 bytes  KSerConfig
 *   [112]      4 bytes  vocab_count (uint32)
 *   [116]   variable    Vocab entries: (id:u32)(len:u16)(token:bytes)
 *   [116+V] variable    Tensor data (absolute offsets in KSerTensorEntry)
 *   [116+V+D]  4 bytes  tensor_count (uint32)
 *   [+4]    variable    KSerTensorEntry[] array
 *   [end-32]  32 bytes  SHA256 checksum of all preceding bytes
 * ============================================================================ */

#include <stdint.h>
#include <stddef.h>

/* ── Magic & versioning ─────────────────────────────────────────────────── */

#define KSER_MAGIC_SIZE  8
#define KSER_HEADER_SIZE 16
#define KSER_VERSION     0x01

/* Magic: "SERENITY" (8 ASCII) + η UTF-8 (2 bytes) + version (1) + pad (5) */
static const uint8_t KSER_MAGIC_BYTES[KSER_HEADER_SIZE] = {
    'S','E','R','E','N','I','T','Y',   /* 8 bytes ASCII         */
    0xCE, 0xB7,                         /* η grec UTF-8          */
    KSER_VERSION,                       /* version = 0x01        */
    0x00, 0x00, 0x00, 0x00, 0x00        /* 5 bytes reserved      */
};

/* ── Data types ─────────────────────────────────────────────────────────── */

typedef enum {
    KSER_FP32 = 0,
    KSER_FP16 = 1,
    KSER_BF16 = 2,
    KSER_INT8 = 3,
} KSerDtype;

typedef enum {
    KSER_OK            =  0,
    KSER_ERR_IO        = -1,
    KSER_ERR_FORMAT    = -2,
    KSER_ERR_MEMORY    = -3,
    KSER_ERR_CHECKSUM  = -4,
    KSER_ERR_NOT_FOUND = -5,
} KSerError;

/* ── Structures (fixed-size, packed layout) ─────────────────────────────── */

/* 96 bytes total */
typedef struct {
    uint32_t vocab_size;       /*  4 */
    uint32_t dim;              /*  4 */
    uint32_t state_size;       /*  4 */
    uint32_t n_layers;         /*  4 */
    uint32_t seq_len;          /*  4 */
    uint32_t d_conv;           /*  4 */
    float    expand_factor;    /*  4 */
    uint8_t  dtype;            /*  1 */
    uint8_t  _pad[3];          /*  3 */
    char     model_name[64];   /* 64 */
    /* total: 96 bytes */
} KSerConfig;

/* 72 bytes total */
typedef struct {
    char     name[32];         /* 32 */
    uint32_t shape[4];         /* 16 */
    uint8_t  dtype;            /*  1 */
    uint8_t  _pad[7];          /*  7 */
    uint64_t offset;           /*  8 — absolute offset in file */
    uint64_t size_bytes;       /*  8 */
    /* total: 72 bytes */
} KSerTensorEntry;

/* Info returned by kser_file_info() — no need to open fully */
typedef struct {
    int      valid;
    char     model_name[64];
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t n_layers;
    uint8_t  dtype;
    uint64_t n_params;
    uint64_t file_size;
} KSerInfo;

/* Opaque handles */
typedef struct KSerWriter KSerWriter;
typedef struct KSerReader KSerReader;

/* Callback for vocab loading */
typedef int (*KSerVocabCallback)(uint32_t id, const char* token,
                                  uint16_t len, void* userdata);

/* ── Writer API ─────────────────────────────────────────────────────────── */

KSerWriter* kser_writer_create  (const char* path, const KSerConfig* cfg);
int         kser_writer_add_vocab(KSerWriter* w, uint32_t id,
                                   const char* token, uint16_t len);
int         kser_writer_add_tensor(KSerWriter* w, const char* name,
                                    const void* data, const uint32_t shape[4],
                                    KSerDtype src_dtype, KSerDtype storage_dtype);
int         kser_writer_finalize (KSerWriter* w);
void        kser_writer_free     (KSerWriter* w);

/* ── Reader API ─────────────────────────────────────────────────────────── */

KSerReader*            kser_reader_open          (const char* path);
const KSerConfig*      kser_reader_config        (KSerReader* r);
int                    kser_reader_load_vocab     (KSerReader* r,
                                                   KSerVocabCallback cb,
                                                   void* userdata);
float*                 kser_reader_load_tensor   (KSerReader* r,
                                                   const char* name);
uint64_t               kser_reader_count_tensors (KSerReader* r);
const KSerTensorEntry* kser_reader_get_tensor_info(KSerReader* r, uint64_t idx);
void                   kser_reader_close         (KSerReader* r);

/* ── Info (fast peek) ───────────────────────────────────────────────────── */

KSerInfo kser_file_info(const char* path);

/* ── Quantization helpers ───────────────────────────────────────────────── */

int    kser_quantize  (const float* src, void* dst, uint64_t n, KSerDtype dtype);
int    kser_dequantize(const void* src, float* dst,  uint64_t n, KSerDtype dtype);
size_t kser_dtype_size(KSerDtype dtype);

/* ── SHA256 (kser_checksum.c) ───────────────────────────────────────────── */

void kser_sha256(const uint8_t* data, size_t len, uint8_t hash[32]);

#endif /* KSER_H */