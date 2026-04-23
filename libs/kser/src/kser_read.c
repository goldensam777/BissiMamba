/* ============================================================================
 * kser_read.c - Lecture de fichiers .ser avec mmap
 * ============================================================================ */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include "kser.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define FSEEKO _fseeki64
#define FTELLO _ftelli64
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#define FSEEKO fseeko
#define FTELLO ftello
#endif

/* Forward declarations */
extern void sha256_init(void* ctx);
extern void sha256_update(void* ctx, const uint8_t* data, size_t len);
extern void sha256_final(void* ctx, uint8_t hash[32]);
typedef struct { uint32_t state[8]; uint64_t count; uint8_t buffer[64]; } SHA256_CTX;
extern void sha256(const uint8_t* data, size_t len, uint8_t hash[32]);

/* Forward declaration for internal function */
static void kser_reader_close(KSerReader* r);

/* Internal reader structure */
struct KSerReader {
    /* File handling */
    int      fd;
    FILE*    fp;
    
    /* Memory mapping */
    uint8_t* mmap_data;
    size_t   mmap_size;
    int      use_mmap;
    
    /* Header info */
    uint8_t  header[KSER_HEADER_SIZE];
    
    /* Config */
    KSerConfig cfg;
    long     cfg_offset;
    
    /* Vocabulary */
    long     vocab_offset;
    uint32_t vocab_count;
    
    /* Tensor index */
    long     tensor_index_offset;
    uint32_t tensor_count;
    KSerTensorEntry* tensors;
    
    /* Data section */
    long     data_offset;
    
    /* Checksum */
    long     checksum_offset;
    uint8_t  stored_checksum[32];
    int      checksum_valid;
};

static int validate_header(const uint8_t* header) {
    /* Check magic bytes: SERENITY + η grec + version */
    if (memcmp(header, KSER_MAGIC, KSER_MAGIC_SIZE) != 0) return 0;
    if (header[8] != 0xCE || header[9] != 0xB7) return 0; /* η UTF-8 */
    if (header[10] != KSER_VERSION) return 0;
    return 1;
}

KSerReader* kser_reader_open(const char* path) {
    KSerReader* r = calloc(1, sizeof(KSerReader));
    if (!r) return NULL;
    
    /* Open file */
    r->fd = open(path, O_RDONLY);
    if (r->fd < 0) {
        free(r);
        return NULL;
    }
    
    /* Get file size */
    struct stat st;
    if (fstat(r->fd, &st) < 0) {
        close(r->fd);
        free(r);
        return NULL;
    }
    r->mmap_size = st.st_size;
    
    /* Memory map the file */
    r->mmap_data = mmap(NULL, r->mmap_size, PROT_READ, MAP_PRIVATE, r->fd, 0);
    if (r->mmap_data == MAP_FAILED) {
        /* Fallback to regular file reading */
        r->mmap_data = NULL;
        r->use_mmap = 0;
        r->fp = fdopen(r->fd, "rb");
        if (!r->fp) {
            close(r->fd);
            free(r);
            return NULL;
        }
    } else {
        r->use_mmap = 1;
    }
    
    /* Read and validate header */
    if (r->use_mmap) {
        memcpy(r->header, r->mmap_data, KSER_HEADER_SIZE);
    } else {
        if (fread(r->header, 1, KSER_HEADER_SIZE, r->fp) != KSER_HEADER_SIZE) {
            kser_reader_close(r);
            return NULL;
        }
    }
    
    if (!validate_header(r->header)) {
        kser_reader_close(r);
        return NULL;
    }
    
    /* Read config */
    r->cfg_offset = KSER_HEADER_SIZE;
    if (r->use_mmap) {
        memcpy(&r->cfg, r->mmap_data + r->cfg_offset, sizeof(KSerConfig));
    } else {
        FSEEKO(r->fp, r->cfg_offset, SEEK_SET);
        if (fread(&r->cfg, sizeof(KSerConfig), 1, r->fp) != 1) {
            kser_reader_close(r);
            return NULL;
        }
    }
    
    /* Vocab starts right after config */
    r->vocab_offset = r->cfg_offset + sizeof(KSerConfig);
    
    /* Read vocab count */
    if (r->use_mmap) {
        memcpy(&r->vocab_count, r->mmap_data + r->vocab_offset, sizeof(uint32_t));
    } else {
        FSEEKO(r->fp, r->vocab_offset, SEEK_SET);
        fread(&r->vocab_count, sizeof(uint32_t), 1, r->fp);
    }
    
    /* Calculate data offset (after vocab entries) */
    r->data_offset = r->vocab_offset + sizeof(uint32_t);
    for (uint32_t i = 0; i < r->vocab_count; i++) {
        uint32_t id;
        uint16_t len;
        if (r->use_mmap) {
            memcpy(&id, r->mmap_data + r->data_offset, sizeof(uint32_t));
            memcpy(&len, r->mmap_data + r->data_offset + sizeof(uint32_t), sizeof(uint16_t));
        } else {
            FSEEKO(r->fp, r->data_offset, SEEK_SET);
            fread(&id, sizeof(uint32_t), 1, r->fp);
            fread(&len, sizeof(uint16_t), 1, r->fp);
        }
        r->data_offset += sizeof(uint32_t) + sizeof(uint16_t) + len;
    }
    
    /* Tensor index is at the end before checksum */
    /* Minimum file size: header(16) + config(96) + vocab(4) + tensor_index(4) + checksum(32) = ~152 bytes */
    if (r->mmap_size < 152) {
        fprintf(stderr, "DEBUG READER: File too small: %zu bytes\n", r->mmap_size);
        kser_reader_close(r);
        return NULL;
    }
    
    /* Calculate: file_size - 32 (checksum) = position of tensor count */
    r->checksum_offset = r->mmap_size - 32;
    long tensor_count_pos = r->checksum_offset - sizeof(uint32_t);
    
    if (r->use_mmap) {
        memcpy(&r->tensor_count, r->mmap_data + tensor_count_pos, sizeof(uint32_t));
    } else {
        FSEEKO(r->fp, tensor_count_pos, SEEK_SET);
        fread(&r->tensor_count, sizeof(uint32_t), 1, r->fp);
    }
    
    fprintf(stderr, "DEBUG READER: tensor_count=%u at pos=%ld\n", r->tensor_count, tensor_count_pos);
    
    /* Calculate tensor index start */
    long tensor_index_size = sizeof(uint32_t) + r->tensor_count * sizeof(KSerTensorEntry);
    r->tensor_index_offset = r->checksum_offset - tensor_index_size;
    
    fprintf(stderr, "DEBUG READER: tensor_index_offset=%ld, data_offset=%ld\n", 
            r->tensor_index_offset, r->data_offset);
    
    if (r->tensor_index_offset < r->data_offset) {
        fprintf(stderr, "DEBUG READER: Invalid tensor index offset\n");
        kser_reader_close(r);
        return NULL;
    }
    
    /* Allocate and read tensor entries */
    r->tensors = calloc(r->tensor_count, sizeof(KSerTensorEntry));
    if (!r->tensors) {
        kser_reader_close(r);
        return NULL;
    }
    
    if (r->use_mmap) {
        memcpy(r->tensors, r->mmap_data + r->tensor_index_offset + sizeof(uint32_t), 
               r->tensor_count * sizeof(KSerTensorEntry));
    } else {
        FSEEKO(r->fp, r->tensor_index_offset + sizeof(uint32_t), SEEK_SET);
        if (fread(r->tensors, sizeof(KSerTensorEntry), r->tensor_count, r->fp) != r->tensor_count) {
            kser_reader_close(r);
            return NULL;
        }
    }
    
    /* Read checksum */
    if (r->use_mmap) {
        memcpy(r->stored_checksum, r->mmap_data + r->checksum_offset, 32);
    } else {
        FSEEKO(r->fp, r->checksum_offset, SEEK_SET);
        size_t n = fread(r->stored_checksum, 1, 32, r->fp);
        (void)n;
    }
    
    return r;
}

const KSerConfig* kser_reader_config(KSerReader* r) {
    return r ? &r->cfg : NULL;
}

int kser_reader_load_vocab(KSerReader* r, KSerVocabCallback cb, void* userdata) {
    if (!r || !cb || r->vocab_count == 0) return KSER_OK;
    
    long pos = r->vocab_offset + sizeof(uint32_t);
    
    for (uint32_t i = 0; i < r->vocab_count; i++) {
        uint32_t id;
        uint16_t len;
        
        if (r->use_mmap) {
            memcpy(&id, r->mmap_data + pos, sizeof(uint32_t));
            memcpy(&len, r->mmap_data + pos + sizeof(uint32_t), sizeof(uint16_t));
        } else {
            FSEEKO(r->fp, pos, SEEK_SET);
            size_t n1 = fread(&id, sizeof(uint32_t), 1, r->fp);
            size_t n2 = fread(&len, sizeof(uint16_t), 1, r->fp);
            (void)n1; (void)n2;
        }
        
        pos += sizeof(uint32_t) + sizeof(uint16_t);
        
        const char* token = r->use_mmap ? 
            (const char*)(r->mmap_data + pos) : NULL;
        
        if (!r->use_mmap) {
            /* Read token to temporary buffer */
            char* temp = malloc(len);
            if (temp) {
                size_t nread = fread(temp, 1, len, r->fp);
                (void)nread; /* Suppress warning - we proceed with what we have */
                int ret = cb(id, temp, len, userdata);
                free(temp);
                if (ret != 0) return ret;
            }
        } else {
            int ret = cb(id, token, len, userdata);
            if (ret != 0) return ret;
        }
        
        pos += len;
    }
    
    return KSER_OK;
}

void* kser_reader_load_tensor(KSerReader* r, const char* name, KSerDtype target_dtype) {
    if (!r || !name) return NULL;
    
    /* Find tensor by name */
    for (uint32_t i = 0; i < r->tensor_count; i++) {
        if (strcmp(r->tensors[i].name, name) == 0) {
            KSerTensorEntry* entry = &r->tensors[i];
            
            /* Calculate number of elements */
            uint64_t n_elements = 1;
            for (int j = 0; j < 4; j++) {
                if (entry->shape[j] > 0) n_elements *= entry->shape[j];
            }
            
            /* Allocate output buffer (always FP32 for k-mamba) */
            float* output = calloc(n_elements, sizeof(float));
            if (!output) return NULL;
            
            /* Get pointer to tensor data */
            const void* src_data;
            if (r->use_mmap) {
                src_data = r->mmap_data + entry->offset;
            } else {
                /* Read from file */
                void* temp = malloc(entry->size_bytes);
                if (!temp) {
                    free(output);
                    return NULL;
                }
                FSEEKO(r->fp, entry->offset, SEEK_SET);
                if (fread(temp, 1, entry->size_bytes, r->fp) != entry->size_bytes) {
                    free(temp);
                    free(output);
                    return NULL;
                }
                
                /* Dequantize */
                extern int kser_dequantize(const void* src, float* dst, uint64_t n, KSerDtype dtype);
                kser_dequantize(temp, output, n_elements, entry->dtype);
                free(temp);
                return output;
            }
            
            /* Dequantize from mmap */
            extern int kser_dequantize(const void* src, float* dst, uint64_t n, KSerDtype dtype);
            kser_dequantize(src_data, output, n_elements, entry->dtype);
            
            return output;
        }
    }
    
    return NULL; /* Tensor not found */
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
    
    if (r->mmap_data) {
        munmap(r->mmap_data, r->mmap_size);
    }
    if (r->fp) fclose(r->fp);
    if (r->fd >= 0) close(r->fd);
    
    free(r->tensors);
    free(r);
}

KSerInfo kser_file_info(const char* path) {
    KSerInfo info = {0};
    
    struct stat st;
    if (stat(path, &st) < 0) return info;
    info.file_size = st.st_size;
    
    /* Quick peek at header */
    FILE* fp = fopen(path, "rb");
    if (!fp) return info;
    
    uint8_t header[KSER_HEADER_SIZE];
    if (fread(header, 1, KSER_HEADER_SIZE, fp) != KSER_HEADER_SIZE) {
        fclose(fp);
        return info;
    }
    
    if (!validate_header(header)) {
        fclose(fp);
        return info;
    }
    
    /* Read config */
    KSerConfig cfg;
    if (fread(&cfg, sizeof(KSerConfig), 1, fp) == 1) {
        info.valid = 1;
        info.vocab_size = cfg.vocab_size;
        info.dim = cfg.dim;
        info.n_layers = cfg.n_layers;
        info.dtype = cfg.dtype;
        memcpy(info.model_name, cfg.model_name, 64);
        
        /* Estimate n_params */
        info.n_params = (uint64_t)cfg.vocab_size * cfg.dim * 2; /* embedding + head */
        info.n_params += (uint64_t)cfg.dim * cfg.state_size * 4 * cfg.n_layers; /* SSM */
        info.n_params += (uint64_t)cfg.dim * cfg.dim * 4 * cfg.n_layers; /* MLP + projections */
    }
    
    fclose(fp);
    return info;
}
