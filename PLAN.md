# K-Mamba Development Plans

> Consolidated project plans for k-mamba serialization and tokenizer strategy.

---

## Plan 1: libkser Serialization Format (.ser v1)

**Status**: ✅ Implementation Complete (23 Avril 2026)

**Location**: `libs/kser/`

### Overview

Bibliothèque C cross-platform (POSIX + Win32) pour sérialisation des modèles k-mamba avec vocabulaire embarqué et quantization FP32/FP16/BF16/INT8.

### Architecture

```
k-mamba/
├── libs/
│   └── kser/
│       ├── include/
│       │   ├── kser.h
│       │   └── kser_checksum.h
│       ├── src/
│       │   ├── kser_write.c    # Writer avec ordre d'écriture strict
│       │   ├── kser_read.c     # Lecture + mmap
│       │   ├── kser_quantize.c # FP32↔FP16/BF16/INT8
│       │   └── kser_checksum.c # SHA256 streaming
│       ├── tests/
│       │   └── test_roundtrip.c
│       └── Makefile
└── src/
    └── kmamba_ser.c        # Wrapper k-mamba → libkser
```

### Format .ser v1 Specification (Final)

```
[0]       16 bytes  Magic (SERENITY + η + version)
[16]      96 bytes  KSerConfig (packed, 96 bytes)
[112]      4 bytes  vocab_count (uint32)
[116]   variable    Vocab entries: (id:u32)(len:u16)(token)
[116+V] variable    Tensor data (offsets absolus)
[116+V+D] 4 bytes   tensor_count (uint32)
[+4]    variable    KSerTensorEntry[] (72 bytes chacun)
[end-32]  32 bytes  SHA256 de tout le précédent
```

#### Header (16 bytes)
| Offset | Size | Content |
|--------|------|---------|
| 0 | 8 | "SERENITY" (ASCII) |
| 8 | 2 | 0xCE 0xB7 (η grec, UTF-8) |
| 10 | 1 | 0x01 (version) |
| 11 | 5 | Reserved (0x00) |

#### KSerConfig (96 bytes - packed)
```c
typedef struct {
    uint32_t vocab_size;       /*  4 bytes */
    uint32_t dim;              /*  4 bytes */
    uint32_t state_size;       /*  4 bytes */
    uint32_t n_layers;         /*  4 bytes */
    uint32_t seq_len;          /*  4 bytes */
    uint32_t d_conv;           /*  4 bytes */
    float    expand_factor;    /*  4 bytes */
    uint8_t  dtype;            /*  1 byte  */
    uint8_t  _pad[3];          /*  3 bytes padding */
    char     model_name[64];   /* 64 bytes */
    /* total: 96 bytes */
} KSerConfig;
```

#### KSerTensorEntry (72 bytes - packed)
```c
typedef struct {
    char     name[32];         /* 32 bytes */
    uint32_t shape[4];         /* 16 bytes */
    uint8_t  dtype;            /*  1 byte  */
    uint8_t  _pad[7];          /*  7 bytes padding */
    uint64_t offset;           /*  8 bytes — absolute offset */
    uint64_t size_bytes;       /*  8 bytes */
    /* total: 72 bytes */
} KSerTensorEntry;
```

### Writer Rules

1. `add_vocab()` doit être appelé **avant** le premier `add_tensor()`
2. Écriture atomique : `.tmp` → rename → fichier final
3. Checksum SHA256 calculé sur tout le fichier sauf les 32 derniers bytes

### API

```c
/* === Writer === */
KSerWriter* kser_writer_create(const char* path, const KSerConfig* cfg);
int         kser_writer_add_vocab(KSerWriter* w, uint32_t id,
                                   const char* token, uint16_t len);
int         kser_writer_add_tensor(KSerWriter* w, const char* name,
                                    const void* data, const uint32_t shape[4],
                                    KSerDtype src_dtype, KSerDtype storage_dtype);
int         kser_writer_finalize(KSerWriter* w);
void        kser_writer_free(KSerWriter* w);

/* === Reader === */
KSerReader*            kser_reader_open(const char* path);
const KSerConfig*      kser_reader_config(KSerReader* r);
int                    kser_reader_load_vocab(KSerReader* r,
                                             KSerVocabCallback cb, void* userdata);
float*                 kser_reader_load_tensor(KSerReader* r, const char* name);
void                   kser_reader_close(KSerReader* r);

/* === Info === */
KSerInfo kser_file_info(const char* path);

/* === Quantization === */
int    kser_quantize(const float* src, void* dst, uint64_t n, KSerDtype dtype);
int    kser_dequantize(const void* src, float* dst, uint64_t n, KSerDtype dtype);
size_t kser_dtype_size(KSerDtype dtype);
size_t kser_quantize_size(uint64_t n, KSerDtype dtype);

/* === SHA256 === */
void kser_sha256(const uint8_t* data, size_t len, uint8_t hash[32]);
int  kser_sha256_file(FILE* fp, long end_pos, uint8_t hash[32]);
```

### Quantization Matrix

| Source | Destination | Méthode |
|--------|-------------|---------|
| FP32 | FP32 | Copie directe |
| FP32 | FP16 | Truncation IEEE 754 |
| FP32 | BF16 | Truncation mantisse 16 bits |
| FP32 | INT8 | Min-max scaling + round (8-byte header) |

### Build

```bash
# libkser
make -C libs/kser

# Tests
make -C libs/kser test
```

---

## Plan 2: cl100k_base Tokenizer Integration (Hybrid Vocab Strategy)

**Status**: 📝 Design Phase

### Strategy Overview

Stratégie hybride: 32K vocab pour petits modèles (PC local), 100K cl100k_base pour Azure Large (cloud).

### Model Configurations

#### CPU Small (32K Vocab) - PC Local
- **VOCAB**: 32,768 | **DIM**: 384 | **STATE**: 512 | **LAYERS**: 2 | **SEQ**: 256
- **Params**: ~120M (embedding: 32K × 384 = 12.3M)
- **ConvND**: Separable cascade
- **Data**: `data/kmamba_tiny_1M.txt` ou byte fallback
- **Location**: PC local (laptop CPU)
- **Usage**: Inference rapide, tests

#### GPU Small MX450 (32K Vocab) - PC Local
- **VOCAB**: 32,768 | **DIM**: 512 | **STATE**: 1024 | **LAYERS**: 4 | **SEQ**: 512
- **Params**: ~350M (embedding: 32K × 512 = 16.7M)
- **ConvND**: Dense unified
- **Data**: `data/kmamba_cpu_10M.txt`
- **Location**: PC local (MX450 2GB VRAM)
- **Usage**: Training local, expérimentations

#### GPU Large Azure (cl100k_base - 100K Vocab) - Azure Cloud
- **VOCAB**: 100,277 | **DIM**: 3072 | **STATE**: 4096 | **LAYERS**: 24 | **SEQ**: 2048
- **Params**: ~7.5B (embedding: 100K × 3072 = 307M)
- **ConvND**: Dense unified
- **Training**: **Full GPU** (embedding + blocks + head all on GPU)
- **Data**: `data/conversations.txt` (1GB) pre-tokenized
- **Location**: **Microsoft Azure** (GPU instances A100/V100)
- **Usage**: Production training, modèle final distribuable

### Rationale

| Modèle | Vocab | Location | Raison |
|--------|-------|----------|--------|
| CPU Small | 32K | PC Local | Rapide, pas besoin qualité tokenization |
| MX450 | 32K | PC Local | VRAM limitée (2GB), training local expérimental |
| Azure Large | 100K | Cloud | Qualité code GPT-4 niveau, compute illimité |

### Tokenizer Setup

#### 32K Models (PC Local)
- **Option A**: Byte-level (0-255) + padding à 32K
- **Option B**: BPE simple 32K tokens sur corpus
- **Fallback**: Caractère ASCII direct
- **No pre-tokenization needed**

#### 100K Model (Cloud)
- **Tiktoken cl100k_base**: 100,277 tokens
- Pre-tokenization: `scripts/prepare_cl100k.py`
- Format binaire: uint32 array `.tok`
- Upload to Kaggle/Colab datasets

### Memory Impact

| Config | Vocab | Embedding | Head | Total Impact |
|--------|-------|-----------|------|--------------|
| 32K/384 | 32,768 | 12.3M | 12.3M | +96MB |
| 32K/512 | 32,768 | 16.7M | 16.7M | +131MB |
| 100K/3072 | 100,277 | 307M | 307M | +2.4GB |

### Implementation Roadmap

```
Phase 1: PC Local Models (32K vocab)
├── models/kmamba_cpu.c: VOCAB_SIZE=32768
├── models/kmamba_cuda.cu: VOCAB_SIZE=32768
├── Tokenizer: byte ou simple BPE
└── Build local avec `make`

Phase 2: Cloud Integration (100K vocab)
├── models/kmamba_azure_large.cu: VOCAB_SIZE=100277
├── Tokenizer: tiktoken cl100k_base
├── Data pipeline: `.tok` uint32 files
└── Setup Kaggle/Colab avec scripts dédiés

Phase 3: Hybrid Deployment
├── kaggle/kmamba_azure_large.ipynb: Notebook Kaggle 7B params
├── colab/kmamba_azure_large.ipynb: Notebook Colab 7B params
└── Pre-tokenized data upload via Kaggle datasets / Google Drive
```

### Next Steps

1. [ ] Implémenter modèles PC local (32K vocab)
2. [ ] Créer setup cloud pour Azure Large (100K vocab)
3. [ ] Test local sur MX450 puis deploy sur cloud
4. [ ] Intégration avec libkser pour save/load modèles

---

## Cross-Cutting Concerns

### Philosophy
- **Zero dependency** — Pas de bibliothèque externe, pas de package manager
- **Inline kernels** — Fonctions simples en C, pas de BLAS complexe
- **Makefile simple** — 20 lignes, pas de CMake
- **NASM + C** — Assembleur pour hot paths, C pour le reste

### Integration Points

```
libkser (.ser v1) ──► kmamba_ser.c ──► kmamba save/load
                            │
                            ▼
                    cl100k integration ──► tokenizer strategy
                            │
                            ▼
                    PC Local (32K) vs Azure (100K)
```

---

## Auteur

**YEVI Mawuli Peniel Samuel** — IFRI-UAC (Bénin)

Devise : **"Optima, immo absoluta perfectio"**
