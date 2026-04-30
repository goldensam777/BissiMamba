/* train.c — K-Mamba Training CLI
 * Usage: ./train --config config.json --data <dataset_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "configs.h"
#include "trainer.h"
#include "kmamba_kernels.h"

/* ============================================================================
 * Multimodal Dataset Loader
 * Supports: vision (.bin, .raw), text (.txt), audio (.wav, .raw), any custom format
 * Format de sortie universel: [N, seq_len, dim] float + labels uint32_t
 * ============================================================================ */

typedef enum {
    DATA_MODALITY_GENERIC,  /* Raw binary, user-preprocessed */
    DATA_MODALITY_MNIST,    /* MNIST: idx3-ubyte + idx1-ubyte */
    DATA_MODALITY_CIFAR10,  /* CIFAR-10: data_batch_*.bin */
    DATA_MODALITY_VISION,   /* Generic images: PNG, JPG */
    DATA_MODALITY_TEXT,     /* Text: token IDs or embeddings */
    DATA_MODALITY_AUDIO,    /* Audio: spectrograms or raw waveform */
    DATA_MODALITY_TIME_SERIES, /* Any 1D/2D time series */
    DATA_MODALITY_SYNTHETIC /* Random data for testing */
} DataModality;

/* Detect modality from file extension or path */
static DataModality detect_modality(const char *path) {
    if (!path || path[0] == '\0') return DATA_MODALITY_SYNTHETIC;
    
    /* Check for MNIST pattern: *-images-idx3-ubyte */
    if (strstr(path, "-images-idx3-ubyte") || strstr(path, "-labels-idx1-ubyte"))
        return DATA_MODALITY_MNIST;
    
    /* Check for CIFAR-10 pattern: data_batch_*.bin or test_batch.bin */
    if (strstr(path, "data_batch_") || strstr(path, "test_batch.bin"))
        return DATA_MODALITY_CIFAR10;
    
    const char *ext = strrchr(path, '.');
    if (!ext) return DATA_MODALITY_GENERIC;
    
    if (strcasecmp(ext, ".txt") == 0 || strcasecmp(ext, ".text") == 0)
        return DATA_MODALITY_TEXT;
    if (strcasecmp(ext, ".wav") == 0 || strcasecmp(ext, ".mp3") == 0 || strcasecmp(ext, ".raw") == 0)
        return DATA_MODALITY_AUDIO;
    if (strcasecmp(ext, ".png") == 0 || strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0)
        return DATA_MODALITY_VISION;
    if (strcasecmp(ext, ".bin") == 0 || strcasecmp(ext, ".npy") == 0)
        return DATA_MODALITY_GENERIC;
    
    return DATA_MODALITY_GENERIC; /* Default: assume preprocessed binary */
}

/* MNIST loader - native format (idx3-ubyte for images, idx1-ubyte for labels) */
static int load_mnist(const char *path, float **data, uint32_t **labels,
                      size_t *num_samples, size_t L, size_t D, int num_classes) {
    /* Parse MNIST path format: /path/to/train-images-idx3-ubyte */
    char images_path[512];
    char labels_path[512];
    
    strncpy(images_path, path, sizeof(images_path) - 1);
    images_path[sizeof(images_path) - 1] = '\0';
    
    /* Construct labels path: train-images-idx3-ubyte -> train-labels-idx1-ubyte */
    char *images_name = strrchr(images_path, '/');
    if (!images_name) images_name = images_path;
    else images_name++; /* Skip the slash */
    
    char dir_buf[512] = "";
    const char *dir = dir_buf;
    if (images_name != images_path) {
        size_t dir_len = images_name - images_path;
        if (dir_len < sizeof(dir_buf)) {
            memcpy(dir_buf, images_path, dir_len);
            dir_buf[dir_len] = '\0';
        }
    }
    
    if (strncmp(images_name, "train-images", 12) == 0) {
        snprintf(labels_path, sizeof(labels_path), "%s%s%s", dir, dir[0] ? "/" : "", "train-labels-idx1-ubyte");
    } else if (strncmp(images_name, "t10k-images", 11) == 0) {
        snprintf(labels_path, sizeof(labels_path), "%s%s%s", dir, dir[0] ? "/" : "", "t10k-labels-idx1-ubyte");
    } else {
        /* Fallback: try to construct labels path by replacing 'images' with 'labels' and idx3 with idx1 */
        snprintf(labels_path, sizeof(labels_path), "%s.labels", path);
    }
    
    printf("Loading MNIST from: %s\n", path);
    printf("Expected labels: %s\n", labels_path);
    
    /* Read images file */
    FILE *img_fp = fopen(path, "rb");
    if (!img_fp) {
        fprintf(stderr, "Error: Cannot open images file %s\n", path);
        return -1;
    }
    
    /* Read MNIST header (big-endian) */
    uint32_t magic, num_images, rows, cols;
    fread(&magic, 4, 1, img_fp);
    fread(&num_images, 4, 1, img_fp);
    fread(&rows, 4, 1, img_fp);
    fread(&cols, 4, 1, img_fp);
    
    /* Convert from big-endian */
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    
    if (magic != 0x00000803) { /* 2051 for images */
        fprintf(stderr, "Error: Invalid MNIST images magic number: %u\n", magic);
        fclose(img_fp);
        return -1;
    }
    
    printf("MNIST: %u images, %ux%u pixels\n", num_images, rows, cols);
    
    size_t n = num_images;
    size_t img_size = rows * cols;
    
    /* Allocate and read images (normalize to [0,1]) */
    *data = (float*)malloc(n * img_size * sizeof(float));
    if (!*data) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        fclose(img_fp);
        return -1;
    }
    
    uint8_t *img_buf = (uint8_t*)malloc(img_size);
    if (!img_buf) {
        fclose(img_fp);
        return -1;
    }
    
    for (size_t i = 0; i < n; i++) {
        if (fread(img_buf, 1, img_size, img_fp) != img_size) {
            fprintf(stderr, "Warning: Could not read image %zu\n", i);
            break;
        }
        for (size_t j = 0; j < img_size; j++) {
            (*data)[i * img_size + j] = img_buf[j] / 255.0f;
        }
    }
    
    free(img_buf);
    fclose(img_fp);
    
    /* Read labels file */
    FILE *lbl_fp = fopen(labels_path, "rb");
    if (lbl_fp) {
        uint32_t lbl_magic, num_labels;
        fread(&lbl_magic, 4, 1, lbl_fp);
        fread(&num_labels, 4, 1, lbl_fp);
        
        lbl_magic = __builtin_bswap32(lbl_magic);
        num_labels = __builtin_bswap32(num_labels);
        
        if (lbl_magic == 0x00000801 && num_labels == n) { /* 2049 for labels */
            *labels = (uint32_t*)malloc(n * sizeof(uint32_t));
            if (*labels) {
                uint8_t *lbl_buf = (uint8_t*)malloc(n);
                if (lbl_buf) {
                    fread(lbl_buf, 1, n, lbl_fp);
                    for (size_t i = 0; i < n; i++) {
                        (*labels)[i] = lbl_buf[i];
                    }
                    free(lbl_buf);
                    printf("Loaded %zu MNIST labels\n", n);
                }
            }
        } else {
            printf("Warning: MNIST labels file mismatch. Using dummy labels.\n");
        }
        fclose(lbl_fp);
    } else {
        printf("Warning: Could not open labels file. Using dummy labels.\n");
        if (num_classes > 0) {
            *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
            if (*labels) {
                for (size_t i = 0; i < n; i++) {
                    (*labels)[i] = (uint32_t)(i % num_classes);
                }
            }
        }
    }
    
    *num_samples = n;
    printf("Loaded %zu MNIST samples successfully.\n", n);
    return 0;
}

/* Generic binary loader - loads raw float32 files */
static int load_modality_generic(const char *path, float **data, uint32_t **labels,
                                  size_t *num_samples, size_t L, size_t D, int num_classes) {
    if (!path || path[0] == '\0') {
        /* No path provided - generate synthetic data */
        printf("No dataset path provided. Using synthetic data.\n");
        size_t n = 100;
        *num_samples = n;
        *data = (float*)calloc(n * L * D, sizeof(float));
        *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
        if (!*data || !*labels) return -1;
        for (size_t i = 0; i < n * L * D; i++) {
            (*data)[i] = (float)(rand() % 256) / 256.0f;
        }
        for (size_t i = 0; i < n; i++) {
            (*labels)[i] = (uint32_t)(rand() % num_classes);
        }
        return 0;
    }

    printf("Loading binary data from: %s\n", path);
    printf("Expected format: [N, %zu, %zu] float32 flattened\n", L, D);
    
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return -1;
    }
    
    /* Get file size */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    /* Calculate expected size per sample */
    size_t sample_size = L * D * sizeof(float);
    size_t n = file_size / sample_size;
    
    if (n == 0) {
        fprintf(stderr, "Error: File too small (need at least %zu bytes)\n", sample_size);
        fclose(fp);
        return -1;
    }
    
    printf("Loading %zu samples (%ld bytes)...\n", n, file_size);
    
    /* Allocate and read data */
    *data = (float*)malloc(n * sample_size);
    if (!*data) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        fclose(fp);
        return -1;
    }
    
    size_t read = fread(*data, 1, n * sample_size, fp);
    fclose(fp);
    
    if (read != n * sample_size) {
        fprintf(stderr, "Warning: Only read %zu of %zu bytes\n", read, n * sample_size);
    }
    
    *num_samples = n;
    
    /* Try to load labels from .labels file */
    char labels_path[512];
    snprintf(labels_path, sizeof(labels_path), "%s.labels", path);
    FILE *lfp = fopen(labels_path, "rb");
    if (lfp) {
        fseek(lfp, 0, SEEK_END);
        long labels_size = ftell(lfp);
        fseek(lfp, 0, SEEK_SET);
        size_t num_labels = labels_size / sizeof(uint32_t);
        if (num_labels == n) {
            *labels = (uint32_t*)malloc(n * sizeof(uint32_t));
            if (*labels) {
                fread(*labels, sizeof(uint32_t), n, lfp);
                printf("Loaded %zu labels from %s.labels\n", n, path);
            }
        } else {
            printf("Warning: labels file has %zu entries, expected %zu. Using dummy labels.\n", 
                   num_labels, n);
        }
        fclose(lfp);
    } else {
        /* Generate dummy labels if num_classes > 0 */
        if (num_classes > 0) {
            *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
            if (*labels) {
                for (size_t i = 0; i < n; i++) {
                    (*labels)[i] = (uint32_t)(i % num_classes);
                }
            }
            printf("Generated dummy labels (0-%d cycle)\n", num_classes - 1);
        } else {
            *labels = NULL;
        }
    }
    
    printf("Loaded %zu samples successfully.\n", n);
    return 0;
}

/* CIFAR-10 native loader - loads from data_batch_*.bin files */
static int load_cifar10(const char *path, float **data, uint32_t **labels,
                        size_t *num_samples, size_t L, size_t D, int num_classes) {
    printf("Loading CIFAR-10 from: %s\n", path);
    
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return -1;
    }
    
    /* CIFAR-10 format: each record is 3073 bytes (1 label + 3072 pixels)
     * 10000 records per batch file
     * Pixels are stored in planar format: R (1024) + G (1024) + B (1024)
     * Each image is 32x32x3 = 3072 bytes
     */
    const size_t img_size = 32 * 32 * 3;  /* 3072 */
    const size_t record_size = 1 + img_size;  /* 3073: 1 byte label + 3072 bytes image */
    
    /* Count records in file */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    size_t n = file_size / record_size;
    if (n == 0) {
        fprintf(stderr, "Error: File too small or invalid format\n");
        fclose(fp);
        return -1;
    }
    
    printf("Loading %zu CIFAR-10 images (32x32x3)...\n", n);
    
    /* Allocate memory */
    *data = (float*)malloc(n * img_size * sizeof(float));
    *labels = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!*data || !*labels) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        fclose(fp);
        return -1;
    }
    
    /* Read all records */
    uint8_t *record = (uint8_t*)malloc(record_size);
    if (!record) {
        fclose(fp);
        return -1;
    }
    
    for (size_t i = 0; i < n; i++) {
        if (fread(record, 1, record_size, fp) != record_size) {
            fprintf(stderr, "Warning: Could not read record %zu\n", i);
            break;
        }
        
        /* First byte is label */
        (*labels)[i] = record[0];
        
        /* Convert pixel values to float [0,1] - CIFAR stores as uint8 */
        for (size_t j = 0; j < img_size; j++) {
            (*data)[i * img_size + j] = record[1 + j] / 255.0f;
        }
    }
    
    free(record);
    fclose(fp);
    
    *num_samples = n;
    printf("Loaded %zu CIFAR-10 samples successfully.\n", n);
    return 0;
}

/* Generic vision loader stub - for PNG/JPG files */
static int load_modality_vision(const char *path, float **data, uint32_t **labels,
                                 size_t *num_samples, size_t L, size_t D, int num_classes) {
    printf("Loading vision data from: %s\n", path);
    printf("Expected: images flattened to [N, %zu, %zu]\n", L, D);
    
    /* TODO: Implement actual image loading (libpng, libjpeg, etc.) */
    /* For now, synthetic data representing images */
    
    size_t n = 100;
    *num_samples = n;
    *data = (float*)calloc(n * L * D, sizeof(float));
    *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (!*data || !*labels) return -1;
    
    /* Simulate normalized image pixels [0,1] */
    for (size_t i = 0; i < n * L * D; i++) {
        (*data)[i] = (float)(rand() % 256) / 255.0f;
    }
    for (size_t i = 0; i < n; i++) {
        (*labels)[i] = (uint32_t)(rand() % num_classes);
    }
    return 0;
}

/* Text loader - token sequences or embeddings */
static int load_modality_text(const char *path, float **data, uint32_t **labels,
                               size_t *num_samples, size_t L, size_t D, int num_classes) {
    printf("Loading text data from: %s\n", path);
    printf("Expected: tokenized text [N, %zu, %zu] as embeddings or one-hot\n", L, D);
    
    /* TODO: Implement text loading with tokenizer */
    /* For now, synthetic data representing token sequences */
    
    size_t n = 100;
    *num_samples = n;
    *data = (float*)calloc(n * L * D, sizeof(float));
    *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (!*data || !*labels) return -1;
    
    /* Simulate token embeddings with sparse structure */
    for (size_t i = 0; i < n * L * D; i++) {
        (*data)[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; /* [-1, 1] */
    }
    for (size_t i = 0; i < n; i++) {
        (*labels)[i] = (uint32_t)(rand() % num_classes);
    }
    return 0;
}

/* Audio loader - spectrograms or waveform */
static int load_modality_audio(const char *path, float **data, uint32_t **labels,
                                size_t *num_samples, size_t L, size_t D, int num_classes) {
    printf("Loading audio data from: %s\n", path);
    printf("Expected: spectrograms [N, %zu, %zu] or waveform patches\n", L, D);
    
    /* TODO: Implement audio loading (libsndfile, etc.) */
    /* For now, synthetic data representing audio features */
    
    size_t n = 100;
    *num_samples = n;
    *data = (float*)calloc(n * L * D, sizeof(float));
    *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (!*data || !*labels) return -1;
    
    /* Simulate audio spectrogram-like values */
    for (size_t i = 0; i < n * L * D; i++) {
        (*data)[i] = ((float)rand() / RAND_MAX); /* [0, 1] like magnitude spectrogram */
    }
    for (size_t i = 0; i < n; i++) {
        (*labels)[i] = (uint32_t)(rand() % num_classes);
    }
    return 0;
}

/* ============================================================================
 * Main entry point: Multimodal dataset loader
 * Auto-detects modality from file extension and dispatches to appropriate loader
 * ============================================================================ */
static int load_dataset(const char *data_path, float **data, uint32_t **labels,
                       size_t *num_samples, size_t L, size_t D, int num_classes) {
    DataModality mod = detect_modality(data_path);
    
    switch (mod) {
        case DATA_MODALITY_MNIST:
            return load_mnist(data_path, data, labels, num_samples, L, D, num_classes);
        case DATA_MODALITY_CIFAR10:
            return load_cifar10(data_path, data, labels, num_samples, L, D, num_classes);
        case DATA_MODALITY_VISION:
            return load_modality_vision(data_path, data, labels, num_samples, L, D, num_classes);
        case DATA_MODALITY_TEXT:
            return load_modality_text(data_path, data, labels, num_samples, L, D, num_classes);
        case DATA_MODALITY_AUDIO:
            return load_modality_audio(data_path, data, labels, num_samples, L, D, num_classes);
        case DATA_MODALITY_SYNTHETIC:
            printf("No dataset path provided. Using synthetic data for testing.\n");
            /* Fall through to generic */
        case DATA_MODALITY_GENERIC:
        case DATA_MODALITY_TIME_SERIES:
        default:
            return load_modality_generic(data_path, data, labels, num_samples, L, D, num_classes);
    }
}

static int parse_flag_value(int argc, char **argv, int *i, const char *flag, const char **out) {
    const size_t n = strlen(flag);
    if (strncmp(argv[*i], flag, n) != 0) return 0;

    if (argv[*i][n] == '=') {
        *out = argv[*i] + n + 1;
        return 1;
    }
    if (argv[*i][n] == '\0' && (*i + 1) < argc) {
        (*i)++;
        *out = argv[*i];
        return 1;
    }
    return -1;
}

static void default_checkpoint_path(const KMambaFullConfig *cfg, char *out, size_t out_cap) {
    if (cfg->checkpoint_path[0] != '\0' &&
        strcmp(cfg->checkpoint_path, "checkpoint.ser") != 0) {
        snprintf(out, out_cap, "%s", cfg->checkpoint_path);
        return;
    }
    if (cfg->model.model_name[0] != '\0') {
        snprintf(out, out_cap, "inference/%s.ser", cfg->model.model_name);
        return;
    }
    snprintf(out, out_cap, "checkpoint.ser");
}

int main(int argc, char **argv) {
    const char *config_path = "config.json";
    const char *data_path_override = NULL;
    const char *checkpoint_path_override = NULL;
    int positional_config_set = 0;

    /* Parse CLI arguments */
    size_t batch_size = 8;  /* default */
    size_t epochs = 3;      /* default */
    int backend_override = -1; /* -1 = use config */

    for (int i = 1; i < argc; i++) {
        const char *val = NULL;
        int rc = parse_flag_value(argc, argv, &i, "--config", &val);
        if (rc == 1) {
            config_path = val;
            positional_config_set = 1;
            continue;
        }
        if (rc == -1) {
            fprintf(stderr, "Error: --config requires a value\n");
            return 1;
        }

        rc = parse_flag_value(argc, argv, &i, "--data", &val);
        if (rc == 1) {
            data_path_override = val;
            continue;
        }
        if (rc == -1) {
            fprintf(stderr, "Error: --data requires a value\n");
            return 1;
        }

        rc = parse_flag_value(argc, argv, &i, "--checkpoint", &val);
        if (rc == 1) {
            checkpoint_path_override = val;
            continue;
        }
        if (rc == -1) {
            fprintf(stderr, "Error: --checkpoint requires a value\n");
            return 1;
        }

        if (strncmp(argv[i], "--batch_size=", 13) == 0) {
            batch_size = (size_t)atoi(argv[i] + 13);
            continue;
        }
        if (strncmp(argv[i], "--epochs=", 9) == 0) {
            epochs = (size_t)atoi(argv[i] + 9);
            continue;
        }
        if (strncmp(argv[i], "--backend=", 10) == 0) {
            const char *b = argv[i] + 10;
            if (strcmp(b, "cpu") == 0) backend_override = 1;
            else if (strcmp(b, "gpu") == 0) backend_override = 2;
            continue;
        }

        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s --config <config.json> --data <dataset_dir> [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --checkpoint=<path> Override checkpoint (.ser) path\n");
            printf("  --batch_size=N       Batch size (default: 8)\n");
            printf("  --epochs=N           Number of epochs (default: 3)\n");
            printf("  --backend=cpu|gpu    Force backend\n");
            printf("Example: %s --config config.json --data data/ --epochs=10\n", argv[0]);
            return 0;
        }

        if (argv[i][0] != '-' && !positional_config_set) {
            config_path = argv[i];
            positional_config_set = 1;
            continue;
        }

        fprintf(stderr, "Error: unknown argument '%s'\n", argv[i]);
        return 1;
    }

    if (argc < 2) {
        printf("Usage: %s --config <config.json> --data <dataset_dir> [options]\n", argv[0]);
        printf("Options:\n");
        printf("  --checkpoint=<path> Override checkpoint (.ser) path\n");
        printf("  --batch_size=N       Batch size (default: 8)\n");
        printf("  --epochs=N           Number of epochs (default: 3)\n");
        printf("  --backend=cpu|gpu    Force backend\n");
        printf("Example: %s --config config.json --data data/ --batch_size=16 --epochs=10\n", argv[0]);
        return 1;
    }

    /* Load configuration */
    KMambaFullConfig cfg;
    if (kmamba_configs_load_json(&cfg, config_path) != 0) {
        fprintf(stderr, "Error: failed to load %s\n", config_path);
        return 1;
    }

    if (data_path_override && data_path_override[0] != '\0') {
        strncpy(cfg.data_path, data_path_override, sizeof(cfg.data_path) - 1);
        cfg.data_path[sizeof(cfg.data_path) - 1] = '\0';
    }
    if (backend_override >= 0) {
        cfg.backend = backend_override;
    }

    char checkpoint_path[512] = {0};
    default_checkpoint_path(&cfg, checkpoint_path, sizeof(checkpoint_path));
    if (checkpoint_path_override && checkpoint_path_override[0] != '\0') {
        snprintf(checkpoint_path, sizeof(checkpoint_path), "%s", checkpoint_path_override);
    }

    /* Load model from checkpoint */
    printf("Loading model from %s...\n", checkpoint_path);
    KMamba *model = kmamba_load(checkpoint_path, 1, &cfg.optim,
                                  cfg.optim.lr, cfg.optim.weight_decay);
    if (!model) {
        fprintf(stderr, "Error: failed to load %s\n", checkpoint_path);
        fprintf(stderr, "Run ./model --config %s --serialize ser first.\n", config_path);
        return 1;
    }
    printf("Model loaded successfully.\n");

    /* Determine number of classes (for CIFAR-10 style tasks) */
    int num_classes = 10; /* Default for CIFAR-10 */
    
    /* Load training data */
    printf("Loading training data...\n");
    float *data = NULL;
    uint32_t *labels = NULL;
    size_t num_samples = 0;
    size_t L = cfg.model.seq_len;
    size_t D = cfg.model.dim;
    
    if (load_dataset(cfg.data_path, &data, &labels, &num_samples, L, D, num_classes) != 0) {
        fprintf(stderr, "Error: failed to load training data\n");
        kmamba_free(model);
        return 1;
    }
    printf("Loaded %zu samples.\n", num_samples);

    /* Create trainer with GC policy */
    TrainerGCConfig gc_cfg = {
        .policy = TRAINER_GC_NONE,  /* Default: no gradient checkpointing */
        .checkpoint_every_n = 2
    };
    
    Trainer *trainer = trainer_create(model, &gc_cfg);
    if (!trainer) {
        fprintf(stderr, "Error: failed to create trainer\n");
        free(data);
        free(labels);
        kmamba_free(model);
        return 1;
    }

    /* Run training */
    printf("\nStarting training...\n");
    printf("  batch_size=%zu, epochs=%zu\n", batch_size, epochs);
    printf("  backend=%s\n\n", 
           cfg.backend == 1 ? "CPU" : (cfg.backend == 2 ? "GPU" : "AUTO"));

    TrainerMetrics metrics = trainer_run(
        trainer,
        data, labels, num_samples,
        L, D, num_classes,
        batch_size, epochs,
        checkpoint_path,
        1  /* verbose = 1: print progress tables */
    );

    printf("\nTraining complete.\n");
    printf("Final loss: %.4f, accuracy: %.2f%%\n", 
           metrics.loss, metrics.accuracy * 100.0f);

    /* Cleanup */
    trainer_free(trainer);
    free(data);
    free(labels);
    kmamba_free(model);
    
    return 0;
}
