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
    DATA_MODALITY_VISION,   /* Images: CIFAR, MNIST, etc. */
    DATA_MODALITY_TEXT,     /* Text: token IDs or embeddings */
    DATA_MODALITY_AUDIO,    /* Audio: spectrograms or raw waveform */
    DATA_MODALITY_TIME_SERIES, /* Any 1D/2D time series */
    DATA_MODALITY_SYNTHETIC /* Random data for testing */
} DataModality;

/* Detect modality from file extension or path */
static DataModality detect_modality(const char *path) {
    if (!path || path[0] == '\0') return DATA_MODALITY_SYNTHETIC;
    
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
    
    /* Generate dummy labels if num_classes > 0 */
    if (num_classes > 0) {
        *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
        if (*labels) {
            for (size_t i = 0; i < n; i++) {
                (*labels)[i] = (uint32_t)(i % num_classes);
            }
        }
    } else {
        *labels = NULL;
    }
    
    printf("Loaded %zu samples successfully.\n", n);
    return 0;
}

/* Vision loader - CIFAR-10 style images */
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
