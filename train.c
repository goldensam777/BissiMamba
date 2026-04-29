/* train.c — K-Mamba Training CLI
 * Usage: ./train --config config.json --data <dataset_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "configs.h"
#include "trainer.h"
#include "kmamba_kernels.h"

/* Stub data loader for vision tasks (CIFAR-10 style) */
static int load_vision_data(const char *data_path, float **data, uint32_t **labels,
                           size_t *num_samples, size_t L, size_t D, int num_classes) {
    if (data_path && data_path[0] != '\0') {
        printf("Dataset path: %s (synthetic loader stub)\n", data_path);
    }
    /* For now, generate synthetic data */
    size_t n = 100; /* Small synthetic dataset */
    *num_samples = n;
    *data = (float*)calloc(n * L * D, sizeof(float));
    *labels = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (!*data || !*labels) return -1;
    
    /* Fill with random data and random labels */
    for (size_t i = 0; i < n * L * D; i++) {
        (*data)[i] = (float)(rand() % 256) / 256.0f;
    }
    for (size_t i = 0; i < n; i++) {
        (*labels)[i] = (uint32_t)(rand() % num_classes);
    }
    return 0;
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
    
    if (load_vision_data(cfg.data_path, &data, &labels, &num_samples, L, D, num_classes) != 0) {
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
