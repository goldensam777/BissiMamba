/* train.c — K-Mamba Training CLI
 * Usage: ./train <config.json> [--batch_size=N] [--epochs=N] [--backend=cpu|gpu]
 * Loads model from checkpoint.ser and runs training via trainer_run.
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
    (void)data_path;
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

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <config.json> [options]\n", argv[0]);
        printf("Options:\n");
        printf("  --batch_size=N    Batch size (default: from config or 8)\n");
        printf("  --epochs=N        Number of epochs (default: from config or 3)\n");
        printf("  --backend=cpu|gpu Force backend (default: from config or auto)\n");
        printf("Example: %s configs/cifar10.json --batch_size=16 --epochs=10\n", argv[0]);
        return 1;
    }

    /* Load configuration */
    KMambaFullConfig cfg;
    if (kmamba_configs_load_json(&cfg, argv[1]) != 0) {
        fprintf(stderr, "Error: failed to load %s\n", argv[1]);
        return 1;
    }

    /* Parse CLI arguments */
    size_t batch_size = 8;  /* default */
    size_t epochs = 3;      /* default */
    int backend_override = -1; /* -1 = use config */
    
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--batch_size=", 13) == 0) {
            batch_size = (size_t)atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--epochs=", 9) == 0) {
            epochs = (size_t)atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--backend=", 10) == 0) {
            const char *b = argv[i] + 10;
            if (strcmp(b, "cpu") == 0) backend_override = 1;
            else if (strcmp(b, "gpu") == 0) backend_override = 2;
        }
    }
    
    if (backend_override >= 0) {
        cfg.backend = backend_override;
    }

    /* Load model from checkpoint */
    printf("Loading model from checkpoint.ser...\n");
    KMamba *model = kmamba_load("checkpoint.ser", 1, &cfg.optim,
                                  cfg.optim.lr, cfg.optim.weight_decay);
    if (!model) {
        fprintf(stderr, "Error: failed to load checkpoint.ser\n");
        fprintf(stderr, "Run ./model %s first to create the model.\n", argv[1]);
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
    
    if (load_vision_data(NULL, &data, &labels, &num_samples, L, D, num_classes) != 0) {
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
        "checkpoint.ser",
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
