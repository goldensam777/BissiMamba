#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../include/kmamba.h"
#include "../include/kmamba_cuda_utils.h"
#include "../libs/train_set/include/trainer.h"
#include "config_presets.h"

/* External dataset loader stubs */
extern int load_cifar10(const char *path, float **data, uint32_t **labels,
                        size_t *num_samples, size_t L, size_t D);
extern int load_moving_mnist(const char *path, float **data, uint32_t **labels,
                             size_t *num_samples, size_t L, size_t D);
extern int generate_synthetic_2d(float **data, uint32_t **labels,
                                   size_t *num_samples, size_t L, size_t D);

static double elapsed_ms(struct timespec *t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0->tv_sec) * 1000.0
         + (double)(t1.tv_nsec - t0->tv_nsec) / 1e6;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  --preset <name>       Preset config (synthetic_2d|cifar10|moving_mnist). Default: synthetic_2d\n");
    printf("  --batch_size <N>      Batch size. Default: 64\n");
    printf("  --epochs <N>          Number of epochs. Default: 100\n");
    printf("  --lr <float>          Override learning rate\n");
    printf("  --backend <cpu|gpu|auto>  Backend selection. Default: auto\n");
    printf("  --data <path>         Dataset path\n");
    printf("  --output <path>       Checkpoint output path. Default: checkpoint.ser\n");
    printf("  --log_dir <path>      Directory for CSV logs. Default: logs/\n");
    printf("  --resume <path>       Resume from checkpoint\n");
    printf("  --gc <none|every_n|all>  Gradient checkpointing policy. Default: none\n");
    printf("  --help                Print this help\n");
}

static void print_config_summary(const KMambaConfig *cfg, const MBOptimConfig *optim,
                                 const char *backend_str, const char *gc_policy,
                                 int batch_size, int epochs) {
    const char *opt_type_str = "AdamW";
    int width = 50;
    char line[width + 1];
    memset(line, '-', width); line[width] = '\0';

    printf("\n┌%s┐\n", line);
    printf("│ %-*s │\n", width - 2, "K-MAMBA EXPERIMENT DETAILS");
    printf("├%s┤\n", line);
    printf("│ %-15s : %-30s │\n", "Model", cfg->model_name);
    printf("│ %-15s : %-30zu │\n", "Dim", cfg->dim);
    printf("│ %-15s : %-30zu │\n", "Layers", cfg->n_layers);
    printf("│ %-15s : %-30zu │\n", "Seq Len", cfg->seq_len);
    printf("│ %-15s : %-30ld │\n", "Spatial Dims", cfg->spatial_ndims);
    printf("├%s┤\n", line);
    printf("│ %-15s : %-30s │\n", "Optimizer", opt_type_str);
    printf("│ %-15s : %-30.4g │\n", "LR", optim->lr);
    printf("│ %-15s : %-30.4g │\n", "Weight Decay", optim->weight_decay);
    printf("│ %-15s : %-30s │\n", "Backend", backend_str);
    printf("│ %-15s : %-30s │\n", "GC Policy", gc_policy);
    printf("├%s┤\n", line);
    printf("│ %-15s : %-30d │\n", "Batch Size", batch_size);
    printf("│ %-15s : %-30d │\n", "Epochs", epochs);
    printf("└%s┘\n\n", line);
}

static TrainerGCPolicy parse_gc_policy(const char *policy) {
    if (strcmp(policy, "every_n") == 0) return TRAINER_GC_EVERY_N;
    if (strcmp(policy, "all") == 0) return TRAINER_GC_ALL;
    return TRAINER_GC_NONE;
}

static void init_random_data(float *data, uint32_t *labels, size_t n_samples, size_t L, size_t D, int num_classes) {
    /* Initialize with random values for stub data */
    for (size_t i = 0; i < n_samples * L * D; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < n_samples; i++) {
        labels[i] = (uint32_t)(rand() % num_classes);
    }
}

int main(int argc, char **argv) {
    const char *preset_name = "synthetic_2d";
    int batch_size = 64;
    int epochs = 100;
    float lr_override = -1.0f;
    const char *backend_str = "auto";
    const char *data_path = NULL;
    const char *output_path = "checkpoint.ser";
    const char *gc_policy_str = "none";
    const char *log_dir = "logs";
    const char *resume_path = NULL;

    /* Parse command-line arguments */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            preset_name = argv[++i];
        } else if (strcmp(argv[i], "--batch_size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr_override = atof(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_str = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--log_dir") == 0 && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            resume_path = argv[++i];
        } else if (strcmp(argv[i], "--gc") == 0 && i + 1 < argc) {
            gc_policy_str = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Load preset configuration */
    KMambaConfig cfg;
    MBOptimConfig optim;
    if (kmamba_config_preset_apply(preset_name, &cfg, &optim) != 0) {
        printf("Error: Preset '%s' not found!\n", preset_name);
        return 1;
    }

    /* Override learning rate if specified */
    if (lr_override > 0.0f) {
        optim.lr = lr_override;
    }

    /* Set backend preference */
    if (strcmp(backend_str, "cpu") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_CPU;
    } else if (strcmp(backend_str, "gpu") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_GPU;
    } else {
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
    }
    kmamba_backend_init();

    /* Print configuration summary */
    print_config_summary(&cfg, &optim, backend_str, gc_policy_str, batch_size, epochs);

    /* Create and initialize model */
    printf("Creating model...\n");
    KMamba *model = kmamba_create(&cfg);
    if (!model) {
        printf("Error: Failed to create model\n");
        return 1;
    }

    printf("Initializing model weights...\n");
    if (kmamba_init(model, (uint32_t)time(NULL)) != 0) {
        printf("Error: Failed to initialize model\n");
        kmamba_free(model);
        return 1;
    }

    /* Setup gradient checkpointing configuration */
    TrainerGCConfig gc_config;
    gc_config.policy = parse_gc_policy(gc_policy_str);
    gc_config.checkpoint_every_n = 2; /* Default: checkpoint every 2 layers */

    /* Setup logging configuration */
    TrainerLogConfig log_cfg;
    log_cfg.log_dir = (char*)log_dir;
    log_cfg.run_name = (char*)preset_name;
    log_cfg.log_step_every = 10;
    log_cfg.log_epoch_every = 1;

    /* Create trainer (with resume support if requested) */
    Trainer *trainer;
    int start_epoch = 1;

    if (resume_path) {
        printf("Resuming from checkpoint: %s\n", resume_path);
        /* First create trainer with existing model, then load checkpoint */
        trainer = trainer_create_with_logging(model, &gc_config, &log_cfg);
        if (!trainer) {
            printf("Error: Failed to create trainer\n");
            kmamba_free(model);
            return 1;
        }
        /* Load checkpoint (model weights + optimizer state) */
        if (trainer_load_checkpoint(trainer, resume_path) == 0) {
            TrainerResumeMetrics metrics;
            trainer_get_metrics(trainer, &metrics);
            start_epoch = metrics.epoch + 1;
            printf("Resumed from epoch %d, step=%zu\n", start_epoch - 1, metrics.global_step);
            /* Update model pointer - trainer_load_checkpoint replaced tr->model */
            model = trainer->model;
        } else {
            printf("Warning: Failed to load checkpoint, starting fresh\n");
            /* Continue with newly created trainer and model */
            /* Need to re-enable training since model was recreated */
            kmamba_enable_training_with_optimizer(model, OPTIMIZER_ADAMW, &optim,
                                                   optim.lr, optim.weight_decay);
        }
    } else {
        printf("Creating trainer with GC policy: %s, logging to: %s/\n", gc_policy_str, log_dir);
        trainer = trainer_create_with_logging(model, &gc_config, &log_cfg);
    }

    if (!trainer) {
        printf("Error: Failed to create trainer\n");
        kmamba_free(model);
        return 1;
    }

    /* Enable training */
    printf("Enabling training...\n");
    if (kmamba_enable_training_with_optimizer(model, OPTIMIZER_ADAMW, &optim,
                                               optim.lr, optim.weight_decay) != 0) {
        printf("Error: Failed to enable training\n");
        trainer_free(trainer);
        kmamba_free(model);
        return 1;
    }

    /* Load dataset based on preset */
    float *data = NULL;
    uint32_t *labels = NULL;
    size_t num_samples = 0;
    size_t L = cfg.seq_len;
    size_t D = cfg.dim;
    int num_classes = 2; /* Default binary classification */

    printf("Loading dataset...\n");
    if (strstr(preset_name, "cifar10") != NULL) {
        num_classes = 10;
        if (load_cifar10(data_path, &data, &labels, &num_samples, L, D) != 0) {
            printf("Error: Failed to load CIFAR-10 data\n");
            trainer_free(trainer);
            kmamba_free(model);
            return 1;
        }
    } else if (strstr(preset_name, "moving_mnist") != NULL) {
        num_classes = 10;
        if (load_moving_mnist(data_path, &data, &labels, &num_samples, L, D) != 0) {
            printf("Error: Failed to load Moving MNIST data\n");
            trainer_free(trainer);
            kmamba_free(model);
            return 1;
        }
    } else {
        /* Default: synthetic_2d */
        num_classes = 2;
        if (generate_synthetic_2d(&data, &labels, &num_samples, L, D) != 0) {
            printf("Error: Failed to generate synthetic data\n");
            trainer_free(trainer);
            kmamba_free(model);
            return 1;
        }
    }

    /* Initialize stub data with random values */
    if (data && labels) {
        init_random_data(data, labels, num_samples, L, D, num_classes);
    }

    printf("Dataset loaded: %zu samples, %d classes\n\n", num_samples, num_classes);

    /* Training loop */
    printf("Starting training...\n");
    size_t num_batches = (num_samples + batch_size - 1) / batch_size;

    /* Print table header once */
    printf("┌───────┬─────────┬─────────┬───────────┬───────────┬───────────┐\n");
    printf("│ Epoch │  Loss   │ Acc (%%) │ Samples/s │ Time (ms) │    LR     │\n");
    printf("├───────┼─────────┼─────────┼───────────┼───────────┼───────────┤\n");

    for (int epoch = start_epoch; epoch <= epochs; ++epoch) {
        struct timespec epoch_t0;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t0);
        double epoch_loss_sum = 0.0;
        double epoch_acc_sum = 0.0;
        int epoch_batches = 0;

        for (size_t batch = 0; batch < num_batches; ++batch) {
            /* Calculate batch boundaries */
            size_t start_idx = batch * batch_size;
            size_t end_idx = start_idx + batch_size;
            if (end_idx > num_samples) end_idx = num_samples;
            size_t current_batch_size = end_idx - start_idx;

            TrainerMetrics metrics;
            struct timespec step_t0;
            clock_gettime(CLOCK_MONOTONIC, &step_t0);

            if (cfg.vocab_size == 0) {
                /* Vision model: use unified API with vision batch */
                const float *batch_data = data + start_idx * L * D;
                const uint32_t *batch_labels = labels + start_idx;
                TrainerBatch batch = TRAINER_BATCH_VISION_INIT(
                    batch_data, batch_labels, current_batch_size, L, D, num_classes);
                metrics = trainer_train_batch_ex(trainer, &batch);
            } else {
                /* Language model: use unified API with token batch */
                uint32_t *batch_tokens = (uint32_t *)malloc(current_batch_size * (L + 1) * sizeof(uint32_t));
                if (!batch_tokens) {
                    printf("Error: Failed to allocate batch tokens\n");
                    break;
                }
                /* Fill with tokens from data (simplified - would need proper tokenization) */
                for (size_t i = 0; i < current_batch_size * (L + 1); i++) {
                    batch_tokens[i] = (uint32_t)(i % cfg.vocab_size);
                }
                TrainerBatch batch = TRAINER_BATCH_TOKENS_INIT(batch_tokens, current_batch_size);
                metrics = trainer_train_batch_ex(trainer, &batch);
                free(batch_tokens);
            }

            /* Log step to CSV */
            double step_ms = elapsed_ms(&step_t0);
            trainer_log_step(trainer, epoch, batch + 1, metrics.loss, metrics.accuracy, step_ms);

            epoch_loss_sum += metrics.loss;
            epoch_acc_sum += metrics.accuracy;
            epoch_batches++;

            /* Progress indicator (no per-batch output to keep table clean) */
            if (batch % 5 == 0 || batch == num_batches - 1) {
                printf("│ %-5d │ %7.4f │ %7.2f │ %9.1f │ %9.1f │ %9.2e │\r",
                       epoch, (float)(epoch_loss_sum / epoch_batches), (float)(epoch_acc_sum / epoch_batches * 100.0),
                       0.0, 0.0, optim.lr);
                fflush(stdout);
            }
        }

        double avg_loss = epoch_batches > 0 ? epoch_loss_sum / epoch_batches : 0.0;
        double avg_acc = epoch_batches > 0 ? epoch_acc_sum / epoch_batches : 0.0;

        /* Check for NaN loss */
        if (!isfinite(avg_loss)) {
            printf("\n[WARNING] NaN/Inf loss detected at epoch %d. Stopping training.\n", epoch);
            break;
        }

        double epoch_ms = elapsed_ms(&epoch_t0);
        double epoch_sec = epoch_ms / 1000.0;
        double samples_per_sec = epoch_sec > 0 ? (epoch_batches * batch_size) / epoch_sec : 0.0;

        /* Print final epoch summary */
        printf("│ %-5d │ %7.4f │ %7.2f │ %9.1f │ %9.1f │ %9.2e │\n",
               epoch, avg_loss, avg_acc * 100.0, samples_per_sec, epoch_ms, optim.lr);

        /* Update trainer resume state */
        trainer->resume.epoch = epoch;
        trainer->resume.global_step += epoch_batches;
        if (avg_loss < trainer->resume.best_val_loss) {
            trainer->resume.best_val_loss = avg_loss;
        }

        /* Log epoch to CSV */
        trainer_log_epoch(trainer, epoch, epoch_batches, avg_loss, avg_acc, epoch_ms);

        /* Save checkpoint after each epoch (for resume capability) */
        if (trainer_save_checkpoint(trainer, output_path) != 0) {
            printf("Warning: Failed to save checkpoint at epoch %d\n", epoch);
        }
    }
    printf("└───────┴─────────┴─────────┴───────────┴───────────┴───────────┘\n");

    printf("Training complete.\n");
    printf("Final checkpoint saved to: %s\n", output_path);
    printf("To resume: %s --resume %s\n", argv[0], output_path);

    /* Cleanup */
    free(data);
    free(labels);
    trainer_free(trainer);
    kmamba_free(model);

    printf("Done.\n");
    return 0;
}
