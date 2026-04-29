/* model.c — Model initialization and serialization CLI.
 * Usage: ./model --config config.json --serialize ser
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "configs.h"

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
    const char *serialize_arg = NULL;

    for (int i = 1; i < argc; i++) {
        const char *val = NULL;

        int rc = parse_flag_value(argc, argv, &i, "--config", &val);
        if (rc == 1) {
            config_path = val;
            continue;
        }
        if (rc == -1) {
            fprintf(stderr, "Error: --config requires a value\n");
            return 1;
        }

        rc = parse_flag_value(argc, argv, &i, "--serialize", &val);
        if (rc == 1) {
            serialize_arg = val;
            continue;
        }
        if (rc == -1) {
            fprintf(stderr, "Error: --serialize requires a value\n");
            return 1;
        }

        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s --config <config.json> --serialize <ser|output.ser>\n", argv[0]);
            printf("Example: %s --config config.json --serialize ser\n", argv[0]);
            return 0;
        }

        if (argv[i][0] != '-') {
            config_path = argv[i];
            continue;
        }

        fprintf(stderr, "Error: unknown argument '%s'\n", argv[i]);
        return 1;
    }

    KMambaFullConfig cfg;
    if (kmamba_configs_load_json(&cfg, config_path) != 0) {
        fprintf(stderr, "Error: failed to load %s\n", config_path);
        return 1;
    }

    printf("Creating model '%s'...\n", cfg.model.model_name);
    KMamba *model = kmamba_configs_create_model(&cfg);
    if (!model) {
        fprintf(stderr, "Error: model creation failed\n");
        return 1;
    }

    printf("Model created successfully.\n");
    printf("  dim=%zu, state_size=%zu, n_layers=%zu, seq_len=%zu\n",
           cfg.model.dim, cfg.model.state_size, cfg.model.n_layers, cfg.model.seq_len);
    printf("  spatial_ndims=%ld, use_convnd=%d\n",
           cfg.model.spatial_ndims, cfg.model.use_convnd);
    printf("  optimizer=AdamW, lr=%.4g, weight_decay=%.4g\n",
           cfg.optim.lr, cfg.optim.weight_decay);
    printf("  backend=%s\n",
           cfg.backend == 0 ? "AUTO" : (cfg.backend == 1 ? "CPU" : "GPU"));

    char checkpoint_path[512] = {0};
    default_checkpoint_path(&cfg, checkpoint_path, sizeof(checkpoint_path));
    if (serialize_arg && strcmp(serialize_arg, "ser") != 0) {
        snprintf(checkpoint_path, sizeof(checkpoint_path), "%s", serialize_arg);
    }

    printf("\nSaving model to %s...\n", checkpoint_path);
    if (kmamba_save(model, checkpoint_path) != 0) {
        fprintf(stderr, "Error: failed to save model\n");
        kmamba_free(model);
        return 1;
    }
    printf("Model saved successfully.\n");

    kmamba_free(model);
    return 0;
}
