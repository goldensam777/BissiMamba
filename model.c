/* model.c — Déterminisme d'existence du modèle K-Mamba.
 * Usage: ./model <config.json>
 * Reads config, creates the model, prints confirmation.
 * The Trainer takes over if the user launches training.
 */
#include <stdio.h>
#include <stdlib.h>
#include "configs.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <config.json>\n", argv[0]);
        printf("Example: %s configs/cifar10.json\n", argv[0]);
        return 1;
    }

    KMambaFullConfig cfg;
    if (kmamba_configs_load_json(&cfg, argv[1]) != 0) {
        fprintf(stderr, "Error: failed to load %s\n", argv[1]);
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

    printf("\nSaving model to checkpoint.ser...\n");
    if (kmamba_save(model, "checkpoint.ser") != 0) {
        fprintf(stderr, "Error: failed to save model\n");
        kmamba_free(model);
        return 1;
    }
    printf("Model saved successfully.\n");

    kmamba_free(model);
    return 0;
}
