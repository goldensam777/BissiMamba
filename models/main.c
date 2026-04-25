#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../include/kmamba.h"
#include "config_presets.h"

void print_usage() {
    printf("Usage: k-mamba-train [--preset <name>] [--batch_size <N>] [--epochs <N>] [--lr <float>]\n");
    printf("                   [--backend <cpu|gpu|auto>] [--data <path>] [--output <path>] [--gc <none|every_n|all>] [--help]\n");
}

int main(int argc, char **argv) {
    const char *preset_name = "synthetic_2d";
    int batch_size = 64, epochs = 100;
    float lr_override = -1.0f;
    const char *backend = "auto", *data_path = NULL, *output_path = "checkpoint.ser", *gc_policy = "none";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--preset") == 0 && i+1 < argc) preset_name = argv[++i];
        else if (strcmp(argv[i], "--batch_size") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr_override = atof(argv[++i]);
        else if (strcmp(argv[i], "--backend") == 0 && i+1 < argc) backend = argv[++i];
        else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--gc") == 0 && i+1 < argc) gc_policy = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) { print_usage(); return 0; }
        else { printf("Unknown arg: %s\n", argv[i]); print_usage(); return 1; }
    }
    KMambaConfig cfg;
    MBOptimConfig optim;
    if (kmamba_config_preset_apply(preset_name, &cfg, &optim) != 0) {
        printf("Preset '%s' not found!\n", preset_name); return 1;
    }
    if (lr_override > 0) optim.lr = lr_override;
    printf("Model: %s\nDim: %zu\nLayers: %zu\nSeq_len: %zu\nSpatial_ndims: %d\nLR: %.4g\nBackend: %s\nGC: %s\n",
        cfg.model_name, cfg.dim, cfg.n_layers, cfg.seq_len, cfg.spatial_ndims, optim.lr, backend, gc_policy);
    // Model creation, trainer, data loading, training loop, checkpointing would go here
    printf("[Stub] Training loop not implemented in this scaffold.\n");
    return 0;
}
