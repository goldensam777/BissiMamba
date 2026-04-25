#include "config_presets.h"
#include <string.h>
#include <stdio.h>

#include "config_presets.h"
#include <string.h>
#include <stdio.h>

static void fill_synthetic_2d(KMambaPreset *p) {
    p->name = "synthetic_2d";
    kmamba_config_set_defaults(&p->cfg);
    p->cfg.vocab_size = 0;
    p->cfg.dim = 32;
    p->cfg.state_size = 8;
    p->cfg.n_layers = 2;
    p->cfg.seq_len = 64;
    p->cfg.spatial_ndims = 2;
    p->cfg.spatial_dims[0] = 8;
    p->cfg.spatial_dims[1] = 8;
    p->cfg.use_convnd = 0;
    p->cfg.max_ndims = 8;
    p->cfg.max_state = 64;
    p->cfg.default_lambda = 0.5f;
    strcpy(p->cfg.model_name, "k-mamba-synthetic-2d");
    kmamba_optim_config_set_defaults(&p->optim);
    p->optim.lr = 1e-3f;
    p->optim.weight_decay = 0.01f;
}

static void fill_cifar10(KMambaPreset *p) {
    p->name = "cifar10";
    kmamba_config_set_defaults(&p->cfg);
    p->cfg.vocab_size = 0;
    p->cfg.dim = 128;
    p->cfg.state_size = 16;
    p->cfg.n_layers = 4;
    p->cfg.seq_len = 64;
    p->cfg.spatial_ndims = 2;
    p->cfg.spatial_dims[0] = 8;
    p->cfg.spatial_dims[1] = 8;
    p->cfg.use_convnd = 1;
    p->cfg.convnd_K = 3;
    p->cfg.convnd_ndims = 2;
    strcpy(p->cfg.model_name, "k-mamba-cifar10");
    kmamba_optim_config_set_defaults(&p->optim);
    p->optim.lr = 3e-4f;
    p->optim.weight_decay = 0.05f;
}

static void fill_moving_mnist(KMambaPreset *p) {
    p->name = "moving_mnist";
    kmamba_config_set_defaults(&p->cfg);
    p->cfg.vocab_size = 0;
    p->cfg.dim = 64;
    p->cfg.state_size = 32;
    p->cfg.n_layers = 6;
    p->cfg.seq_len = 1280;
    p->cfg.spatial_ndims = 3;
    p->cfg.spatial_dims[0] = 20;
    p->cfg.spatial_dims[1] = 8;
    p->cfg.spatial_dims[2] = 8;
    p->cfg.use_convnd = 1;
    p->cfg.convnd_K = 3;
    p->cfg.convnd_ndims = 3;
    strcpy(p->cfg.model_name, "k-mamba-moving-mnist");
    kmamba_optim_config_set_defaults(&p->optim);
    p->optim.lr = 1e-3f;
    p->optim.weight_decay = 0.01f;
}

static KMambaPreset kmamba_presets[3];
static int _presets_initialized = 0;
static int kmamba_num_presets = 3;

static void _init_presets();

const KMambaPreset *kmamba_presets_get_all(int *num) {
    _init_presets();
    if (num) *num = kmamba_num_presets;
    return kmamba_presets;
}

static void _init_presets() {
    if (_presets_initialized) return;
    fill_synthetic_2d(&kmamba_presets[0]);
    fill_cifar10(&kmamba_presets[1]);
    fill_moving_mnist(&kmamba_presets[2]);
    _presets_initialized = 1;
}

const KMambaPreset *kmamba_config_preset_find(const char *name) {
    _init_presets();
    for (int i = 0; i < kmamba_num_presets; ++i) {
        if (strcmp(kmamba_presets[i].name, name) == 0) return &kmamba_presets[i];
    }
    return NULL;
}

int kmamba_config_preset_apply(const char *name, KMambaConfig *cfg, MBOptimConfig *optim) {
    const KMambaPreset *p = kmamba_config_preset_find(name);
    if (!p) return -1;
    *cfg = p->cfg;
    *optim = p->optim;
    return 0;
}
