#include "config_presets.h"
#include <string.h>

static void fill_synthetic_2d(KMambaPreset *p) {
    p->name = "synthetic_2d";
    kmamba_config_set_defaults(&p->cfg);
    p->cfg.vocab_size = 0;
    p->cfg.dim = 1;           /* 1D input - each grid position is 1 float */
    p->cfg.state_size = 64;   /* SSM state size for pattern storage */
    p->cfg.n_layers = 4;      /* Multiple layers for hierarchical feature extraction */
    p->cfg.seq_len = 64;      /* 8x8 = 64 (flattened grid as 1D sequence for 2D scan) */
    p->cfg.spatial_ndims = 2; /* 2D wavefront scan over 8x8 grid */
    p->cfg.spatial_dims[0] = 8;
    p->cfg.spatial_dims[1] = 8;
    p->cfg.use_convnd = 0;
    p->cfg.max_ndims = 8;
    p->cfg.max_state = 128;
    p->cfg.default_lambda = 0.5f;
    p->cfg.expand_factor = 4.0f;  /* Expand dim 1 -> 4 for inner computation */
    p->cfg.dt_min = 0.0001f;  /* Smaller dt range for stability */
    p->cfg.dt_max = 0.01f;
    p->cfg.use_a_log_clamp = 1;  /* Clamp A_log to prevent explosion */
    p->cfg.a_log_min = -5.0f;
    strcpy(p->cfg.model_name, "k-mamba-synthetic-2d");
    kmamba_optim_config_set_defaults(&p->optim);
    p->optim.lr = 3e-4f;      /* Increased for better convergence */
    p->optim.weight_decay = 0.001f; /* Decreased to prevent over-regularization */
    p->optim.clip_norm = 1.0f; /* Gradient clipping */
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

const int kmamba_num_presets = 3;

static KMambaPreset _presets[3];
static int _presets_initialized = 0;

static void _init_presets(void) {
    if (_presets_initialized) return;
    fill_synthetic_2d(&_presets[0]);
    fill_cifar10(&_presets[1]);
    fill_moving_mnist(&_presets[2]);
    _presets_initialized = 1;
}

const KMambaPreset *kmamba_config_preset_find(const char *name) {
    _init_presets();
    for (int i = 0; i < kmamba_num_presets; ++i) {
        if (strcmp(_presets[i].name, name) == 0) return &_presets[i];
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
