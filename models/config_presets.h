#ifndef CONFIG_PRESETS_H
#define CONFIG_PRESETS_H

#include "../include/kmamba.h"

typedef struct {
    const char *name;
    KMambaConfig cfg;
    MBOptimConfig optim;
} KMambaPreset;

extern const int kmamba_num_presets;

const KMambaPreset *kmamba_config_preset_find(const char *name);
int kmamba_config_preset_apply(const char *name, KMambaConfig *cfg, MBOptimConfig *optim);

#endif /* CONFIG_PRESETS_H */
