#include "configs.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void kmamba_configs_default(KMambaFullConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    kmamba_config_set_defaults(&cfg->model);
    kmamba_optim_config_set_defaults(&cfg->optim);
    cfg->backend = KMAMBA_BACKEND_AUTO;
    cfg->gpu_device = -1;
}

/* Minimal JSON parser (no external dependencies).
 * Expects a flat JSON object with keys matching the struct fields.
 * Supports: "dim", "state_size", "n_layers", "seq_len", "spatial_ndims",
 *   "spatial_dims" (array), "use_convnd", "convnd_K", "convnd_ndims",
 *   "mimo_rank", "default_lambda", "use_a_log_clamp", "a_log_min",
 *   "dt_min", "dt_max",
 *   "lr", "mu", "beta2", "eps", "clip_norm", "weight_decay",
 *   "backend", "gpu_device", "model_name".
 * All values have defaults, so missing keys are OK.
 */
int kmamba_configs_load_json(KMambaFullConfig *cfg, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "configs: cannot open %s\n", path); return -1; }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    /* Start with safe defaults */
    kmamba_configs_default(cfg);

    /* Parse simple key:value pairs (string, number, array).
     * Format: "key": value  or  "key": "string"  or  "key": [n1, n2, ...]
     * This is a simplified parser for our specific config format.
     */
    char *p = buf;
    while (*p) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == '\0' || *p == '}') break;

        /* Expect "key" */
        if (*p != '"') { p++; continue; }
        p++;
        char key[64] = {0};
        int k = 0;
        while (*p && *p != '"' && k < 63) key[k++] = *p++;
        if (*p == '"') p++;
        while (*p == ' ' || *p == ':') p++;

        /* Parse value */
        if (*p == '"') {
            /* String value */
            p++;
            char val[256] = {0};
            int v = 0;
            while (*p && *p != '"' && v < 255) val[v++] = *p++;
            if (*p == '"') p++;

            if (strcmp(key, "model_name") == 0)
                strncpy(cfg->model.model_name, val, 63);
            else if (strcmp(key, "data_path") == 0)
                ; /* ignored here, belongs to trainer */
            else if (strcmp(key, "checkpoint_path") == 0)
                ; /* ignored here */
        }
        else if (*p == '[') {
            /* Array value (spatial_dims) */
            p++;
            long arr[32] = {0};
            int ai = 0;
            while (*p && *p != ']' && ai < 32) {
                arr[ai++] = strtol(p, &p, 10);
                while (*p == ' ' || *p == ',') p++;
            }
            if (*p == ']') p++;
            if (strcmp(key, "spatial_dims") == 0)
                memcpy(cfg->model.spatial_dims, arr, ai * sizeof(long));
        }
        else {
            /* Number value */
            double val = strtod(p, &p);
            if (strcmp(key, "dim") == 0) cfg->model.dim = (size_t)val;
            else if (strcmp(key, "state_size") == 0) cfg->model.state_size = (size_t)val;
            else if (strcmp(key, "n_layers") == 0) cfg->model.n_layers = (size_t)val;
            else if (strcmp(key, "seq_len") == 0) cfg->model.seq_len = (size_t)val;
            else if (strcmp(key, "spatial_ndims") == 0) cfg->model.spatial_ndims = (long)val;
            else if (strcmp(key, "use_convnd") == 0) cfg->model.use_convnd = (int)val;
            else if (strcmp(key, "convnd_K") == 0) cfg->model.convnd_K = (long)val;
            else if (strcmp(key, "convnd_ndims") == 0) cfg->model.convnd_ndims = (long)val;
            else if (strcmp(key, "mimo_rank") == 0) cfg->model.mimo_rank = (size_t)val;
            else if (strcmp(key, "default_lambda") == 0) cfg->model.default_lambda = (float)val;
            else if (strcmp(key, "use_a_log_clamp") == 0) cfg->model.use_a_log_clamp = (int)val;
            else if (strcmp(key, "a_log_min") == 0) cfg->model.a_log_min = (float)val;
            else if (strcmp(key, "dt_min") == 0) cfg->model.dt_min = (float)val;
            else if (strcmp(key, "dt_max") == 0) cfg->model.dt_max = (float)val;
            else if (strcmp(key, "lr") == 0) cfg->optim.lr = (float)val;
            else if (strcmp(key, "mu") == 0) cfg->optim.mu = (float)val;
            else if (strcmp(key, "beta2") == 0) cfg->optim.beta2 = (float)val;
            else if (strcmp(key, "eps") == 0) cfg->optim.eps = (float)val;
            else if (strcmp(key, "clip_norm") == 0) cfg->optim.clip_norm = (float)val;
            else if (strcmp(key, "weight_decay") == 0) cfg->optim.weight_decay = (float)val;
            else if (strcmp(key, "backend") == 0) cfg->backend = (int)val;
            else if (strcmp(key, "gpu_device") == 0) cfg->gpu_device = (int)val;
        }
    }
    free(buf);
    return 0;
}

KMamba* kmamba_configs_create_model(const KMambaFullConfig *cfg) {
    if (!cfg) return NULL;

    /* Set backend preference */
    if (cfg->backend == KMAMBA_BACKEND_CPU)
        kmamba_backend_preference = KMAMBA_BACKEND_CPU;
    else if (cfg->backend == KMAMBA_BACKEND_GPU)
        kmamba_backend_preference = KMAMBA_BACKEND_GPU;
    else
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
    kmamba_backend_init();

    KMamba *m = kmamba_create(&cfg->model);
    if (!m) return NULL;

    kmamba_enable_training_with_optimizer(m, OPTIMIZER_ADAMW,
                                          &cfg->optim,
                                          cfg->optim.lr,
                                          cfg->optim.weight_decay);
    return m;
}
