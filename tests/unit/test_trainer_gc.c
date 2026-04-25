#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kmamba.h"
#include "trainer.h"

#define L 32
#define D 16
#define N 16
#define BATCH 1

int main() {
    printf("Starting Trainer GC validation test (L=%d, D=%d, N=%d)...\n", L, D, N);

    /* 1. Configuration du modèle */
    KMambaConfig cfg;
    kmamba_config_set_defaults(&cfg);
    cfg.seq_len = L;
    cfg.dim = D;
    cfg.state_size = N;
    cfg.n_layers = 4;
    cfg.vocab_size = 256;

    KMamba *m = kmamba_create(&cfg);
    kmamba_init(m, 42);

    MBOptimConfig opt_cfg;
    kmamba_optim_config_set_defaults(&opt_cfg);
    kmamba_enable_training(m, &opt_cfg, 1e-3f, 1e-4f);

    /* 2. Données de test */
    float *input = (float*)malloc(L * D * sizeof(float));
    float *target_dy = (float*)malloc(L * D * sizeof(float));
    float *out_no_gc = (float*)malloc(L * D * sizeof(float));
    float *out_gc = (float*)malloc(L * D * sizeof(float));

    for (int i = 0; i < L * D; i++) {
        input[i] = (float)rand() / RAND_MAX;
        target_dy[i] = (float)rand() / RAND_MAX;
    }

    /* 3. Run sans GC (Standard) */
    TrainerGCConfig gc_none = { .policy = TRAINER_GC_NONE };
    Trainer *tr_none = trainer_create(m, &gc_none);
    
    trainer_forward(tr_none, input, out_no_gc, BATCH);
    trainer_backward(tr_none, target_dy, input, BATCH);
    
    /* Sauvegarde des gradients pour comparaison */
    float *grad_ref = (float*)malloc(N * sizeof(float));
    memcpy(grad_ref, m->layers[0]->opt_state->g_A_log, N * sizeof(float));
    
    trainer_free(tr_none);

    /* 4. Reset gradients et Run avec GC EVERY_2 */
    for (size_t i = 0; i < cfg.n_layers; i++) mamba_zero_grads(m->layers[i]);
    
    TrainerGCConfig gc_every = { .policy = TRAINER_GC_EVERY_N, .checkpoint_every_n = 2 };
    Trainer *tr_gc = trainer_create(m, &gc_every);
    
    trainer_forward(tr_gc, input, out_gc, BATCH);
    trainer_backward(tr_gc, target_dy, input, BATCH);

    /* 5. Validation */
    float out_err = 0.0f;
    for (int i = 0; i < L * D; i++) out_err += fabsf(out_no_gc[i] - out_gc[i]);
    
    float grad_err = 0.0f;
    for (int i = 0; i < N; i++) grad_err += fabsf(grad_ref[i] - m->layers[0]->opt_state->g_A_log[i]);

    printf("Forward Output error: %e\n", out_err);
    printf("Backward Gradient error (layer 0): %e\n", grad_err);

    int success = (out_err < 1e-5f && grad_err < 1e-5f);
    if (success) printf("Trainer GC Check: PASS\n");
    else printf("Trainer GC Check: FAIL\n");

    /* Cleanup */
    trainer_free(tr_gc);
    kmamba_free(m);
    free(input); free(target_dy); free(out_no_gc); free(out_gc); free(grad_ref);

    return success ? 0 : 1;
}
