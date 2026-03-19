/*
 * test_kmamba_e2e.c — Test end-to-end du modèle KMamba
 *
 * Crée un micro-modèle (dim=16, state=8, n_layers=1, seq_len=8),
 * vérifie le forward (loss finie, logits non-NaN), et vérifie que
 * la loss décroît sur 20 steps d'entraînement.
 *
 * Ce test ne dépend que de l'API publique kmamba.h — aucun accès
 * aux internals.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "kmamba.h"

#define PASS_TAG "  [PASS]"
#define FAIL_TAG "  [FAIL]"

/* ── Helpers ───────────────────────────────────────────────────── */
static int all_finite_f(const float *v, long n) {
    for (long i = 0; i < n; i++)
        if (!isfinite(v[i])) return 0;
    return 1;
}

/* ── Config micro-modèle ───────────────────────────────────────── */
static KMambaConfig make_micro_cfg(void) {
    KMambaConfig cfg = {
        .vocab_size  = 256,
        .dim         = 16,
        .state_size  = 8,
        .seq_len     = 8,
        .n_layers    = 1,
        .dt_scale    = 1.0f,
        .dt_min      = 0.001f,
        .dt_max      = 0.1f,
        .use_convnd  = 0,
        .convnd_K    = 1,
        .convnd_ndims= 1,
    };
    return cfg;
}

/* ── Test 1 : forward produit des logits finis ─────────────────── */
static int test_forward_finite(void) {
    printf("\n--- forward : logits finis, loss finie ---\n");

    KMambaConfig cfg = make_micro_cfg();
    KMamba *m = kmamba_create(&cfg);
    if (!m) { printf("%s kmamba_create a échoué\n", FAIL_TAG); return 0; }

    kmamba_init(m, 42);

    uint8_t tokens[8] = {72, 101, 108, 108, 111, 10, 0, 0}; /* "Hello\n" */

    long n_logits = (long)(cfg.seq_len * cfg.vocab_size);
    float *logits = (float *)malloc((size_t)n_logits * sizeof(float));
    if (!logits) { kmamba_free(m); return 0; }

    int ret = kmamba_forward(m, tokens, logits);

    int ok = 1;
    if (ret != 0) {
        printf("%s kmamba_forward a retourné %d\n", FAIL_TAG, ret);
        ok = 0;
    } else if (!all_finite_f(logits, n_logits)) {
        printf("%s logits contiennent NaN/Inf\n", FAIL_TAG);
        ok = 0;
    } else {
        printf("%s logits[%ld] tous finis\n", PASS_TAG, n_logits);
    }

    free(logits);
    kmamba_free(m);
    return ok;
}

/* ── Test 2 : train_step produit une loss finie positive ───────── */
static int test_train_step_valid(void) {
    printf("\n--- train_step : loss finie et positive ---\n");

    KMambaConfig cfg = make_micro_cfg();
    KMamba *m = kmamba_create(&cfg);
    if (!m) { printf("%s kmamba_create a échoué\n", FAIL_TAG); return 0; }

    kmamba_init(m, 42);

    MBOptimConfig opt = {
        .lr          = 1e-3f,
        .mu          = 0.9f,
        .beta2       = 0.999f,
        .eps         = 1e-8f,
        .clip_norm   = 1.0f,
        .weight_decay= 1e-5f,
    };
    kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);

    /* tokens_plus1 : seq_len+1 = 9 bytes */
    uint8_t tokens[9] = {72, 101, 108, 108, 111, 10, 0, 0, 72};

    float loss = kmamba_train_step(m, tokens);

    int ok = 1;
    if (!isfinite(loss)) {
        printf("%s loss = %f  (NaN ou Inf)\n", FAIL_TAG, loss);
        ok = 0;
    } else if (loss <= 0.0f) {
        printf("%s loss = %f  (négative ou nulle)\n", FAIL_TAG, loss);
        ok = 0;
    } else {
        printf("%s loss = %.4f  (finie, positive)\n", PASS_TAG, loss);
    }

    kmamba_free(m);
    return ok;
}

/* ── Test 3 : loss décroît sur 20 steps (même séquence) ───────── */
static int test_loss_decreases(void) {
    printf("\n--- loss décroît sur 20 steps ---\n");

    KMambaConfig cfg = make_micro_cfg();
    KMamba *m = kmamba_create(&cfg);
    if (!m) { printf("%s kmamba_create a échoué\n", FAIL_TAG); return 0; }

    kmamba_init(m, 1337);

    MBOptimConfig opt = {
        .lr          = 1e-2f,   /* lr un peu plus grand pour voir la descente */
        .mu          = 0.9f,
        .beta2       = 0.999f,
        .eps         = 1e-8f,
        .clip_norm   = 1.0f,
        .weight_decay= 1e-5f,
    };
    kmamba_enable_training(m, &opt, 1e-2f, 1e-5f);

    /* Séquence fixe répétée : "Hello\nHello" */
    uint8_t tokens[9] = {72, 101, 108, 108, 111, 10, 72, 101, 108};

    float loss_first = 0.0f, loss_last = 0.0f;
    int n_steps = 20;
    int all_finite = 1;

    for (int i = 0; i < n_steps; i++) {
        float l = kmamba_train_step(m, tokens);
        if (!isfinite(l)) { all_finite = 0; break; }
        if (i == 0)          loss_first = l;
        if (i == n_steps - 1) loss_last  = l;
    }

    int ok = 1;
    if (!all_finite) {
        printf("%s loss NaN/Inf détectée avant step %d\n", FAIL_TAG, n_steps);
        ok = 0;
    } else {
        printf("    loss_0=%.4f  loss_%d=%.4f\n", loss_first, n_steps, loss_last);
        if (loss_last < loss_first) {
            printf("%s loss a décru (%.4f → %.4f)\n",
                   PASS_TAG, loss_first, loss_last);
        } else {
            printf("%s loss n'a pas décru (%.4f → %.4f)\n",
                   FAIL_TAG, loss_first, loss_last);
            ok = 0;
        }
    }

    kmamba_free(m);
    return ok;
}

/* ── Test 4 : train_batch cohérent avec train_step ─────────────── */
static int test_train_batch_finite(void) {
    printf("\n--- train_batch : loss finie sur batch_size=4 ---\n");

    KMambaConfig cfg = make_micro_cfg();
    KMamba *m = kmamba_create(&cfg);
    if (!m) { printf("%s kmamba_create a échoué\n", FAIL_TAG); return 0; }

    kmamba_init(m, 99);

    MBOptimConfig opt = {
        .lr          = 1e-3f,
        .mu          = 0.9f,
        .beta2       = 0.999f,
        .eps         = 1e-8f,
        .clip_norm   = 1.0f,
        .weight_decay= 1e-5f,
    };
    kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);

    /* batch de 4 séquences, chacune de seq_len+1=9 bytes */
    size_t batch_size = 4;
    size_t seq_len1   = cfg.seq_len + 1;
    uint8_t batch[4 * 9];
    for (size_t i = 0; i < batch_size * seq_len1; i++)
        batch[i] = (uint8_t)(i % 256);

    float loss = kmamba_train_batch(m, batch, batch_size);

    int ok = 1;
    if (!isfinite(loss) || loss <= 0.0f) {
        printf("%s train_batch loss = %f\n", FAIL_TAG, loss);
        ok = 0;
    } else {
        printf("%s train_batch loss = %.4f\n", PASS_TAG, loss);
    }

    kmamba_free(m);
    return ok;
}

/* ── Main ──────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Tests KMamba end-to-end ===\n");

    int passed = 0, total = 0;

    total++; passed += test_forward_finite();
    total++; passed += test_train_step_valid();
    total++; passed += test_loss_decreases();
    total++; passed += test_train_batch_finite();

    printf("\n=== %d/%d tests passés ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
