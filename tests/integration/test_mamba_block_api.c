/*
 * test_mamba_block_simple.c — Tests MambaBlock avec API CORRECTE
 *
 * Phase 2 : Tests d'intégration MambaBlock (forward/backward)
 * Utilise les vraies signatures de l'API k-mamba
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "kmamba.h"

#define EPSILON 1e-5f

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

static void fill_test_data(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * (max - min) + min;
    }
}

static void print_vector(const char *name, const float *v, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n && i < 8; i++) {
        printf("%.4f", v[i]);
        if (i < 7 && i < n-1) printf(", ");
    }
    if (n > 8) printf("...");
    printf("]\n");
}

/* ============================================================
 * Tests MambaBlock Forward
 * ============================================================ */

static int test_mamba_block_forward_simple() {
    printf("Testing MambaBlock forward simple case...\n");
    
    const size_t L = 4, D = 8, M = 16;
    
    /* Configuration simple */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .seq_len = L,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 1.0f,
        .use_convnd = 0,
        .convnd_K = 0,
        .convnd_ndims = 0
    };
    
    /* Données de test */
    float *x = (float*)malloc(L * D * sizeof(float));
    float *y = (float*)malloc(L * D * sizeof(float));
    
    /* Remplir avec des valeurs simples */
    for (size_t i = 0; i < L * D; i++) {
        x[i] = (float)(i % 10) / 10.0f;  // Valeurs entre 0 et 0.9
    }
    
    printf("Input x: ");
    print_vector("x", x, L * D);
    
    /* Créer et initialiser le MambaBlock */
    MambaBlock *block = mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock\n");
        free(x); free(y);
        return 0;
    }
    
    mamba_block_init(block);  // Pas de seed dans l'API
    
    /* Forward pass */
    mamba_block_forward(block, y, x, L);
    
    printf("Output y: ");
    print_vector("y", y, L * D);
    
    /* Vérifications basiques */
    int success = 1;
    
    /* Vérifier que l'output n'est pas tout zéro */
    int all_zero = 1;
    for (size_t i = 0; i < L * D; i++) {
        if (fabsf(y[i]) > 1e-6f) {
            all_zero = 0;
            break;
        }
    }
    
    if (all_zero) {
        printf("FAIL: Output is all zeros\n");
        success = 0;
    }
    
    /* Vérifier qu'il n'y a pas de NaN ou inf */
    for (size_t i = 0; i < L * D; i++) {
        if (!isfinite(y[i])) {
            printf("FAIL: Output contains NaN or inf at index %zu\n", i);
            success = 0;
            break;
        }
    }
    
    if (success) {
        printf("PASS: MambaBlock forward simple case\n");
    }
    
    /* Nettoyage */
    mamba_block_free(block);
    free(x); free(y);
    
    return success;
}

static int test_mamba_block_forward_deterministic() {
    printf("Testing MambaBlock forward deterministic...\n");
    
    const size_t L = 3, D = 4, M = 8;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .seq_len = L,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 1.0f,
        .use_convnd = 0,
        .convnd_K = 0,
        .convnd_ndims = 0
    };
    
    /* Données de test identiques */
    float x[12] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    float y1[12], y2[12];
    
    /* Créer deux blocks identiques */
    MambaBlock *block1 = mamba_block_create(&cfg);
    MambaBlock *block2 = mamba_block_create(&cfg);
    
    if (block1 == NULL || block2 == NULL) {
        printf("FAIL: Could not create MambaBlocks\n");
        return 0;
    }
    
    /* Initialiser */
    mamba_block_init(block1);
    mamba_block_init(block2);
    
    /* Forward passes */
    mamba_block_forward(block1, y1, x, L);
    mamba_block_forward(block2, y2, x, L);
    
    /* Comparer les outputs */
    int result = 1;
    for (size_t i = 0; i < L * D; i++) {
        float diff = fabsf(y1[i] - y2[i]);
        if (diff > EPSILON) {
            printf("Mismatch at [%zu]: y1=%.6f, y2=%.6f, diff=%.6f\n", 
                   i, y1[i], y2[i], diff);
            result = 0;
        }
    }
    
    if (result) {
        printf("PASS: MambaBlock forward deterministic\n");
    } else {
        printf("FAIL: MambaBlock forward non-deterministic\n");
    }
    
    /* Nettoyage */
    mamba_block_free(block1);
    mamba_block_free(block2);
    
    return result;
}

/* ============================================================
 * Tests MambaBlock Backward
 * ============================================================ */

static int test_mamba_block_backward_simple() {
    printf("Testing MambaBlock backward simple case...\n");
    
    const size_t L = 3, D = 4, M = 8;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .seq_len = L,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 1.0f,
        .use_convnd = 0,
        .convnd_K = 0,
        .convnd_ndims = 0
    };
    
    /* Optimiseur simple */
    MBOptimConfig opt = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    /* Données de test */
    float *x = (float*)malloc(L * D * sizeof(float));
    float *y = (float*)malloc(L * D * sizeof(float));
    float *dy = (float*)malloc(L * D * sizeof(float));
    float *dx = (float*)malloc(L * D * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    fill_test_data(x, L * D, -1.0f, 1.0f);
    fill_test_data(y, L * D, -1.0f, 1.0f);
    fill_test_data(dy, L * D, -0.1f, 0.1f);  // Petits gradients
    
    printf("Input x: ");
    print_vector("x", x, L * D);
    printf("Gradient dy: ");
    print_vector("dy", dy, L * D);
    
    /* Créer et initialiser le MambaBlock */
    MambaBlock *block = mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock\n");
        free(x); free(y); free(dy); free(dx);
        return 0;
    }
    
    mamba_block_init(block);
    
    /* Activer l'entraînement */
    mamba_attach_optimizer(block, OPTIMIZER_ADAM_CLIP, &opt);
    
    /* Forward d'abord pour avoir les activations */
    mamba_block_forward(block, y, x, L);
    
    /* Backward pass */
    mamba_backward(block, dy, x, dx, 0);  // batch_index = 0
    
    printf("Gradient dx: ");
    print_vector("dx", dx, L * D);
    
    /* Vérifications basiques */
    int success = 1;
    
    /* Vérifier que les gradients ne sont pas tout zéro */
    int all_zero = 1;
    for (size_t i = 0; i < L * D; i++) {
        if (fabsf(dx[i]) > 1e-6f) {
            all_zero = 0;
            break;
        }
    }
    
    if (all_zero) {
        printf("FAIL: Input gradient is all zeros\n");
        success = 0;
    }
    
    /* Vérifier qu'il n'y a pas de NaN ou inf */
    for (size_t i = 0; i < L * D; i++) {
        if (!isfinite(dx[i])) {
            printf("FAIL: Gradients contain NaN or inf\n");
            success = 0;
            break;
        }
    }
    
    if (success) {
        printf("PASS: MambaBlock backward simple case\n");
    }
    
    /* Nettoyage */
    mamba_block_free(block);
    free(x); free(y); free(dy); free(dx);
    
    return success;
}

/* ============================================================
 * Tests d'intégration complète
 * ============================================================ */

static int test_mamba_block_integration() {
    printf("Testing MambaBlock integration (forward + backward)...\n");
    
    const size_t L = 2, D = 4, M = 8;
    const int iterations = 5;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .seq_len = L,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 1.0f,
        .use_convnd = 0,
        .convnd_K = 0,
        .convnd_ndims = 0
    };
    
    /* Optimiseur */
    MBOptimConfig opt = {
        .lr = 1e-2f,  // Learning rate plus grand pour voir un effet
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-4f
    };
    
    /* Données de test */
    float x[8] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    float target[8] = {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    
    /* Créer le MambaBlock */
    MambaBlock *block = mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock\n");
        return 0;
    }
    
    mamba_block_init(block);
    mamba_attach_optimizer(block, OPTIMIZER_ADAM_CLIP, &opt);
    
    printf("Training loop:\n");
    for (int iter = 0; iter < iterations; iter++) {
        /* Forward */
        float y[8];
        mamba_block_forward(block, y, x, L);
        
        /* Calculer une loss simple (MSE) */
        float loss = 0.0f;
        for (int i = 0; i < L * D; i++) {
            float diff = y[i] - target[i];
            loss += diff * diff;
        }
        loss /= (float)(L * D);
        
        /* Créer gradient dy = y - target */
        float dy[8];
        for (int i = 0; i < L * D; i++) {
            dy[i] = y[i] - target[i];
        }
        
        /* Backward */
        float dx[8];
        mamba_backward(block, dy, x, dx, 0);  // batch_index = 0
        
        /* Optimizer step */
        mamba_optimizer_step(block, &opt);
        
        printf("Iter %d: loss = %.6f\n", iter, loss);
    }
    
    printf("PASS: MambaBlock integration completed\n");
    
    /* Nettoyage */
    mamba_block_free(block);
    
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_mamba_block() {
    printf("\n=== MambaBlock Performance Benchmarks ===\n");
    
    const size_t L = 128, D = 64, M = 16;
    const int iterations = 50;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .seq_len = L,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 1.0f,
        .use_convnd = 0,
        .convnd_K = 0,
        .convnd_ndims = 0
    };
    
    /* Données de test */
    float *x = (float*)malloc(L * D * sizeof(float));
    float *y = (float*)malloc(L * D * sizeof(float));
    
    fill_test_data(x, L * D, -1.0f, 1.0f);
    
    /* Créer le MambaBlock */
    MambaBlock *block = mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock for benchmark\n");
        free(x); free(y);
        return;
    }
    
    mamba_block_init(block);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        mamba_block_forward(block, y, x, L);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("MambaBlock Performance:\n");
    printf("  Size: L=%zu, D=%zu, M=%zu\n", L, D, M);
    printf("  Time: %.3f sec (%d iterations)\n", elapsed, iterations);
    printf("  Throughput: %.2f forward passes/sec\n", (double)iterations / elapsed);
    
    /* Nettoyage */
    mamba_block_free(block);
    free(x); free(y);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== MambaBlock Integration Test Suite ===\n");
    printf("Testing MambaBlock forward/backward integration\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests forward */
    total++; passed += test_mamba_block_forward_simple();
    total++; passed += test_mamba_block_forward_deterministic();
    
    /* Tests backward */
    total++; passed += test_mamba_block_backward_simple();
    
    /* Test d'intégration */
    total++; passed += test_mamba_block_integration();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        benchmark_mamba_block();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
