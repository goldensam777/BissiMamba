/*
 * test_mamba_block.c — Tests d'intégration MambaBlock
 *
 * Phase 2 : Tests d'intégration MambaBlock (forward/backward)
 * Objectif : Valider le pipeline complet MambaBlock
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* Définitions manuelles des fonctions k-mamba */
void mamba_block_create(MBConfig *cfg, MambaBlock **block);
void mamba_block_free(MambaBlock *block);
int mamba_block_init(MambaBlock *block, unsigned long seed);
void mamba_block_forward(MambaBlock *block, const float *x, float *y, const float *h0, float *h);
void mamba_block_backward(MambaBlock *block, const float *x, const float *h0, const float *dy, float *dx, float *dh0);
void mamba_block_zero_grads(MambaBlock *block);
void mamba_optimizer_step(MambaBlock *block, MBOptimConfig *opt);

#define EPSILON 1e-5f

/* ============================================================
 * Structures de données pour les tests
 * ============================================================ */

typedef struct {
    float *x;       /* Input [L * D] */
    float *y;       /* Output [L * D] */
    float *h0;      /* Initial hidden [D * M] */
    float *h;       /* Hidden states [L * D * M] */
    float *dy;      /* Gradient output [L * D] */
    float *dx;      /* Gradient input [L * D] */
    float *dh0;     /* Gradient initial hidden [D * M] */
    long   L;       /* Sequence length */
    long   D;       /* Feature dimension */
    long   M;       /* State dimension */
} MambaBlockTestData;

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

static int compare_vectors(const float *ref, const float *test, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > eps) {
            printf("Mismatch at [%zu]: ref=%.6f, test=%.6f, diff=%.6f\n", 
                   i, ref[i], test[i], diff);
            return 0;
        }
    }
    return 1;
}

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
    
    const long L = 4, D = 8, M = 16;
    
    /* Configuration simple */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    /* Données de test */
    MambaBlockTestData data;
    data.L = L; data.D = D; data.M = M;
    
    data.x = (float*)malloc(L * D * sizeof(float));
    data.y = (float*)malloc(L * D * sizeof(float));
    data.h0 = (float*)malloc(D * M * sizeof(float));
    data.h = (float*)malloc(L * D * M * sizeof(float));
    
    /* Remplir avec des valeurs simples */
    for (long i = 0; i < L * D; i++) {
        data.x[i] = (float)(i % 10) / 10.0f;  // Valeurs entre 0 et 0.9
    }
    memset(data.h0, 0, D * M * sizeof(float));  // États initiaux à zéro
    
    printf("Input x: ");
    print_vector("x", data.x, L * D);
    
    /* Créer et initialiser le MambaBlock */
    MambaBlock *block;
    mamba_block_create(&cfg, &block);
    mamba_block_init(block, 42);  // Seed fixe pour reproductibilité
    
    /* Forward pass */
    mamba_block_forward(block, data.x, data.y, data.h0, data.h);
    
    printf("Output y: ");
    print_vector("y", data.y, L * D);
    
    /* Vérifications basiques */
    int result = 1;
    
    /* Vérifier que l'output n'est pas tout zéro */
    int all_zero = 1;
    for (long i = 0; i < L * D; i++) {
        if (fabsf(data.y[i]) > 1e-6f) {
            all_zero = 0;
            break;
        }
    }
    
    if (all_zero) {
        printf("FAIL: Output is all zeros\n");
        result = 0;
    }
    
    /* Vérifier qu'il n'y a pas de NaN ou inf */
    for (long i = 0; i < L * D; i++) {
        if (!isfinite(data.y[i])) {
            printf("FAIL: Output contains NaN or inf at index %ld\n", i);
            result = 0;
            break;
        }
    }
    
    if (result) {
        printf("PASS: MambaBlock forward simple case\n");
    }
    
    /* Nettoyage */
    mamba_block_free(block);
    free(data.x); free(data.y); free(data.h0); free(data.h);
    
    return result;
}

static int test_mamba_block_forward_deterministic() {
    printf("Testing MambaBlock forward deterministic...\n");
    
    const long L = 3, D = 4, M = 8;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    /* Données de test identiques */
    float x[12] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    float h0[32] = {0.0f};  // Zéro
    float y1[12], y2[12];
    float h1[96], h2[96];
    
    /* Créer deux blocks identiques */
    MambaBlock *block1, *block2;
    mamba_block_create(&cfg, &block1);
    mamba_block_create(&cfg, &block2);
    
    /* Initialiser avec la même seed */
    mamba_block_init(block1, 123);
    mamba_block_init(block2, 123);
    
    /* Forward passes */
    mamba_block_forward(block1, x, y1, h0, h1);
    mamba_block_forward(block2, x, y2, h0, h2);
    
    /* Comparer les outputs */
    int result = compare_vectors(y1, y2, L * D, EPSILON);
    
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
    
    const long L = 3, D = 4, M = 8;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
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
    MambaBlockTestData data;
    data.L = L; data.D = D; data.M = M;
    
    data.x = (float*)malloc(L * D * sizeof(float));
    data.y = (float*)malloc(L * D * sizeof(float));
    data.h0 = (float*)malloc(D * M * sizeof(float));
    data.h = (float*)malloc(L * D * M * sizeof(float));
    data.dy = (float*)malloc(L * D * sizeof(float));
    data.dx = (float*)malloc(L * D * sizeof(float));
    data.dh0 = (float*)malloc(D * M * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    fill_test_data(data.x, L * D, -1.0f, 1.0f);
    fill_test_data(data.y, L * D, -1.0f, 1.0f);
    memset(data.h0, 0, D * M * sizeof(float));
    fill_test_data(data.dy, L * D, -0.1f, 0.1f);  // Petits gradients
    
    printf("Input x: ");
    print_vector("x", data.x, L * D);
    printf("Gradient dy: ");
    print_vector("dy", data.dy, L * D);
    
    /* Créer et initialiser le MambaBlock */
    MambaBlock *block;
    mamba_block_create(&cfg, &block);
    mamba_block_init(block, 42);
    mamba_block_enable_training(block, &opt);
    
    /* Backward pass */
    mamba_block_backward(block, data.x, data.h0, data.dy, data.dx, data.dh0);
    
    printf("Gradient dx: ");
    print_vector("dx", data.dx, L * D);
    printf("Gradient dh0: ");
    print_vector("dh0", data.dh0, D * M);
    
    /* Vérifications basiques */
    int result = 1;
    
    /* Vérifier que les gradients ne sont pas tout zéro */
    int all_zero = 1;
    for (long i = 0; i < L * D; i++) {
        if (fabsf(data.dx[i]) > 1e-6f) {
            all_zero = 0;
            break;
        }
    }
    
    if (all_zero) {
        printf("FAIL: Input gradient is all zeros\n");
        result = 0;
    }
    
    /* Vérifier qu'il n'y a pas de NaN ou inf */
    for (long i = 0; i < L * D; i++) {
        if (!isfinite(data.dx[i]) || !isfinite(data.dh0[i])) {
            printf("FAIL: Gradients contain NaN or inf\n");
            result = 0;
            break;
        }
    }
    
    if (result) {
        printf("PASS: MambaBlock backward simple case\n");
    }
    
    /* Nettoyage */
    mamba_block_free(block);
    free(data.x); free(data.y); free(data.h0); free(data.h);
    free(data.dy); free(data.dx); free(data.dh0);
    
    return result;
}

/* ============================================================
 * Tests d'intégration complète
 * ============================================================ */

static int test_mamba_block_integration() {
    printf("Testing MambaBlock integration (forward + backward)...\n");
    
    const long L = 2, D = 4, M = 8;
    const int iterations = 10;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
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
    float h0[32] = {0.0f};
    
    /* Créer le MambaBlock */
    MambaBlock *block;
    mamba_block_create(&cfg, &block);
    mamba_block_init(block, 42);
    mamba_block_enable_training(block, &opt);
    
    printf("Training loop:\n");
    for (int iter = 0; iter < iterations; iter++) {
        /* Forward */
        float y[8];
        mamba_block_forward(block, x, y, h0, NULL);
        
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
        float dx[8], dh0[32];
        mamba_block_backward(block, x, h0, dy, dx, dh0);
        
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
    
    const long L = 128, D = 64, M = 16;
    const int iterations = 100;
    
    /* Configuration */
    MBConfig cfg = {
        .dim = D,
        .state_size = M,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    /* Données de test */
    float *x = (float*)malloc(L * D * sizeof(float));
    float *y = (float*)malloc(L * D * sizeof(float));
    float *h0 = (float*)malloc(D * M * sizeof(float));
    
    fill_test_data(x, L * D, -1.0f, 1.0f);
    memset(h0, 0, D * M * sizeof(float));
    
    /* Créer le MambaBlock */
    MambaBlock *block;
    mamba_block_create(&cfg, &block);
    mamba_block_init(block, 42);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        mamba_block_forward(block, x, y, h0, NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("MambaBlock Performance:\n");
    printf("  Size: L=%ld, D=%ld, M=%ld\n", L, D, M);
    printf("  Time: %.3f sec (%d iterations)\n", elapsed, iterations);
    printf("  Throughput: %.2f forward passes/sec\n", (double)iterations / elapsed);
    
    /* Nettoyage */
    mamba_block_free(block);
    free(x); free(y); free(h0);
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
