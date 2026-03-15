/*
 * test_mamba_block_minimal.c — Tests MambaBlock minimal sans conflits
 *
 * Phase 2 : Tests d'intégration MambaBlock (forward/backward)
 * Version minimaliste pour éviter les conflits de types
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* Définitions manuelles pour éviter les conflits */
typedef struct {
    size_t dim;
    size_t state_size;
    size_t seq_len;
    float dt_scale;
    float dt_min;
    float dt_max;
    float dt_rank;
    float dt_init;
    int use_convnd;
    long convnd_K;
    long convnd_ndims;
} SimpleMBConfig;

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} SimpleMatrix;

typedef struct {
    SimpleMBConfig config;
    SimpleMatrix W_in, W_out, A_log, B_mat, C_mat, delta_proj;
    float *hidden, *delta, *scan_B, *scan_C, *scan_delta, *scan_h;
    float *convnd_kernel, *convnd_bias;
    void *convnd_ws;
} SimpleMambaBlock;

/* Signatures des fonctions k-mamba */
SimpleMambaBlock* simple_mamba_block_create(const SimpleMBConfig *config);
void simple_mamba_block_free(SimpleMambaBlock *block);
void simple_mamba_block_init(SimpleMambaBlock *block);
void simple_mamba_block_forward(SimpleMambaBlock *block, float *output, const float *input, size_t batch_size);

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
    SimpleMBConfig cfg = {
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
    SimpleMambaBlock *block = simple_mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock\n");
        free(x); free(y);
        return 0;
    }
    
    simple_mamba_block_init(block);
    
    /* Forward pass */
    simple_mamba_block_forward(block, y, x, L);
    
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
    simple_mamba_block_free(block);
    free(x); free(y);
    
    return success;
}

static int test_mamba_block_forward_deterministic() {
    printf("Testing MambaBlock forward deterministic...\n");
    
    const size_t L = 3, D = 4, M = 8;
    
    /* Configuration */
    SimpleMBConfig cfg = {
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
    SimpleMambaBlock *block1 = simple_mamba_block_create(&cfg);
    SimpleMambaBlock *block2 = simple_mamba_block_create(&cfg);
    
    if (block1 == NULL || block2 == NULL) {
        printf("FAIL: Could not create MambaBlocks\n");
        return 0;
    }
    
    /* Initialiser */
    simple_mamba_block_init(block1);
    simple_mamba_block_init(block2);
    
    /* Forward passes */
    simple_mamba_block_forward(block1, y1, x, L);
    simple_mamba_block_forward(block2, y2, x, L);
    
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
    simple_mamba_block_free(block1);
    simple_mamba_block_free(block2);
    
    return result;
}

/* ============================================================
 * Test de validation de l'architecture
 * ============================================================ */

static int test_mamba_block_architecture() {
    printf("Testing MambaBlock architecture validation...\n");
    
    const size_t L = 2, D = 4, M = 8;
    
    /* Configuration */
    SimpleMBConfig cfg = {
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
    float x[8] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    float y[8];
    
    /* Créer le MambaBlock */
    SimpleMambaBlock *block = simple_mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock\n");
        return 0;
    }
    
    simple_mamba_block_init(block);
    
    printf("MambaBlock created successfully\n");
    printf("  dim: %zu\n", cfg.dim);
    printf("  state_size: %zu\n", cfg.state_size);
    printf("  seq_len: %zu\n", cfg.seq_len);
    printf("  dt_scale: %.3f\n", cfg.dt_scale);
    
    /* Forward pass */
    simple_mamba_block_forward(block, y, x, L);
    
    printf("Forward pass completed\n");
    printf("Input: ");
    print_vector("x", x, L * D);
    printf("Output: ");
    print_vector("y", y, L * D);
    
    /* Vérifier que le block a bien été initialisé */
    int success = 1;
    if (block->W_in.data == NULL || block->W_out.data == NULL) {
        printf("FAIL: Matrices not initialized\n");
        success = 0;
    }
    
    if (block->hidden == NULL || block->scan_h == NULL) {
        printf("FAIL: Buffers not allocated\n");
        success = 0;
    }
    
    if (success) {
        printf("PASS: MambaBlock architecture validation\n");
    }
    
    /* Nettoyage */
    simple_mamba_block_free(block);
    
    return success;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_mamba_block() {
    printf("\n=== MambaBlock Performance Benchmarks ===\n");
    
    const size_t L = 128, D = 64, M = 16;
    const int iterations = 50;
    
    /* Configuration */
    SimpleMBConfig cfg = {
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
    SimpleMambaBlock *block = simple_mamba_block_create(&cfg);
    if (block == NULL) {
        printf("FAIL: Could not create MambaBlock for benchmark\n");
        free(x); free(y);
        return;
    }
    
    simple_mamba_block_init(block);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        simple_mamba_block_forward(block, y, x, L);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("MambaBlock Performance:\n");
    printf("  Size: L=%zu, D=%zu, M=%zu\n", L, D, M);
    printf("  Time: %.3f sec (%d iterations)\n", elapsed, iterations);
    printf("  Throughput: %.2f forward passes/sec\n", (double)iterations / elapsed);
    
    /* Nettoyage */
    simple_mamba_block_free(block);
    free(x); free(y);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== MambaBlock Minimal Test Suite ===\n");
    printf("Testing MambaBlock integration with minimal API\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests forward */
    total++; passed += test_mamba_block_forward_simple();
    total++; passed += test_mamba_block_forward_deterministic();
    
    /* Test d'architecture */
    total++; passed += test_mamba_block_architecture();
    
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
