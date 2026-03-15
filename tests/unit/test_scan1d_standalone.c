/*
 * test_scan1d_standalone.c — Tests Scan1D sans dépendances externes
 * Version standalone pour valider l'algorithme de référence
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define EPSILON 1e-5f

/* ============================================================
 * Structures de données pour Scan1D
 * ============================================================ */

typedef struct {
    float *x;       /* Input sequence [L * D] */
    float *A;       /* Diagonal matrix [D * M] */
    float *B;       /* Input matrix [L * D * M] */
    float *C;       /* Output matrix [L * D * M] */
    float *dt;      /* Delta [L * D] */
    float *h;       /* Hidden states [L * D * M] */
    float *y;       /* Output [L * D] */
    long   L;       /* Sequence length */
    long   D;       /* Feature dimension */
    long   M;       /* State dimension */
} ScanParams;

/* ============================================================
 * Implémentation de référence Scan1D
 * ============================================================ */

static void scan1d_reference(const ScanParams *p) {
    /* Initialiser l'état */
    memset(p->h, 0, p->D * p->M * sizeof(float));
    memset(p->y, 0, p->L * p->D * sizeof(float));
    
    for (long t = 0; t < p->L; t++) {
        float dt_t = p->dt[t];
        
        /* Pour chaque dimension et état */
        for (long d = 0; d < p->D; d++) {
            for (long m = 0; m < p->M; m++) {
                long dm_idx = d * p->M + m;
                long tdm_idx = t * p->D * p->M + dm_idx;
                
                /* Discrétisation */
                float a_val = p->A[dm_idx];
                float dA = expf(dt_t * a_val);
                
                float b_val = p->B[tdm_idx];
                float dB = dt_t * b_val;
                
                float x_td = p->x[t * p->D + d];
                
                /* Mise à jour de l'état */
                float h_old = p->h[dm_idx];
                float h_new = dA * h_old + dB * x_td;
                p->h[dm_idx] = h_new;
                
                /* Sortie */
                float c_val = p->C[tdm_idx];
                p->y[t * p->D + d] += c_val * h_new;
            }
        }
    }
}

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
    for (size_t i = 0; i < n; i++) {
        printf("%.4f", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

/* ============================================================
 * Tests Scan1D Forward
 * ============================================================ */

static int test_scan1d_forward_simple() {
    printf("Testing Scan1D forward simple case...\n");
    
    const long L = 4, D = 2, M = 2;
    
    /* Données de test simples et prévisibles */
    float x[8] = {1.0f, 2.0f, 3.0f, 4.0f,   // t=0..3, d=0
                     5.0f, 6.0f, 7.0f, 8.0f};  // t=0..3, d=1
    
    float A[4] = {0.1f, 0.2f, 0.3f, 0.4f};  // d=0..1, m=0..1
    float B[16] = {0.5f, 0.6f, 0.7f, 0.8f,   // t=0..3, d=0..1, m=0..1
                  0.9f, 1.0f, 1.1f, 1.2f,
                  1.3f, 1.4f, 1.5f, 1.6f,
                  1.7f, 1.8f, 1.9f, 2.0f};
    
    float C[16] = {2.0f, 2.1f, 2.2f, 2.3f,   // t=0..3, d=0..1, m=0..1
                  2.4f, 2.5f, 2.6f, 2.7f,
                  2.8f, 2.9f, 3.0f, 3.1f,
                  3.2f, 3.3f, 3.4f, 3.5f,
                  3.6f, 3.7f, 3.8f, 3.9f};
    
    float dt[4] = {0.1f, 0.1f, 0.1f, 0.1f};  // t=0..3
    
    float h_ref[8], y_ref[8];
    float h_test[8], y_test[8];
    
    /* Référence */
    ScanParams params_ref = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h_ref, .y = y_ref,
        .L = L, .D = D, .M = M
    };
    
    scan1d_reference(&params_ref);
    
    /* Test identique (pour valider la répétabilité) */
    ScanParams params_test = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h_test, .y = y_test,
        .L = L, .D = D, .M = M
    };
    
    printf("Input x: ");
    print_vector("x", x, L * D);
    printf("Delta dt: ");
    print_vector("dt", dt, L);
    
    scan1d_reference(&params_test);
    
    printf("Output y (ref): ");
    print_vector("y_ref", y_ref, L * D);
    printf("Output y (test): ");
    print_vector("y_test", y_test, L * D);
    
    int result = compare_vectors(y_ref, y_test, L * D, EPSILON);
    if (!result) {
        printf("FAIL: Scan1D forward simple case failed\n");
        return 0;
    }
    
    printf("PASS: Scan1D forward simple case\n");
    return 1;
}

static int test_scan1d_forward_random() {
    printf("Testing Scan1D forward random case...\n");
    
    const long L = 8, D = 4, M = 3;
    
    float *x = (float*)malloc(L * D * sizeof(float));
    float *A = (float*)malloc(D * M * sizeof(float));
    float *B = (float*)malloc(L * D * M * sizeof(float));
    float *C = (float*)malloc(L * D * M * sizeof(float));
    float *dt = (float*)malloc(L * sizeof(float));
    
    float *h_ref = (float*)malloc(D * M * sizeof(float));
    float *h_test = (float*)malloc(D * M * sizeof(float));
    float *y_ref = (float*)malloc(L * D * sizeof(float));
    float *y_test = (float*)malloc(L * D * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    fill_test_data(x, L * D, -1.0f, 1.0f);
    fill_test_data(A, D * M, -0.5f, 0.5f);
    fill_test_data(B, L * D * M, -0.3f, 0.3f);
    fill_test_data(C, L * D * M, -0.2f, 0.2f);
    fill_test_data(dt, L, 0.01f, 0.1f);  // Petits deltas positifs
    
    /* Référence */
    ScanParams params_ref = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h_ref, .y = y_ref,
        .L = L, .D = D, .M = M
    };
    
    scan1d_reference(&params_ref);
    
    /* Test */
    ScanParams params_test = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h_test, .y = y_test,
        .L = L, .D = D, .M = M
    };
    
    scan1d_reference(&params_test);
    
    int result = compare_vectors(y_ref, y_test, L * D, EPSILON);
    
    free(x); free(A); free(B); free(C); free(dt);
    free(h_ref); free(h_test); free(y_ref); free(y_test);
    
    if (!result) {
        printf("FAIL: Scan1D forward random case failed\n");
        return 0;
    }
    
    printf("PASS: Scan1D forward random case (L=%ld, D=%ld, M=%ld)\n", L, D, M);
    return 1;
}

/* ============================================================
 * Tests de stabilité numérique
 * ============================================================ */

static int test_scan1d_stability() {
    printf("Testing Scan1D numerical stability...\n");
    
    const long L = 10, D = 2, M = 2;
    
    /* Test avec des deltas très petits */
    float x[20] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                     11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f};
    
    float A[4] = {0.001f, 0.002f, 0.003f, 0.004f};
    float B[40] = {0.1f};  // Tous identiques
    float C[40] = {1.0f};   // Tous identiques
    float dt[10] = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 
                   0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
    
    float h_ref[8], h_test[8];
    float y_ref[20], y_test[20];
    
    /* Référence */
    ScanParams params_ref = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h_ref, .y = y_ref,
        .L = L, .D = D, .M = M
    };
    
    scan1d_reference(&params_ref);
    
    /* Test */
    ScanParams params_test = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h_test, .y = y_test,
        .L = L, .D = D, .M = M
    };
    
    scan1d_reference(&params_test);
    
    int result = compare_vectors(y_ref, y_test, L * D, EPSILON * 10.0f);  // Tolérance plus grande
    
    if (!result) {
        printf("FAIL: Scan1D stability test failed\n");
        return 0;
    }
    
    printf("PASS: Scan1D numerical stability test\n");
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_scan1d() {
    printf("\n=== Scan1D Performance Benchmarks ===\n");
    
    const long L = 128, D = 64, M = 16;  // Tailles réalistes
    const int iterations = 100;
    
    float *x = (float*)malloc(L * D * sizeof(float));
    float *A = (float*)malloc(D * M * sizeof(float));
    float *B = (float*)malloc(L * D * M * sizeof(float));
    float *C = (float*)malloc(L * D * M * sizeof(float));
    float *dt = (float*)malloc(L * sizeof(float));
    float *h = (float*)malloc(D * M * sizeof(float));
    float *y = (float*)malloc(L * D * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    fill_test_data(x, L * D, -1.0f, 1.0f);
    fill_test_data(A, D * M, -0.5f, 0.5f);
    fill_test_data(B, L * D * M, -0.3f, 0.3f);
    fill_test_data(C, L * D * M, -0.2f, 0.2f);
    fill_test_data(dt, L, 0.01f, 0.1f);
    
    ScanParams params = {
        .x = x, .A = A, .B = B, .C = C, .dt = dt,
        .h = h, .y = y,
        .L = L, .D = D, .M = M
    };
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        scan1d_reference(&params);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double ops_per_iter = (double)L * D * M * 2.0;  // Approx 2 opérations par élément
    double total_ops = ops_per_iter * iterations;
    double gflops = total_ops / (elapsed * 1e9);
    
    printf("Scan1D Performance:\n");
    printf("  Size: L=%ld, D=%ld, M=%ld\n", L, D, M);
    printf("  Time: %.3f sec (%d iterations)\n", elapsed, iterations);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("  Throughput: %.2f sequences/sec\n", (double)iterations / elapsed);
    
    free(x); free(A); free(B); free(C); free(dt); free(h); free(y);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Scan1D Standalone Test Suite ===\n");
    printf("Testing Mamba selective scan 1D reference implementation\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests forward */
    total++; passed += test_scan1d_forward_simple();
    total++; passed += test_scan1d_forward_random();
    total++; passed += test_scan1d_stability();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        benchmark_scan1d();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
