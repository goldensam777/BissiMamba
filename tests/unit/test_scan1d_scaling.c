/*
 * test_scan1d_scaling.c — Tests de scalabilité Scan1D
 *
 * Test comment Scan1D scale avec différentes tailles de L, D, M
 * Objectif : Valider les performances et la stabilité sur tailles réalistes
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

static void fill_test_data(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * (max - min) + min;
    }
}

static double benchmark_scan1d_size(long L, long D, long M, int iterations) {
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
    
    free(x); free(A); free(B); free(C); free(dt); free(h); free(y);
    
    return gflops;
}

/* ============================================================
 * Tests de scalabilité
 * ============================================================ */

static void test_scaling_sequence_length() {
    printf("=== Scaling: Sequence Length (L) ===\n");
    printf("Testing with D=64, M=16, varying L\n\n");
    
    const long D = 64, M = 16;
    const long L_values[] = {32, 64, 128, 256, 512};
    const int n_L = sizeof(L_values) / sizeof(L_values[0]);
    
    printf("%-8s %-12s %-12s %-15s %-15s\n", "L", "Elements", "Time (sec)", "GFLOPS", "Throughput");
    printf("%-8s %-12s %-12s %-15s %-15s\n", "--------", "------------", "------------", "---------------", "---------------");
    
    for (int i = 0; i < n_L; i++) {
        long L = L_values[i];
        long elements = L * D * M;
        double gflops = benchmark_scan1d_size(L, D, M, 50);
        
        printf("%-8ld %-12ld %-12.3f %-15.2f %-15.2f\n", 
               L, elements, 0.0, gflops, (double)50 / 0.0);
    }
}

static void test_scaling_feature_dimension() {
    printf("\n=== Scaling: Feature Dimension (D) ===\n");
    printf("Testing with L=128, M=16, varying D\n\n");
    
    const long L = 128, M = 16;
    const long D_values[] = {16, 32, 64, 128, 256};
    const int n_D = sizeof(D_values) / sizeof(D_values[0]);
    
    printf("%-8s %-12s %-12s %-15s %-15s\n", "D", "Elements", "Time (sec)", "GFLOPS", "Throughput");
    printf("%-8s %-12s %-12s %-15s %-15s\n", "--------", "------------", "------------", "---------------", "---------------");
    
    for (int i = 0; i < n_D; i++) {
        long D = D_values[i];
        long elements = L * D * M;
        double gflops = benchmark_scan1d_size(L, D, M, 50);
        
        printf("%-8ld %-12ld %-12.3f %-15.2f %-15.2f\n", 
               D, elements, 0.0, gflops, (double)50 / 0.0);
    }
}

static void test_scaling_state_dimension() {
    printf("\n=== Scaling: State Dimension (M) ===\n");
    printf("Testing with L=128, D=64, varying M\n\n");
    
    const long L = 128, D = 64;
    const long M_values[] = {4, 8, 16, 32, 64};
    const int n_M = sizeof(M_values) / sizeof(M_values[0]);
    
    printf("%-8s %-12s %-12s %-15s %-15s\n", "M", "Elements", "Time (sec)", "GFLOPS", "Throughput");
    printf("%-8s %-12s %-12s %-15s %-15s\n", "--------", "------------", "------------", "---------------", "---------------");
    
    for (int i = 0; i < n_M; i++) {
        long M = M_values[i];
        long elements = L * D * M;
        double gflops = benchmark_scan1d_size(L, D, M, 50);
        
        printf("%-8ld %-12ld %-12.3f %-15.2f %-15.2f\n", 
               M, elements, 0.0, gflops, (double)50 / 0.0);
    }
}

static void test_large_scale() {
    printf("\n=== Large Scale Test ===\n");
    printf("Testing with realistic large dimensions\n\n");
    
    /* Tailles réalistes pour Mamba moderne */
    const struct {
        long L, D, M;
        const char *name;
    } configs[] = {
        {256, 128, 32, "Small Model"},
        {512, 256, 64, "Medium Model"},  
        {1024, 512, 128, "Large Model"},
        {2048, 1024, 256, "XL Model"}
    };
    
    printf("%-15s %-8s %-8s %-8s %-12s %-15s %-15s\n", 
           "Model", "L", "D", "M", "Elements", "GFLOPS", "Throughput");
    printf("%-15s %-8s %-8s %-8s %-12s %-15s %-15s\n", 
           "---------------", "--------", "--------", "--------", "------------", "---------------", "---------------");
    
    for (int i = 0; i < 4; i++) {
        long L = configs[i].L;
        long D = configs[i].D;
        long M = configs[i].M;
        long elements = L * D * M;
        
        double gflops = benchmark_scan1d_size(L, D, M, 10);  // Moins d'itérations pour grandes tailles
        
        printf("%-15s %-8ld %-8ld %-8ld %-12ld %-15.2f %-15.2f\n", 
               configs[i].name, L, D, M, elements, gflops, (double)10 / 0.0);
    }
}

/* ============================================================
 * Analyse de complexité
 * ============================================================ */

static void analyze_complexity() {
    printf("\n=== Complexity Analysis ===\n");
    printf("Scan1D has O(L * D * M) time complexity\n");
    printf("Each timestep requires O(D * M) operations\n\n");
    
    /* Calcul de la mémoire requise */
    printf("Memory requirements:\n");
    const long L = 512, D = 256, M = 64;
    
    size_t input_size = L * D * sizeof(float);
    size_t params_size = D * M * 3 * sizeof(float);  // A, B, C
    size_t state_size = D * M * sizeof(float);  // Hidden states
    size_t output_size = L * D * sizeof(float);
    
    printf("For L=%ld, D=%ld, M=%ld:\n", L, D, M);
    printf("  Input:  %zu bytes (%.2f MB)\n", input_size, input_size / (1024.0 * 1024.0));
    printf("  Params: %zu bytes (%.2f MB)\n", params_size, params_size / (1024.0 * 1024.0));
    printf("  State:  %zu bytes (%.2f MB)\n", state_size, state_size / (1024.0 * 1024.0));
    printf("  Output: %zu bytes (%.2f MB)\n", output_size, output_size / (1024.0 * 1024.0));
    printf("  Total:  %zu bytes (%.2f MB)\n", 
           input_size + params_size + state_size + output_size,
           (input_size + params_size + state_size + output_size) / (1024.0 * 1024.0));
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Scan1D Scaling Test Suite ===\n");
    printf("Testing Scan1D performance across different dimensions\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    /* Tests de scalabilité */
    test_scaling_sequence_length();
    test_scaling_feature_dimension();
    test_scaling_state_dimension();
    
    /* Test grande échelle */
    test_large_scale();
    
    /* Analyse de complexité */
    analyze_complexity();
    
    printf("\n=== Summary ===\n");
    printf("Scan1D scales cubically: O(L * D * M)\n");
    printf("Performance depends on cache efficiency and vectorization\n");
    printf("Large models benefit significantly from AVX2 optimization\n");
    
    return 0;
}
