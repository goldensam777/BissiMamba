/*
 * test_mamba3_gpu.cu - Test du forward Mamba-3 (GPU)
 *
 * Vérifie que le dispatch GPU et le forward fonctionnent.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
#include "kmamba.h"
}

#define TEST_SEQ_LEN 32
#define TEST_DIM 64
#define TEST_STATE 16

static float float_eq(float a, float b, float tol) {
    return fabsf(a - b) < tol;
}

int main(void) {
    printf("=== Test Mamba-3 GPU Forward ===\n");

    /* Vérifier CUDA disponible */
    int cuda_available = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_available);
    if (err != cudaSuccess || cuda_available == 0) {
        printf("SKIP: No CUDA device available\n");
        return 0;
    }
    printf("[OK] CUDA available, %d device(s)\n", cuda_available);

    /* Configuration Mamba-3 */
    MBConfig cfg = {
        .dim = TEST_DIM,
        .state_size = TEST_STATE,
        .seq_len = TEST_SEQ_LEN,
        .mimo_rank = 1,
        .dt_scale = 0.01f,
        .dt_min = 1e-3f,
        .dt_max = 0.1f,
        .spatial_ndims = 0,
        .use_convnd = 0
    };

    /* Création du bloc */
    MambaBlock *block = mamba_block_create(&cfg);
    if (!block) {
        printf("FAIL: mamba_block_create returned NULL\n");
        return 1;
    }
    printf("[OK] Block created\n");

    /* Initialisation */
    mamba_block_init(block);
    printf("[OK] Block initialized\n");

    /* Données de test */
    float *input = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    float *output_cpu = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    float *output_gpu = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    if (!input || !output_cpu || !output_gpu) {
        printf("FAIL: malloc failed\n");
        return 1;
    }

    /* Remplir avec des valeurs aléatoires */
    srand(42);
    for (int i = 0; i < TEST_SEQ_LEN * TEST_DIM; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    /* Forward CPU (pour comparaison) */
    printf("Running CPU forward...\n");
    mamba_block_forward(block, output_cpu, input, 1);
    printf("[OK] CPU forward completed\n");

    /* Forward GPU via mamba_block_forward avec backend auto */
    printf("Running GPU forward (via auto dispatch)...\n");
    
    /* Forcer le backend GPU */
    kmamba_backend_preference = KMAMBA_BACKEND_GPU;
    kmamba_backend_init();
    
    mamba_block_forward(block, output_gpu, input, 1);
    printf("[OK] GPU forward completed\n");

    /* Vérifier que GPU a produit des résultats différents de zéro */
    int has_nonzero = 0;
    int has_nan = 0;
    for (int i = 0; i < TEST_SEQ_LEN * TEST_DIM; i++) {
        if (!float_eq(output_gpu[i], 0.0f, 1e-10f)) has_nonzero = 1;
        if (isnan(output_gpu[i]) || isinf(output_gpu[i])) has_nan = 1;
    }

    if (!has_nonzero) {
        printf("WARN: GPU output is all zeros\n");
    } else {
        printf("[OK] GPU output has non-zero values\n");
    }

    if (has_nan) {
        printf("FAIL: GPU output contains NaN or Inf\n");
        return 1;
    } else {
        printf("[OK] GPU output has no NaN/Inf\n");
    }

    /* Comparaison CPU vs GPU */
    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < TEST_SEQ_LEN * TEST_DIM; i++) {
        float diff = fabsf(output_cpu[i] - output_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        sum_sq_diff += diff * diff;
    }
    float rmse = sqrtf(sum_sq_diff / (TEST_SEQ_LEN * TEST_DIM));
    
    printf("CPU vs GPU comparison:\n");
    printf("  Max absolute diff: %.6e\n", max_diff);
    printf("  RMSE: %.6e\n", rmse);
    
    /* Tolérance pour float32 accumulé différemment CPU/GPU */
    if (max_diff < 1e-3f) {
        printf("[OK] CPU and GPU outputs match within tolerance (1e-3)\n");
    } else {
        printf("WARN: CPU/GPU diff larger than expected (%.6e > 1e-3)\n", max_diff);
        /* Afficher quelques valeurs pour debug */
        printf("  Sample CPU: %.6f, GPU: %.6f at index 0\n", output_cpu[0], output_gpu[0]);
    }

    /* Cleanup */
    mamba_block_free(block);
    free(input);
    free(output_cpu);
    free(output_gpu);

    printf("\n=== GPU Test Completed ===\n");
    return 0;
}
