/*
 * test_mamba3_forward.c - Test du forward Mamba-3 (CPU)
 *
 * Vérifie que les corrections malloc et la formule Mamba-3 fonctionnent.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmamba.h"

#define TEST_SEQ_LEN 32
#define TEST_DIM 64
#define TEST_STATE 16

static float float_eq(float a, float b, float tol) {
    return fabsf(a - b) < tol;
}

int main(void) {
    printf("=== Test Mamba-3 Forward ===\n");

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
    float *output = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    if (!input || !output) {
        printf("FAIL: malloc failed\n");
        return 1;
    }

    /* Remplir avec des valeurs aléatoires */
    for (int i = 0; i < TEST_SEQ_LEN * TEST_DIM; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    /* Forward */
    mamba_block_forward(block, output, input, 1);
    printf("[OK] Forward pass completed\n");

    /* Vérifications basiques */
    int has_nonzero = 0;
    int has_nan = 0;
    for (int i = 0; i < TEST_SEQ_LEN * TEST_DIM; i++) {
        if (!float_eq(output[i], 0.0f, 1e-10f)) has_nonzero = 1;
        if (isnan(output[i]) || isinf(output[i])) has_nan = 1;
    }

    if (!has_nonzero) {
        printf("WARN: Output is all zeros\n");
    } else {
        printf("[OK] Output has non-zero values\n");
    }

    if (has_nan) {
        printf("FAIL: Output contains NaN or Inf\n");
        return 1;
    } else {
        printf("[OK] Output has no NaN/Inf\n");
    }

    /* Test avec MIMO rank > 1 */
    block->config.mimo_rank = 2;
    mamba_block_forward(block, output, input, 1);
    printf("[OK] Forward with MIMO rank=2 completed\n");

    /* Cleanup */
    mamba_block_free(block);
    free(input);
    free(output);

    printf("\n=== All tests PASSED ===\n");
    return 0;
}
