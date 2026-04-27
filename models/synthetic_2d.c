#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/**
 * Generate synthetic 2D pattern classification data.
 * Task: Binary classification where class 1 has anti-diagonal pattern
 * (non-zero values on positions where i+j == 7 in an 8x8 grid).
 * Class 0 is uniform noise.
 * This requires 2D spatial reasoning - 1D scan cannot easily detect anti-diagonals.
 */
int generate_synthetic_2d(float **data, uint32_t **labels,
                          size_t *num_samples, size_t L, size_t D) {
    /* Expect 8x8 grid (64 elements), but handle other sizes gracefully */
    if (L != 8 || D != 8) {
        printf("[synthetic_2d] Warning: Expected 8x8 grid, got %zux%zu\n", L, D);
    }

    *num_samples = 1000;
    *data = (float*)malloc((*num_samples) * L * D * sizeof(float));
    *labels = (uint32_t*)malloc((*num_samples) * sizeof(uint32_t));
    if (!*data || !*labels) {
        free(*data);
        free(*labels);
        *data = NULL;
        *labels = NULL;
        return -1;
    }

    printf("[synthetic_2d] Generating anti-diagonal pattern data: %zu samples\n", *num_samples);

    /* Seed RNG ( caller should call srand() at startup ) */

    for (size_t s = 0; s < *num_samples; s++) {
        /* 50% chance for each class */
        int is_class_1 = (rand() % 2) == 1;
        (*labels)[s] = is_class_1 ? 1 : 0;

        float *sample = (*data) + s * L * D;

        for (size_t i = 0; i < L; i++) {
            for (size_t j = 0; j < D; j++) {
                /* Anti-diagonal: i + j == 7 */
                int is_antidiag = (i + j == 7);

                /* Row-major indexing: i * D + j */
                size_t idx = i * D + j;

                if (is_class_1) {
                    /* Class 1: anti-diagonal has strong signal [0.5, 1.0],
                     * rest has weak noise [-0.2, 0.2] */
                    if (is_antidiag) {
                        sample[idx] = 0.5f + ((float)rand() / RAND_MAX) * 0.5f;
                    } else {
                        sample[idx] = -0.2f + ((float)rand() / RAND_MAX) * 0.4f;
                    }
                } else {
                    /* Class 0: uniform noise [-0.2, 0.2] everywhere */
                    sample[idx] = -0.2f + ((float)rand() / RAND_MAX) * 0.4f;
                }
            }
        }
    }

    return 0;
}
