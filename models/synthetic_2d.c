#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int generate_synthetic_2d(float **data, uint32_t **labels, size_t *num_samples, size_t L, size_t D) {
    printf("[synthetic_2d] Using random dummy data\n");
    *num_samples = 1000;
    *data = (float*)malloc((*num_samples) * L * D * sizeof(float));
    *labels = (uint32_t*)malloc((*num_samples) * sizeof(uint32_t));
    return 0;
}
