#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int load_moving_mnist(const char *path, float **data, uint32_t **labels, size_t *num_samples, size_t L, size_t D) {
    (void)path;
    printf("[moving_mnist] Using random dummy data\n");
    *num_samples = 10000;
    *data = (float*)malloc((*num_samples) * L * D * sizeof(float));
    *labels = (uint32_t*)malloc((*num_samples) * sizeof(uint32_t));
    return 0;
}
