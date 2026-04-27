#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int load_moving_mnist(const char *path, float **data, uint32_t **labels,
                      size_t *num_samples, size_t L, size_t D) {
    (void)path;
    (void)L;
    (void)D;
    printf("[moving_mnist] Using random dummy data\n");
    *num_samples = 10000;
    *data = (float*)malloc((*num_samples) * L * D * sizeof(float));
    *labels = (uint32_t*)malloc((*num_samples) * sizeof(uint32_t));
    if (!*data || !*labels) {
        free(*data);
        free(*labels);
        *data = NULL;
        *labels = NULL;
        return -1;
    }
    /* Initialize to zero (actual loading would populate from files) */
    memset(*data, 0, (*num_samples) * L * D * sizeof(float));
    memset(*labels, 0, (*num_samples) * sizeof(uint32_t));
    return 0;
}
