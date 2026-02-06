#include "mamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(void) {
    MambaConfig config = {
        .dim = 32,
        .state_size = 8,
        .seq_len = 16,
        .dt_rank = 0.1f,
        .dt_scale = 1.0f,
        .dt_init = 0.001f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };

    MambaBlock *m = mamba_block_create(&config);
    if (!m) return 1;
    mamba_block_init(m);

    OptimConfig opt = { .lr = 1e-3f, .mu = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .clip_norm = 1.0f, .weight_decay = 0.0f };
    mamba_attach_optimizer(m, &opt);

    size_t seq_len = config.seq_len; size_t dim = config.dim;
    real_t *input = (real_t *)malloc(seq_len * dim * sizeof(real_t));
    real_t *output = (real_t *)malloc(seq_len * dim * sizeof(real_t));

    /* toy data: sinusoid per dimension */
    for (size_t t = 0; t < seq_len; t++) for (size_t d = 0; d < dim; d++) {
        input[t*dim + d] = sinf(2.0f * 3.14159265f * (t + d) / (float)seq_len);
    }

    size_t epochs = 50;
    for (size_t e = 0; e < epochs; e++) {
        /* forward */
        mamba_forward(m, output, input, 1);

        /* compute simple MSE loss on first output dim vs mean of input dims */
        real_t loss = 0.0f;
        real_t *dY = (real_t *)calloc(seq_len, sizeof(real_t));
        for (size_t t = 0; t < seq_len; t++) {
            real_t pred = output[t*dim + 0];
            real_t target = 0.0f;
            for (size_t d = 0; d < dim; d++) target += input[t*dim + d];
            target /= (real_t)dim;
            real_t err = pred - target;
            loss += err * err;
            dY[t] = 2.0f * err / (real_t)seq_len; /* gradient contribution */
        }
        loss /= (real_t)seq_len;

        /* zero grads */
        mamba_zero_grads(m);

        /* backward */
        mamba_backward(m, dY, input, 0);

        /* step optimizer */
        mamba_optimizer_step(m, &opt);

        free(dY);

        if ((e % 5) == 0) printf("Epoch %zu loss=%.6f\n", e, (double)loss);
    }

    free(input); free(output);
    mamba_free_optimizer(m);
    mamba_block_free(m);
    return 0;
}
