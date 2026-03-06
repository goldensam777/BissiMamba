#include "mamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(void) {
    /* 1M parameter configuration (smoke-test) */
    MambaConfig config = {
        .dim = 706,
        .state_size = 512,
        .seq_len = 32,
        .dt_rank = 0.1f,
        .dt_scale = 1.0f,
        .dt_init = 0.001f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };

    MambaBlock *m = mamba_block_create(&config);
    if (!m) return 1;
    mamba_block_init(m);

    OptimConfig opt = { .lr = 1e-4f, .mu = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .clip_norm = 5.0f, .weight_decay = 1e-5f };
    mamba_attach_optimizer(m, &opt);

    size_t seq_len = config.seq_len; size_t dim = config.dim;
    real_t *input = (real_t *)malloc(seq_len * dim * sizeof(real_t));
    real_t *output = (real_t *)malloc(seq_len * dim * sizeof(real_t));

    /* toy data: sinusoid per dimension */
    for (size_t t = 0; t < seq_len; t++) for (size_t d = 0; d < dim; d++) {
        input[t*dim + d] = sinf(2.0f * 3.14159265f * (t + d) / (float)seq_len);
    }

    size_t epochs = 10; /* short smoke-test */
    FILE *csv = fopen("train_log.csv", "w");
    if (csv) fprintf(csv, "epoch,loss\n");

    for (size_t e = 0; e < epochs; e++) {
        /* forward */
        mamba_forward(m, output, input, 1);

        /* compute simple MSE loss on first output dim vs mean of input dims */
        real_t loss = 0.0f;
        real_t *dY = (real_t *)calloc(seq_len * dim, sizeof(real_t));
        for (size_t t = 0; t < seq_len; t++) {
            real_t pred = output[t*dim + 0];
            real_t target = 0.0f;
            for (size_t d = 0; d < dim; d++) target += input[t*dim + d];
            target /= (real_t)dim;
            real_t err = pred - target;
            loss += err * err;
            dY[t*dim + 0] = 2.0f * err / (real_t)seq_len; /* gradient on dim[0] only */
        }
        loss /= (real_t)seq_len;

        /* zero grads */
        mamba_zero_grads(m);

        /* backward */
        mamba_backward(m, dY, input, 0);

        /* step optimizer */
        mamba_optimizer_step(m, &opt);

        free(dY);

        if (csv) fprintf(csv, "%zu,%.6f\n", e, (double)loss);
        if ((e % 1) == 0) printf("Epoch %zu loss=%.6f\n", e, (double)loss);

        /* checkpoint every 5 epochs */
        if ((e % 5) == 0) {
            FILE *f = fopen("checkpoint_epoch.bin", "wb");
            if (f) {
                /* dump A_log, B_mat, C_mat, W_in, W_out */
                fwrite(m->A_log.data, sizeof(real_t), m->A_log.rows * m->A_log.cols, f);
                fwrite(m->B_mat.data, sizeof(real_t), m->B_mat.rows * m->B_mat.cols, f);
                fwrite(m->C_mat.data, sizeof(real_t), m->C_mat.rows * m->C_mat.cols, f);
                fwrite(m->W_in.data, sizeof(real_t), m->W_in.rows * m->W_in.cols, f);
                fwrite(m->W_out.data, sizeof(real_t), m->W_out.rows * m->W_out.cols, f);
                fclose(f);
            }
        }
    }

    if (csv) fclose(csv);
    free(input); free(output);
    mamba_free_optimizer(m);
    mamba_block_free(m);
    return 0;
}
