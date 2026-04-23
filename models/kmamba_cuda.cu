/* ============================================================================
 * kmamba_cuda.cu - High-Performance GPU Training Instance
 * 
 * Matches PLAN.md: GPU Small MX450 (350M params)
 * 32K Vocab (Byte-level)
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "kmamba.h"
#include "kmamba_ser.h"

/* ═══════════════════════════════════════════════════════════════
 * Configuration (PLAN.md: MX450 Small)
 * ═══════════════════════════════════════════════════════════════ */
#define VOCAB_SIZE      32768
#define MODEL_DIM       512
#define STATE_SIZE      1024
#define LAYERS          4
#define SEQ_LEN         512
#define D_CONV          4
#define EXPAND          2.0f

#define DEFAULT_DATASET "data/kmamba_cpu_10M.txt"
#define SAVE_PATH       "models/kmamba_cuda.ser"

typedef struct {
    uint32_t *tokens;
    size_t    n_tokens;
} DataLoader;

DataLoader dataloader_create(const char *path) {
    DataLoader dl = {NULL, 0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return dl;

    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buf = (char*)malloc(fsize + 1);
    if (!buf) { fclose(fp); return dl; }
    fread(buf, 1, fsize, fp);
    buf[fsize] = '\0';
    fclose(fp);

    kmamba_tokenizer_init("bytes");
    dl.tokens = kmamba_encode(buf, &dl.n_tokens);
    free(buf);
    return dl;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

int main(int argc, char **argv) {
    printf("🚀 K-Mamba CUDA Instance (MX450 Profile)\n");
    printf("════════════════════════════════════════════\n\n");

    /* 1. Init Tokenizer */
    kmamba_tokenizer_init("bytes");
    uint32_t vocab_size = (uint32_t)kmamba_vocab_size();

    KMambaConfig cfg = {0};
    cfg.vocab_size = vocab_size;
    cfg.dim = MODEL_DIM;
    cfg.state_size = STATE_SIZE;
    cfg.n_layers = LAYERS;
    cfg.seq_len = SEQ_LEN;
    cfg.d_conv = D_CONV;
    cfg.expand_factor = EXPAND;
    cfg.weight_tying = 1;
    strncpy(cfg.model_name, "kmamba-cuda-350m", 63);

    KMamba *m = kmamba_create(&cfg);
    kmamba_init(m, 42);

    MBOptimConfig opt = {
        .lr = 8e-4f,
        .mu = 0.9f,
        .beta2 = 0.99f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 0.1f
    };
    kmamba_enable_training(m, &opt, 8e-4f, 0.1f);

    /* 2. Load Data */
    const char *data_path = (argc > 1) ? argv[1] : DEFAULT_DATASET;
    DataLoader dl = dataloader_create(data_path);
    if (!dl.tokens) {
        printf("⚠ Warning: Data not found, using internal fallback for performance test.\n");
        dl.tokens = kmamba_encode("CUDA Mamba is the future of efficient sequence modeling.", &dl.n_tokens);
    }

    /* 3. Batch Training Loop */
    int steps = 1000;
    int batch_size = 16;
    
    printf("🏃 Training on GPU (%d steps, batch=%d, seq=%d)...\n\n", steps, batch_size, SEQ_LEN);
    printf("%-6s | %-8s | %-10s | %-8s\n", "Step", "Loss", "Tokens/s", "Time");
    printf("-------|----------|------------|--------\n");

    double start_time = get_time();
    double last_time = start_time;
    float running_loss = -1.0f;

    uint32_t *batch_buf = (uint32_t*)malloc(batch_size * (SEQ_LEN + 1) * sizeof(uint32_t));

    for (int step = 0; step < steps; step++) {
        /* Construct Batch */
        for (int b = 0; b < batch_size; b++) {
            size_t offset = rand() % (dl.n_tokens - SEQ_LEN - 1);
            memcpy(&batch_buf[b * (SEQ_LEN + 1)], &dl.tokens[offset], (SEQ_LEN + 1) * sizeof(uint32_t));
        }

        float loss = kmamba_train_batch(m, batch_buf, batch_size);

        if (running_loss < 0) running_loss = loss;
        else running_loss = 0.95f * running_loss + 0.05f * loss;

        if (step % 20 == 0 || step == steps - 1) {
            double now = get_time();
            double dt = now - last_time;
            if (step == 0) dt = now - start_time;
            double tps = (20.0 * batch_size * SEQ_LEN) / dt;
            
            printf("%-6d | %-8.4f | %-10.2f | %-6.1fs\n", 
                   step, running_loss, tps, now - start_time);
            last_time = now;
        }

        if (step > 0 && step % 500 == 0) {
            kmamba_save_ser(m, SAVE_PATH, KSER_FP32);
        }
    }

    kmamba_save_ser(m, SAVE_PATH, KSER_FP32);
    printf("\n✅ GPU Training Finished. Model: %s\n", SAVE_PATH);

    free(batch_buf);
    kmamba_free_tokens(dl.tokens, dl.n_tokens);
    kmamba_free(m);
    return 0;
}
