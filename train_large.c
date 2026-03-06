/* train_large.c — Training loop for the ~1B-parameter Mamba LM (CUDA).
 *
 * Usage:
 *   ./train_large [dataset] [model_out] [epochs] [--small]
 *
 *   dataset   : path to text file           (default: data/train.txt)
 *   model_out : checkpoint path             (default: large_checkpoint.bin)
 *   epochs    : number of passes            (default: 10)
 *   --small   : use ML_CFG_SMALL (~7 M)     (default: ML_CFG_1B ~1 B)
 *
 * Requires CUDA + cuBLAS.  Build: see Makefile (target train_large).
 *
 * Memory requirements (fp32):
 *   ~1 B model:  weights ≈16 GB  +  Adam moments ≈32 GB  +  activations
 *   → needs GPU with ≥80 GB VRAM (e.g. A100-80G) for batch=1, seq=2048
 *   --small preset fits in ~4 GB VRAM.
 */

#include "mamba_large.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CORPUS  (256 * 1024 * 1024)   /* 256 MB max corpus */
#define DEFAULT_DS  "data/train.txt"
#define DEFAULT_OUT "large_checkpoint.bin"
#define DEFAULT_EP  10
#define CKPT_EVERY  1                      /* checkpoint every N epochs */
#define LOG_EVERY   100                    /* print loss every N windows */

int main(int argc, char *argv[])
{
    const char *ds_path  = DEFAULT_DS;
    const char *out_path = DEFAULT_OUT;
    int         epochs   = DEFAULT_EP;
    int         use_small= 0;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--small") == 0) use_small = 1;
        else if (i == 1) ds_path  = argv[1];
        else if (i == 2) out_path = argv[2];
        else if (i == 3) epochs   = atoi(argv[3]);
    }

    /* ── 1. Load corpus ────────────────────────────────────────────── */
    FILE *fin = fopen(ds_path, "rb");
    if (!fin) {
        fprintf(stderr, "ERROR: cannot open dataset '%s'\n", ds_path);
        fprintf(stderr, "       Run first:  python3 download_data.py\n");
        return 1;
    }
    char *corpus = (char *)malloc(MAX_CORPUS + 1);
    if (!corpus) { fclose(fin); return 1; }
    size_t corpus_len = fread(corpus, 1, MAX_CORPUS, fin);
    corpus[corpus_len] = '\0';
    fclose(fin);
    printf("Corpus: %zu bytes from '%s'\n", corpus_len, ds_path);

    /* ── 2. Choose config ──────────────────────────────────────────── */
    MLConfig cfg = use_small ? ML_CFG_SMALL : ML_CFG_1B;
    printf("Config: %s\n", use_small ? "SMALL (~7M)" : "1B");

    long long nparams = ml_count_params(&cfg);
    printf("Parameters: %lld (%.2f B)\n\n", nparams, (double)nparams / 1e9);

    /* ── 3. Create model ───────────────────────────────────────────── */
    MLModel *model = ml_create(&cfg);
    if (!model) {
        fprintf(stderr, "ERROR: ml_create failed (out of GPU memory?)\n");
        free(corpus);
        return 1;
    }

    /* Try loading existing checkpoint */
    if (ml_load(model, out_path) == 0)
        printf("Resumed from '%s'\n", out_path);

    /* ── 4. Optimizer config ───────────────────────────────────────── */
    MLOptimConfig opt = ML_OPTIM_DEFAULT;
    if (use_small) opt.lr = 3e-4f;   /* faster lr for small model */

    /* ── 5. Training loop ──────────────────────────────────────────── */
    int   seq_len  = cfg.seq_len;
    int  *in_seq   = (int *)malloc((size_t)seq_len * sizeof(int));
    int  *tgt_seq  = (int *)malloc((size_t)seq_len * sizeof(int));
    if (!in_seq || !tgt_seq) {
        fprintf(stderr, "ERROR: OOM for sequence buffers\n");
        ml_free(model); free(corpus); return 1;
    }

    size_t n_windows = (corpus_len > (size_t)(seq_len + 1))
                       ? (corpus_len - 1) / (size_t)seq_len
                       : 0;
    if (n_windows == 0) {
        fprintf(stderr,
            "ERROR: corpus too short — need > %d chars, got %zu.\n"
            "       Run:  python3 download_data.py\n",
            seq_len + 1, corpus_len);
        ml_free(model); free(corpus); free(in_seq); free(tgt_seq);
        return 1;
    }
    printf("Windows/epoch: %zu  (seq_len=%d)\n", n_windows, seq_len);
    printf("Training for %d epoch(s)…\n\n", epochs);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        size_t count      = 0;
        clock_t t0 = clock();

        for (size_t pos = 0; pos + (size_t)seq_len < corpus_len; pos += (size_t)seq_len) {
            for (int t = 0; t < seq_len; t++) {
                in_seq[t]  = (unsigned char)corpus[pos + (size_t)t];
                tgt_seq[t] = (unsigned char)corpus[pos + (size_t)t + 1];
            }

            float loss = ml_train_step(model, in_seq, tgt_seq, &opt);
            epoch_loss += (double)loss;
            count++;

            if (count % LOG_EVERY == 0) {
                double avg  = epoch_loss / (double)count;
                double ppl  = exp(avg);
                double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
                printf("  epoch %2d  step %6zu/%6zu  loss=%.4f  ppl=%.2f  %.1fs\n",
                       epoch, count, n_windows, avg, ppl, secs);
                fflush(stdout);
            }
        }

        double avg_loss = (count > 0) ? epoch_loss / (double)count : 0.0;
        double ppl      = exp(avg_loss);
        double elapsed  = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("Epoch %2d  loss=%.4f  ppl=%.2f  (%.1f s)\n",
               epoch, avg_loss, ppl, elapsed);
        fflush(stdout);

        if ((epoch + 1) % CKPT_EVERY == 0) {
            if (ml_save(model, out_path) == 0)
                printf("  → checkpoint saved: '%s'\n", out_path);
        }
    }

    /* Final save */
    ml_save(model, out_path);
    printf("\nTraining done.  Model: '%s'\n", out_path);

    /* ── 6. Quick generation sample ────────────────────────────────── */
    printf("\n=== Generation sample ===\n");
    printf("You> Hello!\nBot> ");
    fflush(stdout);
    ml_generate(model, "Human: Hello!\nBot: ", 128, 0.8f);

    free(in_seq); free(tgt_seq); free(corpus);
    ml_free(model);
    return 0;
}
