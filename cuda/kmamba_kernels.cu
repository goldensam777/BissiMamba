#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>

/* Kernel: Embedding Forward */
__global__ void embedding_fwd_kernel(const float *embed, const uint32_t *tokens, float *out, int L, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    
    uint32_t token = tokens[t];
    for (int d = 0; d < D; d++) {
        out[t * D + d] = embed[token * D + d];
    }
}

extern "C" void cuda_embedding_forward(const float *d_embed, const uint32_t *d_tokens, float *d_out, int L, int D) {
    int threads = 256;
    int blocks = (L + threads - 1) / threads;
    embedding_fwd_kernel<<<blocks, threads>>>(d_embed, d_tokens, d_out, L, D);
}

/* Head Forward: Logits = Hidden @ Head^T */
extern "C" void cuda_head_forward(cublasHandle_t handle, const float *d_head, const float *d_hidden, float *d_logits, int L, int D, int V) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    /* Head is [V, D], Hidden is [L, D], Logits is [L, V] */
    /* op(A) @ op(B) = C -> Hidden @ Head^T */
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                V, L, D, 
                &alpha, 
                d_head, D, 
                d_hidden, D, 
                &beta, 
                d_logits, V);
}

/* Kernel: Softmax + CrossEntropy Loss + dLogits */
__global__ void softmax_loss_kernel(const float *logits, const uint32_t *targets, float *loss, float *dlogits, int L, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    
    const float *l_t = &logits[t * V];
    float *dl_t = &dlogits[t * V];
    uint32_t target = targets[t];
    
    /* Max for numerical stability */
    float max_val = -1e30f;
    for (int v = 0; v < V; v++) if (l_t[v] > max_val) max_val = l_t[v];
    
    /* Sum of exp */
    float sum_exp = 0.0f;
    for (int v = 0; v < V; v++) {
        float e = expf(l_t[v] - max_val);
        dl_t[v] = e; /* temporary */
        sum_exp += e;
    }
    
    float inv_sum = 1.0f / sum_exp;
    float log_prob = l_t[target] - max_val - logf(sum_exp);
    
    atomicAdd(loss, -log_prob);
    
    /* Gradients: p_v - delta_{v,target} */
    for (int v = 0; v < V; v++) {
        float prob = dl_t[v] * inv_sum;
        dl_t[v] = prob - (v == target ? 1.0f : 0.0f);
    }
}

extern "C" void cuda_softmax_loss_kernel(const float *d_logits, const uint32_t *d_targets, float *d_loss, float *d_dlogits, int L, int V) {
    int threads = 256;
    int blocks = (L + threads - 1) / threads;
    cudaMemset(d_loss, 0, sizeof(float));
    softmax_loss_kernel<<<blocks, threads>>>(d_logits, d_targets, d_loss, d_dlogits, L, V);
}
