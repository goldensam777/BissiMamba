#ifndef MAMBA_H
#define MAMBA_H

#include <stddef.h>
#include <stdint.h>

/* Type definitions */
typedef float real_t;

/* Matrix structure */
typedef struct {
    real_t *data;
    size_t rows;
    size_t cols;
} Matrix;

/* State space model parameters */
typedef struct {
    Matrix A;           /* State transition matrix (N x N) */
    Matrix B;           /* Input vector (N x 1) */
    Matrix C;           /* Output vector (N x 1) (used as readout weights on state) */
    real_t D;           /* Feedthrough term */
} SSMParams;

/* Mamba block configuration */
typedef struct {
    size_t dim;         /* Model dimension */
    size_t state_size;  /* State space dimension (N) */
    size_t seq_len;     /* Sequence length */
    real_t dt_rank;     /* Delta time rank */
    real_t dt_scale;    /* Delta time scale */
    real_t dt_init;     /* Delta time initialization */
    real_t dt_min;      /* Minimum delta time */
    real_t dt_max;      /* Maximum delta time */
} MambaConfig;

/* Mamba block state */
typedef struct {
    MambaConfig config;
    
    /* Projections */
    /* `W_in` maps input vectors of size `dim` -> controller vectors of size `state_size` (rows=state_size, cols=dim) */
    Matrix W_in;        /* Input projection (state_size x dim) */
    /* `W_out` maps state vectors (state_size) -> output vectors of size `dim` (rows=dim, cols=state_size) */
    Matrix W_out;       /* Output projection (dim x state_size) */
    
    /* State-dependent matrices */
    Matrix A_log;       /* Log of state matrix diagonal */
    Matrix B_mat;       /* Input matrix */
    Matrix C_mat;       /* Output matrix */
    
    /* Delta time parameters */
    /* `delta_proj` projects input -> scalar delta (shape: 1 x dim or dim x 1 depending on allocation) */
    Matrix delta_proj;  /* Delta time projection */
    
    /* Temporary buffers */
    real_t *hidden;     /* Hidden state (state_size) */
    real_t *delta;      /* Delta time values (seq_len) */

    /* Pre-allocated scan1d adapter buffers — évite malloc par forward */
    real_t *scan_B;     /* [seq_len, state_size] — B_mat répliqué */
    real_t *scan_C;     /* [seq_len, state_size] — ones (readout neutre) */
    real_t *scan_delta; /* [seq_len, state_size] — delta[t] répliqué */
    real_t *scan_h;     /* [state_size]          — état caché pour scan1d */
} MambaBlock;

/* Gradient and optimizer state held per-block for CPU training */
typedef struct {
    /* Gradients */
    real_t *g_W_in;    /* state_size x dim (row-major: rows=state_size, cols=dim) */
    real_t *g_W_out;   /* dim x state_size (row-major: rows=dim, cols=state_size) */
    real_t *g_A_log;   /* state_size */
    real_t *g_B_mat;   /* state_size */
    real_t *g_C_mat;   /* state_size */
    real_t *g_delta_proj; /* dim */

    /* Optimizer moments (MUONCLIP) */
    real_t *m_W_in;    real_t *v_W_in;
    real_t *m_W_out;   real_t *v_W_out;
    real_t *m_A_log;   real_t *v_A_log;
    real_t *m_B_mat;   real_t *v_B_mat;
    real_t *m_C_mat;   real_t *v_C_mat;
    real_t *m_delta_proj; real_t *v_delta_proj;

    /* optimizer step counter */
    size_t step;
} OptimState;

/* Optimizer configuration for MUONCLIP */
typedef struct {
    real_t lr;
    real_t mu;      /* momentum for first moment */
    real_t beta2;   /* second moment decay */
    real_t eps;
    real_t clip_norm;
    real_t weight_decay;
} OptimConfig;

/* Attach optimizer state to mamba block */
void mamba_attach_optimizer(MambaBlock *block, const OptimConfig *optconf);
void mamba_free_optimizer(MambaBlock *block);

/* Zero gradients */
void mamba_zero_grads(MambaBlock *block);

/* Backward pass: dY is scalar per timestep (seq_len), input is original inputs, u_seq is projected inputs used in forward */
void mamba_backward(MambaBlock *block, const real_t *dY, const real_t *input, size_t batch_index);

/* Optimizer step: apply MUONCLIP updates to parameters */
void mamba_optimizer_step(MambaBlock *block, const OptimConfig *conf);

/* Function prototypes */

/* Matrix operations */
Matrix* matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *m);
void matrix_zero(Matrix *m);
void matrix_copy(Matrix *dst, const Matrix *src);
void matrix_print(const Matrix *m);

/* Matrix-vector operations */
void matrix_vec_mult(real_t *out, const Matrix *m, const real_t *v);
void vec_add(real_t *y, const real_t *x, size_t n);
void vec_scale(real_t *v, real_t alpha, size_t n);

/* Mamba operations */
MambaBlock* mamba_block_create(const MambaConfig *config);
void mamba_block_free(MambaBlock *block);

/* Initialize Mamba block with standard parameters */
void mamba_block_init(MambaBlock *block);

/* Forward pass through Mamba block
 * input: (seq_len, dim) flattened in row-major order
 * output: (seq_len, dim) flattened in row-major order
 */
void mamba_forward(MambaBlock *block, real_t *output, const real_t *input, size_t batch_size);

/* Forward pass 2D — entree grille [d1, d2, dim], sortie [d1, d2, dim] */
void mamba_forward_2d(MambaBlock *block, real_t *output, const real_t *input,
                      size_t d1, size_t d2);

/* Backward pass 2D — gradient de la loss sur la sortie [d1, d2, dim] */
void mamba_backward_2d(MambaBlock *block, const real_t *dY, const real_t *input,
                       size_t d1, size_t d2);

/* Compute delta time values for each timestep
 * delta_out: (seq_len) vector of delta values
 * delta_in: (seq_len, 1) or similar projection input
 */
void compute_delta(real_t *delta_out, const MambaBlock *block, 
                   const real_t *delta_in, size_t seq_len);

/* Compute discretized A matrix: A_bar = exp(dt * A)
 * Uses matrix exponential approximation
 */
void discretize_A(Matrix *A_bar, const Matrix *A, real_t dt);

/* Compute discretized B vector: B_bar = (dt * A)^(-1) * (exp(dt * A) - I) * B
 */
void discretize_B(real_t *B_bar, const Matrix *A, const real_t *B, real_t dt, size_t state_size);

/* Selective scan operation - core of Mamba
 * Performs parallel scan over sequence with element-wise operations
 */
void selective_scan(real_t *output, real_t *state, 
                   const real_t *input, const real_t *delta,
                   const Matrix *A_bar, const real_t *B_bar,
                   const Matrix *C, real_t D,
                   size_t seq_len, size_t state_size);

/* Softplus activation: log(1 + exp(x)) */
real_t softplus(real_t x);

/* Sigmoid activation */
real_t sigmoid(real_t x);

/* ReLU activation */
real_t relu(real_t x);

#endif /* MAMBA_H */
