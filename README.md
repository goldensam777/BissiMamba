# Mamba State Space Model - C Implementation

A complete implementation of the Mamba state space model architecture in C, including core SSM mathematics, matrix operations, and the selective scan mechanism that makes Mamba efficient.

## Overview

Mamba is a recent advancement in sequence modeling that combines the efficiency of RNNs with the modeling capacity of Transformers. It's based on state space models (SSMs) and features:

- **Selective scan operation**: A parallel algorithm that adapts the dynamics based on input
- **Discretization**: Converts continuous SSMs to discrete-time operations
- **Memory efficiency**: Linear scaling with sequence length (vs. quadratic for Transformers)

## Project Structure

```
BissiMamba/
├── mamba.h           # Header file with public API and data structures
├── mamba.c           # Core implementation of Mamba algorithm
├── main.c            # Example/test program
├── Makefile          # Build configuration
└── README.md         # This file
```

## Key Components

### Data Structures

#### `Matrix`
Represents a 2D matrix with:
- `data`: Flattened array (row-major order)
- `rows`, `cols`: Dimensions

#### `SSMParams`
State space model parameters:
- `A`: State transition matrix (N × N)
- `B`: Input matrix (N × 1)
- `C`: Output matrix (1 × N)
- `D`: Feedthrough term

#### `MambaConfig`
Configuration for Mamba block:
- `dim`: Model dimension
- `state_size`: State space dimension (N)
- `seq_len`: Sequence length
- `dt_*`: Delta time parameters

#### `MambaBlock`
Main Mamba block state:
- Projection matrices (W_in, W_out)
- SSM matrices (A_log, B, C)
- Temporary buffers and parameters

### Core Algorithms

#### 1. **State Space Model Fundamentals**

A continuous-time SSM is:
```
ẋ(t) = A·x(t) + B·u(t)
y(t) = C·x(t) + D·u(t)
```

#### 2. **Discretization**

Convert to discrete time with step size Δt:
```
x[n] = Ā·x[n-1] + B̄·u[n]
y[n] = C·x[n] + D·u[n]
```

Where:
- `Ā = exp(Δt·A)`
- `B̄ = (Δt·A)^(-1)·(exp(Δt·A) - I)·B`

#### 3. **Selective Scan**

The core Mamba operation:
- For each timestep, adapt the state dynamics based on delta time (Δt)
- Apply state transition: `x[n] = Ā[n]·x[n-1] + B̄[n]·u[n]`
- Compute output: `y[n] = C·x[n] + D·u[n]`

This enables data-dependent behavior while maintaining computational efficiency.

## API Functions

### Matrix Operations

```c
/* Create and manage matrices */
Matrix* matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *m);
void matrix_zero(Matrix *m);
void matrix_copy(Matrix *dst, const Matrix *src);
void matrix_print(const Matrix *m);
```

### Vector Operations

```c
/* Vector-matrix and vector-vector operations */
void matrix_vec_mult(real_t *out, const Matrix *m, const real_t *v);
void vec_add(real_t *y, const real_t *x, size_t n);
void vec_scale(real_t *v, real_t alpha, size_t n);
```

### Activation Functions

```c
real_t softplus(real_t x);  /* log(1 + exp(x)) */
real_t sigmoid(real_t x);   /* 1 / (1 + exp(-x)) */
real_t relu(real_t x);      /* max(x, 0) */
```

### Mamba Block

```c
/* Create and manage Mamba block */
MambaBlock* mamba_block_create(const MambaConfig *config);
void mamba_block_free(MambaBlock *block);
void mamba_block_init(MambaBlock *block);

/* Forward pass */
void mamba_forward(MambaBlock *block, real_t *output, 
                   const real_t *input, size_t batch_size);
```

### SSM Operations

```c
/* Discretization functions */
void discretize_A(Matrix *A_bar, const Matrix *A, real_t dt);
void discretize_B(real_t *B_bar, const Matrix *A, const real_t *B, 
                  real_t dt, size_t state_size);

/* Core selective scan */
void selective_scan(real_t *output, real_t *state, 
                   const real_t *input, const real_t *delta,
                   const Matrix *A_bar, const real_t *B_bar,
                   const Matrix *C, real_t D,
                   size_t seq_len, size_t state_size);

/* Delta time computation */
void compute_delta(real_t *delta_out, const MambaBlock *block, 
                   const real_t *delta_in, size_t seq_len);
```

## Building and Running

### Prerequisites

- GCC or compatible C compiler
- Standard C library with math support
- Make (for building)

### Compilation

```bash
# Build the project
make

# Build and run
make run

# Clean build artifacts
make clean

# Rebuild from scratch
make rebuild
```

### Building Manually

```bash
gcc -Wall -Wextra -O2 -std=c99 -lm mamba.c main.c -o mamba_demo
./mamba_demo
```

## Example Usage

```c
#include "mamba.h"

int main() {
    /* Configure Mamba block */
    MambaConfig config = {
        .dim = 64,
        .state_size = 16,
        .seq_len = 10,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        /* ... other parameters ... */
    };
    
    /* Create block */
    MambaBlock *mamba = mamba_block_create(&config);
    mamba_block_init(mamba);
    
    /* Prepare input (batch_size=1) */
    real_t *input = malloc(1 * 10 * 64 * sizeof(real_t));
    real_t *output = malloc(1 * 10 * 64 * sizeof(real_t));
    
    /* Fill input with data... */
    
    /* Run forward pass */
    mamba_forward(mamba, output, input, 1);
    
    /* Process output... */
    
    /* Cleanup */
    free(input);
    free(output);
    mamba_block_free(mamba);
    
    return 0;
}
```

## Implementation Notes

### Simplifications for Educational Purpose

This implementation is designed to be understandable and educational. Production implementations would include:

1. **Optimized discretization**: More sophisticated matrix exponential approximations (Padé approximation, eigendecomposition)
2. **Batched operations**: Vectorized computations for better cache utilization
3. **GPU acceleration**: CUDA/HIP implementations for parallel scan
4. **Mixed precision**: FP16 for efficiency while maintaining accuracy
5. **Gradient computation**: Backpropagation for training
6. **Advanced features**: Convolution modes, attention-like mechanisms

### Numerical Stability

The implementation includes several stabilization techniques:

1. **Softplus for delta times**: Ensures positive time steps
2. **Clamping**: Delta times bounded to valid range
3. **Matrix logarithm for A**: Improves numerical stability
4. **Overflow/underflow handling**: In softplus and sigmoid

## Performance Characteristics

### Computational Complexity

- **Forward pass**: O(seq_len × dim × state_size)
- **Selective scan**: O(seq_len × state_size) - parallelizable
- **Memory**: O(dim + state_size × seq_len)

### Advantages over Transformers

- **Linear time complexity** in sequence length (vs. quadratic)
- **Constant memory** for inference (vs. linear in seq_len)
- **Recurrent nature** allows efficient streaming

## References

- Gu, A., Goel, K., & Ré, C. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Organized state space theory and diagonal SSMs for efficient computing

## Future Enhancements

- [ ] Multi-head Mamba (parallel state space tracks)
- [ ] Bidirectional processing
- [ ] Training with backpropagation
- [ ] GPU acceleration (CUDA)
- [ ] Benchmarking against Transformers
- [ ] Integration with other neural network layers
- [ ] Quantization for mobile deployment

## License

This implementation is provided for educational purposes.

## Contributing

Feel free to extend this implementation with:
- Additional features from the original Mamba paper
- Performance optimizations
- GPU implementations
- Training infrastructure
- Benchmarking tools

## Contact

For questions or improvements, please create an issue or pull request.
