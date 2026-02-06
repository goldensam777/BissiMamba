# Mamba Implementation Guide

## Quick Start

### Building
```bash
make                # Build all executables
make run            # Build and run basic demo
make run-advanced   # Build and run advanced examples
make clean          # Clean build artifacts
```

### First Program
```c
#include "mamba.h"

int main() {
    MambaConfig config = {
        .dim = 64,
        .state_size = 16,
        .seq_len = 10,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
    };
    
    MambaBlock *mamba = mamba_block_create(&config);
    mamba_block_init(mamba);
    
    // Allocate input/output (batch_size=1)
    real_t *input = malloc(1 * 10 * 64 * sizeof(real_t));
    real_t *output = malloc(1 * 10 * 64 * sizeof(real_t));
    
    // Forward pass
    mamba_forward(mamba, output, input, 1);
    
    // Cleanup
    free(input);
    free(output);
    mamba_block_free(mamba);
    
    return 0;
}
```

## Architecture Overview

### State Space Model (SSM)

A continuous SSM describes dynamics:
```
ẋ(t) = A·x(t) + B·u(t)
y(t) = C·x(t) + D·u(t)
```

Where:
- `x(t)` is the hidden state (dimension N)
- `u(t)` is the input
- `y(t)` is the output

### Discretization

For numerical computation, we discretize with step size Δt:
```
x[n] = Ā[n]·x[n-1] + B̄[n]·u[n]
y[n] = C·x[n] + D·u[n]
```

Discretized matrices:
- `Ā = exp(Δt·A)` (state transition)
- `B̄ = (Δt·A)^(-1)·(exp(Δt·A) - I)·B` (input mapping)

### Selective Scan (Key Innovation)

The core Mamba operation:

1. **Input projection**: Transform sequence into state space
2. **Parallel scan**: Apply state transitions with data-dependent Δt values
3. **Output computation**: Linear readout from states

This enables:
- ✅ Linear time complexity O(seq_len × dim)
- ✅ Constant memory inference O(state_size)
- ✅ Data-dependent state dynamics
- ✅ Efficient parallel computation

## Data Layout

### Tensor Shapes

- **Input**: `(batch, seq_len, dim)` - flattened as `(batch * seq_len * dim,)`
- **Output**: `(batch, seq_len, dim)` - flattened as `(batch * seq_len * dim,)`
- **Hidden state**: `(state_size,)` - internal use

### Memory Layout

Row-major (C-style) for 2D data:
```
data[i * cols + j] = element at row i, col j
```

Example for sequence of length 3, dim=2:
```
[t=0, d=0] [t=0, d=1] [t=1, d=0] [t=1, d=1] [t=2, d=0] [t=2, d=1]
```

## Configuration Parameters

### MambaConfig Fields

```c
size_t dim;         // Model dimension (input/output size)
size_t state_size;  // Hidden state dimension (N in SSM)
size_t seq_len;     // Sequence length to process
real_t dt_rank;     // Rank for delta time projection
real_t dt_scale;    // Scaling factor for delta times
real_t dt_init;     // Initialization value for delta
real_t dt_min;      // Minimum delta time (stability)
real_t dt_max;      // Maximum delta time (stability)
```

### Recommended Values

**Small model** (mobile/embedded):
- `dim=32, state_size=8, dt_scale=0.5`

**Medium model** (standard use):
- `dim=64, state_size=16, dt_scale=1.0`

**Large model** (high capacity):
- `dim=256, state_size=64, dt_scale=2.0`

## Implementation Details

### Matrix Operations

**matrix_create/matrix_free**: Allocate/deallocate matrices
```c
Matrix *A = matrix_create(16, 16);
matrix_free(A);
```

**matrix_vec_mult**: y = M·v
```c
real_t out[16];
matrix_vec_mult(out, &A, v);
```

**vec_add**: y += x (in-place addition)
**vec_scale**: v *= α (in-place scaling)

### Activation Functions

- **softplus**: log(1 + exp(x)) - smooth ReLU
- **sigmoid**: 1 / (1 + exp(-x)) - gating
- **relu**: max(x, 0) - sparse activations

### Discretization

**discretize_A**: Compute Ā = exp(Δt·A)
- Uses exponential for stable eigenvalues

**discretize_B**: Compute B̄ = Δt·B
- Simplified version for efficiency

### Selective Scan

Core algorithm in `selective_scan()`:
1. Initialize state to zero
2. For each timestep:
   - Discretize A and B with dt_t
   - Update: x_t = Ā·x_{t-1} + B̄·u_t
   - Output: y_t = C·x_t + D·u_t
3. Return sequence of outputs

### Forward Pass

`mamba_forward()` workflow:
1. Project input: u = SiLU(W_in @ x)
2. Generate delta times for each step
3. Run selective scan
4. Project output: y = W_out @ scan_output

## Example: Training Loop Integration

```c
/* Forward pass */
mamba_forward(mamba, output, input, batch_size);

/* Compute loss (implement your metric) */
real_t loss = compute_loss(output, target);

/* In practice, you would:
 * 1. Compute gradients (backpropagation)
 * 2. Update parameters using optimizer (SGD, Adam, etc.)
 * 3. Repeat for multiple epochs
 */
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Forward pass | O(seq_len × dim × state_size) |
| Selective scan | O(seq_len × state_size) |
| Projection | O(seq_len × dim²) |

### Space Complexity

| Data | Size |
|------|------|
| Parameters | O(dim² + state_size) |
| Intermediate | O(seq_len × dim) |
| Inference state | O(state_size) |

### Comparison with Transformers

| Aspect | Mamba | Transformer |
|--------|-------|-------------|
| Time | O(seq_len) | O(seq_len²) |
| Memory | O(1) inference | O(seq_len) |
| Parallelization | Limited | Full |
| Modeling | Good | Better |

## Debugging Tips

### Verify Dimensions

```c
// Check matrix dimensions match expectations
printf("A matrix: %zu x %zu\n", mamba->A_log.rows, mamba->A_log.cols);
printf("Input size: %zu\n", config.seq_len * config.dim);
```

### Monitor State

```c
// Print hidden state after processing
for (size_t i = 0; i < config.state_size; i++) {
    printf("hidden[%zu] = %f\n", i, mamba->hidden[i]);
}
```

### Check for Numerical Issues

```c
// Look for NaN or infinity
for (size_t i = 0; i < output_size; i++) {
    if (!isfinite(output[i])) {
        printf("Non-finite value at position %zu\n", i);
    }
}
```

### Use Matrix Print

```c
matrix_print(&mamba->A_log);
matrix_print(&mamba->B_mat);
```

## Extending the Implementation

### Add Bidirectional Processing

```c
// Process forward
mamba_forward(mamba_fwd, output_fwd, input, batch_size);

// Process backward
reverse_sequence(input_rev, input, seq_len, dim);
mamba_forward(mamba_bwd, output_bwd, input_rev, batch_size);
reverse_sequence(output_bwd_rev, output_bwd, seq_len, dim);

// Combine
for (i = 0; i < output_size; i++) {
    output[i] = output_fwd[i] + output_bwd_rev[i];
}
```

### Add Attention Mechanism

```c
// Compute attention scores
real_t *scores = malloc(seq_len * seq_len * sizeof(real_t));
matrix_mult(scores, query, key_T);

// Apply softmax and combine with values
// ... implement attention ...

// Combine with SSM output
for (i = 0; i < output_size; i++) {
    output[i] = ssm_output[i] + attention_output[i];
}
```

### Implement Gradient Computation

```c
// Backward pass through selective scan
void selective_scan_backward(real_t *grad_input, real_t *grad_state,
                             real_t *grad_output, ...);

// Gradient w.r.t. parameters
void compute_param_gradients(real_t *grad_A, real_t *grad_B, ...);
```

## Common Issues

### Issue: Output is all zeros
- Check if delta times are properly computed
- Verify B̄ values are non-zero
- Check C matrix initialization

### Issue: Unstable (NaN/Inf)
- Reduce dt_max value
- Ensure A matrix is negative definite
- Check input scaling

### Issue: Slow processing
- Reduce state_size for speed
- Use compiler optimizations (-O3)
- Consider GPU acceleration

## References

1. **Mamba Paper**: Gu, A., Goel, K., & Ré, C. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
2. **SSM Theory**: Hipster, T., et al. "State Space Models"
3. **Numerical Methods**: Numerical recipes for discretization

## Support

For issues, questions, or improvements:
- Review the example files
- Check numerical stability
- Test with smaller configurations first
