# Mamba Mathematical Foundations

## State Space Models (SSMs)

### Continuous-Time Formulation

A linear time-invariant (LTI) state space model:

```
ẋ(t) = A·x(t) + B·u(t)                    [State equation]
y(t) = C·x(t) + D·u(t)                    [Output equation]
```

Where:
- `x(t) ∈ ℝᴺ`: State vector
- `u(t) ∈ ℝ`: Input signal
- `y(t) ∈ ℝ`: Output signal
- `A ∈ ℝᴺˣᴺ`: State transition matrix
- `B ∈ ℝᴺ`: Input matrix
- `C ∈ ℝ¹ˣᴺ`: Output matrix
- `D ∈ ℝ`: Feedthrough term

### Stability Conditions

For stability, all eigenvalues of A must have **negative real parts**:
- `λᵢ(A) < 0` for all i

This ensures state doesn't grow unbounded.

### Solution

Using the matrix exponential:

```
x(t) = eᴬᵗ·x₀ + ∫₀ᵗ eᴬ⁽ᵗ⁻ˢ⁾·B·u(s) ds
```

## Discretization

### Zero-Order Hold (ZOH)

Given sampling period Δt, discretize using ZOH:

```
x[n+1] = Ā·x[n] + B̄·u[n]
y[n] = C·x[n] + D·u[n]
```

### Discretized Matrices

**State transition (closed form)**:
```
Ā = eᴬᐃᵗ
```

**Input mapping (exact)**:
```
B̄ = ∫₀ᐃᵗ eᴬˢ ds · B = A⁻¹·(eᴬᐃᵗ - I)·B
```

### Numerical Computation

For implementation:

1. **Eigenvalue decomposition** (general A):
   ```
   A = Q·Λ·Q⁻¹
   eᴬᐃᵗ = Q·eᴧᐃᵗ·Q⁻¹
   ```

2. **Diagonal case** (efficient):
   ```
   If A is diagonal with elements aᵢᵢ:
   eᴬᐃᵗ[i,i] = eᵃⁱⁱᐃᵗ
   ```

3. **Scaling and squaring** (numerically stable):
   - Scale: compute e^(A·Δt/2ᵏ)
   - Square: repeatedly square result k times

## Selective Scan Operation

### Algorithm

The selective scan processes sequence {u[0], u[1], ..., u[T-1]}:

```
Algorithm: Selective Scan
─────────────────────────────────────────────────
Input: u[0..T-1], A, B, C, D
       Δt[0..T-1] (variable time steps)
       x₀ (initial state)

Initialize: x = x₀

For t = 0 to T-1:
    // Discretize with time-varying Δt[t]
    Ā_t = exp(Δt[t] · A)
    B̄_t = (Δt[t] · A)⁻¹ · (Ā_t - I) · B
    
    // Update state
    x ← Ā_t · x + B̄_t · u[t]
    
    // Compute output
    y[t] ← C · x + D · u[t]

Return: y[0..T-1]
─────────────────────────────────────────────────
```

### Key Properties

1. **Parallelizable**: Scan operation admits parallel computation
   - Associate property: (A⊕B)⊕C = A⊕(B⊕C)
   - Binary tree structure for O(log T) depth

2. **Memory efficient**: 
   - Only requires O(state_size) memory
   - Not O(seq_len × state_size) like attention

3. **Data-dependent dynamics**:
   - Δt values depend on input
   - Enables selective focus on important timesteps

## Mamba Integration

### Full Mamba Block

```
Input: x ∈ ℝ^(T×d) [sequence]
       
Step 1: Input projection with gating
    u = SiLU(W_in · x)  [projection and activation]
    
Step 2: Compute delta times
    Δt = softplus(Δt_proj · x)  [ensure positive]
    Δt = clamp(Δt, dt_min, dt_max)
    
Step 3: Selective scan
    y_ssm = SelectiveScan(u, A, B, C, D, Δt)
    
Step 4: Output projection with residual
    y = W_out · y_ssm
    
Output: y ∈ ℝ^(T×d)
```

### Activation Functions

**SiLU (Swish)**:
```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))
```
- Smooth and differentiable
- Used for gating

**Softplus**:
```
softplus(x) = log(1 + e^x)
```
- Ensures positive delta times
- Smooth approximation of ReLU

**Sigmoid**:
```
σ(x) = 1 / (1 + e^(-x))
```
- Output gating
- Range [0, 1]

## Matrix Operations

### Matrix-Vector Multiplication

```
y = M · v

y[i] = Σⱼ M[i,j] · v[j]
```

**Complexity**: O(m·n) for m×n matrix

### Vector Addition

```
y := y + x

y[i] += x[i] for all i
```

**Complexity**: O(n)

### Matrix Exponential

**Taylor series** (small matrices):
```
eᴬ = I + A + A²/2! + A³/3! + ...
```

**Matrix norm bounds** (stability):
```
‖eᴬ‖ ≤ e^(‖A‖)
```

**Diagonal approximation**:
```
If A = diag(λ₁, ..., λₙ):
eᴬ = diag(e^λ₁, ..., e^λₙ)
```

## Numerical Stability

### Eigenvalue Stability

For discrete SSM to be stable:
```
|λᵢ(Ā)| < 1 for all i
```

Since `Ā = eᴬᐃᵗ`:
```
|e^(λᵢ(A)·Δt)| < 1
⟹ Re(λᵢ(A)) < 0  [A is negative definite]
```

### Condition Number

```
κ(M) = σ_max(M) / σ_min(M)

Large κ ⟹ ill-conditioned ⟹ numerical errors
```

### Numerical Safeguards

1. **Overflow/underflow handling**:
   ```c
   if (x > 20) return max_value;     // e^x overflow
   if (x < -20) return 0;             // e^x underflow
   ```

2. **Scale clipping**:
   ```
   Δt ∈ [dt_min, dt_max]
   Prevents extreme scaling
   ```

3. **Log-space computation**:
   ```
   Store log(A) instead of A
   More numerically stable
   ```

## Complexity Analysis

### Forward Pass Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Input projection | O(T·d²) | O(T·d) |
| Delta computation | O(T·d) | O(T) |
| Selective scan | O(T·n) | O(n) |
| Output projection | O(T·d²) | O(T·d) |
| **Total** | **O(T·d²)** | **O(T·d)** |

Where T = sequence length, d = dimension, n = state size

### Comparison

| Model | Time | Memory (inference) |
|-------|------|-------------------|
| RNN | O(T·d²) | O(d) |
| Transformer | O(T²·d) | O(T·d) |
| Mamba | O(T·d²) | O(d) |

## Optimization Insights

### Input Scaling

Normalize inputs to [-1, 1] for numerical stability:
```
x_normalized = (x - mean) / (std + ε)
```

### Learning Rate Scheduling

Recommended for parameter learning:
```
lr = lr₀ · (1 - t/T)²  [polynomial decay]
```

### Initialization Schemes

**A matrix** (stable eigenvalues):
```
aᵢᵢ ~ U[-10, -0.1]  [log-space]
aᵢⱼ (i≠j) ~ 0 (start with diagonal)
```

**B matrix** (normalized):
```
B[i] ~ N(0, 1/√n)  [normalized random]
```

**C matrix** (normalized):
```
C[i] ~ N(0, 1/√n)  [normalized random]
```

## Advanced Topics

### Variational Inference

For uncertainty quantification:
```
p(y|x) = ∫ p(y|z) p(z|x) dz

Use MC dropout or Bayesian approximation
```

### Multi-scale Processing

Process at different resolutions:
```
y_s = SelectiveScan(u, A_s, B_s, C_s, D_s, Δt)
```

### Hierarchical States

Multiple state space layers:
```
y = W_out · (x⁽ᴸ⁾)
where x⁽ˡ⁾ = SelectiveScan(x⁽ˡ⁻¹⁾, ...)
```

## References

1. **Control Theory**: 
   - Ogata, K. "Modern Control Engineering"
   - Franklin, Powell, Emami-Naeini "Feedback Control"

2. **Numerical Analysis**:
   - Higham, N. "The Scaling and Squaring Method for Computing Matrix Exponentials"
   - Moler, C., & Van Loan, C. F. "Nineteen Dubious Ways to Compute the Exponential of a Matrix"

3. **SSM Theory**:
   - Gu, Albert, et al. "Efficiently Modeling Long Sequences with Structured State Spaces"
   - Orvieto, et al. "Resurrecting Recurrent Neural Networks for Long Sequences"

4. **Mamba**:
   - Gu, A., Goel, K., & Ré, C. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
