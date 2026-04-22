
static void _ssm_scan_backward(MambaBlock *block, MambaBlockWorkspace *ws,
                               const float *u_seq, const float *lambda_seq,
                               const float *d_y_rank) {
    size_t L = block->config.seq_len, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    MBOptimState *grads = block->opt_state;
    
    float *d_h = (float *)calloc(N, sizeof(float));
    float *d_Bu_cur = (float *)calloc(N, sizeof(float));
    float *d_Bu_next = (float *)calloc(N, sizeof(float));

    /* Zero intermediate gradients in workspace */
    memset(ws->d_scan_B, 0, L * NR * sizeof(float));
    memset(ws->d_scan_C, 0, L * NR * sizeof(float));
    memset(ws->d_delta, 0, L * sizeof(float));
    memset(ws->d_lambda, 0, L * sizeof(float));

    /* Reverse scan */
    for (long t = (long)L - 1; t >= 0; t--) {
        float dt_t = ws->delta[t];
        float lam_t = lambda_seq ? lambda_seq[t] : 0.5f;
        const float *c_t = &ws->scan_C[t * NR];
        const float *h_t = &ws->h_seq[t * N];
        const float *h_prev = (t > 0) ? &ws->h_seq[(t - 1) * N] : NULL;
        const float *prev_Bu = (t > 0) ? &ws->Bu_seq[(t - 1) * N] : NULL;
        
        /* 1. d_h += C_t^T @ d_y_rank_t */
        for (size_t r = 0; r < R; r++) {
            float dy_tr = d_y_rank[t * R + r];
            for (size_t n = 0; n < N; n++) {
                d_h[n] += c_t[r * N + n] * dy_tr;
                /* d_scan_C gradient */
                ws->d_scan_C[t * NR + r * N + n] += h_t[n] * dy_tr;
            }
        }

        /* 2. Mamba-3 recurrence: h_t = alpha*R(theta)*h_{t-1} + beta*Bu_{t-1} + gamma*Bu_t */
        for (size_t n = 0; n < N; n++) {
            float a = block->A_log.data[n];
            float alpha = expf(dt_t * a);
            float beta  = (1.0f - lam_t) * dt_t * alpha;
            float gamma = lam_t * dt_t;

            float dh_n = d_h[n];
            d_Bu_cur[n] += dh_n * gamma;
            if (t > 0) d_Bu_next[n] = dh_n * beta; 
            
            /* Gradient for A_log */
            if (h_prev) {
                float h_rot_n;
                if (n & 1) {
                    size_t i = n - 1;
                    float th = block->theta[i >> 1];
                    h_rot_n = sinf(th) * h_prev[i] + cosf(th) * h_prev[i+1];
                } else {
                    size_t i = n;
                    float th = block->theta[i >> 1];
                    h_rot_n = cosf(th) * h_prev[i] - sinf(th) * h_prev[i+1];
                }
                float da = dh_n * (dt_t * alpha * h_rot_n + (1.0f - lam_t) * dt_t * dt_t * alpha * (prev_Bu ? prev_Bu[n] : 0.0f));
                grads->g_A_log[n] += da;
                
                /* Gradient for delta_t (simplified: only through A and Bu terms) */
                ws->d_delta[t] += dh_n * (a * alpha * h_rot_n + (1.0f - lam_t) * (alpha + dt_t * a * alpha) * (prev_Bu ? prev_Bu[n] : 0.0f) + lam_t * ws->Bu_seq[t * N + n]);
            } else {
                /* t=0: h_t = gamma * Bu_t */
                ws->d_delta[t] += dh_n * lam_t * ws->Bu_seq[t * N + n];
            }
            
            /* Gradient for lambda_t */
            if (prev_Bu) {
                ws->d_lambda[t] += dh_n * (-dt_t * alpha * prev_Bu[n] + dt_t * ws->Bu_seq[t * N + n]);
            } else {
                ws->d_lambda[t] += dh_n * dt_t * ws->Bu_seq[t * N + n];
            }

            /* Update d_h for next step (t-1) : it becomes d(R(theta)*h_prev) */
            d_h[n] = dh_n * alpha;
        }

        /* 3. Backward through R(theta) */
        for (size_t i = 0; i + 1 < N; i += 2) {
            float th = block->theta[i >> 1];
            float cv = cosf(th), sv = sinf(th);
            float dh0 = d_h[i], dh1 = d_h[i+1];
            float h_prev0 = (t > 0) ? ws->h_seq[(t-1)*N + i] : 0.0f;
            float h_prev1 = (t > 0) ? ws->h_seq[(t-1)*N + i+1] : 0.0f;

            float new_dh0 = cv * dh0 + sv * dh1;
            float new_dh1 = -sv * dh0 + cv * dh1;

            if (t > 0) {
                grads->g_theta[i >> 1] += ((-sv * h_prev0 - cv * h_prev1) * dh0 +
                                           (cv * h_prev0 - sv * h_prev1) * dh1);
            }
            d_h[i] = new_dh0;
            d_h[i+1] = new_dh1;
        }

        /* 4. d_Bu_cur -> d_scan_B and d_u_seq (implicitly through Bu_cur = sum_r B_t * u_t) */
        const float *u_t = &u_seq[t * R];
        for (size_t r = 0; r < R; r++) {
            float ut_r = u_t[r];
            for (size_t n = 0; n < N; n++) {
                ws->d_scan_B[t * NR + r * N + n] += d_Bu_cur[n] * ut_r;
            }
        }
        
        /* Prepare for t-1 */
        for(size_t n=0; n<N; n++) {
            d_Bu_cur[n] = d_Bu_next[n];
            d_Bu_next[n] = 0.0f;
        }
    }
    free(d_h); free(d_Bu_cur); free(d_Bu_next);
}

void mamba_backward_ws(MambaBlock *block, MambaBlockWorkspace *ws,
                       const float *dY, const float *input,
                       float *d_input, size_t batch_index) {
    (void)batch_index;
    if (!block || !ws || !dY || !input || !d_input) return;
    if (!block->opt_state) return;

    size_t L = block->config.seq_len, D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    MBOptimState *g = block->opt_state;

    /* 1. Output projection backward */
    gemm_f32_ABt(dY, block->W_out.data, ws->d_y_rank, (int)L, (int)D, (int)R);
    gemm_f32_AtB(dY, ws->y_rank, g->g_W_out, (int)D, (int)L, (int)R);

    /* 2. Selective Scan backward */
    _ssm_scan_backward(block, ws, ws->u_seq, ws->lambda_seq, ws->d_y_rank);

    /* 3. Backward through input-dependent parameters */
    for (size_t t = 0; t < L; t++) {
        const float *in_t = &input[t * D];
        
        /* d_scan_B and d_scan_C -> g_W_B, g_W_C, g_b_B, g_b_C */
        for (size_t i = 0; i < NR; i++) {
            float dsb = ws->d_scan_B[t * NR + i];
            float dsc = ws->d_scan_C[t * NR + i];
            g->g_b_B[i] += dsb;
            g->g_b_C[i] += dsc;
            for (size_t d = 0; d < D; d++) {
                g->g_W_B[i * D + d] += dsb * in_t[d];
                g->g_W_C[i * D + d] += dsc * in_t[d];
            }
        }

        /* d_delta -> g_delta_proj and d_input */
        /* Note: simplified backward for softplus and clamp */
        float dd = ws->d_delta[t];
        if (ws->delta[t] > block->config.dt_min && ws->delta[t] < block->config.dt_max) {
             /* gradient of softplus(x) is sigmoid(x). We approximate here. */
             float d_raw = dd * 0.5f; 
             for (size_t d = 0; d < D; d++) g->g_delta_proj[d] += d_raw * in_t[d];
        }

        /* d_lambda -> g_lambda_proj */
        float dl = ws->d_lambda[t];
        float l_val = ws->lambda_seq[t];
        float dl_raw = dl * l_val * (1.0f - l_val); /* sigmoid backward */
        for (size_t d = 0; d < D; d++) g->g_lambda_proj[d] += dl_raw * in_t[d];
    }

    /* 4. Backward through W_in */
    /* u = silu(W_in @ in) */
    /* d_u is implicitly needed but we can compute it for each t */
    /* This is simplified for now. */

    /* d_input residual */
    for (size_t i = 0; i < L * D; i++) d_input[i] += dY[i];
}
