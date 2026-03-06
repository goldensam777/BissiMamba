#!/usr/bin/env python3
"""
train_pytorch.py — Train BissiMamba in PyTorch, export weights for C inference.

Architecture mirrors lm.c / mamba.c exactly so the exported checkpoint
can be loaded by lm_load() and run by chat.c.

Forward per token x_t ∈ R^dim:
  u_t    = SiLU(W_in @ x_t)                           [state_size]
  delta_t = clamp(softplus(delta_proj @ x_t), dt_min, dt_max)  [scalar]
  A_bar_i = exp(A_log_i)                               [static]
  A_t_i  = exp(delta_t * A_bar_i)
  B_bar_i = (A_t_i - 1) / A_bar_i * B_i  (or delta_t*B_i if A≈0)
  h_t    = A_t * h_{t-1} + B_bar * u_t                [state_size]
  y_t    = W_out @ h_t                                  [dim]
  logit_t = head_W @ y_t + head_bias                    [vocab]
"""

import struct
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Config  (must match chat.c exactly)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB      = 128
DIM        = 64
STATE_SIZE = 32
SEQ_LEN    = 128
MAX_GEN    = 256
DT_MIN     = 0.001
DT_MAX     = 0.1

LM_MAGIC   = 0x4C4D3030  # "LM00"

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, dim, state_size, dt_min=DT_MIN, dt_max=DT_MAX):
        super().__init__()
        self.dim        = dim
        self.state_size = state_size
        self.dt_min     = dt_min
        self.dt_max     = dt_max

        self.W_in       = nn.Parameter(torch.empty(state_size, dim))
        self.W_out      = nn.Parameter(torch.empty(dim, state_size))
        self.A_log      = nn.Parameter(torch.empty(state_size))
        self.B_mat      = nn.Parameter(torch.empty(state_size))
        self.C_mat      = nn.Parameter(torch.empty(state_size))  # kept for compat
        self.delta_proj = nn.Parameter(torch.empty(dim))

        self._init_weights()

    def _init_weights(self):
        # Xavier for W_in, W_out (matches C init)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
        # A_log: log of negative eigenvalues → A_bar[i] = exp(A_log[i]) < 0?
        # In C: A_log[i] = -exp(spacing * log(dt_scale)) → negative values
        # We init as small negative values
        nn.init.uniform_(self.A_log, -2.0, -0.5)
        # B, C: 1/sqrt(state_size)
        val = 1.0 / math.sqrt(STATE_SIZE)
        nn.init.constant_(self.B_mat, val)
        nn.init.constant_(self.C_mat, val)
        # delta_proj: small uniform
        nn.init.uniform_(self.delta_proj, -0.01, 0.01)

    def forward(self, x):
        """
        x: [T, dim]  — one sequence (batch=1 for simplicity)
        returns y: [T, dim]
        """
        T   = x.shape[0]
        h   = x.new_zeros(self.state_size)
        ys  = []

        # A_bar: static, exp(A_log)
        A_bar = self.A_log.exp()            # [state_size]

        for t in range(T):
            x_t = x[t]                      # [dim]

            # Controller: u = SiLU(W_in @ x)
            u_t = F.silu(self.W_in @ x_t)  # [state_size]

            # Delta: softplus(delta_proj · x), clamped
            delta_raw = (self.delta_proj * x_t).sum()
            delta_t   = F.softplus(delta_raw).clamp(self.dt_min, self.dt_max)

            # Discretise A and B
            A_t   = (delta_t * A_bar).exp()            # [state_size]
            denom = A_bar.abs().clamp(min=1e-8)
            B_bar = (A_t - 1.0) / denom * self.B_mat   # [state_size]

            # State update
            h = A_t * h + B_bar * u_t                  # [state_size]

            # Output projection
            y_t = self.W_out @ h                        # [dim]
            ys.append(y_t)

        return torch.stack(ys, dim=0)   # [T, dim]


class LM(nn.Module):
    def __init__(self, vocab=VOCAB, dim=DIM, state_size=STATE_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.mamba     = MambaBlock(dim, state_size)
        self.head_W    = nn.Parameter(torch.empty(vocab, dim))
        self.head_bias = nn.Parameter(torch.zeros(vocab))

        nn.init.uniform_(self.embedding.weight, -1/math.sqrt(vocab), 1/math.sqrt(vocab))
        nn.init.xavier_uniform_(self.head_W)

    def forward(self, tokens):
        """
        tokens: [T] int64
        returns logits: [T, vocab]
        """
        x      = self.embedding(tokens)    # [T, dim]
        y      = self.mamba(x)             # [T, dim]
        logits = y @ self.head_W.t() + self.head_bias  # [T, vocab]
        return logits

    def loss(self, tokens):
        """Cross-entropy loss on next-token prediction."""
        logits = self.forward(tokens[:-1])
        target = tokens[1:]
        return F.cross_entropy(logits, target)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint export  (binary format expected by lm_load() in lm.c)
# ─────────────────────────────────────────────────────────────────────────────

def export_checkpoint(model, path):
    """Write a binary checkpoint loadable by lm_load()."""
    def w(arr):
        """Write numpy/tensor as float32 bytes."""
        t = arr.detach().cpu().float()
        f.write(t.numpy().tobytes())

    with open(path, 'wb') as f:
        # 1. Magic
        f.write(struct.pack('<I', LM_MAGIC))

        # 2. LMConfig (5 x size_t = 5 x uint64 on 64-bit Linux)
        f.write(struct.pack('<5Q', VOCAB, DIM, STATE_SIZE, SEQ_LEN, MAX_GEN))

        # 3. Embedding table [V x D]
        w(model.embedding.weight)

        # 4. LM head W [V x D]
        w(model.head_W)

        # 5. LM head bias [V]
        w(model.head_bias)

        # 6. A_log [state_size, 1]  → just state_size floats
        w(model.mamba.A_log)

        # 7. B_mat [state_size, 1]
        w(model.mamba.B_mat)

        # 8. C_mat [state_size, 1]
        w(model.mamba.C_mat)

        # 9. W_in [state_size, dim]
        w(model.mamba.W_in)

        # 10. W_out [dim, state_size]
        w(model.mamba.W_out)

        # 11. delta_proj [1, dim]
        w(model.mamba.delta_proj)

    print(f"Checkpoint saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path):
    with open(path, 'rb') as f:
        raw = f.read()
    # Keep only printable ASCII (< 128)
    data = bytes(b for b in raw if b < 128)
    return torch.tensor(list(data), dtype=torch.long)

def make_windows(corpus, seq_len):
    """Slice corpus into overlapping windows of seq_len+1 tokens."""
    step    = seq_len // 2
    windows = []
    for start in range(0, len(corpus) - seq_len, step):
        windows.append(corpus[start : start + seq_len + 1])
    return windows

def train(data_path='data/train.txt',
          ckpt_path='lm_checkpoint.bin',
          epochs=20,
          lr=3e-3,
          device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    corpus  = load_corpus(data_path)
    windows = make_windows(corpus, SEQ_LEN)
    print(f"Corpus: {len(corpus)} bytes → {len(windows)} windows")

    model = LM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(windows), eta_min=lr * 0.1)

    best_loss = float('inf')

    for epoch in range(epochs):
        # Shuffle windows each epoch
        idx = torch.randperm(len(windows))
        total_loss, count = 0.0, 0

        for i, wi in enumerate(idx):
            tokens = windows[wi].to(device)
            loss   = model.loss(tokens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            count      += 1

            if (i + 1) % 100 == 0 or i == len(idx) - 1:
                avg = total_loss / count
                ppl = math.exp(avg)
                print(f"  epoch {epoch:2d}  step {i+1:5d}/{len(idx)}  "
                      f"loss={avg:.4f}  ppl={ppl:.2f}", flush=True)

        avg_loss = total_loss / count
        ppl      = math.exp(avg_loss)
        print(f"Epoch {epoch:2d}  loss={avg_loss:.4f}  ppl={ppl:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            export_checkpoint(model, ckpt_path)
            print(f"  → checkpoint saved (best so far)")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    export_checkpoint(model, ckpt_path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data',   default='data/train.txt')
    p.add_argument('--ckpt',   default='lm_checkpoint.bin')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr',     type=float, default=3e-3)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    train(args.data, args.ckpt, args.epochs, args.lr, args.device)
