# THEORY.md — Fondements Théoriques de BissiMamba

> Ce document explique la mathématique et l'architecture d'un vrai
> modèle Mamba, depuis les équations d'état jusqu'au forward pass.
> Il sert de référence pour comprendre *pourquoi* chaque opération
> matricielle existe.

---

## 1. Pourquoi pas un Transformer ?

Un Transformer calcule l'attention :

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Le produit `QK^T` est de taille `(L × L)`. Pour une séquence de
longueur L, le coût est **O(L²)**. À L=4096, c'est 16 millions
d'opérations juste pour ce produit.

Mamba remplace l'attention par un **modèle d'état** (SSM) :
coût **O(L)** en espace, **O(L log L)** en temps avec le scan parallèle.

---

## 2. Le Modèle d'État (SSM) — Équations Continues

Un SSM linéaire en temps continu :

```
h'(t) = A · h(t) + B · x(t)      ← équation d'état
y(t)  = C · h(t) + D · x(t)      ← équation de sortie
```

où :
- `x(t) ∈ ℝ^d` : entrée au temps t
- `h(t) ∈ ℝ^N` : état caché (mémoire du système)
- `y(t) ∈ ℝ^d` : sortie
- `A ∈ ℝ^{N×N}` : matrice de transition d'état
- `B ∈ ℝ^{N×d}` : matrice d'entrée
- `C ∈ ℝ^{d×N}` : matrice de sortie
- `D ∈ ℝ`       : terme de feedthrough (skip connection)

**Intuition géométrique** : A est une Volonté sur l'espace d'état —
elle dit "dans quelle direction l'état veut évoluer". B projette
l'entrée dans cet espace. C extrait la sortie de l'état.

---

## 3. Discrétisation — Du Continu au Discret

Pour traiter des séquences discrètes (tokens), on discrétise avec
pas de temps Δ (delta) via la méthode **Zero-Order Hold (ZOH)** :

```
Ā = exp(Δ · A)
B̄ = (ΔA)^{-1} · (exp(ΔA) - I) · ΔB
```

La récurrence discrète devient :

```
h_t = Ā · h_{t-1} + B̄ · x_t
y_t = C · h_t
```

**Pourquoi ZOH ?** C'est l'équivalent discret exact d'un intégrateur
continu maintenu constant entre deux échantillons. Pour A diagonal,
`exp(ΔA)` se réduit à `diag(exp(Δ · a_ii))` — un simple vecteur.

### Approximation dans BissiMamba

BissiMamba utilise une approximation simplifiée :
```
Ā[i] = exp(Δ · A[i])
B̄[i] = (Ā[i] - 1) / A[i] · B[i]     si |A[i]| > ε
B̄[i] = Δ · B[i]                       sinon (limite quand A→0)
```

---

## 4. La Sélectivité — Ce qui rend Mamba unique

Dans un SSM classique (S4), A, B, C sont **fixes** — indépendants de
l'entrée. Le modèle ne peut pas choisir ce qu'il mémorise.

Mamba introduit la **sélectivité** : B, C, et Δ dépendent de l'entrée :

```
Δ(x) = softplus(linear(x))    ← positif, adaptatif
B(x) = linear(x)               ← dépend du token courant
C(x) = linear(x)               ← dépend du token courant
```

**Interprétation** :
- Δ grand → fort couplage à l'état → le modèle mémorise
- Δ petit → faible couplage → le modèle oublie / passe à travers

C'est un mécanisme d'attention implicite, sans le coût quadratique.

---

## 5. Le Selective Scan — Algorithme Central

La récurrence `h_t = Ā_t · h_{t-1} + B̄_t · x_t` est séquentielle
en apparence. Mais elle peut être parallélisée via le **scan associatif**.

### Opérateur de composition

Définir l'opérateur ⊕ :

```
(a₂, b₂) ⊕ (a₁, b₁) = (a₂ · a₁,  a₂ · b₁ + b₂)
```

Alors h_t peut s'écrire :

```
h_t = a_t · h_{t-1} + b_t
    = (a_t, b_t) ⊕ (a_{t-1}, b_{t-1}) ⊕ ... ⊕ (a_1, b_1) ⊕ h_0
```

Ce prefix-scan est **associatif** → **parallélisable en O(log L)**
via l'algorithme de Blelloch (Work-Efficient Parallel Scan).

### Dans BissiMamba (implémentation actuelle)

BissiMamba implémente le scan **séquentiel** (O(L), plus simple) :

```c
for (t = 0; t < seq_len; t++) {
    state[i] = A_diag_t[i] * state[i] + B_bar_t[i] * u_t[i];
}
```

optimatrix fournit la version vectorisée avec `hadamard_avx2`.

---

## 6. Architecture Complète d'un Block Mamba

```
Input x ∈ ℝ^{L×d}
        │
        ├─────────────────────────────────────────┐
        │                                         │
   Linear (W_in)                            Linear (z_proj)
   x_proj = x · W_in^T                     z = x · W_z^T
   ∈ ℝ^{L×2d_inner}                        ∈ ℝ^{L×d_inner}
        │                                         │
   Split en (u, gate)                        SiLU(z)
        │                                         │
   SSM selective scan                             │
   u' = scan(u, Δ, A, B, C)                      │
   ∈ ℝ^{L×d_inner}                               │
        │                                         │
        └─────── u' ⊙ SiLU(z) ──── W_out ────────┘
                                        │
                              Output ∈ ℝ^{L×d}
```

**BissiMamba (simplifié)** implémente le chemin central :
```
x → W_in → SiLU → scan → W_out → y
```
Sans la branche gate (z_proj) pour l'instant.

---

## 7. Initialisation HiPPO de A

Le papier original S4 initialise A avec la **matrice HiPPO** :

```
A[n,k] = -√(2n+1) · √(2k+1)     si n > k
A[n,k] = -(n+1)                  si n = k
A[n,k] = 0                       si n < k
```

Cette matrice encode une mémoire optimale des polynômes de Legendre.
Elle garantit que l'état `h_t` mémorise l'historique complet de `x`
de manière numériquement stable.

Mamba simplifie vers une initialisation diagonale :
```
A = -exp(A_log)     (négatif pour stabilité, log pour constraint ℝ-)
```

BissiMamba utilise :
```c
A_log[i] = -exp(spacing * log(dt_scale))  /* logarithmiquement espacé */
```

---

## 8. Complexité du Modèle

| Opération         | Complexité     | Dominante pour       |
|-------------------|----------------|----------------------|
| Projection W_in   | O(L · d · N)   | L grand, d et N fixes|
| Selective scan    | O(L · N)       | L grand              |
| Projection W_out  | O(L · N · d)   | L grand, d et N fixes|
| Attention (ref)   | O(L² · d)      | L grand              |

Pour L=1024, d=64, N=32 :
- Mamba : ~4.3M opérations
- Attention : ~4.3M opérations (ici comparable, avantage à L >> d)

À L=8192 : Mamba scale linéairement, Attention × 64.

---

## 9. Le Lien avec la Théorie des Volontés

La récurrence SSM peut se lire comme un **système de Volontés** :

| Composante    | Théorie des Volontés                            |
|---------------|-------------------------------------------------|
| `h_t`         | L'état d'équilibre courant                      |
| `A`           | La Volonté de persistance (vers où l'état tend) |
| `B · x_t`     | La Volonté d'input (ce que l'entrée veut changer)|
| `Δ`           | L'intensité des Volontés à chaque pas           |
| `h_{t+1}`     | L'équilibre résultant des Volontés composées    |

La **sélectivité** (Δ adaptatif) est l'arbitrage des Volontés :
le modèle décide à chaque token combien peser la mémoire passée
vs l'information nouvelle.

```
Ā = exp(Δ · A)  ← Volonté de persistance pondérée par Δ
B̄ · x_t         ← Volonté d'input pondérée par Δ
h_t = Ā · h_{t-1} + B̄ · x_t  ← Convergence des Volontés
```

Quand Δ → 0 : Ā → I, le passé domine. Ā → exp(A), l'input domine.
L'équilibre émerge. Ce n'est pas prescrit — il est calculé.

---

## 10. Vers BissiMamba ND

L'extension naturelle est le **scan 2D** implémenté dans optimatrix :

```
h(i,j) = dA1 · h(i-1,j) + dA2 · h(i,j-1) + dB · x(i,j)
```

Pour des entrées 2D (images, spectrogrammes, grilles de tokens) :
- 2 Volontés de persistance (horizontale et verticale)
- Ordonnancement wavefront (diagonales k=i+j indépendantes)

Cela ouvre vers un **modèle multimodal** : même architecture SSM,
appliquée à des données structurées en 2D ou ND.

---

## Références

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023
- Gu et al., *S4: Efficiently Modeling Long Sequences with Structured State Spaces*, 2022
- Gu et al., *HiPPO: Recurrent Memory with Optimal Polynomial Projections*, 2020
- Blelloch, *Prefix Sums and Their Applications*, 1990
