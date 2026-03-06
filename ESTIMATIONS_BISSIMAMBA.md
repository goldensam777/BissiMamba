# ESTIMATIONS_BISSIMAMBA.md

Estimation de complexite, memoire et viabilite d'entrainement pour BissiMamba
(chemins CPU + CUDA), sur le modele de `optimatrix/ESTIMATIONS.md`.

## 1. Etat actuel

- Un fichier d'estimation existe deja pour Optimatrix:
  `optimatrix/ESTIMATIONS.md`
- Il n'existait pas d'equivalent dedie au niveau racine BissiMamba.

## 2. Presets CUDA (mamba_large)

Source: `mamba_large.h`

- `ML_CFG_1B`: `n_layers=38`, `dim=2048`, `d_inner=4096`, `d_state=16`, `dt_rank=128`, `seq_len=2048`
- `ML_CFG_SMALL`: `n_layers=4`, `dim=256`, `d_inner=512`, `d_state=16`, `dt_rank=32`, `seq_len=512`

Param count (`ml_count_params`):

- 1B preset: `1,005,312,000` params (approx 1.005B)
- SMALL preset: `1,883,392` params (approx 1.88M)

Note: le commentaire "`~7 M`" pour `ML_CFG_SMALL` est stale vs la formule actuelle.

## 3. Budget memoire (fp32)

Notation:

- `P` = nombre de parametres
- 1 tenseur fp32 = `4 * P` octets

### Persistant (entrainement)

Dans l'implementation CUDA actuelle, chaque parametre a:

- poids `W`
- gradient `g`
- moments `m` et `v`

Soit environ `4` copies par parametre.

- 1B: `4 * P * 4 bytes` ~= `14.98 GiB`
- SMALL: ~= `0.028 GiB`

### Temporaire (forward layer)

`layer_forward` alloue des buffers transitoires proportionnels a `T`, `D`, `d_inner`.
Pour `ML_CFG_1B` avec batch=1:

- peak transitoire par layer ~= `0.284 GiB` (ordre de grandeur)

Conclusion memoire:

- Avec le code actuel (sans full backprop), la conso semble plutot dans une
  enveloppe ~16-20 GiB + overhead CUDA/cuBLAS.
- Le commentaire `>=80 GB` dans `train_large.c` est tres conservateur et ne
  correspond pas au backward complet actuellement absent.

## 4. Complexite (ordre de grandeur)

Par couche et sequence:

- projections lineaires (in/out/x/dt): terme dominant ~`O(T * D * d_inner)`
- SSM scan (kernel): ~`O(T * d_inner * d_state)`
- softmax LM head tied embedding: ~`O(T * D * Vocab)`

Total modele:

- ~`n_layers * O(T * D * d_inner)` + tete vocab.

Avec `D=2048`, `d_inner=4096`, `T=2048`, `n_layers=38`, le cout est massif et
GPU-bound (cuBLAS + kernels CUDA).

## 5. Efficacite reelle pour entrainer un 1B conversationnel

Point critique: `ml_train_step` n'implemente pas encore le backward complet.

Le code indique explicitement:

- full layer-by-layer backward omis (TODO)
- gradient effectif surtout sur la tete/embedding
- "gradients mostly zero" pour de nombreux buffers

Donc:

- **inference 1B**: plausible si GPU suffisant
- **entrainement 1B utile (qualite conversationnelle)**: **pas encore efficace**
  tant que le full backprop n'est pas complete

## 6. Risques de coherence doc/code

- `mamba_large.h` annonce `>=20 GB` fp32
- `train_large.c` annonce `>=80 GB`
- `ML_CFG_SMALL` commente `~7M` mais la formule donne ~`1.88M`

Ces points devraient etre harmonises dans la doc pour eviter des attentes
incorrectes sur perf/cout.

## 7. Priorites recommandees (avant vrai run 1B)

1. Completer le backward couche par couche dans `mamba_cuda.cu`.
2. Ajouter validation gradients (finite diff) sur SMALL.
3. Mesurer tokens/s + loss/ppl sur SMALL jusqu'a convergence.
4. Recaler budgets VRAM a partir d'un profiler (Nsight Systems + `cudaMemGetInfo`).
5. Passer au preset 1B seulement apres convergence SMALL reproductible.

