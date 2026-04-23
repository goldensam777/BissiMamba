# CONCEPT.md — Mamba-ND Complexe & Convolution Séparable-Parallèle

## 1. La Vision : "L'Harmonie des 11 Dimensions"

L'objectif est de créer un moteur capable de traiter des espaces de haute dimension (jusqu'à 11D±1D, inspiré par la M-Theory) sans l'explosion computationnelle classique. Au lieu de voir les dimensions comme des couches successives, nous les voyons comme des **signaux interférant dans un espace complexe**.

## 2. Le Mécanisme : Convolution Séparable-Parallèle

### Problème Classique
Une convolution ND dense coûte $O(K^N)$, où $K$ est la taille du noyau et $N$ le nombre de dimensions. En 11D, c'est impossible à calculer.

### Solution k-mamba
Nous décomposons la convolution en $N$ noyaux 1D, mais au lieu de les appliquer l'un après l'autre (séquentiel), nous les appliquons **en même temps** via le Wavefront.

$$y_P = \text{Activation} \left( \sum_{d=1}^{N} \text{Conv1D}_d(x, k_d) \cdot e^{i\theta_d} \right)$$

- **$k_d$** : Noyau 1D spécifique à la dimension $d$.
- **$e^{i\theta_d}$** : Signature de phase unique pour chaque dimension. Elle permet d'encoder l'orientation spatiale directement dans le domaine complexe.
- **Fusion de Kernel** : Le thread qui visite le point $P$ calcule les 11 contributions d'un seul coup, réduisant les accès mémoire de 11x.

## 3. L'Espace Complexe (Cauchy-Riemann Discret)

L'état caché $h$ devient un vecteur de nombres complexes.
- **Stabilité** : Les rotations complexes ($e^{i\theta}$) conservent la norme, empêchant l'explosion ou la disparition des gradients sur de très longues séquences ND.
- **Interférence** : Les dimensions ne s'ajoutent pas bêtement ; elles interfèrent. C'est ce qui permet au modèle de capturer des géométries complexes (ex: variétés de Calabi-Yau).

## 4. Architecture Logicielle

### Couche d'Orchestration (C / CPU)
- Orchestration du `KMWavefrontPlan`.
- Gestion de la topologie ND (strides, offsets).
- Dispatch intelligent entre les axes.

### Couche de Calcul (CUDA / cuBLAS)
- **cuBLAS** : Projections linéaires massives ($W_{in}, W_{out}$).
- **Custom Kernels** : ScanND et ConvND complexes avec réduction en mémoire partagée.
- **SIMD AVX2** : Micro-kernels pour le mode CPU ultra-rapide.

## 6. Le Gradient Spatial Wavefront (Moteur de Courbure)

Au lieu d'une simple accumulation statique des états parents, k-mamba utilise une **dérivée spatiale complexe**.

### Principe
Pour un point $P$ en 11D, nous ne calculons pas seulement la moyenne de ses 11 parents, mais nous analysons la **variation (le gradient)** entre ces dimensions :
$$\nabla h_P = \left[ \frac{\partial h}{\partial x_1}, \frac{\partial h}{\partial x_2}, \dots, \frac{\partial h}{\partial x_{11}} \right]$$

### Pourquoi ?
1. **Perception de la Courbure** : Cela permet au modèle de comprendre comment l'information change d'orientation spatiale, essentiel pour modéliser des structures physiques (théorie des cordes).
2. **Filtrage des Invariants** : En se concentrant sur la dérivée, le modèle devient robuste aux décalages constants et se focalise sur la structure dynamique des données.
3. **Harmonisation** : Couplé à l'espace complexe, le gradient devient un flux de phase, simulant une propagation d'onde cohérente à travers le tenseur ND.
