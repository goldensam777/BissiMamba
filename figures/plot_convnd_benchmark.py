#!/usr/bin/env python3
"""
Graphe comparatif: convND Dense vs Séparable
Benchmark sur différentes tailles d'images 2D
"""

import matplotlib.pyplot as plt
import numpy as np

# Données des benchmarks
sizes = [64, 128, 256, 512, 1024]
dense_ms = [127.797, 34.995, 135.513, 927.379, 3497.631]
separable_ms = [1.931, 10.741, 34.616, 204.353, 807.604]
speedups = [66.18, 3.26, 3.91, 4.54, 4.33]

# Calculer les volumes (pixels * canaux)
volumes = [s * s * 64 for s in sizes]  # 64 canaux

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Benchmark convND: Dense K^N vs Séparable (ndims×K)\nGrille 2D, D=64 canaux, K=3', 
             fontsize=14, fontweight='bold')

# 1. Temps d'exécution (log scale)
ax1 = axes[0, 0]
x_pos = np.arange(len(sizes))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, dense_ms, width, label='Dense (K^N=9)', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, separable_ms, width, label='Séparable (2×K=6)', color='#2ecc71', alpha=0.8)
ax1.set_xlabel('Taille grille 2D (N×N)', fontweight='bold')
ax1.set_ylabel('Temps (ms)', fontweight='bold')
ax1.set_title('Temps d\'exécution', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{s}×{s}' for s in sizes])
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Ajouter les valeurs sur les barres
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)

# 2. Speedup ratio
ax2 = axes[0, 1]
colors_speedup = ['#27ae60' if s > 5 else '#f39c12' if s > 3 else '#e74c3c' for s in speedups]
bars = ax2.bar(x_pos, speedups, color=colors_speedup, alpha=0.8, edgecolor='black')
ax2.set_xlabel('Taille grille 2D (N×N)', fontweight='bold')
ax2.set_ylabel('Speedup (Dense/Séparable)', fontweight='bold')
ax2.set_title('Speedup: combien de fois plus rapide?', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{s}×{s}' for s in sizes])
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Seuil (1×)')
ax2.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Objectif (4×)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

for bar, sp in zip(bars, speedups):
    height = bar.get_height()
    ax2.annotate(f'{sp:.1f}×',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Throughput (GB/s)
ax3 = axes[1, 0]
dense_gb = [2.1, 8.4, 33.6, 134.2, 536.9]  # MB converti en GB
separable_gb = [4.2, 16.8, 67.1, 268.4, 1073.7]
dense_gbps = [d / 1000 / (t / 1000) for d, t in zip(dense_gb, dense_ms)]  # GB/s
separable_gbps = [s / 1000 / (t / 1000) for s, t in zip(separable_gb, separable_ms)]

ax3.plot(sizes, dense_gbps, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Dense')
ax3.plot(sizes, separable_gbps, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Séparable')
ax3.set_xlabel('Taille grille (N)', fontweight='bold')
ax3.set_ylabel('Bandwidth (GB/s)', fontweight='bold')
ax3.set_title('Bandwidth mémoire', fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Complexité algorithmique (théorie vs pratique)
ax4 = axes[1, 1]
# Théorie: Dense = O(K^N * spatial * D), Separable = O(N*K * spatial * D)
theoretical_dense = [s * s * 9 for s in sizes]  # K^N = 9
theoretical_separable = [s * s * 6 for s in sizes]  # N*K = 6
theoretical_ratio = [d / s for d, s in zip(theoretical_dense, theoretical_separable)]

ax4.plot(sizes, speedups, 'o-', color='#3498db', linewidth=2, markersize=10, 
         label='Mesuré (pratique)', markerfacecolor='white', markeredgewidth=2)
ax4.plot(sizes, theoretical_ratio, '--', color='#95a5a6', linewidth=2, 
         label='Théorie (K^N / N×K = 9/6 = 1.5x)')
ax4.axhline(y=1.5, color='#95a5a6', linestyle='--', alpha=0.5)
ax4.set_xlabel('Taille grille (N)', fontweight='bold')
ax4.set_ylabel('Speedup ratio', fontweight='bold')
ax4.set_title('Théorie vs Pratique', fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convnd_dense_vs_separable.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('convnd_dense_vs_separable.pdf', bbox_inches='tight')
print("Graphes sauvegardés:")
print("  - convnd_dense_vs_separable.png")
print("  - convnd_dense_vs_separable.pdf")

# Tableau résumé
print("\n" + "="*60)
print("RÉSUMÉ BENCHMARK")
print("="*60)
print(f"{'Taille':<12} {'Dense (ms)':<12} {'Sép. (ms)':<12} {'Speedup':<10}")
print("-"*60)
for s, d, sep, sp in zip(sizes, dense_ms, separable_ms, speedups):
    print(f"{s}×{s:<10} {d:<12.2f} {sep:<12.2f} {sp:<10.1f}x")
print("="*60)
