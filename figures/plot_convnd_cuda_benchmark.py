#!/usr/bin/env python3
"""
Comparative plot: ConvND CUDA Dense vs Separable
Benchmark GPU results from NVIDIA GeForce MX450
"""

import matplotlib.pyplot as plt
import numpy as np

# Real benchmark data from MX450 GPU
# 2D grids
sizes_2d = [64, 128, 256, 512]
dense_2d_ms = [1.143, 2.234, 5.298, 15.098]
separable_2d_ms = [1.506, 2.690, 7.131, 19.880]
speedups_2d = [d/s for d, s in zip(dense_2d_ms, separable_2d_ms)]
bw_dense_2d = [874.87, 1790.76, 3020.13, 4238.89]
bw_sep_2d = [664.21, 1487.03, 2243.80, 3219.38]

# 3D grids
sizes_3d = [32, 64]
dense_3d_ms = [2.987, 19.629]
separable_3d_ms = [5.582, 37.239]
speedups_3d = [d/s for d, s in zip(dense_3d_ms, separable_3d_ms)]
bw_dense_3d = [2678.66, 3260.48]
bw_sep_3d = [1433.08, 1718.62]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ConvND CUDA Benchmark: Dense K^N vs Separable\nGPU: NVIDIA GeForce MX450, D=64 channels, K=3', 
             fontsize=14, fontweight='bold')

# 1. 2D Execution Time
ax1 = axes[0, 0]
x_pos = np.arange(len(sizes_2d))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, dense_2d_ms, width, label='Dense (K^N=9)', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, separable_2d_ms, width, label='Separable (2×K=6)', color='#2ecc71', alpha=0.8)
ax1.set_xlabel('2D Grid Size (N×N)', fontweight='bold')
ax1.set_ylabel('Time (ms)', fontweight='bold')
ax1.set_title('2D Grids: Execution Time', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{s}×{s}' for s in sizes_2d])
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Speedup Ratio (inverse pattern vs CPU!)
ax2 = axes[0, 1]
colors_speedup = ['#e74c3c' if s < 1.0 else '#27ae60' for s in speedups_2d]
bars = ax2.bar(x_pos, speedups_2d, color=colors_speedup, alpha=0.8, edgecolor='black')
ax2.set_xlabel('2D Grid Size (N×N)', fontweight='bold')
ax2.set_ylabel('Speedup (Dense/Separable)', fontweight='bold')
ax2.set_title('Speedup: Dense is faster on GPU!', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{s}×{s}' for s in sizes_2d])
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Parity (1×)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

for bar, sp in zip(bars, speedups_2d):
    height = bar.get_height()
    ax2.annotate(f'{sp:.2f}×',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Memory Bandwidth
ax3 = axes[1, 0]
ax3.plot(sizes_2d, bw_dense_2d, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Dense')
ax3.plot(sizes_2d, bw_sep_2d, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Separable')
ax3.set_xlabel('Grid Size (N)', fontweight='bold')
ax3.set_ylabel('Memory Bandwidth (GB/s)', fontweight='bold')
ax3.set_title('2D Grids: Memory Bandwidth', fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 3D Grids comparison
ax4 = axes[1, 1]
x_pos_3d = np.arange(len(sizes_3d))
bars1_3d = ax4.bar(x_pos_3d - width/2, dense_3d_ms, width, label='Dense (K^N=27)', color='#e74c3c', alpha=0.8)
bars2_3d = ax4.bar(x_pos_3d + width/2, separable_3d_ms, width, label='Separable (3×K=9)', color='#2ecc71', alpha=0.8)
ax4.set_xlabel('3D Grid Size (N×N×N)', fontweight='bold')
ax4.set_ylabel('Time (ms)', fontweight='bold')
ax4.set_title('3D Grids: Execution Time', fontweight='bold')
ax4.set_xticks(x_pos_3d)
ax4.set_xticklabels([f'{s}³' for s in sizes_3d])
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/convnd_cuda_dense_vs_separable.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figures/convnd_cuda_dense_vs_separable.pdf', bbox_inches='tight')
print("Plots saved:")
print("  - figures/convnd_cuda_dense_vs_separable.png")
print("  - figures/convnd_cuda_dense_vs_separable.pdf")

# Summary table
print("\n" + "="*60)
print("CUDA BENCHMARK SUMMARY (NVIDIA GeForce MX450)")
print("="*60)
print(f"{'Size':<12} {'Dense (ms)':<12} {'Sep. (ms)':<12} {'Speedup':<10}")
print("-"*60)
for s, d, sep, sp in zip(sizes_2d, dense_2d_ms, separable_2d_ms, speedups_2d):
    print(f"{s}×{s:<10} {d:<12.3f} {sep:<12.3f} {sp:<10.2f}x")
print("="*60)
print("\nKey Finding: Dense K^N is ~1.3x faster than Separable on GPU!")
print("(Opposite of CPU where Separable is 3-4x faster)")
