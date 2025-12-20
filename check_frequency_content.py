#!/usr/bin/env python3
"""
检查精确解的频谱内容
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 参数
nx = ny = nz = 128
Lx = Ly = Lz = 2 * np.pi
t = 0.0

# 生成网格
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
z = np.linspace(0, Lz, nz, endpoint=False)

# 1D profiles (fix y=0, z=0)
V1_1d = (t**2 + 1) * np.exp(np.sin(3*x + 3*0)) * np.cos(6*0)

# FFT of 1D profile
V1_fft = np.fft.fft(V1_1d)
V1_power = np.abs(V1_fft)**2

# 频率
freq = np.fft.fftfreq(nx, Lx/nx/(2*np.pi))  # Normalized frequency

# Nyquist frequency
k_nyquist = nx // 2

print("=" * 70)
print("频谱分析：精确解 V1(x, 0, 0, t=0)")
print("=" * 70)
print(f"\n网格点数: {nx}")
print(f"Nyquist波数: {k_nyquist}")
print(f"最大可分辨波数: k < {k_nyquist}")

# Find significant modes
threshold = 1e-6 * np.max(V1_power)
significant_modes = np.where(V1_power > threshold)[0]

print(f"\n显著模态 (power > {threshold:.2e}):")
print(f"总共 {len(significant_modes)} 个模态")

# 显示前20个最强的模态
sorted_indices = np.argsort(V1_power)[::-1]
print("\n前20个最强模态:")
print(f"{'Mode':<8} {'Wavenumber':<12} {'Power':<15} {'Aliased?':<10}")
print("-" * 50)
for i in range(min(20, len(sorted_indices))):
    idx = sorted_indices[i]
    k = int(freq[idx])
    power = V1_power[idx]
    if abs(k) >= k_nyquist * 0.67:  # 2/3 rule for dealiasing
        aliased = "YES"
    else:
        aliased = "No"
    print(f"{idx:<8} {k:<12} {power:<15.3e} {aliased:<10}")

# 检查是否有超过2/3 Nyquist的模态
k_23_nyquist = int(2 * k_nyquist / 3)
high_freq_power = np.sum(V1_power[k_23_nyquist:nx-k_23_nyquist])
total_power = np.sum(V1_power)

print(f"\n高频能量 (|k| > 2/3 k_nyquist): {high_freq_power:.3e}")
print(f"总能量: {total_power:.3e}")
print(f"高频占比: {100*high_freq_power/total_power:.2f}%")

if high_freq_power / total_power > 0.01:
    print("\n⚠ 警告：超过1%的能量在高频区域，可能需要去混叠！")
else:
    print("\n✓ 高频能量可忽略")

# 保存图像
plt.figure(figsize=(10, 6))
plt.semilogy(freq[:nx//2], V1_power[:nx//2], 'b-', linewidth=2)
plt.axvline(k_23_nyquist, color='r', linestyle='--', label=f'2/3 Nyquist (k={k_23_nyquist})')
plt.axvline(k_nyquist, color='k', linestyle='--', label=f'Nyquist (k={k_nyquist})')
plt.xlabel('Wavenumber k')
plt.ylabel('Power')
plt.title('Frequency Spectrum of V1(x,0,0,t=0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('frequency_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n频谱图已保存到 frequency_spectrum.png")

# 2D分析：在x-y平面
print("\n" + "=" * 70)
print("2D频谱分析：V1(x, y, z=0, t=0)")
print("=" * 70)

X, Y = np.meshgrid(x, y, indexing='ij')
V1_2d = (t**2 + 1) * np.exp(np.sin(3*X + 3*Y)) * np.cos(6*0)

# 2D FFT
V1_fft_2d = np.fft.fft2(V1_2d)
V1_power_2d = np.abs(V1_fft_2d)**2

# 径向平均功率谱
kx = np.fft.fftfreq(nx, Lx/nx/(2*np.pi))
ky = np.fft.fftfreq(ny, Ly/ny/(2*np.pi))
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K = np.sqrt(KX**2 + KY**2)

# 分bin计算径向谱
k_bins = np.arange(0, k_nyquist+1)
radial_power = np.zeros(len(k_bins)-1)

for i in range(len(k_bins)-1):
    mask = (K >= k_bins[i]) & (K < k_bins[i+1])
    radial_power[i] = np.mean(V1_power_2d[mask]) if np.any(mask) else 0

k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

plt.figure(figsize=(10, 6))
plt.semilogy(k_centers, radial_power, 'b-', linewidth=2)
plt.axvline(k_23_nyquist, color='r', linestyle='--', label=f'2/3 Nyquist (k={k_23_nyquist})')
plt.axvline(k_nyquist, color='k', linestyle='--', label=f'Nyquist (k={k_nyquist})')
plt.xlabel('Radial Wavenumber |k|')
plt.ylabel('Average Power')
plt.title('Radial Power Spectrum of V1(x,y,0,t=0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, k_nyquist])
plt.savefig('radial_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()

print("径向频谱图已保存到 radial_spectrum.png")

# 检查2D高频能量
high_freq_mask = K > k_23_nyquist
high_freq_power_2d = np.sum(V1_power_2d[high_freq_mask])
total_power_2d = np.sum(V1_power_2d)

print(f"\n2D高频能量 (|k| > 2/3 k_nyquist): {high_freq_power_2d:.3e}")
print(f"2D总能量: {total_power_2d:.3e}")
print(f"2D高频占比: {100*high_freq_power_2d/total_power_2d:.2f}%")

if high_freq_power_2d / total_power_2d > 0.01:
    print("\n⚠ 警告：2D分析显示需要去混叠！")
    print("建议：")
    print("  1. 增加网格分辨率（例如 256³ 或更高）")
    print("  2. 实现 3/2 规则去混叠")
    print("  3. 使用更简单的精确解（更低频）")
else:
    print("\n✓ 2D分析：高频能量可忽略")
