import numpy as np
import sympy as sp

# 定义符号变量
x, y, z, t = sp.symbols('x y z t', real=True)

# 定义速度场
V1 = (t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
V2 = (t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
V3 = -(t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(3*x + 3*y) * sp.sin(6*z)

# 计算散度
div_V = sp.diff(V1, x) + sp.diff(V2, y) + sp.diff(V3, z)
div_V_simplified = sp.simplify(div_V)

print("=" * 60)
print("验证自定义速度场的散度")
print("=" * 60)
print(f"\n∇·V = {div_V_simplified}")

if div_V_simplified == 0:
    print("\n✓ 符号计算：散度为零")
else:
    print(f"\n✗ 符号计算：散度不为零！div V = {div_V_simplified}")

# 数值验证
print("\n" + "=" * 60)
print("数值验证（在网格点上）")
print("=" * 60)

# 在几个网格点上计算散度
nx = ny = nz = 128
Lx = Ly = Lz = 2*np.pi
t_val = 0.0

max_div = 0.0
for i in range(0, nx, 16):  # 每隔16个点采样
    for j in range(0, ny, 16):
        for k in range(0, nz, 16):
            x_val = i * Lx / nx
            y_val = j * Ly / ny
            z_val = k * Lz / nz

            div_val = float(div_V.subs([(x, x_val), (y, y_val), (z, z_val), (t, t_val)]))
            max_div = max(max_div, abs(div_val))

print(f"\n网格点上的最大散度: {max_div:.6e}")

if max_div < 1e-10:
    print("✓ 数值验证：散度在网格上近似为零")
else:
    print("✗ 数值验证：散度在网格上不为零！")
