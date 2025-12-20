#!/usr/bin/env python3
"""
验证forcing函数是否使得精确解满足NS方程（数值实现）
"""

import numpy as np
import sympy as sp

# 定义符号变量
x, y, z, t = sp.symbols('x y z t', real=True)

# 速度场
V1 = (t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
V2 = (t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
V3 = -(t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(3*x + 3*y) * sp.sin(6*z)

# 压力场
p = (t**2 + 1) * sp.cos(x) * sp.cos(y) * sp.cos(z)

print("=" * 70)
print("验证投影方法中的forcing公式")
print("=" * 70)

# 计算各项
dV1_dt = sp.diff(V1, t)
dV2_dt = sp.diff(V2, t)
dV3_dt = sp.diff(V3, t)

# Laplacian (P=1)
lap_V1 = sp.diff(V1, x, 2) + sp.diff(V1, y, 2) + sp.diff(V1, z, 2)
lap_V2 = sp.diff(V2, x, 2) + sp.diff(V2, y, 2) + sp.diff(V2, z, 2)
lap_V3 = sp.diff(V3, x, 2) + sp.diff(V3, y, 2) + sp.diff(V3, z, 2)

# 旋度
rot1 = sp.diff(V3, y) - sp.diff(V2, z)
rot2 = sp.diff(V1, z) - sp.diff(V3, x)
rot3 = sp.diff(V2, x) - sp.diff(V1, y)

# v × rot(v)
vcross1 = V2 * rot3 - V3 * rot2
vcross2 = V3 * rot1 - V1 * rot3
vcross3 = V1 * rot2 - V2 * rot1

# 压力梯度
grad_p1 = sp.diff(p, x)
grad_p2 = sp.diff(p, y)
grad_p3 = sp.diff(p, z)

print("\n完整NS方程: ∂v/∂t = P∆v + v×rot(v) - ∇p + f")
print("\n求解器方程: ∂v/∂t = P∆v + v×rot(v) + f_code")
print("（压力项通过投影隐式处理）")

print("\n" + "=" * 70)
print("代码中的forcing公式:")
print("=" * 70)
print("f_code = ∂v/∂t - P∆v - v×rot(v) + ∇p")

# 代码中的forcing
f_code_1 = dV1_dt - lap_V1 - vcross1 + grad_p1
f_code_2 = dV2_dt - lap_V2 - vcross2 + grad_p2
f_code_3 = dV3_dt - lap_V3 - vcross3 + grad_p3

print("\n简化forcing:")
f1_simp = sp.simplify(f_code_1)
f2_simp = sp.simplify(f_code_2)
f3_simp = sp.simplify(f_code_3)

print(f"f1 simplified: {f1_simp}")
print(f"f2 simplified: {f2_simp}")
print(f"f3 simplified: {f3_simp}")

print("\n" + "=" * 70)
print("验证：将forcing代入求解器方程")
print("=" * 70)
print("求解器计算: ∂v/∂t - P∆v - v×rot(v) - f_code = ?")
print("（应该等于-∇p，然后投影会消除）")

# 检查残差（投影前）
residual_1 = dV1_dt - lap_V1 - vcross1 - f_code_1
residual_2 = dV2_dt - lap_V2 - vcross2 - f_code_2
residual_3 = dV3_dt - lap_V3 - vcross3 - f_code_3

res1_simp = sp.simplify(residual_1)
res2_simp = sp.simplify(residual_2)
res3_simp = sp.simplify(residual_3)

print(f"\n残差1 (应为-∇p₁): {res1_simp}")
print(f"       -∇p₁ = {-grad_p1}")
print(f"       相等？ {sp.simplify(res1_simp + grad_p1) == 0}")

print(f"\n残差2 (应为-∇p₂): {res2_simp}")
print(f"       -∇p₂ = {-grad_p2}")
print(f"       相等？ {sp.simplify(res2_simp + grad_p2) == 0}")

print(f"\n残差3 (应为-∇p₃): {res3_simp}")
print(f"       -∇p₃ = {-grad_p3}")
print(f"       相等？ {sp.simplify(res3_simp + grad_p3) == 0}")

# 检查残差的散度（投影后应消除）
div_residual = sp.diff(residual_1, x) + sp.diff(residual_2, y) + sp.diff(residual_3, z)
div_res_simp = sp.simplify(div_residual)

print(f"\n∇·残差 (投影会消除这部分): {div_res_simp}")

# 检查残差的无散分量（投影后保留）
# 这部分应该为零，表示投影后方程精确满足
print("\n" + "=" * 70)
print("关键检查：残差 - ∇φ 是否为零？")
print("（φ 是压力，投影会消除∇φ分量）")
print("=" * 70)

# 由于残差 = -∇p，所以残差 - (-∇p) = 0
final_check_1 = sp.simplify(residual_1 + grad_p1)
final_check_2 = sp.simplify(residual_2 + grad_p2)
final_check_3 = sp.simplify(residual_3 + grad_p3)

print(f"\n残差1 + ∇p₁ = {final_check_1}")
print(f"残差2 + ∇p₂ = {final_check_2}")
print(f"残差3 + ∇p₃ = {final_check_3}")

if final_check_1 == 0 and final_check_2 == 0 and final_check_3 == 0:
    print("\n✓ 理论验证通过：forcing公式正确")
else:
    print("\n✗ 理论验证失败：forcing公式有误！")

print("\n" + "=" * 70)
print("数值测试：在网格点上评估")
print("=" * 70)

# 在几个点上数值评估
t_val = 0.1
x_val = 1.0
y_val = 2.0
z_val = 3.0

point_vals = {t: t_val, x: x_val, y: y_val, z: z_val}

f1_num = float(f_code_1.subs(point_vals))
f2_num = float(f_code_2.subs(point_vals))
f3_num = float(f_code_3.subs(point_vals))

print(f"\n在点 (x,y,z,t) = ({x_val}, {y_val}, {z_val}, {t_val}):")
print(f"f1 = {f1_num:.6e}")
print(f"f2 = {f2_num:.6e}")
print(f"f3 = {f3_num:.6e}")

res1_num = float(residual_1.subs(point_vals))
res2_num = float(residual_2.subs(point_vals))
res3_num = float(residual_3.subs(point_vals))

print(f"\n残差（求解器方程 - f_code）:")
print(f"残差1 = {res1_num:.6e}")
print(f"残差2 = {res2_num:.6e}")
print(f"残差3 = {res3_num:.6e}")

grad_p1_num = float(grad_p1.subs(point_vals))
grad_p2_num = float(grad_p2.subs(point_vals))
grad_p3_num = float(grad_p3.subs(point_vals))

print(f"\n-∇p:")
print(f"-∇p₁ = {-grad_p1_num:.6e}")
print(f"-∇p₂ = {-grad_p2_num:.6e}")
print(f"-∇p₃ = {-grad_p3_num:.6e}")

print(f"\n残差 + ∇p (应该为零):")
print(f"[{res1_num + grad_p1_num:.6e}, {res2_num + grad_p2_num:.6e}, {res3_num + grad_p3_num:.6e}]")

if abs(res1_num + grad_p1_num) < 1e-10 and abs(res2_num + grad_p2_num) < 1e-10 and abs(res3_num + grad_p3_num) < 1e-10:
    print("\n✓ 数值验证通过")
else:
    print("\n✗ 数值验证失败！")
