import numpy as np
import sympy as sp

# 定义符号变量
x, y, z, t = sp.symbols('x y z t', real=True)

# 定义速度场
V1 = (t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
V2 = (t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
V3 = -(t**2 + 1) * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(3*x + 3*y) * sp.sin(6*z)

print("=" * 60)
print("检查函数定义的正确性")
print("=" * 60)

# 检查不可压缩条件
div_V = sp.diff(V1, x) + sp.diff(V2, y) + sp.diff(V3, z)
div_V_simplified = sp.simplify(div_V)
print(f"\n1. 不可压缩条件: ∇·V = {div_V_simplified}")

# 检查时间导数
dV1_dt_correct = sp.diff(V1, t)
dV1_dt_code = 2*t * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
print(f"\n2. ∂V1/∂t 正确值: {dV1_dt_correct}")
print(f"   代码中的值:    {dV1_dt_code}")
print(f"   是否相等: {sp.simplify(dV1_dt_correct - dV1_dt_code) == 0}")

dV2_dt_correct = sp.diff(V2, t)
dV2_dt_code = 2*t * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(6*z)
print(f"\n3. ∂V2/∂t 是否正确: {sp.simplify(dV2_dt_correct - dV2_dt_code) == 0}")

dV3_dt_correct = sp.diff(V3, t)
dV3_dt_code = -2*t * sp.exp(sp.sin(3*x + 3*y)) * sp.cos(3*x + 3*y) * sp.sin(6*z)
print(f"4. ∂V3/∂t 是否正确: {sp.simplify(dV3_dt_correct - dV3_dt_code) == 0}")

# 检查Laplacian
print("\n" + "=" * 60)
print("检查Laplacian计算")
print("=" * 60)

laplace_V1_correct = sp.diff(V1, x, 2) + sp.diff(V1, y, 2) + sp.diff(V1, z, 2)
print(f"\n∇²V1 = {sp.simplify(laplace_V1_correct)}")

laplace_V3_correct = sp.diff(V3, x, 2) + sp.diff(V3, y, 2) + sp.diff(V3, z, 2)  
print(f"\n∇²V3 = {sp.simplify(laplace_V3_correct)}")

# 检查旋度
print("\n" + "=" * 60)
print("检查旋度计算")
print("=" * 60)

rot1_correct = sp.diff(V3, y) - sp.diff(V2, z)
rot2_correct = sp.diff(V1, z) - sp.diff(V3, x)
rot3_correct = sp.diff(V2, x) - sp.diff(V1, y)

print(f"\nrot1 = ∂V3/∂y - ∂V2/∂z = {sp.simplify(rot1_correct)}")
print(f"rot2 = ∂V1/∂z - ∂V3/∂x = {sp.simplify(rot2_correct)}")
print(f"rot3 = ∂V2/∂x - ∂V1/∂y = {sp.simplify(rot3_correct)}")

