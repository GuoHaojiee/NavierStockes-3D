#!/usr/bin/env python3
"""
ETD4RK 系数的独立数值检查（配合 NavierStokes_periodic_accfft_etd.cpp）

格式：Cox & Matthews (2002) "Exponential time differencing for stiff systems",
J. Comput. Phys. 176, 430-455, 式 (26)-(29)。
系数的围道积分算法：Kassam & Trefethen (2005), SIAM J. Sci. Comput. 26(4), 1214-1233。

对标量 / 逐模态方程  u' = c u + N(u, t)，h = dt，z = c h：
  E  = e^z,  E2 = e^{z/2}
  Q  = (E2 - 1)/c                                  = h * phiQ(z),  phiQ(z) = (e^{z/2}-1)/z
  f1 = [-4 - z + E(4 - 3z + z^2)] / (h^2 c^3)      = h * g1(z),    g1(z) = [-4 - z + e^z(4-3z+z^2)]/z^3
  f2 = [ 2 + z + E(-2 + z)]       / (h^2 c^3)      = h * g2(z),    g2(z) = [ 2 + z + e^z(z-2)]/z^3
  f3 = [-4 - 3z - z^2 + E(4 - z)] / (h^2 c^3)      = h * g3(z),    g3(z) = [-4 -3z - z^2 + e^z(4-z)]/z^3
（f1,f2,f3,Q 都已经包含因子 h，与 C++ 中系数表的约定相同。）
极限 z -> 0：phiQ -> 1/2，g1 -> 1/6，g2 -> 1/6，g3 -> 1/6
  ==> Q -> h/2, f1 -> h/6, f2 -> h/6, f3 -> h/6。
  注意：更新式中 f2 乘以 2（2 f2 (N_a + N_b)），所以有效权重 h/6, h/3, h/3, h/6，
  即经典 RK4 权重。任务说明里写的 "f2 -> 1/3" 实际是 2*f2 的极限，本脚本数值验证了这一点。

用法：python3 etd_coeff_check.py   （需要 numpy, mpmath, sympy）
全部检查通过时退出码为 0。
"""
import sys
import numpy as np
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
M_CONTOUR = 32       # 围道点数（与 C++ 相同）
Z_CONTOUR = 0.5      # 0 < |z| < Z_CONTOUR 用围道积分，否则用显式公式（与 C++ 相同）
# 说明：半径 1 的圆在 |z|≈1 时会经过原点附近，圆周上的点本身又落入相消区，
# 精度降到 ~1e-13（见第 2 节 z=-1 一行）。取 Z_CONTOUR=0.5 时：围道点满足 |w|>=0.5，
# 显式公式在 |z|>=0.5 处相对误差 ~1e-14，两段都在 1e-13 以内。

failures = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        failures.append(name)


# ----------------------------------------------------------------------------
# 三种实现：显式公式（双精度）、Kassam-Trefethen 围道积分（双精度）、mpmath 高精度
# ----------------------------------------------------------------------------
def phi_direct(z):
    """显式公式（双精度），返回 (phiQ, g1, g2, g3)。小 |z| 时有严重相消。"""
    ez, ez2 = np.exp(z), np.exp(z / 2)
    return ((ez2 - 1) / z,
            (-4 - z + ez * (4 - 3 * z + z * z)) / z**3,
            (2 + z + ez * (z - 2)) / z**3,
            (-4 - 3 * z - z * z + ez * (4 - z)) / z**3)


def phi_contour(z, M=M_CONTOUR):
    """Kassam-Trefethen (2005) 围道积分：以 z 为圆心、半径 1 的圆上取 M 个点取平均。
    圆周点 r_j = exp(i*pi*(j+0.5)*2/M)，避开实轴上的点；z 为实数，取实部。"""
    r = np.exp(2j * np.pi * (np.arange(M) + 0.5) / M)
    zz = z + r
    vals = phi_direct(zz)
    return tuple(float(np.real(np.mean(v))) for v in vals)


def phi_hybrid(z):
    """与 C++ 中 build_etd_table 相同的组合方式。"""
    if z == 0.0:
        return (0.5, 1 / 6, 1 / 6, 1 / 6)
    if abs(z) < Z_CONTOUR:
        return phi_contour(z)
    return tuple(float(v) for v in phi_direct(z))


def phi_mp(z):
    """mpmath 60 位精度的显式公式（作为参考值）。"""
    z = mp.mpf(z)
    if z == 0:
        return (mp.mpf(1) / 2, mp.mpf(1) / 6, mp.mpf(1) / 6, mp.mpf(1) / 6)
    ez, ez2 = mp.e**z, mp.e**(z / 2)
    return ((ez2 - 1) / z,
            (-4 - z + ez * (4 - 3 * z + z * z)) / z**3,
            (2 + z + ez * (z - 2)) / z**3,
            (-4 - 3 * z - z * z + ez * (4 - z)) / z**3)


# ----------------------------------------------------------------------------
# 1. c -> 0 极限
# ----------------------------------------------------------------------------
print("== 1. 极限 z = c h -> 0 ==")
for z in [0.0, -1e-13, -1e-8, -1e-4]:
    v = phi_hybrid(z)
    print(f"  z={z:9.1e}  phiQ={v[0]:.15f}  g1={v[1]:.15f}  g2={v[2]:.15f}  g3={v[3]:.15f}")
v = phi_hybrid(-1e-13)   # 偏离极限为 O(z)，故取很小的 z
check("phiQ->1/2, g1,g2,g3->1/6 (z=-1e-13)", max(abs(v[0] - 0.5), abs(v[1] - 1 / 6),
                                     abs(v[2] - 1 / 6), abs(v[3] - 1 / 6)) < 1e-12)
w = (v[1], 2 * v[2], 2 * v[2], v[3])
check("RK4 权重 (f1, 2f2, 2f2, f3)/h -> (1/6,1/3,1/3,1/6)",
      np.allclose(w, (1 / 6, 1 / 3, 1 / 3, 1 / 6), atol=1e-12))

# ----------------------------------------------------------------------------
# 2. 围道积分 / 显式公式 vs mpmath
# ----------------------------------------------------------------------------
print("\n== 2. 与 mpmath(60 位) 比较的最大相对误差 ==")
print(f"  {'z':>10} | {'显式(双精度)':>13} | {'围道 M=32':>10} | {'C++ 组合':>10}")
zs = [-1e-8, -1e-5, -1e-3, -0.1, -0.5, -1.0, -2.0, -5.0, -7.9, -8.0, -20.0,
      -100.0, -1e3, -1e4, -1e5]
worst_hybrid = 0.0
worst_contour_far = 0.0
for z in zs:
    ref = [float(x) for x in phi_mp(z)]
    rel = lambda a: max(abs(a[i] - ref[i]) / abs(ref[i]) for i in range(4))
    ed, ec, eh = rel(phi_direct(z)), rel(phi_contour(z)), rel(phi_hybrid(z))
    worst_hybrid = max(worst_hybrid, eh)
    if abs(z) <= Z_CONTOUR or abs(z) >= 2.0:   # 圆周不经过原点附近的区域
        worst_contour_far = max(worst_contour_far, ec)
    print(f"  {z:10.1e} | {ed:13.2e} | {ec:10.2e} | {eh:10.2e}")
check("C++ 组合方式在所有 z 上相对误差 < 1e-13", worst_hybrid < 1e-13,
      f"(max {worst_hybrid:.2e})")
check("围道积分与高精度直接公式一致 (|z|<=0.5 或 |z|>=2, < 1e-13)", worst_contour_far < 1e-13,
      f"(max {worst_contour_far:.2e})")
ed_small = max(abs(phi_direct(-1e-5)[i] - float(phi_mp(-1e-5)[i])) for i in range(1, 4))
print(f"  （说明：显式公式在 z=-1e-5 处绝对误差 {ed_small:.1e}，相消严重，这正是要用围道积分的原因）")

# ----------------------------------------------------------------------------
# 3. 标量方程 u' = c u + sin t, c = -100（Cox & Matthews 图 5 的测试问题）
# ----------------------------------------------------------------------------
print("\n== 3. 标量方程 u' = c u + sin t, c=-100, u(0)=0, T=pi/2 ==")
c = -100.0
T = np.pi / 2


def exact(t, u0=0.0):
    up = lambda s: (-c * np.sin(s) - np.cos(s)) / (c * c + 1)
    return up(t) + (u0 - up(0.0)) * np.exp(c * t)


def etd4rk(h, nsteps):
    """Cox & Matthews (2002) 式 (26)-(29)；N(u,t) = sin t（与 u 无关，但阶段时间不同）"""
    z = c * h
    E, E2 = np.exp(z), np.exp(z / 2)
    pq, g1, g2, g3 = phi_hybrid(z)
    Q, f1, f2, f3 = h * pq, h * g1, h * g2, h * g3
    N = lambda u, t: np.sin(t)
    u, t = 0.0, 0.0
    for _ in range(nsteps):
        Nu = N(u, t)
        a = E2 * u + Q * Nu                         # (26)
        Na = N(a, t + h / 2)
        b = E2 * u + Q * Na                         # (27)
        Nb = N(b, t + h / 2)
        cc = E2 * a + Q * (2 * Nb - Nu)             # (28)
        Nc = N(cc, t + h)
        u = E * u + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc   # (29)
        t += h
    return u


def rk4(h, nsteps):
    F = lambda u, t: c * u + np.sin(t)
    u, t = 0.0, 0.0
    for _ in range(nsteps):
        k1 = F(u, t); k2 = F(u + h / 2 * k1, t + h / 2)
        k3 = F(u + h / 2 * k2, t + h / 2); k4 = F(u + h * k3, t + h)
        u += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
    return u


print(f"  {'n':>6} {'h':>10} {'|c|h':>7} {'ETD4RK err':>12} {'阶':>6} {'RK4 err':>12}")
ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
errs = []
for n in ns:
    h = T / n
    e = abs(etd4rk(h, n) - exact(T))
    r = abs(rk4(h, n) - exact(T))
    errs.append(e)
    order = "" if len(errs) < 2 else f"{np.log2(errs[-2] / errs[-1]):6.2f}"
    print(f"  {n:6d} {h:10.3e} {abs(c) * h:7.2f} {e:12.3e} {order:>6} {r:12.3e}")
# 渐近区（|c|h <= 1.25）拟合斜率；误差需高于舍入水平（|u|~1e-2，eps|u|~1e-18）
sel = [i for i, n in enumerate(ns) if abs(c) * T / n <= 1.25 and errs[i] > 1e-16]
slope = np.polyfit(np.log([T / ns[i] for i in sel]), np.log([errs[i] for i in sel]), 1)[0]
check("ETD4RK 渐近区 h-误差斜率约为 4", 3.7 < slope < 4.3,
      f"(斜率 {slope:.2f}，拟合点 n={[ns[i] for i in sel]})")
check("|c|h=39 时 ETD4RK 仍稳定（误差有界）", errs[0] < 1e-3, f"(err {errs[0]:.2e})")

# ----------------------------------------------------------------------------
# 4. 构造解（MMS）符号检查：C++ 中空间形状函数的旋度/拉普拉斯/散度
# ----------------------------------------------------------------------------
print("\n== 4. 构造解符号检查（sympy）==")
x, y, z_, t, P, om, A = sp.symbols("x y z t P omega A", real=True)
s = 3 * x + 3 * y
S = sp.Matrix([sp.exp(sp.sin(s)) * sp.cos(6 * z_),
               sp.exp(sp.sin(s)) * sp.cos(6 * z_),
               -sp.exp(sp.sin(s)) * sp.cos(s) * sp.sin(6 * z_)])
X = (x, y, z_)
div = sum(sp.diff(S[i], X[i]) for i in range(3))
check("div S = 0", sp.simplify(div) == 0)
rot = sp.Matrix([sp.diff(S[2], y) - sp.diff(S[1], z_),
                 sp.diff(S[0], z_) - sp.diff(S[2], x),
                 sp.diff(S[1], x) - sp.diff(S[0], y)])
lap = sp.Matrix([sum(sp.diff(S[i], v, 2) for v in X) for i in range(3)])


# 与 C++ 中 shape_* 函数逐字对应的数值实现
def cpp_shapes(xv, yv, zv):
    cc, ss = np.cos(3 * xv + 3 * yv), np.sin(3 * xv + 3 * yv)
    e = np.exp(ss)
    lap12 = 2 * 9 * e * (cc * cc - ss) * np.cos(6 * zv) - 36 * e * np.cos(6 * zv)
    lap3 = -2 * 9 * e * cc * ((cc * cc - ss) - (2 * ss + 1)) * np.sin(6 * zv) \
        + 36 * e * cc * np.sin(6 * zv)
    r1 = -3 * e * (cc * cc - ss) * np.sin(6 * zv) + 6 * e * np.sin(6 * zv)
    r2 = -6 * e * np.sin(6 * zv) + 3 * e * (cc * cc - ss) * np.sin(6 * zv)
    return (lap12, lap12, lap3), (r1, r2, 0.0)


rng = np.random.default_rng(0)
err_lap = err_rot = 0.0
fl = sp.lambdify(X, lap, "numpy")
fr = sp.lambdify(X, rot, "numpy")
for _ in range(50):
    p = rng.uniform(0, 2 * np.pi, 3)
    L, R = cpp_shapes(*p)
    Ls, Rs = np.array(fl(*p)).ravel(), np.array(fr(*p)).ravel()
    err_lap = max(err_lap, max(abs(L[i] - Ls[i]) for i in range(3)))
    err_rot = max(err_rot, max(abs(R[i] - Rs[i]) for i in range(3)))
check("C++ Laplace(S) 与 sympy 一致", err_lap < 1e-10, f"(max {err_lap:.1e})")
check("C++ rot(S) 与 sympy 一致", err_rot < 1e-10, f"(max {err_rot:.1e})")

# 外力：f = g'(t) S - P g(t) ΔS - g(t)^2 (S × rot S) + g(t) ∇p0 , p = g(t) cos x cos y cos z
# 检查：v = g S 代入 v_t = P Δv + v×rot v - ∇p + f 恒成立（两种时间因子）
p0 = sp.cos(x) * sp.cos(y) * sp.cos(z_)
gradp0 = sp.Matrix([sp.diff(p0, v) for v in X])
for name, g in [("poly  g=t^2+1", t**2 + 1), ("sin   g=1+A sin(w t)", 1 + A * sp.sin(om * t))]:
    v = g * S
    f = sp.diff(g, t) * S - P * g * lap - g**2 * S.cross(rot) + g * gradp0
    rotv = sp.Matrix([sp.diff(v[2], y) - sp.diff(v[1], z_),
                      sp.diff(v[0], z_) - sp.diff(v[2], x),
                      sp.diff(v[1], x) - sp.diff(v[0], y)])
    lapv = sp.Matrix([sum(sp.diff(v[i], w, 2) for w in X) for i in range(3)])
    res = sp.diff(v, t) - (P * lapv + v.cross(rotv) - g * gradp0 + f)
    check(f"MMS 残差为 0 [{name}]", all(sp.simplify(r) == 0 for r in res))

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
