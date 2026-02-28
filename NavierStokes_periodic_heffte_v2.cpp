/**
 * Navier-Stokes 求解器 - heFFTe 版本（2D pencil 分解）
 *
 * 基于 NavierStokes_periodic_fftw.cpp 的核心算法，使用 heFFTe 库代替 FFTW MPI。
 *
 * 与 FFTW 版本的关键区别：
 *   - heFFTe fft3d_r2c 自动进行 2D pencil 分解（非 FFTW 的 1D slab）
 *   - R2C 压缩第三维 (dim=2, 即 z)：kz = 0..nz/2（非负）
 *   - kx = (i<=nx/2)?i:i-nx，ky = (j<=ny/2)?j:j-ny（需折叠）
 *   - 正变换使用 scale::none（不归一化），逆变换使用 scale::full（自动除以 N）
 *   - 所有频谱量均为 DFT（非归一化）表示，即 V_c = N * FFT_norm(V_phys)
 *   - heFFTe backward(scale::full) 自动处理归一化，无需手动除以 N
 *
 * 编译：参见 Makefile_heffte
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <heffte.h>

using namespace std;
using Cvec = vector<complex<double>>;
using Rvec = vector<double>;

// ==============================================================================
// 解析解与强迫项（与 FFTW 版本完全一致）
// ==============================================================================

double func_V1(double x, double y, double z, double t) {
    return (t * t + 1) * exp(sin(3 * x + 3 * y)) * cos(6 * z);
}
double func_V2(double x, double y, double z, double t) {
    return (t * t + 1) * exp(sin(3 * x + 3 * y)) * cos(6 * z);
}
double func_V3(double x, double y, double z, double t) {
    return -(t * t + 1) * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) * sin(6 * z);
}

double func_dV1_dt(double x, double y, double z, double t) {
    return 2 * t * exp(sin(3 * x + 3 * y)) * cos(6 * z);
}
double func_dV2_dt(double x, double y, double z, double t) {
    return 2 * t * exp(sin(3 * x + 3 * y)) * cos(6 * z);
}
double func_dV3_dt(double x, double y, double z, double t) {
    return -2 * t * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) * sin(6 * z);
}

double func_laplace_V1(double x, double y, double z, double t) {
    double d2v1_dx2 = (t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) *
                      (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * cos(6 * z);
    double d2v1_dy2 = (t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) *
                      (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * cos(6 * z);
    double d2v1_dz2 = -(t * t + 1) * 36 * exp(sin(3 * x + 3 * y)) * cos(6 * z);
    return d2v1_dx2 + d2v1_dy2 + d2v1_dz2;
}
double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x, y, z, t);
}
double func_laplace_V3(double x, double y, double z, double t) {
    double d2v3_dx2 = -(t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) *
                      ((cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) -
                       (2 * sin(3 * x + 3 * y) + 1)) * sin(6 * z);
    double d2v3_dy2 = d2v3_dx2;
    double d2v3_dz2 = (t * t + 1) * 36 * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) * sin(6 * z);
    return d2v3_dx2 + d2v3_dy2 + d2v3_dz2;
}

double func_rot1(double x, double y, double z, double t) {
    double dv3_dy = -(t * t + 1) * 3 * exp(sin(3 * x + 3 * y)) *
                    (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * sin(6 * z);
    double dv2_dz = -(t * t + 1) * 6 * exp(sin(3 * x + 3 * y)) * sin(6 * z);
    return dv3_dy - dv2_dz;
}
double func_rot2(double x, double y, double z, double t) {
    double dv1_dz = -(t * t + 1) * 6 * exp(sin(3 * x + 3 * y)) * sin(6 * z);
    double dv3_dx = -(t * t + 1) * 3 * exp(sin(3 * x + 3 * y)) *
                    (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * sin(6 * z);
    return dv1_dz - dv3_dx;
}
double func_rot3(double, double, double, double) { return 0.0; }

double func_v_cross_rot1(double x, double y, double z, double t) {
    return func_V2(x, y, z, t) * func_rot3(x, y, z, t) - func_V3(x, y, z, t) * func_rot2(x, y, z, t);
}
double func_v_cross_rot2(double x, double y, double z, double t) {
    return func_V3(x, y, z, t) * func_rot1(x, y, z, t) - func_V1(x, y, z, t) * func_rot3(x, y, z, t);
}
double func_v_cross_rot3(double x, double y, double z, double t) {
    return func_V1(x, y, z, t) * func_rot2(x, y, z, t) - func_V2(x, y, z, t) * func_rot1(x, y, z, t);
}

double func_grad_p1(double x, double y, double z, double t) {
    return -(t * t + 1) * sin(x) * cos(y) * cos(z);
}
double func_grad_p2(double x, double y, double z, double t) {
    return -(t * t + 1) * cos(x) * sin(y) * cos(z);
}
double func_grad_p3(double x, double y, double z, double t) {
    return -(t * t + 1) * cos(x) * cos(y) * sin(z);
}

// 强迫项：使解析解精确满足 NS 方程（投影方法，P=1）
double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x, y, z, t) - func_laplace_V1(x, y, z, t)
           - func_v_cross_rot1(x, y, z, t) + func_grad_p1(x, y, z, t);
}
double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x, y, z, t) - func_laplace_V2(x, y, z, t)
           - func_v_cross_rot2(x, y, z, t) + func_grad_p2(x, y, z, t);
}
double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x, y, z, t) - func_laplace_V3(x, y, z, t)
           - func_v_cross_rot3(x, y, z, t) + func_grad_p3(x, y, z, t);
}

// ==============================================================================
// heFFTe 频谱算子
//
// 归一化约定：
//   V_c = forward(V_r, scale::none) = DFT(V_r)  [非归一化，= N * FFT_norm(V_phys)]
//   V_r = backward(V_c, scale::full) = IDFT(V_c)/N = V_phys [自动归一化]
//   所有频谱量（V_c, rot_c, nl_c, visc_c, f_c）均使用相同的 N 因子，保持一致
// ==============================================================================

// 线性索引：heFFTe box3d，x 最快变化（C 顺序）
// i,j,k 为全局坐标（来自 box.low..box.high）
inline size_t box_idx(const heffte::box3d<>& box, int i, int j, int k) {
    return static_cast<size_t>(i - box.low[0])
         + static_cast<size_t>(j - box.low[1]) * box.size[0]
         + static_cast<size_t>(k - box.low[2]) * box.size[0] * box.size[1];
}

// 波数：heFFTe R2C 沿 dim=2（z 方向被压缩为 0..nz/2）
// kx 和 ky 需要折叠处理（Nyquist 对称）
inline double wave_kx(int i, int nx) { return i <= nx / 2 ? (double)i : (double)(i - nx); }
inline double wave_ky(int j, int ny) { return j <= ny / 2 ? (double)j : (double)(j - ny); }
inline double wave_kz(int k)         { return (double)k; }  // kz = 0..nz/2，始终非负

// 计算旋度 rot(V) = i·k × V（频谱空间）
// rot_1 = i(ky*V3 - kz*V2)
// rot_2 = i(kz*V1 - kx*V3)
// rot_3 = i(kx*V2 - ky*V1)
void compute_rot(const Cvec& V1, const Cvec& V2, const Cvec& V3,
                 Cvec& rot1, Cvec& rot2, Cvec& rot3,
                 int nx, int ny, int nz,
                 const heffte::box3d<>& box_c) {
    #pragma omp parallel for collapse(3)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx);
                double ky = wave_ky(j, ny);
                double kz = wave_kz(k);

                // i * kx * (a + ib) = -kx*b + i*kx*a
                rot1[idx] = { -(ky * V3[idx].imag() - kz * V2[idx].imag()),
                                ky * V3[idx].real() - kz * V2[idx].real() };
                rot2[idx] = { -(kz * V1[idx].imag() - kx * V3[idx].imag()),
                                kz * V1[idx].real() - kx * V3[idx].real() };
                rot3[idx] = { -(kx * V2[idx].imag() - ky * V1[idx].imag()),
                                kx * V2[idx].real() - ky * V1[idx].real() };
            }
        }
    }
}

// 计算粘性项：visc = -k² * V（频谱空间）
void compute_viscous(const Cvec& V, Cvec& visc,
                     int nx, int ny, int nz,
                     const heffte::box3d<>& box_c) {
    #pragma omp parallel for collapse(3)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx), ky = wave_ky(j, ny), kz = wave_kz(k);
                double k2 = kx * kx + ky * ky + kz * kz;
                visc[idx] = -k2 * V[idx];
            }
        }
    }
}

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间）
// 步骤：div = ik·V，phi = div/(-k²)，V -= ik*phi
void make_div_free(Cvec& V1, Cvec& V2, Cvec& V3,
                   Cvec& div, Cvec& phi,
                   int nx, int ny, int nz,
                   const heffte::box3d<>& box_c) {
    #pragma omp parallel for collapse(3)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx), ky = wave_ky(j, ny), kz = wave_kz(k);

                // div = i(kx*V1 + ky*V2 + kz*V3)
                div[idx] = { -(kx * V1[idx].imag() + ky * V2[idx].imag() + kz * V3[idx].imag()),
                               kx * V1[idx].real() + ky * V2[idx].real() + kz * V3[idx].real() };

                double k2 = kx * kx + ky * ky + kz * kz;
                phi[idx] = (k2 > 1e-14) ? (div[idx] / (-k2)) : complex<double>(0.0, 0.0);

                // V -= i*k*phi（即 ∇phi = i*k*phi）
                V1[idx] -= complex<double>(-kx * phi[idx].imag(), kx * phi[idx].real());
                V2[idx] -= complex<double>(-ky * phi[idx].imag(), ky * phi[idx].real());
                V3[idx] -= complex<double>(-kz * phi[idx].imag(), kz * phi[idx].real());
            }
        }
    }
}

// 计算 max|div(V)| 用于诊断
double compute_div_max(const Cvec& V1, const Cvec& V2, const Cvec& V3,
                       Cvec& div,
                       int nx, int ny, int nz,
                       const heffte::box3d<>& box_c) {
    double local_max = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:local_max)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx), ky = wave_ky(j, ny), kz = wave_kz(k);
                div[idx] = { -(kx * V1[idx].imag() + ky * V2[idx].imag() + kz * V3[idx].imag()),
                               kx * V1[idx].real() + ky * V2[idx].real() + kz * V3[idx].real() };
                local_max = max(local_max, abs(div[idx]));
            }
        }
    }
    double global_max = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global_max;
}

// ==============================================================================
// 非线性项：v × rot(v)（伪谱方法）
// ==============================================================================

// 计算非线性对流项 v × rot(v)，输入输出均在频谱空间
// 算法：1) 频谱空间计算旋度 2) 逆变换到实空间 3) 实空间叉乘 4) 正变换回频谱空间
//
// 注意：V1_r/V2_r/V3_r 和 rot1_r/rot2_r/rot3_r 作为工作数组会被修改
void compute_nonlinear(heffte::fft3d_r2c<heffte::backend::fftw>& fft,
                       const Cvec& V1_c, const Cvec& V2_c, const Cvec& V3_c,
                       Cvec& nl1_c, Cvec& nl2_c, Cvec& nl3_c,
                       Rvec& V1_r, Rvec& V2_r, Rvec& V3_r,
                       Cvec& rot1_c, Cvec& rot2_c, Cvec& rot3_c,
                       Rvec& rot1_r, Rvec& rot2_r, Rvec& rot3_r,
                       int nx, int ny, int nz,
                       const heffte::box3d<>& inbox_r,
                       const heffte::box3d<>& outbox_c) {

    // 1. 频谱空间计算旋度（rot_c = ik × V_c，与 V_c 同 N 因子）
    compute_rot(V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c, nx, ny, nz, outbox_c);

    // 2. 逆变换到实空间（scale::full 自动除以 N，得到物理值）
    //    heFFTe backward 不破坏输入，无需复制
    fft.backward(V1_c.data(), V1_r.data(), heffte::scale::full);
    fft.backward(V2_c.data(), V2_r.data(), heffte::scale::full);
    fft.backward(V3_c.data(), V3_r.data(), heffte::scale::full);
    fft.backward(rot1_c.data(), rot1_r.data(), heffte::scale::full);
    fft.backward(rot2_c.data(), rot2_r.data(), heffte::scale::full);
    fft.backward(rot3_c.data(), rot3_r.data(), heffte::scale::full);

    // 3. 实空间计算 v × rot(v)，结果覆盖 rot_r（复用为工作数组）
    size_t nr = inbox_r.count();
    #pragma omp parallel for
    for (size_t idx = 0; idx < nr; ++idx) {
        double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
        double w1 = rot1_r[idx], w2 = rot2_r[idx], w3 = rot3_r[idx];
        rot1_r[idx] = v2 * w3 - v3 * w2;  // (v×rot)_x
        rot2_r[idx] = v3 * w1 - v1 * w3;  // (v×rot)_y
        rot3_r[idx] = v1 * w2 - v2 * w1;  // (v×rot)_z
    }

    // 4. 正变换回频谱空间（scale::none，得到 DFT = N × FFT_norm，与 V_c 一致）
    fft.forward(rot1_r.data(), nl1_c.data(), heffte::scale::none);
    fft.forward(rot2_r.data(), nl2_c.data(), heffte::scale::none);
    fft.forward(rot3_r.data(), nl3_c.data(), heffte::scale::none);
}

// ==============================================================================
// 完整右端项：RHS = v×rot(v) + P∆v + f，投影到无散空间
// ==============================================================================

void compute_rhs(heffte::fft3d_r2c<heffte::backend::fftw>& fft,
                 const Cvec& V1_c, const Cvec& V2_c, const Cvec& V3_c,
                 Cvec& rhs1_c, Cvec& rhs2_c, Cvec& rhs3_c,
                 Rvec& V1_r, Rvec& V2_r, Rvec& V3_r,
                 Rvec& work_r1, Rvec& work_r2, Rvec& work_r3,
                 Cvec& rot1_c, Cvec& rot2_c, Cvec& rot3_c,
                 Rvec& rot1_r, Rvec& rot2_r, Rvec& rot3_r,
                 Cvec& nl1_c, Cvec& nl2_c, Cvec& nl3_c,
                 Cvec& visc1_c, Cvec& visc2_c, Cvec& visc3_c,
                 Cvec& f1_c, Cvec& f2_c, Cvec& f3_c,
                 Cvec& div_c, Cvec& phi_c,
                 int nx, int ny, int nz,
                 double t, double dx, double dy, double dz,
                 const heffte::box3d<>& inbox_r,
                 const heffte::box3d<>& outbox_c) {

    // 1. 非线性项 v × rot(v)（伪谱）
    compute_nonlinear(fft, V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                      V1_r, V2_r, V3_r,
                      rot1_c, rot2_c, rot3_c,
                      rot1_r, rot2_r, rot3_r,
                      nx, ny, nz, inbox_r, outbox_c);

    // 2. 粘性项 P∆V = -k² V（频谱空间，P=1）
    compute_viscous(V1_c, visc1_c, nx, ny, nz, outbox_c);
    compute_viscous(V2_c, visc2_c, nx, ny, nz, outbox_c);
    compute_viscous(V3_c, visc3_c, nx, ny, nz, outbox_c);

    // 3. 外力项 f（实空间 → 频谱空间）
    for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                size_t idx = box_idx(inbox_r, i, j, k);
                double x = i * dx, y = j * dy, z = k * dz;
                work_r1[idx] = func_f1(x, y, z, t);
                work_r2[idx] = func_f2(x, y, z, t);
                work_r3[idx] = func_f3(x, y, z, t);
            }
        }
    }
    // scale::none：f_c = DFT(f) = N * FFT_norm(f)，与 V_c 约定一致
    fft.forward(work_r1.data(), f1_c.data(), heffte::scale::none);
    fft.forward(work_r2.data(), f2_c.data(), heffte::scale::none);
    fft.forward(work_r3.data(), f3_c.data(), heffte::scale::none);

    // 4. 组合各项：RHS = v×rot(v) + P∆V + f
    size_t nc = outbox_c.count();
    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        rhs1_c[i] = nl1_c[i] + visc1_c[i] + f1_c[i];
        rhs2_c[i] = nl2_c[i] + visc2_c[i] + f2_c[i];
        rhs3_c[i] = nl3_c[i] + visc3_c[i] + f3_c[i];
    }

    // 5. 投影：使 RHS 无散（P 算子）
    make_div_free(rhs1_c, rhs2_c, rhs3_c, div_c, phi_c, nx, ny, nz, outbox_c);
}

// ==============================================================================
// RK4 时间积分（频谱空间）
// ==============================================================================

void rk4_step(heffte::fft3d_r2c<heffte::backend::fftw>& fft,
              Cvec& V1_c, Cvec& V2_c, Cvec& V3_c,
              Rvec& V1_r, Rvec& V2_r, Rvec& V3_r,
              Rvec& work_r1, Rvec& work_r2, Rvec& work_r3,
              Cvec& k1_v1, Cvec& k1_v2, Cvec& k1_v3,
              Cvec& k2_v1, Cvec& k2_v2, Cvec& k2_v3,
              Cvec& k3_v1, Cvec& k3_v2, Cvec& k3_v3,
              Cvec& k4_v1, Cvec& k4_v2, Cvec& k4_v3,
              Cvec& tmp_v1, Cvec& tmp_v2, Cvec& tmp_v3,
              Cvec& rot1_c, Cvec& rot2_c, Cvec& rot3_c,
              Rvec& rot1_r, Rvec& rot2_r, Rvec& rot3_r,
              Cvec& nl1_c, Cvec& nl2_c, Cvec& nl3_c,
              Cvec& visc1_c, Cvec& visc2_c, Cvec& visc3_c,
              Cvec& f1_c, Cvec& f2_c, Cvec& f3_c,
              Cvec& div_c, Cvec& phi_c,
              int nx, int ny, int nz,
              const heffte::box3d<>& inbox_r,
              const heffte::box3d<>& outbox_c,
              double dt, double t,
              double dx, double dy, double dz) {

    size_t nc = outbox_c.count();

    // k1 = RHS(V^n, t)
    compute_rhs(fft, V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c,
                nx, ny, nz, t, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        tmp_v1[i] = V1_c[i] + 0.5 * dt * k1_v1[i];
        tmp_v2[i] = V2_c[i] + 0.5 * dt * k1_v2[i];
        tmp_v3[i] = V3_c[i] + 0.5 * dt * k1_v3[i];
    }

    // k2 = RHS(V^n + dt/2 * k1, t + dt/2)
    compute_rhs(fft, tmp_v1, tmp_v2, tmp_v3, k2_v1, k2_v2, k2_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c,
                nx, ny, nz, t + 0.5 * dt, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        tmp_v1[i] = V1_c[i] + 0.5 * dt * k2_v1[i];
        tmp_v2[i] = V2_c[i] + 0.5 * dt * k2_v2[i];
        tmp_v3[i] = V3_c[i] + 0.5 * dt * k2_v3[i];
    }

    // k3 = RHS(V^n + dt/2 * k2, t + dt/2)
    compute_rhs(fft, tmp_v1, tmp_v2, tmp_v3, k3_v1, k3_v2, k3_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c,
                nx, ny, nz, t + 0.5 * dt, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        tmp_v1[i] = V1_c[i] + dt * k3_v1[i];
        tmp_v2[i] = V2_c[i] + dt * k3_v2[i];
        tmp_v3[i] = V3_c[i] + dt * k3_v3[i];
    }

    // k4 = RHS(V^n + dt * k3, t + dt)
    compute_rhs(fft, tmp_v1, tmp_v2, tmp_v3, k4_v1, k4_v2, k4_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c,
                nx, ny, nz, t + dt, dx, dy, dz, inbox_r, outbox_c);

    // V^{n+1} = V^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        V1_c[i] += (dt / 6.0) * (k1_v1[i] + 2.0 * k2_v1[i] + 2.0 * k3_v1[i] + k4_v1[i]);
        V2_c[i] += (dt / 6.0) * (k1_v2[i] + 2.0 * k2_v2[i] + 2.0 * k3_v2[i] + k4_v2[i]);
        V3_c[i] += (dt / 6.0) * (k1_v3[i] + 2.0 * k2_v3[i] + 2.0 * k3_v3[i] + k4_v3[i]);
    }
}

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int nx = 128, ny = 128, nz = 128;
    const double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
    const double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    const ptrdiff_t nt_total = 20000;  // 总步数（用于计算 dt）
    const ptrdiff_t nt_run   = 10;     // 实际运行步数（验证用）
    const double T  = 1.0;
    const double dt = T / nt_total;    // dt = 5e-5

    int max_threads = omp_get_max_threads();
    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - heFFTe Version (2D Pencil)\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz << "\n";
        cout << "Domain: [0, 2π]^3\n";
        cout << "MPI processes: " << nprocs << "\n";
        cout << "OpenMP threads/process: " << max_threads << "\n";
        cout << "Steps: " << nt_run << " / " << nt_total << ", dt = " << dt << "\n";
        cout << "============================================================\n";
    }

    // ------------------------------------------------------------------
    // 初始化 heFFTe（自动 2D pencil 分解）
    // ------------------------------------------------------------------
    // 注意：heFFTe fft 对象析构时会释放内部 MPI 通信子，
    //       必须在 MPI_Finalize() 之前完成析构，故用块作用域包围。
    {
    heffte::box3d<> world_r = {{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
    heffte::box3d<> world_c = {{0, 0, 0}, {nx - 1, ny - 1, nz / 2}};  // R2C 压缩 z

    auto proc_grid   = heffte::proc_setup_min_surface(world_r, nprocs);
    auto all_inboxes = heffte::split_world(world_r, proc_grid);
    auto all_outboxes = heffte::split_world(world_c, proc_grid);

    heffte::box3d<> inbox_r  = all_inboxes[rank];
    heffte::box3d<> outbox_c = all_outboxes[rank];

    if (rank == 0) {
        cout << "heFFTe pencil grid: "
             << proc_grid[0] << "x" << proc_grid[1] << "x" << proc_grid[2] << "\n";
    }

    // 创建 FFT 引擎（使用 FFTW 后端，R2C 沿 dim=2 即 z 方向）
    heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox_r, outbox_c, 2, MPI_COMM_WORLD);

    size_t nr = inbox_r.count();
    size_t nc = outbox_c.count();

    // ------------------------------------------------------------------
    // 分配内存
    // ------------------------------------------------------------------
    Rvec V1_r(nr), V2_r(nr), V3_r(nr);
    Cvec V1_c(nc), V2_c(nc), V3_c(nc);

    Rvec work_r1(nr), work_r2(nr), work_r3(nr);
    Rvec rot1_r(nr), rot2_r(nr), rot3_r(nr);

    Cvec rot1_c(nc), rot2_c(nc), rot3_c(nc);
    Cvec nl1_c(nc),  nl2_c(nc),  nl3_c(nc);
    Cvec visc1_c(nc), visc2_c(nc), visc3_c(nc);
    Cvec f1_c(nc),   f2_c(nc),   f3_c(nc);
    Cvec div_c(nc),  phi_c(nc);

    Cvec k1_v1(nc), k1_v2(nc), k1_v3(nc);
    Cvec k2_v1(nc), k2_v2(nc), k2_v3(nc);
    Cvec k3_v1(nc), k3_v2(nc), k3_v3(nc);
    Cvec k4_v1(nc), k4_v2(nc), k4_v3(nc);
    Cvec tmp_v1(nc), tmp_v2(nc), tmp_v3(nc);

    // ------------------------------------------------------------------
    // 初始条件（t=0）
    // ------------------------------------------------------------------
    if (rank == 0) cout << "Setting initial conditions...\n";
    for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                size_t idx = box_idx(inbox_r, i, j, k);
                double x = i * dx, y = j * dy, z = k * dz;
                V1_r[idx] = func_V1(x, y, z, 0.0);
                V2_r[idx] = func_V2(x, y, z, 0.0);
                V3_r[idx] = func_V3(x, y, z, 0.0);
            }
        }
    }

    // 正变换：scale::none → V_c = DFT(V_r) = N × FFT_norm(V_phys)
    fft.forward(V1_r.data(), V1_c.data(), heffte::scale::none);
    fft.forward(V2_r.data(), V2_c.data(), heffte::scale::none);
    fft.forward(V3_r.data(), V3_c.data(), heffte::scale::none);

    // 投影初始条件到无散空间
    if (rank == 0) cout << "Projecting initial condition to divergence-free space...\n";
    make_div_free(V1_c, V2_c, V3_c, div_c, phi_c, nx, ny, nz, outbox_c);

    // ------------------------------------------------------------------
    // 误差计算辅助函数
    // ------------------------------------------------------------------
    auto compute_error = [&](double time) -> pair<double, double> {
        // backward(scale::full) 自动除以 N，得到物理值
        fft.backward(V1_c.data(), V1_r.data(), heffte::scale::full);
        fft.backward(V2_c.data(), V2_r.data(), heffte::scale::full);
        fft.backward(V3_c.data(), V3_r.data(), heffte::scale::full);

        double local_err = 0.0, local_max = 0.0;
        for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
            for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
                for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                    size_t idx = box_idx(inbox_r, i, j, k);
                    double x = i * dx, y = j * dy, z = k * dz;
                    double d1 = V1_r[idx] - func_V1(x, y, z, time);
                    double d2 = V2_r[idx] - func_V2(x, y, z, time);
                    double d3 = V3_r[idx] - func_V3(x, y, z, time);
                    double e  = d1 * d1 + d2 * d2 + d3 * d3;
                    local_err += e;
                    local_max  = max(local_max, sqrt(e));
                }
            }
        }
        double global_err = 0.0, global_max = 0.0;
        MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return {sqrt(global_err * dx * dy * dz), global_max};
    };

    // ------------------------------------------------------------------
    // 打印初始误差
    // ------------------------------------------------------------------
    {
        auto [err0, err0inf] = compute_error(0.0);
        double div0 = compute_div_max(V1_c, V2_c, V3_c, div_c, nx, ny, nz, outbox_c);
        if (rank == 0) {
            cout << "\nInitial condition (t=0):\n";
            cout << "  L2 error:   " << scientific << err0    << "\n";
            cout << "  L∞ error:   " << err0inf   << "\n";
            cout << "  max|div V|: " << div0       << "\n\n";
        }
    }

    // ------------------------------------------------------------------
    // 时间推进（RK4）
    // ------------------------------------------------------------------
    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Time Integration (RK4)\n";
        cout << "============================================================\n";
        cout << setw(6)  << "Step"
             << setw(14) << "Wall(s)"
             << setw(16) << "L2 Error"
             << setw(16) << "max|div V|\n";
        cout << "------------------------------------------------------------\n";
    }

    double t_wall_total = 0.0;  // 累计挂钟时间（秒）

    for (ptrdiff_t it = 0; it <= nt_run; ++it) {
        double t_cur = it * dt;

        auto [errL2, errLinf] = compute_error(t_cur);
        double div_max = compute_div_max(V1_c, V2_c, V3_c, div_c, nx, ny, nz, outbox_c);

        if (rank == 0) {
            cout << setw(6) << it
                 << setw(14) << fixed << setprecision(4) << t_wall_total
                 << setw(16) << scientific << setprecision(4) << errL2
                 << setw(16) << div_max << "\n";
        }

        if (it < nt_run) {
            double t0 = MPI_Wtime();
            rk4_step(fft, V1_c, V2_c, V3_c,
                     V1_r, V2_r, V3_r,
                     work_r1, work_r2, work_r3,
                     k1_v1, k1_v2, k1_v3,
                     k2_v1, k2_v2, k2_v3,
                     k3_v1, k3_v2, k3_v3,
                     k4_v1, k4_v2, k4_v3,
                     tmp_v1, tmp_v2, tmp_v3,
                     rot1_c, rot2_c, rot3_c,
                     rot1_r, rot2_r, rot3_r,
                     nl1_c, nl2_c, nl3_c,
                     visc1_c, visc2_c, visc3_c,
                     f1_c, f2_c, f3_c,
                     div_c, phi_c,
                     nx, ny, nz, inbox_r, outbox_c,
                     dt, t_cur, dx, dy, dz);
            double step_time = MPI_Wtime() - t0;
            double global_step;
            MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            t_wall_total += global_step;
        }
    }

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Timing Summary\n";
        cout << "------------------------------------------------------------\n";
        cout << "  Total steps:     " << nt_run << "\n";
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s\n";
        cout << "  Avg per step:    " << t_wall_total / nt_run << " s\n";
        cout << "============================================================\n";
    }
    }  // heFFTe fft 对象在此析构（MPI_Finalize 之前）

    MPI_Finalize();
    return 0;
}
