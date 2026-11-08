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
#include <fftw3.h>
#include "common_mode.h"

using namespace std;

// 运行模式与分阶段计时（见 common_mode.h）
//   g_with_force = true  (verify)：每次 RHS 计算解析外力 f（原始行为）
//   g_with_force = false (bench) ：f = 0，跳过外力填充、3 次正向 FFT 与相加
static bool g_with_force = true;
static nsmode::CpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;
using Cvec = vector<complex<double>>;
using Rvec = vector<double>;

// heFFTe 常驻工作区（main 中分配一次，所有正/逆变换共用）。不传工作区时，heFFTe 每次调用都会
// 新建并清零一个 size_workspace() 大小的缓冲区，在单进程多线程时占 FFT 时间的很大一部分。
static complex<double>* g_fft_ws = nullptr;

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

// ==============================================================================
// 频谱算子与时间推进
//
// 时间格式：经典 RK4，按累加形式存储（不存 k1..k4，k 只在一次循环内使用）：
//   k    = P[ nl + (-k^2 V) (+ f，仅 verify) ]
//   阶段1: acc  = V_orig + dt/6*k;   V = V_orig + dt/2*k
//   阶段2: acc += dt/3*k;            V = V_orig + dt/2*k
//   阶段3: acc += dt/3*k;            V = V_orig + dt*k
//   阶段4:                           V = acc + dt/6*k
// 谱空间循环用 double[2] 视图访问 complex<double>（C++11 保证二者布局兼容）。
// ==============================================================================

using C2 = double[2];
static inline C2* c2(Cvec& v) { return reinterpret_cast<C2*>(v.data()); }

// 所有全尺寸数组（main 中分配一次）
struct Fields {
    Cvec V[3];      // 阶段值（谱空间）
    Cvec rot[3];    // 旋度；正变换之后存放非线性项 nl
    Cvec rhs[3];    // -k^2 V（粘性项，由旋度循环同时写入）
    Cvec orig[3];   // V^n
    Cvec acc[3];    // RK4 累加量
    Rvec Vr[3];     // 实空间速度；verify 时在叉乘之后用来存放外力 f
    Rvec rotr[3];   // 实空间旋度，叉乘结果覆盖于此
};

enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间，in-place；仅用于初始条件）
void make_div_free(Cvec& V1_c, Cvec& V2_c, Cvec& V3_c,
                   int nx, int ny, const heffte::box3d<>& box_c) {
    C2 *V1 = c2(V1_c), *V2 = c2(V2_c), *V3 = c2(V3_c);
    #pragma omp parallel for collapse(3)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx), ky = wave_ky(j, ny), kz = wave_kz(k);
                double div_r = -(kx * V1[idx][1] + ky * V2[idx][1] + kz * V3[idx][1]);
                double div_i =   kx * V1[idx][0] + ky * V2[idx][0] + kz * V3[idx][0];
                double k2 = kx * kx + ky * ky + kz * kz;
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-14) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                // V -= i*k*phi（即 ∇phi = i*k*phi）
                V1[idx][0] -= -kx * phi_i;  V1[idx][1] -= kx * phi_r;
                V2[idx][0] -= -ky * phi_i;  V2[idx][1] -= ky * phi_r;
                V3[idx][0] -= -kz * phi_i;  V3[idx][1] -= kz * phi_r;
            }
        }
    }
}

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
    // 谱系数为非归一化 DFT（= N × 归一化系数），除以 N 后才是物理量，与其他版本可比
    return global_max / ((double)nx * ny * nz);
}

// 每个 RHS 的第一步（谱空间，一个循环）：rot = i·k × V，同时 rhs = -k² V
void compute_rot_and_viscous(Fields& F, int nx, int ny, const heffte::box3d<>& box_c) {
    C2 *V[3]   = { c2(F.V[0]),   c2(F.V[1]),   c2(F.V[2]) };
    C2 *rot[3] = { c2(F.rot[0]), c2(F.rot[1]), c2(F.rot[2]) };
    C2 *rhs[3] = { c2(F.rhs[0]), c2(F.rhs[1]), c2(F.rhs[2]) };
    #pragma omp parallel for collapse(3)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx), ky = wave_ky(j, ny), kz = wave_kz(k);
                double k2 = kx * kx + ky * ky + kz * kz;
                // i * kx * (a + ib) = -kx*b + i*kx*a
                rot[0][idx][0] = -(ky * V[2][idx][1] - kz * V[1][idx][1]);
                rot[0][idx][1] =   ky * V[2][idx][0] - kz * V[1][idx][0];
                rot[1][idx][0] = -(kz * V[0][idx][1] - kx * V[2][idx][1]);
                rot[1][idx][1] =   kz * V[0][idx][0] - kx * V[2][idx][0];
                rot[2][idx][0] = -(kx * V[1][idx][1] - ky * V[0][idx][1]);
                rot[2][idx][1] =   kx * V[1][idx][0] - ky * V[0][idx][0];
                for (int c = 0; c < 3; ++c) {
                    rhs[c][idx][0] = -k2 * V[c][idx][0];
                    rhs[c][idx][1] = -k2 * V[c][idx][1];
                }
            }
        }
    }
}

// RK4 的一个阶段：在当前阶段值 V 处计算 RHS，并把 V 推进到下一阶段值（阶段 4 之后为 V^{n+1}）。
void rk4_stage(heffte::fft3d_r2c<heffte::backend::fftw>& fft, Fields& F,
               int nx, int ny, const heffte::box3d<>& inbox_r, const heffte::box3d<>& outbox_c,
               double dt, double t, double dx, double dy, double dz, int stage) {
    // 1. 旋度 + 粘性项（谱空间）
    compute_rot_and_viscous(F, nx, ny, outbox_c);
    g_timer.mark(PH_SPEC);

    // 2. 逆变换到实空间（scale::full 自动除以 N，得到物理值）
    for (int c = 0; c < 3; ++c) fft.backward(F.V[c].data(),   F.Vr[c].data(),   g_fft_ws, heffte::scale::full);
    for (int c = 0; c < 3; ++c) fft.backward(F.rot[c].data(), F.rotr[c].data(), g_fft_ws, heffte::scale::full);
    g_timer.mark(PH_FFT);

    // 3. 实空间叉乘 v × rot(v)，覆盖 rotr
    size_t nr = inbox_r.count();
    #pragma omp parallel for
    for (size_t idx = 0; idx < nr; ++idx) {
        double v1 = F.Vr[0][idx], v2 = F.Vr[1][idx], v3 = F.Vr[2][idx];
        double w1 = F.rotr[0][idx], w2 = F.rotr[1][idx], w3 = F.rotr[2][idx];
        F.rotr[0][idx] = v2 * w3 - v3 * w2;  // (v×rot)_x
        F.rotr[1][idx] = v3 * w1 - v1 * w3;  // (v×rot)_y
        F.rotr[2][idx] = v1 * w2 - v2 * w1;  // (v×rot)_z
    }
    g_timer.mark(PH_PHYS);

    // 4. 正变换：叉乘 → rot（scale::none，nl = DFT，与 V_c 一致）
    for (int c = 0; c < 3; ++c) fft.forward(F.rotr[c].data(), F.rot[c].data(), g_fft_ws, heffte::scale::none);
    g_timer.mark(PH_FFT);

    // 5. 外力 [仅 verify]：填入空闲的 Vr，正变换到 V（scale::none）
    if (g_with_force) {
        for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
            for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
                for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                    size_t idx = box_idx(inbox_r, i, j, k);
                    double x = i * dx, y = j * dy, z = k * dz;
                    F.Vr[0][idx] = func_f1(x, y, z, t);
                    F.Vr[1][idx] = func_f2(x, y, z, t);
                    F.Vr[2][idx] = func_f3(x, y, z, t);
                }
            }
        }
        for (int c = 0; c < 3; ++c) fft.forward(F.Vr[c].data(), F.V[c].data(), g_fft_ws, heffte::scale::none);
        g_timer.mark(PH_FORCE);
    }

    // 6. 求和 + 投影 + RK4 更新（一个循环，k 只存在于局部变量中）
    C2 *V[3]   = { c2(F.V[0]),    c2(F.V[1]),    c2(F.V[2]) };
    C2 *rot[3] = { c2(F.rot[0]),  c2(F.rot[1]),  c2(F.rot[2]) };
    C2 *rhs[3] = { c2(F.rhs[0]),  c2(F.rhs[1]),  c2(F.rhs[2]) };
    C2 *acc[3] = { c2(F.acc[0]),  c2(F.acc[1]),  c2(F.acc[2]) };
    C2 *base[3]= { c2(F.orig[0]), c2(F.orig[1]), c2(F.orig[2]) };
    double a_acc = 0.0, a_v = 0.0;
    int acc_mode = ACC_ADD;
    switch (stage) {
        case 1:  acc_mode = ACC_INIT; a_acc = dt / 6.0; a_v = 0.5 * dt; break;
        case 2:                       a_acc = dt / 3.0; a_v = 0.5 * dt; break;
        case 3:                       a_acc = dt / 3.0; a_v = dt;       break;
        default:
            acc_mode = ACC_KEEP; a_v = dt / 6.0;
            for (int c = 0; c < 3; ++c) base[c] = acc[c];
            break;
    }
    const bool force = g_with_force;
    const heffte::box3d<>& box_c = outbox_c;
    #pragma omp parallel for collapse(3)
    for (int i = box_c.low[0]; i <= box_c.high[0]; ++i) {
        for (int j = box_c.low[1]; j <= box_c.high[1]; ++j) {
            for (int k = box_c.low[2]; k <= box_c.high[2]; ++k) {
                size_t idx = box_idx(box_c, i, j, k);
                double kx = wave_kx(i, nx), ky = wave_ky(j, ny), kz = wave_kz(k);
                double k2 = kx * kx + ky * ky + kz * kz;
                // RHS = nl + visc + f（与原代码相同的运算顺序）
                double q[3][2];
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 2; ++r) {
                        double v = rot[c][idx][r] + rhs[c][idx][r];
                        if (force) v += V[c][idx][r];
                        q[c][r] = v;
                    }
                // 投影
                double div_r = -(kx * q[0][1] + ky * q[1][1] + kz * q[2][1]);
                double div_i =   kx * q[0][0] + ky * q[1][0] + kz * q[2][0];
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-14) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                q[0][0] -= -kx * phi_i;  q[0][1] -= kx * phi_r;
                q[1][0] -= -ky * phi_i;  q[1][1] -= ky * phi_r;
                q[2][0] -= -kz * phi_i;  q[2][1] -= kz * phi_r;
                // RK4
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 2; ++r) {
                        double bv = base[c][idx][r];
                        if      (acc_mode == ACC_INIT) acc[c][idx][r]  = bv + a_acc * q[c][r];
                        else if (acc_mode == ACC_ADD)  acc[c][idx][r] += a_acc * q[c][r];
                        V[c][idx][r] = bv + a_v * q[c][r];
                    }
            }
        }
    }
    g_timer.mark(PH_SPEC);
}

void rk4_step(heffte::fft3d_r2c<heffte::backend::fftw>& fft, Fields& F,
              int nx, int ny, const heffte::box3d<>& inbox_r, const heffte::box3d<>& outbox_c,
              double dt, double t, double dx, double dy, double dz) {
    // 保存 V^n
    for (int c = 0; c < 3; ++c) F.orig[c] = F.V[c];
    g_timer.mark(PH_SPEC);

    rk4_stage(fft, F, nx, ny, inbox_r, outbox_c, dt, t,            dx, dy, dz, 1);
    rk4_stage(fft, F, nx, ny, inbox_r, outbox_c, dt, t + 0.5 * dt, dx, dy, dz, 2);
    rk4_stage(fft, F, nx, ny, inbox_r, outbox_c, dt, t + 0.5 * dt, dx, dy, dz, 3);
    rk4_stage(fft, F, nx, ny, inbox_r, outbox_c, dt, t + dt,       dx, dy, dz, 4);
}

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 6) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " " << nsmode::usage_tail(true) << "\n";
            cerr << "  Example: mpirun -np 4 " << argv[0] << " 64 64 64 0.00001 100 4 bench\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    nsmode::RunArgs rargs;
    {
        string perr;
        if (!nsmode::parse_tail_args(argc, argv, 6, /*allow_threads=*/true, rargs, perr)) {
            if (rank == 0) cerr << "Error: " << perr << "\nUsage: " << argv[0] << " "
                                << nsmode::usage_tail(true) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const nsmode::RunMode mode = rargs.mode;
    g_with_force = (mode == nsmode::MODE_VERIFY);

    int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    double dt = atof(argv[4]);
    int nt_run = atoi(argv[5]);
    int nthreads = (rargs.omp_threads > 0) ? rargs.omp_threads : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    // heFFTe 的 FFTW 后端在首次执行时才创建 FFTW 计划，并沿用全局的线程设置；
    // 不调用这两行，每个进程内的 FFT 只用 1 个线程。需链接 libfftw3_omp（原因见 FFTW 版）。
    fftw_init_threads();
    fftw_plan_with_nthreads(nthreads);
    int max_threads = nthreads;

    const double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
    double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - heFFTe Version (2D Pencil)\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz
             << ", dt=" << scientific << dt << ", steps=" << nt_run << "\n";
        cout << "Domain: [0, 2π]^3\n";
        cout << "MPI processes: " << nprocs << "\n";
        cout << "OpenMP threads/process: " << max_threads << "\n";
        cout << "Run mode: " << nsmode::mode_name(mode)
             << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << "\n";
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
    // Library defaults for the fftw backend (alltoallv, use_reorder = true), unless
    // overridden by NS_HEFFTE_ALGO / NS_HEFFTE_REORDER.
    auto options = heffte::default_options<heffte::backend::fftw>();
    nsmode::heffte_options_from_env(options, rank);
    heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox_r, outbox_c, 2, MPI_COMM_WORLD, options);

    size_t nr = inbox_r.count();
    size_t nc = outbox_c.count();
    vector<complex<double>> fft_ws(fft.size_workspace());
    g_fft_ws = fft_ws.data();

    // ------------------------------------------------------------------
    // 分配内存
    // ------------------------------------------------------------------
    // 全尺寸数组：15 个复数（V、rot、rhs、V_orig、acc 各 3 个）+ 6 个实数（Vr、rotr 各 3 个），
    // bench 与 verify 相同（verify 的外力复用 Vr 与 V）
    Fields F;
    for (int c = 0; c < 3; ++c) {
        F.V[c].resize(nc);   F.rot[c].resize(nc);  F.rhs[c].resize(nc);
        F.orig[c].resize(nc); F.acc[c].resize(nc);
        F.Vr[c].resize(nr);  F.rotr[c].resize(nr);
    }
    Rvec &V1_r = F.Vr[0], &V2_r = F.Vr[1], &V3_r = F.Vr[2];
    Cvec &V1_c = F.V[0],  &V2_c = F.V[1],  &V3_c = F.V[2];

    {
        const double GB = 1024.0 * 1024.0 * 1024.0;
        double mx[3] = { (double)(nc * sizeof(complex<double>)), (double)(nr * sizeof(double)),
                         (double)(fft.size_workspace() * sizeof(complex<double>)) }, g[3];
        MPI_Allreduce(mx, g, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Memory report (per process, max over ranks):\n");
            printf("  Full-size arrays: 15 complex of %.3f GB, 6 real of %.3f GB\n", g[0] / GB, g[1] / GB);
            printf("  heFFTe workspace: %.3f GB\n", g[2] / GB);
            fflush(stdout);
        }
    }

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
    fft.forward(V1_r.data(), V1_c.data(), g_fft_ws, heffte::scale::none);
    fft.forward(V2_r.data(), V2_c.data(), g_fft_ws, heffte::scale::none);
    fft.forward(V3_r.data(), V3_c.data(), g_fft_ws, heffte::scale::none);

    // 投影初始条件到无散空间
    if (rank == 0) cout << "Projecting initial condition to divergence-free space...\n";
    make_div_free(V1_c, V2_c, V3_c, nx, ny, outbox_c);

    // ------------------------------------------------------------------
    // 误差计算辅助函数
    // ------------------------------------------------------------------
    // V_c → V_r：backward(scale::full) 自动除以 N，得到物理值（不破坏 V_c）
    auto to_physical = [&]() {
        fft.backward(V1_c.data(), V1_r.data(), g_fft_ws, heffte::scale::full);
        fft.backward(V2_c.data(), V2_r.data(), g_fft_ws, heffte::scale::full);
        fft.backward(V3_c.data(), V3_r.data(), g_fft_ws, heffte::scale::full);
    };

    auto compute_error = [&](double time) -> pair<double, double> {
        to_physical();

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

    // 总动能 0.5·Σ|v|²（物理空间逐点求和，bench 模式下的结果合理性检查）
    auto compute_energy = [&]() -> double {
        to_physical();
        double local_e = 0.0;
        #pragma omp parallel for reduction(+:local_e)
        for (size_t idx = 0; idx < nr; ++idx)
            local_e += V1_r[idx]*V1_r[idx] + V2_r[idx]*V2_r[idx] + V3_r[idx]*V3_r[idx];
        double global_e = 0.0;
        MPI_Reduce(&local_e, &global_e, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        return 0.5 * global_e;
    };

    // ------------------------------------------------------------------
    // 时间推进（RK4）
    // ------------------------------------------------------------------
    double t_wall_total = 0.0;

    for (ptrdiff_t it = 0; it < nt_run; ++it) {
        double t_cur = it * dt;
        double t0 = MPI_Wtime();
        g_timer.start();
        rk4_step(fft, F, nx, ny, inbox_r, outbox_c, dt, t_cur, dx, dy, dz);
        double step_time = MPI_Wtime() - t0;
        double global_step;
        MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_wall_total += global_step;
    }

    double phase_max[nsmode::PH_COUNT];
    nsmode::reduce_phases_max(g_timer.acc, phase_max);

    // 以下诊断均不计入计时
    {
        double t_final = nt_run * dt;
        // compute_div_max 需要一个暂存数组：复用 rhs1（计时结束后已无用）
        double max_div = compute_div_max(V1_c, V2_c, V3_c, F.rhs[0], nx, ny, nz, outbox_c);
        if (mode == nsmode::MODE_VERIFY) {
            auto [errL2, errLinf] = compute_error(t_final);
            (void)errLinf;
            if (rank == 0)
                cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(4) << errL2 << "\n";
        } else {
            double ekin = compute_energy();
            if (rank == 0)
                cout << "  Kinetic energy 0.5*sum|v|^2 (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(10) << ekin << "\n";
        }
        if (rank == 0)
            cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << max_div << "\n";
    }

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Timing Summary\n";
        cout << "------------------------------------------------------------\n";
        cout << "  Total steps:     " << nt_run << "\n";
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s\n";
        cout << "  Avg per step:    " << t_wall_total / nt_run << " s\n";
        cout << flush;
        nsmode::print_phase_summary(phase_max, t_wall_total, nt_run, mode);
        cout << "============================================================\n";
    }
    }  // heFFTe fft 对象在此析构（MPI_Finalize 之前）

    MPI_Finalize();
    return 0;
}
