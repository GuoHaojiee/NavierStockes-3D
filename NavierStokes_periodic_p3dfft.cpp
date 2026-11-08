/**
 * Navier-Stokes 求解器 - p3dfft 版本（2D pencil 分解）
 *
 * 基于 NavierStokes_periodic_fftw.cpp 的核心算法，使用 p3dfft C 接口代替 FFTW MPI。
 *
 * 与 FFTW 版本的关键区别：
 *   - p3dfft 支持 2D pencil 分解（非 FFTW 的 1D slab），可更好地利用多节点并行
 *   - R2C 压缩第一维（x 方向）：kx = 0..nx/2（非负），ky/kz 需折叠处理
 *   - 数据布局：Fortran 列主序（x 变化最快），使用 1-based 全局索引
 *   - p3dfft btran_c2r 会销毁输入数组：时间推进中逆变换之后不再使用其谱空间内容；诊断时先复制
 *   - 归一化约定：所有频谱量均为 DFT（非归一化），即 V_c = DFT(V_phys)
 *     逆变换后除以 N=nx*ny*nz 得到物理值：V_phys = btran(V_c_copy) / N
 *
 * 编译：参见 Makefile_p3dfft
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <fftw3.h>
#include <p3dfft.h>
#include "common_mode.h"

using namespace std;

// 运行模式与分阶段计时（见 common_mode.h）
//   g_with_force = true  (verify)：每次 RHS 计算解析外力 f（原始行为）
//   g_with_force = false (bench) ：f = 0，跳过外力填充、3 次正向 FFT 与相加
static bool g_with_force = true;
static nsmode::CpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;

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
    double d2_xy = (t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) *
                   (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * cos(6 * z);
    double d2_z  = -(t * t + 1) * 36 * exp(sin(3 * x + 3 * y)) * cos(6 * z);
    return d2_xy + d2_xy + d2_z;  // d²/dx² + d²/dy² + d²/dz²
}
double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x, y, z, t);
}
double func_laplace_V3(double x, double y, double z, double t) {
    double c = cos(3 * x + 3 * y), s = sin(3 * x + 3 * y);
    double d2_xy = -(t * t + 1) * 9 * exp(s) * c *
                   ((c * c - s) - (2 * s + 1)) * sin(6 * z);
    double d2_z  = (t * t + 1) * 36 * exp(s) * c * sin(6 * z);
    return d2_xy + d2_xy + d2_z;
}

double func_rot1(double x, double y, double z, double t) {
    double c = cos(3 * x + 3 * y), s = sin(3 * x + 3 * y);
    double dv3_dy = -(t * t + 1) * 3 * exp(s) * (c * c - s) * sin(6 * z);
    double dv2_dz = -(t * t + 1) * 6 * exp(s) * sin(6 * z);
    return dv3_dy - dv2_dz;
}
double func_rot2(double x, double y, double z, double t) {
    double c = cos(3 * x + 3 * y), s = sin(3 * x + 3 * y);
    double dv1_dz = -(t * t + 1) * 6 * exp(s) * sin(6 * z);
    double dv3_dx = -(t * t + 1) * 3 * exp(s) * (c * c - s) * sin(6 * z);
    return dv1_dz - dv3_dx;
}
double func_rot3(double, double, double, double) { return 0.0; }

double func_v_cross_rot1(double x, double y, double z, double t) {
    return func_V2(x,y,z,t)*func_rot3(x,y,z,t) - func_V3(x,y,z,t)*func_rot2(x,y,z,t);
}
double func_v_cross_rot2(double x, double y, double z, double t) {
    return func_V3(x,y,z,t)*func_rot1(x,y,z,t) - func_V1(x,y,z,t)*func_rot3(x,y,z,t);
}
double func_v_cross_rot3(double x, double y, double z, double t) {
    return func_V1(x,y,z,t)*func_rot2(x,y,z,t) - func_V2(x,y,z,t)*func_rot1(x,y,z,t);
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
    return func_dV1_dt(x,y,z,t) - func_laplace_V1(x,y,z,t)
           - func_v_cross_rot1(x,y,z,t) + func_grad_p1(x,y,z,t);
}
double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t) - func_laplace_V2(x,y,z,t)
           - func_v_cross_rot2(x,y,z,t) + func_grad_p2(x,y,z,t);
}
double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t) - func_laplace_V3(x,y,z,t)
           - func_v_cross_rot3(x,y,z,t) + func_grad_p3(x,y,z,t);
}

// ==============================================================================
// p3dfft 上下文与索引工具
// ==============================================================================

struct P3DCtx {
    int istart[3], iend[3], isize[3];   // 实空间本地范围（1-based 全局索引）
    int fstart[3], fend[3], fsize[3];   // 频谱空间本地范围
    int memsize[3];                     // 实空间内存大小（含 padding）
    int nx, ny, nz;
    int idx_offset;                     // 索引偏移：0-based 时为 0，1-based 时为 1
    double norm_inv;                    // 1/(nx*ny*nz)，用于逆变换后归一化
};

// 实空间线性索引（x 最快变化，Fortran 列主序）
// i, j, k 为 1-based 全局索引
inline size_t idx_r(const P3DCtx& ctx, int i, int j, int k) {
    return static_cast<size_t>(i - ctx.istart[0])
         + static_cast<size_t>(j - ctx.istart[1]) * ctx.isize[0]
         + static_cast<size_t>(k - ctx.istart[2]) * ctx.isize[0] * ctx.isize[1];
}

// 频谱空间线性索引（x 最快变化，Fortran 列主序）
inline size_t idx_c(const P3DCtx& ctx, int i, int j, int k) {
    return static_cast<size_t>(i - ctx.fstart[0])
         + static_cast<size_t>(j - ctx.fstart[1]) * ctx.fsize[0]
         + static_cast<size_t>(k - ctx.fstart[2]) * ctx.fsize[0] * ctx.fsize[1];
}

// 物理坐标（考虑 1-based 偏移）
inline double phys_x(const P3DCtx& ctx, int i, double dx) {
    return (i - ctx.idx_offset) * dx;
}
inline double phys_y(const P3DCtx& ctx, int j, double dy) {
    return (j - ctx.idx_offset) * dy;
}
inline double phys_z(const P3DCtx& ctx, int k, double dz) {
    return (k - ctx.idx_offset) * dz;
}

// 波数：p3dfft R2C 沿 x 方向（i 为 1-based，kx = i - offset 非负）
inline double wave_kx(const P3DCtx& ctx, int i) {
    return (double)(i - ctx.idx_offset);  // 0..nx/2，始终非负
}

// 波数折叠：ky, kz 为 C2C 方向，需要折叠负频率
// idx_0based = j - offset，fold to [-n/2, n/2]
inline double wave_fold(int idx_1based, int offset, int n) {
    int k = idx_1based - offset;  // 0-based: 0..n-1
    return (k <= n / 2) ? (double)k : (double)(k - n);
}

// ==============================================================================
// 频谱算子（p3dfft 坐标约定）
//
// 归一化约定：
//   V_c = DFT(V_phys)  [非归一化，= N * FFT_norm(V_phys)]
//   所有频谱量（rot、-k²V、nl、f）均为非归一化表示
//   逆变换后：V_phys = btran(V_c) / N（btran_c2r 会破坏输入）
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
static inline C2* c2(vector<complex<double>>& v) { return reinterpret_cast<C2*>(v.data()); }

// 所有全尺寸数组（main 中分配一次）
struct Fields {
    vector<complex<double>> V[3];     // 阶段值；btran 会破坏其内容，之后由阶段循环重写
    vector<complex<double>> rot[3];   // 旋度；ftran 之后存放非线性项 nl
    vector<complex<double>> rhs[3];   // -k^2 V（粘性项，由旋度循环同时写入）
    vector<complex<double>> orig[3];  // V^n
    vector<complex<double>> acc[3];   // RK4 累加量
    vector<double> Vr[3];             // 实空间速度；verify 时在叉乘之后用来存放外力 f
    vector<double> rotr[3];           // 实空间旋度，叉乘结果覆盖于此
};

enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间，in-place；仅用于初始条件）
// div = i·k·V，phi = div/(-k²)，V -= i·k·phi
void make_div_free(const P3DCtx& ctx,
                   vector<complex<double>>& V1_c,
                   vector<complex<double>>& V2_c,
                   vector<complex<double>>& V3_c) {
    int ny = ctx.ny, nz = ctx.nz;
    C2 *V1 = c2(V1_c), *V2 = c2(V2_c), *V3 = c2(V3_c);
    #pragma omp parallel for collapse(3)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);
                double div_r = -(kx * V1[idx][1] + ky * V2[idx][1] + kz * V3[idx][1]);
                double div_i =   kx * V1[idx][0] + ky * V2[idx][0] + kz * V3[idx][0];
                double k2 = kx * kx + ky * ky + kz * kz;
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-14) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                V1[idx][0] -= -kx * phi_i;  V1[idx][1] -= kx * phi_r;
                V2[idx][0] -= -ky * phi_i;  V2[idx][1] -= ky * phi_r;
                V3[idx][0] -= -kz * phi_i;  V3[idx][1] -= kz * phi_r;
            }
        }
    }
}

// 计算 max|div(V)| 用于诊断
double compute_div_max(const P3DCtx& ctx,
                       const vector<complex<double>>& V1_c,
                       const vector<complex<double>>& V2_c,
                       const vector<complex<double>>& V3_c) {
    int ny = ctx.ny, nz = ctx.nz;
    double local_max = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:local_max)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);
                complex<double> div = {
                    -(kx * V1_c[idx].imag() + ky * V2_c[idx].imag() + kz * V3_c[idx].imag()),
                      kx * V1_c[idx].real() + ky * V2_c[idx].real() + kz * V3_c[idx].real()
                };
                local_max = max(local_max, abs(div));
            }
        }
    }
    double global_max = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // 谱系数为非归一化 DFT（= N × 归一化系数），除以 N 后才是物理量，与其他版本可比
    return global_max * ctx.norm_inv;
}

// 每个 RHS 的第一步（谱空间，一个循环）：rot = i·k × V，同时 rhs = -k² V
void compute_rot_and_viscous(const P3DCtx& ctx, Fields& F) {
    int ny = ctx.ny, nz = ctx.nz;
    C2 *V[3]   = { c2(F.V[0]),   c2(F.V[1]),   c2(F.V[2]) };
    C2 *rot[3] = { c2(F.rot[0]), c2(F.rot[1]), c2(F.rot[2]) };
    C2 *rhs[3] = { c2(F.rhs[0]), c2(F.rhs[1]), c2(F.rhs[2]) };
    #pragma omp parallel for collapse(3)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);
                double k2 = kx * kx + ky * ky + kz * kz;
                // rot_1 = i(ky*V3 - kz*V2)：复数乘法 i*(a+ib) = -b+ia
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
// btran 前不再复制输入：之后不再使用 V 与 rot 的谱空间内容。
void rk4_stage(const P3DCtx& ctx, Fields& F,
               double dt, double t, double dx, double dy, double dz, int stage) {
    unsigned char op_f[] = "fft";
    unsigned char op_b[] = "tff";
    int ny = ctx.ny, nz = ctx.nz;

    // 1. 旋度 + 粘性项（谱空间）
    compute_rot_and_viscous(ctx, F);
    g_timer.mark(PH_SPEC);

    // 2. 逆变换 V、rot → 实空间（结果为 N * 物理值）
    for (int c = 0; c < 3; ++c) Cp3dfft_btran_c2r(reinterpret_cast<double*>(F.V[c].data()),   F.Vr[c].data(),   op_b);
    for (int c = 0; c < 3; ++c) Cp3dfft_btran_c2r(reinterpret_cast<double*>(F.rot[c].data()), F.rotr[c].data(), op_b);
    g_timer.mark(PH_FFT);

    // 3. 除以 N 得到物理值 + 实空间叉乘 v × rot(v)，覆盖 rotr
    double inv_n = ctx.norm_inv;
    size_t local_r_size = static_cast<size_t>(ctx.isize[0]) * ctx.isize[1] * ctx.isize[2];
    #pragma omp parallel for
    for (size_t idx = 0; idx < local_r_size; ++idx) {
        for (int c = 0; c < 3; ++c) { F.Vr[c][idx] *= inv_n; F.rotr[c][idx] *= inv_n; }
    }
    // 循环顺序 k(z)→j(y)→i(x)：p3dfft Fortran 列主序 x 最快（stride=1）
    #pragma omp parallel for collapse(3)
    for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
        for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
            for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                size_t idx = idx_r(ctx, i, j, k);
                double v1 = F.Vr[0][idx], v2 = F.Vr[1][idx], v3 = F.Vr[2][idx];
                double w1 = F.rotr[0][idx], w2 = F.rotr[1][idx], w3 = F.rotr[2][idx];
                F.rotr[0][idx] = v2 * w3 - v3 * w2;  // (v×rot)_x
                F.rotr[1][idx] = v3 * w1 - v1 * w3;  // (v×rot)_y
                F.rotr[2][idx] = v1 * w2 - v2 * w1;  // (v×rot)_z
            }
        }
    }
    g_timer.mark(PH_PHYS);

    // 4. 正变换：叉乘 → rot（存放 nl，不归一化）
    for (int c = 0; c < 3; ++c) Cp3dfft_ftran_r2c(F.rotr[c].data(), reinterpret_cast<double*>(F.rot[c].data()), op_f);
    g_timer.mark(PH_FFT);

    // 5. 外力 [仅 verify]：填入空闲的 Vr，正变换到 V（不归一化）
    if (g_with_force) {
        #pragma omp parallel for collapse(3)
        for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
            for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
                for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                    size_t idx = idx_r(ctx, i, j, k);
                    double x = phys_x(ctx, i, dx);
                    double y = phys_y(ctx, j, dy);
                    double z = phys_z(ctx, k, dz);
                    F.Vr[0][idx] = func_f1(x, y, z, t);
                    F.Vr[1][idx] = func_f2(x, y, z, t);
                    F.Vr[2][idx] = func_f3(x, y, z, t);
                }
            }
        }
        for (int c = 0; c < 3; ++c) Cp3dfft_ftran_r2c(F.Vr[c].data(), reinterpret_cast<double*>(F.V[c].data()), op_f);
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
    #pragma omp parallel for collapse(3)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);
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

void rk4_step(const P3DCtx& ctx, Fields& F,
              double dt, double t, double dx, double dy, double dz) {
    // 保存 V^n
    for (int c = 0; c < 3; ++c) F.orig[c] = F.V[c];
    g_timer.mark(PH_SPEC);

    rk4_stage(ctx, F, dt, t,            dx, dy, dz, 1);
    rk4_stage(ctx, F, dt, t + 0.5 * dt, dx, dy, dz, 2);
    rk4_stage(ctx, F, dt, t + 0.5 * dt, dx, dy, dz, 3);
    rk4_stage(ctx, F, dt, t + dt,       dx, dy, dz, 4);
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

    const double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
    double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    // ------------------------------------------------------------------
    // 初始化 p3dfft（2D pencil 分解）
    // ------------------------------------------------------------------
    P3DCtx ctx;
    ctx.nx = nx; ctx.ny = ny; ctx.nz = nz;

    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);

    // ------------------------------------------------------------------
    // FFTW 多线程初始化（必须在 Cp3dfft_setup 之前调用，使 p3dfft 内部的
    // FFTW 计划创建时使用多线程，与 FFTW / AccFFT 版本保持一致）
    // ------------------------------------------------------------------
    omp_set_num_threads(nthreads);
    int max_threads = nthreads;
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - p3dfft Version (2D Pencil)\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz
             << ", dt=" << scientific << dt << ", steps=" << nt_run << "\n";
        cout << "Domain: [0, 2π]^3\n";
        cout << "MPI processes: " << nprocs
             << " (2D grid " << dims[0] << "x" << dims[1] << ")\n";
        cout << "OpenMP threads/process: " << max_threads << "\n";
        cout << "Run mode: " << nsmode::mode_name(mode)
             << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << "\n";
        cout << "============================================================\n";
    }

    int memsize[3];
    Cp3dfft_setup(dims, nx, ny, nz,
                  MPI_Comm_c2f(MPI_COMM_WORLD),
                  nx, ny, nz,
                  1,       // OW=1: btran_c2r 允许覆盖输入（我们始终复制，此处设为1）
                  memsize);

    ctx.memsize[0] = memsize[0];
    ctx.memsize[1] = memsize[1];
    ctx.memsize[2] = memsize[2];

    // 获取实空间（conf=1）和频谱空间（conf=2）本地范围（1-based 全局索引）
    int conf = 1;
    Cp3dfft_get_dims(ctx.istart, ctx.iend, ctx.isize, conf);
    conf = 2;
    Cp3dfft_get_dims(ctx.fstart, ctx.fend, ctx.fsize, conf);

    // 确定索引偏移（0-based 或 1-based）
    ctx.idx_offset = (ctx.istart[0] == 0) ? 0 : 1;

    // 归一化因子（逆变换后除以 N）
    ctx.norm_inv = 1.0 / (static_cast<double>(nx) * ny * nz);

    if (rank == 0) {
        cout << "idx_offset = " << ctx.idx_offset << "\n";
        cout << "Real space:     ["
             << ctx.istart[0] << "," << ctx.iend[0] << "] x ["
             << ctx.istart[1] << "," << ctx.iend[1] << "] x ["
             << ctx.istart[2] << "," << ctx.iend[2] << "]\n";
        cout << "Spectral space: ["
             << ctx.fstart[0] << "," << ctx.fend[0] << "] x ["
             << ctx.fstart[1] << "," << ctx.fend[1] << "] x ["
             << ctx.fstart[2] << "," << ctx.fend[2] << "]\n";
    }

    // ------------------------------------------------------------------
    // 分配内存
    // ------------------------------------------------------------------
    // 实空间：使用 memsize[2] 为最后一维（含 FFT padding）
    size_t local_size_r = static_cast<size_t>(ctx.isize[0]) * ctx.isize[1] * ctx.memsize[2];
    size_t local_size_c = static_cast<size_t>(ctx.fsize[0]) * ctx.fsize[1] * ctx.fsize[2];

    // 全尺寸数组：15 个复数（V、rot、rhs、V_orig、acc 各 3 个）+ 6 个实数（Vr、rotr 各 3 个），
    // bench 与 verify 相同（verify 的外力复用 Vr 与 V）
    Fields F;
    for (int c = 0; c < 3; ++c) {
        F.V[c].resize(local_size_c);   F.rot[c].resize(local_size_c);  F.rhs[c].resize(local_size_c);
        F.orig[c].resize(local_size_c); F.acc[c].resize(local_size_c);
        F.Vr[c].resize(local_size_r);  F.rotr[c].resize(local_size_r);
    }
    vector<double> &V1_r = F.Vr[0], &V2_r = F.Vr[1], &V3_r = F.Vr[2];
    vector<complex<double>> &V1_c = F.V[0], &V2_c = F.V[1], &V3_c = F.V[2];

    {
        const double GB = 1024.0 * 1024.0 * 1024.0;
        double mx[2] = { (double)(local_size_c * sizeof(complex<double>)), (double)(local_size_r * sizeof(double)) }, g[2];
        MPI_Allreduce(mx, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Memory report (per process, max over ranks):\n");
            printf("  Full-size arrays: 15 complex of %.3f GB, 6 real of %.3f GB\n", g[0] / GB, g[1] / GB);
            fflush(stdout);
        }
    }

    // ------------------------------------------------------------------
    // 初始条件（t=0）
    // ------------------------------------------------------------------
    // 循环顺序 k(z)→j(y)→i(x)：x 最内层，stride=1，缓存友好。
    for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
        for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
            for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                size_t ridx = idx_r(ctx, i, j, k);
                double x = phys_x(ctx, i, dx);
                double y = phys_y(ctx, j, dy);
                double z = phys_z(ctx, k, dz);
                V1_r[ridx] = func_V1(x, y, z, 0.0);
                V2_r[ridx] = func_V2(x, y, z, 0.0);
                V3_r[ridx] = func_V3(x, y, z, 0.0);
            }
        }
    }

    // 正变换（不归一化：V_c = DFT(V_r) = N * FFT_norm(V_phys)）
    unsigned char op_f[] = "fft";
    Cp3dfft_ftran_r2c(V1_r.data(), reinterpret_cast<double*>(V1_c.data()), op_f);
    Cp3dfft_ftran_r2c(V2_r.data(), reinterpret_cast<double*>(V2_c.data()), op_f);
    Cp3dfft_ftran_r2c(V3_r.data(), reinterpret_cast<double*>(V3_c.data()), op_f);

    // 投影初始条件到无散空间
    if (rank == 0) cout << "Projecting initial condition to divergence-free space...\n";
    make_div_free(ctx, V1_c, V2_c, V3_c);

    // ------------------------------------------------------------------
    // 误差计算辅助函数
    // ------------------------------------------------------------------
    // V_c → V_r（物理值；btran_c2r 会破坏输入，先复制到 rhs：计时结束后 rhs 已无用）
    auto to_physical = [&]() {
        vector<complex<double>> &c1 = F.rhs[0], &c2 = F.rhs[1], &c3 = F.rhs[2];
        c1 = V1_c; c2 = V2_c; c3 = V3_c;
        unsigned char op_b[] = "tff";
        Cp3dfft_btran_c2r(reinterpret_cast<double*>(c1.data()), V1_r.data(), op_b);
        Cp3dfft_btran_c2r(reinterpret_cast<double*>(c2.data()), V2_r.data(), op_b);
        Cp3dfft_btran_c2r(reinterpret_cast<double*>(c3.data()), V3_r.data(), op_b);

        // 除以 N 得到物理值
        double inv_n = ctx.norm_inv;
        size_t local_r = static_cast<size_t>(ctx.isize[0]) * ctx.isize[1] * ctx.isize[2];
        #pragma omp parallel for
        for (size_t idx = 0; idx < local_r; ++idx) {
            V1_r[idx] *= inv_n;
            V2_r[idx] *= inv_n;
            V3_r[idx] *= inv_n;
        }
    };

    auto compute_error = [&](double time) -> pair<double, double> {
        to_physical();
        // 循环顺序 k(z)→j(y)→i(x)：x 最内层，stride=1，缓存友好。
        double local_err = 0.0, local_max = 0.0;
        for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
            for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
                for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                    size_t idx = idx_r(ctx, i, j, k);
                    double x = phys_x(ctx, i, dx);
                    double y = phys_y(ctx, j, dy);
                    double z = phys_z(ctx, k, dz);
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
        #pragma omp parallel for collapse(3) reduction(+:local_e)
        for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
            for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
                for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                    size_t idx = idx_r(ctx, i, j, k);
                    local_e += V1_r[idx]*V1_r[idx] + V2_r[idx]*V2_r[idx] + V3_r[idx]*V3_r[idx];
                }
            }
        }
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
        rk4_step(ctx, F, dt, t_cur, dx, dy, dz);
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
        double max_div = compute_div_max(ctx, V1_c, V2_c, V3_c);
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

    MPI_Finalize();
    return 0;
}
