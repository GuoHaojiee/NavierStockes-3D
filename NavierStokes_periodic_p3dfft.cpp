/**
 * Navier-Stokes 求解器 - p3dfft 版本（2D pencil 分解）
 *
 * 基于 NavierStokes_periodic_fftw.cpp 的核心算法，使用 p3dfft C 接口代替 FFTW MPI。
 *
 * 与 FFTW 版本的关键区别：
 *   - p3dfft 支持 2D pencil 分解（非 FFTW 的 1D slab），可更好地利用多节点并行
 *   - R2C 压缩第一维（x 方向）：kx = 0..nx/2（非负），ky/kz 需折叠处理
 *   - 数据布局：Fortran 列主序（x 变化最快），使用 1-based 全局索引
 *   - p3dfft btran_c2r 会销毁输入数组，使用前必须复制
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

using namespace std;

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
//   所有频谱量（rot_c, visc_c, nl_c, f_c）均为非归一化表示
//   逆变换后：V_phys = btran(V_c_copy) / N
// ==============================================================================

// 计算旋度 rot(V) = i·k × V（频谱空间）
// p3dfft R2C 约定：kx 为非负（x 方向被压缩），ky/kz 需折叠
void compute_rot(const P3DCtx& ctx,
                 const vector<complex<double>>& V1_c,
                 const vector<complex<double>>& V2_c,
                 const vector<complex<double>>& V3_c,
                 vector<complex<double>>& rot1_c,
                 vector<complex<double>>& rot2_c,
                 vector<complex<double>>& rot3_c) {
    int nx = ctx.nx, ny = ctx.ny, nz = ctx.nz;
    #pragma omp parallel for collapse(3)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);

                // rot_1 = i(ky*V3 - kz*V2)：复数乘法 i*(a+ib) = -b+ia
                rot1_c[idx] = { -(ky * V3_c[idx].imag() - kz * V2_c[idx].imag()),
                                   ky * V3_c[idx].real() - kz * V2_c[idx].real() };
                rot2_c[idx] = { -(kz * V1_c[idx].imag() - kx * V3_c[idx].imag()),
                                   kz * V1_c[idx].real() - kx * V3_c[idx].real() };
                rot3_c[idx] = { -(kx * V2_c[idx].imag() - ky * V1_c[idx].imag()),
                                   kx * V2_c[idx].real() - ky * V1_c[idx].real() };
            }
        }
    }
}

// 计算粘性项：visc = -k² * V（频谱空间）
void compute_viscous(const P3DCtx& ctx,
                     const vector<complex<double>>& V_c,
                     vector<complex<double>>& visc_c) {
    int ny = ctx.ny, nz = ctx.nz;
    #pragma omp parallel for collapse(3)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);
                double k2 = kx * kx + ky * ky + kz * kz;
                visc_c[idx] = -k2 * V_c[idx];
            }
        }
    }
}

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间，in-place）
// div = i·k·V，phi = div/(-k²)，V -= i·k·phi
void make_div_free(const P3DCtx& ctx,
                   vector<complex<double>>& V1_c,
                   vector<complex<double>>& V2_c,
                   vector<complex<double>>& V3_c,
                   vector<complex<double>>& div_c,
                   vector<complex<double>>& phi_c) {
    int ny = ctx.ny, nz = ctx.nz;
    #pragma omp parallel for collapse(3)
    for (int i = ctx.fstart[0]; i <= ctx.fend[0]; ++i) {
        for (int j = ctx.fstart[1]; j <= ctx.fend[1]; ++j) {
            for (int k = ctx.fstart[2]; k <= ctx.fend[2]; ++k) {
                size_t idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_fold(j, ctx.idx_offset, ny);
                double kz = wave_fold(k, ctx.idx_offset, nz);

                // div = i(kx*V1 + ky*V2 + kz*V3)
                div_c[idx] = {
                    -(kx * V1_c[idx].imag() + ky * V2_c[idx].imag() + kz * V3_c[idx].imag()),
                      kx * V1_c[idx].real() + ky * V2_c[idx].real() + kz * V3_c[idx].real()
                };

                double k2 = kx * kx + ky * ky + kz * kz;
                phi_c[idx] = (k2 > 1e-14) ? (div_c[idx] / (-k2))
                                           : complex<double>(0.0, 0.0);

                // V -= i*k*phi
                V1_c[idx] -= complex<double>(-kx * phi_c[idx].imag(), kx * phi_c[idx].real());
                V2_c[idx] -= complex<double>(-ky * phi_c[idx].imag(), ky * phi_c[idx].real());
                V3_c[idx] -= complex<double>(-kz * phi_c[idx].imag(), kz * phi_c[idx].real());
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
    return global_max;
}

// ==============================================================================
// 非线性项：v × rot(v)（伪谱方法）
// ==============================================================================

// 计算非线性对流项 v × rot(v)，输入输出均在频谱空间（非归一化 DFT）
//
// 注意：
//   - p3dfft btran_c2r 会销毁输入，因此对 V_c 和 rot_c 进行复制
//   - 逆变换后除以 N 得到物理值（DFT 归一化约定）
//   - 正变换（ftran_r2c）不除以 N，保持非归一化约定
void compute_nonlinear(const P3DCtx& ctx,
                       const vector<complex<double>>& V1_c,
                       const vector<complex<double>>& V2_c,
                       const vector<complex<double>>& V3_c,
                       vector<complex<double>>& nl1_c,
                       vector<complex<double>>& nl2_c,
                       vector<complex<double>>& nl3_c,
                       vector<double>& V1_r,
                       vector<double>& V2_r,
                       vector<double>& V3_r,
                       vector<complex<double>>& rot1_c,
                       vector<complex<double>>& rot2_c,
                       vector<complex<double>>& rot3_c,
                       vector<double>& rot1_r,
                       vector<double>& rot2_r,
                       vector<double>& rot3_r) {

    unsigned char op_f[] = "fft";
    unsigned char op_b[] = "tff";

    // 1. 频谱空间计算旋度（rot_c = ik × V_c，非归一化）
    compute_rot(ctx, V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c);

    // 2. 逆变换到实空间（btran_c2r 会破坏输入，必须先复制）
    //    结果为 N * V_phys，除以 N 得到物理值
    vector<complex<double>> V1_tmp(V1_c), V2_tmp(V2_c), V3_tmp(V3_c);
    vector<complex<double>> rot1_tmp(rot1_c), rot2_tmp(rot2_c), rot3_tmp(rot3_c);

    Cp3dfft_btran_c2r(reinterpret_cast<double*>(V1_tmp.data()),   V1_r.data(), op_b);
    Cp3dfft_btran_c2r(reinterpret_cast<double*>(V2_tmp.data()),   V2_r.data(), op_b);
    Cp3dfft_btran_c2r(reinterpret_cast<double*>(V3_tmp.data()),   V3_r.data(), op_b);
    Cp3dfft_btran_c2r(reinterpret_cast<double*>(rot1_tmp.data()), rot1_r.data(), op_b);
    Cp3dfft_btran_c2r(reinterpret_cast<double*>(rot2_tmp.data()), rot2_r.data(), op_b);
    Cp3dfft_btran_c2r(reinterpret_cast<double*>(rot3_tmp.data()), rot3_r.data(), op_b);

    // btran_c2r 输出 = N * V_phys，除以 N 得到物理值
    double inv_n = ctx.norm_inv;
    size_t local_r_size = static_cast<size_t>(ctx.isize[0]) * ctx.isize[1] * ctx.isize[2];
    #pragma omp parallel for
    for (size_t idx = 0; idx < local_r_size; ++idx) {
        V1_r[idx] *= inv_n;
        V2_r[idx] *= inv_n;
        V3_r[idx] *= inv_n;
        rot1_r[idx] *= inv_n;
        rot2_r[idx] *= inv_n;
        rot3_r[idx] *= inv_n;
    }

    // 3. 实空间计算 v × rot(v)，覆盖 rot_r（复用为工作数组）
    // 循环顺序 k(z)→j(y)→i(x)：p3dfft Fortran 列主序 x 最快（stride=1），
    // x 放最内层可保证连续内存访问，避免大步长缓存缺失。
    #pragma omp parallel for collapse(3)
    for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
        for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
            for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                size_t idx = idx_r(ctx, i, j, k);
                double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
                double w1 = rot1_r[idx], w2 = rot2_r[idx], w3 = rot3_r[idx];
                rot1_r[idx] = v2 * w3 - v3 * w2;  // (v×rot)_x
                rot2_r[idx] = v3 * w1 - v1 * w3;  // (v×rot)_y
                rot3_r[idx] = v1 * w2 - v2 * w1;  // (v×rot)_z
            }
        }
    }

    // 4. 正变换回频谱空间（不归一化，nl_c = DFT(cross) = N * FFT_norm(cross)）
    Cp3dfft_ftran_r2c(rot1_r.data(), reinterpret_cast<double*>(nl1_c.data()), op_f);
    Cp3dfft_ftran_r2c(rot2_r.data(), reinterpret_cast<double*>(nl2_c.data()), op_f);
    Cp3dfft_ftran_r2c(rot3_r.data(), reinterpret_cast<double*>(nl3_c.data()), op_f);
}

// ==============================================================================
// 完整右端项：RHS = v×rot(v) + P∆v + f，投影到无散空间
// ==============================================================================

void compute_rhs(const P3DCtx& ctx,
                 const vector<complex<double>>& V1_c,
                 const vector<complex<double>>& V2_c,
                 const vector<complex<double>>& V3_c,
                 vector<complex<double>>& rhs1_c,
                 vector<complex<double>>& rhs2_c,
                 vector<complex<double>>& rhs3_c,
                 vector<double>& V1_r,
                 vector<double>& V2_r,
                 vector<double>& V3_r,
                 vector<double>& work_r1,
                 vector<double>& work_r2,
                 vector<double>& work_r3,
                 vector<complex<double>>& rot1_c,
                 vector<complex<double>>& rot2_c,
                 vector<complex<double>>& rot3_c,
                 vector<double>& rot1_r,
                 vector<double>& rot2_r,
                 vector<double>& rot3_r,
                 vector<complex<double>>& nl1_c,
                 vector<complex<double>>& nl2_c,
                 vector<complex<double>>& nl3_c,
                 vector<complex<double>>& visc1_c,
                 vector<complex<double>>& visc2_c,
                 vector<complex<double>>& visc3_c,
                 vector<complex<double>>& f1_c,
                 vector<complex<double>>& f2_c,
                 vector<complex<double>>& f3_c,
                 vector<complex<double>>& div_c,
                 vector<complex<double>>& phi_c,
                 double t, double dx, double dy, double dz) {

    unsigned char op_f[] = "fft";

    // 1. 非线性项 v × rot(v)（伪谱方法）
    compute_nonlinear(ctx, V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                      V1_r, V2_r, V3_r,
                      rot1_c, rot2_c, rot3_c,
                      rot1_r, rot2_r, rot3_r);

    // 2. 粘性项 P∆V = -k² V（频谱空间，P=1）
    compute_viscous(ctx, V1_c, visc1_c);
    compute_viscous(ctx, V2_c, visc2_c);
    compute_viscous(ctx, V3_c, visc3_c);

    // 3. 外力项 f（实空间 → 频谱空间，不归一化）
    // 循环顺序 k(z)→j(y)→i(x)：x 最内层，stride=1，缓存友好。
    #pragma omp parallel for collapse(3)
    for (int k = ctx.istart[2]; k <= ctx.iend[2]; ++k) {
        for (int j = ctx.istart[1]; j <= ctx.iend[1]; ++j) {
            for (int i = ctx.istart[0]; i <= ctx.iend[0]; ++i) {
                size_t idx = idx_r(ctx, i, j, k);
                double x = phys_x(ctx, i, dx);
                double y = phys_y(ctx, j, dy);
                double z = phys_z(ctx, k, dz);
                work_r1[idx] = func_f1(x, y, z, t);
                work_r2[idx] = func_f2(x, y, z, t);
                work_r3[idx] = func_f3(x, y, z, t);
            }
        }
    }
    // f_c = DFT(f) = N * FFT_norm(f)，与 V_c, nl_c, visc_c 约定一致（不归一化）
    Cp3dfft_ftran_r2c(work_r1.data(), reinterpret_cast<double*>(f1_c.data()), op_f);
    Cp3dfft_ftran_r2c(work_r2.data(), reinterpret_cast<double*>(f2_c.data()), op_f);
    Cp3dfft_ftran_r2c(work_r3.data(), reinterpret_cast<double*>(f3_c.data()), op_f);

    // 4. 组合各项：RHS = v×rot(v) + P∆V + f（均为非归一化 DFT）
    size_t nc = static_cast<size_t>(ctx.fsize[0]) * ctx.fsize[1] * ctx.fsize[2];
    #pragma omp parallel for
    for (size_t idx = 0; idx < nc; ++idx) {
        rhs1_c[idx] = nl1_c[idx] + visc1_c[idx] + f1_c[idx];
        rhs2_c[idx] = nl2_c[idx] + visc2_c[idx] + f2_c[idx];
        rhs3_c[idx] = nl3_c[idx] + visc3_c[idx] + f3_c[idx];
    }

    // 5. 投影：使 RHS 无散（P 算子）
    make_div_free(ctx, rhs1_c, rhs2_c, rhs3_c, div_c, phi_c);
}

// ==============================================================================
// RK4 时间积分（频谱空间）
// ==============================================================================

void rk4_step(const P3DCtx& ctx,
              vector<complex<double>>& V1_c,
              vector<complex<double>>& V2_c,
              vector<complex<double>>& V3_c,
              vector<double>& V1_r, vector<double>& V2_r, vector<double>& V3_r,
              vector<double>& work_r1, vector<double>& work_r2, vector<double>& work_r3,
              vector<complex<double>>& k1_v1, vector<complex<double>>& k1_v2, vector<complex<double>>& k1_v3,
              vector<complex<double>>& k2_v1, vector<complex<double>>& k2_v2, vector<complex<double>>& k2_v3,
              vector<complex<double>>& k3_v1, vector<complex<double>>& k3_v2, vector<complex<double>>& k3_v3,
              vector<complex<double>>& k4_v1, vector<complex<double>>& k4_v2, vector<complex<double>>& k4_v3,
              vector<complex<double>>& tmp_v1, vector<complex<double>>& tmp_v2, vector<complex<double>>& tmp_v3,
              vector<complex<double>>& rot1_c, vector<complex<double>>& rot2_c, vector<complex<double>>& rot3_c,
              vector<double>& rot1_r, vector<double>& rot2_r, vector<double>& rot3_r,
              vector<complex<double>>& nl1_c, vector<complex<double>>& nl2_c, vector<complex<double>>& nl3_c,
              vector<complex<double>>& visc1_c, vector<complex<double>>& visc2_c, vector<complex<double>>& visc3_c,
              vector<complex<double>>& f1_c, vector<complex<double>>& f2_c, vector<complex<double>>& f3_c,
              vector<complex<double>>& div_c, vector<complex<double>>& phi_c,
              double dt, double t, double dx, double dy, double dz) {

    size_t nc = static_cast<size_t>(ctx.fsize[0]) * ctx.fsize[1] * ctx.fsize[2];

    // k1 = RHS(V^n, t)
    compute_rhs(ctx, V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, t, dx, dy, dz);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        tmp_v1[i] = V1_c[i] + 0.5 * dt * k1_v1[i];
        tmp_v2[i] = V2_c[i] + 0.5 * dt * k1_v2[i];
        tmp_v3[i] = V3_c[i] + 0.5 * dt * k1_v3[i];
    }

    // k2 = RHS(V^n + dt/2 * k1, t + dt/2)
    compute_rhs(ctx, tmp_v1, tmp_v2, tmp_v3, k2_v1, k2_v2, k2_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, t + 0.5 * dt, dx, dy, dz);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        tmp_v1[i] = V1_c[i] + 0.5 * dt * k2_v1[i];
        tmp_v2[i] = V2_c[i] + 0.5 * dt * k2_v2[i];
        tmp_v3[i] = V3_c[i] + 0.5 * dt * k2_v3[i];
    }

    // k3 = RHS(V^n + dt/2 * k2, t + dt/2)
    compute_rhs(ctx, tmp_v1, tmp_v2, tmp_v3, k3_v1, k3_v2, k3_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, t + 0.5 * dt, dx, dy, dz);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        tmp_v1[i] = V1_c[i] + dt * k3_v1[i];
        tmp_v2[i] = V2_c[i] + dt * k3_v2[i];
        tmp_v3[i] = V3_c[i] + dt * k3_v3[i];
    }

    // k4 = RHS(V^n + dt * k3, t + dt)
    compute_rhs(ctx, tmp_v1, tmp_v2, tmp_v3, k4_v1, k4_v2, k4_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, t + dt, dx, dy, dz);

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

    if (argc < 6) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " NX NY NZ dt NSTEPS [OMP_threads]\n";
            cerr << "  Example: mpirun -np 4 " << argv[0] << " 64 64 64 0.00001 100\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    double dt = atof(argv[4]);
    int nt_run = atoi(argv[5]);
    int nthreads = (argc > 6) ? atoi(argv[6]) : omp_get_max_threads();

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
    ctx.norm_inv = 1.0 / static_cast<double>(nx * ny * nz);

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

    vector<double>               V1_r(local_size_r), V2_r(local_size_r), V3_r(local_size_r);
    vector<complex<double>>      V1_c(local_size_c), V2_c(local_size_c), V3_c(local_size_c);

    vector<double>               work_r1(local_size_r), work_r2(local_size_r), work_r3(local_size_r);
    vector<double>               rot1_r(local_size_r), rot2_r(local_size_r), rot3_r(local_size_r);

    vector<complex<double>>      rot1_c(local_size_c), rot2_c(local_size_c), rot3_c(local_size_c);
    vector<complex<double>>      nl1_c(local_size_c), nl2_c(local_size_c), nl3_c(local_size_c);
    vector<complex<double>>      visc1_c(local_size_c), visc2_c(local_size_c), visc3_c(local_size_c);
    vector<complex<double>>      f1_c(local_size_c), f2_c(local_size_c), f3_c(local_size_c);
    vector<complex<double>>      div_c(local_size_c), phi_c(local_size_c);

    vector<complex<double>>      k1_v1(local_size_c), k1_v2(local_size_c), k1_v3(local_size_c);
    vector<complex<double>>      k2_v1(local_size_c), k2_v2(local_size_c), k2_v3(local_size_c);
    vector<complex<double>>      k3_v1(local_size_c), k3_v2(local_size_c), k3_v3(local_size_c);
    vector<complex<double>>      k4_v1(local_size_c), k4_v2(local_size_c), k4_v3(local_size_c);
    vector<complex<double>>      tmp_v1(local_size_c), tmp_v2(local_size_c), tmp_v3(local_size_c);

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
    make_div_free(ctx, V1_c, V2_c, V3_c, div_c, phi_c);

    // ------------------------------------------------------------------
    // 误差计算辅助函数
    // ------------------------------------------------------------------
    auto compute_error = [&](double time) -> pair<double, double> {
        // btran_c2r 会破坏输入，必须复制
        vector<complex<double>> c1(V1_c), c2(V2_c), c3(V3_c);
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

    // ------------------------------------------------------------------
    // 时间推进（RK4）
    // ------------------------------------------------------------------
    double t_wall_total = 0.0;

    for (ptrdiff_t it = 0; it < nt_run; ++it) {
        double t_cur = it * dt;
        double t0 = MPI_Wtime();
        rk4_step(ctx, V1_c, V2_c, V3_c,
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
                 dt, t_cur, dx, dy, dz);
        double step_time = MPI_Wtime() - t0;
        double global_step;
        MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_wall_total += global_step;
    }

    {
        double t_final = nt_run * dt;
        auto [errL2, errLinf] = compute_error(t_final);
        if (rank == 0)
            cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << errL2 << "\n";
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

    MPI_Barrier(MPI_COMM_WORLD);
    Cp3dfft_clean();
    MPI_Finalize();
    return 0;
}
