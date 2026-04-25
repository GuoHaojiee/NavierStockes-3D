/**
 * Navier-Stokes 求解器 - AccFFT 版本（2D pencil 分解）
 *
 * 基于 NavierStokes_periodic_fftw.cpp 的核心算法，使用 AccFFT 库代替 FFTW MPI。
 *
 * 与 FFTW 版本的关键区别（库差异，算法相同）：
 *   - AccFFT 2D pencil 分解（非 FFTW 的 1D slab）
 *   - R2C 压缩第三维（z 方向）：kz = 0..nz/2（非负）
 *   - kx = (gi<=nx/2)?gi:gi-nx，ky 类似（折叠，Nyquist 置零）
 *   - Complex = double[2]，访问方式 A[idx][0] / A[idx][1]
 *   - 实空间索引步长用全局 nz：ptr = i*isize[1]*nz + j*nz + k
 *   - 频谱空间索引：ptr = i*osize[1]*osize[2] + j*osize[2] + k
 *   - 逆变换不自动归一化，需手动除以 N = nx*ny*nz
 *   - accfft_execute_c2r 调用前必须复制输入（保守处理，与 p3dfft 一致）
 *   - 所有频谱量均为 DFT（非归一化）：V_c = DFT(V_phys)
 *
 * 编译：参见 Makefile_accfft
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <utility>
#include <mpi.h>
#include <omp.h>
#include <accfft.h>

using namespace std;

// ==============================================================================
// AccFFT 上下文结构体
// ==============================================================================

struct AccCtx {
    int isize[3], istart[3];  // 实空间本地大小和全局起始（0-based）
    int osize[3], ostart[3];  // 频谱空间本地大小和全局起始（0-based）
    int nx, ny, nz;
    long local_size_r;        // = isize[0]*isize[1]*nz（实空间元素数）
    long local_size_c;        // = osize[0]*osize[1]*osize[2]（频谱空间复数元素数）
    double norm_inv;          // 1/(nx*ny*nz)，用于逆变换后归一化
};

// 实空间线性索引（0-based 本地，z 步长为全局 nz）
inline long idx_r(const AccCtx& ctx, int i, int j, int k) {
    return (long)i * ctx.isize[1] * ctx.nz + j * ctx.nz + k;
}

// 频谱空间线性索引（0-based 本地）
inline long idx_c(const AccCtx& ctx, int i, int j, int k) {
    return (long)i * ctx.osize[1] * ctx.osize[2] + j * ctx.osize[2] + k;
}

// 物理坐标
inline double phys_x(const AccCtx& ctx, int i, double dx) { return (i + ctx.istart[0]) * dx; }
inline double phys_y(const AccCtx& ctx, int j, double dy) { return (j + ctx.istart[1]) * dy; }
inline double phys_z(int k, double dz)                    { return k * dz; }

// 波数（AccFFT R2C 沿 z 方向；kx/ky 折叠，Nyquist 置零；kz = 0..nz/2 非负）
inline double wave_kx(const AccCtx& ctx, int i) {
    int gi = i + ctx.ostart[0];
    if (gi == ctx.nx / 2) return 0.0;
    return (gi <= ctx.nx / 2) ? (double)gi : (double)(gi - ctx.nx);
}
inline double wave_ky(const AccCtx& ctx, int j) {
    int gj = j + ctx.ostart[1];
    if (gj == ctx.ny / 2) return 0.0;
    return (gj <= ctx.ny / 2) ? (double)gj : (double)(gj - ctx.ny);
}
inline double wave_kz(const AccCtx& ctx, int k) {
    return (double)(k + ctx.ostart[2]);
}

// ==============================================================================
// 解析解与强迫项（与 FFTW / heFFTe / p3dfft 版本完全一致）
// ==============================================================================

double func_V1(double x, double y, double z, double t) {
    return (t*t+1)*exp(sin(3*x+3*y))*cos(6*z);
}
double func_V2(double x, double y, double z, double t) {
    return (t*t+1)*exp(sin(3*x+3*y))*cos(6*z);
}
double func_V3(double x, double y, double z, double t) {
    return -(t*t+1)*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
}

double func_dV1_dt(double x, double y, double z, double t) {
    return 2*t*exp(sin(3*x+3*y))*cos(6*z);
}
double func_dV2_dt(double x, double y, double z, double t) {
    return 2*t*exp(sin(3*x+3*y))*cos(6*z);
}
double func_dV3_dt(double x, double y, double z, double t) {
    return -2*t*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
}

double func_laplace_V1(double x, double y, double z, double t) {
    double c = cos(3*x+3*y), s = sin(3*x+3*y);
    double d2_xy = (t*t+1)*9*exp(s)*(c*c - s)*cos(6*z);
    double d2_z  = -(t*t+1)*36*exp(s)*cos(6*z);
    return d2_xy + d2_xy + d2_z;
}
double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x, y, z, t);
}
double func_laplace_V3(double x, double y, double z, double t) {
    double c = cos(3*x+3*y), s = sin(3*x+3*y);
    double d2_xy = -(t*t+1)*9*exp(s)*c*((c*c - s) - (2*s + 1))*sin(6*z);
    double d2_z  = (t*t+1)*36*exp(s)*c*sin(6*z);
    return d2_xy + d2_xy + d2_z;
}

double func_rot1(double x, double y, double z, double t) {
    double c = cos(3*x+3*y), s = sin(3*x+3*y);
    double dv3_dy = -(t*t+1)*3*exp(s)*(c*c - s)*sin(6*z);
    double dv2_dz = -(t*t+1)*6*exp(s)*sin(6*z);
    return dv3_dy - dv2_dz;
}
double func_rot2(double x, double y, double z, double t) {
    double c = cos(3*x+3*y), s = sin(3*x+3*y);
    double dv1_dz = -(t*t+1)*6*exp(s)*sin(6*z);
    double dv3_dx = -(t*t+1)*3*exp(s)*(c*c - s)*sin(6*z);
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
    return -(t*t+1)*sin(x)*cos(y)*cos(z);
}
double func_grad_p2(double x, double y, double z, double t) {
    return -(t*t+1)*cos(x)*sin(y)*cos(z);
}
double func_grad_p3(double x, double y, double z, double t) {
    return -(t*t+1)*cos(x)*cos(y)*sin(z);
}

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
// 频谱算子（AccFFT 坐标约定）
//
// 归一化约定：
//   V_c = DFT(V_phys)  [非归一化，= N * FFT_norm(V_phys)]
//   所有频谱量（rot_c, visc_c, nl_c, f_c）均为非归一化表示
//   逆变换后：V_phys = accfft_c2r(V_c_copy) / N
// ==============================================================================

// 计算旋度 rot(V) = i·k × V（频谱空间）
// rot_1 = i(ky*V3 - kz*V2)
// rot_2 = i(kz*V1 - kx*V3)
// rot_3 = i(kx*V2 - ky*V1)
void compute_rot(const AccCtx& ctx,
                 const Complex* V1_c, const Complex* V2_c, const Complex* V3_c,
                 Complex* rot1_c, Complex* rot2_c, Complex* rot3_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);

                // i*(a+ib) = -b+ia
                rot1_c[idx][0] = -(ky * V3_c[idx][1] - kz * V2_c[idx][1]);
                rot1_c[idx][1] =   ky * V3_c[idx][0] - kz * V2_c[idx][0];
                rot2_c[idx][0] = -(kz * V1_c[idx][1] - kx * V3_c[idx][1]);
                rot2_c[idx][1] =   kz * V1_c[idx][0] - kx * V3_c[idx][0];
                rot3_c[idx][0] = -(kx * V2_c[idx][1] - ky * V1_c[idx][1]);
                rot3_c[idx][1] =   kx * V2_c[idx][0] - ky * V1_c[idx][0];
            }
        }
    }
}

// 计算粘性项：visc = -k² * V（频谱空间）
void compute_viscous(const AccCtx& ctx,
                     const Complex* V_c, Complex* visc_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double k2 = kx*kx + ky*ky + kz*kz;
                visc_c[idx][0] = -k2 * V_c[idx][0];
                visc_c[idx][1] = -k2 * V_c[idx][1];
            }
        }
    }
}

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间，in-place）
// div = i·k·V，phi = div/(-k²)，V -= i·k·phi
void make_div_free(const AccCtx& ctx,
                   Complex* V1_c, Complex* V2_c, Complex* V3_c,
                   Complex* div_c, Complex* phi_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);

                // div = i(kx*V1 + ky*V2 + kz*V3)
                div_c[idx][0] = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                div_c[idx][1] =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];

                double k2 = kx*kx + ky*ky + kz*kz;
                if (k2 > 1e-14) {
                    phi_c[idx][0] = div_c[idx][0] / (-k2);
                    phi_c[idx][1] = div_c[idx][1] / (-k2);
                } else {
                    phi_c[idx][0] = phi_c[idx][1] = 0.0;
                }

                // V -= i*k*phi：i*kx*(a+ib) = -kx*b + i*kx*a
                V1_c[idx][0] -= -kx * phi_c[idx][1];
                V1_c[idx][1] -=  kx * phi_c[idx][0];
                V2_c[idx][0] -= -ky * phi_c[idx][1];
                V2_c[idx][1] -=  ky * phi_c[idx][0];
                V3_c[idx][0] -= -kz * phi_c[idx][1];
                V3_c[idx][1] -=  kz * phi_c[idx][0];
            }
        }
    }
}

// 计算 max|div(V)| 用于诊断
double compute_div_max(const AccCtx& ctx,
                       const Complex* V1_c, const Complex* V2_c, const Complex* V3_c) {
    double local_max = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:local_max)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double dr = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                double di =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
                double val = sqrt(dr*dr + di*di);
                if (val > local_max) local_max = val;
            }
        }
    }
    double global_max = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global_max;
}

// ==============================================================================
// 非线性对流项：v × rot(v)（伪谱方法）
//
// 算法（与 FFTW / heFFTe / p3dfft 版本完全相同）：
//   1. 频谱空间计算旋度 rot_c = ik × V_c              （无 FFT）
//   2. 逆变换 V_c → V_r                               （3 次 FFT）
//   3. 逆变换 rot_c → rot_r                            （3 次 FFT）
//   4. 实空间叉乘，覆盖 rot_r                           （无 FFT）
//   5. 正变换 rot_r → nl_c                             （3 次 FFT）
// 合计：9 次 FFT（与其他版本相同）
//
// preserve_V：
//   true  - V_c 在调用后仍需使用（k1 阶段），用 tmp_c 复制保护（3 次 memcpy）
//   false - V_c 调用后即被覆盖（k2/k3/k4 阶段），直接传入 c2r（0 次 memcpy）
// rot_c 始终是临时量（每次由 compute_rot 重新生成），直接传入 c2r（0 次 memcpy）
// ==============================================================================
void compute_nonlinear(const AccCtx& ctx, accfft_plan* plan,
                       Complex* V1_c, Complex* V2_c, Complex* V3_c,
                       Complex* nl1_c, Complex* nl2_c, Complex* nl3_c,
                       double* V1_r, double* V2_r, double* V3_r,
                       Complex* rot1_c, Complex* rot2_c, Complex* rot3_c,
                       double* rot1_r, double* rot2_r, double* rot3_r,
                       Complex* tmp_c,
                       bool preserve_V) {

    long csize = ctx.local_size_c;
    double inv_n = ctx.norm_inv;

    // 1. 频谱空间计算旋度（rot_c = ik × V_c，非归一化）
    compute_rot(ctx, V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c);

    // 2. 逆变换 V_c → V_r
    //    preserve_V=true：先复制到 tmp_c 再 c2r（保护 V_c，供后续 RK4 阶段使用）
    //    preserve_V=false：直接传 V_c（调用方在此后立即覆盖 tmp_v，无需保护）
    if (preserve_V) {
        memcpy(tmp_c, V1_c, csize * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V1_r);
        memcpy(tmp_c, V2_c, csize * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V2_r);
        memcpy(tmp_c, V3_c, csize * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V3_r);
    } else {
        accfft_execute_c2r(plan, V1_c, V1_r);
        accfft_execute_c2r(plan, V2_c, V2_r);
        accfft_execute_c2r(plan, V3_c, V3_r);
    }

    // 3. 逆变换 rot_c → rot_r
    //    rot_c 由 compute_rot 临时生成，c2r 后不再使用，直接传入（0 次 memcpy）
    accfft_execute_c2r(plan, rot1_c, rot1_r);
    accfft_execute_c2r(plan, rot2_c, rot2_r);
    accfft_execute_c2r(plan, rot3_c, rot3_r);

    // 4. 归一化（除以 N 得到物理值）
    #pragma omp parallel for
    for (long idx = 0; idx < ctx.local_size_r; ++idx) {
        V1_r[idx] *= inv_n;    V2_r[idx] *= inv_n;    V3_r[idx] *= inv_n;
        rot1_r[idx] *= inv_n;  rot2_r[idx] *= inv_n;  rot3_r[idx] *= inv_n;
    }

    // 5. 实空间叉乘 v × rot(v)，覆盖 rot_r（用作正变换输入）
    #pragma omp parallel for
    for (long idx = 0; idx < ctx.local_size_r; ++idx) {
        double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
        double w1 = rot1_r[idx], w2 = rot2_r[idx], w3 = rot3_r[idx];
        rot1_r[idx] = v2*w3 - v3*w2;   // (v×rot)_x
        rot2_r[idx] = v3*w1 - v1*w3;   // (v×rot)_y
        rot3_r[idx] = v1*w2 - v2*w1;   // (v×rot)_z
    }

    // 6. 正变换（不归一化：nl_c = DFT(cross)，与 V_c 约定一致）
    accfft_execute_r2c(plan, rot1_r, nl1_c);
    accfft_execute_r2c(plan, rot2_r, nl2_c);
    accfft_execute_r2c(plan, rot3_r, nl3_c);
}

// ==============================================================================
// 完整右端项：RHS = v×rot(v) + P∆v + f，投影到无散空间
// ==============================================================================

void compute_rhs(const AccCtx& ctx, accfft_plan* plan,
                 Complex* V1_c, Complex* V2_c, Complex* V3_c,
                 Complex* rhs1_c, Complex* rhs2_c, Complex* rhs3_c,
                 double* V1_r, double* V2_r, double* V3_r,
                 double* work_r1, double* work_r2, double* work_r3,
                 Complex* rot1_c, Complex* rot2_c, Complex* rot3_c,
                 double* rot1_r, double* rot2_r, double* rot3_r,
                 Complex* nl1_c, Complex* nl2_c, Complex* nl3_c,
                 Complex* visc1_c, Complex* visc2_c, Complex* visc3_c,
                 Complex* f1_c, Complex* f2_c, Complex* f3_c,
                 Complex* div_c, Complex* phi_c,
                 Complex* tmp_c,
                 double t, double dx, double dy, double dz,
                 bool preserve_V) {

    // 1. 粘性项 P∆V = -k² V（纯频谱运算，必须在 compute_nonlinear 之前完成）
    //    当 preserve_V=false 时，compute_nonlinear 会销毁 V_c，故先在此读取
    compute_viscous(ctx, V1_c, visc1_c);
    compute_viscous(ctx, V2_c, visc2_c);
    compute_viscous(ctx, V3_c, visc3_c);

    // 2. 非线性项 v × rot(v)（伪谱方法；preserve_V 控制是否保护 V_c）
    compute_nonlinear(ctx, plan, V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                      V1_r, V2_r, V3_r,
                      rot1_c, rot2_c, rot3_c,
                      rot1_r, rot2_r, rot3_r,
                      tmp_c, preserve_V);

    // 3. 外力项 f（实空间填充 → 正变换，不归一化：f_c = DFT(f)）
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.isize[0]; ++i) {
        for (int j = 0; j < ctx.isize[1]; ++j) {
            for (int k = 0; k < ctx.nz; ++k) {
                long ridx = idx_r(ctx, i, j, k);
                double x = phys_x(ctx, i, dx);
                double y = phys_y(ctx, j, dy);
                double z = phys_z(k, dz);
                work_r1[ridx] = func_f1(x, y, z, t);
                work_r2[ridx] = func_f2(x, y, z, t);
                work_r3[ridx] = func_f3(x, y, z, t);
            }
        }
    }
    accfft_execute_r2c(plan, work_r1, f1_c);
    accfft_execute_r2c(plan, work_r2, f2_c);
    accfft_execute_r2c(plan, work_r3, f3_c);

    // 4. 组合各项：RHS = nl + visc + f（均为非归一化 DFT）
    long nc = ctx.local_size_c;
    #pragma omp parallel for
    for (long idx = 0; idx < nc; ++idx) {
        rhs1_c[idx][0] = nl1_c[idx][0] + visc1_c[idx][0] + f1_c[idx][0];
        rhs1_c[idx][1] = nl1_c[idx][1] + visc1_c[idx][1] + f1_c[idx][1];
        rhs2_c[idx][0] = nl2_c[idx][0] + visc2_c[idx][0] + f2_c[idx][0];
        rhs2_c[idx][1] = nl2_c[idx][1] + visc2_c[idx][1] + f2_c[idx][1];
        rhs3_c[idx][0] = nl3_c[idx][0] + visc3_c[idx][0] + f3_c[idx][0];
        rhs3_c[idx][1] = nl3_c[idx][1] + visc3_c[idx][1] + f3_c[idx][1];
    }

    // 5. 投影：使 RHS 无散（P 算子）
    make_div_free(ctx, rhs1_c, rhs2_c, rhs3_c, div_c, phi_c);
}

// ==============================================================================
// RK4 时间积分（频谱空间）
// ==============================================================================

void rk4_step(const AccCtx& ctx, accfft_plan* plan,
              Complex* V1_c, Complex* V2_c, Complex* V3_c,
              double* V1_r, double* V2_r, double* V3_r,
              double* work_r1, double* work_r2, double* work_r3,
              Complex* k1_v1, Complex* k1_v2, Complex* k1_v3,
              Complex* k2_v1, Complex* k2_v2, Complex* k2_v3,
              Complex* k3_v1, Complex* k3_v2, Complex* k3_v3,
              Complex* k4_v1, Complex* k4_v2, Complex* k4_v3,
              Complex* tmp_v1, Complex* tmp_v2, Complex* tmp_v3,
              Complex* rot1_c, Complex* rot2_c, Complex* rot3_c,
              double* rot1_r, double* rot2_r, double* rot3_r,
              Complex* nl1_c, Complex* nl2_c, Complex* nl3_c,
              Complex* visc1_c, Complex* visc2_c, Complex* visc3_c,
              Complex* f1_c, Complex* f2_c, Complex* f3_c,
              Complex* div_c, Complex* phi_c,
              Complex* tmp_c,
              double dt, double t, double dx, double dy, double dz) {

    long nc = ctx.local_size_c;

    // k1 = RHS(V^n, t)
    // preserve_V=true：V_c 在 k1 后仍需使用（构建 tmp_v，以及 k2/k3/k4 中更新）
    compute_rhs(ctx, plan, V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, tmp_c,
                t, dx, dy, dz, /*preserve_V=*/true);

    // tmp = V^n + dt/2 * k1
    #pragma omp parallel for
    for (long i = 0; i < nc; ++i) {
        tmp_v1[i][0] = V1_c[i][0] + 0.5*dt*k1_v1[i][0];
        tmp_v1[i][1] = V1_c[i][1] + 0.5*dt*k1_v1[i][1];
        tmp_v2[i][0] = V2_c[i][0] + 0.5*dt*k1_v2[i][0];
        tmp_v2[i][1] = V2_c[i][1] + 0.5*dt*k1_v2[i][1];
        tmp_v3[i][0] = V3_c[i][0] + 0.5*dt*k1_v3[i][0];
        tmp_v3[i][1] = V3_c[i][1] + 0.5*dt*k1_v3[i][1];
    }

    // k2 = RHS(V^n + dt/2*k1, t + dt/2)
    // preserve_V=false：tmp_v 在此调用后立即被覆盖（tmp_v = V_c + dt/2*k2），无需保护
    compute_rhs(ctx, plan, tmp_v1, tmp_v2, tmp_v3, k2_v1, k2_v2, k2_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, tmp_c,
                t + 0.5*dt, dx, dy, dz, /*preserve_V=*/false);

    #pragma omp parallel for
    for (long i = 0; i < nc; ++i) {
        tmp_v1[i][0] = V1_c[i][0] + 0.5*dt*k2_v1[i][0];
        tmp_v1[i][1] = V1_c[i][1] + 0.5*dt*k2_v1[i][1];
        tmp_v2[i][0] = V2_c[i][0] + 0.5*dt*k2_v2[i][0];
        tmp_v2[i][1] = V2_c[i][1] + 0.5*dt*k2_v2[i][1];
        tmp_v3[i][0] = V3_c[i][0] + 0.5*dt*k2_v3[i][0];
        tmp_v3[i][1] = V3_c[i][1] + 0.5*dt*k2_v3[i][1];
    }

    // k3 = RHS(V^n + dt/2*k2, t + dt/2)
    // preserve_V=false：tmp_v 在此调用后立即被覆盖（tmp_v = V_c + dt*k3），无需保护
    compute_rhs(ctx, plan, tmp_v1, tmp_v2, tmp_v3, k3_v1, k3_v2, k3_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, tmp_c,
                t + 0.5*dt, dx, dy, dz, /*preserve_V=*/false);

    #pragma omp parallel for
    for (long i = 0; i < nc; ++i) {
        tmp_v1[i][0] = V1_c[i][0] + dt*k3_v1[i][0];
        tmp_v1[i][1] = V1_c[i][1] + dt*k3_v1[i][1];
        tmp_v2[i][0] = V2_c[i][0] + dt*k3_v2[i][0];
        tmp_v2[i][1] = V2_c[i][1] + dt*k3_v2[i][1];
        tmp_v3[i][0] = V3_c[i][0] + dt*k3_v3[i][0];
        tmp_v3[i][1] = V3_c[i][1] + dt*k3_v3[i][1];
    }

    // k4 = RHS(V^n + dt*k3, t + dt)
    // preserve_V=false：tmp_v 在此调用后不再使用（直接进入最终 RK4 更新），无需保护
    compute_rhs(ctx, plan, tmp_v1, tmp_v2, tmp_v3, k4_v1, k4_v2, k4_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                f1_c, f2_c, f3_c, div_c, phi_c, tmp_c,
                t + dt, dx, dy, dz, /*preserve_V=*/false);

    // V^{n+1} = V^n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    #pragma omp parallel for
    for (long i = 0; i < nc; ++i) {
        V1_c[i][0] += (dt/6.0)*(k1_v1[i][0] + 2*k2_v1[i][0] + 2*k3_v1[i][0] + k4_v1[i][0]);
        V1_c[i][1] += (dt/6.0)*(k1_v1[i][1] + 2*k2_v1[i][1] + 2*k3_v1[i][1] + k4_v1[i][1]);
        V2_c[i][0] += (dt/6.0)*(k1_v2[i][0] + 2*k2_v2[i][0] + 2*k3_v2[i][0] + k4_v2[i][0]);
        V2_c[i][1] += (dt/6.0)*(k1_v2[i][1] + 2*k2_v2[i][1] + 2*k3_v2[i][1] + k4_v2[i][1]);
        V3_c[i][0] += (dt/6.0)*(k1_v3[i][0] + 2*k2_v3[i][0] + 2*k3_v3[i][0] + k4_v3[i][0]);
        V3_c[i][1] += (dt/6.0)*(k1_v3[i][1] + 2*k2_v3[i][1] + 2*k3_v3[i][1] + k4_v3[i][1]);
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
    omp_set_num_threads(nthreads);
    int max_threads = nthreads;

    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    // ------------------------------------------------------------------
    // 初始化 AccFFT（2D pencil 分解）
    // ------------------------------------------------------------------
    AccCtx ctx;
    ctx.nx = nx;  ctx.ny = ny;  ctx.nz = nz;
    ctx.norm_inv = 1.0 / static_cast<double>(nx * ny * nz);

    int c_dims[2] = {0, 0};   // {0,0} 让 AccFFT 自动选择最优进程网格
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

    int n[3] = {nx, ny, nz};
    int alloc_max = accfft_local_size_dft_r2c(n, ctx.isize, ctx.istart,
                                               ctx.osize, ctx.ostart, c_comm);

    ctx.local_size_r = (long)ctx.isize[0] * ctx.isize[1] * nz;
    ctx.local_size_c = (long)ctx.osize[0] * ctx.osize[1] * ctx.osize[2];

    accfft_init(max_threads);

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - AccFFT Version (2D Pencil)\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz
             << ", dt=" << scientific << dt << ", steps=" << nt_run << "\n";
        cout << "Domain: [0, 2π]^3\n";
        cout << "MPI processes: " << nprocs << "\n";
        cout << "OpenMP threads/process: " << max_threads << "\n";
        cout << "============================================================\n";
        cout << "Real space local:     ["
             << ctx.istart[0] << "+" << ctx.isize[0] << "] x ["
             << ctx.istart[1] << "+" << ctx.isize[1] << "] x [0+" << ctx.isize[2] << "]\n";
        cout << "Spectral space local: ["
             << ctx.ostart[0] << "+" << ctx.osize[0] << "] x ["
             << ctx.ostart[1] << "+" << ctx.osize[1] << "] x ["
             << ctx.ostart[2] << "+" << ctx.osize[2] << "]\n";
    }

    // ------------------------------------------------------------------
    // 预分配所有内存（热循环中零动态分配）
    // ------------------------------------------------------------------
    long rsz = ctx.local_size_r * sizeof(double);
    long csz = (long)alloc_max;

    // 速度场（实空间 + 频谱空间）
    double*  V1_r = (double*)  accfft_alloc(rsz);
    double*  V2_r = (double*)  accfft_alloc(rsz);
    double*  V3_r = (double*)  accfft_alloc(rsz);
    Complex* V1_c = (Complex*) accfft_alloc(csz);
    Complex* V2_c = (Complex*) accfft_alloc(csz);
    Complex* V3_c = (Complex*) accfft_alloc(csz);

    // 旋度场（实空间 + 频谱空间）
    double*  rot1_r = (double*)  accfft_alloc(rsz);
    double*  rot2_r = (double*)  accfft_alloc(rsz);
    double*  rot3_r = (double*)  accfft_alloc(rsz);
    Complex* rot1_c = (Complex*) accfft_alloc(csz);
    Complex* rot2_c = (Complex*) accfft_alloc(csz);
    Complex* rot3_c = (Complex*) accfft_alloc(csz);

    // 外力实空间工作数组
    double*  work_r1 = (double*) accfft_alloc(rsz);
    double*  work_r2 = (double*) accfft_alloc(rsz);
    double*  work_r3 = (double*) accfft_alloc(rsz);

    // 投影工作数组
    Complex* div_c = (Complex*) accfft_alloc(csz);
    Complex* phi_c = (Complex*) accfft_alloc(csz);

    // 非线性、粘性、外力频谱项
    Complex* nl1_c    = (Complex*) accfft_alloc(csz);
    Complex* nl2_c    = (Complex*) accfft_alloc(csz);
    Complex* nl3_c    = (Complex*) accfft_alloc(csz);
    Complex* visc1_c  = (Complex*) accfft_alloc(csz);
    Complex* visc2_c  = (Complex*) accfft_alloc(csz);
    Complex* visc3_c  = (Complex*) accfft_alloc(csz);
    Complex* f1_c     = (Complex*) accfft_alloc(csz);
    Complex* f2_c     = (Complex*) accfft_alloc(csz);
    Complex* f3_c     = (Complex*) accfft_alloc(csz);

    // RK4 工作数组
    Complex* k1_v1 = (Complex*) accfft_alloc(csz);
    Complex* k1_v2 = (Complex*) accfft_alloc(csz);
    Complex* k1_v3 = (Complex*) accfft_alloc(csz);
    Complex* k2_v1 = (Complex*) accfft_alloc(csz);
    Complex* k2_v2 = (Complex*) accfft_alloc(csz);
    Complex* k2_v3 = (Complex*) accfft_alloc(csz);
    Complex* k3_v1 = (Complex*) accfft_alloc(csz);
    Complex* k3_v2 = (Complex*) accfft_alloc(csz);
    Complex* k3_v3 = (Complex*) accfft_alloc(csz);
    Complex* k4_v1 = (Complex*) accfft_alloc(csz);
    Complex* k4_v2 = (Complex*) accfft_alloc(csz);
    Complex* k4_v3 = (Complex*) accfft_alloc(csz);
    Complex* tmp_v1 = (Complex*) accfft_alloc(csz);
    Complex* tmp_v2 = (Complex*) accfft_alloc(csz);
    Complex* tmp_v3 = (Complex*) accfft_alloc(csz);

    // 单个通用临时复数缓冲（逆变换前复制用，顺序执行可复用）
    Complex* tmp_c = (Complex*) accfft_alloc(csz);

    // ------------------------------------------------------------------
    // 创建 FFT 计划（一个计划可用于所有相同大小的变换）
    // ------------------------------------------------------------------
    if (rank == 0) cout << "Creating AccFFT plan...\n";
    MPI_Barrier(c_comm);
    double t_plan = MPI_Wtime();

    accfft_plan* plan = accfft_plan_dft_3d_r2c(n, V1_r, (double*)V1_c,
                                                c_comm, ACCFFT_MEASURE);
    MPI_Barrier(c_comm);
    if (rank == 0)
        cout << "AccFFT plan time: " << MPI_Wtime() - t_plan << " s\n";

    // ------------------------------------------------------------------
    // 初始条件（t=0）
    // ------------------------------------------------------------------
    if (rank == 0) cout << "Setting initial conditions...\n";

    for (int i = 0; i < ctx.isize[0]; ++i) {
        for (int j = 0; j < ctx.isize[1]; ++j) {
            for (int k = 0; k < nz; ++k) {
                long ridx = idx_r(ctx, i, j, k);
                double x = phys_x(ctx, i, dx);
                double y = phys_y(ctx, j, dy);
                double z = phys_z(k, dz);
                V1_r[ridx] = func_V1(x, y, z, 0.0);
                V2_r[ridx] = func_V2(x, y, z, 0.0);
                V3_r[ridx] = func_V3(x, y, z, 0.0);
            }
        }
    }

    // 正变换（不归一化：V_c = DFT(V_r) = N * FFT_norm(V_phys)）
    accfft_execute_r2c(plan, V1_r, V1_c);
    accfft_execute_r2c(plan, V2_r, V2_c);
    accfft_execute_r2c(plan, V3_r, V3_c);

    // 投影初始条件到无散空间
    if (rank == 0) cout << "Projecting initial condition to divergence-free space...\n";
    make_div_free(ctx, V1_c, V2_c, V3_c, div_c, phi_c);

    // ------------------------------------------------------------------
    // 误差计算辅助函数（使用 tmp_c 作为临时缓冲）
    // ------------------------------------------------------------------
    auto compute_error = [&](double time) -> pair<double, double> {
        // 复制 V_c 到 tmp_c，逆变换（保护原 V_c）
        memcpy(tmp_c, V1_c, ctx.local_size_c * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V1_r);
        memcpy(tmp_c, V2_c, ctx.local_size_c * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V2_r);
        memcpy(tmp_c, V3_c, ctx.local_size_c * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V3_r);

        // 除以 N 得到物理值
        double inv_n = ctx.norm_inv;
        #pragma omp parallel for
        for (long idx = 0; idx < ctx.local_size_r; ++idx) {
            V1_r[idx] *= inv_n;
            V2_r[idx] *= inv_n;
            V3_r[idx] *= inv_n;
        }

        double local_err = 0.0, local_max = 0.0;
        for (int i = 0; i < ctx.isize[0]; ++i) {
            for (int j = 0; j < ctx.isize[1]; ++j) {
                for (int k = 0; k < nz; ++k) {
                    long ridx = idx_r(ctx, i, j, k);
                    double x = phys_x(ctx, i, dx);
                    double y = phys_y(ctx, j, dy);
                    double z = phys_z(k, dz);
                    double d1 = V1_r[ridx] - func_V1(x, y, z, time);
                    double d2 = V2_r[ridx] - func_V2(x, y, z, time);
                    double d3 = V3_r[ridx] - func_V3(x, y, z, time);
                    double e  = d1*d1 + d2*d2 + d3*d3;
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
        rk4_step(ctx, plan,
                 V1_c, V2_c, V3_c,
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
                 div_c, phi_c, tmp_c,
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

    // ------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------
    accfft_free(V1_r);    accfft_free(V2_r);    accfft_free(V3_r);
    accfft_free(V1_c);    accfft_free(V2_c);    accfft_free(V3_c);
    accfft_free(rot1_r);  accfft_free(rot2_r);  accfft_free(rot3_r);
    accfft_free(rot1_c);  accfft_free(rot2_c);  accfft_free(rot3_c);
    accfft_free(work_r1); accfft_free(work_r2); accfft_free(work_r3);
    accfft_free(div_c);   accfft_free(phi_c);
    accfft_free(nl1_c);   accfft_free(nl2_c);   accfft_free(nl3_c);
    accfft_free(visc1_c); accfft_free(visc2_c); accfft_free(visc3_c);
    accfft_free(f1_c);    accfft_free(f2_c);    accfft_free(f3_c);
    accfft_free(k1_v1);   accfft_free(k1_v2);   accfft_free(k1_v3);
    accfft_free(k2_v1);   accfft_free(k2_v2);   accfft_free(k2_v3);
    accfft_free(k3_v1);   accfft_free(k3_v2);   accfft_free(k3_v3);
    accfft_free(k4_v1);   accfft_free(k4_v2);   accfft_free(k4_v3);
    accfft_free(tmp_v1);  accfft_free(tmp_v2);  accfft_free(tmp_v3);
    accfft_free(tmp_c);

    accfft_destroy_plan(plan);
    accfft_cleanup();
    MPI_Comm_free(&c_comm);

    MPI_Finalize();
    return 0;
}
