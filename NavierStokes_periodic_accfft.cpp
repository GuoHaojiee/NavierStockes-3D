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
 *   - accfft_execute_c2r 会破坏输入：时间推进中逆变换之后不再使用其谱空间内容；诊断时先复制
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
#include "common_mode.h"

using namespace std;

// 运行模式与分阶段计时（见 common_mode.h）
//   g_with_force = true  (verify)：每次 RHS 计算解析外力 f（原始行为）
//   g_with_force = false (bench) ：f = 0，跳过外力填充、3 次正向 FFT 与相加
static bool g_with_force = true;
static nsmode::CpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;

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
    return (long)i * ctx.isize[1] * ctx.nz + (long)j * ctx.nz + k;
}

// 频谱空间线性索引（0-based 本地）
inline long idx_c(const AccCtx& ctx, int i, int j, int k) {
    return (long)i * ctx.osize[1] * ctx.osize[2] + (long)j * ctx.osize[2] + k;
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
//   所有频谱量（rot、-k²V、nl、f）均为非归一化表示
//   逆变换后：V_phys = accfft_c2r(V_c) / N（c2r 会破坏输入）
//
// 时间格式：经典 RK4，按累加形式存储（不存 k1..k4，k 只在一次循环内使用）：
//   k    = P[ nl + (-k^2 V) (+ f，仅 verify) ]
//   阶段1: acc  = V_orig + dt/6*k;   V = V_orig + dt/2*k
//   阶段2: acc += dt/3*k;            V = V_orig + dt/2*k
//   阶段3: acc += dt/3*k;            V = V_orig + dt*k
//   阶段4:                           V = acc + dt/6*k
// ==============================================================================

// 所有全尺寸数组（main 中分配一次）
struct Fields {
    Complex *V[3];      // 阶段值（谱空间）；c2r 会破坏其内容，之后由阶段循环重写
    Complex *rot[3];    // 旋度；r2c 之后存放非线性项 nl = DFT(v x rot v)
    Complex *rhs[3];    // -k^2 V（粘性项，由旋度循环同时写入）
    Complex *orig[3];   // V^n
    Complex *acc[3];    // RK4 累加量
    double  *Vr[3];     // 实空间速度；verify 时在叉乘之后用来存放外力 f
    double  *rotr[3];   // 实空间旋度，叉乘结果覆盖于此
};

enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间，in-place；仅用于初始条件）
// div = i·k·V，phi = div/(-k²)，V -= i·k·phi
void make_div_free(const AccCtx& ctx, Complex* V1_c, Complex* V2_c, Complex* V3_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double div_r = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                double div_i =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
                double k2 = kx*kx + ky*ky + kz*kz;
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-14) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                // V -= i*k*phi：i*kx*(a+ib) = -kx*b + i*kx*a
                V1_c[idx][0] -= -kx * phi_i;  V1_c[idx][1] -=  kx * phi_r;
                V2_c[idx][0] -= -ky * phi_i;  V2_c[idx][1] -=  ky * phi_r;
                V3_c[idx][0] -= -kz * phi_i;  V3_c[idx][1] -=  kz * phi_r;
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
    // 谱系数为非归一化 DFT（= N × 归一化系数），除以 N 后才是物理量，与其他版本可比
    return global_max * ctx.norm_inv;
}

// 每个 RHS 的第一步（谱空间，一个循环）：rot = i·k × V，同时 rhs = -k² V
void compute_rot_and_viscous(const AccCtx& ctx, Fields& F) {
    Complex *V1 = F.V[0], *V2 = F.V[1], *V3 = F.V[2];
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double k2 = kx*kx + ky*ky + kz*kz;
                // i*(a+ib) = -b+ia
                F.rot[0][idx][0] = -(ky * V3[idx][1] - kz * V2[idx][1]);
                F.rot[0][idx][1] =   ky * V3[idx][0] - kz * V2[idx][0];
                F.rot[1][idx][0] = -(kz * V1[idx][1] - kx * V3[idx][1]);
                F.rot[1][idx][1] =   kz * V1[idx][0] - kx * V3[idx][0];
                F.rot[2][idx][0] = -(kx * V2[idx][1] - ky * V1[idx][1]);
                F.rot[2][idx][1] =   kx * V2[idx][0] - ky * V1[idx][0];
                for (int c = 0; c < 3; ++c) {
                    F.rhs[c][idx][0] = -k2 * F.V[c][idx][0];
                    F.rhs[c][idx][1] = -k2 * F.V[c][idx][1];
                }
            }
        }
    }
}

// RK4 的一个阶段：在当前阶段值 V 处计算 RHS，并把 V 推进到下一阶段值（阶段 4 之后为 V^{n+1}）。
// 每阶段 9 次 FFT（verify 另加 3 次外力 FFT）。c2r 前不再需要复制输入：之后不再使用 V 与 rot 的谱空间内容。
void rk4_stage(const AccCtx& ctx, accfft_plan* plan, Fields& F,
               double dt, double t, double dx, double dy, double dz, int stage) {
    const double inv_n = ctx.norm_inv;

    // 1. 旋度 + 粘性项（谱空间）
    compute_rot_and_viscous(ctx, F);
    g_timer.mark(PH_SPEC);

    // 2. 逆变换 V、rot → 实空间
    for (int c = 0; c < 3; ++c) accfft_execute_c2r(plan, F.V[c],   F.Vr[c]);
    for (int c = 0; c < 3; ++c) accfft_execute_c2r(plan, F.rot[c], F.rotr[c]);
    g_timer.mark(PH_FFT);

    // 3. 归一化（除以 N 得到物理值）+ 实空间叉乘 v × rot(v)，覆盖 rotr
    #pragma omp parallel for
    for (long idx = 0; idx < ctx.local_size_r; ++idx) {
        for (int c = 0; c < 3; ++c) { F.Vr[c][idx] *= inv_n; F.rotr[c][idx] *= inv_n; }
    }
    #pragma omp parallel for
    for (long idx = 0; idx < ctx.local_size_r; ++idx) {
        double v1 = F.Vr[0][idx], v2 = F.Vr[1][idx], v3 = F.Vr[2][idx];
        double w1 = F.rotr[0][idx], w2 = F.rotr[1][idx], w3 = F.rotr[2][idx];
        F.rotr[0][idx] = v2*w3 - v3*w2;   // (v×rot)_x
        F.rotr[1][idx] = v3*w1 - v1*w3;   // (v×rot)_y
        F.rotr[2][idx] = v1*w2 - v2*w1;   // (v×rot)_z
    }
    g_timer.mark(PH_PHYS);

    // 4. 正变换：叉乘 → rot（存放 nl，不归一化）
    for (int c = 0; c < 3; ++c) accfft_execute_r2c(plan, F.rotr[c], F.rot[c]);
    g_timer.mark(PH_FFT);

    // 5. 外力 [仅 verify]：填入空闲的 Vr，正变换到 V（f_c = DFT(f)，不归一化）
    if (g_with_force) {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < ctx.isize[0]; ++i) {
            for (int j = 0; j < ctx.isize[1]; ++j) {
                for (int k = 0; k < ctx.nz; ++k) {
                    long ridx = idx_r(ctx, i, j, k);
                    double x = phys_x(ctx, i, dx);
                    double y = phys_y(ctx, j, dy);
                    double z = phys_z(k, dz);
                    F.Vr[0][ridx] = func_f1(x, y, z, t);
                    F.Vr[1][ridx] = func_f2(x, y, z, t);
                    F.Vr[2][ridx] = func_f3(x, y, z, t);
                }
            }
        }
        for (int c = 0; c < 3; ++c) accfft_execute_r2c(plan, F.Vr[c], F.V[c]);
        g_timer.mark(PH_FORCE);
    }

    // 6. 求和 + 投影 + RK4 更新（一个循环，k 只存在于局部变量中）
    Complex *const *base = F.orig;
    double a_acc = 0.0, a_v = 0.0;
    int acc_mode = ACC_ADD;
    switch (stage) {
        case 1:  acc_mode = ACC_INIT; a_acc = dt/6.0; a_v = 0.5*dt; break;
        case 2:                       a_acc = dt/3.0; a_v = 0.5*dt; break;
        case 3:                       a_acc = dt/3.0; a_v = dt;     break;
        default: acc_mode = ACC_KEEP;                 a_v = dt/6.0; base = F.acc; break;
    }
    const bool force = g_with_force;
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double k2 = kx*kx + ky*ky + kz*kz;
                // RHS = nl + visc + f（与原代码相同的运算顺序）
                double q[3][2];
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 2; ++r) {
                        double v = F.rot[c][idx][r] + F.rhs[c][idx][r];
                        if (force) v += F.V[c][idx][r];
                        q[c][r] = v;
                    }
                // 投影
                double div_r = -(kx*q[0][1] + ky*q[1][1] + kz*q[2][1]);
                double div_i =   kx*q[0][0] + ky*q[1][0] + kz*q[2][0];
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-14) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                q[0][0] -= -kx * phi_i;  q[0][1] -=  kx * phi_r;
                q[1][0] -= -ky * phi_i;  q[1][1] -=  ky * phi_r;
                q[2][0] -= -kz * phi_i;  q[2][1] -=  kz * phi_r;
                // RK4
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 2; ++r) {
                        double bv = base[c][idx][r];
                        if      (acc_mode == ACC_INIT) F.acc[c][idx][r]  = bv + a_acc * q[c][r];
                        else if (acc_mode == ACC_ADD)  F.acc[c][idx][r] += a_acc * q[c][r];
                        F.V[c][idx][r] = bv + a_v * q[c][r];
                    }
            }
        }
    }
    g_timer.mark(PH_SPEC);
}

void rk4_step(const AccCtx& ctx, accfft_plan* plan, Fields& F,
              double dt, double t, double dx, double dy, double dz) {
    // 保存 V^n
    for (int c = 0; c < 3; ++c) memcpy(F.orig[c], F.V[c], ctx.local_size_c * sizeof(Complex));
    g_timer.mark(PH_SPEC);

    rk4_stage(ctx, plan, F, dt, t,          dx, dy, dz, 1);
    rk4_stage(ctx, plan, F, dt, t + 0.5*dt, dx, dy, dz, 2);
    rk4_stage(ctx, plan, F, dt, t + 0.5*dt, dx, dy, dz, 3);
    rk4_stage(ctx, plan, F, dt, t + dt,     dx, dy, dz, 4);
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
    int max_threads = nthreads;

    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    // ------------------------------------------------------------------
    // 初始化 AccFFT（2D pencil 分解）
    // ------------------------------------------------------------------
    AccCtx ctx;
    ctx.nx = nx;  ctx.ny = ny;  ctx.nz = nz;
    ctx.norm_inv = 1.0 / (static_cast<double>(nx) * ny * nz);

    int c_dims[2] = {0, 0};   // {0,0} 让 AccFFT 自动选择最优进程网格
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

    int n[3] = {nx, ny, nz};
    int alloc_max = accfft_local_size_dft_r2c(n, ctx.isize, ctx.istart,
                                               ctx.osize, ctx.ostart, c_comm);

    ctx.local_size_r = (long)ctx.isize[0] * ctx.isize[1] * nz;
    ctx.local_size_c = (long)ctx.osize[0] * ctx.osize[1] * ctx.osize[2];

    // AccFFT 内部用 int 保存缓冲区字节数（alloc_max / alloc_local），单进程超过 2 GB 就会溢出，
    // 属于库的限制。这里提前报错，避免分配错误大小后内存越界或得到错误结果。
    {
        const long avg_bytes = (long)nx * ny * (nz / 2 + 1) * (long)sizeof(Complex) / nprocs;
        const long local_bytes = ctx.local_size_c * (long)sizeof(Complex);
        int bad_local = (avg_bytes >= 2147483647L || (long)alloc_max < local_bytes) ? 1 : 0, bad = 0;
        MPI_Allreduce(&bad_local, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (bad) {
            if (rank == 0)
                cerr << "Error: AccFFT stores buffer sizes in int; this grid needs about "
                     << avg_bytes / 1048576L << " MB of FFT buffer per rank (limit 2048 MB). "
                     << "Use more MPI ranks.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

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
        cout << "Run mode: " << nsmode::mode_name(mode)
             << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << "\n";
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

    // 全尺寸数组：15 个复数（V、rot、rhs、V_orig、acc 各 3 个）+ 6 个实数（Vr、rotr 各 3 个），
    // bench 与 verify 相同（verify 的外力复用 Vr 与 V）
    Fields F;
    for (int c = 0; c < 3; ++c) {
        F.V[c]    = (Complex*) accfft_alloc(csz);
        F.rot[c]  = (Complex*) accfft_alloc(csz);
        F.rhs[c]  = (Complex*) accfft_alloc(csz);
        F.orig[c] = (Complex*) accfft_alloc(csz);
        F.acc[c]  = (Complex*) accfft_alloc(csz);
        F.Vr[c]   = (double*)  accfft_alloc(rsz);
        F.rotr[c] = (double*)  accfft_alloc(rsz);
    }
    double  *V1_r = F.Vr[0], *V2_r = F.Vr[1], *V3_r = F.Vr[2];
    Complex *V1_c = F.V[0],  *V2_c = F.V[1],  *V3_c = F.V[2];
    // 诊断（计时结束后）用的临时缓冲：复用 rhs1 的内存（此时 rhs 已无用）
    Complex* tmp_c = F.rhs[0];

    {
        const double GB = 1024.0*1024.0*1024.0;
        double mx[2] = { (double)csz, (double)rsz }, g[2];
        MPI_Allreduce(mx, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Memory report (per process, max over ranks):\n");
            printf("  Full-size arrays: 15 complex of %.3f GB, 6 real of %.3f GB\n", g[0]/GB, g[1]/GB);
            fflush(stdout);
        }
    }

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
    make_div_free(ctx, V1_c, V2_c, V3_c);

    // ------------------------------------------------------------------
    // 误差计算辅助函数（使用 tmp_c 作为临时缓冲）
    // ------------------------------------------------------------------
    // V_c → V_r（物理值，保护 V_c）
    auto to_physical = [&]() {
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
    };

    auto compute_error = [&](double time) -> pair<double, double> {
        to_physical();
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

    // 总动能 0.5·Σ|v|²（物理空间逐点求和，bench 模式下的结果合理性检查）
    auto compute_energy = [&]() -> double {
        to_physical();
        double local_e = 0.0;
        #pragma omp parallel for reduction(+:local_e)
        for (long idx = 0; idx < ctx.local_size_r; ++idx)
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
        rk4_step(ctx, plan, F, dt, t_cur, dx, dy, dz);
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

    // ------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------
    for (int c = 0; c < 3; ++c) {
        accfft_free(F.V[c]);    accfft_free(F.rot[c]);  accfft_free(F.rhs[c]);
        accfft_free(F.orig[c]); accfft_free(F.acc[c]);
        accfft_free(F.Vr[c]);   accfft_free(F.rotr[c]);
    }

    accfft_destroy_plan(plan);
    MPI_Comm_free(&c_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    accfft_cleanup();
    MPI_Finalize();
    return 0;
}
