#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3-mpi.h>
#include <iomanip>
#include <mpi.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <omp.h>
#include <sched.h>
#include "common_mode.h"
#include "fftlib/ns_fft_fftw.h"   // plan / execute code shared with scripts/fftbench

using namespace std;

// 运行模式与分阶段计时（见 common_mode.h）
//   g_with_force = true  (verify)：每次 RHS 计算解析外力 f（原始行为）
//   g_with_force = false (bench) ：f = 0，跳过外力填充、3 次正向 FFT、归一化与相加
static bool g_with_force = true;
static nsmode::CpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;

// ==============================================================================
// 全周期性边界条件的Navier-Stokes求解器
// 使用FFTW MPI进行3D R2C/C2R变换
// ==============================================================================

// FFTW MPI 3D r2c/c2r: 2 plans (r2c, c2r) reused for all arrays via the new-array execute API
// (fftw_mpi_execute_dft_r2c/c2r); see fftlib/ns_fft_fftw.h
static nsfft::Fftw g_fft;

// ==============================================================================
// 粘性系数、初值、时间步长控制（环境变量；都不设时行为与原程序逐位一致）
//   NS_NU=<nu>              粘性系数（默认 1；NS_IC=tg 且未设 NS_NU 时默认 1/1600，仅作保底）
//                           verify 的外力 f 中 Δv 项同样乘 nu，解析解对任意 nu 都成立
//   NS_IC=poly|tg           初值：poly = 原解析解（默认）；tg = Taylor–Green（只能配合 bench，f = 0）
//   NS_DT_MODE=fixed|cfl    默认 fixed（argv 的 dt 与 NSTEPS）
//   NS_CFL=<C>              cfl：对流界 dt_c = C / max_x Σ|u_i|/dx_i（默认 0.5）
//   NS_CFL_VISC=<Cv>        cfl：粘性界 dt_v = Cv / (nu·k²max)（默认 1.0；RK4 实轴稳定界约 2.785）
//   NS_DT_MAX=<dt>          cfl：dt 上限（可选）
//   NS_T_END=<T>            cfl：必须设置；按物理时间结束，最后一步截短正好落在 T
//                           （cfl 模式下 argv 的 dt 不使用，NSTEPS 是步数上限，达到上限仍未到 T 时报错）
//   NS_DT_PRINT=<K>         每 K 步打印一行 "DT ..."（cfl 默认 100；fixed 只有显式设置时才打印）
// 设置了以上任何一个变量时才输出新增的行（"DT"、"DTCFL_*"、"FINAL"），默认输出逐字不变。
// ==============================================================================
static double g_nu = 1.0;
enum { IC_POLY = 0, IC_TG = 1 };
static int g_ic = IC_POLY;

// ==============================================================================
// 解析解与强迫项（与 heFFTe/p3dfft 版本完全一致）
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
    double d2v1_dx2 = (t*t+1)*9*exp(sin(3*x+3*y))*((cos(3*x+3*y)*cos(3*x+3*y))-sin(3*x+3*y))*cos(6*z);
    double d2v1_dy2 = (t*t+1)*9*exp(sin(3*x+3*y))*((cos(3*x+3*y)*cos(3*x+3*y))-sin(3*x+3*y))*cos(6*z);
    double d2v1_dz2 = -(t*t+1)*36*exp(sin(3*x+3*y))*cos(6*z);
    return d2v1_dx2 + d2v1_dy2 + d2v1_dz2;
}
double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x,y,z,t);
}
double func_laplace_V3(double x, double y, double z, double t) {
    double d2v3_dx2 = -(t*t+1)*9*exp(sin(3*x+3*y))*cos(3*x+3*y)*((cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))-(2*sin(3*x+3*y)+1))*sin(6*z);
    double d2v3_dy2 = -(t*t+1)*9*exp(sin(3*x+3*y))*cos(3*x+3*y)*((cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))-(2*sin(3*x+3*y)+1))*sin(6*z);
    double d2v3_dz2 = (t*t+1)*36*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
    return d2v3_dx2 + d2v3_dy2 + d2v3_dz2;
}

double func_rot1(double x, double y, double z, double t){
    double dv3_dy = -(t*t+1)*3*exp(sin(3*x+3*y))*(cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))*sin(6*z);
    double dv2_dz = -(t*t+1)*6*exp(sin(3*x+3*y))*sin(6*z);
    return dv3_dy - dv2_dz;
}
double func_rot2(double x, double y, double z, double t){
    double dv1_dz = -(t*t+1)*6*exp(sin(3*x+3*y))*sin(6*z);
    double dv3_dx = -(t*t+1)*3*exp(sin(3*x+3*y))*(cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))*sin(6*z);
    return dv1_dz - dv3_dx;
}
double func_rot3(double x, double y, double z, double t){
    return 0;
}

double func_v_cross_rot1(double x, double y, double z, double t) {
    return func_V2(x,y,z,t)*func_rot3(x,y,z,t)-func_V3(x,y,z,t)*func_rot2(x,y,z,t);
}
double func_v_cross_rot2(double x, double y, double z, double t) {
    return func_V3(x,y,z,t)*func_rot1(x,y,z,t)-func_V1(x,y,z,t)*func_rot3(x,y,z,t);
}
double func_v_cross_rot3(double x, double y, double z, double t) {
    return func_V1(x,y,z,t)*func_rot2(x,y,z,t)-func_V2(x,y,z,t)*func_rot1(x,y,z,t);
}

double func_p(double x, double y, double z, double t) {
    return (t*t+1)*cos(x)*cos(y)*cos(z);
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

// g_nu = 1 时 g_nu*Δv 与 Δv 逐位相同（乘 1.0 是精确的）
double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x,y,z,t) - g_nu*func_laplace_V1(x,y,z,t) - func_v_cross_rot1(x,y,z,t) + func_grad_p1(x,y,z,t);
}
double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t) - g_nu*func_laplace_V2(x,y,z,t) - func_v_cross_rot2(x,y,z,t) + func_grad_p2(x,y,z,t);
}
double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t) - g_nu*func_laplace_V3(x,y,z,t) - func_v_cross_rot3(x,y,z,t) + func_grad_p3(x,y,z,t);
}

// Taylor–Green 初值（Re = 1/nu，U = 1，L = 1）
inline void tg_V(double x, double y, double z, double v[3]) {
    v[0] =  sin(x) * cos(y) * cos(z);
    v[1] = -cos(x) * sin(y) * cos(z);
    v[2] =  0.0;
}

// ==============================================================================
// FFT 初始化和辅助函数
// ==============================================================================

// FFT initialisation and plans: fftlib/ns_fft_fftw.h (nsfft::Fftw)

// ==============================================================================
// 频谱空间操作（谱空间为转置布局 [y_local][x][z]，y 按进程分布）
//
// 时间格式：经典 RK4，按累加形式存储（不存 k1..k4，k 只在一次循环内使用）：
//   k    = P[ nl/N + (-k^2 V) (+ f/N，仅 verify) ]
//   阶段1: acc  = V_orig + tau/6*k;   V = V_orig + tau/2*k
//   阶段2: acc += tau/3*k;            V = V_orig + tau/2*k
//   阶段3: acc += tau/3*k;            V = V_orig + tau*k
//   阶段4:                            V = acc + tau/6*k
// ==============================================================================

// 所有全尺寸数组（main 中分配一次）
struct Fields {
    fftw_complex *V[3];     // 阶段值（谱空间）；c2r 之后内容不再可靠（多维 c2r 可能破坏输入）
    fftw_complex *rot[3];   // 旋度；r2c 之后存放非线性项 nl = DFT(v x rot v)
    fftw_complex *rhs[3];   // -k^2 V（粘性项，由旋度循环同时写入）
    fftw_complex *orig[3];  // V^n
    fftw_complex *acc[3];   // RK4 累加量
    double *Vr[3];          // 实空间速度；verify 时在叉乘之后用来存放外力 f
    double *rotr[3];        // 实空间旋度，叉乘结果覆盖于此
};

struct Grid {
    ptrdiff_t nx, ny, nz;
    ptrdiff_t local_n0, local_0_start;   // 实空间：x 分布
    ptrdiff_t local_n1, local_1_start;   // 谱空间（转置）：y 分布
    double dx, dy, dz;
};

enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

/**
 * 投影到无散空间：V -= ∇(∇⁻²·∇·V)（频谱空间，in-place；仅用于初始条件）
 * div = ik·V，phi = div/(-k²)，V -= ik*phi
 */
void make_div_free(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c, const Grid& g) {
    const ptrdiff_t nx = g.nx, ny = g.ny, nz_c = g.nz/2 + 1;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < g.local_n1; ++j) {          // 转置布局 [y_local][x][z]
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (j * nx + i) * nz_c + k;
                ptrdiff_t jg = g.local_1_start + j;
                double kx = (i  <= nx/2) ? (double)i  : (double)(i  - nx);
                double ky = (jg <= ny/2) ? (double)jg : (double)(jg - ny);
                double kz = (double)k;
                double k2 = kx*kx + ky*ky + kz*kz;
                double div_r = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                double div_i =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-10) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                // V -= i*k*phi：i*kx*(a+ib) = -kx*b + i*kx*a
                V1_c[idx][0] -= -kx * phi_i;  V1_c[idx][1] -=  kx * phi_r;
                V2_c[idx][0] -= -ky * phi_i;  V2_c[idx][1] -=  ky * phi_r;
                V3_c[idx][0] -= -kz * phi_i;  V3_c[idx][1] -=  kz * phi_r;
            }
        }
    }
}

/**
 * 诊断：max|div V| = max|ik·V|（频谱空间，与 make_div_free 相同的波数约定）
 */
double compute_div_max(const fftw_complex* V1_c, const fftw_complex* V2_c, const fftw_complex* V3_c,
                       ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                       ptrdiff_t local_n1, ptrdiff_t local_1_start) {
    ptrdiff_t nz_c = nz/2 + 1;
    double local_max = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:local_max)
    for(ptrdiff_t j = 0; j < local_n1; ++j) {          // 转置布局 [y_local][x][z]
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (j * nx + i) * nz_c + k;
                ptrdiff_t jg = local_1_start + j;
                double kx = (i  <= nx/2) ? (double)i  : (double)(i  - nx);
                double ky = (jg <= ny/2) ? (double)jg : (double)(jg - ny);
                double kz = (double)k;
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

/**
 * 每个 RHS 的第一步（谱空间，一个循环）：rot = ik × V，同时 rhs = -k² V
 */
void compute_rot_and_viscous(Fields& F, const Grid& g) {
    const ptrdiff_t nx = g.nx, ny = g.ny, nz_c = g.nz/2 + 1;
    fftw_complex *V1 = F.V[0], *V2 = F.V[1], *V3 = F.V[2];
    const double nu = g_nu;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < g.local_n1; ++j) {          // 转置布局 [y_local][x][z]
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (j * nx + i) * nz_c + k;
                ptrdiff_t jg = g.local_1_start + j;
                double kx = (i  <= nx/2) ? (double)i  : (double)(i  - nx);
                double ky = (jg <= ny/2) ? (double)jg : (double)(jg - ny);
                double kz = (double)k;
                double k2 = nu * (kx*kx + ky*ky + kz*kz);   // nu = 1 时与原来逐位相同

                F.rot[0][idx][0] = -(ky*V3[idx][1] - kz*V2[idx][1]);
                F.rot[0][idx][1] =   ky*V3[idx][0] - kz*V2[idx][0];
                F.rot[1][idx][0] = -(kz*V1[idx][1] - kx*V3[idx][1]);
                F.rot[1][idx][1] =   kz*V1[idx][0] - kx*V3[idx][0];
                F.rot[2][idx][0] = -(kx*V2[idx][1] - ky*V1[idx][1]);
                F.rot[2][idx][1] =   kx*V2[idx][0] - ky*V1[idx][0];
                for (int c = 0; c < 3; ++c) {
                    F.rhs[c][idx][0] = -k2 * F.V[c][idx][0];
                    F.rhs[c][idx][1] = -k2 * F.V[c][idx][1];
                }
            }
        }
    }
}

// ==============================================================================
// 时间步长控制（NS_DT_MODE=cfl）与逐步诊断
//   每步只在 RK 第 1 级扫描一次：第 1 级逆变换得到的 Vr = u^n、rotr = ω^n（叉乘之前），
//   不额外做 FFT。扫描（含 Allreduce）单独计时，不计入 PH_* 各阶段（但计入每步墙钟时间）。
// ==============================================================================
enum { LIM_CONV = 0, LIM_VISC = 1, LIM_DTMAX = 2, LIM_TEND = 3, LIM_FIXED = 4, LIM_COUNT = 5 };
static const char* lim_name(int l) {
    static const char* N[LIM_COUNT] = {"conv", "visc", "dtmax", "tend", "fixed"};
    return N[l];
}

struct StepCtl {
    bool cfl = false;           // NS_DT_MODE=cfl
    bool scan = false;          // 本步是否在第 1 级扫描（cfl：每步；fixed：仅打印步）
    double C = 0.5, Cv = 1.0, dt_max = 0.0, t_end = 0.0;
    double k2max = 0.0;         // 最大 |k|²（本程序波数约定：各方向 0..N/2）
    double inv_dx[3] = {0, 0, 0};
    double t = 0.0;             // 本步开始时刻（cfl：由 dt 累加）
    // 第 1 级扫描结果（全局）
    double maxS = 0.0;          // max_x Σ|u_i|/dx_i
    double umax = 0.0;          // max_x |u|
    double E = 0.0;             // 0.5 <|u|²>
    double eps = 0.0;           // nu <|ω|²>
    // 本步的 dt 选择
    double dt = 0.0;
    int lim = LIM_FIXED;
    double scan_time = 0.0;     // 扫描总耗时（max over ranks 的逐步和由调用方处理）
};

static void scan_stage1(const Fields& F, const Grid& g, StepCtl& ctl) {
    double t0 = nsmode::wall_now();
    const ptrdiff_t ny = g.ny, nz = g.nz, rstride = 2*(nz/2+1);
    const double ix = ctl.inv_dx[0], iy = ctl.inv_dx[1], iz = ctl.inv_dx[2];
    double mS = 0.0, mU2 = 0.0, sU2 = 0.0, sW2 = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:mS,mU2) reduction(+:sU2,sW2)
    for(ptrdiff_t i = 0; i < g.local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t idx = (i * ny + j) * rstride + k;
                double u = F.Vr[0][idx], v = F.Vr[1][idx], w = F.Vr[2][idx];
                double a = F.rotr[0][idx], b = F.rotr[1][idx], c = F.rotr[2][idx];
                double S = fabs(u)*ix + fabs(v)*iy + fabs(w)*iz;
                double u2 = u*u + v*v + w*w;
                // NaN 必须传播出来（max 归约会吞掉 NaN），用 !(x <= m) 判断
                if (!(S  <= mS))  mS  = S;
                if (!(u2 <= mU2)) mU2 = u2;
                sU2 += u2;
                sW2 += a*a + b*b + c*c;
            }
        }
    }
    double lmax[2] = {mS, mU2}, gmax[2], lsum[2] = {sU2, sW2}, gsum[2];
    MPI_Allreduce(lmax, gmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(lsum, gsum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double N = (double)(g.nx * g.ny * g.nz);
    // MPI_MAX 对 NaN 的结果不确定：本地出现 NaN 时把总和也置为 NaN（求和会传播）
    ctl.maxS = gmax[0];
    ctl.umax = sqrt(gmax[1]);
    ctl.E    = 0.5 * gsum[0] / N;
    ctl.eps  = g_nu * gsum[1] / N;
    if (!std::isfinite(ctl.E)) { ctl.maxS = ctl.E; ctl.umax = ctl.E; }

    if (ctl.cfl) {
        double dt_c = (ctl.maxS > 0.0) ? ctl.C / ctl.maxS : HUGE_VAL;
        double dt_v = (g_nu * ctl.k2max > 0.0) ? ctl.Cv / (g_nu * ctl.k2max) : HUGE_VAL;
        double dt = dt_c; int lim = LIM_CONV;
        if (dt_v < dt) { dt = dt_v; lim = LIM_VISC; }
        if (ctl.dt_max > 0.0 && ctl.dt_max < dt) { dt = ctl.dt_max; lim = LIM_DTMAX; }
        if (ctl.t + dt >= ctl.t_end) { dt = ctl.t_end - ctl.t; lim = LIM_TEND; }
        ctl.dt = dt; ctl.lim = lim;
    }
    ctl.scan_time += nsmode::wall_now() - t0;
}

/**
 * RK4 的一个阶段：在当前阶段值 V 处计算 RHS，并把 V 推进到下一阶段值
 * （阶段 4 之后为 V^{n+1}）。每阶段 9 次 FFT（verify 另加 3 次外力 FFT）。
 * cfl 模式：第 1 级在逆变换之后由 scan_stage1 确定 tau（之前 tau 未使用）。
 */
void rk4_stage(Fields& F, const Grid& g, double& tau, double t, int stage, StepCtl& ctl) {
    const ptrdiff_t nx = g.nx, ny = g.ny, nz = g.nz, nz_c = nz/2 + 1;
    const double norm = (double)(nx * ny * nz);
    const ptrdiff_t rstride = 2*(nz/2+1);   // R2C 实空间 padding 后的行长

    // 1. 旋度 + 粘性项（谱空间）
    compute_rot_and_viscous(F, g);
    g_timer.mark(PH_SPEC);

    // 2. 逆变换：V、rot → 实空间（之后 V 与 rot 的谱空间内容都不再使用）
    for (int c = 0; c < 3; ++c) g_fft.backward(F.V[c],   F.Vr[c]);
    for (int c = 0; c < 3; ++c) g_fft.backward(F.rot[c], F.rotr[c]);
    g_timer.mark(PH_FFT);

    // 2b. [cfl 每步 / fixed 打印步] 第 1 级：u^n、ω^n 的扫描与 dt 选择（单独计时，不计入 PH_*）
    if (stage == 1 && ctl.scan) {
        scan_stage1(F, g, ctl);
        if (ctl.cfl) tau = ctl.dt;
        g_timer.start();
    }

    // 3. 实空间叉乘 v × rot(v)，覆盖 rotr
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < g.local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t idx = (i * ny + j) * rstride + k;
                double v1 = F.Vr[0][idx], v2 = F.Vr[1][idx], v3 = F.Vr[2][idx];
                double w1 = F.rotr[0][idx], w2 = F.rotr[1][idx], w3 = F.rotr[2][idx];
                F.rotr[0][idx] = v2*w3 - v3*w2;  // (v×rot)_x
                F.rotr[1][idx] = v3*w1 - v1*w3;  // (v×rot)_y
                F.rotr[2][idx] = v1*w2 - v2*w1;  // (v×rot)_z
            }
        }
    }
    g_timer.mark(PH_PHYS);

    // 4. 正变换：叉乘 → rot（存放 nl，未归一化）
    for (int c = 0; c < 3; ++c) g_fft.forward(F.rotr[c], F.rot[c]);
    g_timer.mark(PH_FFT);

    // 5. 外力 [仅 verify]：填入空闲的 Vr，正变换到 V（未归一化，阶段循环中除以 N）
    if (g_with_force) {
        #pragma omp parallel for collapse(3)
        for(ptrdiff_t i = 0; i < g.local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz; ++k) {
                    ptrdiff_t ig  = g.local_0_start + i;
                    double x = ig * g.dx, y = j * g.dy, z = k * g.dz;
                    ptrdiff_t idx = (i * ny + j) * rstride + k;
                    F.Vr[0][idx] = func_f1(x, y, z, t);
                    F.Vr[1][idx] = func_f2(x, y, z, t);
                    F.Vr[2][idx] = func_f3(x, y, z, t);
                }
            }
        }
        for (int c = 0; c < 3; ++c) g_fft.forward(F.Vr[c], F.V[c]);
        g_timer.mark(PH_FORCE);
    }

    // 6. 归一化 + 求和 + 投影 + RK4 更新（一个循环，k 只存在于局部变量中）
    fftw_complex *const *base = F.orig;
    double a_acc = 0.0, a_v = 0.0;
    int acc_mode = ACC_ADD;
    switch (stage) {
        case 1:  acc_mode = ACC_INIT; a_acc = tau/6.0; a_v = 0.5*tau; break;
        case 2:                       a_acc = tau/3.0; a_v = 0.5*tau; break;
        case 3:                       a_acc = tau/3.0; a_v = tau;     break;
        default: acc_mode = ACC_KEEP;                  a_v = tau/6.0; base = F.acc; break;
    }
    const bool force = g_with_force;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < g.local_n1; ++j) {          // 转置布局 [y_local][x][z]
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (j * nx + i) * nz_c + k;
                ptrdiff_t jg = g.local_1_start + j;
                double kx = (i  <= nx/2) ? (double)i  : (double)(i  - nx);
                double ky = (jg <= ny/2) ? (double)jg : (double)(jg - ny);
                double kz = (double)k;
                double k2 = kx*kx + ky*ky + kz*kz;

                // RHS = nl + visc + f（与原代码相同的运算顺序）
                double q[3][2];
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 2; ++r) {
                        double v = F.rot[c][idx][r] / norm + F.rhs[c][idx][r];
                        if (force) v += F.V[c][idx][r] / norm;
                        q[c][r] = v;
                    }
                // 投影
                double div_r = -(kx*q[0][1] + ky*q[1][1] + kz*q[2][1]);
                double div_i =   kx*q[0][0] + ky*q[1][0] + kz*q[2][0];
                double phi_r = 0.0, phi_i = 0.0;
                if (k2 > 1e-10) { phi_r = div_r / (-k2); phi_i = div_i / (-k2); }
                q[0][0] -= -kx * phi_i;  q[0][1] -=  kx * phi_r;
                q[1][0] -= -ky * phi_i;  q[1][1] -=  ky * phi_r;
                q[2][0] -= -kz * phi_i;  q[2][1] -=  kz * phi_r;
                // RK4
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 2; ++r) {
                        double b = base[c][idx][r];
                        if      (acc_mode == ACC_INIT) F.acc[c][idx][r]  = b + a_acc * q[c][r];
                        else if (acc_mode == ACC_ADD)  F.acc[c][idx][r] += a_acc * q[c][r];
                        F.V[c][idx][r] = b + a_v * q[c][r];
                    }
            }
        }
    }
    g_timer.mark(PH_SPEC);
}

// tau：fixed 为输入；cfl 由第 1 级确定并返回
void rk4_step(Fields& F, const Grid& g, ptrdiff_t alloc_local, double& tau, double t, StepCtl& ctl) {
    // 保存 V^n
    for (int c = 0; c < 3; ++c) memcpy(F.orig[c], F.V[c], alloc_local * sizeof(fftw_complex));
    g_timer.mark(PH_SPEC);

    rk4_stage(F, g, tau, t,           1, ctl);
    rk4_stage(F, g, tau, t + 0.5*tau, 2, ctl);
    rk4_stage(F, g, tau, t + 0.5*tau, 3, ctl);
    rk4_stage(F, g, tau, t + tau,     4, ctl);
}

// ==============================================================================
// 环境变量解析（各进程环境相同，srun/mpirun 导出，无需广播）
// ==============================================================================
static bool g_env_err = false;
static bool env_set(const char* name) { const char* s = getenv(name); return s && *s; }
static double env_double(const char* name, double def, bool positive) {
    const char* s = getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    double v = strtod(s, &end);
    if (*end || !std::isfinite(v) || (positive ? !(v > 0.0) : !(v >= 0.0))) {
        fprintf(stderr, "Error: %s=%s (expected a %s number)\n", name, s, positive ? "positive" : "non-negative");
        g_env_err = true;
        return def;
    }
    return v;
}
static long env_long(const char* name, long def) {
    const char* s = getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (*end || v <= 0) { fprintf(stderr, "Error: %s=%s (expected a positive integer)\n", name, s); g_env_err = true; return def; }
    return v;
}

// 最近若干步的扫描结果（BLOWUP 时打印，帮助判断是 CFL 还是混叠能量堆积）
struct HistRec { long step; double t, dt, umax, E, eps, Ceff; };

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char **argv) {
    // MPI + OpenMP + FFTW 初始化
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " " << nsmode::usage_tail(true) << endl;
            cerr << "  Example: mpirun -np 4 " << argv[0] << " 64 64 64 0.00001 100 4 bench" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    nsmode::RunArgs rargs;
    {
        string perr;
        if (!nsmode::parse_tail_args(argc, argv, 6, /*allow_threads=*/true, rargs, perr)) {
            if (rank == 0) cerr << "Error: " << perr << "\nUsage: " << argv[0] << " "
                                << nsmode::usage_tail(true) << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const nsmode::RunMode mode = rargs.mode;
    g_with_force = (mode == nsmode::MODE_VERIFY);

    ptrdiff_t nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    double tau = atof(argv[4]);
    ptrdiff_t nt_run = atoi(argv[5]);
    int nthreads = (rargs.omp_threads > 0) ? rargs.omp_threads : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    int max_threads = nthreads;

    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;
    double dx = L_x/nx, dy = L_y/ny, dz = L_z/nz;

    // ---- 粘性系数、初值、时间步长控制（见文件开头说明） ----
    const bool ext_out = env_set("NS_NU") || env_set("NS_IC") || env_set("NS_DT_MODE") || env_set("NS_CFL") ||
                         env_set("NS_CFL_VISC") || env_set("NS_DT_MAX") || env_set("NS_T_END") || env_set("NS_DT_PRINT");
    StepCtl ctl;
    if (const char* s = getenv("NS_IC")) {
        if      (!strcmp(s, "poly") || !*s) g_ic = IC_POLY;
        else if (!strcmp(s, "tg"))          g_ic = IC_TG;
        else { if (rank == 0) fprintf(stderr, "Error: NS_IC=%s (expected poly|tg)\n", s); g_env_err = true; }
    }
    const bool nu_given = env_set("NS_NU");
    g_nu = env_double("NS_NU", (g_ic == IC_TG) ? 1.0 / 1600.0 : 1.0, /*positive=*/false);
    if (const char* s = getenv("NS_DT_MODE")) {
        if      (!strcmp(s, "fixed") || !*s) ctl.cfl = false;
        else if (!strcmp(s, "cfl"))          ctl.cfl = true;
        else { if (rank == 0) fprintf(stderr, "Error: NS_DT_MODE=%s (expected fixed|cfl)\n", s); g_env_err = true; }
    }
    ctl.C      = env_double("NS_CFL", 0.5, true);
    ctl.Cv     = env_double("NS_CFL_VISC", 1.0, true);
    ctl.dt_max = env_double("NS_DT_MAX", 0.0, true);
    ctl.t_end  = env_double("NS_T_END", 0.0, true);
    const long dt_print = env_long("NS_DT_PRINT", ctl.cfl ? 100 : 0);   // 0 = 不打印
    if (g_ic == IC_TG && mode == nsmode::MODE_VERIFY) {
        if (rank == 0) fprintf(stderr, "Error: NS_IC=tg has no analytic solution; use bench mode (f = 0)\n");
        g_env_err = true;
    }
    if (ctl.cfl && !env_set("NS_T_END")) {
        if (rank == 0) fprintf(stderr, "Error: NS_DT_MODE=cfl needs NS_T_END\n");
        g_env_err = true;
    }
    if (!ctl.cfl && (env_set("NS_T_END") || env_set("NS_CFL") || env_set("NS_CFL_VISC") || env_set("NS_DT_MAX"))) {
        if (rank == 0) fprintf(stderr, "Error: NS_T_END / NS_CFL / NS_CFL_VISC / NS_DT_MAX need NS_DT_MODE=cfl "
                                       "(fixed mode: dt and NSTEPS from the command line)\n");
        g_env_err = true;
    }
    if (g_env_err) MPI_Abort(MPI_COMM_WORLD, 1);
    ctl.inv_dx[0] = 1.0 / dx; ctl.inv_dx[1] = 1.0 / dy; ctl.inv_dx[2] = 1.0 / dz;
    ctl.k2max = (double)((nx/2)*(nx/2) + (ny/2)*(ny/2) + (nz/2)*(nz/2));

    // FFTW 多线程：须在 fftw_mpi_init 之前 init_threads，在建计划之前设定线程数。
    // 必须链接 libfftw3_omp（而非 libfftw3_threads）：_threads 版用 pthread 创建工作线程，
    // 工作线程继承主线程的 CPU 亲和性；设置 OMP_PROC_BIND/OMP_PLACES 时 libgomp 在启动时
    // 就把主线程绑到第一个 place，所有 FFTW 工作线程挤在同一个 place 上，FFT 退化为串行。
    // _omp 版直接使用 OpenMP 线程组，与逐点循环的线程绑定方式相同。
    g_fft.init(max_threads);      // fftw_init_threads, fftw_plan_with_nthreads, fftw_mpi_init

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Navier-Stokes Solver - FFTW MPI Version (1D Slab)" << endl;
        cout << "============================================================" << endl;
        cout << "Grid: " << nx << " x " << ny << " x " << nz
             << ", dt=" << scientific << tau << ", steps=" << nt_run << endl;
        cout << "Domain: [0, 2π]³" << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads/process: " << max_threads << endl;
        {   // threads really in effect (runtime), for the logs of scripts/cpu_final
            int in_par = 0;
            #pragma omp parallel
            {
                #pragma omp master
                in_par = omp_get_num_threads();
            }
            cout << "Threads (runtime): omp_get_max_threads()=" << omp_get_max_threads()
                 << " omp_get_num_threads(in parallel)=" << in_par
                 << "\n";
        }
        {
            cpu_set_t cs; CPU_ZERO(&cs);
            int ncpu = (sched_getaffinity(0, sizeof(cs), &cs) == 0) ? CPU_COUNT(&cs) : -1;
            cout << "FFTW planner threads: " << fftw_planner_nthreads()
                 << "  (main-thread CPU affinity: " << ncpu << " CPUs)" << endl;
        }
        cout << "Run mode: " << nsmode::mode_name(mode)
             << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << endl;
        // FFTW MPI 只有 slab（x 切分）；NS_DECOMP 不起作用，只打印出来便于核对
        g_fft.print_decomp();
        if (ext_out) {
            printf("DTCFL_CONFIG dt_mode=%s ic=%s nu=%.10e%s cfl=%.4f cfl_visc=%.4f dt_max=%.6e t_end=%.6f "
                   "dt_print=%ld k2max=%.0f dt_visc=%.6e\n",
                   ctl.cfl ? "cfl" : "fixed", g_ic == IC_TG ? "tg" : "poly", g_nu,
                   nu_given ? "" : "(default)", ctl.C, ctl.Cv, ctl.dt_max, ctl.t_end, dt_print, ctl.k2max,
                   g_nu > 0 ? ctl.Cv / (g_nu * ctl.k2max) : HUGE_VAL);
            if (ctl.cfl)
                printf("  cfl mode: argv dt (%.3e) unused; NSTEPS (%ld) = step cap\n", tau, (long)nt_run);
            fflush(stdout);
        }
        cout << "============================================================" << endl;
    }

    // 获取本地分配大小（FFTW MPI 1D slab 分解）
    // 实空间：x 分布 [x_local][y][z]；谱空间（转置输出）：y 分布 [y_local][x][z]
    g_fft.decompose(nx, ny, nz);
    g_fft.setup();
    const ptrdiff_t alloc_local = g_fft.alloc_local, local_n0 = g_fft.local_n0, local_0_start = g_fft.local_0_start,
                    local_n1 = g_fft.local_n1, local_1_start = g_fft.local_1_start;

    // ------------------------------------------------------------------
    // 预分配所有内存（热循环中零动态分配）
    // ------------------------------------------------------------------

    // 全尺寸数组：15 个复数（V、rot、rhs、V_orig、acc 各 3 个）+ 6 个实数（Vr、rotr 各 3 个），
    // bench 与 verify 相同（verify 的外力复用 Vr 与 V）
    Fields F;
    for (int c = 0; c < 3; ++c) {
        F.V[c]    = fftw_alloc_complex(alloc_local);
        F.rot[c]  = fftw_alloc_complex(alloc_local);
        F.rhs[c]  = fftw_alloc_complex(alloc_local);
        F.orig[c] = fftw_alloc_complex(alloc_local);
        F.acc[c]  = fftw_alloc_complex(alloc_local);
        F.Vr[c]   = fftw_alloc_real(2 * alloc_local);
        F.rotr[c] = fftw_alloc_real(2 * alloc_local);
    }
    fftw_complex *V1_c = F.V[0], *V2_c = F.V[1], *V3_c = F.V[2];
    double *V1_r = F.Vr[0], *V2_r = F.Vr[1], *V3_r = F.Vr[2];
    Grid grid{nx, ny, nz, local_n0, local_0_start, local_n1, local_1_start, dx, dy, dz};

    if (rank == 0) {
        const double GB = 1024.0*1024.0*1024.0;
        printf("Memory report (per process):\n");
        printf("  Full-size arrays: 15 complex of %.3f GB, 6 real of %.3f GB\n",
               alloc_local*sizeof(fftw_complex)/GB, 2*alloc_local*sizeof(double)/GB);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // 初始化所有 FFTW 计划（所有数组已预分配，指针固定）
    // ------------------------------------------------------------------
    // 规划强度：NS_FFTW_PLAN = estimate | measure（默认）| patient | exhaustive；NS_PRINT_PLAN=1 打印 plan
    // 只需任意一对合法缓冲区创建 plan；执行时通过 fftw_mpi_execute_* 换指针（fftlib/ns_fft_fftw.h）
    g_fft.plan(V1_r, V1_c);

    // ------------------------------------------------------------------
    // 初始条件（t=0）
    // ------------------------------------------------------------------
    if (rank == 0) cout << "Setting initial conditions..." << endl;

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t ig  = local_0_start + i;
                double x = ig * dx, y = j * dy, z = k * dz;
                ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                if (g_ic == IC_TG) {
                    double v[3];
                    tg_V(x, y, z, v);
                    V1_r[idx] = v[0];  V2_r[idx] = v[1];  V3_r[idx] = v[2];
                } else {
                    V1_r[idx] = func_V1(x, y, z, 0.0);
                    V2_r[idx] = func_V2(x, y, z, 0.0);
                    V3_r[idx] = func_V3(x, y, z, 0.0);
                }
            }
        }
    }

    // 正变换 → 归一化（V_c = DFT(V_r)/N）
    g_fft.forward(V1_r, V1_c);
    g_fft.forward(V2_r, V2_c);
    g_fft.forward(V3_r, V3_c);

    double norm_factor = (double)(nx * ny * nz);
    ptrdiff_t nz_c = nz/2+1;
    ptrdiff_t total_c = local_n1 * nx * nz_c;   // 谱空间（转置布局）本地元素数
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        V1_c[i][0] /= norm_factor;  V1_c[i][1] /= norm_factor;
        V2_c[i][0] /= norm_factor;  V2_c[i][1] /= norm_factor;
        V3_c[i][0] /= norm_factor;  V3_c[i][1] /= norm_factor;
    }

    // 投影初始条件到无散空间
    if (rank == 0) cout << "Projecting initial condition to divergence-free space..." << endl;
    make_div_free(V1_c, V2_c, V3_c, grid);

    // ------------------------------------------------------------------
    // 时间推进（RK4）
    // ------------------------------------------------------------------
    double t_wall_total = 0.0;

    // fixed：t = it*tau（与原程序相同）；cfl：t 由 dt 累加，最后一步截短到 NS_T_END
    long steps_done = 0;
    bool blowup = false, cap_hit = false;
    double dt_min = HUGE_VAL, dt_big = 0.0, E0 = -1.0;
    long lim_count[LIM_COUNT] = {0, 0, 0, 0, 0};
    const int NHIST = 8;
    HistRec hist[NHIST];
    long nhist = 0;
    ctl.t = 0.0;
    if (rank == 0 && dt_print > 0)
        printf("DT_HEADER step t dt umax E eps Ceff lim   (E = 0.5<|u|^2>, eps = nu<|w|^2>, at the start of the step)\n");

    for(ptrdiff_t it = 0; ctl.cfl || it < nt_run; ++it) {
        if (ctl.cfl) {
            if (!(ctl.t < ctl.t_end)) break;
            if (it >= nt_run) { cap_hit = true; break; }
        }
        double t_cur = ctl.cfl ? ctl.t : it * tau;
        const bool pstep = dt_print > 0 && (it % dt_print == 0);
        ctl.scan = ctl.cfl || pstep;
        double tau_step = tau;          // cfl：第 1 级改写为本步 dt
        {
            double t0 = MPI_Wtime();
            g_timer.start();
            rk4_step(F, grid, alloc_local, tau_step, t_cur, ctl);
            double step_time = MPI_Wtime() - t0;
            double global_step;
            MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            t_wall_total += global_step;
        }
        ++steps_done;
        const int lim = ctl.cfl ? ctl.lim : LIM_FIXED;
        lim_count[lim]++;
        if (ctl.cfl) {
            if (lim != LIM_TEND && tau_step < dt_min) dt_min = tau_step;   // 不含截短的最后一步
            if (tau_step > dt_big) dt_big = tau_step;
        }
        if (ctl.scan) {
            if (E0 < 0.0) E0 = ctl.E;
            const double Ceff = tau_step * ctl.maxS;
            hist[nhist % NHIST] = HistRec{(long)it, t_cur, tau_step, ctl.umax, ctl.E, ctl.eps, Ceff};
            ++nhist;
            // 爆炸判据（扫描值在所有进程上相同，因此所有进程同时退出）
            if (!std::isfinite(ctl.E) || !std::isfinite(ctl.maxS) || ctl.E > 1e3 * E0 || !(tau_step > 0.0)) {
                blowup = true;
                if (rank == 0) {
                    printf("BLOWUP step=%ld t=%.10e dt=%.6e umax=%.6e E=%.10e (E0=%.10e) cfl=%.4f\n",
                           (long)it, t_cur, tau_step, ctl.umax, ctl.E, E0, ctl.C);
                    printf("BLOWUP_HIST step t dt umax E eps Ceff   (last %ld scanned steps)\n",
                           nhist < NHIST ? nhist : (long)NHIST);
                    for (long h = (nhist > NHIST ? nhist - NHIST : 0); h < nhist; ++h) {
                        const HistRec& r = hist[h % NHIST];
                        printf("BLOWUP_HIST %ld %.10e %.6e %.6e %.10e %.10e %.4f\n",
                               r.step, r.t, r.dt, r.umax, r.E, r.eps, r.Ceff);
                    }
                    fflush(stdout);
                }
                break;
            }
            if (rank == 0 && dt_print > 0 && (pstep || (ctl.cfl && lim == LIM_TEND))) {
                printf("DT %ld %.10e %.6e %.6e %.10e %.10e %.4f %s\n",
                       (long)it, t_cur, tau_step, ctl.umax, ctl.E, ctl.eps, Ceff, lim_name(lim));
                fflush(stdout);
            }
        }
        if (ctl.cfl) ctl.t = (lim == LIM_TEND) ? ctl.t_end : ctl.t + tau_step;
    }
    if (cap_hit && rank == 0) {
        fprintf(stderr, "Error: NSTEPS cap (%ld) reached at t=%.6e < NS_T_END=%.6e\n", (long)nt_run, ctl.t, ctl.t_end);
        printf("DTCFL_CAP_HIT steps=%ld t=%.10e t_end=%.10e\n", steps_done, ctl.t, ctl.t_end);
    }
    // 报告用的步数：默认（fixed、未爆炸）与原程序相同，为 NSTEPS
    const long nsteps_rep = (ctl.cfl || blowup) ? steps_done : (long)nt_run;

    double phase_max[nsmode::PH_COUNT];
    nsmode::reduce_phases_max(g_timer.acc, phase_max);
    double scan_max = 0.0;
    MPI_Allreduce(&ctl.scan_time, &scan_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 最终诊断（不计入计时）：verify → L2 误差；bench → 总动能；两者都输出 max|div V|
    {
        double t_final = ctl.cfl ? ctl.t : (blowup ? steps_done * tau : nt_run * tau);
        // 散度须在 c2r 之前计算（多维 c2r 可能破坏输入 V_c）
        double max_div = compute_div_max(V1_c, V2_c, V3_c, nx, ny, nz, local_n1, local_1_start);
        if (ext_out) compute_rot_and_viscous(F, grid);    // t=T 的旋度（谱空间），供 E/eps 诊断
        g_fft.backward(V1_c, V1_r);
        g_fft.backward(V2_c, V2_r);
        g_fft.backward(V3_c, V3_r);
        if (ext_out) {
            for (int c = 0; c < 3; ++c) g_fft.backward(F.rot[c], F.rotr[c]);
            StepCtl fin = ctl;
            fin.cfl = false;
            scan_stage1(F, grid, fin);
            if (rank == 0) {
                printf("FINAL t=%.10e E=%.12e eps=%.12e umax=%.10e maxS=%.10e\n", t_final, fin.E, fin.eps, fin.umax, fin.maxS);
                fflush(stdout);
            }
        }
        double local_err = 0.0, local_e = 0.0;
        if (mode == nsmode::MODE_VERIFY) {
            #pragma omp parallel for collapse(3) reduction(+:local_err)
            for(ptrdiff_t i = 0; i < local_n0; ++i) {
                for(ptrdiff_t j = 0; j < ny; ++j) {
                    for(ptrdiff_t k = 0; k < nz; ++k) {
                        ptrdiff_t ig  = local_0_start + i;
                        double x = ig * dx, y = j * dy, z = k * dz;
                        ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                        double d1 = V1_r[idx] - func_V1(x, y, z, t_final);
                        double d2 = V2_r[idx] - func_V2(x, y, z, t_final);
                        double d3 = V3_r[idx] - func_V3(x, y, z, t_final);
                        local_err += d1*d1 + d2*d2 + d3*d3;
                    }
                }
            }
        } else {
            // 总动能 0.5·Σ|v|²（物理空间逐点求和）
            #pragma omp parallel for collapse(3) reduction(+:local_e)
            for(ptrdiff_t i = 0; i < local_n0; ++i) {
                for(ptrdiff_t j = 0; j < ny; ++j) {
                    for(ptrdiff_t k = 0; k < nz; ++k) {
                        ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                        local_e += V1_r[idx]*V1_r[idx] + V2_r[idx]*V2_r[idx] + V3_r[idx]*V3_r[idx];
                    }
                }
            }
        }
        double global_err = 0.0, global_e = 0.0;
        MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_e,   &global_e,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (mode == nsmode::MODE_VERIFY)
                cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(4) << sqrt(global_err * dx * dy * dz) << endl;
            else
                cout << "  Kinetic energy 0.5*sum|v|^2 (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(10) << 0.5 * global_e << endl;
            cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << max_div << endl;
        }
    }

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Timing Summary" << endl;
        cout << "------------------------------------------------------------" << endl;
        cout << "  Total steps:     " << nsteps_rep << endl;
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s" << endl;
        cout << "  Avg per step:    " << t_wall_total / (nsteps_rep > 0 ? nsteps_rep : 1) << " s" << endl;
        nsmode::print_phase_summary(phase_max, t_wall_total, nsteps_rep, mode);
        if (ext_out) {
            const int fft_per_step = (mode == nsmode::MODE_BENCH) ? 36 : 48;
            printf("  Scan (max/E/eps + Allreduce, stage 1): %.6f s total, %.3e s/step (not in the phases above)\n",
                   scan_max, scan_max / (nsteps_rep > 0 ? nsteps_rep : 1));
            printf("DTCFL_SUMMARY dt_mode=%s cfl=%.4f steps=%ld rhs_calls=%ld ffts=%ld t_end=%.10e "
                   "dt_min=%.10e dt_maxused=%.10e scan_time=%.6f wall=%.6f "
                   "lim_conv=%ld lim_visc=%ld lim_dtmax=%ld lim_tend=%ld blowup=%d cap_hit=%d\n",
                   ctl.cfl ? "cfl" : "fixed", ctl.C, steps_done, 4 * steps_done, (long)fft_per_step * steps_done,
                   ctl.cfl ? ctl.t : steps_done * tau,
                   ctl.cfl ? (dt_min < HUGE_VAL ? dt_min : dt_big) : tau, ctl.cfl ? dt_big : tau,
                   scan_max, t_wall_total,
                   lim_count[LIM_CONV], lim_count[LIM_VISC], lim_count[LIM_DTMAX], lim_count[LIM_TEND],
                   blowup ? 1 : 0, cap_hit ? 1 : 0);
            fflush(stdout);
        }
        cout << "============================================================" << endl;
    }

    // ------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------
    g_fft.destroy();

    for (int c = 0; c < 3; ++c) {
        fftw_free(F.V[c]);  fftw_free(F.rot[c]);  fftw_free(F.rhs[c]);
        fftw_free(F.orig[c]); fftw_free(F.acc[c]);
        fftw_free(F.Vr[c]); fftw_free(F.rotr[c]);
    }

    MPI_Finalize();
    return (blowup || cap_hit) ? 2 : 0;   // 默认运行不会出现这两种情况，退出码仍为 0
}
