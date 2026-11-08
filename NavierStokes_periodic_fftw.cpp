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

// 全局FFTW计划（仅 2 个，通过 fftw_mpi_execute_dft_r2c/c2r 新执行 API 复用）
// FFTW plan 描述"变换几何"（网格大小、类型、布局），与具体数组无关；
// 只要所有数组均由 fftw_alloc_real/complex 分配（对齐保证），即可在执行时换指针。
fftw_plan plan_r2c;  // 所有正变换（R2C）共用
fftw_plan plan_c2r;  // 所有逆变换（C2R）共用

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

double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x,y,z,t) - func_laplace_V1(x,y,z,t) - func_v_cross_rot1(x,y,z,t) + func_grad_p1(x,y,z,t);
}
double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t) - func_laplace_V2(x,y,z,t) - func_v_cross_rot2(x,y,z,t) + func_grad_p2(x,y,z,t);
}
double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t) - func_laplace_V3(x,y,z,t) - func_v_cross_rot3(x,y,z,t) + func_grad_p3(x,y,z,t);
}

// ==============================================================================
// FFT 初始化和辅助函数
// ==============================================================================

/**
 * 初始化 FFTW MPI 3D 计划（仅创建 2 个）
 *
 * plan_r2c / plan_c2r 描述"变换几何"（nx×ny×nz，R2C/C2R，MPI 分解），
 * 与具体数组无关。执行时通过 fftw_mpi_execute_dft_r2c / _c2r 传入实际指针，
 * 前提：所有数组均由 fftw_alloc_real / fftw_alloc_complex 分配（对齐一致）。
 *
 * 对比旧方案：15 个 plan × FFTW_MEASURE 预计算 → 2 个，初始化开销显著降低。
 *
 * any_r : 任意一个已分配的实数组（大小 2*alloc_local），用于 plan 创建
 * any_c : 任意一个已分配的复数组（大小 alloc_local），用于 plan 创建
 */
void initialize_fftw_3d(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                        double *any_r, fftw_complex *any_c) {
    // 谱空间保持转置布局 [y][x][z]（y 分布），省去正变换末尾和逆变换开头的全局转置
    plan_r2c = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, any_r, any_c,
                                         MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
    plan_c2r = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, any_c, any_r,
                                         MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
}

void finalize_fft_plans() {
    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
}

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

/**
 * RK4 的一个阶段：在当前阶段值 V 处计算 RHS，并把 V 推进到下一阶段值
 * （阶段 4 之后为 V^{n+1}）。每阶段 9 次 FFT（verify 另加 3 次外力 FFT）。
 */
void rk4_stage(Fields& F, const Grid& g, double tau, double t, int stage) {
    const ptrdiff_t nx = g.nx, ny = g.ny, nz = g.nz, nz_c = nz/2 + 1;
    const double norm = (double)(nx * ny * nz);
    const ptrdiff_t rstride = 2*(nz/2+1);   // R2C 实空间 padding 后的行长

    // 1. 旋度 + 粘性项（谱空间）
    compute_rot_and_viscous(F, g);
    g_timer.mark(PH_SPEC);

    // 2. 逆变换：V、rot → 实空间（之后 V 与 rot 的谱空间内容都不再使用）
    for (int c = 0; c < 3; ++c) fftw_mpi_execute_dft_c2r(plan_c2r, F.V[c],   F.Vr[c]);
    for (int c = 0; c < 3; ++c) fftw_mpi_execute_dft_c2r(plan_c2r, F.rot[c], F.rotr[c]);
    g_timer.mark(PH_FFT);

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
    for (int c = 0; c < 3; ++c) fftw_mpi_execute_dft_r2c(plan_r2c, F.rotr[c], F.rot[c]);
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
        for (int c = 0; c < 3; ++c) fftw_mpi_execute_dft_r2c(plan_r2c, F.Vr[c], F.V[c]);
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

void rk4_step(Fields& F, const Grid& g, ptrdiff_t alloc_local, double tau, double t) {
    // 保存 V^n
    for (int c = 0; c < 3; ++c) memcpy(F.orig[c], F.V[c], alloc_local * sizeof(fftw_complex));
    g_timer.mark(PH_SPEC);

    rk4_stage(F, g, tau, t,           1);
    rk4_stage(F, g, tau, t + 0.5*tau, 2);
    rk4_stage(F, g, tau, t + 0.5*tau, 3);
    rk4_stage(F, g, tau, t + tau,     4);
}

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

    // FFTW 多线程：须在 fftw_mpi_init 之前 init_threads，在建计划之前设定线程数。
    // 必须链接 libfftw3_omp（而非 libfftw3_threads）：_threads 版用 pthread 创建工作线程，
    // 工作线程继承主线程的 CPU 亲和性；设置 OMP_PROC_BIND/OMP_PLACES 时 libgomp 在启动时
    // 就把主线程绑到第一个 place，所有 FFTW 工作线程挤在同一个 place 上，FFT 退化为串行。
    // _omp 版直接使用 OpenMP 线程组，与逐点循环的线程绑定方式相同。
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);
    fftw_mpi_init();

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Navier-Stokes Solver - FFTW MPI Version (1D Slab)" << endl;
        cout << "============================================================" << endl;
        cout << "Grid: " << nx << " x " << ny << " x " << nz
             << ", dt=" << scientific << tau << ", steps=" << nt_run << endl;
        cout << "Domain: [0, 2π]³" << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads/process: " << max_threads << endl;
        {
            cpu_set_t cs; CPU_ZERO(&cs);
            int ncpu = (sched_getaffinity(0, sizeof(cs), &cs) == 0) ? CPU_COUNT(&cs) : -1;
            cout << "FFTW planner threads: " << fftw_planner_nthreads()
                 << "  (main-thread CPU affinity: " << ncpu << " CPUs)" << endl;
        }
        cout << "Run mode: " << nsmode::mode_name(mode)
             << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << endl;
        cout << "============================================================" << endl;
    }

    // 获取本地分配大小（FFTW MPI 1D slab 分解）
    // 实空间：x 分布 [x_local][y][z]；谱空间（转置输出）：y 分布 [y_local][x][z]
    ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;
    alloc_local = fftw_mpi_local_size_3d_transposed(nx, ny, nz/2+1, MPI_COMM_WORLD,
                                                    &local_n0, &local_0_start,
                                                    &local_n1, &local_1_start);

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
    if (rank == 0) cout << "Initializing FFTW plans..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_plan_start = MPI_Wtime();

    // 只需任意一对合法缓冲区创建 plan；执行时通过 fftw_mpi_execute_* 换指针
    initialize_fftw_3d(nx, ny, nz, V1_r, V1_c);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "FFTW planning time: " << MPI_Wtime() - t_plan_start << " s" << endl;

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
                V1_r[idx] = func_V1(x, y, z, 0.0);
                V2_r[idx] = func_V2(x, y, z, 0.0);
                V3_r[idx] = func_V3(x, y, z, 0.0);
            }
        }
    }

    // 正变换 → 归一化（V_c = DFT(V_r)/N）
    fftw_mpi_execute_dft_r2c(plan_r2c, V1_r, V1_c);
    fftw_mpi_execute_dft_r2c(plan_r2c, V2_r, V2_c);
    fftw_mpi_execute_dft_r2c(plan_r2c, V3_r, V3_c);

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

    for(ptrdiff_t it = 0; it < nt_run; ++it) {
        double t_cur = it * tau;
        {
            double t0 = MPI_Wtime();
            g_timer.start();
            rk4_step(F, grid, alloc_local, tau, t_cur);
            double step_time = MPI_Wtime() - t0;
            double global_step;
            MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            t_wall_total += global_step;
        }
    }

    double phase_max[nsmode::PH_COUNT];
    nsmode::reduce_phases_max(g_timer.acc, phase_max);

    // 最终诊断（不计入计时）：verify → L2 误差；bench → 总动能；两者都输出 max|div V|
    {
        double t_final = nt_run * tau;
        // 散度须在 c2r 之前计算（多维 c2r 可能破坏输入 V_c）
        double max_div = compute_div_max(V1_c, V2_c, V3_c, nx, ny, nz, local_n1, local_1_start);
        fftw_mpi_execute_dft_c2r(plan_c2r, V1_c, V1_r);
        fftw_mpi_execute_dft_c2r(plan_c2r, V2_c, V2_r);
        fftw_mpi_execute_dft_c2r(plan_c2r, V3_c, V3_r);
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
        cout << "  Total steps:     " << nt_run << endl;
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s" << endl;
        cout << "  Avg per step:    " << t_wall_total / nt_run << " s" << endl;
        nsmode::print_phase_summary(phase_max, t_wall_total, nt_run, mode);
        cout << "============================================================" << endl;
    }

    // ------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------
    finalize_fft_plans();

    for (int c = 0; c < 3; ++c) {
        fftw_free(F.V[c]);  fftw_free(F.rot[c]);  fftw_free(F.rhs[c]);
        fftw_free(F.orig[c]); fftw_free(F.acc[c]);
        fftw_free(F.Vr[c]); fftw_free(F.rotr[c]);
    }

    MPI_Finalize();
    return 0;
}
