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

using namespace std;

// ==============================================================================
// 全周期性边界条件的Navier-Stokes求解器
// 使用FFTW MPI进行3D R2C/C2R变换
// ==============================================================================

// 全局FFTW计划
// 速度场（plan 绑定到 V_r <-> V_c）
fftw_plan plan_fwd_v1, plan_fwd_v2, plan_fwd_v3;  // V_r -> V_c
fftw_plan plan_bwd_v1, plan_bwd_v2, plan_bwd_v3;  // V_c -> V_r

// 旋度场（plan 绑定到 rot_c -> rot_r）
fftw_plan plan_bwd_rot1, plan_bwd_rot2, plan_bwd_rot3;  // rot_c -> rot_r

// 非线性项（plan 绑定到 rot_r -> nl_c，叉乘后覆盖 rot_r）
fftw_plan plan_fwd_nl1, plan_fwd_nl2, plan_fwd_nl3;   // rot_r -> nl_c

// 外力项（plan 绑定到 work_r -> f_c）
fftw_plan plan_fwd_f1, plan_fwd_f2, plan_fwd_f3;      // work_r -> f_c

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
 * 初始化所有 FFTW MPI 3D 计划
 *
 * 归一化约定（与 heFFTe 版本对齐）：
 *   V_c = DFT(V_r) / N （归一化频谱系数）
 *   forward 后手动 /N；backward 后直接得到物理值（N * IDFT(V_c_norm) = V_phys）
 *
 * 为消除热循环中的动态内存分配，所有需要专用 plan 的 (src, dst) 对均在此创建：
 *   plan_bwd_v{1,2,3}  : V_c -> V_r    （速度逆变换，已存在）
 *   plan_fwd_v{1,2,3}  : V_r -> V_c    （速度正变换，已存在）
 *   plan_bwd_rot{1,2,3}: rot_c -> rot_r （旋度逆变换，新增）
 *   plan_fwd_nl{1,2,3} : rot_r -> nl_c  （叉乘结果正变换，新增；叉乘后 rot_r 被覆盖）
 *   plan_fwd_f{1,2,3}  : work_r -> f_c  （外力正变换，新增）
 */
void initialize_fftw_3d(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                        double *V1_r,    double *V2_r,    double *V3_r,
                        fftw_complex *V1_c,   fftw_complex *V2_c,   fftw_complex *V3_c,
                        double *rot1_r,  double *rot2_r,  double *rot3_r,
                        fftw_complex *rot1_c, fftw_complex *rot2_c, fftw_complex *rot3_c,
                        fftw_complex *nl1_c,  fftw_complex *nl2_c,  fftw_complex *nl3_c,
                        double *work_r1, double *work_r2, double *work_r3,
                        fftw_complex *f1_c,   fftw_complex *f2_c,   fftw_complex *f3_c) {

    // 速度正/逆变换（已有）
    plan_fwd_v1 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V1_r, V1_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_v2 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V2_r, V2_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_v3 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V3_r, V3_c, MPI_COMM_WORLD, FFTW_ESTIMATE);

    plan_bwd_v1 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V1_c, V1_r, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v2 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V2_c, V2_r, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v3 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V3_c, V3_r, MPI_COMM_WORLD, FFTW_ESTIMATE);

    // 旋度逆变换（新增）：rot_c -> rot_r
    plan_bwd_rot1 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, rot1_c, rot1_r, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_rot2 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, rot2_c, rot2_r, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_rot3 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, rot3_c, rot3_r, MPI_COMM_WORLD, FFTW_ESTIMATE);

    // 非线性项正变换（新增）：rot_r -> nl_c（叉乘后 rot_r 保存叉乘结果）
    plan_fwd_nl1 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, rot1_r, nl1_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_nl2 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, rot2_r, nl2_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_nl3 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, rot3_r, nl3_c, MPI_COMM_WORLD, FFTW_ESTIMATE);

    // 外力正变换（新增）：work_r -> f_c
    plan_fwd_f1 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, work_r1, f1_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_f2 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, work_r2, f2_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_f3 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, work_r3, f3_c, MPI_COMM_WORLD, FFTW_ESTIMATE);
}

void finalize_fft_plans() {
    fftw_destroy_plan(plan_fwd_v1);   fftw_destroy_plan(plan_fwd_v2);   fftw_destroy_plan(plan_fwd_v3);
    fftw_destroy_plan(plan_bwd_v1);   fftw_destroy_plan(plan_bwd_v2);   fftw_destroy_plan(plan_bwd_v3);
    fftw_destroy_plan(plan_bwd_rot1); fftw_destroy_plan(plan_bwd_rot2); fftw_destroy_plan(plan_bwd_rot3);
    fftw_destroy_plan(plan_fwd_nl1);  fftw_destroy_plan(plan_fwd_nl2);  fftw_destroy_plan(plan_fwd_nl3);
    fftw_destroy_plan(plan_fwd_f1);   fftw_destroy_plan(plan_fwd_f2);   fftw_destroy_plan(plan_fwd_f3);
}

// ==============================================================================
// 频谱空间操作
// ==============================================================================

/**
 * 计算旋度 rot(V) = ik × V（频谱空间）
 * rot_1 = i(ky*V3 - kz*V2)
 * rot_2 = i(kz*V1 - kx*V3)
 * rot_3 = i(kx*V2 - ky*V1)
 */
void compute_rot(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                 fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                 ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                 ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_c = nz/2 + 1;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (i * ny + j) * nz_c + k;
                ptrdiff_t ig = local_0_start + i;
                double kx = (ig <= nx/2) ? (double)ig : (double)(ig - nx);
                double ky = (j  <= ny/2) ? (double)j  : (double)(j  - ny);
                double kz = (double)k;

                rot1_c[idx][0] = -(ky*V3_c[idx][1] - kz*V2_c[idx][1]);
                rot1_c[idx][1] =   ky*V3_c[idx][0] - kz*V2_c[idx][0];

                rot2_c[idx][0] = -(kz*V1_c[idx][1] - kx*V3_c[idx][1]);
                rot2_c[idx][1] =   kz*V1_c[idx][0] - kx*V3_c[idx][0];

                rot3_c[idx][0] = -(kx*V2_c[idx][1] - ky*V1_c[idx][1]);
                rot3_c[idx][1] =   kx*V2_c[idx][0] - ky*V1_c[idx][0];
            }
        }
    }
}

/**
 * 投影到无散空间：V -= ∇(∇⁻²·∇·V)（频谱空间，in-place）
 * div = ik·V，phi = div/(-k²)，V -= ik*phi
 */
void make_div_free(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                   fftw_complex* div_c, fftw_complex* phi_c,
                   ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                   ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_c = nz/2 + 1;

    // 步骤 1：计算散度 div = ik·V
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (i * ny + j) * nz_c + k;
                ptrdiff_t ig = local_0_start + i;
                double kx = (ig <= nx/2) ? (double)ig : (double)(ig - nx);
                double ky = (j  <= ny/2) ? (double)j  : (double)(j  - ny);
                double kz = (double)k;

                div_c[idx][0] = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                div_c[idx][1] =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
            }
        }
    }

    // 步骤 2：求解泊松方程 phi = div/(-k²)，然后 V -= ik*phi
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (i * ny + j) * nz_c + k;
                ptrdiff_t ig = local_0_start + i;
                double kx = (ig <= nx/2) ? (double)ig : (double)(ig - nx);
                double ky = (j  <= ny/2) ? (double)j  : (double)(j  - ny);
                double kz = (double)k;
                double k2 = kx*kx + ky*ky + kz*kz;

                if (k2 > 1e-10) {
                    phi_c[idx][0] = div_c[idx][0] / (-k2);
                    phi_c[idx][1] = div_c[idx][1] / (-k2);
                } else {
                    phi_c[idx][0] = 0.0;
                    phi_c[idx][1] = 0.0;
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

/**
 * 计算粘性项：visc = -k² * V（频谱空间）
 */
void compute_viscous_term(fftw_complex* V_c, fftw_complex* viscous_c,
                          ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                          ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_c = nz/2 + 1;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_c; ++k) {
                ptrdiff_t idx = (i * ny + j) * nz_c + k;
                ptrdiff_t ig = local_0_start + i;
                double kx = (ig <= nx/2) ? (double)ig : (double)(ig - nx);
                double ky = (j  <= ny/2) ? (double)j  : (double)(j  - ny);
                double kz = (double)k;
                double k2 = kx*kx + ky*ky + kz*kz;
                viscous_c[idx][0] = -k2 * V_c[idx][0];
                viscous_c[idx][1] = -k2 * V_c[idx][1];
            }
        }
    }
}

// ==============================================================================
// 非线性对流项：v × rot(v)（伪谱方法）
//
// 算法（与 heFFTe/p3dfft 版本完全相同）：
//   1. 频谱空间计算旋度 rot_c = ik × V_c                 （无 FFT）
//   2. 逆变换 V_c → V_r（plan_bwd_v）                    （3 FFTs）
//   3. 逆变换 rot_c → rot_r（plan_bwd_rot）               （3 FFTs）
//   4. 实空间叉乘，结果覆盖 rot_r                         （无 FFT）
//   5. 正变换 rot_r → nl_c（plan_fwd_nl），归一化          （3 FFTs）
// 合计：9 次 FFT，零动态内存分配
//
// 注意：V_c 在此函数中只读（plan_bwd_v 为 out-of-place，不破坏 V_c）
// ==============================================================================
void compute_nonlinear_term(fftw_complex* V1_c,  fftw_complex* V2_c,  fftw_complex* V3_c,
                            fftw_complex* nl1_c,  fftw_complex* nl2_c,  fftw_complex* nl3_c,
                            double* V1_r,   double* V2_r,   double* V3_r,
                            double* rot1_r, double* rot2_r, double* rot3_r,
                            fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                            ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                            ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_c = nz/2 + 1;
    ptrdiff_t total_c = local_n0 * ny * nz_c;
    double norm = (double)(nx * ny * nz);

    // 1. 频谱空间计算旋度（不涉及 FFT）
    compute_rot(V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c,
                nx, ny, nz, local_n0, local_0_start);

    // 2. 速度逆变换：V_c → V_r（plan_bwd_v 为 out-of-place，V_c 不被修改）
    fftw_execute(plan_bwd_v1);
    fftw_execute(plan_bwd_v2);
    fftw_execute(plan_bwd_v3);

    // 3. 旋度逆变换：rot_c → rot_r（专用 plan，无需借用 V_c 作中转）
    fftw_execute(plan_bwd_rot1);
    fftw_execute(plan_bwd_rot2);
    fftw_execute(plan_bwd_rot3);

    // 4. 实空间叉乘 v × rot(v)，结果覆盖 rot_r（后续用于正变换）
    //    FFTW R2C 实空间数组 padding：行步长 = 2*(nz/2+1)
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
                double w1 = rot1_r[idx], w2 = rot2_r[idx], w3 = rot3_r[idx];
                rot1_r[idx] = v2*w3 - v3*w2;  // (v×rot)_x
                rot2_r[idx] = v3*w1 - v1*w3;  // (v×rot)_y
                rot3_r[idx] = v1*w2 - v2*w1;  // (v×rot)_z
            }
        }
    }

    // 5. 正变换：rot_r（叉乘结果）→ nl_c（专用 plan，不触碰 V_c）
    fftw_execute(plan_fwd_nl1);
    fftw_execute(plan_fwd_nl2);
    fftw_execute(plan_fwd_nl3);

    // 归一化（FFTW 正变换不归一化，除以 N 得到归一化频谱系数）
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        nl1_c[i][0] /= norm;  nl1_c[i][1] /= norm;
        nl2_c[i][0] /= norm;  nl2_c[i][1] /= norm;
        nl3_c[i][0] /= norm;  nl3_c[i][1] /= norm;
    }
}

// ==============================================================================
// 完整右端项：RHS = v×rot(v) + P∆v + f，投影到无散空间
//
// 与 heFFTe/p3dfft 版本结构相同，零动态内存分配
// 所有工作数组均从外部传入（在 main() 中预分配）
// ==============================================================================
void compute_rhs(fftw_complex* V1_c,  fftw_complex* V2_c,  fftw_complex* V3_c,
                 fftw_complex* rhs1_c, fftw_complex* rhs2_c, fftw_complex* rhs3_c,
                 double* V1_r,   double* V2_r,   double* V3_r,
                 double* work_r1, double* work_r2, double* work_r3,
                 double* rot1_r,  double* rot2_r,  double* rot3_r,
                 fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                 fftw_complex* nl1_c,  fftw_complex* nl2_c,  fftw_complex* nl3_c,
                 fftw_complex* visc1_c, fftw_complex* visc2_c, fftw_complex* visc3_c,
                 fftw_complex* f1_c,   fftw_complex* f2_c,   fftw_complex* f3_c,
                 fftw_complex* div_c,  fftw_complex* phi_c,
                 ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                 ptrdiff_t local_n0, ptrdiff_t local_0_start,
                 double t, double dx, double dy, double dz) {

    ptrdiff_t nz_c = nz/2 + 1;
    ptrdiff_t total_c = local_n0 * ny * nz_c;
    double norm = (double)(nx * ny * nz);

    // 1. 非线性项 v×rot(v)（伪谱，plan_bwd_v 不破坏 V_c，plan_bwd_rot 和 plan_fwd_nl 专用）
    compute_nonlinear_term(V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                           V1_r, V2_r, V3_r, rot1_r, rot2_r, rot3_r,
                           rot1_c, rot2_c, rot3_c,
                           nx, ny, nz, local_n0, local_0_start);

    // 2. 粘性项 P∆V = -k² V（频谱空间，V_c 在整个 compute_rhs 中只读）
    compute_viscous_term(V1_c, visc1_c, nx, ny, nz, local_n0, local_0_start);
    compute_viscous_term(V2_c, visc2_c, nx, ny, nz, local_n0, local_0_start);
    compute_viscous_term(V3_c, visc3_c, nx, ny, nz, local_n0, local_0_start);

    // 3. 外力项 f（实空间填充 → 专用 plan_fwd_f 直接写到 f_c，不触碰 V_c）
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t ig  = local_0_start + i;
                double x = ig * dx, y = j * dy, z = k * dz;
                ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                work_r1[idx] = func_f1(x, y, z, t);
                work_r2[idx] = func_f2(x, y, z, t);
                work_r3[idx] = func_f3(x, y, z, t);
            }
        }
    }
    fftw_execute(plan_fwd_f1);  // work_r1 -> f1_c
    fftw_execute(plan_fwd_f2);
    fftw_execute(plan_fwd_f3);
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        f1_c[i][0] /= norm;  f1_c[i][1] /= norm;
        f2_c[i][0] /= norm;  f2_c[i][1] /= norm;
        f3_c[i][0] /= norm;  f3_c[i][1] /= norm;
    }

    // 4. 组合：RHS = nl + visc + f
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        rhs1_c[i][0] = nl1_c[i][0] + visc1_c[i][0] + f1_c[i][0];
        rhs1_c[i][1] = nl1_c[i][1] + visc1_c[i][1] + f1_c[i][1];
        rhs2_c[i][0] = nl2_c[i][0] + visc2_c[i][0] + f2_c[i][0];
        rhs2_c[i][1] = nl2_c[i][1] + visc2_c[i][1] + f2_c[i][1];
        rhs3_c[i][0] = nl3_c[i][0] + visc3_c[i][0] + f3_c[i][0];
        rhs3_c[i][1] = nl3_c[i][1] + visc3_c[i][1] + f3_c[i][1];
    }

    // 5. 投影：使 RHS 无散（div_c/phi_c 作为工作数组，在 main() 中预分配）
    make_div_free(rhs1_c, rhs2_c, rhs3_c, div_c, phi_c,
                  nx, ny, nz, local_n0, local_0_start);
}

// ==============================================================================
// RK4 时间积分（频谱空间）
//
// 与 heFFTe/p3dfft 版本结构相同，零动态内存分配
// V_c_orig 在 main() 中预分配，用于保存 V^n
// ==============================================================================
void rk4_step(fftw_complex* V1_c,  fftw_complex* V2_c,  fftw_complex* V3_c,
              double* V1_r,   double* V2_r,   double* V3_r,
              double* work_r1, double* work_r2, double* work_r3,
              double* rot1_r,  double* rot2_r,  double* rot3_r,
              fftw_complex* k1_v1, fftw_complex* k1_v2, fftw_complex* k1_v3,
              fftw_complex* k2_v1, fftw_complex* k2_v2, fftw_complex* k2_v3,
              fftw_complex* k3_v1, fftw_complex* k3_v2, fftw_complex* k3_v3,
              fftw_complex* k4_v1, fftw_complex* k4_v2, fftw_complex* k4_v3,
              fftw_complex* tmp_v1, fftw_complex* tmp_v2, fftw_complex* tmp_v3,
              fftw_complex* V1_c_orig, fftw_complex* V2_c_orig, fftw_complex* V3_c_orig,
              fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
              fftw_complex* nl1_c,  fftw_complex* nl2_c,  fftw_complex* nl3_c,
              fftw_complex* visc1_c, fftw_complex* visc2_c, fftw_complex* visc3_c,
              fftw_complex* f1_c,   fftw_complex* f2_c,   fftw_complex* f3_c,
              fftw_complex* div_c,  fftw_complex* phi_c,
              ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
              ptrdiff_t local_n0, ptrdiff_t local_0_start,
              ptrdiff_t alloc_local, double tau, double t,
              double dx, double dy, double dz) {

    ptrdiff_t nz_c = nz/2 + 1;
    ptrdiff_t total_c = local_n0 * ny * nz_c;

    // 保存 V^n（一次 memcpy，无动态分配）
    memcpy(V1_c_orig, V1_c, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c_orig, V2_c, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c_orig, V3_c, alloc_local * sizeof(fftw_complex));

    // k1 = RHS(V^n, t)
    compute_rhs(V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_r, rot2_r, rot3_r,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                div_c, phi_c,
                nx, ny, nz, local_n0, local_0_start, t, dx, dy, dz);

    // tmp = V^n + τ/2 * k1，写入 V_c 供 k2 使用
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        V1_c[i][0] = V1_c_orig[i][0] + 0.5*tau*k1_v1[i][0];
        V1_c[i][1] = V1_c_orig[i][1] + 0.5*tau*k1_v1[i][1];
        V2_c[i][0] = V2_c_orig[i][0] + 0.5*tau*k1_v2[i][0];
        V2_c[i][1] = V2_c_orig[i][1] + 0.5*tau*k1_v2[i][1];
        V3_c[i][0] = V3_c_orig[i][0] + 0.5*tau*k1_v3[i][0];
        V3_c[i][1] = V3_c_orig[i][1] + 0.5*tau*k1_v3[i][1];
    }

    // k2 = RHS(V^n + τ/2*k1, t + τ/2)
    compute_rhs(V1_c, V2_c, V3_c, k2_v1, k2_v2, k2_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_r, rot2_r, rot3_r,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                div_c, phi_c,
                nx, ny, nz, local_n0, local_0_start, t+0.5*tau, dx, dy, dz);

    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        V1_c[i][0] = V1_c_orig[i][0] + 0.5*tau*k2_v1[i][0];
        V1_c[i][1] = V1_c_orig[i][1] + 0.5*tau*k2_v1[i][1];
        V2_c[i][0] = V2_c_orig[i][0] + 0.5*tau*k2_v2[i][0];
        V2_c[i][1] = V2_c_orig[i][1] + 0.5*tau*k2_v2[i][1];
        V3_c[i][0] = V3_c_orig[i][0] + 0.5*tau*k2_v3[i][0];
        V3_c[i][1] = V3_c_orig[i][1] + 0.5*tau*k2_v3[i][1];
    }

    // k3 = RHS(V^n + τ/2*k2, t + τ/2)
    compute_rhs(V1_c, V2_c, V3_c, k3_v1, k3_v2, k3_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_r, rot2_r, rot3_r,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                div_c, phi_c,
                nx, ny, nz, local_n0, local_0_start, t+0.5*tau, dx, dy, dz);

    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        V1_c[i][0] = V1_c_orig[i][0] + tau*k3_v1[i][0];
        V1_c[i][1] = V1_c_orig[i][1] + tau*k3_v1[i][1];
        V2_c[i][0] = V2_c_orig[i][0] + tau*k3_v2[i][0];
        V2_c[i][1] = V2_c_orig[i][1] + tau*k3_v2[i][1];
        V3_c[i][0] = V3_c_orig[i][0] + tau*k3_v3[i][0];
        V3_c[i][1] = V3_c_orig[i][1] + tau*k3_v3[i][1];
    }

    // k4 = RHS(V^n + τ*k3, t + τ)
    compute_rhs(V1_c, V2_c, V3_c, k4_v1, k4_v2, k4_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_r, rot2_r, rot3_r,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                div_c, phi_c,
                nx, ny, nz, local_n0, local_0_start, t+tau, dx, dy, dz);

    // V^{n+1} = V^n + τ/6 * (k1 + 2k2 + 2k3 + k4)
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        V1_c[i][0] = V1_c_orig[i][0] + (tau/6.0)*(k1_v1[i][0] + 2*k2_v1[i][0] + 2*k3_v1[i][0] + k4_v1[i][0]);
        V1_c[i][1] = V1_c_orig[i][1] + (tau/6.0)*(k1_v1[i][1] + 2*k2_v1[i][1] + 2*k3_v1[i][1] + k4_v1[i][1]);
        V2_c[i][0] = V2_c_orig[i][0] + (tau/6.0)*(k1_v2[i][0] + 2*k2_v2[i][0] + 2*k3_v2[i][0] + k4_v2[i][0]);
        V2_c[i][1] = V2_c_orig[i][1] + (tau/6.0)*(k1_v2[i][1] + 2*k2_v2[i][1] + 2*k3_v2[i][1] + k4_v2[i][1]);
        V3_c[i][0] = V3_c_orig[i][0] + (tau/6.0)*(k1_v3[i][0] + 2*k2_v3[i][0] + 2*k3_v3[i][0] + k4_v3[i][0]);
        V3_c[i][1] = V3_c_orig[i][1] + (tau/6.0)*(k1_v3[i][1] + 2*k2_v3[i][1] + 2*k3_v3[i][1] + k4_v3[i][1]);
    }
}

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char **argv) {
    const ptrdiff_t nx = 128, ny = 128, nz = 128;
    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;
    const double dx = L_x/nx, dy = L_y/ny, dz = L_z/nz;

    const ptrdiff_t nt_total = 20000;
    const ptrdiff_t nt_run   = 2000;
    const double T   = 1.0;
    const double tau = T / nt_total;  // dt = 5e-5

    // MPI + OpenMP + FFTW 初始化
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int max_threads = omp_get_max_threads();
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);
    fftw_mpi_init();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Navier-Stokes Solver - FFTW MPI Version (1D Slab)" << endl;
        cout << "============================================================" << endl;
        cout << "Grid: " << nx << " x " << ny << " x " << nz << endl;
        cout << "Domain: [0, 2π]³" << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads/process: " << max_threads << endl;
        cout << "Steps: " << nt_run << " / " << nt_total << ", dt = " << tau << endl;
        cout << "============================================================" << endl;
    }

    // 获取本地分配大小（FFTW MPI 1D slab 分解）
    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(nx, ny, nz/2+1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);

    // ------------------------------------------------------------------
    // 预分配所有内存（热循环中零动态分配）
    // ------------------------------------------------------------------

    // 速度场：实空间（含 R2C padding）和频谱空间
    double       *V1_r = fftw_alloc_real(2 * alloc_local);
    double       *V2_r = fftw_alloc_real(2 * alloc_local);
    double       *V3_r = fftw_alloc_real(2 * alloc_local);
    fftw_complex *V1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c = fftw_alloc_complex(alloc_local);

    // 旋度场（实空间 + 频谱空间）
    double       *rot1_r = fftw_alloc_real(2 * alloc_local);
    double       *rot2_r = fftw_alloc_real(2 * alloc_local);
    double       *rot3_r = fftw_alloc_real(2 * alloc_local);
    fftw_complex *rot1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *rot2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *rot3_c = fftw_alloc_complex(alloc_local);

    // 投影工作数组
    fftw_complex *div_c = fftw_alloc_complex(alloc_local);
    fftw_complex *phi_c = fftw_alloc_complex(alloc_local);

    // 非线性、粘性、外力频谱项
    fftw_complex *nl1_c   = fftw_alloc_complex(alloc_local);
    fftw_complex *nl2_c   = fftw_alloc_complex(alloc_local);
    fftw_complex *nl3_c   = fftw_alloc_complex(alloc_local);
    fftw_complex *visc1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *visc2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *visc3_c = fftw_alloc_complex(alloc_local);
    fftw_complex *f1_c    = fftw_alloc_complex(alloc_local);
    fftw_complex *f2_c    = fftw_alloc_complex(alloc_local);
    fftw_complex *f3_c    = fftw_alloc_complex(alloc_local);

    // 外力实空间工作数组（plan_fwd_f 绑定到这里）
    double *work_r1 = fftw_alloc_real(2 * alloc_local);
    double *work_r2 = fftw_alloc_real(2 * alloc_local);
    double *work_r3 = fftw_alloc_real(2 * alloc_local);

    // RK4 工作数组
    fftw_complex *k1_v1 = fftw_alloc_complex(alloc_local);
    fftw_complex *k1_v2 = fftw_alloc_complex(alloc_local);
    fftw_complex *k1_v3 = fftw_alloc_complex(alloc_local);
    fftw_complex *k2_v1 = fftw_alloc_complex(alloc_local);
    fftw_complex *k2_v2 = fftw_alloc_complex(alloc_local);
    fftw_complex *k2_v3 = fftw_alloc_complex(alloc_local);
    fftw_complex *k3_v1 = fftw_alloc_complex(alloc_local);
    fftw_complex *k3_v2 = fftw_alloc_complex(alloc_local);
    fftw_complex *k3_v3 = fftw_alloc_complex(alloc_local);
    fftw_complex *k4_v1 = fftw_alloc_complex(alloc_local);
    fftw_complex *k4_v2 = fftw_alloc_complex(alloc_local);
    fftw_complex *k4_v3 = fftw_alloc_complex(alloc_local);
    fftw_complex *tmp_v1 = fftw_alloc_complex(alloc_local);
    fftw_complex *tmp_v2 = fftw_alloc_complex(alloc_local);
    fftw_complex *tmp_v3 = fftw_alloc_complex(alloc_local);

    // V^n 备份（用于 RK4，避免热循环中动态分配）
    fftw_complex *V1_c_orig = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c_orig = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c_orig = fftw_alloc_complex(alloc_local);

    // ------------------------------------------------------------------
    // 初始化所有 FFTW 计划（所有数组已预分配，指针固定）
    // ------------------------------------------------------------------
    if (rank == 0) cout << "Initializing FFTW plans..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_plan_start = MPI_Wtime();

    initialize_fftw_3d(nx, ny, nz,
                       V1_r,   V2_r,   V3_r,   V1_c,   V2_c,   V3_c,
                       rot1_r, rot2_r, rot3_r, rot1_c, rot2_c, rot3_c,
                       nl1_c,  nl2_c,  nl3_c,
                       work_r1, work_r2, work_r3,
                       f1_c,   f2_c,   f3_c);

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
    fftw_execute(plan_fwd_v1);
    fftw_execute(plan_fwd_v2);
    fftw_execute(plan_fwd_v3);

    double norm_factor = (double)(nx * ny * nz);
    ptrdiff_t nz_c = nz/2+1;
    ptrdiff_t total_c = local_n0 * ny * nz_c;
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_c; ++i) {
        V1_c[i][0] /= norm_factor;  V1_c[i][1] /= norm_factor;
        V2_c[i][0] /= norm_factor;  V2_c[i][1] /= norm_factor;
        V3_c[i][0] /= norm_factor;  V3_c[i][1] /= norm_factor;
    }

    // 投影初始条件到无散空间
    if (rank == 0) cout << "Projecting initial condition to divergence-free space..." << endl;
    make_div_free(V1_c, V2_c, V3_c, div_c, phi_c, nx, ny, nz, local_n0, local_0_start);

    // ------------------------------------------------------------------
    // 打印初始误差
    // ------------------------------------------------------------------
    {
        fftw_execute(plan_bwd_v1);
        fftw_execute(plan_bwd_v2);
        fftw_execute(plan_bwd_v3);

        double local_err = 0.0, local_max = 0.0;
        #pragma omp parallel for collapse(3) reduction(+:local_err) reduction(max:local_max)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz; ++k) {
                    ptrdiff_t ig  = local_0_start + i;
                    double x = ig * dx, y = j * dy, z = k * dz;
                    ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                    double d1 = V1_r[idx] - func_V1(x, y, z, 0.0);
                    double d2 = V2_r[idx] - func_V2(x, y, z, 0.0);
                    double d3 = V3_r[idx] - func_V3(x, y, z, 0.0);
                    double e  = d1*d1 + d2*d2 + d3*d3;
                    local_err += e;
                    local_max  = max(local_max, sqrt(e));
                }
            }
        }
        double global_err = 0.0, global_max = 0.0;
        MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "\nInitial condition (t=0):" << endl;
            cout << "  L2 error:   " << scientific << sqrt(global_err * dx * dy * dz) << endl;
            cout << "  L∞ error:   " << global_max << endl;
        }

        // 恢复 V_c（backward 为 out-of-place，V_c 本身未被修改；但后续 rk4 需要 V_c 正确）
        // 这里重新做一次正变换保证 V_c 与 V_r 一致（V_r 此时是物理值）
        fftw_execute(plan_fwd_v1);
        fftw_execute(plan_fwd_v2);
        fftw_execute(plan_fwd_v3);
        #pragma omp parallel for
        for(ptrdiff_t i = 0; i < total_c; ++i) {
            V1_c[i][0] /= norm_factor;  V1_c[i][1] /= norm_factor;
            V2_c[i][0] /= norm_factor;  V2_c[i][1] /= norm_factor;
            V3_c[i][0] /= norm_factor;  V3_c[i][1] /= norm_factor;
        }

        // 计算 max|div V|
        double local_div = 0.0;
        #pragma omp parallel for collapse(3) reduction(max:local_div)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz_c; ++k) {
                    ptrdiff_t ig  = local_0_start + i;
                    ptrdiff_t idx = (i * ny + j) * nz_c + k;
                    double kx = (ig <= nx/2) ? (double)ig : (double)(ig - nx);
                    double ky = (j  <= ny/2) ? (double)j  : (double)(j  - ny);
                    double kz = (double)k;
                    double dr = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                    double di =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
                    local_div = max(local_div, sqrt(dr*dr + di*di));
                }
            }
        }
        double global_div = 0.0;
        MPI_Reduce(&local_div, &global_div, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            cout << "  max|div V|: " << global_div << "\n" << endl;
    }

    // ------------------------------------------------------------------
    // 时间推进（RK4）
    // ------------------------------------------------------------------
    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Time Integration (RK4)" << endl;
        cout << "============================================================" << endl;
        cout << setw(6) << "Step" << setw(12) << "Wall(s)"
             << setw(15) << "L2 Error" << setw(15) << "Max |div V|" << endl;
        cout << "------------------------------------------------------------------------" << endl;
    }

    double t_wall_total = 0.0;  // 累计挂钟时间（秒）

    for(ptrdiff_t it = 0; it <= nt_run; ++it) {
        double t_cur = it * tau;

        // --- 误差计算（不计入 wall time）---
        fftw_execute(plan_bwd_v1);
        fftw_execute(plan_bwd_v2);
        fftw_execute(plan_bwd_v3);

        double local_err = 0.0;
        #pragma omp parallel for collapse(3) reduction(+:local_err)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz; ++k) {
                    ptrdiff_t ig  = local_0_start + i;
                    double x = ig * dx, y = j * dy, z = k * dz;
                    ptrdiff_t idx = (i * ny + j) * (2*(nz/2+1)) + k;
                    double d1 = V1_r[idx] - func_V1(x, y, z, t_cur);
                    double d2 = V2_r[idx] - func_V2(x, y, z, t_cur);
                    double d3 = V3_r[idx] - func_V3(x, y, z, t_cur);
                    local_err += d1*d1 + d2*d2 + d3*d3;
                }
            }
        }
        double global_err = 0.0;
        MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // 恢复 V_c（从 V_r 重新正变换；V_r 此时包含当前步物理值）
        fftw_execute(plan_fwd_v1);
        fftw_execute(plan_fwd_v2);
        fftw_execute(plan_fwd_v3);
        #pragma omp parallel for
        for(ptrdiff_t i = 0; i < total_c; ++i) {
            V1_c[i][0] /= norm_factor;  V1_c[i][1] /= norm_factor;
            V2_c[i][0] /= norm_factor;  V2_c[i][1] /= norm_factor;
            V3_c[i][0] /= norm_factor;  V3_c[i][1] /= norm_factor;
        }

        // 计算 max|div V|
        double local_div = 0.0;
        #pragma omp parallel for collapse(3) reduction(max:local_div)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz_c; ++k) {
                    ptrdiff_t ig  = local_0_start + i;
                    ptrdiff_t idx = (i * ny + j) * nz_c + k;
                    double kx = (ig <= nx/2) ? (double)ig : (double)(ig - nx);
                    double ky = (j  <= ny/2) ? (double)j  : (double)(j  - ny);
                    double kz = (double)k;
                    double dr = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                    double di =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
                    local_div = max(local_div, sqrt(dr*dr + di*di));
                }
            }
        }
        double global_div = 0.0;
        MPI_Reduce(&local_div, &global_div, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double L2_err = sqrt(global_err * dx * dy * dz);
            cout << setw(6) << it
                 << setw(12) << fixed << setprecision(4) << t_wall_total
                 << setw(15) << scientific << setprecision(4) << L2_err
                 << setw(15) << global_div << endl;
        }

        // --- RK4 步进（计入 wall time）---
        if (it < nt_run) {
            double t0 = MPI_Wtime();
            rk4_step(V1_c, V2_c, V3_c,
                     V1_r, V2_r, V3_r,
                     work_r1, work_r2, work_r3,
                     rot1_r, rot2_r, rot3_r,
                     k1_v1, k1_v2, k1_v3,
                     k2_v1, k2_v2, k2_v3,
                     k3_v1, k3_v2, k3_v3,
                     k4_v1, k4_v2, k4_v3,
                     tmp_v1, tmp_v2, tmp_v3,
                     V1_c_orig, V2_c_orig, V3_c_orig,
                     rot1_c, rot2_c, rot3_c,
                     nl1_c, nl2_c, nl3_c,
                     visc1_c, visc2_c, visc3_c,
                     f1_c, f2_c, f3_c,
                     div_c, phi_c,
                     nx, ny, nz, local_n0, local_0_start,
                     alloc_local, tau, t_cur, dx, dy, dz);
            double step_time = MPI_Wtime() - t0;
            double global_step;
            MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            t_wall_total += global_step;
        }
    }

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Timing Summary" << endl;
        cout << "------------------------------------------------------------" << endl;
        cout << "  Total steps:     " << nt_run << endl;
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s" << endl;
        cout << "  Avg per step:    " << t_wall_total / nt_run << " s" << endl;
        cout << "============================================================" << endl;
    }

    // ------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------
    finalize_fft_plans();

    fftw_free(V1_r);     fftw_free(V2_r);     fftw_free(V3_r);
    fftw_free(V1_c);     fftw_free(V2_c);     fftw_free(V3_c);
    fftw_free(rot1_r);   fftw_free(rot2_r);   fftw_free(rot3_r);
    fftw_free(rot1_c);   fftw_free(rot2_c);   fftw_free(rot3_c);
    fftw_free(div_c);    fftw_free(phi_c);
    fftw_free(nl1_c);    fftw_free(nl2_c);    fftw_free(nl3_c);
    fftw_free(visc1_c);  fftw_free(visc2_c);  fftw_free(visc3_c);
    fftw_free(f1_c);     fftw_free(f2_c);     fftw_free(f3_c);
    fftw_free(work_r1);  fftw_free(work_r2);  fftw_free(work_r3);
    fftw_free(k1_v1);    fftw_free(k1_v2);    fftw_free(k1_v3);
    fftw_free(k2_v1);    fftw_free(k2_v2);    fftw_free(k2_v3);
    fftw_free(k3_v1);    fftw_free(k3_v2);    fftw_free(k3_v3);
    fftw_free(k4_v1);    fftw_free(k4_v2);    fftw_free(k4_v3);
    fftw_free(tmp_v1);   fftw_free(tmp_v2);   fftw_free(tmp_v3);
    fftw_free(V1_c_orig); fftw_free(V2_c_orig); fftw_free(V3_c_orig);

    MPI_Finalize();
    return 0;
}
