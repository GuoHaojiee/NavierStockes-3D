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
fftw_plan plan_fwd_v1, plan_fwd_v2, plan_fwd_v3;  // 3D R2C正向变换
fftw_plan plan_bwd_v1, plan_bwd_v2, plan_bwd_v3;  // 3D C2R反向变换

// ==============================================================================
// Taylor-Green涡解析解
// 这是全周期性边界条件下Navier-Stokes方程的精确解
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

// 投影方法：求解器求解 ∂v/∂t = P∆v + v×rot(v) + f_code（无-∇p项）
// 完整方程：∂v/∂t = P∆v + v×rot(v) - ∇p + f_image
// 压力项通过make_div_free()投影隐式处理
// 所以：f_code = f_image = ∂v/∂t - P∆v - v×rot(v) + ∇p（P=1）
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
// FFT初始化和辅助函数
// ==============================================================================

/**
 * 初始化FFTW MPI 3D变换
 * 为每个速度分量创建独立的计划
 */
void initialize_fftw_3d(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                        double *V1_r, double *V2_r, double *V3_r,
                        fftw_complex *V1_c, fftw_complex *V2_c, fftw_complex *V3_c) {

    // 为V1创建计划
    plan_fwd_v1 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V1_r, V1_c,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v1 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V1_c, V1_r,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);

    // 为V2创建计划
    plan_fwd_v2 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V2_r, V2_c,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v2 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V2_c, V2_r,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);

    // 为V3创建计划
    plan_fwd_v3 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V3_r, V3_c,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v3 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V3_c, V3_r,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
}

void finalize_fft_plans() {
    fftw_destroy_plan(plan_fwd_v1);
    fftw_destroy_plan(plan_fwd_v2);
    fftw_destroy_plan(plan_fwd_v3);
    fftw_destroy_plan(plan_bwd_v1);
    fftw_destroy_plan(plan_bwd_v2);
    fftw_destroy_plan(plan_bwd_v3);
}

/**
 * 归一化频谱数据
 */
void normalization(fftw_complex* data, ptrdiff_t size, double factor) {
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < size; ++i) {
        data[i][0] /= factor;
        data[i][1] /= factor;
    }
}

/**
 * 计算能量（实空间）
 */
double calculateEnergy(double* V1_r, double* V2_r, double* V3_r,
                       ptrdiff_t local_n0, ptrdiff_t ny, ptrdiff_t nz,
                       double dx, double dy, double dz) {
    double sumSquares = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:sumSquares)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;
                double v2 = V1_r[index]*V1_r[index] +
                           V2_r[index]*V2_r[index] +
                           V3_r[index]*V3_r[index];
                sumSquares += v2;
            }
        }
    }

    // 能量 = 0.5 * ∫ v² dV
    return 0.5 * sumSquares * dx * dy * dz;
}

// ==============================================================================
// 频谱空间操作和时间步进
// ==============================================================================

/**
 * 计算旋度 rot(V)
 * 在频谱空间中：rot(V) = ik × V_hat
 */
void compute_rot(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                 fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                 ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                 ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_complex = nz/2 + 1;

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;

                // 全局索引
                ptrdiff_t i_global = local_0_start + i;

                // 波数（考虑周期性）
                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;  // R2C变换，kz总是非负

                // rot = ik × V
                // rot_x = i(k_y*V_z - k_z*V_y)
                rot1_c[index][0] = -(ky*V3_c[index][1] - kz*V2_c[index][1]);
                rot1_c[index][1] =  (ky*V3_c[index][0] - kz*V2_c[index][0]);

                // rot_y = i(k_z*V_x - k_x*V_z)
                rot2_c[index][0] = -(kz*V1_c[index][1] - kx*V3_c[index][1]);
                rot2_c[index][1] =  (kz*V1_c[index][0] - kx*V3_c[index][0]);

                // rot_z = i(k_x*V_y - k_y*V_x)
                rot3_c[index][0] = -(kx*V2_c[index][1] - ky*V1_c[index][1]);
                rot3_c[index][1] =  (kx*V2_c[index][0] - ky*V1_c[index][0]);
            }
        }
    }
}

/**
 * 计算v×rot(v)（对流项）
 * 在频谱空间计算旋度，转到实空间计算叉乘，再转回频谱空间
 */
void compute_v_cross_rot(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                         fftw_complex* result1_c, fftw_complex* result2_c, fftw_complex* result3_c,
                         double* work_r1, double* work_r2, double* work_r3,
                         double* work_r4, double* work_r5, double* work_r6,
                         fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                         ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                         ptrdiff_t local_n0, ptrdiff_t local_0_start,
                         ptrdiff_t alloc_local) {

    // 步骤1: 计算旋度（频谱空间）
    compute_rot(V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c,
                nx, ny, nz, local_n0, local_0_start);

    // 步骤2: 速度和旋度都转到实空间
    fftw_execute(plan_bwd_v1);  // V1_c -> work_r1
    fftw_execute(plan_bwd_v2);  // V2_c -> work_r2
    fftw_execute(plan_bwd_v3);  // V3_c -> work_r3

    // 为rot创建临时变换（使用相同的plan结构）
    // 注意：这里我们直接执行，因为rot和V有相同的维度
    memcpy(work_r4, rot1_c, alloc_local * sizeof(fftw_complex));
    memcpy(work_r5, rot2_c, alloc_local * sizeof(fftw_complex));
    memcpy(work_r6, rot3_c, alloc_local * sizeof(fftw_complex));

    // 实际上需要为rot创建独立的逆变换
    // 简化：直接使用已有plan（假设rot也是复数数组）

    // 步骤3: 在实空间计算叉乘 v × rot(v)
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                // v × rot(v)
                // 注意：这里work_r4,5,6实际上还是复数，需要修正
                // 暂时简化处理
                work_r4[index] = work_r2[index] * 0 - work_r3[index] * 0;  // v2*rot3 - v3*rot2
                work_r5[index] = work_r3[index] * 0 - work_r1[index] * 0;  // v3*rot1 - v1*rot3
                work_r6[index] = work_r1[index] * 0 - work_r2[index] * 0;  // v1*rot2 - v2*rot1
            }
        }
    }

    // 步骤4: 转回频谱空间
    fftw_execute(plan_fwd_v1);  // work_r4 -> result1_c
    fftw_execute(plan_fwd_v2);  // work_r5 -> result2_c
    fftw_execute(plan_fwd_v3);  // work_r6 -> result3_c

    // 归一化
    double norm = nx * ny * nz;
    normalization(result1_c, alloc_local, norm);
    normalization(result2_c, alloc_local, norm);
    normalization(result3_c, alloc_local, norm);
}

/**
 * 计算散度 div(V)
 * 在频谱空间中：div(V) = ik · V_hat
 */
void compute_div(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                 fftw_complex* div_c,
                 ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                 ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_complex = nz/2 + 1;

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;

                ptrdiff_t i_global = local_0_start + i;
                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                // div = ik · V
                div_c[index][0] = -(kx*V1_c[index][1] + ky*V2_c[index][1] + kz*V3_c[index][1]);
                div_c[index][1] =  (kx*V1_c[index][0] + ky*V2_c[index][0] + kz*V3_c[index][0]);
            }
        }
    }
}

/**
 * 使速度场无散（投影方法）
 * V = V - ∇(∇⁻²·∇·V)
 */
void make_div_free(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                   fftw_complex* div_c, fftw_complex* phi_c,
                   ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                   ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    // 计算散度
    compute_div(V1_c, V2_c, V3_c, div_c, nx, ny, nz, local_n0, local_0_start);

    ptrdiff_t nz_complex = nz/2 + 1;

    // 求解泊松方程：∇²φ = ∇·V
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;

                ptrdiff_t i_global = local_0_start + i;
                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                double k2 = kx*kx + ky*ky + kz*kz;

                if (k2 > 1e-10) {
                    phi_c[index][0] = div_c[index][0] / (-k2);
                    phi_c[index][1] = div_c[index][1] / (-k2);
                } else {
                    phi_c[index][0] = 0.0;
                    phi_c[index][1] = 0.0;
                }
            }
        }
    }

    // V = V - ∇φ
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;

                ptrdiff_t i_global = local_0_start + i;
                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                // ∇φ = ikφ
                V1_c[index][0] -= -kx*phi_c[index][1];
                V1_c[index][1] -=  kx*phi_c[index][0];

                V2_c[index][0] -= -ky*phi_c[index][1];
                V2_c[index][1] -=  ky*phi_c[index][0];

                V3_c[index][0] -= -kz*phi_c[index][1];
                V3_c[index][1] -=  kz*phi_c[index][0];
            }
        }
    }
}

// ==============================================================================
// 时间步进
// ==============================================================================

/**
 * 计算粘性项（频谱空间）：ν∇²V = -νk²V
 */
void compute_viscous_term(fftw_complex* V_c, fftw_complex* viscous_c,
                          ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                          ptrdiff_t local_n0, ptrdiff_t local_0_start) {

    ptrdiff_t nz_complex = nz/2 + 1;

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;
                ptrdiff_t i_global = local_0_start + i;

                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;
                double k2 = kx*kx + ky*ky + kz*kz;

                // P∆V = -k²V (频谱空间，P=1)
                viscous_c[index][0] = -k2 * V_c[index][0];
                viscous_c[index][1] = -k2 * V_c[index][1];
            }
        }
    }
}

/**
 * 计算非线性对流项（伪谱方法）：-v×rot(v)
 * 步骤：1) 在频谱空间计算rot(v)
 *       2) 转到实空间计算 v×rot(v)
 *       3) FFT回频谱空间得到 -v×rot(v)
 */
void compute_nonlinear_term(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                            fftw_complex* nl1_c, fftw_complex* nl2_c, fftw_complex* nl3_c,
                            double* V1_r, double* V2_r, double* V3_r,
                            double* work_r1, double* work_r2, double* work_r3,
                            fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                            fftw_complex* tmp1_c, fftw_complex* tmp2_c, fftw_complex* tmp3_c,
                            ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                            ptrdiff_t local_n0, ptrdiff_t local_0_start,
                            ptrdiff_t alloc_local) {

    ptrdiff_t nz_complex = nz/2 + 1;
    ptrdiff_t total_size = local_n0 * ny * nz_complex;

    // 1. 在频谱空间计算旋度
    compute_rot(V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c,
                nx, ny, nz, local_n0, local_0_start);

    // 2. 速度转到实空间（V_c -> V_r）
    fftw_execute(plan_bwd_v1);
    fftw_execute(plan_bwd_v2);
    fftw_execute(plan_bwd_v3);

    // 3. 旋度转到实空间（重用FFT计划）
    // 将rot_c复制到tmp_c，然后使用plan_bwd转换到work_r
    memcpy(tmp1_c, rot1_c, alloc_local * sizeof(fftw_complex));
    memcpy(tmp2_c, rot2_c, alloc_local * sizeof(fftw_complex));
    memcpy(tmp3_c, rot3_c, alloc_local * sizeof(fftw_complex));

    // 临时使用V的plan来转换rot（将tmp_c中的数据转换到对应的V_r位置，然后复制）
    // 这需要小心处理：先保存V_c，然后用tmp_c替换，执行逆变换，再恢复
    fftw_complex *V1_c_backup = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c_backup = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c_backup = fftw_alloc_complex(alloc_local);

    memcpy(V1_c_backup, V1_c, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c_backup, V2_c, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c_backup, V3_c, alloc_local * sizeof(fftw_complex));

    // 用rot替换V_c，然后逆变换到work_r
    memcpy(V1_c, tmp1_c, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c, tmp2_c, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c, tmp3_c, alloc_local * sizeof(fftw_complex));

    // 注意：plan_bwd绑定到V1_c->work_r1等，所以需要临时调整
    // 实际上这样不行，因为plan绑定到特定内存地址
    // 简化方案：直接将V1_r等的结果复制到临时数组，然后再用工作数组
    double *rot1_r = work_r1;  // 重用work数组
    double *rot2_r = work_r2;
    double *rot3_r = work_r3;

    // 执行逆FFT（rot的频谱 -> 实空间）
    fftw_execute(plan_bwd_v1);  // tmp1_c（即rot1_c） -> V1_r
    memcpy(rot1_r, V1_r, 2 * alloc_local * sizeof(double));

    fftw_execute(plan_bwd_v2);  // tmp2_c -> V2_r
    memcpy(rot2_r, V2_r, 2 * alloc_local * sizeof(double));

    fftw_execute(plan_bwd_v3);  // tmp3_c -> V3_r
    memcpy(rot3_r, V3_r, 2 * alloc_local * sizeof(double));

    // 恢复V_c和V_r（重新计算V_r from V_c）
    memcpy(V1_c, V1_c_backup, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c, V2_c_backup, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c, V3_c_backup, alloc_local * sizeof(fftw_complex));

    fftw_execute(plan_bwd_v1);  // 重新计算V_r
    fftw_execute(plan_bwd_v2);
    fftw_execute(plan_bwd_v3);

    fftw_free(V1_c_backup);
    fftw_free(V2_c_backup);
    fftw_free(V3_c_backup);

    // 4. 在实空间计算 v×rot(v)，结果存入work数组（会被覆盖，所以用新的临时数组）
    double *cross_r1 = fftw_alloc_real(2 * alloc_local);
    double *cross_r2 = fftw_alloc_real(2 * alloc_local);
    double *cross_r3 = fftw_alloc_real(2 * alloc_local);

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                double v1 = V1_r[index];
                double v2 = V2_r[index];
                double v3 = V3_r[index];
                double w1 = rot1_r[index];
                double w2 = rot2_r[index];
                double w3 = rot3_r[index];

                // v × rot(v) 的三个分量
                cross_r1[index] = v2 * w3 - v3 * w2;  // (v×rot)_x
                cross_r2[index] = v3 * w1 - v1 * w3;  // (v×rot)_y
                cross_r3[index] = v1 * w2 - v2 * w1;  // (v×rot)_z
            }
        }
    }

    // 5. cross转回频谱空间（使用V_c作为临时，然后复制到nl_c）
    memcpy(V1_r, cross_r1, 2 * alloc_local * sizeof(double));
    fftw_execute(plan_fwd_v1);
    memcpy(nl1_c, V1_c, alloc_local * sizeof(fftw_complex));

    memcpy(V2_r, cross_r2, 2 * alloc_local * sizeof(double));
    fftw_execute(plan_fwd_v2);
    memcpy(nl2_c, V2_c, alloc_local * sizeof(fftw_complex));

    memcpy(V3_r, cross_r3, 2 * alloc_local * sizeof(double));
    fftw_execute(plan_fwd_v3);
    memcpy(nl3_c, V3_c, alloc_local * sizeof(fftw_complex));

    fftw_free(cross_r1);
    fftw_free(cross_r2);
    fftw_free(cross_r3);

    // 归一化得到 v×rot(v)（图片方程中是正号）
    double norm = nx * ny * nz;
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        nl1_c[i][0] = nl1_c[i][0] / norm;
        nl1_c[i][1] = nl1_c[i][1] / norm;
        nl2_c[i][0] = nl2_c[i][0] / norm;
        nl2_c[i][1] = nl2_c[i][1] / norm;
        nl3_c[i][0] = nl3_c[i][0] / norm;
        nl3_c[i][1] = nl3_c[i][1] / norm;
    }
}

/**
 * 计算完整右端项：dV/dt = v×rot(v) + ν∇²v + f
 * 使用伪谱方法（符合图片方程）
 */
void compute_rhs(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                 fftw_complex* rhs1_c, fftw_complex* rhs2_c, fftw_complex* rhs3_c,
                 double* V1_r, double* V2_r, double* V3_r,
                 double* work_r1, double* work_r2, double* work_r3,
                 fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
                 fftw_complex* nl1_c, fftw_complex* nl2_c, fftw_complex* nl3_c,
                 fftw_complex* visc1_c, fftw_complex* visc2_c, fftw_complex* visc3_c,
                 fftw_complex* f1_c, fftw_complex* f2_c, fftw_complex* f3_c,
                 ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                 ptrdiff_t local_n0, ptrdiff_t local_0_start,
                 ptrdiff_t alloc_local, double t,
                 double dx, double dy, double dz) {

    ptrdiff_t nz_complex = nz/2 + 1;
    ptrdiff_t total_size = local_n0 * ny * nz_complex;

    // 保存原始V_c（因为计算过程会使用V1_c等存储中间结果）
    fftw_complex *V1_c_save = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c_save = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c_save = fftw_alloc_complex(alloc_local);
    memcpy(V1_c_save, V1_c, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c_save, V2_c, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c_save, V3_c, alloc_local * sizeof(fftw_complex));

    // 分配临时数组用于compute_nonlinear_term
    fftw_complex *tmp1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *tmp2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *tmp3_c = fftw_alloc_complex(alloc_local);

    // 1. 计算非线性项 -v×rot(v)（使用V_c_save，因为compute_nonlinear_term会修改V_c）
    // 注意：V_c现在可能包含临时数据，我们需要从V_c_save恢复后再使用
    memcpy(V1_c, V1_c_save, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c, V2_c_save, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c, V3_c_save, alloc_local * sizeof(fftw_complex));

    // 重新计算V_r（用于非线性项计算）
    fftw_execute(plan_bwd_v1);
    fftw_execute(plan_bwd_v2);
    fftw_execute(plan_bwd_v3);

    compute_nonlinear_term(V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                          V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                          rot1_c, rot2_c, rot3_c, tmp1_c, tmp2_c, tmp3_c,
                          nx, ny, nz, local_n0, local_0_start, alloc_local);

    // 2. 计算粘性项 P∆v（使用保存的V_c，P=1）
    compute_viscous_term(V1_c_save, visc1_c, nx, ny, nz, local_n0, local_0_start);
    compute_viscous_term(V2_c_save, visc2_c, nx, ny, nz, local_n0, local_0_start);
    compute_viscous_term(V3_c_save, visc3_c, nx, ny, nz, local_n0, local_0_start);

    // 3. 计算外力项 f（实空间）
    // 使用work_r，不要破坏V_r！
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t i_global = local_0_start + i;
                double x = i_global * dx;
                double y = j * dy;
                double z = k * dz;
                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                work_r1[index] = func_f1(x, y, z, t);
                work_r2[index] = func_f2(x, y, z, t);
                work_r3[index] = func_f3(x, y, z, t);
            }
        }
    }

    // f转到频谱空间（需要复制work_r到V_r，因为plan绑定到V_r）
    memcpy(V1_r, work_r1, 2 * alloc_local * sizeof(double));
    memcpy(V2_r, work_r2, 2 * alloc_local * sizeof(double));
    memcpy(V3_r, work_r3, 2 * alloc_local * sizeof(double));

    fftw_execute(plan_fwd_v1);  // V1_r -> V1_c
    fftw_execute(plan_fwd_v2);  // V2_r -> V2_c
    fftw_execute(plan_fwd_v3);  // V3_r -> V3_c

    // 复制并归一化f
    double norm = nx * ny * nz;
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        f1_c[i][0] = V1_c[i][0] / norm;
        f1_c[i][1] = V1_c[i][1] / norm;
        f2_c[i][0] = V2_c[i][0] / norm;
        f2_c[i][1] = V2_c[i][1] / norm;
        f3_c[i][0] = V3_c[i][0] / norm;
        f3_c[i][1] = V3_c[i][1] / norm;
    }

    // 注意：我们不再恢复V_c和V_r，因为rk4_step会管理这些数组
    // V_c和V_r现在可能包含临时数据，但这是预期的

    // 4. 组合所有项：F = v×rot(v) + P∆v + f
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        rhs1_c[i][0] = nl1_c[i][0] + visc1_c[i][0] + f1_c[i][0];
        rhs1_c[i][1] = nl1_c[i][1] + visc1_c[i][1] + f1_c[i][1];
        rhs2_c[i][0] = nl2_c[i][0] + visc2_c[i][0] + f2_c[i][0];
        rhs2_c[i][1] = nl2_c[i][1] + visc2_c[i][1] + f2_c[i][1];
        rhs3_c[i][0] = nl3_c[i][0] + visc3_c[i][0] + f3_c[i][0];
        rhs3_c[i][1] = nl3_c[i][1] + visc3_c[i][1] + f3_c[i][1];
    }

    // 5. 投影到无散空间（图片中的方法）
    // 计算 div(F) = ik·F
    fftw_complex *div_F = fftw_alloc_complex(alloc_local);
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;
                ptrdiff_t i_global = local_0_start + i;

                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                // div(F) = i*kx*F1 + i*ky*F2 + i*kz*F3
                // i*kx*F = i*kx*(Fr + i*Fi) = i*kx*Fr - kx*Fi
                double div_real = -kx*rhs1_c[index][1] - ky*rhs2_c[index][1] - kz*rhs3_c[index][1];
                double div_imag =  kx*rhs1_c[index][0] + ky*rhs2_c[index][0] + kz*rhs3_c[index][0];

                div_F[index][0] = div_real;
                div_F[index][1] = div_imag;
            }
        }
    }

    // 求解泊松方程：∆p = div(F) → p = div(F) / (-k²)
    fftw_complex *p_c = fftw_alloc_complex(alloc_local);
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;
                ptrdiff_t i_global = local_0_start + i;

                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;
                double k2 = kx*kx + ky*ky + kz*kz;

                if (k2 > 1e-10) {
                    p_c[index][0] = div_F[index][0] / (-k2);
                    p_c[index][1] = div_F[index][1] / (-k2);
                } else {
                    p_c[index][0] = 0.0;
                    p_c[index][1] = 0.0;
                }
            }
        }
    }

    // 6. 从F中减去压力梯度：rhs = F - ∇p
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz_complex; ++k) {
                ptrdiff_t index = (i * ny + j) * nz_complex + k;
                ptrdiff_t i_global = local_0_start + i;

                double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                // ∇p = ik*p，所以从F中减去
                // ∂F/∂x = i*kx*p → F - ∂p/∂x
                rhs1_c[index][0] -= -kx*p_c[index][1];
                rhs1_c[index][1] -=  kx*p_c[index][0];

                rhs2_c[index][0] -= -ky*p_c[index][1];
                rhs2_c[index][1] -=  ky*p_c[index][0];

                rhs3_c[index][0] -= -kz*p_c[index][1];
                rhs3_c[index][1] -=  kz*p_c[index][0];
            }
        }
    }

    // 释放临时数组
    fftw_free(div_F);
    fftw_free(p_c);
    fftw_free(V1_c_save);
    fftw_free(V2_c_save);
    fftw_free(V3_c_save);
    fftw_free(tmp1_c);
    fftw_free(tmp2_c);
    fftw_free(tmp3_c);
}

/**
 * Runge-Kutta 4阶时间积分
 * 包含完整的NS右端项：-v×rot(v) + ν∇²v + f
 */
void rk4_step(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
              double* V1_r, double* V2_r, double* V3_r,
              double* work_r1, double* work_r2, double* work_r3,
              fftw_complex* k1_v1, fftw_complex* k1_v2, fftw_complex* k1_v3,
              fftw_complex* k2_v1, fftw_complex* k2_v2, fftw_complex* k2_v3,
              fftw_complex* k3_v1, fftw_complex* k3_v2, fftw_complex* k3_v3,
              fftw_complex* k4_v1, fftw_complex* k4_v2, fftw_complex* k4_v3,
              fftw_complex* tmp_v1, fftw_complex* tmp_v2, fftw_complex* tmp_v3,
              fftw_complex* rot1_c, fftw_complex* rot2_c, fftw_complex* rot3_c,
              fftw_complex* nl1_c, fftw_complex* nl2_c, fftw_complex* nl3_c,
              fftw_complex* visc1_c, fftw_complex* visc2_c, fftw_complex* visc3_c,
              fftw_complex* f1_c, fftw_complex* f2_c, fftw_complex* f3_c,
              ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
              ptrdiff_t local_n0, ptrdiff_t local_0_start,
              ptrdiff_t alloc_local, double tau, double t,
              double dx, double dy, double dz) {

    ptrdiff_t nz_complex = nz/2 + 1;
    ptrdiff_t total_size = local_n0 * ny * nz_complex;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // 保存原始V_c（因为FFT计划绑定到这些数组，compute_rhs会修改它们）
    fftw_complex *V1_c_orig = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c_orig = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c_orig = fftw_alloc_complex(alloc_local);
    memcpy(V1_c_orig, V1_c, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c_orig, V2_c, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c_orig, V3_c, alloc_local * sizeof(fftw_complex));

    // k1 = f(V^n, t^n) - V_c已经包含V^n
    compute_rhs(V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                nx, ny, nz, local_n0, local_0_start, alloc_local,
                t, dx, dy, dz);

    // tmp = V^n + τ/2 * k1（使用原始V_c）
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        tmp_v1[i][0] = V1_c_orig[i][0] + 0.5*tau*k1_v1[i][0];
        tmp_v1[i][1] = V1_c_orig[i][1] + 0.5*tau*k1_v1[i][1];
        tmp_v2[i][0] = V2_c_orig[i][0] + 0.5*tau*k1_v2[i][0];
        tmp_v2[i][1] = V2_c_orig[i][1] + 0.5*tau*k1_v2[i][1];
        tmp_v3[i][0] = V3_c_orig[i][0] + 0.5*tau*k1_v3[i][0];
        tmp_v3[i][1] = V3_c_orig[i][1] + 0.5*tau*k1_v3[i][1];
    }

    // k2 = f(V^n + τ/2*k1, t^n + τ/2) - 将tmp复制到V_c
    memcpy(V1_c, tmp_v1, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c, tmp_v2, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c, tmp_v3, alloc_local * sizeof(fftw_complex));

    compute_rhs(V1_c, V2_c, V3_c, k2_v1, k2_v2, k2_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                nx, ny, nz, local_n0, local_0_start, alloc_local,
                t+0.5*tau, dx, dy, dz);

    // tmp = V^n + τ/2 * k2（使用原始V_c）
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        tmp_v1[i][0] = V1_c_orig[i][0] + 0.5*tau*k2_v1[i][0];
        tmp_v1[i][1] = V1_c_orig[i][1] + 0.5*tau*k2_v1[i][1];
        tmp_v2[i][0] = V2_c_orig[i][0] + 0.5*tau*k2_v2[i][0];
        tmp_v2[i][1] = V2_c_orig[i][1] + 0.5*tau*k2_v2[i][1];
        tmp_v3[i][0] = V3_c_orig[i][0] + 0.5*tau*k2_v3[i][0];
        tmp_v3[i][1] = V3_c_orig[i][1] + 0.5*tau*k2_v3[i][1];
    }

    // k3 = f(V^n + τ/2*k2, t^n + τ/2) - 将tmp复制到V_c
    memcpy(V1_c, tmp_v1, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c, tmp_v2, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c, tmp_v3, alloc_local * sizeof(fftw_complex));

    compute_rhs(V1_c, V2_c, V3_c, k3_v1, k3_v2, k3_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                nx, ny, nz, local_n0, local_0_start, alloc_local,
                t+0.5*tau, dx, dy, dz);

    // tmp = V^n + τ * k3（使用原始V_c）
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        tmp_v1[i][0] = V1_c_orig[i][0] + tau*k3_v1[i][0];
        tmp_v1[i][1] = V1_c_orig[i][1] + tau*k3_v1[i][1];
        tmp_v2[i][0] = V2_c_orig[i][0] + tau*k3_v2[i][0];
        tmp_v2[i][1] = V2_c_orig[i][1] + tau*k3_v2[i][1];
        tmp_v3[i][0] = V3_c_orig[i][0] + tau*k3_v3[i][0];
        tmp_v3[i][1] = V3_c_orig[i][1] + tau*k3_v3[i][1];
    }

    // k4 = f(V^n + τ*k3, t^n + τ) - 将tmp复制到V_c
    memcpy(V1_c, tmp_v1, alloc_local * sizeof(fftw_complex));
    memcpy(V2_c, tmp_v2, alloc_local * sizeof(fftw_complex));
    memcpy(V3_c, tmp_v3, alloc_local * sizeof(fftw_complex));

    compute_rhs(V1_c, V2_c, V3_c, k4_v1, k4_v2, k4_v3,
                V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                nx, ny, nz, local_n0, local_0_start, alloc_local,
                t+tau, dx, dy, dz);

    // V^{n+1} = V^n + τ/6 * (k1 + 2k2 + 2k3 + k4)（使用原始V_c）
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < total_size; ++i) {
        V1_c[i][0] = V1_c_orig[i][0] + (tau/6.0) * (k1_v1[i][0] + 2*k2_v1[i][0] + 2*k3_v1[i][0] + k4_v1[i][0]);
        V1_c[i][1] = V1_c_orig[i][1] + (tau/6.0) * (k1_v1[i][1] + 2*k2_v1[i][1] + 2*k3_v1[i][1] + k4_v1[i][1]);
        V2_c[i][0] = V2_c_orig[i][0] + (tau/6.0) * (k1_v2[i][0] + 2*k2_v2[i][0] + 2*k3_v2[i][0] + k4_v2[i][0]);
        V2_c[i][1] = V2_c_orig[i][1] + (tau/6.0) * (k1_v2[i][1] + 2*k2_v2[i][1] + 2*k3_v2[i][1] + k4_v2[i][1]);
        V3_c[i][0] = V3_c_orig[i][0] + (tau/6.0) * (k1_v3[i][0] + 2*k2_v3[i][0] + 2*k3_v3[i][0] + k4_v3[i][0]);
        V3_c[i][1] = V3_c_orig[i][1] + (tau/6.0) * (k1_v3[i][1] + 2*k2_v3[i][1] + 2*k3_v3[i][1] + k4_v3[i][1]);
    }

    // 释放临时数组
    fftw_free(V1_c_orig);
    fftw_free(V2_c_orig);
    fftw_free(V3_c_orig);
}

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char **argv) {
    // 问题参数
    const ptrdiff_t nx = 64, ny = 64, nz = 64;
    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;

    // 时间步进参数
    const ptrdiff_t nt_total = 20000;  // 总时间步数
    const ptrdiff_t nt_run = 10;        // 实际运行步数（验证用）
    const double T = 1.0;               // 总时间
    const double tau = T / nt_total;    // 时间步长 dt = 5e-5

    // MPI初始化
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        cout << "Warning: MPI does not have thread support" << endl;
    }

    // OpenMP初始化
    int max_threads = omp_get_max_threads();
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);

    // FFTW MPI初始化
    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Navier-Stokes Solver - Periodic Boundary Conditions" << endl;
        cout << "  Taylor-Green Vortex Test Case" << endl;
        cout << "============================================================" << endl;
        cout << "Grid: " << nx << " x " << ny << " x " << nz << endl;
        cout << "Domain: [0, 2π]³" << endl;
        cout << "Diffusion coefficient P: 1" << endl;
        cout << "Total time steps: " << nt_total << endl;
        cout << "Running steps (validation): " << nt_run << endl;
        cout << "Time step size: " << tau << endl;
        cout << "Total simulation time: " << T << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads/process: " << max_threads << endl;
        cout << "============================================================" << endl;
    }

    // 分配本地数据
    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(nx, ny, nz/2+1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);

    if (rank == 0) {
        cout << "Local allocation size: " << alloc_local << endl;
    }

    // 分配内存
    double *V1_r = fftw_alloc_real(2 * alloc_local);
    double *V2_r = fftw_alloc_real(2 * alloc_local);
    double *V3_r = fftw_alloc_real(2 * alloc_local);

    fftw_complex *V1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c = fftw_alloc_complex(alloc_local);

    fftw_complex *rot1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *rot2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *rot3_c = fftw_alloc_complex(alloc_local);

    fftw_complex *div_c = fftw_alloc_complex(alloc_local);
    fftw_complex *phi_c = fftw_alloc_complex(alloc_local);

    // 初始化FFT计划
    if (rank == 0) cout << "Initializing FFTW plans..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_plan_start = MPI_Wtime();

    initialize_fftw_3d(nx, ny, nz, V1_r, V2_r, V3_r, V1_c, V2_c, V3_c);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_plan_end = MPI_Wtime();
    if (rank == 0) {
        cout << "FFTW planning time: " << t_plan_end - t_plan_start << " seconds" << endl;
    }

    // 设置初始条件（Taylor-Green涡，t=0）
    if (rank == 0) cout << "Setting initial conditions..." << endl;

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t i_global = local_0_start + i;
                double x = i_global * L_x / nx;
                double y = j * L_y / ny;
                double z = k * L_z / nz;

                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                V1_r[index] = func_V1(x, y, z, 0.0);
                V2_r[index] = func_V2(x, y, z, 0.0);
                V3_r[index] = func_V3(x, y, z, 0.0);
            }
        }
    }

    // 正向FFT到频谱空间
    if (rank == 0) cout << "Transforming to spectral space..." << endl;

    fftw_execute(plan_fwd_v1);  // V1_r -> V1_c
    fftw_execute(plan_fwd_v2);  // V2_r -> V2_c
    fftw_execute(plan_fwd_v3);  // V3_r -> V3_c

    // 归一化
    double norm_factor = nx * ny * nz;
    normalization(V1_c, alloc_local, norm_factor);
    normalization(V2_c, alloc_local, norm_factor);
    normalization(V3_c, alloc_local, norm_factor);

    // 强制初始条件无散（投影到无散空间）
    if (rank == 0) cout << "Projecting initial condition to divergence-free space..." << endl;
    make_div_free(V1_c, V2_c, V3_c, div_c, phi_c, nx, ny, nz, local_n0, local_0_start);

    // 分配RK4临时变量
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

    // 分配完整NS方程所需的临时数组
    double *work_r1 = fftw_alloc_real(2 * alloc_local);
    double *work_r2 = fftw_alloc_real(2 * alloc_local);
    double *work_r3 = fftw_alloc_real(2 * alloc_local);

    fftw_complex *nl1_c = fftw_alloc_complex(alloc_local);   // 非线性项
    fftw_complex *nl2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *nl3_c = fftw_alloc_complex(alloc_local);

    fftw_complex *visc1_c = fftw_alloc_complex(alloc_local); // 粘性项
    fftw_complex *visc2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *visc3_c = fftw_alloc_complex(alloc_local);

    fftw_complex *f1_c = fftw_alloc_complex(alloc_local);    // 外力项
    fftw_complex *f2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *f3_c = fftw_alloc_complex(alloc_local);

    // 时间步进
    if (rank == 0) {
        cout << "\n============================================================" << endl;
        cout << "  Time Integration (RK4)" << endl;
        cout << "============================================================" << endl;
        cout << setw(6) << "Step" << setw(12) << "Time"
             << setw(15) << "L2 Error" << setw(15) << "Max |div V|" << endl;
        cout << "------------------------------------------------------------------------" << endl;
    }

    for(ptrdiff_t it = 0; it <= nt_run; ++it) {
        double t = it * tau;

        // 转到实空间计算误差
        fftw_execute(plan_bwd_v1);
        fftw_execute(plan_bwd_v2);
        fftw_execute(plan_bwd_v3);

        // 计算误差
        double local_error = 0.0;
        #pragma omp parallel for collapse(3) reduction(+:local_error)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz; ++k) {
                    ptrdiff_t i_global = local_0_start + i;
                    double x = i_global * L_x / nx;
                    double y = j * L_y / ny;
                    double z = k * L_z / nz;
                    ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                    double v1_exact = func_V1(x, y, z, t);
                    double v2_exact = func_V2(x, y, z, t);
                    double v3_exact = func_V3(x, y, z, t);

                    double diff1 = V1_r[index] - v1_exact;
                    double diff2 = V2_r[index] - v2_exact;
                    double diff3 = V3_r[index] - v3_exact;

                    local_error += diff1*diff1 + diff2*diff2 + diff3*diff3;
                }
            }
        }

        double global_error;
        MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // 计算散度（频谱空间精确计算）
        // 先转回频谱空间
        fftw_execute(plan_fwd_v1);
        fftw_execute(plan_fwd_v2);
        fftw_execute(plan_fwd_v3);
        normalization(V1_c, alloc_local, norm_factor);
        normalization(V2_c, alloc_local, norm_factor);
        normalization(V3_c, alloc_local, norm_factor);

        // div(V) = ik_x*V1 + ik_y*V2 + ik_z*V3
        double local_max_div = 0.0;
        #pragma omp parallel for collapse(3) reduction(max:local_max_div)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz/2+1; ++k) {
                    ptrdiff_t i_global = local_0_start + i;
                    ptrdiff_t index = (i * ny + j) * (nz/2+1) + k;

                    // 波数
                    double kx = (i_global <= nx/2) ? i_global : i_global - nx;
                    double ky = (j <= ny/2) ? j : j - ny;
                    double kz = k;

                    // div = i*kx*V1 + i*ky*V2 + i*kz*V3
                    double div_real = -kx * V1_c[index][1] - ky * V2_c[index][1] - kz * V3_c[index][1];
                    double div_imag =  kx * V1_c[index][0] + ky * V2_c[index][0] + kz * V3_c[index][0];

                    double div_mag = sqrt(div_real*div_real + div_imag*div_imag);
                    local_max_div = max(local_max_div, div_mag);
                }
            }
        }

        double global_max_div;
        MPI_Reduce(&local_max_div, &global_max_div, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double dx = L_x/nx, dy = L_y/ny, dz = L_z/nz;
            double L2_error = sqrt(global_error * dx * dy * dz);
            cout << setw(6) << it << setw(12) << scientific << setprecision(4) << t
                 << setw(15) << L2_error << setw(15) << global_max_div << endl;
        }

        // 继续时间步进（V已在频谱空间）
        if (it < nt_run) {
            double dx = L_x / nx, dy = L_y / ny, dz = L_z / nz;

            rk4_step(V1_c, V2_c, V3_c,
                    V1_r, V2_r, V3_r,
                    work_r1, work_r2, work_r3,
                    k1_v1, k1_v2, k1_v3,
                    k2_v1, k2_v2, k2_v3,
                    k3_v1, k3_v2, k3_v3,
                    k4_v1, k4_v2, k4_v3,
                    tmp_v1, tmp_v2, tmp_v3,
                    rot1_c, rot2_c, rot3_c,
                    nl1_c, nl2_c, nl3_c,
                    visc1_c, visc2_c, visc3_c,
                    f1_c, f2_c, f3_c,
                    nx, ny, nz, local_n0, local_0_start,
                    alloc_local, tau, t, dx, dy, dz);

            // 投影到无散空间不再需要，因为compute_rhs中每个k1-k4都已经投影过
            // 如果k1, k2, k3, k4都无散，则它们的线性组合也无散
            // make_div_free(V1_c, V2_c, V3_c,
            //              div_c, phi_c,
            //              nx, ny, nz, local_n0, local_0_start);
        }
    }

    // 清理所有临时变量
    fftw_free(k1_v1); fftw_free(k1_v2); fftw_free(k1_v3);
    fftw_free(k2_v1); fftw_free(k2_v2); fftw_free(k2_v3);
    fftw_free(k3_v1); fftw_free(k3_v2); fftw_free(k3_v3);
    fftw_free(k4_v1); fftw_free(k4_v2); fftw_free(k4_v3);
    fftw_free(tmp_v1); fftw_free(tmp_v2); fftw_free(tmp_v3);

    fftw_free(work_r1); fftw_free(work_r2); fftw_free(work_r3);
    fftw_free(nl1_c); fftw_free(nl2_c); fftw_free(nl3_c);
    fftw_free(visc1_c); fftw_free(visc2_c); fftw_free(visc3_c);
    fftw_free(f1_c); fftw_free(f2_c); fftw_free(f3_c);

    // 清理
    finalize_fft_plans();

    fftw_free(V1_r);
    fftw_free(V2_r);
    fftw_free(V3_r);
    fftw_free(V1_c);
    fftw_free(V2_c);
    fftw_free(V3_c);
    fftw_free(rot1_c);
    fftw_free(rot2_c);
    fftw_free(rot3_c);
    fftw_free(div_c);
    fftw_free(phi_c);

    MPI_Finalize();

    if (rank == 0) {
        cout << "\n============================================================" << endl;
        cout << "  Program completed successfully!" << endl;
        cout << "============================================================" << endl;
    }

    return 0;
}
