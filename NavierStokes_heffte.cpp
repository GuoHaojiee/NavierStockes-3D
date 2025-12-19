#include <iostream>
#include <cmath>
#include <complex>
#include <heffte.h>  // heFFTe库
#include <fftw3.h>   // 保留FFTW用于z方向R2R
#include <iomanip>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <memory>
#include <vector>

using namespace std;

// ==============================================================================
// 全局变量
// ==============================================================================

// FFTW plans (仅用于z方向局部R2R变换)
fftw_plan plan_r2r_cos_z;
fftw_plan plan_r2r_sin_z;

// heFFTe对象 (用于xy平面分布式变换)
// 注意：这里我们使用unique_ptr来管理heFFTe对象的生命周期
std::unique_ptr<heffte::fft3d<heffte::backend::fftw>> heffte_cos_fwd;
std::unique_ptr<heffte::fft3d<heffte::backend::fftw>> heffte_cos_bwd;
std::unique_ptr<heffte::fft3d<heffte::backend::fftw>> heffte_sin_fwd;
std::unique_ptr<heffte::fft3d<heffte::backend::fftw>> heffte_sin_bwd;

// ==============================================================================
// 数学函数定义 (与原代码相同)
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
    return func_dV1_dt(x,y,z,t)- func_laplace_V1(x,y,z,t)- func_v_cross_rot1(x,y,z,t) + func_grad_p1(x,y,z,t);
}

double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t)- func_laplace_V2(x,y,z,t)- func_v_cross_rot2(x,y,z,t) + func_grad_p2(x,y,z,t);
}

double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t)- func_laplace_V3(x,y,z,t)- func_v_cross_rot3(x,y,z,t) + func_grad_p3(x,y,z,t);
}

// ==============================================================================
// heFFTe初始化函数
// ==============================================================================

/**
 * 初始化z方向的FFTW R2R变换（局部，无MPI通信）
 * 这部分与原代码相同
 */
void initialize_r2r_cos_z(ptrdiff_t local_nx, ptrdiff_t ny, ptrdiff_t nz,
                           double *re_in, double *re_out)
{
    const fftw_r2r_kind kind_cos[] = {FFTW_REDFT00};  // DCT-I型
    const int nz_int = nz/2+1;
    plan_r2r_cos_z = fftw_plan_many_r2r(
        1, &nz_int, local_nx*(2*(ny/2+1)),  // rank=1, n=nz_int, howmany
        re_in, &nz_int, 1, nz_int,           // 输入
        re_out, &nz_int, 1, nz_int,          // 输出
        kind_cos, FFTW_PATIENT
    );
}

void initialize_r2r_sin_z(ptrdiff_t local_nx, ptrdiff_t ny, ptrdiff_t nz,
                           double *re_in, double *re_out)
{
    const fftw_r2r_kind kind_sin[] = {FFTW_RODFT00};  // DST-I型
    const int nz_int = nz/2-1;
    plan_r2r_sin_z = fftw_plan_many_r2r(
        1, &nz_int, local_nx*(2*(ny/2+1)),  // rank=1, n=nz_int, howmany
        re_in, &nz_int, 1, nz_int,           // 输入
        re_out, &nz_int, 1, nz_int,          // 输出
        kind_sin, FFTW_PATIENT
    );
}

/**
 * 初始化heFFTe用于xy平面的分布式FFT
 *
 * 重要说明：
 * 由于heFFTe是为3D FFT设计的，但我们已经在z方向做了R2R变换，
 * 这里我们需要在xy平面做2D FFT对每个z层进行批处理。
 *
 * 目前的实现策略：
 * 暂时保留FFTW MPI的实现，因为heFFTe主要优势在于3D完整FFT。
 * 未来可以考虑重新设计算法，完全使用heFFTe的3D FFT。
 */

// TODO: 这里需要决定如何使用heFFTe
// 选项1: 继续使用FFTW MPI（保守方案）
// 选项2: 研究heFFTe是否支持2D+批处理
// 选项3: 重新设计算法，使用heFFTe的3D C2C（需要数学推导）

/**
 * 清理资源
 */
void finalize_fft_plans() {
    fftw_destroy_plan(plan_r2r_cos_z);
    fftw_destroy_plan(plan_r2r_sin_z);

    // heFFTe对象通过unique_ptr自动释放
    heffte_cos_fwd.reset();
    heffte_cos_bwd.reset();
    heffte_sin_fwd.reset();
    heffte_sin_bwd.reset();
}

// ==============================================================================
// 辅助函数 (与原代码相同)
// ==============================================================================

void normalization(fftw_complex* V_c_, ptrdiff_t nx,ptrdiff_t local_ny, ptrdiff_t nz, double factor)
{
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (j * nx + i) * nz + k;
                V_c_[index][0] /= factor;
                V_c_[index][1] /= factor;
            }
        }
    }
}

double calculateEnergy(double* V_r_, ptrdiff_t local_nx,ptrdiff_t ny, ptrdiff_t nz, double factor)
{
    double sumSquares = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:sumSquares)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (i * (2*(ny/2+1)) + j) * nz + k;
                sumSquares += V_r_[index] * V_r_[index];
            }
        }
    }
    double energy = 0.5 * sumSquares / factor;
    return energy;
}

// ==============================================================================
// 注意：此文件是重构工作的起点
// ==============================================================================

/**
 * 重构说明：
 *
 * 当前状态：
 * 1. 保留了所有数学函数（func_V1, func_V2等）
 * 2. 保留了z方向的FFTW R2R初始化
 * 3. 添加了heFFTe头文件和对象声明
 *
 * 需要完成的工作：
 * 1. 决定heFFTe的使用策略（见下方分析）
 * 2. 实现相应的变换函数
 * 3. 修改compute_rot, compute_div, compute_v_cross_rot等函数
 * 4. 修改主函数
 *
 * heFFTe使用策略分析：
 *
 * 当前代码结构：
 *   实空间 -> z方向R2R(DCT/DST) -> xy平面R2C(MPI) -> 频谱空间
 *
 * 问题：
 *   heFFTe设计用于3D FFT，不直接支持"2D FFT + 批处理"
 *
 * 可能的解决方案：
 *
 * A) 保守方案：保留FFTW MPI
 *    - z方向：FFTW R2R（局部）
 *    - xy平面：FFTW MPI R2C（分布式）
 *    - 优点：稳定，风险低
 *    - 缺点：无法利用heFFTe的优化
 *
 * B) 部分迁移方案：
 *    - 研究heFFTe是否有API可以只在xy平面做FFT
 *    - 或者使用heFFTe做3D变换，但z方向用"identity"
 *    - 需要深入研究heFFTe文档
 *
 * C) 完全重构方案：
 *    - 放弃分离的R2R+R2C策略
 *    - 使用heFFTe的3D C2C变换
 *    - 通过数学技巧实现DCT/DST边界条件
 *    - 需要重新推导数学方程
 *    - 优点：完全利用heFFTe，最佳性能
 *    - 缺点：工作量大，需要数学功底
 *
 * 推荐：
 *   - 短期：方案A（保守）
 *   - 中期：方案B（如果heFFTe支持）
 *   - 长期：方案C（最优但需要时间）
 */

int main(int argc, char **argv) {
    // 网格参数
    const ptrdiff_t nx = 128, ny = 128, nz = 128;
    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;

    // 时间步进参数
    ptrdiff_t nt = 1;
    double T = 1;
    double tau = T / 20000;

    // MPI初始化
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        cout << "Warning: MPI does not have full thread support" << endl;
    }

    // FFTW线程初始化
    int max_threads = omp_get_max_threads();
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);

    // FFTW MPI初始化
    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "===========================================================" << endl;
        cout << "  Navier-Stokes Solver with heFFTe (Work in Progress)" << endl;
        cout << "===========================================================" << endl;
        cout << "Grid size: " << nx << " x " << ny << " x " << nz << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads per process: " << max_threads << endl;
        cout << "===========================================================" << endl;
        cout << endl;
        cout << "NOTE: This version is under development." << endl;
        cout << "Currently using FFTW for all transforms." << endl;
        cout << "heFFTe integration is in progress..." << endl;
        cout << endl;
    }

    // 确定本地数据分布
    ptrdiff_t alloc_local, local_nx, local_x_start, local_ny, local_y_start;
    alloc_local = fftw_mpi_local_size_2d_transposed(
        nx, ny/2+1, MPI_COMM_WORLD,
        &local_nx, &local_x_start,
        &local_ny, &local_y_start
    );

    if (rank == 0) {
        cout << "Data distribution:" << endl;
        cout << "  alloc_local = " << alloc_local << endl;
    }

    // TODO: 在这里添加heFFTe的初始化代码
    // 当前先使用原始的FFTW实现

    cout << "\n[Rank " << rank << "] heFFTe initialization placeholder..." << endl;
    cout << "[Rank " << rank << "] Current implementation: FFTW-only" << endl;

    // 清理
    finalize_fft_plans();

    MPI_Finalize();

    if (rank == 0) {
        cout << "\n===========================================================" << endl;
        cout << "  Program completed successfully" << endl;
        cout << "===========================================================" << endl;
    }

    return 0;
}
