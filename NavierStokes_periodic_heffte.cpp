#include <iostream>
#include <cmath>
#include <complex>
#include <iomanip>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <vector>
#include <memory>
#include <heffte.h>  // heFFTe library with FFTW backend

using namespace std;

// ==============================================================================
// 全周期性边界条件的Navier-Stokes求解器
// 使用heFFTe进行3D R2C/C2R变换 (FFTW backend, pencil decomposition)
// ==============================================================================

// 全局heFFTe对象 (使用unique_ptr延迟初始化)
std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_v1;  // V1分量的R2C变换
std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_v2;  // V2分量的R2C变换
std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_v3;  // V3分量的R2C变换

// 数据分布信息 (pencil decomposition)
heffte::box3d<> inbox_r;   // 实空间输入box (每个进程的局部域)
heffte::box3d<> outbox_c;  // 频谱空间输出box (每个进程的局部域)

// ==============================================================================
// heFFTe辅助函数：索引和波数计算
// ==============================================================================

/**
 * 将box内的3D坐标(i, j, k)转换为线性索引
 * @param i, j, k: 全局坐标
 * @param box: heFFTe box
 * @return 线性索引（在当前box的数组中的位置）
 */
inline size_t box_index(int i, int j, int k, const heffte::box3d<>& box) {
    int ni = box.size[0];  // x方向大小
    int nj = box.size[1];  // y方向大小
    int nk = box.size[2];  // z方向大小

    int local_i = i - box.low[0];
    int local_j = j - box.low[1];
    int local_k = k - box.low[2];

    return local_i * nj * nk + local_j * nk + local_k;
}

/**
 * 计算波数（处理Nyquist折叠）
 * @param idx: 全局索引 (0 to n-1)
 * @param n: 该方向的总网格数
 * @return 波数k（可能为负）
 */
inline double get_wavenumber(int idx, int n) {
    return (idx <= n/2) ? idx : idx - n;
}

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
// 包含heFFTe专用频谱操作库
// ==============================================================================
#include "heffte_spectral_ops.hpp"

// ==============================================================================
// FFT初始化和辅助函数
// ==============================================================================

/**
 * 初始化heFFTe的3D R2C/C2R变换
 * 使用pencil decomposition (2D分解)
 */
void initialize_heffte_3d(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // 创建2D进程网格用于pencil分解
    // heFFTe会自动优化分解策略
    std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(
        {nx, ny, nz}, nprocs
    );

    if (rank == 0) {
        cout << "heFFTe pencil decomposition: "
             << proc_grid[0] << " x " << proc_grid[1] << " x " << proc_grid[2] << endl;
    }

    // 定义全局域box [0, nx-1] x [0, ny-1] x [0, nz-1]
    heffte::box3d<> world_box = {{0, 0, 0}, {nx-1, ny-1, nz-1}};

    // 使用heFFTe的split_world创建实空间和频谱空间的box
    std::vector<heffte::box3d<>> all_inboxes = heffte::split_world(world_box, proc_grid);
    inbox_r = all_inboxes[rank];

    // R2C变换：频谱空间z方向压缩 (0 to nz/2)
    heffte::box3d<> world_box_c = {{0, 0, 0}, {nx-1, ny-1, nz/2}};
    std::vector<heffte::box3d<>> all_outboxes = heffte::split_world(world_box_c, proc_grid);
    outbox_c = all_outboxes[rank];

    if (rank == 0) {
        cout << "Example inbox (rank 0): ["
             << inbox_r.low[0] << ":" << inbox_r.high[0] << ", "
             << inbox_r.low[1] << ":" << inbox_r.high[1] << ", "
             << inbox_r.low[2] << ":" << inbox_r.high[2] << "]" << endl;
        cout << "Example outbox (rank 0): ["
             << outbox_c.low[0] << ":" << outbox_c.high[0] << ", "
             << outbox_c.low[1] << ":" << outbox_c.high[1] << ", "
             << outbox_c.low[2] << ":" << outbox_c.high[2] << "]" << endl;
    }

    // 创建heFFTe R2C变换对象 (每个速度分量一个)
    // 使用FFTW backend
    fft_v1 = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(
        inbox_r, outbox_c, comm
    );
    fft_v2 = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(
        inbox_r, outbox_c, comm
    );
    fft_v3 = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(
        inbox_r, outbox_c, comm
    );

    if (rank == 0) {
        cout << "heFFTe initialization complete" << endl;
    }
}

void finalize_fft_plans() {
    // heFFTe使用unique_ptr，会自动清理
    fft_v1.reset();
    fft_v2.reset();
    fft_v3.reset();
}

// ==============================================================================
// heFFTe专用时间步进函数
// ==============================================================================

/**
 * 计算非线性项 v×rot(v)（伪谱方法）
 * 使用heFFTe变换
 */
void heffte_compute_nonlinear_term(
    const std::vector<std::complex<double>>& V1_c,
    const std::vector<std::complex<double>>& V2_c,
    const std::vector<std::complex<double>>& V3_c,
    std::vector<std::complex<double>>& nl1_c,
    std::vector<std::complex<double>>& nl2_c,
    std::vector<std::complex<double>>& nl3_c,
    std::vector<double>& V1_r,
    std::vector<double>& V2_r,
    std::vector<double>& V3_r,
    std::vector<double>& rot1_r,
    std::vector<double>& rot2_r,
    std::vector<double>& rot3_r,
    std::vector<std::complex<double>>& rot1_c,
    std::vector<std::complex<double>>& rot2_c,
    std::vector<std::complex<double>>& rot3_c,
    int nx, int ny, int nz)
{
    // 1. 在频谱空间计算旋度
    heffte_compute_rot(V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c,
                       nx, ny, nz, outbox_c);

    // 2. 速度和旋度都转到实空间
    fft_v1->backward(V1_c.data(), V1_r.data(), heffte::scale::full);
    fft_v2->backward(V2_c.data(), V2_r.data(), heffte::scale::full);
    fft_v3->backward(V3_c.data(), V3_r.data(), heffte::scale::full);

    fft_v1->backward(rot1_c.data(), rot1_r.data(), heffte::scale::full);
    fft_v2->backward(rot2_c.data(), rot2_r.data(), heffte::scale::full);
    fft_v3->backward(rot3_c.data(), rot3_r.data(), heffte::scale::full);

    // 3. 在实空间计算 v × rot(v)
    size_t local_size_r = inbox_r.count();
    #pragma omp parallel for
    for(size_t i = 0; i < local_size_r; ++i) {
        double v1 = V1_r[i];
        double v2 = V2_r[i];
        double v3 = V3_r[i];
        double w1 = rot1_r[i];
        double w2 = rot2_r[i];
        double w3 = rot3_r[i];

        // v × rot(v) 的三个分量
        rot1_r[i] = v2 * w3 - v3 * w2;  // 复用数组存储结果
        rot2_r[i] = v3 * w1 - v1 * w3;
        rot3_r[i] = v1 * w2 - v2 * w1;
    }

    // 4. 转回频谱空间
    fft_v1->forward(rot1_r.data(), nl1_c.data(), heffte::scale::full);
    fft_v2->forward(rot2_r.data(), nl2_c.data(), heffte::scale::full);
    fft_v3->forward(rot3_r.data(), nl3_c.data(), heffte::scale::full);
}

/**
 * 计算完整的RHS：dV/dt = v×rot(v) + P∆v + f
 * 并投影到无散空间
 */
void heffte_compute_rhs(
    const std::vector<std::complex<double>>& V1_c,
    const std::vector<std::complex<double>>& V2_c,
    const std::vector<std::complex<double>>& V3_c,
    std::vector<std::complex<double>>& rhs1_c,
    std::vector<std::complex<double>>& rhs2_c,
    std::vector<std::complex<double>>& rhs3_c,
    std::vector<double>& V1_r,
    std::vector<double>& V2_r,
    std::vector<double>& V3_r,
    std::vector<double>& work_r1,
    std::vector<double>& work_r2,
    std::vector<double>& work_r3,
    std::vector<std::complex<double>>& rot1_c,
    std::vector<std::complex<double>>& rot2_c,
    std::vector<std::complex<double>>& rot3_c,
    std::vector<std::complex<double>>& nl1_c,
    std::vector<std::complex<double>>& nl2_c,
    std::vector<std::complex<double>>& nl3_c,
    std::vector<std::complex<double>>& visc1_c,
    std::vector<std::complex<double>>& visc2_c,
    std::vector<std::complex<double>>& visc3_c,
    std::vector<std::complex<double>>& f1_c,
    std::vector<std::complex<double>>& f2_c,
    std::vector<std::complex<double>>& f3_c,
    std::vector<std::complex<double>>& div_c,
    std::vector<std::complex<double>>& phi_c,
    int nx, int ny, int nz,
    double t, double dx, double dy, double dz)
{
    size_t local_size_c = outbox_c.count();
    size_t local_size_r = inbox_r.count();

    // 1. 计算非线性项 v×rot(v)
    heffte_compute_nonlinear_term(V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                                   V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                                   rot1_c, rot2_c, rot3_c, nx, ny, nz);

    // 2. 计算粘性项 P∆v
    heffte_compute_viscous_term(V1_c, visc1_c, nx, ny, nz, outbox_c);
    heffte_compute_viscous_term(V2_c, visc2_c, nx, ny, nz, outbox_c);
    heffte_compute_viscous_term(V3_c, visc3_c, nx, ny, nz, outbox_c);

    // 3. 计算外力项 f（实空间）
    auto inbox_low = inbox_r.low;
    auto inbox_high = inbox_r.high;

    size_t idx = 0;
    for(int i = inbox_low[0]; i <= inbox_high[0]; ++i) {
        for(int j = inbox_low[1]; j <= inbox_high[1]; ++j) {
            for(int k = inbox_low[2]; k <= inbox_high[2]; ++k) {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;

                work_r1[idx] = func_f1(x, y, z, t);
                work_r2[idx] = func_f2(x, y, z, t);
                work_r3[idx] = func_f3(x, y, z, t);
                ++idx;
            }
        }
    }

    // f转到频谱空间
    fft_v1->forward(work_r1.data(), f1_c.data(), heffte::scale::full);
    fft_v2->forward(work_r2.data(), f2_c.data(), heffte::scale::full);
    fft_v3->forward(work_r3.data(), f3_c.data(), heffte::scale::full);

    // 4. 组合所有项：F = v×rot(v) + P∆v + f
    #pragma omp parallel for
    for(size_t i = 0; i < local_size_c; ++i) {
        rhs1_c[i] = nl1_c[i] + visc1_c[i] + f1_c[i];
        rhs2_c[i] = nl2_c[i] + visc2_c[i] + f2_c[i];
        rhs3_c[i] = nl3_c[i] + visc3_c[i] + f3_c[i];
    }

    // 5. 投影到无散空间：rhs = rhs - ∇(∇⁻²·∇·rhs)
    heffte_make_div_free(rhs1_c, rhs2_c, rhs3_c, div_c, phi_c, nx, ny, nz, outbox_c);
}

/**
 * RK4时间步进（heFFTe版本）
 */
void heffte_rk4_step(
    std::vector<std::complex<double>>& V1_c,
    std::vector<std::complex<double>>& V2_c,
    std::vector<std::complex<double>>& V3_c,
    std::vector<double>& V1_r,
    std::vector<double>& V2_r,
    std::vector<double>& V3_r,
    std::vector<double>& work_r1,
    std::vector<double>& work_r2,
    std::vector<double>& work_r3,
    std::vector<std::complex<double>>& k1_v1,
    std::vector<std::complex<double>>& k1_v2,
    std::vector<std::complex<double>>& k1_v3,
    std::vector<std::complex<double>>& k2_v1,
    std::vector<std::complex<double>>& k2_v2,
    std::vector<std::complex<double>>& k2_v3,
    std::vector<std::complex<double>>& k3_v1,
    std::vector<std::complex<double>>& k3_v2,
    std::vector<std::complex<double>>& k3_v3,
    std::vector<std::complex<double>>& k4_v1,
    std::vector<std::complex<double>>& k4_v2,
    std::vector<std::complex<double>>& k4_v3,
    std::vector<std::complex<double>>& tmp_v1,
    std::vector<std::complex<double>>& tmp_v2,
    std::vector<std::complex<double>>& tmp_v3,
    std::vector<std::complex<double>>& rot1_c,
    std::vector<std::complex<double>>& rot2_c,
    std::vector<std::complex<double>>& rot3_c,
    std::vector<std::complex<double>>& nl1_c,
    std::vector<std::complex<double>>& nl2_c,
    std::vector<std::complex<double>>& nl3_c,
    std::vector<std::complex<double>>& visc1_c,
    std::vector<std::complex<double>>& visc2_c,
    std::vector<std::complex<double>>& visc3_c,
    std::vector<std::complex<double>>& f1_c,
    std::vector<std::complex<double>>& f2_c,
    std::vector<std::complex<double>>& f3_c,
    std::vector<std::complex<double>>& div_c,
    std::vector<std::complex<double>>& phi_c,
    int nx, int ny, int nz,
    double tau, double t,
    double dx, double dy, double dz)
{
    size_t local_size_c = outbox_c.count();

    // 保存原始 V_c
    auto V1_c_orig = V1_c;
    auto V2_c_orig = V2_c;
    auto V3_c_orig = V3_c;

    // k1 = f(V^n, t^n)
    heffte_compute_rhs(V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                       V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                       rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                       visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                       div_c, phi_c, nx, ny, nz, t, dx, dy, dz);

    // tmp = V^n + τ/2 * k1
    #pragma omp parallel for
    for(size_t i = 0; i < local_size_c; ++i) {
        tmp_v1[i] = V1_c_orig[i] + 0.5*tau*k1_v1[i];
        tmp_v2[i] = V2_c_orig[i] + 0.5*tau*k1_v2[i];
        tmp_v3[i] = V3_c_orig[i] + 0.5*tau*k1_v3[i];
    }

    // k2 = f(V^n + τ/2*k1, t^n + τ/2)
    heffte_compute_rhs(tmp_v1, tmp_v2, tmp_v3, k2_v1, k2_v2, k2_v3,
                       V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                       rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                       visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                       div_c, phi_c, nx, ny, nz, t+0.5*tau, dx, dy, dz);

    // tmp = V^n + τ/2 * k2
    #pragma omp parallel for
    for(size_t i = 0; i < local_size_c; ++i) {
        tmp_v1[i] = V1_c_orig[i] + 0.5*tau*k2_v1[i];
        tmp_v2[i] = V2_c_orig[i] + 0.5*tau*k2_v2[i];
        tmp_v3[i] = V3_c_orig[i] + 0.5*tau*k2_v3[i];
    }

    // k3 = f(V^n + τ/2*k2, t^n + τ/2)
    heffte_compute_rhs(tmp_v1, tmp_v2, tmp_v3, k3_v1, k3_v2, k3_v3,
                       V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                       rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                       visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                       div_c, phi_c, nx, ny, nz, t+0.5*tau, dx, dy, dz);

    // tmp = V^n + τ * k3
    #pragma omp parallel for
    for(size_t i = 0; i < local_size_c; ++i) {
        tmp_v1[i] = V1_c_orig[i] + tau*k3_v1[i];
        tmp_v2[i] = V2_c_orig[i] + tau*k3_v2[i];
        tmp_v3[i] = V3_c_orig[i] + tau*k3_v3[i];
    }

    // k4 = f(V^n + τ*k3, t^n + τ)
    heffte_compute_rhs(tmp_v1, tmp_v2, tmp_v3, k4_v1, k4_v2, k4_v3,
                       V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                       rot1_c, rot2_c, rot3_c, nl1_c, nl2_c, nl3_c,
                       visc1_c, visc2_c, visc3_c, f1_c, f2_c, f3_c,
                       div_c, phi_c, nx, ny, nz, t+tau, dx, dy, dz);

    // V^{n+1} = V^n + τ/6 * (k1 + 2k2 + 2k3 + k4)
    #pragma omp parallel for
    for(size_t i = 0; i < local_size_c; ++i) {
        V1_c[i] = V1_c_orig[i] + (tau/6.0) * (k1_v1[i] + 2.0*k2_v1[i] + 2.0*k3_v1[i] + k4_v1[i]);
        V2_c[i] = V2_c_orig[i] + (tau/6.0) * (k1_v2[i] + 2.0*k2_v2[i] + 2.0*k3_v2[i] + k4_v2[i]);
        V3_c[i] = V3_c_orig[i] + (tau/6.0) * (k1_v3[i] + 2.0*k2_v3[i] + 2.0*k3_v3[i] + k4_v3[i]);
    }
}

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char **argv) {
    // 问题参数
    const int nx = 64, ny = 64, nz = 64;
    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;
    const double dx = L_x / nx, dy = L_y / ny, dz = L_z / nz;

    // 时间步进参数
    const int nt_total = 20000;
    const int nt_run = 5000;
    const double T = 1.0;
    const double tau = T / nt_total;

    // MPI初始化
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Navier-Stokes Solver - heFFTe Version" << endl;
        cout << "  Periodic Boundary Conditions" << endl;
        cout << "============================================================" << endl;
        cout << "Grid: " << nx << " x " << ny << " x " << nz << endl;
        cout << "Domain: [0, 2π]³" << endl;
        cout << "Total time steps: " << nt_total << endl;
        cout << "Running steps: " << nt_run << endl;
        cout << "Time step size: " << tau << endl;
        cout << "MPI processes: " << size << endl;
        cout << "============================================================" << endl;
    }

    // 初始化heFFTe
    if (rank == 0) cout << "Initializing heFFTe..." << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_plan_start = MPI_Wtime();

    initialize_heffte_3d(nx, ny, nz, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_plan_end = MPI_Wtime();
    if (rank == 0) {
        cout << "heFFTe planning time: " << t_plan_end - t_plan_start << " seconds" << endl;
    }

    // 分配本地数据
    size_t local_size_r = inbox_r.count();
    size_t local_size_c = outbox_c.count();

    if (rank == 0) {
        cout << "Local real space size: " << local_size_r << endl;
        cout << "Local complex space size: " << local_size_c << endl;
    }

    // 分配内存（使用std::vector）
    std::vector<double> V1_r(local_size_r);
    std::vector<double> V2_r(local_size_r);
    std::vector<double> V3_r(local_size_r);

    std::vector<std::complex<double>> V1_c(local_size_c);
    std::vector<std::complex<double>> V2_c(local_size_c);
    std::vector<std::complex<double>> V3_c(local_size_c);

    std::vector<std::complex<double>> rot1_c(local_size_c);
    std::vector<std::complex<double>> rot2_c(local_size_c);
    std::vector<std::complex<double>> rot3_c(local_size_c);

    std::vector<std::complex<double>> div_c(local_size_c);
    std::vector<std::complex<double>> phi_c(local_size_c);

    std::vector<double> work_r1(local_size_r);
    std::vector<double> work_r2(local_size_r);
    std::vector<double> work_r3(local_size_r);

    // RK4临时变量
    std::vector<std::complex<double>> k1_v1(local_size_c), k1_v2(local_size_c), k1_v3(local_size_c);
    std::vector<std::complex<double>> k2_v1(local_size_c), k2_v2(local_size_c), k2_v3(local_size_c);
    std::vector<std::complex<double>> k3_v1(local_size_c), k3_v2(local_size_c), k3_v3(local_size_c);
    std::vector<std::complex<double>> k4_v1(local_size_c), k4_v2(local_size_c), k4_v3(local_size_c);
    std::vector<std::complex<double>> tmp_v1(local_size_c), tmp_v2(local_size_c), tmp_v3(local_size_c);

    std::vector<std::complex<double>> nl1_c(local_size_c), nl2_c(local_size_c), nl3_c(local_size_c);
    std::vector<std::complex<double>> visc1_c(local_size_c), visc2_c(local_size_c), visc3_c(local_size_c);
    std::vector<std::complex<double>> f1_c(local_size_c), f2_c(local_size_c), f3_c(local_size_c);

    // 设置初始条件（t=0）
    if (rank == 0) cout << "Setting initial conditions..." << endl;

    auto inbox_low = inbox_r.low;
    auto inbox_high = inbox_r.high;

    size_t idx = 0;
    for(int i = inbox_low[0]; i <= inbox_high[0]; ++i) {
        for(int j = inbox_low[1]; j <= inbox_high[1]; ++j) {
            for(int k = inbox_low[2]; k <= inbox_high[2]; ++k) {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;

                V1_r[idx] = func_V1(x, y, z, 0.0);
                V2_r[idx] = func_V2(x, y, z, 0.0);
                V3_r[idx] = func_V3(x, y, z, 0.0);
                ++idx;
            }
        }
    }

    // 正向FFT到频谱空间
    if (rank == 0) cout << "Transforming to spectral space..." << endl;
    fft_v1->forward(V1_r.data(), V1_c.data(), heffte::scale::full);
    fft_v2->forward(V2_r.data(), V2_c.data(), heffte::scale::full);
    fft_v3->forward(V3_r.data(), V3_c.data(), heffte::scale::full);

    // 强制初始条件无散
    if (rank == 0) cout << "Projecting initial condition to divergence-free space..." << endl;
    heffte_make_div_free(V1_c, V2_c, V3_c, div_c, phi_c, nx, ny, nz, outbox_c);

    // 时间步进
    if (rank == 0) {
        cout << "\n============================================================" << endl;
        cout << "  Time Integration (RK4)" << endl;
        cout << "============================================================" << endl;
        cout << setw(6) << "Step" << setw(12) << "Time"
             << setw(15) << "L2 Error" << setw(15) << "Max |div V|" << endl;
        cout << "------------------------------------------------------------------------" << endl;
    }

    for(int it = 0; it <= nt_run; ++it) {
        double t = it * tau;

        // 转到实空间计算误差
        fft_v1->backward(V1_c.data(), V1_r.data(), heffte::scale::full);
        fft_v2->backward(V2_c.data(), V2_r.data(), heffte::scale::full);
        fft_v3->backward(V3_c.data(), V3_r.data(), heffte::scale::full);

        // 计算L2误差
        double local_error = 0.0;
        idx = 0;
        for(int i = inbox_low[0]; i <= inbox_high[0]; ++i) {
            for(int j = inbox_low[1]; j <= inbox_high[1]; ++j) {
                for(int k = inbox_low[2]; k <= inbox_high[2]; ++k) {
                    double x = i * dx;
                    double y = j * dy;
                    double z = k * dz;

                    double v1_exact = func_V1(x, y, z, t);
                    double v2_exact = func_V2(x, y, z, t);
                    double v3_exact = func_V3(x, y, z, t);

                    double diff1 = V1_r[idx] - v1_exact;
                    double diff2 = V2_r[idx] - v2_exact;
                    double diff3 = V3_r[idx] - v3_exact;

                    local_error += diff1*diff1 + diff2*diff2 + diff3*diff3;
                    ++idx;
                }
            }
        }

        double global_error;
        MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // 计算散度（频谱空间）
        // 先转回频谱空间
        fft_v1->forward(V1_r.data(), V1_c.data(), heffte::scale::full);
        fft_v2->forward(V2_r.data(), V2_c.data(), heffte::scale::full);
        fft_v3->forward(V3_r.data(), V3_c.data(), heffte::scale::full);

        heffte_compute_div(V1_c, V2_c, V3_c, div_c, nx, ny, nz, outbox_c);

        double local_max_div = 0.0;
        for(size_t i = 0; i < local_size_c; ++i) {
            double div_mag = std::abs(div_c[i]);
            local_max_div = std::max(local_max_div, div_mag);
        }

        double global_max_div;
        MPI_Reduce(&local_max_div, &global_max_div, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0 && it % 100 == 0) {
            double L2_error = sqrt(global_error * dx * dy * dz);
            cout << setw(6) << it << setw(12) << scientific << setprecision(4) << t
                 << setw(15) << L2_error << setw(15) << global_max_div << endl;
        }

        // 继续时间步进
        if (it < nt_run) {
            heffte_rk4_step(V1_c, V2_c, V3_c, V1_r, V2_r, V3_r,
                           work_r1, work_r2, work_r3,
                           k1_v1, k1_v2, k1_v3, k2_v1, k2_v2, k2_v3,
                           k3_v1, k3_v2, k3_v3, k4_v1, k4_v2, k4_v3,
                           tmp_v1, tmp_v2, tmp_v3,
                           rot1_c, rot2_c, rot3_c,
                           nl1_c, nl2_c, nl3_c,
                           visc1_c, visc2_c, visc3_c,
                           f1_c, f2_c, f3_c,
                           div_c, phi_c,
                           nx, ny, nz, tau, t, dx, dy, dz);
        }
    }

    // 清理
    finalize_fft_plans();

    MPI_Finalize();

    if (rank == 0) {
        cout << "\n============================================================" << endl;
        cout << "  Program completed successfully!" << endl;
        cout << "============================================================" << endl;
    }

    return 0;
}
