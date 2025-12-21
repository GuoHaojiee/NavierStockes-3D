/**
 * Navier-Stokes 求解器 - heFFTe 版本（保持与 NavierStokes_periodic_fftw.cpp 相同的核心逻辑）
 * - 使用 heFFTe (FFTW backend) 进行 3D R2C/C2R 变换，pencil 分解
 * - 谱域作为主状态变量，RK4 时间推进
 * - 投影法保证速度场无散度
 * - 解析解/强迫项与 FFTW 版本完全一致，目标是数值结果一致
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <heffte.h>
#include "heffte_spectral_ops.hpp"

using namespace std;

// ==============================================================================
// 解析解与强迫项（保持与 FFTW 版本一致）
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
    double d2v1_dx2 = (t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) *
                      ((cos(3 * x + 3 * y) * cos(3 * x + 3 * y)) - sin(3 * x + 3 * y)) * cos(6 * z);
    double d2v1_dy2 = (t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) *
                      ((cos(3 * x + 3 * y) * cos(3 * x + 3 * y)) - sin(3 * x + 3 * y)) * cos(6 * z);
    double d2v1_dz2 = -(t * t + 1) * 36 * exp(sin(3 * x + 3 * y)) * cos(6 * z);
    return d2v1_dx2 + d2v1_dy2 + d2v1_dz2;
}

double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x, y, z, t);
}

double func_laplace_V3(double x, double y, double z, double t) {
    double d2v3_dx2 = -(t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) *
                      ((cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) -
                       (2 * sin(3 * x + 3 * y) + 1)) *
                      sin(6 * z);
    double d2v3_dy2 = -(t * t + 1) * 9 * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) *
                      ((cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) -
                       (2 * sin(3 * x + 3 * y) + 1)) *
                      sin(6 * z);
    double d2v3_dz2 = (t * t + 1) * 36 * exp(sin(3 * x + 3 * y)) * cos(3 * x + 3 * y) * sin(6 * z);
    return d2v3_dx2 + d2v3_dy2 + d2v3_dz2;
}

double func_rot1(double x, double y, double z, double t) {
    double dv3_dy = -(t * t + 1) * 3 * exp(sin(3 * x + 3 * y)) *
                    (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * sin(6 * z);
    double dv2_dz = -(t * t + 1) * 6 * exp(sin(3 * x + 3 * y)) * sin(6 * z);
    return dv3_dy - dv2_dz;
}

double func_rot2(double x, double y, double z, double t) {
    double dv1_dz = -(t * t + 1) * 6 * exp(sin(3 * x + 3 * y)) * sin(6 * z);
    double dv3_dx = -(t * t + 1) * 3 * exp(sin(3 * x + 3 * y)) *
                    (cos(3 * x + 3 * y) * cos(3 * x + 3 * y) - sin(3 * x + 3 * y)) * sin(6 * z);
    return dv1_dz - dv3_dx;
}

double func_rot3(double, double, double, double) { return 0.0; }

double func_v_cross_rot1(double x, double y, double z, double t) {
    return func_V2(x, y, z, t) * func_rot3(x, y, z, t) - func_V3(x, y, z, t) * func_rot2(x, y, z, t);
}

double func_v_cross_rot2(double x, double y, double z, double t) {
    return func_V3(x, y, z, t) * func_rot1(x, y, z, t) - func_V1(x, y, z, t) * func_rot3(x, y, z, t);
}

double func_v_cross_rot3(double x, double y, double z, double t) {
    return func_V1(x, y, z, t) * func_rot2(x, y, z, t) - func_V2(x, y, z, t) * func_rot1(x, y, z, t);
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

// 强迫项：f = ∂v/∂t - P∆v - v×rot(v) + ∇p（P=1）
double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x, y, z, t) - func_laplace_V1(x, y, z, t) - func_v_cross_rot1(x, y, z, t) +
           func_grad_p1(x, y, z, t);
}

double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x, y, z, t) - func_laplace_V2(x, y, z, t) - func_v_cross_rot2(x, y, z, t) +
           func_grad_p2(x, y, z, t);
}

double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x, y, z, t) - func_laplace_V3(x, y, z, t) - func_v_cross_rot3(x, y, z, t) +
           func_grad_p3(x, y, z, t);
}

// ==============================================================================
// heFFTe 工具
// ==============================================================================

// 计算非线性项 v × rot(v)，输入/输出均在频谱空间
void compute_nonlinear_term_spectral(
    heffte::fft3d_r2c<heffte::backend::fftw> &fft,
    const vector<complex<double>> &V1_c,
    const vector<complex<double>> &V2_c,
    const vector<complex<double>> &V3_c,
    vector<complex<double>> &nl1_c,
    vector<complex<double>> &nl2_c,
    vector<complex<double>> &nl3_c,
    vector<double> &V1_r,
    vector<double> &V2_r,
    vector<double> &V3_r,
    vector<complex<double>> &rot1_c,
    vector<complex<double>> &rot2_c,
    vector<complex<double>> &rot3_c,
    vector<double> &rot1_r,
    vector<double> &rot2_r,
    vector<double> &rot3_r,
    int nx, int ny, int nz,
    const heffte::box3d<> &inbox_r,
    const heffte::box3d<> &outbox_c)
{
    heffte_compute_rot(V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c, nx, ny, nz, outbox_c);

    fft.backward(V1_c.data(), V1_r.data(), heffte::scale::full);
    fft.backward(V2_c.data(), V2_r.data(), heffte::scale::full);
    fft.backward(V3_c.data(), V3_r.data(), heffte::scale::full);

    fft.backward(rot1_c.data(), rot1_r.data(), heffte::scale::full);
    fft.backward(rot2_c.data(), rot2_r.data(), heffte::scale::full);
    fft.backward(rot3_c.data(), rot3_r.data(), heffte::scale::full);

    size_t local_size_r = inbox_r.count();
    #pragma omp parallel for
    for (size_t i = 0; i < local_size_r; ++i) {
        double v1 = V1_r[i];
        double v2 = V2_r[i];
        double v3 = V3_r[i];
        double w1 = rot1_r[i];
        double w2 = rot2_r[i];
        double w3 = rot3_r[i];

        rot1_r[i] = v2 * w3 - v3 * w2;
        rot2_r[i] = v3 * w1 - v1 * w3;
        rot3_r[i] = v1 * w2 - v2 * w1;
    }

    fft.forward(rot1_r.data(), nl1_c.data(), heffte::scale::none);
    fft.forward(rot2_r.data(), nl2_c.data(), heffte::scale::none);
    fft.forward(rot3_r.data(), nl3_c.data(), heffte::scale::none);
}

// 计算 RHS：v×rot(v) + P∆v + f，并投影无散
void compute_rhs_spectral(
    heffte::fft3d_r2c<heffte::backend::fftw> &fft,
    const vector<complex<double>> &V1_c,
    const vector<complex<double>> &V2_c,
    const vector<complex<double>> &V3_c,
    vector<complex<double>> &rhs1_c,
    vector<complex<double>> &rhs2_c,
    vector<complex<double>> &rhs3_c,
    vector<double> &V1_r,
    vector<double> &V2_r,
    vector<double> &V3_r,
    vector<double> &work_r1,
    vector<double> &work_r2,
    vector<double> &work_r3,
    vector<complex<double>> &rot1_c,
    vector<complex<double>> &rot2_c,
    vector<complex<double>> &rot3_c,
    vector<double> &rot1_r,
    vector<double> &rot2_r,
    vector<double> &rot3_r,
    vector<complex<double>> &nl1_c,
    vector<complex<double>> &nl2_c,
    vector<complex<double>> &nl3_c,
    vector<complex<double>> &visc1_c,
    vector<complex<double>> &visc2_c,
    vector<complex<double>> &visc3_c,
    vector<complex<double>> &f1_c,
    vector<complex<double>> &f2_c,
    vector<complex<double>> &f3_c,
    vector<complex<double>> &div_c,
    vector<complex<double>> &phi_c,
    int nx, int ny, int nz,
    double t, double dx, double dy, double dz,
    const heffte::box3d<> &inbox_r,
    const heffte::box3d<> &outbox_c)
{
    compute_nonlinear_term_spectral(fft, V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                                    V1_r, V2_r, V3_r,
                                    rot1_c, rot2_c, rot3_c,
                                    rot1_r, rot2_r, rot3_r,
                                    nx, ny, nz, inbox_r, outbox_c);

    heffte_compute_viscous_term(V1_c, visc1_c, nx, ny, nz, outbox_c);
    heffte_compute_viscous_term(V2_c, visc2_c, nx, ny, nz, outbox_c);
    heffte_compute_viscous_term(V3_c, visc3_c, nx, ny, nz, outbox_c);

    for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                size_t idx = heffte_box_index(inbox_r, i, j, k);
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;
                work_r1[idx] = func_f1(x, y, z, t);
                work_r2[idx] = func_f2(x, y, z, t);
                work_r3[idx] = func_f3(x, y, z, t);
            }
        }
    }

    fft.forward(work_r1.data(), f1_c.data(), heffte::scale::none);
    fft.forward(work_r2.data(), f2_c.data(), heffte::scale::none);
    fft.forward(work_r3.data(), f3_c.data(), heffte::scale::none);

    size_t local_size_c = outbox_c.count();
    #pragma omp parallel for
    for (size_t i = 0; i < local_size_c; ++i) {
        rhs1_c[i] = nl1_c[i] + visc1_c[i] + f1_c[i];
        rhs2_c[i] = nl2_c[i] + visc2_c[i] + f2_c[i];
        rhs3_c[i] = nl3_c[i] + visc3_c[i] + f3_c[i];
    }

    heffte_make_div_free(rhs1_c, rhs2_c, rhs3_c, div_c, phi_c, nx, ny, nz, outbox_c);
}

// RK4（谱域）推进一步
void rk4_step_spectral(
    heffte::fft3d_r2c<heffte::backend::fftw> &fft,
    vector<complex<double>> &V1_c,
    vector<complex<double>> &V2_c,
    vector<complex<double>> &V3_c,
    vector<double> &V1_r,
    vector<double> &V2_r,
    vector<double> &V3_r,
    vector<double> &work_r1,
    vector<double> &work_r2,
    vector<double> &work_r3,
    vector<complex<double>> &k1_v1,
    vector<complex<double>> &k1_v2,
    vector<complex<double>> &k1_v3,
    vector<complex<double>> &k2_v1,
    vector<complex<double>> &k2_v2,
    vector<complex<double>> &k2_v3,
    vector<complex<double>> &k3_v1,
    vector<complex<double>> &k3_v2,
    vector<complex<double>> &k3_v3,
    vector<complex<double>> &k4_v1,
    vector<complex<double>> &k4_v2,
    vector<complex<double>> &k4_v3,
    vector<complex<double>> &tmp_v1,
    vector<complex<double>> &tmp_v2,
    vector<complex<double>> &tmp_v3,
    vector<complex<double>> &rot1_c,
    vector<complex<double>> &rot2_c,
    vector<complex<double>> &rot3_c,
    vector<double> &rot1_r,
    vector<double> &rot2_r,
    vector<double> &rot3_r,
    vector<complex<double>> &nl1_c,
    vector<complex<double>> &nl2_c,
    vector<complex<double>> &nl3_c,
    vector<complex<double>> &visc1_c,
    vector<complex<double>> &visc2_c,
    vector<complex<double>> &visc3_c,
    vector<complex<double>> &f1_c,
    vector<complex<double>> &f2_c,
    vector<complex<double>> &f3_c,
    vector<complex<double>> &div_c,
    vector<complex<double>> &phi_c,
    int nx, int ny, int nz,
    const heffte::box3d<> &inbox_r,
    const heffte::box3d<> &outbox_c,
    double dt, double t,
    double dx, double dy, double dz)
{
    size_t local_size_c = outbox_c.count();

    compute_rhs_spectral(fft, V1_c, V2_c, V3_c, k1_v1, k1_v2, k1_v3,
                         V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                         rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                         nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                         f1_c, f2_c, f3_c, div_c, phi_c,
                         nx, ny, nz, t, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < local_size_c; ++i) {
        tmp_v1[i] = V1_c[i] + 0.5 * dt * k1_v1[i];
        tmp_v2[i] = V2_c[i] + 0.5 * dt * k1_v2[i];
        tmp_v3[i] = V3_c[i] + 0.5 * dt * k1_v3[i];
    }

    compute_rhs_spectral(fft, tmp_v1, tmp_v2, tmp_v3, k2_v1, k2_v2, k2_v3,
                         V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                         rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                         nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                         f1_c, f2_c, f3_c, div_c, phi_c,
                         nx, ny, nz, t + 0.5 * dt, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < local_size_c; ++i) {
        tmp_v1[i] = V1_c[i] + 0.5 * dt * k2_v1[i];
        tmp_v2[i] = V2_c[i] + 0.5 * dt * k2_v2[i];
        tmp_v3[i] = V3_c[i] + 0.5 * dt * k2_v3[i];
    }

    compute_rhs_spectral(fft, tmp_v1, tmp_v2, tmp_v3, k3_v1, k3_v2, k3_v3,
                         V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                         rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                         nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                         f1_c, f2_c, f3_c, div_c, phi_c,
                         nx, ny, nz, t + 0.5 * dt, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < local_size_c; ++i) {
        tmp_v1[i] = V1_c[i] + dt * k3_v1[i];
        tmp_v2[i] = V2_c[i] + dt * k3_v2[i];
        tmp_v3[i] = V3_c[i] + dt * k3_v3[i];
    }

    compute_rhs_spectral(fft, tmp_v1, tmp_v2, tmp_v3, k4_v1, k4_v2, k4_v3,
                         V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                         rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                         nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                         f1_c, f2_c, f3_c, div_c, phi_c,
                         nx, ny, nz, t + dt, dx, dy, dz, inbox_r, outbox_c);

    #pragma omp parallel for
    for (size_t i = 0; i < local_size_c; ++i) {
        V1_c[i] += dt / 6.0 * (k1_v1[i] + 2.0 * k2_v1[i] + 2.0 * k3_v1[i] + k4_v1[i]);
        V2_c[i] += dt / 6.0 * (k1_v2[i] + 2.0 * k2_v2[i] + 2.0 * k3_v2[i] + k4_v2[i]);
        V3_c[i] += dt / 6.0 * (k1_v3[i] + 2.0 * k2_v3[i] + 2.0 * k3_v3[i] + k4_v3[i]);
    }
}

// ==============================================================================
// heFFTe 初始化
// ==============================================================================

tuple<heffte::box3d<>, heffte::box3d<>, unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>>>
initialize_heffte(int nx, int ny, int nz, MPI_Comm comm, int rank, int nprocs) {
    heffte::box3d<> world_r = {{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
    heffte::box3d<> world_c = {{0, 0, 0}, {nx - 1, ny - 1, nz / 2}};

    auto proc_grid = heffte::proc_setup_min_surface(world_r, nprocs);
    if (rank == 0) {
        cout << "heFFTe pencil: " << proc_grid[0] << "x" << proc_grid[1] << "x" << proc_grid[2] << endl;
    }

    auto all_inboxes = heffte::split_world(world_r, proc_grid);
    auto all_outboxes = heffte::split_world(world_c, proc_grid);

    auto inbox_r = all_inboxes[rank];
    auto outbox_c = all_outboxes[rank];

    auto fft_engine = make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(inbox_r, outbox_c, 2, comm);
    return make_tuple(inbox_r, outbox_c, move(fft_engine));
}

// ==============================================================================
// 主程序
// ==============================================================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int nx = 64, ny = 64, nz = 64;
    const double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;

    // 时间步与 FFTW 版本保持一致
    const ptrdiff_t nt_total = 20000;
    const ptrdiff_t nt_run = 100;
    const double T = 1.0;
    const double dt = T / nt_total; // 5e-5

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - heFFTe Version (FFTW parity)\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz << endl;
        cout << "Domain: [0, 2π]^3\n";
        cout << "MPI processes: " << nprocs << endl;
        cout << "Total time steps: " << nt_total << ", run steps: " << nt_run << endl;
        cout << "dt: " << dt << endl;
        cout << "============================================================\n";
    }

    auto [inbox_r, outbox_c, fft_engine] = initialize_heffte(nx, ny, nz, MPI_COMM_WORLD, rank, nprocs);

    size_t local_size_r = inbox_r.count();
    size_t local_size_c = outbox_c.count();
    vector<double> V1_r(local_size_r), V2_r(local_size_r), V3_r(local_size_r);
    vector<complex<double>> V1_c(local_size_c), V2_c(local_size_c), V3_c(local_size_c);

    if (rank == 0) {
        cout << "Local real size: " << local_size_r << endl;
        cout << "Local complex size: " << local_size_c << endl;
    }

    // 初始条件 t=0
    double t = 0.0;
    for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                double x = i * Lx / nx;
                double y = j * Ly / ny;
                double z = k * Lz / nz;
                size_t idx = heffte_box_index(inbox_r, i, j, k);
                V1_r[idx] = func_V1(x, y, z, t);
                V2_r[idx] = func_V2(x, y, z, t);
                V3_r[idx] = func_V3(x, y, z, t);
            }
        }
    }

    V1_c = fft_engine->forward(V1_r, heffte::scale::none);
    V2_c = fft_engine->forward(V2_r, heffte::scale::none);
    V3_c = fft_engine->forward(V3_r, heffte::scale::none);

    if (rank == 0) cout << "Projecting initial condition to divergence-free space...\n";
    {
        vector<complex<double>> div_tmp(local_size_c), phi_tmp(local_size_c);
        heffte_make_div_free(V1_c, V2_c, V3_c, div_tmp, phi_tmp, nx, ny, nz, outbox_c);
    }

    // 工作数组
    vector<double> work_r1(local_size_r), work_r2(local_size_r), work_r3(local_size_r);
    vector<double> rot1_r(local_size_r), rot2_r(local_size_r), rot3_r(local_size_r);

    vector<complex<double>> rot1_c(local_size_c), rot2_c(local_size_c), rot3_c(local_size_c);
    vector<complex<double>> nl1_c(local_size_c), nl2_c(local_size_c), nl3_c(local_size_c);
    vector<complex<double>> visc1_c(local_size_c), visc2_c(local_size_c), visc3_c(local_size_c);
    vector<complex<double>> f1_c(local_size_c), f2_c(local_size_c), f3_c(local_size_c);
    vector<complex<double>> div_c(local_size_c), phi_c(local_size_c);

    vector<complex<double>> k1_v1(local_size_c), k1_v2(local_size_c), k1_v3(local_size_c);
    vector<complex<double>> k2_v1(local_size_c), k2_v2(local_size_c), k2_v3(local_size_c);
    vector<complex<double>> k3_v1(local_size_c), k3_v2(local_size_c), k3_v3(local_size_c);
    vector<complex<double>> k4_v1(local_size_c), k4_v2(local_size_c), k4_v3(local_size_c);
    vector<complex<double>> tmp_v1(local_size_c), tmp_v2(local_size_c), tmp_v3(local_size_c);

    auto compute_error = [&](double time) -> tuple<double, double> {
        V1_r = fft_engine->backward(V1_c, heffte::scale::full);
        V2_r = fft_engine->backward(V2_c, heffte::scale::full);
        V3_r = fft_engine->backward(V3_c, heffte::scale::full);

        double local_error = 0.0;
        double local_max = 0.0;
        for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
            for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
                for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                    size_t idx_local = heffte_box_index(inbox_r, i, j, k);
                    double x = i * Lx / nx;
                    double y = j * Ly / ny;
                    double z = k * Lz / nz;
                    double exact1 = func_V1(x, y, z, time);
                    double exact2 = func_V2(x, y, z, time);
                    double exact3 = func_V3(x, y, z, time);
                    double d1 = V1_r[idx_local] - exact1;
                    double d2 = V2_r[idx_local] - exact2;
                    double d3 = V3_r[idx_local] - exact3;
                    double err_sq = d1 * d1 + d2 * d2 + d3 * d3;
                    local_error += err_sq;
                    local_max = max(local_max, sqrt(err_sq));
                }
            }
        }

        double global_error, global_max;
        MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;
        return make_tuple(sqrt(global_error * dx * dy * dz), global_max);
    };

    auto compute_divergence = [&]() -> double {
        heffte_compute_div(V1_c, V2_c, V3_c, div_c, nx, ny, nz, outbox_c);
        double local_max = 0.0;
        for (size_t i = 0; i < local_size_c; ++i) {
            local_max = max(local_max, abs(div_c[i]));
        }
        double global_max;
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return global_max;
    };

    auto [err_L2_init, err_Linf_init] = compute_error(t);
    double div_init = compute_divergence();
    if (rank == 0) {
        cout << "\nInitial condition (t=0):\n";
        cout << "  L2 error: " << scientific << err_L2_init << endl;
        cout << "  L∞ error: " << err_Linf_init << endl;
        cout << "  max|div(V)|: " << div_init << endl;
    }

    if (rank == 0) {
        cout << "\n============================================================\n";
        cout << "  Time Integration (RK4)\n";
        cout << "============================================================\n";
        cout << setw(6) << "Step" << setw(12) << "Time"
             << setw(15) << "L2 Error" << setw(15) << "Max |div V|" << endl;
        cout << "------------------------------------------------------------------------" << endl;
    }

    double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;
    for (ptrdiff_t it = 0; it <= nt_run; ++it) {
        double t_cur = it * dt;

        auto [err_L2, err_Linf] = compute_error(t_cur);
        double div_max = compute_divergence();
        if (rank == 0) {
            cout << setw(6) << it << setw(12) << scientific << setprecision(4) << t_cur
                 << setw(15) << err_L2 << setw(15) << div_max << endl;
        }

        if (it < nt_run) {
            rk4_step_spectral(*fft_engine, V1_c, V2_c, V3_c,
                              V1_r, V2_r, V3_r, work_r1, work_r2, work_r3,
                              k1_v1, k1_v2, k1_v3, k2_v1, k2_v2, k2_v3,
                              k3_v1, k3_v2, k3_v3, k4_v1, k4_v2, k4_v3,
                              tmp_v1, tmp_v2, tmp_v3,
                              rot1_c, rot2_c, rot3_c, rot1_r, rot2_r, rot3_r,
                              nl1_c, nl2_c, nl3_c, visc1_c, visc2_c, visc3_c,
                              f1_c, f2_c, f3_c, div_c, phi_c,
                              nx, ny, nz, inbox_r, outbox_c, dt, t_cur, dx, dy, dz);
        }
    }

    fft_engine.reset();
    MPI_Finalize();
    return 0;
}
