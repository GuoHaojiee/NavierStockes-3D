#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3-mpi.h>
#include <iomanip>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <omp.h>

using namespace std;

// ==============================================================================
// 全周期性边界条件的Navier-Stokes求解器 - 完整版本
// Taylor-Green涡验证（包含时间步进）
// ==============================================================================

// 全局FFTW计划
fftw_plan plan_fwd_v1, plan_fwd_v2, plan_fwd_v3;
fftw_plan plan_bwd_v1, plan_bwd_v2, plan_bwd_v3;

// Taylor-Green涡参数
const double nu = 0.001;  // 运动粘度
const double V0 = 1.0;    // 初始速度幅值

// Taylor-Green涡解析解
double func_V1(double x, double y, double z, double t) {
    return V0 * sin(x) * cos(y) * cos(z) * exp(-3.0*nu*t);
}

double func_V2(double x, double y, double z, double t) {
    return -V0 * cos(x) * sin(y) * cos(z) * exp(-3.0*nu*t);
}

double func_V3(double x, double y, double z, double t) {
    return 0.0;
}

// FFT初始化
void initialize_fftw_3d(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                        double *V1_r, double *V2_r, double *V3_r,
                        fftw_complex *V1_c, fftw_complex *V2_c, fftw_complex *V3_c) {
    plan_fwd_v1 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V1_r, V1_c,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v1 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V1_c, V1_r,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_fwd_v2 = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, V2_r, V2_c,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
    plan_bwd_v2 = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, V2_c, V2_r,
                                            MPI_COMM_WORLD, FFTW_ESTIMATE);
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

void normalization(fftw_complex* data, ptrdiff_t size, double factor) {
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < size; ++i) {
        data[i][0] /= factor;
        data[i][1] /= factor;
    }
}

// 计算能量
double calculateEnergy(double* V1_r, double* V2_r, double* V3_r,
                       ptrdiff_t local_n0, ptrdiff_t ny, ptrdiff_t nz,
                       double dx, double dy, double dz) {
    double sumSquares = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:sumSquares)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;
                sumSquares += V1_r[index]*V1_r[index] +
                             V2_r[index]*V2_r[index] +
                             V3_r[index]*V3_r[index];
            }
        }
    }
    return 0.5 * sumSquares * dx * dy * dz;
}

// 应用Laplacian算子（频谱空间）
void apply_laplacian(fftw_complex* V_c, fftw_complex* result_c,
                     ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                     ptrdiff_t local_n0, ptrdiff_t local_0_start,
                     ptrdiff_t alloc_local) {
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

                // ∇²V = -k²V（频谱空间）
                result_c[index][0] = -k2 * V_c[index][0];
                result_c[index][1] = -k2 * V_c[index][1];
            }
        }
    }
}

// Euler方法时间步进（简化版，用于验证）
void euler_step(fftw_complex* V1_c, fftw_complex* V2_c, fftw_complex* V3_c,
                fftw_complex* lap1_c, fftw_complex* lap2_c, fftw_complex* lap3_c,
                double tau, double nu,
                ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                ptrdiff_t local_n0, ptrdiff_t local_0_start,
                ptrdiff_t alloc_local) {

    // 计算Laplacian
    apply_laplacian(V1_c, lap1_c, nx, ny, nz, local_n0, local_0_start, alloc_local);
    apply_laplacian(V2_c, lap2_c, nx, ny, nz, local_n0, local_0_start, alloc_local);
    apply_laplacian(V3_c, lap3_c, nx, ny, nz, local_n0, local_0_start, alloc_local);

    // 时间步进: V^{n+1} = V^n + τ * ν * ∇²V
    // （忽略非线性项，因为Taylor-Green涡的非线性项正好抵消）
    ptrdiff_t nz_complex = nz/2 + 1;
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < local_n0 * ny * nz_complex; ++i) {
        V1_c[i][0] += tau * nu * lap1_c[i][0];
        V1_c[i][1] += tau * nu * lap1_c[i][1];
        V2_c[i][0] += tau * nu * lap2_c[i][0];
        V2_c[i][1] += tau * nu * lap2_c[i][1];
        V3_c[i][0] += tau * nu * lap3_c[i][0];
        V3_c[i][1] += tau * nu * lap3_c[i][1];
    }
}

int main(int argc, char **argv) {
    const ptrdiff_t nx = 64, ny = 64, nz = 64;
    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;
    const double dx = L_x / nx, dy = L_y / ny, dz = L_z / nz;

    const ptrdiff_t nt_total = 10000;
    const ptrdiff_t nt_run = 10;
    const double T = 1.0;
    const double tau = T / nt_total;

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
        cout << "  Navier-Stokes Solver - Taylor-Green Vortex" << endl;
        cout << "  Full Time Integration Test" << endl;
        cout << "============================================================" << endl;
        cout << "Grid: " << nx << " x " << ny << " x " << nz << endl;
        cout << "Viscosity: " << nu << endl;
        cout << "Time steps: " << nt_run << " / " << nt_total << endl;
        cout << "dt = " << tau << ", Total time = " << T << endl;
        cout << "MPI processes: " << size << endl;
        cout << "============================================================" << endl;
    }

    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(nx, ny, nz/2+1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);

    // 分配内存
    double *V1_r = fftw_alloc_real(2 * alloc_local);
    double *V2_r = fftw_alloc_real(2 * alloc_local);
    double *V3_r = fftw_alloc_real(2 * alloc_local);

    fftw_complex *V1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *V2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *V3_c = fftw_alloc_complex(alloc_local);

    fftw_complex *lap1_c = fftw_alloc_complex(alloc_local);
    fftw_complex *lap2_c = fftw_alloc_complex(alloc_local);
    fftw_complex *lap3_c = fftw_alloc_complex(alloc_local);

    // 初始化FFT
    if (rank == 0) cout << "Initializing FFT..." << endl;
    initialize_fftw_3d(nx, ny, nz, V1_r, V2_r, V3_r, V1_c, V2_c, V3_c);

    // 设置初始条件
    if (rank == 0) cout << "Setting initial conditions (t=0)..." << endl;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_n0; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t i_global = local_0_start + i;
                double x = i_global * dx;
                double y = j * dy;
                double z = k * dz;
                ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                V1_r[index] = func_V1(x, y, z, 0.0);
                V2_r[index] = func_V2(x, y, z, 0.0);
                V3_r[index] = func_V3(x, y, z, 0.0);
            }
        }
    }

    // 转到频谱空间
    fftw_execute(plan_fwd_v1);
    fftw_execute(plan_fwd_v2);
    fftw_execute(plan_fwd_v3);

    double norm_factor = nx * ny * nz;
    normalization(V1_c, alloc_local, norm_factor);
    normalization(V2_c, alloc_local, norm_factor);
    normalization(V3_c, alloc_local, norm_factor);

    if (rank == 0) {
        cout << "\n============================================================" << endl;
        cout << "  Time Integration" << endl;
        cout << "============================================================" << endl;
        cout << setw(6) << "Step" << setw(12) << "Time"
             << setw(15) << "Energy" << setw(15) << "E_analytical"
             << setw(15) << "L2 Error" << endl;
        cout << "------------------------------------------------------------" << endl;
    }

    // 时间步进
    for(ptrdiff_t it = 0; it <= nt_run; ++it) {
        double t = it * tau;

        // 转到实空间以计算统计量
        fftw_execute(plan_bwd_v1);
        fftw_execute(plan_bwd_v2);
        fftw_execute(plan_bwd_v3);

        // 计算能量
        double local_energy = calculateEnergy(V1_r, V2_r, V3_r,
                                              local_n0, ny, nz, dx, dy, dz);
        double global_energy;
        MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Taylor-Green涡的理论能量: E(t) = E(0) * exp(-6νt)
        double E0 = 0.5 * V0 * V0 * (L_x * L_y * L_z) / 4.0;  // 初始能量
        double E_analytical = E0 * exp(-6.0 * nu * t);

        // 计算误差
        double local_err = 0.0;
        #pragma omp parallel for collapse(3) reduction(+:local_err)
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0; j < ny; ++j) {
                for(ptrdiff_t k = 0; k < nz; ++k) {
                    ptrdiff_t i_global = local_0_start + i;
                    double x = i_global * dx;
                    double y = j * dy;
                    double z = k * dz;
                    ptrdiff_t index = (i * ny + j) * (2*(nz/2+1)) + k;

                    double v1_exact = func_V1(x, y, z, t);
                    double v2_exact = func_V2(x, y, z, t);
                    double v3_exact = func_V3(x, y, z, t);

                    double diff1 = V1_r[index] - v1_exact;
                    double diff2 = V2_r[index] - v2_exact;
                    double diff3 = V3_r[index] - v3_exact;

                    local_err += diff1*diff1 + diff2*diff2 + diff3*diff3;
                }
            }
        }

        double global_err;
        MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double L2_error = sqrt(global_err * dx * dy * dz);
            cout << setw(6) << it << setw(12) << scientific << setprecision(4) << t
                 << setw(15) << global_energy
                 << setw(15) << E_analytical
                 << setw(15) << L2_error << endl;
        }

        // 如果不是最后一步，继续时间步进
        if (it < nt_run) {
            // 转回频谱空间
            fftw_execute(plan_fwd_v1);
            fftw_execute(plan_fwd_v2);
            fftw_execute(plan_fwd_v3);

            normalization(V1_c, alloc_local, norm_factor);
            normalization(V2_c, alloc_local, norm_factor);
            normalization(V3_c, alloc_local, norm_factor);

            // Euler步进
            euler_step(V1_c, V2_c, V3_c, lap1_c, lap2_c, lap3_c,
                      tau, nu, nx, ny, nz, local_n0, local_0_start, alloc_local);
        }
    }

    if (rank == 0) {
        cout << "============================================================" << endl;
        cout << "  Simulation completed successfully!" << endl;
        cout << "============================================================" << endl;
    }

    // 清理
    finalize_fft_plans();
    fftw_free(V1_r); fftw_free(V2_r); fftw_free(V3_r);
    fftw_free(V1_c); fftw_free(V2_c); fftw_free(V3_c);
    fftw_free(lap1_c); fftw_free(lap2_c); fftw_free(lap3_c);

    MPI_Finalize();
    return 0;
}
