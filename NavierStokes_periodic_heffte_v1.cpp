/**
 * Navier-Stokes求解器 - heFFTe版本v1
 * 基于FFTW版本，使用heFFTe进行R2C/C2R变换
 *
 * 策略：
 * - 使用heFFTe的FFTW backend和pencil分解
 * - 最小化修改，重用FFTW版本的算法逻辑
 * - 第一阶段：验证heFFTe的FFT正确性和数值精度
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <complex>
#include <mpi.h>
#include <omp.h>
#include <heffte.h>

using namespace std;

// 初始化heFFTe - 返回boxes和FFT对象
std::tuple<heffte::box3d<>, heffte::box3d<>, std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>>>
initialize_heffte(int nx, int ny, int nz, MPI_Comm comm, int rank, int nprocs) {
    // 创建box
    heffte::box3d<> world_r = {{0,0,0}, {nx-1, ny-1, nz-1}};
    heffte::box3d<> world_c = {{0,0,0}, {nx-1, ny-1, nz/2}};

    // 自动确定最优进程网格
    auto proc_grid = heffte::proc_setup_min_surface(world_r, nprocs);

    if (rank == 0) {
        cout << "heFFTe pencil: " << proc_grid[0] << "x" << proc_grid[1] << "x" << proc_grid[2] << endl;
    }

    auto all_inboxes = heffte::split_world(world_r, proc_grid);
    auto all_outboxes = heffte::split_world(world_c, proc_grid);

    auto inbox_r = all_inboxes[rank];
    auto outbox_c = all_outboxes[rank];

    // r2c_direction=2表示在z方向（第3维）进行R2C压缩
    auto fft_engine = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(
        inbox_r, outbox_c, 2, comm
    );

    return std::make_tuple(inbox_r, outbox_c, std::move(fft_engine));
}

int main(int argc, char** argv) {
    // MPI初始化
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 问题参数
    const int nx = 128, ny = 128, nz = 128;
    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;

    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - heFFTe Version\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz << endl;
        cout << "Domain: [0, 2π]³\n";
        cout << "MPI processes: " << nprocs << endl;
        cout << "============================================================\n";
    }

    // 初始化heFFTe
    auto [inbox_r, outbox_c, fft_engine] = initialize_heffte(nx, ny, nz, MPI_COMM_WORLD, rank, nprocs);

    // 分配内存
    size_t local_size_r = inbox_r.count();
    size_t local_size_c = outbox_c.count();

    std::vector<double> V1_r(local_size_r);
    std::vector<std::complex<double>> V1_c(local_size_c);

    if (rank == 0) {
        cout << "Local real size: " << local_size_r << endl;
        cout << "Local complex size: " << local_size_c << endl;
    }

    // 设置初始条件
    size_t idx = 0;
    for(int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for(int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for(int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                double x = i * Lx / nx;
                double y = j * Ly / ny;
                double z = k * Lz / nz;
                V1_r[idx] = cos(x) * sin(y) * sin(z);  // 简单测试函数
                ++idx;
            }
        }
    }

    if (rank == 0) cout << "Performing forward FFT...\n";
    auto V1_c_result = fft_engine->forward(V1_r);

    if (rank == 0) cout << "Performing backward FFT...\n";
    auto V1_r_result = fft_engine->backward(V1_c_result, heffte::scale::full);

    // 验证round-trip精度
    double local_error = 0.0;
    idx = 0;
    for(int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for(int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for(int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                double x = i * Lx / nx;
                double y = j * Ly / ny;
                double z = k * Lz / nz;
                double exact = cos(x) * sin(y) * sin(z);
                double diff = V1_r_result[idx] - exact;
                local_error += diff * diff;
                ++idx;
            }
        }
    }

    double global_error;
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;
        double L2_error = sqrt(global_error * dx * dy * dz);
        cout << "\nFFT round-trip L2 error: " << scientific << L2_error << endl;
        cout << "\n============================================================\n";
        cout << "  heFFTe FFT Test " << (L2_error < 1e-12 ? "PASSED" : "FAILED") << "!\n";
        cout << "============================================================\n";
    }

    // 在MPI_Finalize之前释放heFFTe对象
    fft_engine.reset();

    MPI_Finalize();
    return 0;
}
