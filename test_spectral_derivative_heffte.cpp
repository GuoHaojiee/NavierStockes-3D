// ==============================================================================
// 验证 heFFTe 3D R2C/C2R 频谱导数
// 测试函数: u(x,y,z) = sin(2x) * cos(3y) * sin(4z)
//
// heFFTe 与 FFTW MPI 的关键区别：
//   - 用 box3d<> + fft3d_r2c 替代 fftw_mpi_plan_dft_r2c_3d
//   - backward(scale::full) 自动除以 N，无需手动归一化
//   - box3d 下标从 0 开始，索引格式: x 最快变化（与 FFTW MPI slab 相同）
//   - R2C 在 z 方向 (dim=2)，kz=0..nz/2 非负；kx/ky 需要折叠处理
// ==============================================================================

#include <cmath>
#include <cstdio>
#include <complex>
#include <vector>
#include <array>
#include <heffte.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {  // 作用域块：确保 heFFTe 对象在 MPI_Finalize 之前析构

    const int nx = 64, ny = 64, nz = 64;
    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    const double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    // ===========================================================================
    // 1. 定义全局 box
    //    world_r: 实空间全网格 [0..nx-1] x [0..ny-1] x [0..nz-1]
    //    world_c: 频谱空间，z 方向 R2C 压缩为 nz/2+1 → [0..nz/2]
    // ===========================================================================
    heffte::box3d<> world_r = {{0, 0, 0}, {nx-1, ny-1, nz-1}};
    heffte::box3d<> world_c = {{0, 0, 0}, {nx-1, ny-1, nz/2}};

    // 让 heFFTe 自动分配 MPI 进程网格
    std::array<int,3> proc_grid = heffte::proc_setup_min_surface(world_r, size);
    auto all_inboxes  = heffte::split_world(world_r, proc_grid);
    auto all_outboxes = heffte::split_world(world_c, proc_grid);

    heffte::box3d<> inbox_r  = all_inboxes[rank];
    heffte::box3d<> outbox_c = all_outboxes[rank];

    // 创建 FFT 引擎：第三维 (dim=2, 即 z) 做 R2C
    heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox_r, outbox_c, 2, MPI_COMM_WORLD);

    size_t local_r_size = inbox_r.count();
    size_t local_c_size = outbox_c.count();

    // ===========================================================================
    // 2. 分配内存
    //    u_r / du_r: 实空间，大小 = inbox_r 中的点数
    //    u_c / du_c: 频谱空间，大小 = outbox_c 中的点数
    // ===========================================================================
    std::vector<double>               u_r(local_r_size);
    std::vector<std::complex<double>> u_c(local_c_size);
    std::vector<std::complex<double>> du_c(local_c_size);
    std::vector<double>               du_r(local_r_size);

    // ===========================================================================
    // 3. 填充实空间数据 u(x,y,z) = sin(2x)*cos(3y)*sin(4z)
    //    heFFTe 索引格式（C-order，x 最快）:
    //    idx = (i-low[0]) + (j-low[1])*size[0] + (k-low[2])*size[0]*size[1]
    // ===========================================================================
    for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
        for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
            for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                double x = i * dx, y = j * dy, z = k * dz;
                int li = i - inbox_r.low[0];
                int lj = j - inbox_r.low[1];
                int lk = k - inbox_r.low[2];
                size_t idx = li + lj * inbox_r.size[0]
                              + lk * inbox_r.size[0] * inbox_r.size[1];
                u_r[idx] = sin(2*x) * cos(3*y) * sin(4*z);
            }
        }
    }

    // ===========================================================================
    // 4. 正变换：实空间 -> 频谱空间（不做归一化）
    // ===========================================================================
    fft.forward(u_r.data(), u_c.data(), heffte::scale::none);

    // ===========================================================================
    // 5. 打印标题
    // ===========================================================================
    if (rank == 0) {
        printf("=====================================================\n");
        printf(" heFFTe 3D R2C/C2R 频谱导数验证\n");
        printf(" 函数: u = sin(2x)*cos(3y)*sin(4z)\n");
        printf(" 区域: [0,2π]³,  网格: %dx%dx%d,  进程: %d\n",
               nx, ny, nz, size);
        printf("=====================================================\n");
        printf("%-8s  %-16s  %-16s\n", "方向", "L2误差", "最大误差");
        printf("-----------------------------------------------------\n");
    }

    // ===========================================================================
    // 6. 依次测试 d/dx, d/dy, d/dz
    // ===========================================================================
    const char *names[] = {"d/dx", "d/dy", "d/dz"};

    for (int dir = 0; dir < 3; dir++) {

        // --- 6a. 复制 u_c 到 du_c（backward 不破坏输入，但每次循环要从 u_c 重新开始）
        du_c = u_c;

        // --- 6b. 频谱空间乘以 i*k（导数算子）
        //
        // 波数约定（与 FFTW MPI R2C 完全一致）：
        //   kx: i 是 [0..nx-1]，需折叠：kx = i (i<=nx/2) 或 i-nx (i>nx/2)
        //   ky: j 是 [0..ny-1]，需折叠：ky = j (j<=ny/2) 或 j-ny (j>ny/2)
        //   kz: k 是 [0..nz/2]，R2C 只存非负，kz = k
        //
        // 复数乘法：i*wave*(a+ib) = -wave*b + i*wave*a
        //
        for (int i = outbox_c.low[0]; i <= outbox_c.high[0]; ++i) {
            for (int j = outbox_c.low[1]; j <= outbox_c.high[1]; ++j) {
                for (int k = outbox_c.low[2]; k <= outbox_c.high[2]; ++k) {
                    int li = i - outbox_c.low[0];
                    int lj = j - outbox_c.low[1];
                    int lk = k - outbox_c.low[2];
                    size_t idx = li + lj * outbox_c.size[0]
                                  + lk * outbox_c.size[0] * outbox_c.size[1];

                    double kx = (i <= nx/2) ? (double)i : (double)(i - nx);
                    double ky = (j <= ny/2) ? (double)j : (double)(j - ny);
                    double kz = (double)k;  // R2C: 只存 0..nz/2，直接使用

                    double wave;
                    if      (dir == 0) wave = kx;
                    else if (dir == 1) wave = ky;
                    else               wave = kz;

                    double a = du_c[idx].real();
                    double b = du_c[idx].imag();
                    du_c[idx] = {-wave * b, wave * a};
                }
            }
        }

        // --- 6c. 逆变换：heffte::scale::full 自动除以 N=nx*ny*nz
        fft.backward(du_c.data(), du_r.data(), heffte::scale::full);

        // --- 6d. 与精确导数比较
        double local_l2 = 0.0, local_max = 0.0;
        for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i) {
            for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j) {
                for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k) {
                    double x = i * dx, y = j * dy, z = k * dz;
                    int li = i - inbox_r.low[0];
                    int lj = j - inbox_r.low[1];
                    int lk = k - inbox_r.low[2];
                    size_t idx = li + lj * inbox_r.size[0]
                                  + lk * inbox_r.size[0] * inbox_r.size[1];

                    double du_num = du_r[idx];  // scale::full 已归一化

                    double du_exact;
                    if      (dir == 0) du_exact =  2.0 * cos(2*x) * cos(3*y) * sin(4*z);
                    else if (dir == 1) du_exact = -3.0 * sin(2*x) * sin(3*y) * sin(4*z);
                    else               du_exact =  4.0 * sin(2*x) * cos(3*y) * cos(4*z);

                    double err = fabs(du_num - du_exact);
                    local_l2  += err * err;
                    local_max  = (err > local_max) ? err : local_max;
                }
            }
        }

        double global_l2, global_max;
        MPI_Reduce(&local_l2,  &global_l2,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double l2_err = sqrt(global_l2 * dx * dy * dz);
            printf("%-8s  %-16.3e  %-16.3e\n", names[dir], l2_err, global_max);
        }
    }

    if (rank == 0) {
        printf("=====================================================\n");
        printf("误差在机器精度 (~1e-13) 级别即为正确\n");
    }

    }  // heFFTe 对象在此析构，MPI 通信子被正确释放

    MPI_Finalize();
    return 0;
}
