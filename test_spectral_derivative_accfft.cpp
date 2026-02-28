// ==============================================================================
// 验证 AccFFT 3D R2C/C2R 频谱导数
// 测试函数: u(x,y,z) = sin(2x) * cos(3y) * sin(4z)
//
// AccFFT 与 FFTW MPI 的关键区别：
//   - 用 accfft_create_comm + accfft_plan_dft_3d_r2c 替代 fftw_mpi_plan_dft_r2c_3d
//   - Complex = double[2]，访问方式 A[idx][0]/A[idx][1]
//   - 实空间索引步长用全局 n[2]：ptr = i*isize[1]*n[2] + j*n[2] + k
//   - 频谱空间索引：ptr = i*osize[1]*osize[2] + j*osize[2] + k
//   - 全局波数：gi=i+ostart[0], gj=j+ostart[1], gk=k+ostart[2]
//   - kx/ky 需要折叠处理；kz（R2C，0..nz/2）直接使用
//   - Nyquist 模（kx 或 ky == N/2）置零（与 AccFFT operators 保持一致）
//   - 逆变换不自动归一化，需手动除以 N = nx*ny*nz
// ==============================================================================

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <accfft.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nx = 64, ny = 64, nz = 64;
    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    const double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;
    const double N_total = (double)nx * ny * nz;   // 归一化因子

    int n[3] = {nx, ny, nz};

    // ===========================================================================
    // 1. 创建 2D Cartesian 通信子（AccFFT 2D pencil 分解）
    // ===========================================================================
    int c_dims[2] = {0, 0};   // {0,0} 让 AccFFT 自动选择最优进程网格
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

    // ===========================================================================
    // 2. 查询本地大小
    //    isize/istart: 实空间本地网格大小和全局起始索引
    //    osize/ostart: 频谱空间本地网格大小和全局起始索引
    //    alloc_max:    最大分配字节数（取实空间与频谱空间所需的较大值）
    // ===========================================================================
    int isize[3], istart[3], osize[3], ostart[3];
    int alloc_max = accfft_local_size_dft_r2c(n, isize, istart, osize, ostart, c_comm);

    // ===========================================================================
    // 3. 分配对齐内存
    //    实空间：步长用全局 n[2]（AccFFT 约定），大小 isize[0]*isize[1]*n[2]
    //    频谱空间：大小 osize[0]*osize[1]*osize[2] Complex
    // ===========================================================================
    long r_size = (long)isize[0] * isize[1] * n[2];
    long c_size = (long)osize[0] * osize[1] * osize[2];

    double*  u_r  = (double*)  accfft_alloc(r_size * sizeof(double));
    Complex* u_c  = (Complex*) accfft_alloc(alloc_max);
    Complex* du_c = (Complex*) accfft_alloc(alloc_max);
    double*  du_r = (double*)  accfft_alloc(r_size * sizeof(double));

    // ===========================================================================
    // 4. 初始化 AccFFT（设置 OpenMP 线程数，这里用单线程）
    // ===========================================================================
    accfft_init(1);

    // ===========================================================================
    // 5. 创建 FFT 计划（ACCFFT_MEASURE 会在几种算法中选择最快的）
    // ===========================================================================
    accfft_plan* plan = accfft_plan_dft_3d_r2c(n, u_r, (double*)u_c,
                                                c_comm, ACCFFT_MEASURE);

    // ===========================================================================
    // 6. 填充实空间数据 u(x,y,z) = sin(2x)*cos(3y)*sin(4z)
    //    实空间索引格式：ptr = i*isize[1]*n[2] + j*n[2] + k
    //    全局坐标：x = (i + istart[0])*dx, y = (j + istart[1])*dy, z = k*dz
    // ===========================================================================
    for (int i = 0; i < isize[0]; ++i) {
        for (int j = 0; j < isize[1]; ++j) {
            for (int k = 0; k < n[2]; ++k) {
                double x = (i + istart[0]) * dx;
                double y = (j + istart[1]) * dy;
                double z = k * dz;
                long ptr = (long)i * isize[1] * n[2] + j * n[2] + k;
                u_r[ptr] = sin(2*x) * cos(3*y) * sin(4*z);
            }
        }
    }

    // ===========================================================================
    // 7. 正变换：实空间 -> 频谱空间
    // ===========================================================================
    accfft_execute_r2c(plan, u_r, u_c);

    // ===========================================================================
    // 8. 打印标题
    // ===========================================================================
    if (rank == 0) {
        printf("=====================================================\n");
        printf(" AccFFT 3D R2C/C2R 频谱导数验证\n");
        printf(" 函数: u = sin(2x)*cos(3y)*sin(4z)\n");
        printf(" 区域: [0,2π]³,  网格: %dx%dx%d,  进程: %d\n",
               nx, ny, nz, size);
        printf("=====================================================\n");
        printf("%-8s  %-16s  %-16s\n", "方向", "L2误差", "最大误差");
        printf("-----------------------------------------------------\n");
    }

    // ===========================================================================
    // 9. 依次测试 d/dx, d/dy, d/dz
    // ===========================================================================
    const char *names[] = {"d/dx", "d/dy", "d/dz"};

    for (int dir = 0; dir < 3; dir++) {

        // --- 9a. 复制 u_c -> du_c，避免修改原始频谱数据
        memcpy(du_c, u_c, c_size * sizeof(Complex));

        // --- 9b. 频谱空间乘以 i*k（导数算子）
        //
        // 波数约定（与 AccFFT operators.txx 保持一致）：
        //   kx: gi = i + ostart[0]，折叠：kx = gi（gi <= nx/2）或 gi-nx（gi > nx/2）
        //       Nyquist（gi == nx/2）置零
        //   ky: gj = j + ostart[1]，同上
        //   kz: gk = k + ostart[2]，R2C 只存非负，直接使用
        //
        // 复数乘法：i*wave*(a+ib) = -wave*b + i*wave*a
        //
        for (int i = 0; i < osize[0]; ++i) {
            for (int j = 0; j < osize[1]; ++j) {
                for (int k = 0; k < osize[2]; ++k) {
                    long ptr = (long)i * osize[1] * osize[2] + j * osize[2] + k;

                    int gi = i + ostart[0];
                    int gj = j + ostart[1];
                    int gk = k + ostart[2];

                    double kx = (gi <= nx/2) ? (double)gi : (double)(gi - nx);
                    double ky = (gj <= ny/2) ? (double)gj : (double)(gj - ny);
                    double kz = (double)gk;

                    // Nyquist 模置零（与 AccFFT operators.txx 一致）
                    if (gi == nx/2) kx = 0.0;
                    if (gj == ny/2) ky = 0.0;

                    double wave;
                    if      (dir == 0) wave = kx;
                    else if (dir == 1) wave = ky;
                    else               wave = kz;

                    double a = du_c[ptr][0];
                    double b = du_c[ptr][1];
                    du_c[ptr][0] = -wave * b;
                    du_c[ptr][1] =  wave * a;
                }
            }
        }

        // --- 9c. 逆变换：AccFFT 不自动归一化，需手动除以 N_total
        accfft_execute_c2r(plan, du_c, du_r);

        // --- 9d. 与精确导数比较
        double local_l2 = 0.0, local_max = 0.0;
        for (int i = 0; i < isize[0]; ++i) {
            for (int j = 0; j < isize[1]; ++j) {
                for (int k = 0; k < n[2]; ++k) {
                    double x = (i + istart[0]) * dx;
                    double y = (j + istart[1]) * dy;
                    double z = k * dz;
                    long ptr = (long)i * isize[1] * n[2] + j * n[2] + k;

                    double du_num = du_r[ptr] / N_total;   // 手动归一化

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

    // ===========================================================================
    // 10. 清理
    // ===========================================================================
    accfft_free(u_r);
    accfft_free(u_c);
    accfft_free(du_c);
    accfft_free(du_r);
    accfft_destroy_plan(plan);
    accfft_cleanup();
    MPI_Comm_free(&c_comm);

    MPI_Finalize();
    return 0;
}
