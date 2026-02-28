// ==============================================================================
// 验证 FFTW MPI 3D R2C/C2R 频谱导数
// 测试函数: u(x,y,z) = sin(2x) * cos(3y) * sin(4z)
// 精确导数:
//   du/dx = 2*cos(2x)*cos(3y)*sin(4z)
//   du/dy = -3*sin(2x)*sin(3y)*sin(4z)
//   du/dz = 4*sin(2x)*cos(3y)*cos(4z)
// ==============================================================================

#include <cmath>
#include <cstdio>
#include <fftw3-mpi.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 网格参数，区域 [0, 2π]³
    const ptrdiff_t nx = 64, ny = 64, nz = 64;
    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    const double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    // ===========================================================================
    // 1. 获取 MPI 本地分块大小
    //    FFTW MPI 沿 x 方向做 slab 分解
    //    每个进程分到 local_n0 个 x 切片，从 local_0_start 开始
    // ===========================================================================
    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(nx, ny, nz/2+1,
                                          MPI_COMM_WORLD,
                                          &local_n0, &local_0_start);

    // ===========================================================================
    // 2. 分配内存
    //    u_r : 实空间，z 方向有 padding，大小 local_n0 × ny × 2*(nz/2+1)
    //    u_c : 频谱空间，大小 local_n0 × ny × (nz/2+1)
    //    du_c: 频谱空间导数（临时），C2R 执行后会被销毁
    //    du_r: 实空间导数结果
    // ===========================================================================
    double       *u_r  = fftw_alloc_real(2 * alloc_local);
    fftw_complex *u_c  = fftw_alloc_complex(alloc_local);
    fftw_complex *du_c = fftw_alloc_complex(alloc_local);
    double       *du_r = fftw_alloc_real(2 * alloc_local);

    // ===========================================================================
    // 3. 创建 FFT 计划
    //    R2C: u_r  -> u_c   (正变换)
    //    C2R: du_c -> du_r  (逆变换，会销毁 du_c 的内容！)
    // ===========================================================================
    fftw_plan plan_r2c = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, u_r, u_c,
                                                    MPI_COMM_WORLD, FFTW_ESTIMATE);
    fftw_plan plan_c2r = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, du_c, du_r,
                                                    MPI_COMM_WORLD, FFTW_ESTIMATE);

    // ===========================================================================
    // 4. 填充实空间数据 u(x,y,z) = sin(2x)*cos(3y)*sin(4z)
    //    注意 z 方向的 padding：实际步长是 2*(nz/2+1)，不是 nz
    // ===========================================================================
    const ptrdiff_t nz_pad = 2*(nz/2+1);  // z 方向 padded 步长
    const ptrdiff_t nz_c   = nz/2+1;      // z 方向复数点数

    for (ptrdiff_t i = 0; i < local_n0; i++) {
        for (ptrdiff_t j = 0; j < ny; j++) {
            for (ptrdiff_t k = 0; k < nz; k++) {
                ptrdiff_t i_global = local_0_start + i;
                double x = i_global * dx;
                double y = j * dy;
                double z = k * dz;
                ptrdiff_t idx = (i * ny + j) * nz_pad + k;
                u_r[idx] = sin(2*x) * cos(3*y) * sin(4*z);
            }
        }
    }

    // ===========================================================================
    // 5. 正变换: 实空间 -> 频谱空间
    // ===========================================================================
    fftw_execute(plan_r2c);

    // ===========================================================================
    // 6. 依次测试 x、y、z 三个方向的导数
    // ===========================================================================
    if (rank == 0) {
        printf("=====================================================\n");
        printf(" FFTW MPI 3D R2C/C2R 频谱导数验证\n");
        printf(" 函数: u = sin(2x)*cos(3y)*sin(4z)\n");
        printf(" 区域: [0,2π]³,  网格: %td×%td×%td,  进程: %d\n",
               nx, ny, nz, size);
        printf("=====================================================\n");
        printf("%-8s  %-16s  %-16s\n", "方向", "L2误差", "最大误差");
        printf("-----------------------------------------------------\n");
    }

    const double norm = (double)(nx * ny * nz);

    for (int dir = 0; dir < 3; dir++) {

        // --- 6a. 将 u_c 复制到 du_c ---
        // 必须复制，因为 C2R 执行后 du_c 内容会被销毁
        for (ptrdiff_t n = 0; n < alloc_local; n++) {
            du_c[n][0] = u_c[n][0];
            du_c[n][1] = u_c[n][1];
        }

        // --- 6b. 频谱空间中乘以导数算子 i*k ---
        //
        // 波数约定（由 FFTW R2C 布局决定）:
        //   kx: 数组下标 0..nx-1 对应物理波数 0,1,..,nx/2,-nx/2+1,..,-1
        //       即 kx = i         (i <= nx/2)
        //          kx = i - nx    (i > nx/2)
        //   ky: 同 kx 的规则
        //   kz: R2C 只存 0..nz/2，全为非负，kz = k 直接使用
        //
        // 导数算子: (∂u/∂x)^ = i*kx * û
        // 复数乘法: i*wave*(a+ib) = -wave*b + i*wave*a
        //
        for (ptrdiff_t i = 0; i < local_n0; i++) {
            for (ptrdiff_t j = 0; j < ny; j++) {
                for (ptrdiff_t k = 0; k < nz_c; k++) {
                    ptrdiff_t i_global = local_0_start + i;
                    ptrdiff_t idx = (i * ny + j) * nz_c + k;

                    double kx = (i_global <= nx/2) ? (double)i_global
                                                   : (double)(i_global - nx);
                    double ky = (j        <= ny/2) ? (double)j
                                                   : (double)(j - ny);
                    double kz = (double)k;  // R2C: 只存非负频率，直接用

                    double wave;
                    if      (dir == 0) wave = kx;
                    else if (dir == 1) wave = ky;
                    else               wave = kz;

                    double a = du_c[idx][0];
                    double b = du_c[idx][1];
                    du_c[idx][0] = -wave * b;   // Re(i*wave*(a+ib)) = -wave*b
                    du_c[idx][1] =  wave * a;   // Im(i*wave*(a+ib)) =  wave*a
                }
            }
        }

        // --- 6c. 逆变换: 频谱 -> 实空间 (FFTW 不自带归一化) ---
        fftw_execute(plan_c2r);

        // --- 6d. 与精确导数比较 ---
        double local_l2  = 0.0;
        double local_max = 0.0;

        for (ptrdiff_t i = 0; i < local_n0; i++) {
            for (ptrdiff_t j = 0; j < ny; j++) {
                for (ptrdiff_t k = 0; k < nz; k++) {
                    ptrdiff_t i_global = local_0_start + i;
                    double x = i_global * dx;
                    double y = j * dy;
                    double z = k * dz;
                    ptrdiff_t idx = (i * ny + j) * nz_pad + k;

                    // FFTW C2R 未归一化，除以 N = nx*ny*nz
                    double du_num = du_r[idx] / norm;

                    double du_exact;
                    if      (dir == 0) du_exact =  2 * cos(2*x) * cos(3*y) * sin(4*z);
                    else if (dir == 1) du_exact = -3 * sin(2*x) * sin(3*y) * sin(4*z);
                    else               du_exact =  4 * sin(2*x) * cos(3*y) * cos(4*z);

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
            const char *names[] = {"d/dx", "d/dy", "d/dz"};
            printf("%-8s  %-16.3e  %-16.3e\n", names[dir], l2_err, global_max);
        }
    }

    if (rank == 0) {
        printf("=====================================================\n");
        printf("误差在机器精度 (~1e-13) 级别即为正确\n");
    }

    // 清理
    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(u_r);
    fftw_free(u_c);
    fftw_free(du_c);
    fftw_free(du_r);

    fftw_mpi_cleanup();
    MPI_Finalize();
    return 0;
}
