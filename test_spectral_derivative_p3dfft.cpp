// ==============================================================================
// 验证 p3dfft 3D R2C/C2R 频谱导数
// 测试函数: u(x,y,z) = sin(2x) * cos(3y) * sin(4z)
//
// p3dfft 与 FFTW MPI 的关键区别：
//   ★ R2C 压缩的是 x 方向（第一维），不是 z 方向！
//      FFTW/heFFTe: kz 非负 (0..nz/2)，kx/ky 需折叠
//      p3dfft:      kx 非负 (0..nx/2)，ky/kz 需折叠
//   ★ 使用 1-based 全局索引 (Fortran 风格)，idx_offset = 1
//   ★ 数据按 Fortran 列主序存储，x 变化最快
//   ★ 逆变换不做归一化，需手动除以 N=nx*ny*nz
// ==============================================================================

#include <cmath>
#include <cstdio>
#include <complex>
#include <vector>
#include <mpi.h>
#include <p3dfft.h>

// 折叠波数：将 1-based 下标转为有符号波数（0-based 逻辑）
inline double k_wrap(int idx_1based, int offset, int n) {
    int k = idx_1based - offset;  // 转为 0-based：0..n-1
    return (k <= n/2) ? (double)k : (double)(k - n);
}

// 实空间线性索引（x 最快，Fortran 列主序，1-based 输入）
inline size_t idx_r(const int istart[3], const int isize[3],
                    int i, int j, int k) {
    return (size_t)(i - istart[0])
         + (size_t)(j - istart[1]) * isize[0]
         + (size_t)(k - istart[2]) * isize[0] * isize[1];
}

// 频谱空间线性索引（x 最快，Fortran 列主序，1-based 输入）
inline size_t idx_c(const int fstart[3], const int fsize[3],
                    int i, int j, int k) {
    return (size_t)(i - fstart[0])
         + (size_t)(j - fstart[1]) * fsize[0]
         + (size_t)(k - fstart[2]) * fsize[0] * fsize[1];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nx = 64, ny = 64, nz = 64;
    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    const double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;
    const double norm = (double)(nx * ny * nz);

    // ===========================================================================
    // 1. 初始化 p3dfft
    //    dims[2]: 2D 进程网格（p3dfft 自行分割 y 和 z 方向）
    //    memsize[3]: 返回所需内存大小（含 padding）
    // ===========================================================================
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int memsize[3];
    // 参数: dims, nx, ny, nz, MPI_Comm (Fortran handle), nx, ny, nz, OW=1, memsize
    // OW=1: 允许 btran_c2r 覆盖（销毁）输入数组
    Cp3dfft_setup(dims, nx, ny, nz,
                  MPI_Comm_c2f(MPI_COMM_WORLD),
                  nx, ny, nz, 1, memsize);

    // 获取本地数据范围（1-based 全局索引）
    // conf=1: 实空间 (X-pencil)，conf=2: 频谱空间 (Z-pencil)
    int istart[3], iend[3], isize[3];  // 实空间
    int fstart[3], fend[3], fsize[3];  // 频谱空间
    int conf = 1;
    Cp3dfft_get_dims(istart, iend, isize, conf);
    conf = 2;
    Cp3dfft_get_dims(fstart, fend, fsize, conf);

    // p3dfft 使用 1-based 索引，offset = istart[0] - 1（通常 = 0 或 1）
    // 若 istart[0] == 1，则 offset = 1（1-based）
    const int offset = istart[0];  // 1-based offset

    // ===========================================================================
    // 2. 分配内存
    //    实空间: isize[0]*isize[1]*memsize[2]（memsize 包含 R2C 所需 padding）
    //    频谱空间: fsize[0]*fsize[1]*fsize[2] 个复数
    // ===========================================================================
    size_t local_r_size = (size_t)isize[0] * isize[1] * memsize[2];
    size_t local_c_size = (size_t)fsize[0] * fsize[1] * fsize[2];

    std::vector<double>               u_r(local_r_size, 0.0);
    std::vector<std::complex<double>> u_c(local_c_size);
    std::vector<std::complex<double>> du_c(local_c_size);
    std::vector<double>               du_r(local_r_size, 0.0);

    // ===========================================================================
    // 3. 填充实空间数据 u(x,y,z) = sin(2x)*cos(3y)*sin(4z)
    //    p3dfft 坐标：x = (i - offset) * dx，其中 i 是 1-based
    //    p3dfft 实空间采用 X-pencil（x 变化最快），Fortran 列主序
    // ===========================================================================
    for (int i = istart[0]; i <= iend[0]; ++i) {
        for (int j = istart[1]; j <= iend[1]; ++j) {
            for (int k = istart[2]; k <= iend[2]; ++k) {
                double x = (i - offset) * dx;
                double y = (j - offset) * dy;
                double z = (k - offset) * dz;
                size_t idx = idx_r(istart, isize, i, j, k);
                u_r[idx] = sin(2*x) * cos(3*y) * sin(4*z);
            }
        }
    }

    // ===========================================================================
    // 4. 正变换：实空间 -> 频谱空间
    //    op = "fft": 所有三个维度做 FFT（"t" 表示 transpose，"f" 表示 FFT）
    //    "fft" = z 方向: "f"，y 方向: "f"，x 方向: "f" (最后的那个是 R2C)
    //    实际上 op_f = "fft" 表示全 FFT，R2C 在 x 方向
    // ===========================================================================
    unsigned char op_f[] = "fft";
    Cp3dfft_ftran_r2c(u_r.data(),
                      reinterpret_cast<double*>(u_c.data()),
                      op_f);

    // ===========================================================================
    // 5. 打印标题
    // ===========================================================================
    if (rank == 0) {
        printf("=====================================================\n");
        printf(" p3dfft 3D R2C/C2R 频谱导数验证\n");
        printf(" 函数: u = sin(2x)*cos(3y)*sin(4z)\n");
        printf(" 区域: [0,2π]³,  网格: %dx%dx%d,  进程: %d\n",
               nx, ny, nz, size);
        printf("=====================================================\n");
        printf(" ★ p3dfft 的 R2C 方向是 x（第一维）\n");
        printf("   kx = 0..nx/2（非负），ky/kz 需要折叠处理\n");
        printf("=====================================================\n");
        printf("%-8s  %-16s  %-16s\n", "方向", "L2误差", "最大误差");
        printf("-----------------------------------------------------\n");
    }

    // ===========================================================================
    // 6. 依次测试 d/dx, d/dy, d/dz
    // ===========================================================================
    const char *names[] = {"d/dx", "d/dy", "d/dz"};

    // op_b = "tff"：逆变换（先转置恢复 X-pencil，然后 IFFT）
    unsigned char op_b[] = "tff";

    for (int dir = 0; dir < 3; dir++) {

        // --- 6a. 复制 u_c 到 du_c
        // p3dfft btran_c2r 会销毁输入（OW=1），必须复制后使用
        du_c = u_c;

        // --- 6b. 频谱空间乘以 i*k（导数算子）
        //
        // p3dfft 频谱空间波数约定（★与 FFTW/heFFTe 不同）：
        //   i (x): R2C 方向，1-based，kx = i - offset = 0..nx/2（全非负）
        //   j (y): C2C 方向，1-based，ky = k_wrap(j, offset, ny)（有正负）
        //   k (z): C2C 方向，1-based，kz = k_wrap(k, offset, nz)（有正负）
        //
        // 复数乘法：i*wave*(a+ib) = -wave*b + i*wave*a
        //
        for (int i = fstart[0]; i <= fend[0]; ++i) {
            for (int j = fstart[1]; j <= fend[1]; ++j) {
                for (int kk = fstart[2]; kk <= fend[2]; ++kk) {
                    size_t idx = idx_c(fstart, fsize, i, j, kk);

                    // x: R2C，只有非负波数，直接减去 offset
                    double kx = (double)(i - offset);

                    // y, z: C2C，需要折叠（负频率在后半段）
                    double ky = k_wrap(j,  offset, ny);
                    double kz = k_wrap(kk, offset, nz);

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

        // --- 6c. 逆变换：p3dfft 不做归一化，需手动除以 N
        Cp3dfft_btran_c2r(reinterpret_cast<double*>(du_c.data()),
                          du_r.data(), op_b);

        // --- 6d. 与精确导数比较
        double local_l2 = 0.0, local_max = 0.0;
        for (int i = istart[0]; i <= iend[0]; ++i) {
            for (int j = istart[1]; j <= iend[1]; ++j) {
                for (int k = istart[2]; k <= iend[2]; ++k) {
                    double x = (i - offset) * dx;
                    double y = (j - offset) * dy;
                    double z = (k - offset) * dz;
                    size_t idx = idx_r(istart, isize, i, j, k);

                    double du_num = du_r[idx] / norm;  // 手动归一化

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

    Cp3dfft_clean();
    MPI_Finalize();
    return 0;
}
