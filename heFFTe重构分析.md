# heFFTe重构分析与实施方案

## 1. 当前代码FFT使用分析

### 1.1 FFT类型和用途

#### V1和V2速度分量（Cosine变换）
```
物理空间 → z方向DCT (REDFT00, nz/2+1点) → xy平面R2C (MPI) → 频谱空间
```

**数据尺寸变化：**
- 输入: `[local_nx][2*(ny/2+1)][nz/2+1]` (实数)
- z-DCT后: `[local_nx][2*(ny/2+1)][nz/2+1]` (实数)
- xy-R2C后: `[local_ny][nx][nz/2+1]` (复数，转置)

#### V3速度分量（Sine变换）
```
物理空间 → z方向DST (RODFT00, nz/2-1点) → xy平面R2C (MPI) → 频谱空间
```

**数据尺寸变化：**
- 输入: `[local_nx][2*(ny/2+1)][nz/2-1]` (实数)
- z-DST后: `[local_nx][2*(ny/2+1)][nz/2-1]` (实数)
- xy-R2C后: `[local_ny][nx][nz/2-1]` (复数，转置)

### 1.2 关键代码位置

| 功能 | 行号 | 说明 |
|-----|------|------|
| 全局plan声明 | 15-23 | 6个FFTW plan变量 |
| R2R初始化 | 26-46 | cos和sin的z方向局部变换 |
| R2C/C2R初始化 | 48-83 | xy平面MPI并行变换 |
| plan销毁 | 85-92 | 清理资源 |
| compute_v_cross_rot | 362-438 | V×rotV计算，多次FFT |
| compute_f | 441-477 | 外力项计算，多次FFT |
| 主函数初始化 | 870-876 | 创建所有plans |
| 主函数变换 | 905-942 | 初值和最终结果变换 |

### 1.3 数据布局

**FFTW MPI transposed布局：**
- **正向变换**:
  - 输入: `[local_nx][ny][nz]` - x方向分布
  - 输出: `[local_ny][nx][nz]` - y方向分布（转置）

- **反向变换**:
  - 输入: `[local_ny][nx][nz]` - y方向分布（转置）
  - 输出: `[local_nx][ny][nz]` - x方向分布

**为什么转置？**
- FFTW使用Slab分解
- z和y方向可以局部完成
- x方向需要通信，转置后可以高效处理

---

## 2. heFFTe重构策略

### 2.1 混合策略（推荐）

**方案：保留z方向FFTW R2R + heFFTe替换xy平面MPI并行**

**优点：**
- ✅ 改动最小，风险低
- ✅ 充分利用heFFTe的MPI通信优化
- ✅ 保持z方向R2R变换的正确性
- ✅ 代码结构清晰

**缺点：**
- ⚠️ 仍依赖FFTW（但这是成熟库）
- ⚠️ 不能完全GPU化（z方向仍在CPU）

### 2.2 完全heFFTe策略（备选）

**方案：所有FFT都用heFFTe**

**优点：**
- ✅ 代码统一
- ✅ 可能更好的整体性能
- ✅ 完全GPU化的可能性

**缺点：**
- ❌ heFFTe的R2R支持较新（需要2.4+）
- ❌ 需要大量测试验证正确性
- ❌ 3D混合变换的支持不明确

---

## 3. 实施方案（混合策略）

### 3.1 架构设计

```
┌─────────────────────────────────────────────┐
│           Navier-Stokes Solver              │
├─────────────────────────────────────────────┤
│  z方向变换          xy平面变换               │
│  ─────────          ─────────               │
│  FFTW R2R           heFFTe R2C/C2R          │
│  (局部，无MPI)      (分布式，MPI+Pencil)     │
│                                             │
│  • DCT (REDFT00)    • 3D R2C forward        │
│  • DST (RODFT00)    • 3D C2R backward       │
└─────────────────────────────────────────────┘
```

### 3.2 新的类结构

```cpp
class NavierStokesSolver {
private:
    // === FFTW部分：z方向局部R2R ===
    fftw_plan plan_r2r_cos_z;
    fftw_plan plan_r2r_sin_z;

    // === heFFTe部分：xy平面分布式R2C/C2R ===
    std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_cos;
    std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_sin;

    // 数据分布
    heffte::box3d<> inbox_cos;   // 输入box
    heffte::box3d<> outbox_cos;  // 输出box（转置）
    heffte::box3d<> inbox_sin;
    heffte::box3d<> outbox_sin;

    // MPI信息
    MPI_Comm comm;
    int rank, size;

public:
    void initialize(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, MPI_Comm comm);
    void forward_transform_cos(double* real_in, fftw_complex* spec_out);
    void backward_transform_cos(fftw_complex* spec_in, double* real_out);
    void forward_transform_sin(double* real_in, fftw_complex* spec_out);
    void backward_transform_sin(fftw_complex* spec_in, double* real_out);
    ~NavierStokesSolver();
};
```

### 3.3 初始化流程

```cpp
void NavierStokesSolver::initialize(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                                     MPI_Comm mpi_comm) {
    comm = mpi_comm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // ===== 第一步：确定数据分布 =====

    // heFFTe需要全局索引范围，获取每个进程的local范围
    ptrdiff_t local_nx, local_x_start, local_ny, local_y_start;
    ptrdiff_t alloc_local = fftw_mpi_local_size_2d_transposed(
        nx, ny/2+1, comm,
        &local_nx, &local_x_start,
        &local_ny, &local_y_start
    );

    // ===== 第二步：定义heFFTe的box =====

    // Cosine: nz/2+1 点
    int nz_cos = nz/2 + 1;

    // 输入box：x方向分布，每个进程处理 [local_x_start, local_x_start+local_nx-1]
    inbox_cos = heffte::box3d<>(
        {static_cast<int>(local_x_start), 0, 0},
        {static_cast<int>(local_x_start + local_nx - 1),
         static_cast<int>(ny - 1),
         nz_cos - 1}
    );

    // 输出box：y方向分布（转置），处理 [local_y_start, local_y_start+local_ny-1]
    outbox_cos = heffte::box3d<>(
        {0, static_cast<int>(local_y_start), 0},
        {static_cast<int>(nx - 1),
         static_cast<int>(local_y_start + local_ny - 1),
         nz_cos - 1}
    );

    // Sine: nz/2-1 点
    int nz_sin = nz/2 - 1;
    inbox_sin = heffte::box3d<>(
        {static_cast<int>(local_x_start), 0, 0},
        {static_cast<int>(local_x_start + local_nx - 1),
         static_cast<int>(ny - 1),
         nz_sin - 1}
    );
    outbox_sin = heffte::box3d<>(
        {0, static_cast<int>(local_y_start), 0},
        {static_cast<int>(nx - 1),
         static_cast<int>(local_y_start + local_ny - 1),
         nz_sin - 1}
    );

    // ===== 第三步：配置heFFTe选项 =====

    heffte::plan_options options;
    options.use_pencils = (size > 16);  // 大规模用pencil分解
    options.use_reorder = true;         // 允许数据重排优化
    options.algorithm = heffte::reshape_algorithm::alltoallv;  // 通信算法

    // ===== 第四步：创建heFFTe对象 =====

    fft_cos = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(
        inbox_cos, outbox_cos, comm, options
    );

    fft_sin = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(
        inbox_sin, outbox_sin, comm, options
    );

    // ===== 第五步：创建FFTW z方向R2R plans =====

    // Cosine z方向
    int nz_cos_arr = nz_cos;
    plan_r2r_cos_z = fftw_plan_many_r2r(
        1,                          // rank=1 (一维变换)
        &nz_cos_arr,                // 变换尺寸
        local_nx * (2*(ny/2+1)),    // howmany（批次数）
        nullptr, nullptr, 1, nz_cos_arr,  // 输入参数
        nullptr, nullptr, 1, nz_cos_arr,  // 输出参数
        &FFTW_REDFT00,              // DCT-I
        FFTW_PATIENT                // 规划标志
    );

    // Sine z方向
    int nz_sin_arr = nz_sin;
    plan_r2r_sin_z = fftw_plan_many_r2r(
        1, &nz_sin_arr,
        local_nx * (2*(ny/2+1)),
        nullptr, nullptr, 1, nz_sin_arr,
        nullptr, nullptr, 1, nz_sin_arr,
        &FFTW_RODFT00,              // DST-I
        FFTW_PATIENT
    );
}
```

### 3.4 正向变换（Cosine）

```cpp
void NavierStokesSolver::forward_transform_cos(double* real_in,
                                                fftw_complex* spec_out) {
    // ===== 步骤1: z方向DCT (局部，FFTW) =====
    // real_in尺寸: [local_nx][2*(ny/2+1)][nz/2+1]
    // 输出: 同样尺寸
    double* z_transformed = fftw_alloc_real(/* 适当大小 */);
    fftw_execute_r2r(plan_r2r_cos_z, real_in, z_transformed);

    // ===== 步骤2: xy平面R2C (分布式，heFFTe) =====
    // 输入: [local_nx][ny][nz/2+1] (x方向分布)
    // 输出: [local_ny][nx][nz/2+1] (y方向分布，复数)

    std::vector<double> input_real(fft_cos->size_inbox());
    std::vector<std::complex<double>> output_complex(fft_cos->size_outbox());

    // 复制数据到heFFTe输入缓冲区
    // 注意：需要处理FFTW的padding（2*(ny/2+1)）
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz/2+1; ++k) {
                ptrdiff_t src_idx = (i * (2*(ny/2+1)) + j) * (nz/2+1) + k;
                ptrdiff_t dst_idx = (i * ny + j) * (nz/2+1) + k;
                input_real[dst_idx] = z_transformed[src_idx];
            }
        }
    }

    // 执行heFFTe R2C变换
    fft_cos->forward(input_real.data(), output_complex.data());

    // 复制到输出（需要转换数据布局）
    // output_complex是[local_ny][nx][nz/2+1]，转置后的
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz/2+1; ++k) {
                ptrdiff_t src_idx = (j * nx + i) * (nz/2+1) + k;
                ptrdiff_t dst_idx = src_idx;  // 根据实际需要调整
                spec_out[dst_idx][0] = output_complex[src_idx].real();
                spec_out[dst_idx][1] = output_complex[src_idx].imag();
            }
        }
    }

    // 归一化（heFFTe和FFTW的归一化约定可能不同）
    double norm_factor = 1.0 / (nx * ny * nz);
    #pragma omp parallel for
    for(size_t i = 0; i < fft_cos->size_outbox(); ++i) {
        spec_out[i][0] *= norm_factor;
        spec_out[i][1] *= norm_factor;
    }

    fftw_free(z_transformed);
}
```

### 3.5 反向变换（Cosine）

```cpp
void NavierStokesSolver::backward_transform_cos(fftw_complex* spec_in,
                                                 double* real_out) {
    // ===== 步骤1: xy平面C2R (分布式，heFFTe) =====
    std::vector<std::complex<double>> input_complex(fft_cos->size_outbox());
    std::vector<double> output_real(fft_cos->size_inbox());

    // 复制输入数据
    #pragma omp parallel for
    for(size_t i = 0; i < input_complex.size(); ++i) {
        input_complex[i] = std::complex<double>(spec_in[i][0], spec_in[i][1]);
    }

    // 执行heFFTe C2R变换
    fft_cos->backward(input_complex.data(), output_real.data());

    // ===== 步骤2: z方向逆DCT (局部，FFTW) =====
    double* xy_transformed = fftw_alloc_real(/* 适当大小 */);

    // 从heFFTe输出复制到FFTW输入（处理padding）
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz/2+1; ++k) {
                ptrdiff_t src_idx = (i * ny + j) * (nz/2+1) + k;
                ptrdiff_t dst_idx = (i * (2*(ny/2+1)) + j) * (nz/2+1) + k;
                xy_transformed[dst_idx] = output_real[src_idx];
            }
        }
    }

    // 执行z方向逆DCT
    fftw_execute_r2r(plan_r2r_cos_z, xy_transformed, real_out);

    fftw_free(xy_transformed);
}
```

---

## 4. 实施步骤

### 第一阶段：准备和测试环境

**任务清单：**
- [ ] 在MGU-270上安装heFFTe库
  - 确认FFTW已安装（作为backend）
  - 编译heFFTe with FFTW support
  - 测试heFFTe example程序

- [ ] 创建测试分支
  ```bash
  git checkout -b heffte-migration
  ```

- [ ] 编写简单测试程序
  - 测试heFFTe R2C/C2R的基本功能
  - 验证与FFTW的结果一致性
  - 测试数据布局转换

### 第二阶段：代码重构

**任务清单：**
- [ ] 创建新文件 `NavierStokes_heffte.cpp`
  - 复制原始文件
  - 添加heFFTe头文件

- [ ] 重写初始化函数
  - 实现 `initialize_heffte_r2c_cos`
  - 实现 `initialize_heffte_r2c_sin`
  - 保留 `initialize_r2r_cos/sin`（FFTW）

- [ ] 重写变换执行函数
  - 修改 `compute_v_cross_rot`
  - 修改 `compute_f`
  - 修改主函数的FFT调用

- [ ] 处理数据布局
  - 编写数据复制/转换辅助函数
  - 处理FFTW的padding问题
  - 确保heFFTe输入/输出与现有代码兼容

### 第三阶段：测试和验证

**测试清单：**
- [ ] 单元测试
  - 测试每个FFT变换单独执行
  - 验证正向+反向=恒等变换
  - 检查归一化因子

- [ ] 集成测试
  - 运行完整求解器
  - 对比能量守恒（Energy1, Energy2, Energy3）
  - 对比误差指标（err1, err2, err3）

- [ ] 正确性验证
  - 小规模测试（32³, 64³）
  - 与原FFTW版本结果对比
  - 相对误差应小于1e-10

- [ ] 性能测试
  - 不同网格大小（128³, 256³, 512³）
  - 不同进程数（1, 4, 16, 64, 256）
  - 记录时间分解（初始化、FFT、计算）

### 第四阶段：优化和调优

**优化清单：**
- [ ] heFFTe参数调优
  - 测试 `use_pencils` vs `use_reorder`
  - 测试不同通信算法
  - 测试不同backend（如果有MKL）

- [ ] OpenMP优化
  - 检查数据复制循环的并行化
  - 避免并行区域的竞争条件

- [ ] 内存优化
  - 减少临时缓冲区
  - 重用内存分配
  - 检查内存对齐

---

## 5. 预期挑战和解决方案

### 5.1 数据布局不匹配

**问题：** FFTW使用padding（`2*(ny/2+1)`），heFFTe可能不用

**解决：**
```cpp
void copy_with_padding(double* src, double* dst,
                       ptrdiff_t local_nx, ptrdiff_t ny, ptrdiff_t nz) {
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                dst[(i*ny + j)*nz + k] = src[(i*(2*(ny/2+1)) + j)*nz + k];
            }
        }
    }
}
```

### 5.2 归一化约定不同

**问题：** FFTW和heFFTe的归一化因子可能不同

**解决：**
1. 阅读heFFTe文档确认约定
2. 在变换后手动调整归一化
3. 通过单位测试验证

### 5.3 转置布局处理

**问题：** 原代码使用`FFTW_MPI_TRANSPOSED_OUT/IN`

**解决：**
- heFFTe通过不同的`inbox`和`outbox`实现转置
- 确保`outbox`的第一维是y方向分布
- 可能需要额外的数据重排

### 5.4 R2R变换的兼容性

**问题：** heFFTe的R2R支持可能有限

**解决方案（当前采用）：**
- z方向继续使用FFTW R2R
- 仅xy平面使用heFFTe
- 未来如需完全迁移，可考虑：
  - 使用heFFTe的R2R接口（如果稳定）
  - 或将R2R转换为R2C（通过对称性）

---

## 6. 编译和链接

### 6.1 修改Makefile

```makefile
# heFFTe设置
HEFFTE_DIR = /path/to/heffte
HEFFTE_INC = -I$(HEFFTE_DIR)/include
HEFFTE_LIB = -L$(HEFFTE_DIR)/lib -lheffte

# FFTW设置（保留，用于R2R）
FFTW_INC = -I/path/to/fftw/include
FFTW_LIB = -L/path/to/fftw/lib -lfftw3_mpi -lfftw3 -lfftw3_threads

# 编译器
CXX = mpicxx
CXXFLAGS = -O3 -std=c++11 -fopenmp $(HEFFTE_INC) $(FFTW_INC)
LDFLAGS = $(HEFFTE_LIB) $(FFTW_LIB) -lm

# 目标
navier_stokes_heffte: NavierStokes_heffte.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
```

### 6.2 编译命令

```bash
# 编译
make navier_stokes_heffte

# 运行（示例）
mpirun -np 16 ./navier_stokes_heffte
```

---

## 7. 性能预期

### 7.1 预期改进

根据heFFTe论文和相关研究：

| 场景 | 预期加速比 | 原因 |
|------|-----------|------|
| 小规模（≤16进程） | 1.0-1.2x | Slab vs Pencil差异不大 |
| 中等规模（64进程） | 1.5-2.0x | Pencil分解优势显现 |
| 大规模（256+进程） | 2.0-5.0x | 优化的通信+Pencil分解 |
| GPU版本 | 10-20x | GPU加速+优化通信 |

### 7.2 弱扩展性

heFFTe设计用于exascale，应该展现出色的弱扩展性：
- 固定每进程工作量
- 增加进程数和总问题规模
- 运行时间应保持相对稳定

### 7.3 通信时间占比

- FFTW: 通信占40-60%
- heFFTe: 通信占20-30%（通过重叠优化）

---

## 8. 后续GPU扩展路径

一旦CPU版本稳定，可以考虑GPU扩展：

### 8.1 切换backend

```cpp
// 从FFTW切换到cuFFT
heffte::fft3d_r2c<heffte::backend::cufft> fft_gpu(
    inbox, outbox, comm, options
);
```

### 8.2 内存管理

```cpp
// GPU内存分配
double* d_real_in;
cudaMalloc(&d_real_in, size * sizeof(double));

// 数据传输
cudaMemcpy(d_real_in, h_real_in, size * sizeof(double),
           cudaMemcpyHostToDevice);

// FFT执行（在GPU上）
fft_gpu->forward(d_real_in, d_complex_out);
```

### 8.3 混合CPU-GPU

- z方向R2R在CPU（FFTW）
- xy平面在GPU（heFFTe+cuFFT）
- 需要CPU-GPU数据传输

---

## 9. 检查清单

### 代码完成度
- [ ] 所有FFT初始化函数已重写
- [ ] 所有FFT执行点已更新
- [ ] 数据布局转换已实现
- [ ] 归一化处理正确
- [ ] OpenMP并行化保留

### 测试完成度
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 正确性验证通过（误差< 1e-10）
- [ ] 小规模性能测试完成
- [ ] 大规模性能测试完成

### 文档完成度
- [ ] 代码注释完整
- [ ] README更新（依赖、编译、运行）
- [ ] 性能报告编写
- [ ] 与FFTW版本对比报告

---

## 10. 参考资源

### heFFTe文档
- 官方文档: https://icl-utk-edu.github.io/heffte/
- API参考: https://icl-utk-edu.github.io/heffte/doxygen/
- 示例代码: https://github.com/icl-utk-edu/heffte/tree/master/examples

### 相关论文
- heFFTe原始论文: https://pmc.ncbi.nlm.nih.gov/articles/PMC7302276/
- 分布式FFT综述: 查找相关文献

### 技术支持
- GitHub Issues: https://github.com/icl-utk-edu/heffte/issues
- Email: heffte@icl.utk.edu

---

## 附录：代码模板

### A. 完整的类定义模板

```cpp
// NavierStokes_heffte.hpp
#pragma once

#include <heffte.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <memory>
#include <complex>
#include <vector>

class NavierStokesSolver {
private:
    // MPI通信
    MPI_Comm comm;
    int rank, size;

    // 网格参数
    ptrdiff_t nx, ny, nz;
    ptrdiff_t local_nx, local_x_start;
    ptrdiff_t local_ny, local_y_start;
    ptrdiff_t alloc_local;

    // heFFTe对象
    std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_cos;
    std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> fft_sin;

    // FFTW plans (z方向R2R)
    fftw_plan plan_r2r_cos_z;
    fftw_plan plan_r2r_sin_z;

    // 数据box
    heffte::box3d<> inbox_cos, outbox_cos;
    heffte::box3d<> inbox_sin, outbox_sin;

public:
    NavierStokesSolver(ptrdiff_t nx_, ptrdiff_t ny_, ptrdiff_t nz_,
                       MPI_Comm comm_);
    ~NavierStokesSolver();

    void initialize();
    void forward_transform_cos(double* real_in, fftw_complex* spec_out);
    void backward_transform_cos(fftw_complex* spec_in, double* real_out);
    void forward_transform_sin(double* real_in, fftw_complex* spec_out);
    void backward_transform_sin(fftw_complex* spec_in, double* real_out);

    // Getters
    ptrdiff_t get_local_nx() const { return local_nx; }
    ptrdiff_t get_local_ny() const { return local_ny; }
    ptrdiff_t get_local_x_start() const { return local_x_start; }
    ptrdiff_t get_local_y_start() const { return local_y_start; }
};
```

### B. 测试程序模板

```cpp
// test_heffte_migration.cpp
#include "NavierStokes_heffte.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const ptrdiff_t nx = 64, ny = 64, nz = 64;

    NavierStokesSolver solver(nx, ny, nz, MPI_COMM_WORLD);
    solver.initialize();

    // 分配内存
    ptrdiff_t local_nx = solver.get_local_nx();
    double* input = fftw_alloc_real(local_nx * (2*(ny/2+1)) * (nz/2+1));
    fftw_complex* spectrum = fftw_alloc_complex(/* size */);
    double* output = fftw_alloc_real(local_nx * (2*(ny/2+1)) * (nz/2+1));

    // 初始化测试数据
    for(ptrdiff_t i = 0; i < local_nx * (2*(ny/2+1)) * (nz/2+1); ++i) {
        input[i] = 1.0;  // 或其他测试函数
    }

    // 正向变换
    solver.forward_transform_cos(input, spectrum);

    // 反向变换
    solver.backward_transform_cos(spectrum, output);

    // 验证：output应该等于input
    double max_error = 0.0;
    for(ptrdiff_t i = 0; i < local_nx * (2*(ny/2+1)) * (nz/2+1); ++i) {
        double error = fabs(output[i] - input[i]);
        if(error > max_error) max_error = error;
    }

    double global_max_error;
    MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::cout << "Maximum error: " << global_max_error << std::endl;
        if(global_max_error < 1e-10) {
            std::cout << "TEST PASSED!" << std::endl;
        } else {
            std::cout << "TEST FAILED!" << std::endl;
        }
    }

    fftw_free(input);
    fftw_free(spectrum);
    fftw_free(output);

    MPI_Finalize();
    return 0;
}
```

---

## 总结

本文档提供了将现有Navier-Stokes求解器从纯FFTW迁移到heFFTe+FFTW混合方案的完整路线图。

**核心策略：**
- z方向：保留FFTW R2R（DCT/DST）
- xy平面：使用heFFTe分布式R2C/C2R

**预期收益：**
- 更好的可扩展性（Pencil分解）
- 优化的MPI通信（重叠技术）
- GPU加速的可能性

**实施难点：**
- 数据布局转换
- 归一化处理
- 正确性验证

按照本文档的步骤逐步实施，可以安全、高效地完成迁移工作。
