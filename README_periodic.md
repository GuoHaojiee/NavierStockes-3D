# 全周期性边界条件的Navier-Stokes求解器

## 概述

这是一个使用FFTW库的全周期性边界条件Navier-Stokes求解器。

### 主要特点

1. ✅ **全周期性边界条件**（所有方向）
2. ✅ **Taylor-Green涡验证算例**（有解析解）
3. ✅ **简化的FFT结构**（不再需要R2R变换）
4. ✅ **3D R2C/C2R变换**（标准FFTW MPI）
5. ✅ **MPI + OpenMP混合并行**

### 与原版本的主要区别

| 特性 | 原版本 | 新版本（周期性） |
|------|--------|------------------|
| 边界条件 | x,y周期; z第一类/第二类 | 所有方向周期 |
| FFT类型 | R2R (DCT/DST) + R2C | 仅R2C |
| 数据结构 | V1,V2用cos; V3用sin | 统一处理 |
| 验证算例 | 自定义函数 | Taylor-Green涡 |
| 代码复杂度 | 高（混合变换） | 低（统一变换） |

## 编译

### 前提条件

需要安装：
- MPI（OpenMPI或MPICH）
- FFTW3（包括MPI支持）
- OpenMP支持的编译器

### 编译命令

```bash
# 使用Makefile
make -f Makefile_periodic

# 或者手动编译
mpicxx -O3 -std=c++11 -fopenmp \
    NavierStokes_periodic_fftw.cpp \
    -o navier_stokes_periodic \
    -lfftw3_mpi -lfftw3 -lfftw3_threads -lm
```

## 运行

### 基本运行

```bash
# 使用4个MPI进程
mpirun -np 4 ./navier_stokes_periodic

# 使用8个MPI进程
mpirun -np 8 ./navier_stokes_periodic
```

### 在超算上运行

创建作业脚本 `job_periodic.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ns_periodic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun ./navier_stokes_periodic
```

提交作业：
```bash
sbatch job_periodic.sh
```

## Taylor-Green涡验证

### 解析解

Taylor-Green涡是Navier-Stokes方程的精确解：

**速度场**：
```
v1(x,y,z,t) = V0 * sin(x) * cos(y) * cos(z) * exp(-3νt)
v2(x,y,z,t) = -V0 * cos(x) * sin(y) * cos(z) * exp(-3νt)
v3(x,y,z,t) = 0
```

**压力场**：
```
p(x,y,z,t) = (ρV0²/16) * [cos(2x) + cos(2y)] * [cos(2z) + 2] * exp(-6νt)
```

### 参数设置

- 域：`[0, 2π]³`
- 粘度：`ν = 0.001`
- 初始速度幅值：`V0 = 1.0`

### 验证指标

程序会自动检查：

1. **FFT往返测试**：
   - 实空间 → 频谱空间 → 实空间
   - 误差应 < 1e-10

2. **散度检查**：
   - `∇·v ≈ 0`（无散场）

3. **与解析解对比**（未来添加）：
   - L2范数误差
   - 能量衰减率

## 输出说明

运行时会显示：

```
============================================================
  Navier-Stokes Solver - Periodic Boundary Conditions
  Taylor-Green Vortex Test Case
============================================================
Grid: 64 x 64 x 64
Domain: [0, 2π]³
Viscosity: 0.001
Time steps: 10, dt = 0.01
MPI processes: 4
OpenMP threads/process: 8
============================================================
Local allocation size: XXX
Initializing FFTW plans...
FFTW planning time: X.XX seconds
Setting initial conditions...
Transforming to spectral space...
Testing transformation...

Initial condition verification:
  V1 L2 error: 1.23e-11
  V2 L2 error: 1.45e-11
  V3 L2 error: 0.00e+00
  ✓ FFT round-trip test PASSED!

============================================================
  Program completed successfully!
============================================================
```

## 当前实现状态

### ✅ 已完成

- [x] 全周期性边界条件
- [x] Taylor-Green涡解析解函数
- [x] 3D FFT初始化（FFTW MPI）
- [x] 频谱空间旋度计算
- [x] 频谱空间散度计算
- [x] 无散投影方法
- [x] FFT往返测试

### 🚧 进行中

- [ ] 时间步进（Runge-Kutta）
- [ ] 完整的Navier-Stokes求解循环
- [ ] 能量统计
- [ ] 与解析解的误差分析

### 📋 待完成

- [ ] 完整的验证测试套件
- [ ] 性能测试和profiling
- [ ] 可视化输出（VTK格式）
- [ ] 参数配置文件

## 代码结构

```
NavierStokes_periodic_fftw.cpp
├── 全局变量
│   └── FFTW计划
├── 数学函数
│   ├── Taylor-Green涡解析解
│   ├── 速度场 (v1, v2, v3)
│   ├── 压力场 (p)
│   ├── 旋度 (rot)
│   └── 外力项 (f)
├── FFT函数
│   ├── initialize_fftw_3d()
│   ├── normalization()
│   └── finalize_fft_plans()
├── 频谱空间操作
│   ├── compute_rot()
│   ├── compute_div()
│   └── make_div_free()
└── 主程序
    ├── MPI/OpenMP初始化
    ├── 内存分配
    ├── FFT计划创建
    ├── 初始条件设置
    ├── FFT测试
    └── 清理
```

## 下一步工作

### 立即任务

1. **完成时间步进**：
   ```cpp
   void rungeKutta4(...) {
       // 实现RK4时间积分
   }
   ```

2. **添加完整求解循环**：
   ```cpp
   for (int it = 0; it < nt; ++it) {
       // 计算右端项
       // 时间步进
       // 保持无散
   }
   ```

3. **添加验证测试**：
   - 每个时间步与解析解对比
   - 输出L2误差
   - 检查能量守恒

### 中期任务

4. **性能优化**：
   - Profiling找到热点
   - 优化数据布局
   - 减少内存拷贝

5. **扩展测试**：
   - 不同网格大小（32³ 到 512³）
   - 不同MPI进程数（1 到 1024）
   - 弱扩展性测试

### 长期任务

6. **迁移到heFFTe**：
   - 替换FFTW MPI为heFFTe
   - 利用Pencil分解
   - GPU加速（可选）

## 参考

### Taylor-Green涡

- Taylor, G. I., & Green, A. E. (1937). "Mechanism of the production of small eddies from large ones"
- 标准DNS验证算例

### FFTW文档

- [FFTW官网](http://www.fftw.org/)
- [FFTW MPI文档](http://www.fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html)

## 问题排查

### 编译错误

**问题**：找不到FFTW头文件
```
fatal error: fftw3-mpi.h: No such file or directory
```

**解决**：
```bash
# 查找FFTW安装位置
find /usr -name "fftw3.h"

# 修改Makefile中的FFTW_DIR
FFTW_DIR = /path/to/your/fftw
```

### 运行时错误

**问题**：段错误
```
Segmentation fault
```

**可能原因**：
1. 内存不足 → 减小网格大小
2. MPI进程数过多 → 减少进程数
3. FFTW未正确初始化 → 检查fftw_mpi_init()

**问题**：FFT测试失败（误差太大）

**可能原因**：
1. 归一化因子错误 → 检查normalization()
2. 数据索引错误 → 检查数组访问
3. 边界处理错误 → 检查padding (2*(nz/2+1))

## 联系

如有问题，请：
1. 检查日志输出
2. 尝试更小的网格（如32³）
3. 单进程运行调试

---

**版本**: 1.0
**日期**: 2025-12-20
**作者**: Guo Haojie
**状态**: 开发中（基础框架完成，待完善时间步进）
