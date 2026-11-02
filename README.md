# 三维周期性 Navier-Stokes 方程伪谱求解器

## 项目简介

本项目实现了三维周期性边界条件下不可压缩粘性流体的伪谱数值求解器，覆盖从本科阶段的 MPI/OpenMP 并行实现，到研究生阶段的多 FFT 库移植与 GPU 加速版本。


---

## 目录

- [物理模型](#物理模型)
- [数值方法](#数值方法)
- [本科阶段工作（FFTW + MPI/OpenMP）](#本科阶段工作)
- [研究生阶段工作（多库移植 + GPU 加速）](#研究生阶段工作)
- [代码结构](#代码结构)
- [编译与运行](#编译与运行)
- [验证与结果](#验证与结果)

---

## 物理模型

求解三维周期性边界条件下的不可压缩 Navier-Stokes 方程：

$$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\nabla p + \nu \Delta \mathbf{v} + \mathbf{f}$$

$$\nabla \cdot \mathbf{v} = 0$$

其中 $\mathbf{v}$ 为速度场，$p$ 为压力，$\nu$ 为运动粘度，$\mathbf{f}$ 为外力。

求解域为 $[0, 2\pi]^3$ 的周期性立方体，网格分辨率 $64^3$，时间步长 $dt = 5 \times 10^{-5}$。

**热对流扩展：** 在方程中加入浮力项 $\Pr \cdot \theta \, \mathbf{e}_3$，描述温差引起的对流运动，其中 $\theta$ 为温度扰动，$\Pr$ 为普朗特数。

---

## 数值方法

### 空间离散：伪谱方法

将速度场展开为傅里叶级数，利用快速傅里叶变换（FFT）在谱空间中高效计算导数（谱空间中微分等价于乘以波数 $ik$），在物理空间中计算非线性项，从而获得谱方法的高精度。

### 时间积分：RK4

使用经典四阶 Runge-Kutta 方法进行时间推进，精度为 $O(\tau^4)$。

### 投影法（压力修正）

每个时间步结束后，在谱空间对速度场做 Helmholtz 投影，消除散度，保证不可压缩约束 $\nabla \cdot \mathbf{v} = 0$。

### 归一化约定（统一）

所有版本均采用**非归一化 DFT 约定**：
- 正变换：$\hat{V} = \mathrm{DFT}(V)$（不除以 $N$）
- 逆变换后手动除以 $N$ 得到物理值

---

## 本科阶段工作

**题目：** 基于并行计算系统的粘性不可压缩流体流动伪谱方法数值模拟

**计算平台：** 莫斯科国立大学 VMK 学院 Polus 集群（4 节点，每节点 2 块 10 核处理器）

### 并行算法

核心算法如下图所示。谱空间（带帽子的量 $\hat{V}$）和物理空间（不带帽子的量 $V$）之间通过 FFT/iFFT 转换，非线性项在物理空间中计算，导数在谱空间中计算。

![并行算法示意图](pic/并行算法.png)

### OpenMP 并行优化

实现了三种线程绑定策略：
- **无绑定**：线程由操作系统动态调度
- **顺序绑定**：线程依次绑定到同一处理器的核，再到另一处理器
- **均匀绑定**：线程均匀分布在两个处理器的核上

实验结果表明，**均匀绑定**加速效果最佳（红线）：

![OpenMP 不同绑定策略对比](pic/不同绑定下的openmp结果对比.png)

### MPI 并行优化

数据按 x 方向分层分布（1D slab 分解）。主要优化手段：

**优化 1：消除逆转置**

3D FFT 需对 x 方向做全局转置，FFTW 完成后会自动执行逆转置还原数据布局。由于我们在谱空间中直接操作，可跳过逆转置，直接使用转置后的数据布局计算：

![消除逆转置优化对比](pic/MPI版本优化1取消装置结果对比.png)

**优化 2：FFTW 计划选择**

对比 `FFTW_ESTIMATE`、`FFTW_MEASURE`、`FFTW_PATIENT` 三种计划，`FFTW_PATIENT` 预计算时间最长但执行性能最优（紫色柱）：

![MPI 不同 FFTW plan 优化](pic/MPI不同plan的优化.png)

### MPI + OpenMP 混合并行

结合两种并行技术，混合版本在所有参数下均取得最佳加速比：

![MPI+OpenMP 混合版本加速效果](pic/MPI+OpenMp混合版本得到最佳优化.png)

### 热对流数值实验

在 Navier-Stokes 方程中加入热浮力项，对 Rayleigh-Bénard 对流问题进行数值模拟。

**可视化结果：** 速度场和竖直截面流场清晰呈现对流涡胞结构，与理论预期吻合。

![热对流速度场可视化 1](pic/对流实验可视化图片1.png)

![热对流速度场可视化 2（竖直截面）](pic/对流实验可视化图片2.png)

数值解的动能随时间演化收敛于解析值，验证了程序的正确性。

### 本科阶段主要成果

- 实现了基于 MPI、OpenMP、MPI+OpenMP 的三种并行求解器
- 验证了良好的并行可扩展性
- 完成热对流数值实验，结果与理论吻合

---

## 研究生阶段工作

### 动机：突破 FFTW 的 1D Slab 分解限制

FFTW MPI 版本仅支持 **1D slab 分解**（沿单一维度切分），当进程数超过网格一维大小时无法继续扩展。为支持更大规模并行，需要引入 **2D pencil 分解**（沿两个维度切分），理论上可将并行进程数从 $O(N)$ 扩展到 $O(N^2)$。

我们对三个支持 2D pencil 分解的 FFT 库进行了移植与性能研究：

### 多 FFT 库移植

#### heFFTe（Highly Efficient FFT for Exascale）

- **文件：** `NavierStokes_periodic_heffte_v2.cpp`
- **特性：** 支持 CPU/GPU 后端，2D pencil 分解，ORNL 开发
- R2C 沿 z 方向，$k_z \in [0, N_z/2]$；$k_x, k_y$ 按标准折叠规则
- 逆变换不破坏输入，无需复制缓冲区
- `forward(scale::none)` + `backward(scale::full)`（自动 $/N$）

#### p3dfft

- **文件：** `NavierStokes_periodic_p3dfft.cpp`
- **特性：** 成熟的 2D pencil 分解库，广泛用于气候/流体模拟
- R2C 沿 x 方向，$k_x \in [0, N_x/2]$；$k_y, k_z$ 需手动折叠
- `btran_c2r` 销毁输入，每次调用前必须复制缓冲区
- 实空间含 padding，需用 `memsize` 计算内存大小

#### AccFFT

- **文件：** `NavierStokes_periodic_accfft.cpp`
- **特性：** 面向加速器的 2D pencil 分解库
- 逆变换不自动归一化，需手动 $/N_{\mathrm{total}}$
- Nyquist 分量（$= N/2$）置零以保证实对称性
- 已验证：L2 误差 $\approx 2.5 \times 10^{-14}$

### 谱导数验证

针对每个库，单独编写了谱导数验证程序：

| 文件 | 库 | 状态 |
|------|-----|------|
| `test_spectral_derivative.cpp` | FFTW | 已通过 |
| `test_spectral_derivative_heffte.cpp` | heFFTe | 已通过 |
| `test_spectral_derivative_p3dfft.cpp` | p3dfft | 已通过 |
| `test_spectral_derivative_accfft.cpp` | AccFFT | 已通过（误差 $\sim 10^{-14}$）|

### 解析解（Taylor-Green 涡变体）

用于代码验证的解析解：

$$V_1 = V_2 = (t^2+1) e^{\sin(3x+3y)} \cos(6z)$$
$$V_3 = -(t^2+1) e^{\sin(3x+3y)} \cos(3x+3y) \sin(6z)$$
$$p = (t^2+1) \cos x \cos y \cos z$$

验证参数：$64^3$ 网格，$dt = 5 \times 10^{-5}$，运行 10 步。

### GPU 加速

| 文件 | 描述 |
|------|------|
| `NavierStokes_periodic_cufft.cu` | 单节点单 GPU（cuFFT） |
| `NavierStokes_periodic_cufftxt.cu` | 单节点多 GPU（cuFFTXt） |
| `NavierStokes_periodic_heffte_gpu1.cu` | 单节点多 GPU（heFFTe+cuFFT，MPI） |
| `NavierStokes_periodic_heffte_mgpu.cu` | 单节点多 GPU（heFFTe+cuFFT，MPI） |
| `NavierStokes_periodic_heffte_multigpu.cu` | 多节点多 GPU（heFFTe+cuFFT，MPI） |
| `NavierStokes_periodic_cufftmp_mgpu.cu` | 单节点多 GPU（cuFFTMp，MPI） |
| `NavierStokes_periodic_cufftmp_multinode.cu` | 多节点多 GPU（cuFFTMp，MPI） |

---

## 代码结构

```
.
├── NavierStokes_periodic_fftw.cpp           # FFTW MPI（1D slab，参考实现）
├── NavierStokes_periodic_heffte_v2.cpp      # heFFTe CPU（2D pencil）
├── NavierStokes_periodic_p3dfft.cpp         # p3dfft（2D pencil）
├── NavierStokes_periodic_accfft.cpp         # AccFFT（2D pencil）
├── NavierStokes_periodic_cufft.cu           # 单 GPU（cuFFT）
├── NavierStokes_periodic_cufftxt.cu         # 单节点多 GPU（cuFFTXt）
├── NavierStokes_periodic_heffte_gpu1.cu     # 单节点多 GPU（heFFTe+cuFFT）
├── NavierStokes_periodic_heffte_mgpu.cu     # 单节点多 GPU（heFFTe+cuFFT）
├── NavierStokes_periodic_heffte_multigpu.cu # 多节点多 GPU（heFFTe+cuFFT）
├── NavierStokes_periodic_cufftmp_mgpu.cu    # 单节点多 GPU（cuFFTMp）
├── NavierStokes_periodic_cufftmp_multinode.cu # 多节点多 GPU（cuFFTMp）
├── common_mode.h                            # 公共：verify/bench 模式解析、分阶段计时
├── test_spectral_derivative*.cpp            # 各库谱导数验证
├── Makefile_*                               # 对应编译文件
└── pic/                                     # 实验结果图片
```

---

## 编译与运行

### 依赖

| 库 | 版本要求 | 安装路径 |
|----|---------|---------|
| FFTW3 MPI | ≥ 3.3 | 系统（`/usr/lib/x86_64-linux-gnu`）|
| heFFTe | ≥ 2.3 | `~/.local` |
| p3dfft | ≥ 2.7 | `~/.local` |
| AccFFT | latest | `~/.local`（源码：`~/accfft`）|
| MPI | OpenMPI 或 MPICH | 系统 |

### 编译

```bash
# FFTW 版
make -f Makefile_periodic

# heFFTe 版
make -f Makefile_heffte

# p3dfft 版
make -f Makefile_p3dfft

# AccFFT 版
make -f Makefile_accfft
```

### 运行

统一格式：

```
mpirun -np <NP> ./<CPU 版> NX NY NZ dt NSTEPS [OMP_THREADS] [verify|bench]
mpirun -np <NP> ./<GPU 版> NX NY NZ dt NSTEPS [verify|bench]
```

- `OMP_THREADS`（可选）仅 CPU 版支持，默认使用系统最大线程数
- 运行模式（可选，默认 `verify`），详见下方「运行模式」

#### CPU 版

```bash
# FFTW MPI（1D slab）
mpirun -np 4 ./navier_stokes_periodic 64 64 64 1e-4 100 4

# heFFTe（2D pencil）
mpirun -np 4 ./navier_stokes_heffte_v2 64 64 64 1e-4 100 4

# p3dfft（2D pencil）
mpirun -np 4 ./navier_stokes_p3dfft 64 64 64 1e-4 100 4

# AccFFT（2D pencil）
mpirun -np 4 ./navier_stokes_accfft 64 64 64 1e-4 100 4
```

#### 单 GPU

```bash
./navier_stokes_cufft 64 64 64 1e-4 100
```

#### 多 GPU（单节点）

```bash
# cuFFTXt（无需 MPI）
./navier_stokes_cufftxt 128 128 128 1e-4 100

# heFFTe + cuFFT
mpirun -np 4 ./navier_stokes_heffte_gpu1 128 128 128 1e-4 100
mpirun -np 4 ./navier_stokes_heffte_mgpu 128 128 128 1e-4 100

# cuFFTMp
mpirun -np 4 ./navier_stokes_cufftmp_mgpu 128 128 128 1e-4 100
```

#### 多 GPU（多节点）

```bash
# 2 节点 × 4 GPU = 8 进程
mpirun -np 8 ./navier_stokes_heffte_multigpu 256 256 256 1e-4 100
mpirun -np 8 ./navier_stokes_cufftmp_multinode 256 256 256 1e-4 100
```

### 运行模式（verify / bench）

所有 11 个求解器都支持两种模式，由 NSTEPS 之后的可选参数选择：

| 模式 | 外力 f | 每步 FFT 次数 | 计时结束后输出 | 用途 |
|------|--------|---------------|----------------|------|
| `verify`（默认） | 解析外力，每次 RHS 逐点计算并做 3 次正向 FFT | 48 | L2 误差 + max\|div V\| | 正确性验证，行为与原程序一致 |
| `bench` | f = 0，跳过外力填充 / 3 次 FFT / 复制与缩放，且不分配外力缓冲 | 36 | 总动能 0.5·Σ\|v\|² + max\|div V\| | 性能测试（外力是已知条件，不应计入求解器耗时） |

- 两种模式的初始条件相同（解析 V(t=0)，投影到无散空间）。
- bench 没有解析解，因此不输出 L2；动能和散度用于确认结果有限、无 NaN。
  各库在相同网格/步数下的动能应一致（本机 64³、dt=1e-4、10 步，4 个 CPU 版均为 3.2040343055e+05），可作交叉检查。
- 诊断（L2 / 动能 / 散度）都在计时循环之后计算，不计入耗时。
- max|div V| 按各版本自身的谱系数约定计算：FFTW / cuFFT / cuFFTXt / cuFFTMp 存储归一化系数，
  heFFTe / p3dfft / AccFFT 存储非归一化 DFT 系数（数值大 N 倍），因此不同库之间该值不可直接比较，只用于判断是否接近 0。

**参数解析规则（CPU 版 OMP_THREADS 与 mode 共存，无歧义）：**
NSTEPS 之后的每个参数按类型识别——纯数字为 `OMP_THREADS`，`verify` / `bench`（也接受 `mode=bench`、`--mode=bench`）为模式，
其他任何内容报错退出；每类最多出现一次，顺序不限。GPU 版不接受数字参数。

```bash
mpirun -np 4 ./navier_stokes_accfft 64 64 64 1e-4 10            # verify，默认线程数
mpirun -np 4 ./navier_stokes_accfft 64 64 64 1e-4 10 4          # verify，4 线程（与旧用法完全兼容）
mpirun -np 4 ./navier_stokes_accfft 64 64 64 1e-4 10 bench      # bench，默认线程数
mpirun -np 4 ./navier_stokes_accfft 64 64 64 1e-4 10 4 bench    # bench，4 线程
mpirun -np 4 ./navier_stokes_cufftmp_multinode 256 256 256 1e-4 10 bench
```

### 分阶段计时

两种模式都在原有的 `Avg per step:` 行（格式不变，批处理脚本依赖它）之后输出每步平均的分阶段时间：

```
  Avg per step:    0.0253 s
  Phase breakdown (avg per step, max over ranks):
    FFT (incl. transpose/comm): 0.018910 s
    Spectral pointwise:         0.005548 s
    Physical pointwise:         0.001225 s
    Forcing f (fill+3 FFT):     0.000000 s
    Sum of phases:              0.025684 s
    Total (wall, per step):     0.025306 s
    FFTs per step:              36
```

- **FFT**：所有求解器内的正/逆变换（含库内转置通信）；AccFFT / p3dfft 在逆变换前为保护输入所做的复制也计入此项。
- **Spectral pointwise**：旋度、粘性项、投影、RHS 合并、归一化缩放、RK4 组合与复制等。
- **Physical pointwise**：实空间归一化与叉乘 v×rot(v)。
- **Forcing**：外力填充 + 其 3 次 FFT + 复制/缩放（bench 下为 0）。外力的 FFT 不计入 FFT 项，因此两种模式的 FFT 项可直接比较。
- 多进程时每一项分别对各进程取最大值，因此 Sum of phases 可能略大于 Total。
- CPU 版用主机墙钟；单 GPU / 每进程一 GPU 的版本（cuFFT、cuFFTMp、heFFTe GPU）用默认流上的 cudaEvent，不额外同步；
  cuFFTXt（单进程多 GPU）在每个计时点前同步全部 GPU（复用已有的 SYNC 点，bench 模式不增加同步）。

---

## 验证与结果

### 精度验证

各版本在 Taylor-Green 涡解析解上的 L2 误差（10 步，$dt=5\times10^{-5}$）：

| 实现 | L2 误差 |
|------|---------|
| FFTW MPI | $\sim 10^{-14}$ |
| heFFTe | $\sim 10^{-14}$ |
| p3dfft | $\sim 10^{-14}$ |
| AccFFT | $\sim 2.5\times10^{-14}$ |

所有版本均达到机器精度量级，验证了实现的正确性。

### 可扩展性

- OpenMP：均匀绑定策略在双路服务器上加速效果最佳
- MPI：通过消除逆转置和使用 `FFTW_PATIENT` 计划显著降低计算时间
- MPI+OpenMP 混合：加速比优于纯 MPI 或纯 OpenMP
- 2D pencil 分解（heFFTe/p3dfft/AccFFT）：突破 1D slab 的进程数限制，支持更大规模并行
