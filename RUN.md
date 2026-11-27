# 运行指南

## 参数格式

```
mpirun -np <NP> ./<CPU 程序> NX NY NZ dt NSTEPS [OMP_THREADS] [verify|bench]
mpirun -np <NP> ./<GPU 程序> NX NY NZ dt NSTEPS [verify|bench]
```

| 参数 | 含义 |
|------|------|
| `NP` | MPI 进程数 |
| `NX NY NZ` | 三维网格尺寸 |
| `dt` | 时间步长 |
| `NSTEPS` | 时间步总数 |
| `OMP_THREADS` | OpenMP 线程数（可选，仅 CPU 版，默认系统最大值） |
| `verify` / `bench` | 运行模式（可选，默认 `verify`） |

### 可选参数的解析规则（无歧义）

NSTEPS 之后的参数按**类型**识别，与位置无关：

- 纯数字（如 `4`）→ `OMP_THREADS`（仅 CPU 版；GPU 版给数字会报错）
- `verify` / `bench`（也接受 `mode=bench`、`--mode=bench`）→ 运行模式
- 其他内容、或同一类参数出现两次 → 打印错误与用法并退出

因此 `... 10`、`... 10 4`、`... 10 bench`、`... 10 4 bench` 都合法，且旧的调用方式（只给 OMP_THREADS）行为不变。

### 两种模式

| | `verify`（默认） | `bench` |
|---|---|---|
| 外力 f | 解析外力，每次 RHS 计算 | f = 0（跳过外力填充、3 次正向 FFT、复制/缩放；不分配外力缓冲） |
| 每步 FFT | 48 | 36 |
| 初始条件 | 解析 V(t=0) | 解析 V(t=0) |
| 结束后输出（不计时） | L2 误差、max\|div V\| | 总动能 0.5·Σ\|v\|²、max\|div V\| |

两种模式都会在 `Avg per step:` 行（格式不变）之后输出分阶段计时：
FFT（含转置通信）/ 谱空间逐点 / 物理空间逐点 / 外力（bench 为 0），均为每步平均、各进程取最大值。

---

## CPU 版本

```bash
# verify（正确性）
mpirun -np 4 ./navier_stokes_periodic   64 64 64 1e-4 100 4   # FFTW MPI
mpirun -np 4 ./navier_stokes_heffte_v2  64 64 64 1e-4 100 4   # heFFTe
mpirun -np 4 ./navier_stokes_p3dfft     64 64 64 1e-4 100 4   # p3dfft
mpirun -np 4 ./navier_stokes_accfft     64 64 64 1e-4 100 4   # AccFFT

# bench（性能）
mpirun -np 4 ./navier_stokes_periodic   64 64 64 1e-4 100 4 bench
mpirun -np 4 ./navier_stokes_accfft     64 64 64 1e-4 100 bench   # 默认线程数
```

---

## GPU 版本

### 单节点单 GPU

```bash
./navier_stokes_cufft 64 64 64 1e-4 100            # verify
./navier_stokes_cufft 64 64 64 1e-4 100 bench
```

### 单节点多 GPU

```bash
# cuFFTXt（无需 MPI，自动使用节点上所有 GPU）
./navier_stokes_cufftxt 128 128 128 1e-4 100 bench

# heFFTe + cuFFT（每个 MPI 进程占用一块 GPU）
mpirun -np 4 ./navier_stokes_heffte_gpu1  128 128 128 1e-4 100 bench
mpirun -np 4 ./navier_stokes_heffte_mgpu  128 128 128 1e-4 100 bench

# cuFFTMp：cufftmp_mgpu 已 deprecated（≥2 GPU 时结果为 NaN，不再维护），单节点也请用 multinode 版
mpirun -np 4 ./navier_stokes_cufftmp_multinode 128 128 128 1e-4 100 bench
```

### 多节点多 GPU

```bash
# 示例：2 节点，每节点 4 GPU，共 8 进程
mpirun -np 8 ./navier_stokes_heffte_multigpu   256 256 256 1e-4 100 bench
mpirun -np 8 ./navier_stokes_cufftmp_multinode 256 256 256 1e-4 100 bench
```

---

## 环境变量

| 变量 | 含义 |
|------|------|
| `NS_SHARE_WORK_AREA=0` | 仅 cuFFTMp 多节点版：不共用 cuFFT 工作区，改回每个计划各自分配。默认共用，以节省显存 |
| `NS_XT_THREADS=1/3/6` | 仅 cuFFTXt 版：同一批 3 次或 6 次变换由几个线程同时发起，每个线程使用自己的一对 plan。默认 1，与原来相同（依次执行）。每个 plan 都有自己的 cuFFT 工作区，大小见 Memory report |

### 粘性系数、Taylor–Green 初值、自适应时间步长（仅 CPU FFTW 版 `navier_stokes_periodic`）

都不设置时，结果和输出都与原来逐位、逐字相同。设置了其中任何一个，才会多输出 `DTCFL_CONFIG`、`DT ...`、`FINAL`、`DTCFL_SUMMARY` 这几行。

| 变量 | 默认 | 含义 |
|------|------|------|
| `NS_NU=<ν>` | 1 | 粘性系数。方程为 ∂v/∂t = νΔv + v×rot v − ∇p + f；verify 的外力里 Δv 项同样乘 ν，所以解析解对任意 ν 都成立。`NS_IC=tg` 且没有设置 `NS_NU` 时，默认 1/1600（只作保底，脚本里应显式设置） |
| `NS_IC=poly\|tg` | poly | 初值。poly 为原解析解；tg 为 Taylor–Green：u = sin x cos y cos z，v = −cos x sin y cos z，w = 0，E(0) = 0.125。tg 没有解析解，只能配合 `bench` 使用（f = 0） |
| `NS_DT_MODE=fixed\|cfl` | fixed | fixed：dt 和 NSTEPS 取自命令行。cfl：每步自适应确定 dt，命令行的 dt 不使用，NSTEPS 作为步数上限（达到上限仍未到 T_END 时报错，退出码 2） |
| `NS_CFL=<C>` | 0.5 | cfl：对流界 dt_c = C / max_x(\|u\|/dx + \|v\|/dy + \|w\|/dz) |
| `NS_CFL_VISC=<Cv>` | 1.0 | cfl：粘性界 dt_v = Cv / (ν·k²max)，k²max = (NX/2)² + (NY/2)² + (NZ/2)² |
| `NS_DT_MAX=<dt>` | 无 | cfl：dt 的上限 |
| `NS_T_END=<T>` | cfl 下必须设置 | cfl：按物理时间结束，最后一步截短，正好落在 T 上 |
| `NS_NYQ0=0\|1` | 0 | 1 时去掉三个方向的 Nyquist 平面（kx = NX/2、ky = NY/2、kz = NZ/2，只对偶数 N 有效）。投影后的初值以及每个 RK 级的更新，都把这些模态（V 和 acc）置 0，所以求导和投影里 Nyquist 的 k 取什么值不再起作用，两者自然一致。默认 0 保持原约定（Nyquist 处 k = +N/2），结果逐位不变。原约定会破坏 Nyquist 平面的共轭对称性：在 64³ 的 TG（Re=1600）中，从 t≈3 开始出现虚假耗散，E(10) 为 0.0026，而用 NS_NYQ0=1 时为 0.052（文献参考值 0.074）。verify 解在 Nyquist 附近几乎没有能量，L2 基本不变 |
| `NS_DT_PRINT=<K>` | cfl 100；fixed 不打印 | 每 K 步打印一行 `DT step t dt umax E eps Ceff lim`。数值取自该步开始时刻：E = 0.5⟨\|u\|²⟩，eps = ν⟨\|ω\|²⟩，Ceff = dt·max Σ\|u_i\|/dx_i，lim 表示哪一项限制了 dt（conv、visc、dtmax、tend，fixed 模式为 fixed） |

cfl 模式下，dt = min(dt_c, dt_v, NS_DT_MAX, T_END − t)。
- 不为求 max 额外做 FFT：在 RK 第 1 级逆变换之后扫描一遍实空间速度（即 u^n）和旋度（即 ω^n）。
- 扫描和 Allreduce 单独计时，打印为 `Scan ...`，不计入各阶段时间，但计入墙钟时间。fixed 模式只在打印步扫描。

**C 与稳定界的换算。** 对流项在最高波数 k_max,i = π/dx_i 处的特征值是 i·Σ u_i k_max,i。RK4 在虚轴上的稳定界约为 2.83，因此要求 π·C ≲ 2.83，即 **C ≲ 0.9**。
- 实际流场的 max Σ|u_i| 不一定出现在高波数有能量的地方，所以实际稳定界可能略高。
- 粘性界：RK4 在实轴上的稳定界约为 2.785，即 Cv ≲ 2.785。

**取 min 并不严格保证稳定。** RK4 的稳定域不是矩形；对流和粘性同时接近各自上限时，两者的组合特征值可能落在稳定域之外。本次 TG（ν = 1/1600）和 verify（ν = 0.01）的粘性项都远小于粘性界，全程都是对流项在限制 dt（见 `DTCFL_SUMMARY` 中的 `lim_conv`、`lim_visc`），不受这一点影响。

**发散判据。** 满足以下任一条件即视为发散：E 或 max Σ|u_i|/dx_i 为非有限值，E > 1000·E(0)，或 dt ≤ 0。
- 发散时打印 `BLOWUP ...`，以及最后 8 次扫描的 `BLOWUP_HIST`（t、dt、umax、E、eps、Ceff），退出码为 2。
- fixed 模式只在打印步检查是否发散。

结束时输出：
- `FINAL t E eps umax maxS`：t = T 时的诊断，不计时。
- `DTCFL_SUMMARY`：步数、RHS 调用次数（4 × 步数）、FFT 次数（bench 每步 36 次，verify 每步 48 次）、`dt_min`（不含截短的最后一步）、`dt_maxused`、扫描总耗时、墙钟时间、各项限制 dt 的步数、blowup、cap_hit。

```bash
# 64^3 verify，nu = 0.01，自适应到 T = 1（4 阶：C = 0.5 → 1.0 时误差约增大 16–19 倍）
NS_NU=0.01 NS_DT_MODE=cfl NS_CFL=0.5 NS_T_END=1 mpirun -np 2 ./navier_stokes_periodic 64 64 64 0 100000 4 verify
# Taylor–Green，Re = 1600
NS_IC=tg NS_NU=0.000625 NS_DT_MODE=cfl NS_CFL=0.9 NS_T_END=10 mpirun -np 2 ./navier_stokes_periodic 64 64 64 0 100000 4 bench
```

---

## 批处理脚本兼容性

`Bench/files/*.sbatch` 通过 `grep "Avg per step:" | awk '{print $(NF-1)}'` 取耗时，该行格式未变。
现有脚本不带模式参数，因此默认运行 `verify`（与修改前行为一致）；
做性能测试时在命令行末尾追加 `bench` 即可（bench 不输出 `L2 error` 行，`parse_l2` 会得到空值）。
