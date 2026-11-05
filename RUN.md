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

## 批处理脚本兼容性

`Bench/files/*.sbatch` 通过 `grep "Avg per step:" | awk '{print $(NF-1)}'` 取耗时，该行格式未变。
现有脚本不带模式参数，因此默认运行 `verify`（与修改前行为一致）；
做性能测试时在命令行末尾追加 `bench` 即可（bench 不输出 `L2 error` 行，`parse_l2` 会得到空值）。
