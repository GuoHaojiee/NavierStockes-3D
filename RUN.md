# 运行指南

## 参数格式

```
mpirun -np <NP> ./<程序> NX NY NZ dt NSTEPS [OMP_THREADS]
```

| 参数 | 含义 |
|------|------|
| `NP` | MPI 进程数 |
| `NX NY NZ` | 三维网格尺寸 |
| `dt` | 时间步长 |
| `NSTEPS` | 时间步总数 |
| `OMP_THREADS` | OpenMP 线程数（可选，仅 CPU 版，默认系统最大值） |

所有版本完成全部时间步后输出一次 L2 误差。

---

## CPU 版本

```bash
mpirun -np 4 ./navier_stokes_periodic   64 64 64 1e-4 100 4   # FFTW MPI
mpirun -np 4 ./navier_stokes_heffte_v2  64 64 64 1e-4 100 4   # heFFTe
mpirun -np 4 ./navier_stokes_p3dfft     64 64 64 1e-4 100 4   # p3dfft
mpirun -np 4 ./navier_stokes_accfft     64 64 64 1e-4 100 4   # AccFFT
```

---

## GPU 版本

### 单节点单 GPU

```bash
./navier_stokes_cufft 64 64 64 1e-4 100
```

### 单节点多 GPU

```bash
# cuFFTXt（无需 MPI，自动使用节点上所有 GPU）
./navier_stokes_cufftxt 128 128 128 1e-4 100

# heFFTe + cuFFT（每个 MPI 进程占用一块 GPU）
mpirun -np 4 ./navier_stokes_heffte_gpu1  128 128 128 1e-4 100
mpirun -np 4 ./navier_stokes_heffte_mgpu  128 128 128 1e-4 100

# cuFFTMp
mpirun -np 4 ./navier_stokes_cufftmp_mgpu 128 128 128 1e-4 100
```

### 多节点多 GPU

```bash
# 示例：2 节点，每节点 4 GPU，共 8 进程
mpirun -np 8 ./navier_stokes_heffte_multigpu   256 256 256 1e-4 100
mpirun -np 8 ./navier_stokes_cufftmp_multinode 256 256 256 1e-4 100
```
