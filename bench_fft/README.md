# bench_fft — 纯 FFT 基准（v2，2026‑11‑13）

只做 FFT、不带 Navier–Stokes 物理的小程序，用来回答三个问题：

| 作业 | 回答的问题 | 时长（估计） |
|---|---|---|
| `fb1_cufftmp.sbatch` | cuFFTMp 离开单节点后时间花在哪；IB 带宽实际用了多少；官方文档里的多节点开关有没有用 | 20–30 分钟 |
| `fb2_heffte_gpu.sbatch` | heFFTe GPU 版慢，有多少来自布局 / reorder / 批处理这些可调参数 | 30–40 分钟 |
| `fb3_cpu.sbatch` | heFFTe CPU 版比 FFTW 慢一倍，是不是因为砖块分解（4 次重排） | 15–25 分钟 |

## 使用

```bash
# 0) 只需一次：核对 fb_common.sh 里的 srun 写法与你们现有脚本一致
grep -n "srun" ~/guohaojie/runs/dD*.sbatch ~/guohaojie/runs/fl_scan.sbatch | head
head -20 ~/guohaojie/build/bind_hca.sh

# 1) 编译（计算节点、GCC 容器内）
source ~/guohaojie/runs/common.sh
srun -N1 -n1 -c16 --exclude=cn34,cn18 $M bash /work/src/NavierStockes-3D/bench_fft/build_fb.sh

# 2) 提交。fb1 最重要，先交它
cd ~/guohaojie/src/NavierStockes-3D/bench_fft
sbatch fb1_cufftmp.sbatch
sbatch fb3_cpu.sbatch
sbatch fb2_heffte_gpu.sbatch

# 3) 看结果
bash fb_parse.sh $(ls -t ~/guohaojie/logs/fb1_*.out | head -1)
bash fb_parse.sh $(ls -t ~/guohaojie/logs/fb3_*.out | head -1)
bash fb_parse.sh $(ls -t ~/guohaojie/logs/fb2_*.out | head -1)
```

每个作业的第一步都是小规模试跑，日志里必须出现 `[check] ... (OK)`。

## fb1 的表怎么读

时间单位 ms / 每个 FFT；B 单位 GB/s / 每节点；百分数相对 200 GB/s。

| 列 | 含义 |
|---|---|
| `dist` | 一次分布式 FFT（正、逆平均）。`fwd` 和 `inv` 应当接近，否则某个方向多做了一次转置 |
| `local` | 本地 FFT 部分的估计（普通 cuFFT，无通信） |
| `reshape` | 只做一次转置、不做 FFT（cuFFTMp 独立 reshape 接口）——最直接的带宽测量 |
| `a2a` | 同样数据量的 `MPI_Alltoall`（GPU 缓冲区）——基于 MPI 的库（heFFTe）能达到的水平 |
| `B_dist` | D / dist，下限 |
| `B_comm` | D / (dist − local)，扣除本地部分后的估计 |
| `B_resh` | D / reshape，直接测量 |
| `switches` | 该次运行打开的开关：`pack` = `CUFFT_RESHAPE_USE_PACKING=1`，`sl1` = `NVSHMEM_IB_SL=1`，`ibgda` = `NVSHMEM_IB_ENABLE_IBGDA=1` |

D = V/n²·(n−1)，V = NX·NY·(NZ/2+1)·16 字节。日志开头还会打印"100% 带宽下这些数据在线路上至少要走多少 ms"。

给导师的数字：2 / 4 节点那几行的 `B_resh` 和 `B_comm` 的百分比。

## v2 相对上一版改了什么

- **cuFFTMp**：`nvshmem_free` 移到 `cufftDestroy` 之前（文档：`cufftDestroy` 会结束 NVSHMEM；求解器里就是这个顺序）；新增 `[R]` 纯转置测量；新增开关扫描；本地测试 / MPI 测试显存不够时跳过而不是中止；每个阶段结束立刻输出 `RES_x` 行，后面的阶段出错也不丢前面的数。
- **heFFTe**：布局改名为 `xslab|yslab|zslab|xpencil|ypencil|zpencil|brick`（上一版的 `pencil` 实际是砖块）；打印每个阶段的进程网格和 MPI 重排次数；单进程且不重排时不走批处理接口（heFFTe 2.4.1 的 `compute_transform` 在这种情况下偏移算错）。
- **新增 `est_vs_meas.cpp`**：串行小程序，量化 heFFTe 的 FFTW 规划方式（`FFTW_ESTIMATE` + NULL 数组）对本地 1D FFT 的影响。
- **脚本**：每次运行记录退出码；挂住的运行 12 分钟后被杀掉，作业继续；IBGDA 先用一次 4 分钟的探测。

## 没有验证的部分

- `fftbench_cufftmp.cu` 和 heFFTe 的 GPU 版没有在 GPU 上编译过（写代码的环境没有 CUDA）。主机端逻辑用模拟头文件编译并跑通过；`fftbench_heffte.cpp` 的 CPU 版对着 heFFTe 2.4.1 头文件做过语法检查；`fftbench_fftw.cpp`、`est_vs_meas.cpp`、`fb_parse.sh` 实际运行过。
- `build_fb.sh` 遇到 cuFFTMp 版编译失败会自动去掉可选测试重试；三次都失败时把 `build/bin_fb/fb_cufftmp*.log` 发回来。
- `fb_common.sh` 里的 `srun` 写法、`bind_hca.sh` 的调用方式是按交接文档推测的。
- `CUFFT_RESHAPE_USE_PACKING` 在文档里写在独立 reshape 接口下面，是否也作用于 FFT plan 内部的转置，文档没说——fb1 的 `dist` 列会直接显示。
