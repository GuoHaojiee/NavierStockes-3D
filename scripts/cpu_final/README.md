# CPU 正式测试：Intel 工具链，slab，512³

## 测试设置

CPU 正式测试只用 Intel 工具链（icpx + Intel MPI，v2 镜像），分解方式全部用 slab，网格 512³。

| 库 | 程序 |
|---|---|
| FFTW | `/work/build/bin_intel/ns_fftw` |
| AccFFT | `/work/build/bin_intel/ns_accfft` |
| P3DFFT | `/work/build/bin_p3d/intel/ns_p3dfft` |
| heFFTe | `/work/build/bin_intel/ns_heffte_v2_mkl`，加 `NS_HEFFTE_REORDER=1` |

公共参数：

- 环境变量：`NS_DECOMP=slab`、`NS_FFTW_PLAN=measure`、`KMP_BLOCKTIME=0`、`MKL_NUM_THREADS=<线程数>`；
- 512³，dt=1e-6，10 步，bench 模式；
- 动能参考值 1.7950250065e+08；
- 每个配置跑 3 次，重复放外层循环。

各节点数的布局和启动方式：

| 节点数 | 每节点布局 | 启动方式 |
|---|---|---|
| 1 | 128×1、64×2、1×128 | 单个 srun 任务，容器内 `mpirun`（fork + shm），和 cmp_toolchain.sbatch 单节点 Intel 的方式相同。1×128 另加 `/work/tools/numactl --interleave=all`、`OMP_PLACES=cores`、`OMP_PROC_BIND=spread` |
| 2、4 | 128×1、64×2 | `irun_mn`，即 hello_2node E6d 的设置：pmi2、mlx、HPC-X UCX 1.17、`I_MPI_ADJUST_ALLTOALL=3` |

## 文件

| 文件 | 作用 |
|---|---|
| `cpu_final.sbatch <1\|2\|4> [dry]` | 每个节点数一个作业，用 `sbatch -N <n>` 提交，节点数和参数不一致时直接退出。每次运行向 CSV 追加一行 |
| `ucx_probe.c` | LD_PRELOAD 用的小库，每个作业在镜像里用 icx 编一次。程序退出时，MPI rank 0 打印 `UCXPROBE version=<实际加载的 libucp 版本\|none> lib=<路径>`。单节点 shm 不加载 UCX，记为 none |
| `cpu_final_parse.sh [csv]` | 汇总 CSV：每个配置的中位数和范围、每个 (库, 节点数) 的最佳布局、强扩展表，以及有问题的运行 |
| `plot_cpu_final.py [csv] [输出目录]` | 画 `fig_cpu_1node` 和 `fig_cpu_scaling` 两张图（png 和 pdf），需要 matplotlib |

### CSV

正式结果写到 `~/guohaojie/results/cpu_final.csv`，dry-run 写到 `cpu_final_dry.csv`。列依次为：

```
commit,jobid,nodelist,lib,nodes,ranks_per_node,threads,rep,step_s,fft_s,plan_s,energy,energy_ok,ucx_version,ucx_warn,exit_code,omp_threads,mkl_threads
```

- **ucx_version**：rank 0 实际加载的 UCX 版本，单节点 shm 时为 none。
- **ucx_warn**：多节点运行时如果版本不是 1.17.0，这一列为 `WARN`，日志里同时打印 `WARNING: UCX <版本> loaded, expected 1.17.0`。
- **omp_threads、mkl_threads**：求解器启动时打印的实际线程数，即 `Threads (runtime): omp_get_max_threads()=… omp_get_num_threads(in parallel)=…`；heFFTe-MKL 另有 `mkl_get_max_threads()=…`。和设定的线程数不一致时，日志里打印 WARNING。
- **nodelist**：里面的逗号换成了 `;`。

### 求解器的改动

改动后需要重新编译求解器：

- 四个求解器都加了一行 `Threads (runtime): …`；
- heFFTe 新增 `heFFTe plan time: … s`，即 fft3d_r2c 构造的时间。MKL 后端在构造时提交 DFTI 描述符；FFTW 后端的计划要到第一次执行时才建，所以 FFTW 后端的这个时间不包括 FFTW 规划；
- P3DFFT 自己会打印 `Running on N threads`，这一行也保留在日志里。

## 时长估算

512³，10 步；每次运行包括容器启动、MEASURE 规划和计算。

| 节点数 | 运行次数 | 每次 | 合计 | `-t` |
|---|---|---|---|---|
| 1 | 4 库 × 3 布局 × 3 次 = 36 | 1.5–3 分钟（1×128 规划更慢） | 55–110 分钟 | 02:30:00 |
| 2 | 4 × 2 × 3 = 24 | 1–2 分钟 | 25–50 分钟 | 01:15:00 |
| 4 | 24 | 1–1.5 分钟 | 25–40 分钟 | 01:00:00 |
| dry | 8 次（1 节点）或 4 次（2 节点） | 很短 | 约 10 分钟 | 00:20:00 |

## 超算上的运行顺序

```bash
cd ~/guohaojie/src/NavierStockes-3D && git pull --ff-only
source scripts/intel/intel_common.sh
```

**0. 重新编译求解器**

求解器加了线程数和 heFFTe 规划时间的打印，需要重编，库不用重编。

```bash
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/build_intel_solvers.sh
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:20:00 $MI env P3O_SOLVER_ONLY=1 bash /work/src/NavierStockes-3D/scripts/p3dfft/build_p3dfft_opt.sh intel
```

成功时第一条命令有 8 个 OK，第二条有 `OK intel_check` 和 `OK intel_solver`。

**1. dry-run：64³，1 个节点和 2 个节点各一次**

```bash
cd scripts/cpu_final
sbatch -N 1 -J cpufdry1 -t 00:20:00 cpu_final.sbatch 1 dry
sbatch -N 2 -J cpufdry2 -t 00:20:00 cpu_final.sbatch 2 dry
grep -E "^numactl|^>>>|Threads \(runtime\)|Running on|UCXPROBE|intel_mn_env:|^WARNING|^CSV" $(ls -t ~/guohaojie/logs/cpuf_cpufdry1_*.out | head -1)
grep -E "^>>>|Threads \(runtime\)|Running on|UCXPROBE|intel_mn_env:|^WARNING|^CSV" $(ls -t ~/guohaojie/logs/cpuf_cpufdry2_*.out | head -1)
bash cpu_final_parse.sh ~/guohaojie/results/cpu_final_dry.csv
```

dry-run 通过的标志：

- **1 节点**：
  - 布局是 64×2 和 1×128；
  - 有一行 `numactl: … OK`；
  - 64×2 的运行显示 `omp_get_max_threads()=2 omp_get_num_threads(in parallel)=2`，1×128 的显示 128；heFFTe 另有 `mkl_get_max_threads()=128`；
  - P3DFFT 打印 `Running on 128 threads`；
  - 每次运行都有 `UCXPROBE version=none`，CSV 的 ucx 列为 none；
- **2 节点**：
  - 每节点 32×4，共 64 个进程；
  - 有 `intel_mn_env: … UCX=/work/opt_ucx/hpcx/lib/ucx …`；
  - 每次运行都有 `UCXPROBE version=1.17.0 lib=/work/opt_ucx/hpcx/lib/libucp.so.0.0.0`；
  - 没有 `WARNING: UCX`；
- **两个都要满足**：
  - 动能都是 3.3526466678e+05，`energy_ok=yes`；
  - 每行 `exit_code=0`；
  - parse 最后一段的问题运行数为 `(0)`。

**2. 正式测试**

三个作业可以同时提交。

```bash
sbatch -N 1 -J cpuf1 -t 02:30:00 cpu_final.sbatch 1
sbatch -N 2 -J cpuf2 -t 01:15:00 cpu_final.sbatch 2
sbatch -N 4 -J cpuf4 -t 01:00:00 cpu_final.sbatch 4
```

**3. 结果**

```bash
bash cpu_final_parse.sh                                 # ~/guohaojie/results/cpu_final.csv
CPF_COMMIT=<hash> bash cpu_final_parse.sh               # 只看某个 commit 的运行
python3 plot_cpu_final.py ~/guohaojie/results/cpu_final.csv ~/guohaojie/results
```

`cpu_final_parse.sh` 的输出：

- 每个 (库, 节点数, 布局)：step 中位数（最小–最大）、FFT 和规划时间、`能量正确次数/运行次数`、UCX 版本；
- 每个 (库, 节点数) 的最佳布局：只在所有运行都能量正确、退出码为 0 的布局里选；
- 强扩展表：库 × 1、2、4 节点，每格是最佳布局的每步时间 `s/step`、加速比 `S`（以 1 节点最佳配置为基准）和并行效率 `E`；
- 能量不对、退出码不为 0、UCX 版本不对的运行。

两张图：

- `fig_cpu_1node`：单节点分组柱状图，4 个库 × 3 种模式，柱顶标数值；
- `fig_cpu_scaling`：强扩展，横轴节点数 1、2、4，纵轴每步时间（对数轴），4 个库各一条线，每个点用该节点数下的最佳布局。

图上只有数据，没有标题和说明文字。颜色只用深蓝 1F4E79、深红 8B1E1E 和灰色；heFFTe 用深蓝虚线和菱形标记，和 FFTW 区分。字体依次找 Calibri、Carlito、DejaVu Sans。

## 本机测过的部分

- 用本机 oneAPI 编了 heFFTe 的 MKL 和 FFTW 两个后端，以及 Intel 版求解器。新加的线程行正确，heFFTe-MKL 显示 `mkl_get_max_threads()`；64³ 的动能是 3.3526466678e+05。
- gnu 版 P3DFFT 打印 `Running on 4 threads` 和新的线程行。
- ucx_probe：
  - 只用 shm 时为 none；
  - shm:ofi 时显示系统 UCX 1.16.0；
  - 把 HPC-X 的 UCX 放到 LD_LIBRARY_PATH 前面时显示 1.17.0。
- cpu_final.sbatch 用假 srun 跑通了：
  - 1 节点（8×2 和 1×8）：numactl、线程数、CSV 都正确；
  - 2 节点（irun_mn）：本机加载的是 UCX 1.16.0，所以正确地打出了 `WARNING: UCX 1.16.0 loaded, expected 1.17.0` 并在 CSV 里标 `WARN`；
  - 节点数和参数不一致时报错退出。
- parse 和 plot 用一份虚构的完整 CSV 测过。

**没测**：v2 镜像里的真实运行、numactl 的 libnuma 路径、Intel 版 P3DFFT（本机没有 ifx），以及 512³ 的耗时。
