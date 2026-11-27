# dtcfl — CFL 自适应时间步长（CPU FFTW 版）

求解器的开关（`NS_DT_MODE`、`NS_CFL`、`NS_NU`、`NS_IC=tg`、`NS_NYQ0` 等）见 `RUN.md` 的“粘性系数、Taylor–Green 初值、自适应时间步长”一节。

## 1. 物理验证（在 dtcfl 实验之前）

| 文件 | 内容 |
|---|---|
| `tg_validate.sbatch` | 1 节点，Intel 工具链，FFTW 64×2，256³ Taylor–Green，Re=1600，`NS_NYQ0=1`，fixed dt=0.005，2400 步（T=12），每 0.1 输出一次 E 和 ε |
| `tg_validate_parse.sh` | 与 512³ 谱方法参考解比较：ε 的峰值及其时刻，t=5/8/9/10/12 时的相对偏差；可选：与本机的部分结果逐行比较 |
| `ref/` | 参考数据（van Rees 等 2011，HiOCFD 算例 C3.5），来源见 `ref/README.md` |

**限制：** 没有去混叠（2/3 规则）。Re=1600 时 256³ 的分辨率不足，与参考解的偏差有一部分来自这里。
所有 dtcfl 结论都只针对“与本程序参考解相比的相对收益”，不代表物理上准确的 TG。2/3 去混叠以后另做（`NS_DEALIAS`）。

验证结果（作业 303632，256³）：ε 峰值 0.012932，出现在 t=8.87；参考值 0.012858，出现在 t=8.97（偏差 +0.5%）。t=5–12 范围内，E 偏差 ≤0.7%，ε 偏差 ≤1.9%。

## 2. dtcfl 实验

| 文件 | 内容 |
|---|---|
| `dtcfl.sbatch <cfl\|probe\|ref\|fixed>` | 每个运行组一个作业；每次运行的完整输出写到 `results/dtcfl/<jobid>/<label>.txt` |
| `dtcfl_parse.sh <cfl> <fixed> <ref> [probe]` | 参数都是作业号；输出步数比、墙钟时间比、相对 (a) 的偏差、探测结果，以及 dt(t) 和 u_max(t) 的 CSV |

共同设置：256³，Intel 工具链，FFTW 64×2，MEASURE，`NS_NYQ0=1 NS_IC=tg NS_NU=0.000625`，bench，`NS_DT_PRINT=1`（每步一行 DT）。

| 组 | 内容 |
|---|---|
| cfl | C = 0.5、0.75、0.9，T_END = 20 |
| fixed (b) | 每个 C 一个基线：dt_min 取自同一 C 的 cfl 运行（不含截短的最后一步），NSTEPS = ceil(20/dt_min)，dt = 20/NSTEPS（≤ dt_min，正好结束在 t=20）。dt_min 从 cfl 作业的结果文件读出，不依赖 Slurm 是否还保留该作业的信息；提交时加 `--dependency=afterok:<cfl> --kill-on-invalid-dep=yes` |
| ref (a) | fixed，dt = 0.00125，16000 步 |
| probe | cfl，C = 1.0、1.1、1.25，T_END = 12；C < 0.9 时若出现 BLOWUP，额外标注“可能是混叠导致” |

- **主指标是步数比**（fixed/cfl），它是确定性的，所以每组只跑 1 次；墙钟时间只作为佐证。
- **t=10 的值不单独跑 T=10**，而是从同一次 T=20 运行的 DT 行读出：E 和 ε 用 4 点三次插值，误差为 O(dt⁴)；到 t=10 的步数取开始时刻 t<10 的步数。
- **t=20 的值**取自 FINAL 行。
