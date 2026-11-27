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

## 2. dtcfl 实验
`dtcfl.sbatch` 和 `dtcfl_parse.sh` 在物理验证通过后再写。
