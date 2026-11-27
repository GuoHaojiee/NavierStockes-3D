# Taylor–Green 涡 Re=1600 参考数据

`spectral_Re1600_512.gdiag`：512³ 伪谱 DNS（不可压），Taylor–Green 涡，Re = 1600，t ∈ [0, 20]，Δt = 0.01，共 2000 行。

| 列 | 含义 |
|---|---|
| 1 | t |
| 2 | 动能 E = 0.5⟨\|u\|²⟩（E(0) = 0.125） |
| 3 | 耗散率 −dE/dt |
| 4 | 涡量拟能 Ω = 0.5⟨\|ω\|²⟩，于是 2νΩ = ν⟨\|ω\|²⟩（ν = 1/1600），等于求解器输出的 `eps` |

关键值：−dE/dt 的峰值为 0.012858，出现在 t = 8.97；E(10) = 0.07448。

## 来源
- 文件下载自 High-Order CFD Workshop 的算例 C3.5（Taylor–Green vortex，Re=1600）：
  <https://cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag>（2026-11-27 下载，sha256 `60bfeaee2dd32de1b9e7171c70cf3774b6053d82a785d526c055547a0757017a`）。
- 算例说明：<https://cfd.ku.edu/hiocfd/case_c3.5.pdf>。
- 数据出自 W. M. van Rees, A. Leonard, D. I. Pullin, P. Koumoutsakos, "A comparison of vortex and pseudo-spectral
  methods for the simulation of periodic vortical flows at high Reynolds numbers", *J. Comput. Phys.* 230 (2011) 2794–2805。
- 文件内容未做任何修改。
