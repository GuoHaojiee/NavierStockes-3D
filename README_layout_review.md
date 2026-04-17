# 数组布局审查报告

## 汇总表

| 文件 | 实空间索引 | 含 padding | Pencil 方向对齐 | 归一化位置 |
|------|-----------|-----------|----------------|-----------|
| fftw.cpp      | `(i*ny+j)*(2*(nz/2+1))+k` | 是（R2C padding） | 正确（z stride=1，R2C 沿 z） | 正变换后手动 `/N`，逆变换 out-of-place 不需要 |
| heffte_v2.cpp | `box_idx`: x stride=1 | 不涉及（heFFTe 内部管理） | 正确 | `backward(scale::full)` 自动 /N |
| p3dfft.cpp    | `(i-is0)+(j-is1)*isize[0]+(k-is2)*isize[0]*isize[1]` | 是（`memsize[2]` 分配，尾部填充） | **错误**（x stride=1 最快，但循环 i 为最外层、k 为最内层） | `btran_c2r` 后手动 `/N` |
| accfft.cpp    | `i*isize[1]*nz + j*nz + k` | 不涉及 | 正确（k/z stride=1，R2C 沿 z） | `c2r` 后手动 `/N` |
| cufft.cu      | `i*NY*NZ + j*NZ + k` | 不涉及（无 FFTW padding） | 正确（k/z stride=1，R2C 沿 z） | D2Z 后 `kernel_scale_cplx /N`，viscous 在 Z2D 销毁 V_c 之前 |

---

## 逐文件分析

### fftw.cpp
[正常] 未发现布局问题。

- 实空间 padding 正确：行 350/416/696/738/822 全部用 `(i*ny+j)*(2*(nz/2+1))+k`，步长为 `2*(nz/2+1)`（FFTW R2C 约定）。
- 谱空间：行 201/236 用 `(i*ny+j)*nz_c+k`，`nz_c=nz/2+1`，截断正确。
- z（kz）为最快变化维度，与 R2C 方向一致。

---

### heffte_v2.cpp
[正常] 未发现布局问题。

- `box_idx`（行 136–139）：`(i-low[0]) + (j-low[1])*size[0] + (k-low[2])*size[0]*size[1]`，x 最快；heFFTe 内部转置到所需 pencil 方向，用户无需处理。
- 实空间向量用 `nr = inbox_r.count()`（无 padding），谱空间用 `nc = outbox_c.count()`。
- `world_c = {{0,0,0},{nx-1,ny-1,nz/2}}`（行 483），R2C z 截断正确。
- 归一化：`forward(scale::none)` + `backward(scale::full)` 自动 /N，nl 的 forward 也用 `scale::none`，约定一致。

---

### p3dfft.cpp

正确的部分：
- `idx_r`（行 134–138）公式正确：x stride=1，y stride=`isize[0]`，z stride=`isize[0]*isize[1]`。数据连续无行间 padding（`memsize[2]` 只是尾部额外分配）。
- 内存分配 `isize[0]*isize[1]*memsize[2]`（行 627）：`memsize[2]≥isize[2]`，额外空间在尾部，不影响 `idx_r` 的寻址正确性。
- 归一化：`btran_c2r` 后手动 `/N`（行 344–351, 690–693），正确。

**问题（已修复）：**

[问题] `p3dfft.cpp:354–366` — 叉乘循环 `i(x)→j(y)→k(z)`，内层 k 步长为
`isize[0]*isize[1]`（约 128×64=8192 doubles=64KB/step），而 p3dfft Fortran
列主序中 x（stride=1）应为最内层。修复：改为 `k→j→i`（z 最外、x 最内）。

[问题] `p3dfft.cpp:424–437` — 强迫项填充循环同样为 `i→j→k`，同上问题。

[问题] `p3dfft.cpp:651–663` — 初始条件循环 `i→j→k`，同上。

[问题] `p3dfft.cpp:697–712` — 误差计算循环 `i→j→k`，同上。

---

### accfft.cpp
[正常] 未发现布局问题。

- `idx_r`（行 47）：`i*isize[1]*nz + j*nz + k`，k（z）stride=1 最快，与 AccFFT R2C 沿 z 一致。
- `local_size_r = isize[0]*isize[1]*nz`（行 560）与索引公式一致。
- `idx_c`（行 51）：`i*osize[1]*osize[2] + j*osize[2] + k`，`osize[2]=nz/2+1`，截断正确。
- 归一化：行 334–338 手动 `/N`，顺序正确。

---

### cufft.cu
[正常] 未发现布局问题。

- 实空间（行 178–184）：`idx = i*NY*NZ + j*NZ + k`，k（z）stride=1，与 R2C（cuFFT D2Z 沿 z）一致。无 FFTW 式 padding（`NR=NX*NY*NZ`，行 21）。
- 谱空间（行 200–203）：`idx = i*NY*NZC + j*NZC + k`，`NZC=NZ/2+1`，截断正确。
- 归一化顺序关键路径（`compute_rhs`，行 411–440）：
  1. `kernel_compute_viscous`（读 V_c，行 418–420）
  2. `compute_nonlinear`（Z2D **销毁** V_c，行 423）
  3. 强迫项 D2Z + scale（行 426–432）
  4. 合并 + 投影

  viscous 在 Z2D 之前计算，顺序正确（行 417 注释已明确）。

- `compute_diagnostics` 中 V_c 被 Z2D 销毁后，D2Z+scale 重建（行 513–518），之后 rk4_step 使用重建后的 V_c，正确。

---

## p3dfft padding 专项分析

| 位置 | 描述 | 判断 |
|------|------|------|
| 行 627 `local_size_r` 分配 | `isize[0]*isize[1]*memsize[2]`，用 `memsize[2]` 保证足够内存 | 安全（尾部填充，不影响 `idx_r`） |
| 行 134–138 `idx_r` | z 步长用 `isize[0]*isize[1]`（不是 `memsize[0]*isize[1]`） | 安全（Cp3dfft_setup 传 `nx_in=nx`，无 x padding，`memsize[0]=isize[0]`） |
| 行 344 normalize loop | 遍历 `isize[0]*isize[1]*isize[2]` 个元素（flat 索引） | 安全（数据连续，无行间 padding） |
| 行 690 normalize loop | 同上 | 安全 |
| 行 354–366 cross product | 通过 `idx_r(ctx,i,j,k)` 访问，地址正确 | 安全（正确性无误，但循环顺序有性能问题，**已修复**） |

---

## 优先级排序

**【高】** `p3dfft.cpp:354–366, 424–437, 651–663, 697–712` — 用户侧所有三重循环均为
`i(x)→j(y)→k(z)`，而 p3dfft Fortran 列主序的最快维度是 x（stride=1）。内层 k
步长 = `isize[0]×isize[1]` ≈ 128×64 = 8192 doubles = 64 KB/step，每次内层迭代跨越
整个 L1/L2 缓存，预计 20–30× 性能下降。

**修复方案**：将所有实空间三重循环改为 `k→j→i`（z 最外，x 最内），使内层循环以
stride=1 连续访问内存。
