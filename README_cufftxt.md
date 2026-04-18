# cuFFTXt 多 GPU 版本说明

## 文件列表

| 文件 | 说明 |
|------|------|
| `NavierStokes_periodic_cufftxt.cu` | cuFFTXt 多 GPU 求解器 |
| `Makefile_cufftxt` | 编译文件 |

---

## 编译

```bash
# 修改 Makefile_cufftxt 中的 ARCH、CUDA_DIR 适配集群环境
make -f Makefile_cufftxt
```

运行：
```bash
./navier_stokes_cufftxt
```

---

## 关键设计说明

### 1. In-place 变换（仅限 in-place）

cuFFTXt 多 GPU 模式**不支持 out-of-place 变换**（即 `cufftXtExecDescriptorD2Z` 的 input 和 output 指针必须相同）。  
因此每个速度/涡量分量只分配一个 `cudaLibXtDesc*` 缓冲区，兼做实空间和频谱空间存储：

- **实空间模式**（D2Z 前或 Z2D 后）：将缓冲区指针转为 `double*`，最后一维步长为 `2*NZC`（标准 in-place 填充）
- **频谱模式**（D2Z 后或 Z2D 前）：将缓冲区指针转为 `GCplx*`，最后一维步长为 `NZC`
- 两种访问方式物理字节完全一致（`sizeof(GCplx) == 2*sizeof(double)`）

### 2. subFormat 字段 hack（内部未文档化行为）

`cudaLibXtDesc` 结构体（来自 `cudalibxt.h`）：
```c
struct cudaLibXtDesc_t {
    int version;
    cudaXtDesc *descriptor;   // 含 nGPUs, GPUs[], data[], size[]
    libFormat library;
    int subFormat;            // ← 状态字段，控制允许哪种方向变换
    void *libDescriptor;
};
```

`subFormat` 枚举值（来自 `cufftXt.h`）：
```c
CUFFT_XT_FORMAT_INPLACE           = 0x02  // 线性分布（实空间），D2Z 的合法输入
CUFFT_XT_FORMAT_INPLACE_SHUFFLED  = 0x03  // shuffled 分布（频谱空间），Z2D 的合法输入
```

**状态转换**：
- `cufftXtExecDescriptorD2Z()` 执行后：`desc->subFormat` 自动变为 `3`（SHUFFLED）
- `cufftXtExecDescriptorZ2D()` 执行后：`desc->subFormat` 自动变为 `2`（INPLACE）

**问题场景**：算法中需要将频谱涡量直接写入 `rot_buf`（不经过 D2Z），然后调用 Z2D。  
此时 `rot_buf->subFormat = 2`（从 `cufftXtMalloc(INPLACE)` 初始化），但 Z2D 要求 `subFormat = 3`，会返回 `CUFFT_INVALID_TYPE` 错误。

**修复方案**：直接写入缓冲区后，手动设置：
```cpp
rot_buf->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;  // = 3
```

类似地，强迫 D2Z 的格式：
```cpp
work_buf->subFormat = CUFFT_XT_FORMAT_INPLACE;  // = 2
```

**注意**：此 hack 仅修改主机端标志位，不移动 GPU 间数据。前提条件是数据已经处于目标格式的正确布局（即"逻辑上"确实是那种格式的数据）。该行为与 CUDA 版本绑定，将来可能失效。

### 3. 流同步（SYNC [A–I]）

cuFFTXt 多 GPU 使用**每 GPU 独立内部 CUDA 流**，与用户代码的默认流（stream 0）并发执行。  
若不显式同步，用户内核（stream 0）写完缓冲区后，cuFFT 内部流立即读该缓冲区，会产生数据竞争。

代码中共有 9 处 `sync_all_gpus()` 调用（标记为 SYNC [A]–[I]）：

| 标记 | 位置 | 原因 |
|------|------|------|
| A | `compute_nonlinear` rot 内核之后 | rot 内核写 rot_buf，Z2D 即将读/写 V_buf 和 rot_buf |
| B | Z2D(V+rot) 之后 | Z2D 异步，cross_product 内核需要 Z2D 的实空间结果 |
| C | cross_product 内核之后 | 内核写 rot_buf，D2Z 即将读 rot_buf |
| D | D2Z(rot) 之后 | D2Z 异步，copy_cplx 内核需要频谱结果 |
| E | `compute_rhs` viscous 内核之后 | viscous 内核读 V_buf，compute_nonlinear 的 Z2D 将写 V_buf |
| F | fill_forcing 内核之后 | 内核写 work_buf，D2Z 即将读 work_buf |
| G | D2Z(work) 之后 | D2Z 异步，copy_cplx 内核需要频谱结果 |
| H | `diagnostics` Z2D(V) 之后 | Z2D 异步，error_sq 内核需要实空间结果 |
| I | D2Z(V) 之后（diagnostics） | D2Z 异步，scale_cplx 内核需要频谱结果 |

### 4. 索引映射假设（**上集群后必须验证**）

**本代码假设**：D2Z 后频谱数据按 X 轴 slab 分布，GPU `g` 持有全局 X 索引 `[g*nx_local, (g+1)*nx_local)`，对应波数：
```
kx = (gi <= NX/2) ? gi : gi - NX    其中 gi = g*nx_local + lx
ky = (j  <= NY/2) ? j  : j  - NY
kz = kc   (0..NZC-1)
```

**NVIDIA 官方未文档化该分布规则**。若实际 `INPLACE_SHUFFLED` 布局不是 X-slab（如按 Y 轴分布或有内部转置），则所有频谱运算（涡量、粘性、投影）的波数计算错误，结果不正确。

**验证方法**（在集群上）：
```cpp
// 用单一正弦波初始化，D2Z 后打印每块 GPU 持有的模式
// 例如初始化为 V1(x,y,z) = sin(kx0*x)，D2Z 后 GPU(kx0/nx_local) 应在 lx=kx0%nx_local 位置有峰值
```

### 5. 内存布局（in-place padded）

每块 GPU 分配 `nx_local * NY * NZC` 个 `GCplx` 元素（= `nx_local * NY * (NZ+2)` 个 `double`）：

```
实空间访问（double*）：
  V[lx][j][k]   →  ptr_d[ lx*NY*2*NZC + j*2*NZC + k ]   k = 0..NZ-1
  padding 位置：  ptr_d[ lx*NY*2*NZC + j*2*NZC + NZ ]  和  ...+NZ+1 （不使用）

频谱访问（GCplx*）：
  V[lx][j][kc]  →  ptr_c[ lx*NY*NZC + j*NZC + kc ]       kc = 0..NZC-1
```

---

## 对比单 GPU cufft 版本

| 方面 | cufft（单 GPU） | cufftxt（多 GPU） |
|------|-----------------|------------------|
| 变换 | `cufftExecD2Z/Z2D` | `cufftXtExecDescriptorD2Z/Z2D` |
| 内存 | `cudaMalloc`（独立 real/complex） | `cufftXtMalloc(INPLACE)`（合并 real+complex） |
| 内核 | 全局 NX\*NY\*NZC 元素 | 本地 nx_local\*NY\*NZC，传入 x_offset |
| 同步 | cuFFT 默认 stream 0，自动有序 | 需要显式 `cudaDeviceSynchronize()` |
| 索引 | `RIDX(i,j,k) = i*NY*NZ+j*NZ+k` | `RIDX_PAD(lx,j,k) = lx*NY*2*NZC+j*2*NZC+k` |
