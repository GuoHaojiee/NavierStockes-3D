/**
 * Navier-Stokes 求解器 - AccFFT 版本（2D pencil 分解）+ ETD4RK 时间推进
 *
 * 由 NavierStokes_periodic_accfft.cpp 复制后改动；空间离散、FFT 调用、投影、
 * 归一化约定（非归一化 DFT：V_c = DFT(V_phys)，逆变换后手动 /N）全部保持不变。
 *
 * 新增内容：
 *   1. ETD4RK（Cox & Matthews 2002, J. Comput. Phys. 176:430, 式 (26)-(29)）。
 *      每个 Fourier 模态：dv̂/dt = c v̂ + N̂,  c = -P|k|²,
 *      N̂ = Proj[ DFT(v×rot v) + DFT(f) ]（不含粘性项）。
 *      粘性项用 e^{ch} 精确积分 → 没有 |c|h ≤ 2.785 的粘性稳定性限制。
 *      Helmholtz 投影与乘标量 c(|k|) 可交换，故只需投影 N̂，v̂ 仍保持无散。
 *   2. 系数表：|k|² 为整数，按 |k|² 建表（E, E2, Q, f1, f2, f3），dt 确定后只算一次。
 *      小 |ch| 用 Kassam & Trefethen (2005) 复围道积分避免相消。
 *   3. 可选测试问题：poly（原解析解，时间因子 t²+1）、sin（时间因子 1+A·sin(ωt)，
 *      用于测时间收敛阶）、tg（Taylor-Green 涡，无外力，无解析解）。
 *   4. 粘性系数 P 可调（原程序固定 P=1）；外力随 P 相应改变，解析解不变。
 *
 * 用法：
 *   mpirun -np NP ./navier_stokes_accfft_etd NX NY NZ dt NSTEPS [OMP_threads] [scheme] [key=value ...]
 *     scheme : etd4rk（默认）| rk4 | compare（先 rk4 后 etd4rk，打印两者之差）
 *     problem=poly|sin|tg   （默认 poly，即原程序的解析解）
 *     P=<粘性系数>           （默认 poly/sin: 1，tg: 1/1600）
 *     omega=<ω> amp=<A>     （sin 问题的时间因子 1 + A·sin(ωt)，默认 ω=2, A=0.5）
 *     U=<幅值>               （速度幅值缩放，默认 1；poly/sin 解析解与外力相应缩放）
 *     diag=<K>              （每 K 步打印一次动能；0 = 不打印，默认 0）
 *     check=<K>             （每 K 步检查一次是否爆炸，默认 10；判据见 ETD_NOTES.md）
 *     spec=<文件名>          （结束时把壳平均能谱 E(k) 写到文件）
 *     ehi=<K>               （每 K 步记录最高波数壳层能量 E_hi，并启用 γ 判据；0 = 关闭，默认 0）
 *     ehi_file=<文件名>      （E_hi 记录输出文件：step t E_hi E_total umax）
 *     gamma_thr=<阈值>       （γ 判据阈值，默认 1e-3 /步）
 *     gamma_win=<W>          （拟合用的最后记录数，默认 20）
 *   输出格式与原程序一致（含 "L2 error (t=...)" 与 "Avg per step:"）。
 *
 * 编译：make -f Makefile_accfft_etd
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <accfft.h>

using namespace std;

// ==============================================================================
// AccFFT 上下文结构体
// ==============================================================================

struct AccCtx {
    int isize[3], istart[3];  // 实空间本地大小和全局起始（0-based）
    int osize[3], ostart[3];  // 频谱空间本地大小和全局起始（0-based）
    int nx, ny, nz;
    long local_size_r;        // = isize[0]*isize[1]*nz（实空间元素数）
    long local_size_c;        // = osize[0]*osize[1]*osize[2]（频谱空间复数元素数）
    double norm_inv;          // 1/(nx*ny*nz)，用于逆变换后归一化
};

// 实空间线性索引（0-based 本地，z 步长为全局 nz）
inline long idx_r(const AccCtx& ctx, int i, int j, int k) {
    return (long)i * ctx.isize[1] * ctx.nz + j * ctx.nz + k;
}

// 频谱空间线性索引（0-based 本地）
inline long idx_c(const AccCtx& ctx, int i, int j, int k) {
    return (long)i * ctx.osize[1] * ctx.osize[2] + j * ctx.osize[2] + k;
}

// 物理坐标
inline double phys_x(const AccCtx& ctx, int i, double dx) { return (i + ctx.istart[0]) * dx; }
inline double phys_y(const AccCtx& ctx, int j, double dy) { return (j + ctx.istart[1]) * dy; }
inline double phys_z(int k, double dz)                    { return k * dz; }

// 波数（AccFFT R2C 沿 z 方向；kx/ky 折叠，Nyquist 置零；kz = 0..nz/2 非负）
inline double wave_kx(const AccCtx& ctx, int i) {
    int gi = i + ctx.ostart[0];
    if (gi == ctx.nx / 2) return 0.0;
    return (gi <= ctx.nx / 2) ? (double)gi : (double)(gi - ctx.nx);
}
inline double wave_ky(const AccCtx& ctx, int j) {
    int gj = j + ctx.ostart[1];
    if (gj == ctx.ny / 2) return 0.0;
    return (gj <= ctx.ny / 2) ? (double)gj : (double)(gj - ctx.ny);
}
inline double wave_kz(const AccCtx& ctx, int k) {
    return (double)(k + ctx.ostart[2]);
}

// 与 wave_kx/ky 相同的折叠规则（按全局下标），用于求 |k|² 的最大值
inline int fold_nyq0(int g, int n) {
    if (g == n / 2) return 0;
    return (g <= n / 2) ? g : g - n;
}

// ==============================================================================
// 测试问题与参数
//
//   poly：原程序的解析解，v = g(t) S(x)，g = t²+1（P=1 时外力与原程序逐项相同）
//   sin ：同一空间形状，g = 1 + A·sin(ωt)（非多项式，用于测时间收敛阶）
//   tg  ：Taylor-Green 涡 v0 = U(sin x cos y cos z, -cos x sin y cos z, 0)，f = 0
//
// 外力（构造解）：v_t = PΔv + v×rot v − ∇p + f，p = g(t)·cos x cos y cos z
//   f = g'(t)·S − P·g·ΔS − g²·(S × rot S) + g·∇p0
// 形状函数 S、ΔS、rot S 与原程序 func_laplace_* / func_rot* 相同
// （etd_coeff_check.py 第 4 节用 sympy 核对：div S = 0、ΔS、rot S、残差恒为 0）。
// ==============================================================================

enum Problem { PROB_POLY, PROB_SIN, PROB_TG };

struct Config {
    Problem prob  = PROB_POLY;
    double  P     = 1.0;   // 粘性系数
    double  omega = 2.0;   // sin 问题：ω
    double  amp   = 0.5;   // sin 问题：A
    double  U     = 1.0;   // 速度幅值缩放：poly/sin 的 g(t) 乘以 U，tg 初值乘以 U
};
static Config g_cfg;

inline bool has_forcing()  { return g_cfg.prob != PROB_TG; }
inline bool has_analytic() { return g_cfg.prob != PROB_TG; }

// 时间因子 g(t) 及其导数（含幅值缩放 U；U=1 时与原程序相同）
inline double tfac(double t) {
    if (g_cfg.prob == PROB_SIN) return g_cfg.U * (1.0 + g_cfg.amp * sin(g_cfg.omega * t));
    return g_cfg.U * (t*t + 1.0);
}
inline double dtfac(double t) {
    if (g_cfg.prob == PROB_SIN) return g_cfg.U * g_cfg.amp * g_cfg.omega * cos(g_cfg.omega * t);
    return g_cfg.U * 2.0 * t;
}

// 解析解 v(x,t)（poly/sin），或 TG 初值（t 被忽略）
inline void exact_V(double x, double y, double z, double t, double v[3]) {
    if (g_cfg.prob == PROB_TG) {
        v[0] =  g_cfg.U * sin(x) * cos(y) * cos(z);
        v[1] = -g_cfg.U * cos(x) * sin(y) * cos(z);
        v[2] =  0.0;
        return;
    }
    double g = tfac(t);
    double e = exp(sin(3*x+3*y));
    v[0] =  g * e * cos(6*z);
    v[1] =  g * e * cos(6*z);
    v[2] = -g * e * cos(3*x+3*y) * sin(6*z);
}

// 外力 f(x,t)；每个点只算一次 exp/sin/cos（原程序每个分量重复计算，结果相同）
inline void forcing(double x, double y, double z, double t, double f[3]) {
    double g = tfac(t), dg = dtfac(t), P = g_cfg.P;
    double c = cos(3*x+3*y), s = sin(3*x+3*y), e = exp(s);
    double cz = cos(6*z), sz = sin(6*z);
    // 形状函数 S
    double S1 = e*cz, S2 = e*cz, S3 = -e*c*sz;
    // ΔS（与原 func_laplace_V1/V3 相同，去掉时间因子）
    double d2xy1 = 9*e*(c*c - s)*cz;
    double lap1  = 2*d2xy1 - 36*e*cz;
    double d2xy3 = -9*e*c*((c*c - s) - (2*s + 1))*sz;
    double lap3  = 2*d2xy3 + 36*e*c*sz;
    // rot S（与原 func_rot1/rot2 相同，rot3 = 0）
    double r1 = -3*e*(c*c - s)*sz + 6*e*sz;
    double r2 = -6*e*sz + 3*e*(c*c - s)*sz;
    double r3 = 0.0;
    // S × rot S
    double x1 = S2*r3 - S3*r2, x2 = S3*r1 - S1*r3, x3 = S1*r2 - S2*r1;
    // ∇p0，p0 = cos x cos y cos z
    double gp1 = -sin(x)*cos(y)*cos(z);
    double gp2 = -cos(x)*sin(y)*cos(z);
    double gp3 = -cos(x)*cos(y)*sin(z);
    f[0] = dg*S1 - P*g*lap1 - g*g*x1 + g*gp1;
    f[1] = dg*S2 - P*g*lap1 - g*g*x2 + g*gp2;
    f[2] = dg*S3 - P*g*lap3 - g*g*x3 + g*gp3;
}

// ==============================================================================
// 频谱算子（AccFFT 坐标约定）
//
// 归一化约定：
//   V_c = DFT(V_phys)  [非归一化，= N * FFT_norm(V_phys)]
//   所有频谱量（rot_c, visc_c, nl_c, f_c）均为非归一化表示
//   逆变换后：V_phys = accfft_c2r(V_c_copy) / N
// ==============================================================================

// 计算旋度 rot(V) = i·k × V（频谱空间）
// rot_1 = i(ky*V3 - kz*V2)
// rot_2 = i(kz*V1 - kx*V3)
// rot_3 = i(kx*V2 - ky*V1)
void compute_rot(const AccCtx& ctx,
                 const Complex* V1_c, const Complex* V2_c, const Complex* V3_c,
                 Complex* rot1_c, Complex* rot2_c, Complex* rot3_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);

                // i*(a+ib) = -b+ia
                rot1_c[idx][0] = -(ky * V3_c[idx][1] - kz * V2_c[idx][1]);
                rot1_c[idx][1] =   ky * V3_c[idx][0] - kz * V2_c[idx][0];
                rot2_c[idx][0] = -(kz * V1_c[idx][1] - kx * V3_c[idx][1]);
                rot2_c[idx][1] =   kz * V1_c[idx][0] - kx * V3_c[idx][0];
                rot3_c[idx][0] = -(kx * V2_c[idx][1] - ky * V1_c[idx][1]);
                rot3_c[idx][1] =   kx * V2_c[idx][0] - ky * V1_c[idx][0];
            }
        }
    }
}

// 计算粘性项：visc = -P k² * V（频谱空间；原程序 P 固定为 1）
void compute_viscous(const AccCtx& ctx,
                     const Complex* V_c, Complex* visc_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double k2 = kx*kx + ky*ky + kz*kz;
                visc_c[idx][0] = -g_cfg.P * k2 * V_c[idx][0];
                visc_c[idx][1] = -g_cfg.P * k2 * V_c[idx][1];
            }
        }
    }
}

// 投影到无散空间：V -= ∇(∇⁻²·div V)（频谱空间，in-place）
// div = i·k·V，phi = div/(-k²)，V -= i·k·phi
void make_div_free(const AccCtx& ctx,
                   Complex* V1_c, Complex* V2_c, Complex* V3_c,
                   Complex* div_c, Complex* phi_c) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);

                // div = i(kx*V1 + ky*V2 + kz*V3)
                div_c[idx][0] = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                div_c[idx][1] =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];

                double k2 = kx*kx + ky*ky + kz*kz;
                if (k2 > 1e-14) {
                    phi_c[idx][0] = div_c[idx][0] / (-k2);
                    phi_c[idx][1] = div_c[idx][1] / (-k2);
                } else {
                    phi_c[idx][0] = phi_c[idx][1] = 0.0;
                }

                // V -= i*k*phi：i*kx*(a+ib) = -kx*b + i*kx*a
                V1_c[idx][0] -= -kx * phi_c[idx][1];
                V1_c[idx][1] -=  kx * phi_c[idx][0];
                V2_c[idx][0] -= -ky * phi_c[idx][1];
                V2_c[idx][1] -=  ky * phi_c[idx][0];
                V3_c[idx][0] -= -kz * phi_c[idx][1];
                V3_c[idx][1] -=  kz * phi_c[idx][0];
            }
        }
    }
}

// 计算 max|div(V)| 用于诊断
double compute_div_max(const AccCtx& ctx,
                       const Complex* V1_c, const Complex* V2_c, const Complex* V3_c) {
    double local_max = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:local_max)
    for (int i = 0; i < ctx.osize[0]; ++i) {
        for (int j = 0; j < ctx.osize[1]; ++j) {
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double kx = wave_kx(ctx, i);
                double ky = wave_ky(ctx, j);
                double kz = wave_kz(ctx, k);
                double dr = -(kx*V1_c[idx][1] + ky*V2_c[idx][1] + kz*V3_c[idx][1]);
                double di =   kx*V1_c[idx][0] + ky*V2_c[idx][0] + kz*V3_c[idx][0];
                double val = sqrt(dr*dr + di*di);
                if (val > local_max) local_max = val;
            }
        }
    }
    double global_max = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global_max;
}

// ==============================================================================
// 非线性对流项：v × rot(v)（伪谱方法）
//
// 算法（与 FFTW / heFFTe / p3dfft 版本完全相同）：
//   1. 频谱空间计算旋度 rot_c = ik × V_c              （无 FFT）
//   2. 逆变换 V_c → V_r                               （3 次 FFT）
//   3. 逆变换 rot_c → rot_r                            （3 次 FFT）
//   4. 实空间叉乘，覆盖 rot_r                           （无 FFT）
//   5. 正变换 rot_r → nl_c                             （3 次 FFT）
// 合计：9 次 FFT（与其他版本相同）
//
// preserve_V：
//   true  - V_c 在调用后仍需使用（k1 阶段），用 tmp_c 复制保护（3 次 memcpy）
//   false - V_c 调用后即被覆盖（k2/k3/k4 阶段），直接传入 c2r（0 次 memcpy）
// rot_c 始终是临时量（每次由 compute_rot 重新生成），直接传入 c2r（0 次 memcpy）
// ==============================================================================
void compute_nonlinear(const AccCtx& ctx, accfft_plan* plan,
                       Complex* V1_c, Complex* V2_c, Complex* V3_c,
                       Complex* nl1_c, Complex* nl2_c, Complex* nl3_c,
                       double* V1_r, double* V2_r, double* V3_r,
                       Complex* rot1_c, Complex* rot2_c, Complex* rot3_c,
                       double* rot1_r, double* rot2_r, double* rot3_r,
                       Complex* tmp_c,
                       bool preserve_V) {

    long csize = ctx.local_size_c;
    double inv_n = ctx.norm_inv;

    // 1. 频谱空间计算旋度（rot_c = ik × V_c，非归一化）
    compute_rot(ctx, V1_c, V2_c, V3_c, rot1_c, rot2_c, rot3_c);

    // 2. 逆变换 V_c → V_r
    //    preserve_V=true：先复制到 tmp_c 再 c2r（保护 V_c，供后续 RK4 阶段使用）
    //    preserve_V=false：直接传 V_c（调用方在此后立即覆盖 tmp_v，无需保护）
    if (preserve_V) {
        memcpy(tmp_c, V1_c, csize * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V1_r);
        memcpy(tmp_c, V2_c, csize * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V2_r);
        memcpy(tmp_c, V3_c, csize * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V3_r);
    } else {
        accfft_execute_c2r(plan, V1_c, V1_r);
        accfft_execute_c2r(plan, V2_c, V2_r);
        accfft_execute_c2r(plan, V3_c, V3_r);
    }

    // 3. 逆变换 rot_c → rot_r
    //    rot_c 由 compute_rot 临时生成，c2r 后不再使用，直接传入（0 次 memcpy）
    accfft_execute_c2r(plan, rot1_c, rot1_r);
    accfft_execute_c2r(plan, rot2_c, rot2_r);
    accfft_execute_c2r(plan, rot3_c, rot3_r);

    // 4. 归一化（除以 N 得到物理值）
    #pragma omp parallel for
    for (long idx = 0; idx < ctx.local_size_r; ++idx) {
        V1_r[idx] *= inv_n;    V2_r[idx] *= inv_n;    V3_r[idx] *= inv_n;
        rot1_r[idx] *= inv_n;  rot2_r[idx] *= inv_n;  rot3_r[idx] *= inv_n;
    }

    // 5. 实空间叉乘 v × rot(v)，覆盖 rot_r（用作正变换输入）
    #pragma omp parallel for
    for (long idx = 0; idx < ctx.local_size_r; ++idx) {
        double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
        double w1 = rot1_r[idx], w2 = rot2_r[idx], w3 = rot3_r[idx];
        rot1_r[idx] = v2*w3 - v3*w2;   // (v×rot)_x
        rot2_r[idx] = v3*w1 - v1*w3;   // (v×rot)_y
        rot3_r[idx] = v1*w2 - v2*w1;   // (v×rot)_z
    }

    // 6. 正变换（不归一化：nl_c = DFT(cross)，与 V_c 约定一致）
    accfft_execute_r2c(plan, rot1_r, nl1_c);
    accfft_execute_r2c(plan, rot2_r, nl2_c);
    accfft_execute_r2c(plan, rot3_r, nl3_c);
}

// ==============================================================================
// 右端项
//   include_viscous = true ：RHS = Proj[ v×rot(v) + PΔv + f ]   （RK4 用，与原程序相同）
//   include_viscous = false：N   = Proj[ v×rot(v) + f ]          （ETD4RK 用，粘性项由 e^{ch} 处理）
// FFT 次数：每次调用 12 次（3 c2r V + 3 c2r rot + 3 r2c nl + 3 r2c f）；
//           tg 问题无外力，跳过外力的 3 次 r2c（两种格式都一样，为 9 次）。
// ==============================================================================

void compute_rhs(const AccCtx& ctx, accfft_plan* plan,
                 Complex* V1_c, Complex* V2_c, Complex* V3_c,
                 Complex* rhs1_c, Complex* rhs2_c, Complex* rhs3_c,
                 double* V1_r, double* V2_r, double* V3_r,
                 double* work_r1, double* work_r2, double* work_r3,
                 Complex* rot1_c, Complex* rot2_c, Complex* rot3_c,
                 double* rot1_r, double* rot2_r, double* rot3_r,
                 Complex* nl1_c, Complex* nl2_c, Complex* nl3_c,
                 Complex* visc1_c, Complex* visc2_c, Complex* visc3_c,
                 Complex* f1_c, Complex* f2_c, Complex* f3_c,
                 Complex* div_c, Complex* phi_c,
                 Complex* tmp_c,
                 double t, double dx, double dy, double dz,
                 bool preserve_V, bool include_viscous) {

    // 1. 粘性项 P∆V = -P k² V（纯频谱运算，必须在 compute_nonlinear 之前完成）
    //    当 preserve_V=false 时，compute_nonlinear 会销毁 V_c，故先在此读取
    if (include_viscous) {
        compute_viscous(ctx, V1_c, visc1_c);
        compute_viscous(ctx, V2_c, visc2_c);
        compute_viscous(ctx, V3_c, visc3_c);
    }

    // 2. 非线性项 v × rot(v)（伪谱方法；preserve_V 控制是否保护 V_c）
    compute_nonlinear(ctx, plan, V1_c, V2_c, V3_c, nl1_c, nl2_c, nl3_c,
                      V1_r, V2_r, V3_r,
                      rot1_c, rot2_c, rot3_c,
                      rot1_r, rot2_r, rot3_r,
                      tmp_c, preserve_V);

    // 3. 外力项 f（实空间填充 → 正变换，不归一化：f_c = DFT(f)）
    bool frc = has_forcing();
    if (frc) {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < ctx.isize[0]; ++i) {
            for (int j = 0; j < ctx.isize[1]; ++j) {
                for (int k = 0; k < ctx.nz; ++k) {
                    long ridx = idx_r(ctx, i, j, k);
                    double f[3];
                    forcing(phys_x(ctx, i, dx), phys_y(ctx, j, dy), phys_z(k, dz), t, f);
                    work_r1[ridx] = f[0];
                    work_r2[ridx] = f[1];
                    work_r3[ridx] = f[2];
                }
            }
        }
        accfft_execute_r2c(plan, work_r1, f1_c);
        accfft_execute_r2c(plan, work_r2, f2_c);
        accfft_execute_r2c(plan, work_r3, f3_c);
    }

    // 4. 组合各项（均为非归一化 DFT）
    long nc = ctx.local_size_c;
    #pragma omp parallel for
    for (long idx = 0; idx < nc; ++idx) {
        for (int r = 0; r < 2; ++r) {
            double v1 = include_viscous ? visc1_c[idx][r] : 0.0;
            double v2 = include_viscous ? visc2_c[idx][r] : 0.0;
            double v3 = include_viscous ? visc3_c[idx][r] : 0.0;
            double g1 = frc ? f1_c[idx][r] : 0.0;
            double g2 = frc ? f2_c[idx][r] : 0.0;
            double g3 = frc ? f3_c[idx][r] : 0.0;
            rhs1_c[idx][r] = nl1_c[idx][r] + v1 + g1;
            rhs2_c[idx][r] = nl2_c[idx][r] + v2 + g2;
            rhs3_c[idx][r] = nl3_c[idx][r] + v3 + g3;
        }
    }

    // 5. 投影：使 RHS 无散（P 算子）
    make_div_free(ctx, rhs1_c, rhs2_c, rhs3_c, div_c, phi_c);
}

// 所有 compute_rhs 共用的工作数组，打包以缩短 rk4_step / etd4rk_step 的参数列表
struct Work {
    double  *V1_r, *V2_r, *V3_r;
    double  *work_r1, *work_r2, *work_r3;
    Complex *rot1_c, *rot2_c, *rot3_c;
    double  *rot1_r, *rot2_r, *rot3_r;
    Complex *nl1_c, *nl2_c, *nl3_c;
    Complex *visc1_c, *visc2_c, *visc3_c;
    Complex *f1_c, *f2_c, *f3_c;
    Complex *div_c, *phi_c, *tmp_c;
    Complex *k1[3], *k2[3], *k3[3], *k4[3], *tmp_v[3];
};

inline void rhs(const AccCtx& ctx, accfft_plan* plan, Work& w,
                Complex* const in[3], Complex* const out[3],
                double t, double dx, double dy, double dz,
                bool preserve_V, bool include_viscous) {
    compute_rhs(ctx, plan, in[0], in[1], in[2], out[0], out[1], out[2],
                w.V1_r, w.V2_r, w.V3_r, w.work_r1, w.work_r2, w.work_r3,
                w.rot1_c, w.rot2_c, w.rot3_c, w.rot1_r, w.rot2_r, w.rot3_r,
                w.nl1_c, w.nl2_c, w.nl3_c, w.visc1_c, w.visc2_c, w.visc3_c,
                w.f1_c, w.f2_c, w.f3_c, w.div_c, w.phi_c, w.tmp_c,
                t, dx, dy, dz, preserve_V, include_viscous);
}

// ==============================================================================
// RK4 时间积分（频谱空间）— 与原程序 rk4_step 的组合公式与 preserve_V 用法相同
// ==============================================================================

void rk4_step(const AccCtx& ctx, accfft_plan* plan, Work& w, Complex* const V[3],
              double dt, double t, double dx, double dy, double dz) {
    long nc = ctx.local_size_c;
    Complex** k1 = w.k1; Complex** k2 = w.k2; Complex** k3 = w.k3; Complex** k4 = w.k4;
    Complex** tv = w.tmp_v;

    // k1 = RHS(V^n, t)，V^n 之后还要用 → preserve_V=true
    rhs(ctx, plan, w, V, k1, t, dx, dy, dz, true, true);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for
        for (long i = 0; i < nc; ++i) {
            tv[d][i][0] = V[d][i][0] + 0.5*dt*k1[d][i][0];
            tv[d][i][1] = V[d][i][1] + 0.5*dt*k1[d][i][1];
        }
    }
    // k2 = RHS(V^n + dt/2*k1, t + dt/2)，tmp_v 随后被覆盖 → preserve_V=false
    rhs(ctx, plan, w, tv, k2, t + 0.5*dt, dx, dy, dz, false, true);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for
        for (long i = 0; i < nc; ++i) {
            tv[d][i][0] = V[d][i][0] + 0.5*dt*k2[d][i][0];
            tv[d][i][1] = V[d][i][1] + 0.5*dt*k2[d][i][1];
        }
    }
    // k3 = RHS(V^n + dt/2*k2, t + dt/2)
    rhs(ctx, plan, w, tv, k3, t + 0.5*dt, dx, dy, dz, false, true);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for
        for (long i = 0; i < nc; ++i) {
            tv[d][i][0] = V[d][i][0] + dt*k3[d][i][0];
            tv[d][i][1] = V[d][i][1] + dt*k3[d][i][1];
        }
    }
    // k4 = RHS(V^n + dt*k3, t + dt)
    rhs(ctx, plan, w, tv, k4, t + dt, dx, dy, dz, false, true);

    // V^{n+1} = V^n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for
        for (long i = 0; i < nc; ++i) {
            for (int r = 0; r < 2; ++r)
                V[d][i][r] += (dt/6.0)*(k1[d][i][r] + 2*k2[d][i][r] + 2*k3[d][i][r] + k4[d][i][r]);
        }
    }
}

// ==============================================================================
// ETD4RK 系数表
//
// 对 c = -P·K，K = |k|² ∈ {0, 1, ..., Kmax}（整数，因 α=1、kx,ky,kz 为整数），z = c·h：
//   E  = e^{z},  E2 = e^{z/2}
//   Q  = (E2 − 1)/c                                 = h·φQ(z),  φQ(z) = (e^{z/2} − 1)/z
//   f1 = [−4 − z + E(4 − 3z + z²)]/(h²c³)          = h·g1(z)            —— Cox & Matthews 式 (29)
//   f2 = [ 2 + z + E(−2 + z)]/(h²c³)               = h·g2(z)
//   f3 = [−4 − 3z − z² + E(4 − z)]/(h²c³)          = h·g3(z)
// 约定：表中 Q、f1、f2、f3 **已包含因子 h**（单位为时间）。
// 极限 z→0（c=0，即 k=0 或 P=0）：Q = h/2，f1 = h/6，f2 = h/6，f3 = h/6；
//   更新式中 f2 前有系数 2，所以等效权重 h/6, h/3, h/3, h/6 = 经典 RK4。
//
// 数值计算（Kassam & Trefethen 2005）：
//   0 < |z| < ETD_Z_CONTOUR：以 z 为圆心、半径 1、M=32 个点的圆周上对 φ 函数取平均（取实部）；
//   |z| ≥ ETD_Z_CONTOUR：直接用显式公式（此时无严重相消）。
//   阈值取 0.5：半径 1 的圆在 |z|≈1 时会经过原点附近，圆周点本身陷入相消区，
//   精度只有 ~1e-13；取 0.5 时两段相对误差都 ≲ 1.3e-14（见 etd_coeff_check.py 第 2 节）。
// ==============================================================================

static const int    ETD_M_CONTOUR = 32;
static const double ETD_Z_CONTOUR = 0.5;

struct EtdTable {
    int Kmax = 0;
    vector<double> E, E2, Q, f1, f2, f3;
};

// 显式公式（复数版），返回 φQ, g1, g2, g3
static void etd_phi_direct(complex<double> z, complex<double> out[4]) {
    complex<double> ez = exp(z), ez2 = exp(0.5*z), z3 = z*z*z;
    out[0] = (ez2 - 1.0) / z;
    out[1] = (-4.0 - z + ez*(4.0 - 3.0*z + z*z)) / z3;
    out[2] = ( 2.0 + z + ez*(z - 2.0)) / z3;
    out[3] = (-4.0 - 3.0*z - z*z + ez*(4.0 - z)) / z3;
}

static void etd_phi(double z, double out[4]) {
    if (z == 0.0) {
        out[0] = 0.5; out[1] = out[2] = out[3] = 1.0/6.0;
    } else if (fabs(z) < ETD_Z_CONTOUR) {
        double acc[4] = {0, 0, 0, 0};
        for (int j = 0; j < ETD_M_CONTOUR; ++j) {
            complex<double> r = polar(1.0, 2.0*M_PI*(j + 0.5)/ETD_M_CONTOUR);
            complex<double> v[4];
            etd_phi_direct(z + r, v);
            for (int q = 0; q < 4; ++q) acc[q] += v[q].real();
        }
        for (int q = 0; q < 4; ++q) out[q] = acc[q] / ETD_M_CONTOUR;
    } else {
        complex<double> v[4];
        etd_phi_direct(complex<double>(z, 0.0), v);
        for (int q = 0; q < 4; ++q) out[q] = v[q].real();
    }
}

EtdTable build_etd_table(int nx, int ny, int nz, double P, double h) {
    // Kmax = 三个方向各取最大 |k| 的平方和（kx/ky 的 Nyquist 已置零，kz 含 nz/2）
    int mx = 0, my = 0;
    for (int g = 0; g < nx; ++g) mx = max(mx, abs(fold_nyq0(g, nx)));
    for (int g = 0; g < ny; ++g) my = max(my, abs(fold_nyq0(g, ny)));
    int mz = nz / 2;
    EtdTable T;
    T.Kmax = mx*mx + my*my + mz*mz;
    int n = T.Kmax + 1;
    T.E.resize(n); T.E2.resize(n); T.Q.resize(n); T.f1.resize(n); T.f2.resize(n); T.f3.resize(n);
    for (int K = 0; K < n; ++K) {
        double z = -P * K * h;
        double ph[4];
        etd_phi(z, ph);
        T.E[K]  = exp(z);
        T.E2[K] = exp(0.5*z);
        T.Q[K]  = h * ph[0];
        T.f1[K] = h * ph[1];
        T.f2[K] = h * ph[2];
        T.f3[K] = h * ph[3];
    }
    return T;
}

// |k|² 的整数表索引（与 compute_viscous 使用相同的 wave_k*）
inline int k2_index(const AccCtx& ctx, int i, int j, int k) {
    double kx = wave_kx(ctx, i), ky = wave_ky(ctx, j), kz = wave_kz(ctx, k);
    return (int)lround(kx*kx + ky*ky + kz*kz);
}

// ==============================================================================
// ETD4RK 一步（Cox & Matthews 2002, 式 (26)-(29)）
//   a  = E2·u + Q·N(u, t)                             (26)
//   b  = E2·u + Q·N(a, t+h/2)                         (27)
//   c̃  = E2·a + Q·(2N(b, t+h/2) − N(u, t))           (28)
//   u⁺ = E·u + f1·N(u) + 2f2·(N(a) + N(b)) + f3·N(c̃, t+h)   (29)
//
// 缓冲区复用（无需额外矢量场，内存增量只有系数表）：
//   k1 ← N(u)            preserve_V=true （u 后面还要用）
//   tmp_v ← a
//   k2 ← N(a)            preserve_V=true （a 在 (28) 中还要用）
//   k4 ← b（临时存放）
//   k3 ← N(b)            preserve_V=false（b 之后不再用，k4 中的 b 被销毁）
//   tmp_v ← c̃（原地覆盖 a：每个点先读 a 再写 c̃）
//   k4 ← N(c̃)            preserve_V=false（c̃ 之后不再用，tmp_v 被销毁）
// ==============================================================================

// 逐模态循环：func(idx, K)
template <class F>
inline void for_each_mode(const AccCtx& ctx, F func) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < ctx.osize[0]; ++i)
        for (int j = 0; j < ctx.osize[1]; ++j)
            for (int k = 0; k < ctx.osize[2]; ++k)
                func(idx_c(ctx, i, j, k), k2_index(ctx, i, j, k));
}

void etd4rk_step(const AccCtx& ctx, accfft_plan* plan, Work& w, Complex* const V[3],
                 const EtdTable& T, double h, double t, double dx, double dy, double dz) {
    Complex** Nu = w.k1; Complex** Na = w.k2; Complex** Nb = w.k3; Complex** Nc = w.k4;
    Complex** a  = w.tmp_v;   // 先存 a，后存 c̃
    Complex** b  = w.k4;      // 先存 b，N(b) 算完后再存 N(c̃)
    const double *E = T.E.data(), *E2 = T.E2.data(), *Q = T.Q.data();
    const double *f1 = T.f1.data(), *f2 = T.f2.data(), *f3 = T.f3.data();

    // N(u, t)
    rhs(ctx, plan, w, V, Nu, t, dx, dy, dz, /*preserve_V=*/true, /*include_viscous=*/false);

    // (26) a = E2·u + Q·N(u)
    for_each_mode(ctx, [&](long idx, int K) {
        for (int d = 0; d < 3; ++d)
            for (int r = 0; r < 2; ++r)
                a[d][idx][r] = E2[K]*V[d][idx][r] + Q[K]*Nu[d][idx][r];
    });

    // N(a, t+h/2)
    rhs(ctx, plan, w, a, Na, t + 0.5*h, dx, dy, dz, /*preserve_V=*/true, false);

    // (27) b = E2·u + Q·N(a)
    for_each_mode(ctx, [&](long idx, int K) {
        for (int d = 0; d < 3; ++d)
            for (int r = 0; r < 2; ++r)
                b[d][idx][r] = E2[K]*V[d][idx][r] + Q[K]*Na[d][idx][r];
    });

    // N(b, t+h/2)
    rhs(ctx, plan, w, b, Nb, t + 0.5*h, dx, dy, dz, /*preserve_V=*/false, false);

    // (28) c̃ = E2·a + Q·(2N(b) − N(u))，原地覆盖 a
    for_each_mode(ctx, [&](long idx, int K) {
        for (int d = 0; d < 3; ++d)
            for (int r = 0; r < 2; ++r)
                a[d][idx][r] = E2[K]*a[d][idx][r] + Q[K]*(2.0*Nb[d][idx][r] - Nu[d][idx][r]);
    });

    // N(c̃, t+h)
    rhs(ctx, plan, w, a, Nc, t + h, dx, dy, dz, /*preserve_V=*/false, false);

    // (29) u⁺ = E·u + f1·N(u) + 2f2·(N(a) + N(b)) + f3·N(c̃)
    for_each_mode(ctx, [&](long idx, int K) {
        for (int d = 0; d < 3; ++d)
            for (int r = 0; r < 2; ++r)
                V[d][idx][r] = E[K]*V[d][idx][r] + f1[K]*Nu[d][idx][r]
                             + 2.0*f2[K]*(Na[d][idx][r] + Nb[d][idx][r]) + f3[K]*Nc[d][idx][r];
    });
}

// ==============================================================================
// 诊断：动能与壳平均能谱
//   mean(|v|²) = (1/N²)·Σ_k |v̂_k|²（非归一化 DFT 的 Parseval）
//   R2C 只存 kz ≥ 0：kz ∈ (0, nz/2) 的模态权重为 2，kz = 0 或 nz/2 权重为 1
//   动能 E = ½·mean(|v|²)
// ==============================================================================

inline double r2c_weight(const AccCtx& ctx, int k) {
    int gk = k + ctx.ostart[2];
    return (gk == 0 || 2*gk == ctx.nz) ? 1.0 : 2.0;
}

double kinetic_energy(const AccCtx& ctx, Complex* const V[3]) {
    double local = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:local)
    for (int i = 0; i < ctx.osize[0]; ++i)
        for (int j = 0; j < ctx.osize[1]; ++j)
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                double s = 0.0;
                for (int d = 0; d < 3; ++d)
                    s += V[d][idx][0]*V[d][idx][0] + V[d][idx][1]*V[d][idx][1];
                local += r2c_weight(ctx, k) * s;
            }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return 0.5 * global * ctx.norm_inv * ctx.norm_inv;
}

// 壳平均能谱 E(k)，k = round(|k|)，Σ_k E(k) = 动能；rank 0 写入文件
void write_spectrum(const AccCtx& ctx, Complex* const V[3], const string& fname, double time) {
    int nsh = (int)ceil(sqrt(3.0) * max(ctx.nx, max(ctx.ny, ctx.nz)) / 2.0) + 2;
    vector<double> loc(nsh, 0.0), glob(nsh, 0.0);
    for (int i = 0; i < ctx.osize[0]; ++i)
        for (int j = 0; j < ctx.osize[1]; ++j)
            for (int k = 0; k < ctx.osize[2]; ++k) {
                long idx = idx_c(ctx, i, j, k);
                int sh = (int)lround(sqrt((double)k2_index(ctx, i, j, k)));
                double s = 0.0;
                for (int d = 0; d < 3; ++d)
                    s += V[d][idx][0]*V[d][idx][0] + V[d][idx][1]*V[d][idx][1];
                loc[sh] += 0.5 * r2c_weight(ctx, k) * s * ctx.norm_inv * ctx.norm_inv;
            }
    MPI_Reduce(loc.data(), glob.data(), nsh, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        FILE* fp = fopen(fname.c_str(), "w");
        if (!fp) { cerr << "cannot open " << fname << "\n"; return; }
        fprintf(fp, "# shell-averaged energy spectrum, t = %.6f\n# k  E(k)\n", time);
        for (int s = 0; s < nsh; ++s) fprintf(fp, "%d %.16e\n", s, glob[s]);
        fclose(fp);
    }
}

// 最高波数壳层能量 E_hi = Σ_{|k| ≥ 0.8·kmax} ½|v̂|²/N²（含 R2C 权重）。
// kmax ≡ sqrt(Kmax) 为网格上最大的 |k|（三个方向各取最大波数），
// 即壳层条件 |k|² ≥ 0.64·Kmax。这样壳层包含 |k|²=Kmax 的"角"模态——
// RK4 粘性不稳定最先、最快出现在这些模态上（z = −P·Kmax·dt）。
double high_shell_energy(const AccCtx& ctx, Complex* const V[3], int Kmax) {
    double Kmin = 0.64 * Kmax;
    double local = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:local)
    for (int i = 0; i < ctx.osize[0]; ++i)
        for (int j = 0; j < ctx.osize[1]; ++j)
            for (int k = 0; k < ctx.osize[2]; ++k) {
                if (k2_index(ctx, i, j, k) < Kmin) continue;
                long idx = idx_c(ctx, i, j, k);
                double s = 0.0;
                for (int d = 0; d < 3; ++d)
                    s += V[d][idx][0]*V[d][idx][0] + V[d][idx][1]*V[d][idx][1];
                local += r2c_weight(ctx, k) * s;
            }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return 0.5 * global * ctx.norm_inv * ctx.norm_inv;
}

// 最后 W 个记录上 ln E_hi 对步数的最小二乘斜率 γ（单位：1/步）；记录不足或含非正值时返回 NaN
double fit_log_slope(const vector<long>& step, const vector<double>& e, int W) {
    int n = (int)e.size();
    if (n < W || W < 2) return NAN;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int q = n - W; q < n; ++q) {
        if (!(e[q] > 0.0) || !std::isfinite(e[q])) return NAN;
        double x = (double)step[q], y = log(e[q]);
        sx += x; sy += y; sxx += x*x; sxy += x*y;
    }
    double den = W*sxx - sx*sx;
    return den > 0 ? (W*sxy - sx*sy) / den : NAN;
}

// ==============================================================================
// 主程序
// ==============================================================================

enum Scheme { SCHEME_RK4, SCHEME_ETD4RK, SCHEME_COMPARE };

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 6) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0]
                 << " NX NY NZ dt NSTEPS [OMP_threads] [rk4|etd4rk|compare] [key=value ...]\n";
            cerr << "  keys: problem=poly|sin|tg  P=<visc>  U=<scale>  omega=<w>  amp=<A>"
                    "  diag=<K>  check=<K>  spec=<file>\n";
            cerr << "  Example: mpirun -np 4 " << argv[0] << " 64 64 64 0.001 1000 4 etd4rk\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    double dt = atof(argv[4]);
    int nt_run = atoi(argv[5]);
    int nthreads = (argc > 6 && atoi(argv[6]) > 0) ? atoi(argv[6]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    int max_threads = nthreads;

    // 可选参数：scheme（不含 '='）与 key=value
    Scheme scheme = SCHEME_ETD4RK;
    bool P_given = false;
    int diag_every = 0, check_every = 10;
    string spec_file;
    int ehi_every = 0, gamma_win = 20;
    double gamma_thr = 1e-3;
    string ehi_file;
    for (int a = 7; a < argc; ++a) {
        string s = argv[a];
        size_t eq = s.find('=');
        if (eq == string::npos) {
            if      (s == "rk4")     scheme = SCHEME_RK4;
            else if (s == "etd4rk")  scheme = SCHEME_ETD4RK;
            else if (s == "compare") scheme = SCHEME_COMPARE;
            else { if (rank == 0) cerr << "Unknown scheme: " << s << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
            continue;
        }
        string key = s.substr(0, eq), val = s.substr(eq + 1);
        if (key == "problem") {
            if      (val == "poly") g_cfg.prob = PROB_POLY;
            else if (val == "sin")  g_cfg.prob = PROB_SIN;
            else if (val == "tg")   g_cfg.prob = PROB_TG;
            else { if (rank == 0) cerr << "Unknown problem: " << val << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        else if (key == "P")     { g_cfg.P = atof(val.c_str()); P_given = true; }
        else if (key == "omega") g_cfg.omega = atof(val.c_str());
        else if (key == "amp")   g_cfg.amp = atof(val.c_str());
        else if (key == "U")     g_cfg.U = atof(val.c_str());
        else if (key == "diag")  diag_every = atoi(val.c_str());
        else if (key == "check") check_every = atoi(val.c_str());
        else if (key == "spec")  spec_file = val;
        else if (key == "ehi")       ehi_every = atoi(val.c_str());
        else if (key == "ehi_file")  ehi_file = val;
        else if (key == "gamma_thr") gamma_thr = atof(val.c_str());
        else if (key == "gamma_win") gamma_win = atoi(val.c_str());
        else { if (rank == 0) cerr << "Unknown option: " << key << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (g_cfg.prob == PROB_TG && !P_given) g_cfg.P = 1.0 / 1600.0;   // Re = 1600

    const double Lx = 2*M_PI, Ly = 2*M_PI, Lz = 2*M_PI;
    double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    // ------------------------------------------------------------------
    // 初始化 AccFFT（2D pencil 分解）
    // ------------------------------------------------------------------
    AccCtx ctx;
    ctx.nx = nx;  ctx.ny = ny;  ctx.nz = nz;
    ctx.norm_inv = 1.0 / static_cast<double>((long)nx * ny * nz);

    int c_dims[2] = {0, 0};   // {0,0} 让 AccFFT 自动选择最优进程网格
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

    int n[3] = {nx, ny, nz};
    int alloc_max = accfft_local_size_dft_r2c(n, ctx.isize, ctx.istart,
                                               ctx.osize, ctx.ostart, c_comm);

    ctx.local_size_r = (long)ctx.isize[0] * ctx.isize[1] * nz;
    ctx.local_size_c = (long)ctx.osize[0] * ctx.osize[1] * ctx.osize[2];

    accfft_init(max_threads);

    const char* prob_name = g_cfg.prob == PROB_POLY ? "poly (t^2+1)"
                          : g_cfg.prob == PROB_SIN  ? "sin (1+A sin(wt))" : "taylor-green";
    const char* scheme_name = scheme == SCHEME_RK4 ? "rk4"
                            : scheme == SCHEME_ETD4RK ? "etd4rk" : "compare (rk4 vs etd4rk)";
    if (rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes Solver - AccFFT Version (2D Pencil) + ETD4RK\n";
        cout << "============================================================\n";
        cout << "Grid: " << nx << " x " << ny << " x " << nz
             << ", dt=" << scientific << dt << ", steps=" << nt_run << "\n";
        cout << "Domain: [0, 2π]^3\n";
        cout << "MPI processes: " << nprocs << "\n";
        cout << "OpenMP threads/process: " << max_threads << "\n";
        cout << "Scheme: " << scheme_name << "\n";
        cout << "Problem: " << prob_name << ", P=" << g_cfg.P << ", U=" << g_cfg.U;
        if (g_cfg.prob == PROB_SIN) cout << ", omega=" << g_cfg.omega << ", A=" << g_cfg.amp;
        cout << "\n";
        cout << "============================================================\n";
        cout << "Real space local:     ["
             << ctx.istart[0] << "+" << ctx.isize[0] << "] x ["
             << ctx.istart[1] << "+" << ctx.isize[1] << "] x [0+" << ctx.isize[2] << "]\n";
        cout << "Spectral space local: ["
             << ctx.ostart[0] << "+" << ctx.osize[0] << "] x ["
             << ctx.ostart[1] << "+" << ctx.osize[1] << "] x ["
             << ctx.ostart[2] << "+" << ctx.osize[2] << "]\n";
    }

    // ------------------------------------------------------------------
    // 预分配所有内存（热循环中零动态分配）— 与原程序相同
    // ------------------------------------------------------------------
    long rsz = ctx.local_size_r * sizeof(double);
    long csz = (long)alloc_max;

    double*  V1_r = (double*)  accfft_alloc(rsz);
    double*  V2_r = (double*)  accfft_alloc(rsz);
    double*  V3_r = (double*)  accfft_alloc(rsz);
    Complex* V1_c = (Complex*) accfft_alloc(csz);
    Complex* V2_c = (Complex*) accfft_alloc(csz);
    Complex* V3_c = (Complex*) accfft_alloc(csz);

    Work w;
    w.V1_r = V1_r; w.V2_r = V2_r; w.V3_r = V3_r;
    w.rot1_r = (double*) accfft_alloc(rsz);
    w.rot2_r = (double*) accfft_alloc(rsz);
    w.rot3_r = (double*) accfft_alloc(rsz);
    w.rot1_c = (Complex*) accfft_alloc(csz);
    w.rot2_c = (Complex*) accfft_alloc(csz);
    w.rot3_c = (Complex*) accfft_alloc(csz);
    w.work_r1 = (double*) accfft_alloc(rsz);
    w.work_r2 = (double*) accfft_alloc(rsz);
    w.work_r3 = (double*) accfft_alloc(rsz);
    w.div_c = (Complex*) accfft_alloc(csz);
    w.phi_c = (Complex*) accfft_alloc(csz);
    w.nl1_c = (Complex*) accfft_alloc(csz);
    w.nl2_c = (Complex*) accfft_alloc(csz);
    w.nl3_c = (Complex*) accfft_alloc(csz);
    w.visc1_c = (Complex*) accfft_alloc(csz);
    w.visc2_c = (Complex*) accfft_alloc(csz);
    w.visc3_c = (Complex*) accfft_alloc(csz);
    w.f1_c = (Complex*) accfft_alloc(csz);
    w.f2_c = (Complex*) accfft_alloc(csz);
    w.f3_c = (Complex*) accfft_alloc(csz);
    for (int d = 0; d < 3; ++d) {
        w.k1[d] = (Complex*) accfft_alloc(csz);
        w.k2[d] = (Complex*) accfft_alloc(csz);
        w.k3[d] = (Complex*) accfft_alloc(csz);
        w.k4[d] = (Complex*) accfft_alloc(csz);
        w.tmp_v[d] = (Complex*) accfft_alloc(csz);
    }
    w.tmp_c = (Complex*) accfft_alloc(csz);

    Complex* const V[3] = {V1_c, V2_c, V3_c};

    // compare 模式额外保存 rk4 的结果（只有这个模式多 3 个复数数组）
    Complex* Vsave[3] = {nullptr, nullptr, nullptr};
    if (scheme == SCHEME_COMPARE)
        for (int d = 0; d < 3; ++d) Vsave[d] = (Complex*) accfft_alloc(csz);

    // ------------------------------------------------------------------
    // 创建 FFT 计划
    // ------------------------------------------------------------------
    if (rank == 0) cout << "Creating AccFFT plan...\n";
    MPI_Barrier(c_comm);
    double t_plan = MPI_Wtime();
    accfft_plan* plan = accfft_plan_dft_3d_r2c(n, V1_r, (double*)V1_c,
                                                c_comm, ACCFFT_MEASURE);
    MPI_Barrier(c_comm);
    if (rank == 0)
        cout << "AccFFT plan time: " << MPI_Wtime() - t_plan << " s\n";

    // ------------------------------------------------------------------
    // ETD4RK 系数表（dt 确定后只算一次）
    // ------------------------------------------------------------------
    EtdTable etd;
    {
        double t_tab = MPI_Wtime();
        etd = build_etd_table(nx, ny, nz, g_cfg.P, dt);
        t_tab = MPI_Wtime() - t_tab;
        if (rank == 0) {
            cout << scientific << setprecision(4);
            cout << "Kmax=|k|^2_max: " << etd.Kmax
                 << ",  |c|max*dt = P*Kmax*dt = " << g_cfg.P * etd.Kmax * dt << "\n";
            cout << "RK4 viscous stability bound dt <= 2.785/(P*Kmax) = "
                 << 2.785 / (g_cfg.P * etd.Kmax) << "\n";
            cout << "ETD table: " << etd.Kmax + 1 << " entries, "
                 << (etd.Kmax + 1) * 6 * sizeof(double) / 1024.0 << " KiB, built in "
                 << t_tab << " s\n";
        }
    }

    // ------------------------------------------------------------------
    // 初始条件（t=0）：实空间赋值 → r2c → 投影
    // ------------------------------------------------------------------
    auto set_initial = [&]() {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < ctx.isize[0]; ++i)
            for (int j = 0; j < ctx.isize[1]; ++j)
                for (int k = 0; k < nz; ++k) {
                    long ridx = idx_r(ctx, i, j, k);
                    double v[3];
                    exact_V(phys_x(ctx, i, dx), phys_y(ctx, j, dy), phys_z(k, dz), 0.0, v);
                    V1_r[ridx] = v[0]; V2_r[ridx] = v[1]; V3_r[ridx] = v[2];
                }
        accfft_execute_r2c(plan, V1_r, V1_c);
        accfft_execute_r2c(plan, V2_r, V2_c);
        accfft_execute_r2c(plan, V3_r, V3_c);
        make_div_free(ctx, V1_c, V2_c, V3_c, w.div_c, w.phi_c);
    };

    // ------------------------------------------------------------------
    // 误差：返回 {L2 误差, L∞ 误差, L2 范数(精确解)}
    // ------------------------------------------------------------------
    auto compute_error = [&](double time) -> vector<double> {
        Complex* tmp_c = w.tmp_c;
        memcpy(tmp_c, V1_c, ctx.local_size_c * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V1_r);
        memcpy(tmp_c, V2_c, ctx.local_size_c * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V2_r);
        memcpy(tmp_c, V3_c, ctx.local_size_c * sizeof(Complex));
        accfft_execute_c2r(plan, tmp_c, V3_r);

        double inv_n = ctx.norm_inv;
        #pragma omp parallel for
        for (long idx = 0; idx < ctx.local_size_r; ++idx) {
            V1_r[idx] *= inv_n; V2_r[idx] *= inv_n; V3_r[idx] *= inv_n;
        }

        double loc[3] = {0, 0, 0};   // Σ|e|², max|e|, Σ|v_exact|²
        for (int i = 0; i < ctx.isize[0]; ++i)
            for (int j = 0; j < ctx.isize[1]; ++j)
                for (int k = 0; k < nz; ++k) {
                    long ridx = idx_r(ctx, i, j, k);
                    double v[3];
                    exact_V(phys_x(ctx, i, dx), phys_y(ctx, j, dy), phys_z(k, dz), time, v);
                    double d1 = V1_r[ridx] - v[0], d2 = V2_r[ridx] - v[1], d3 = V3_r[ridx] - v[2];
                    double e = d1*d1 + d2*d2 + d3*d3;
                    loc[0] += e;
                    loc[1]  = max(loc[1], sqrt(e));
                    loc[2] += v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
                }
        double gs[2] = {0, 0}, gm = 0.0, ls[2] = {loc[0], loc[2]};
        MPI_Reduce(ls, gs, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc[1], &gm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double dV = dx * dy * dz;
        return {sqrt(gs[0] * dV), gm, sqrt(gs[1] * dV)};
    };

    // ------------------------------------------------------------------
    // max|u|（实空间，c2r 前复制保护 V_c；使用 V_r 作为工作区，诊断不计入计时）
    // ------------------------------------------------------------------
    auto velocity_max = [&]() -> double {
        double* Vr[3] = {V1_r, V2_r, V3_r};
        for (int d = 0; d < 3; ++d) {
            memcpy(w.tmp_c, V[d], ctx.local_size_c * sizeof(Complex));
            accfft_execute_c2r(plan, w.tmp_c, Vr[d]);
        }
        double local = 0.0;
        #pragma omp parallel for reduction(max:local)
        for (long idx = 0; idx < ctx.local_size_r; ++idx) {
            double u1 = V1_r[idx], u2 = V2_r[idx], u3 = V3_r[idx];
            double m = sqrt(u1*u1 + u2*u2 + u3*u3);
            if (!(m <= local)) local = m;   // NaN 也会传播
        }
        local *= ctx.norm_inv;
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return global;
    };

    // ------------------------------------------------------------------
    // 时间推进
    // γ 判据（ehi>0 时启用）：每 ehi 步记录 E_hi；运行结束（或爆炸）时用最后 gamma_win 个记录
    //   拟合 ln E_hi 对步数的斜率 γ，γ > gamma_thr → 判为不稳定。爆炸判据保留作兜底。
    // 爆炸判据：每 check_every 步（及最后一步）计算动能 E；
    //   E 非有限 或 E > 1e3·E(0) → 判为爆炸，提前结束（诊断不计入计时）
    // ------------------------------------------------------------------
    auto run = [&](Scheme s) {
        set_initial();
        double E0 = kinetic_energy(ctx, V);
        double t_wall_total = 0.0;
        long steps_done = 0;
        bool blowup = false;
        const char* tag = (s == SCHEME_RK4) ? "rk4" : "etd4rk";
        vector<long> rec_step;
        vector<double> rec_ehi, rec_ratio;   // E_hi 与 E_hi/E_total
        double umax_rec = 0.0;
        FILE* fe = nullptr;
        if (ehi_every > 0 && rank == 0 && !ehi_file.empty()) {
            string f = ehi_file;
            if (scheme == SCHEME_COMPARE) f += string(".") + tag;
            fe = fopen(f.c_str(), "w");
            if (fe) fprintf(fe, "# scheme=%s N=%d dt=%.10e P=%.6e Kmax=%d shell |k|^2>=0.64*Kmax\n"
                                "# step t E_hi E_total umax\n", tag, nx, dt, g_cfg.P, etd.Kmax);
        }
        auto record = [&](long step) {
            double eh = high_shell_energy(ctx, V, etd.Kmax);
            double et = kinetic_energy(ctx, V);
            double um = velocity_max();
            rec_step.push_back(step);
            rec_ehi.push_back(eh);
            rec_ratio.push_back(eh / et);
            if (std::isfinite(um)) umax_rec = max(umax_rec, um);
            if (fe) fprintf(fe, "%ld %.10e %.10e %.10e %.10e\n", step, step * dt, eh, et, um);
            return et;
        };
        if (ehi_every > 0) record(0);

        for (long it = 0; it < nt_run; ++it) {
            double t_cur = it * dt;
            double t0 = MPI_Wtime();
            if (s == SCHEME_RK4) rk4_step(ctx, plan, w, V, dt, t_cur, dx, dy, dz);
            else                 etd4rk_step(ctx, plan, w, V, etd, dt, t_cur, dx, dy, dz);
            double step_time = MPI_Wtime() - t0;
            double global_step;
            MPI_Allreduce(&step_time, &global_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            t_wall_total += global_step;
            steps_done = it + 1;

            bool last = (it + 1 == nt_run);
            bool do_diag  = diag_every > 0 && ((it + 1) % diag_every == 0 || last);
            bool do_check = check_every > 0 && ((it + 1) % check_every == 0 || last);
            bool do_ehi   = ehi_every > 0 && ((it + 1) % ehi_every == 0 || last);
            double Ek_rec = do_ehi ? record(it + 1) : NAN;
            if (do_diag || do_check || do_ehi) {
                double Ek = do_ehi ? Ek_rec : kinetic_energy(ctx, V);
                if (do_diag && rank == 0)
                    cout << "  [" << tag << "] step " << it + 1 << "  t=" << fixed << setprecision(6)
                         << (it + 1) * dt << "  E=" << scientific << setprecision(10) << Ek << "\n";
                if (!std::isfinite(Ek) || Ek > 1e3 * E0) {
                    blowup = true;
                    if (rank == 0)
                        cout << "  BLOWUP at step " << it + 1 << " (t=" << fixed << setprecision(6)
                             << (it + 1) * dt << "), E=" << scientific << Ek
                             << ", E0=" << E0 << "\n";
                    break;
                }
            }
        }

        if (fe) fclose(fe);
        double gamma = (ehi_every > 0) ? fit_log_slope(rec_step, rec_ehi, gamma_win) : NAN;
        bool gamma_unstable = (ehi_every > 0) && std::isfinite(gamma) && gamma > gamma_thr;
        // 归一化变体 γ_n：ln(E_hi/E_total) 的斜率。有外力且 E(t) 随时间变化（如 sin 问题，E ∝ g(t)²）时，
        // 物理的 E_hi 也 ∝ E，原始 γ 会把物理变化误判为不稳定；γ_n 消去了这部分（见 stability/STABILITY.md）。
        double gamma_n = (ehi_every > 0) ? fit_log_slope(rec_step, rec_ratio, gamma_win) : NAN;
        bool gamma_n_unstable = (ehi_every > 0) && std::isfinite(gamma_n) && gamma_n > gamma_thr;
        double t_final = steps_done * dt;
        double Ek = kinetic_energy(ctx, V);
        double divmax = compute_div_max(ctx, V1_c, V2_c, V3_c) * ctx.norm_inv;
        if (rank == 0) cout << "------------------------------------------------------------\n"
                            << "  Result [" << tag << "]\n";
        if (has_analytic()) {
            auto err = compute_error(t_final);
            if (rank == 0) {
                cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(4) << err[0] << "\n";
                cout << "  Rel L2 error:    " << err[0] / err[2] << "\n";
                cout << "  Linf error:      " << err[1] << "\n";
            }
        }
        if (rank == 0) {
            cout << "  Kinetic energy (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(10) << Ek << "\n";
            cout << "  max|div V|:      " << setprecision(4) << divmax << "\n";
            if (ehi_every > 0) {
                cout << "  E_hi final:      " << scientific << setprecision(4) << rec_ehi.back() << "\n";
                cout << "  Gamma (ln E_hi slope, last " << gamma_win << " records): "
                     << setprecision(4) << gamma << " /step"
                     << "   amp factor/step exp(gamma/2) = " << setprecision(6) << exp(0.5 * gamma) << "\n";
                cout << "  Gamma_n (ln(E_hi/E) slope): " << setprecision(4) << gamma_n << " /step\n";
                cout << "  Stable (gamma_n criterion): "
                     << (blowup ? "NO (blowup)" : gamma_n_unstable ? "NO (gamma_n)" : "yes") << "\n";
                cout << "  Umax (max over records): " << setprecision(6) << umax_rec << "\n";
            }
            cout << "  Stable:          "
                 << (blowup ? "NO (blowup)" : gamma_unstable ? "NO (gamma)" : "yes") << "\n";
            cout << "============================================================\n";
            cout << "  Timing Summary [" << tag << "]\n";
            cout << "------------------------------------------------------------\n";
            cout << "  Total steps:     " << steps_done << "\n";
            cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s\n";
            cout << "  Avg per step:    " << t_wall_total / max(steps_done, 1L) << " s\n";
            cout << "============================================================\n";
        }
        if (!spec_file.empty()) {
            string f = spec_file;
            if (scheme == SCHEME_COMPARE) f += string(".") + tag;
            write_spectrum(ctx, V, f, t_final);
        }
    };

    if (scheme == SCHEME_COMPARE) {
        run(SCHEME_RK4);
        for (int d = 0; d < 3; ++d)
            memcpy(Vsave[d], V[d], ctx.local_size_c * sizeof(Complex));
        run(SCHEME_ETD4RK);
        // ||V_rk4 − V_etd||_L2 = sqrt( 2·E(diff)·(2π)³ )（Parseval，与实空间求和等价）
        for (int d = 0; d < 3; ++d) {
            #pragma omp parallel for
            for (long i = 0; i < ctx.local_size_c; ++i) {
                w.tmp_v[d][i][0] = Vsave[d][i][0] - V[d][i][0];
                w.tmp_v[d][i][1] = Vsave[d][i][1] - V[d][i][1];
            }
        }
        double Ed = kinetic_energy(ctx, w.tmp_v);
        double En = kinetic_energy(ctx, V);
        if (rank == 0)
            cout << "  Diff L2 (rk4 vs etd4rk): " << scientific << setprecision(4)
                 << sqrt(2.0 * Ed * Lx * Ly * Lz)
                 << "   relative: " << sqrt(Ed / En) << "\n";
    } else {
        run(scheme);
    }

    // ------------------------------------------------------------------
    // 清理
    // ------------------------------------------------------------------
    accfft_free(V1_r);    accfft_free(V2_r);    accfft_free(V3_r);
    accfft_free(V1_c);    accfft_free(V2_c);    accfft_free(V3_c);
    accfft_free(w.rot1_r); accfft_free(w.rot2_r); accfft_free(w.rot3_r);
    accfft_free(w.rot1_c); accfft_free(w.rot2_c); accfft_free(w.rot3_c);
    accfft_free(w.work_r1); accfft_free(w.work_r2); accfft_free(w.work_r3);
    accfft_free(w.div_c);  accfft_free(w.phi_c);
    accfft_free(w.nl1_c);  accfft_free(w.nl2_c);  accfft_free(w.nl3_c);
    accfft_free(w.visc1_c); accfft_free(w.visc2_c); accfft_free(w.visc3_c);
    accfft_free(w.f1_c);   accfft_free(w.f2_c);   accfft_free(w.f3_c);
    for (int d = 0; d < 3; ++d) {
        accfft_free(w.k1[d]); accfft_free(w.k2[d]); accfft_free(w.k3[d]);
        accfft_free(w.k4[d]); accfft_free(w.tmp_v[d]);
        if (Vsave[d]) accfft_free(Vsave[d]);
    }
    accfft_free(w.tmp_c);

    accfft_destroy_plan(plan);
    MPI_Comm_free(&c_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    accfft_cleanup();
    MPI_Finalize();
    return 0;
}
