/**
 * heFFTe专用的频谱空间操作
 * 适配pencil decomposition
 */
#pragma once

#include <vector>
#include <complex>
#include <heffte.h>
#include <cmath>

// 线性索引助手：C-order，x 最快，z 最慢
inline size_t heffte_box_index(const heffte::box3d<> &box, int i, int j, int k) {
    int nx = box.size[0];
    int ny = box.size[1];
    int local_i = i - box.low[0];
    int local_j = j - box.low[1];
    int local_k = k - box.low[2];
    return static_cast<size_t>(local_i) + static_cast<size_t>(local_j) * nx +
           static_cast<size_t>(local_k) * nx * ny;
}

/**
 * 计算旋度 rot(V) = ∇ × V (频谱空间)
 */
inline void heffte_compute_rot(
    const std::vector<std::complex<double>>& V1_c,
    const std::vector<std::complex<double>>& V2_c,
    const std::vector<std::complex<double>>& V3_c,
    std::vector<std::complex<double>>& rot1_c,
    std::vector<std::complex<double>>& rot2_c,
    std::vector<std::complex<double>>& rot3_c,
    int nx, int ny, int nz,
    const heffte::box3d<>& box)
{
    for(int i = box.low[0]; i <= box.high[0]; ++i) {
        for(int j = box.low[1]; j <= box.high[1]; ++j) {
            for(int k = box.low[2]; k <= box.high[2]; ++k) {
                size_t idx = heffte_box_index(box, i, j, k);
                // 整数波数（域长度=2π时，物理波数=整数波数）
                double kx = (i <= nx/2) ? i : i - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;  // R2C: kz总是非负

                // rot = ik × V
                // i * complex = complex(- imag, real)
                double rot1_real = -(ky * V3_c[idx].imag() - kz * V2_c[idx].imag());
                double rot1_imag =  (ky * V3_c[idx].real() - kz * V2_c[idx].real());

                double rot2_real = -(kz * V1_c[idx].imag() - kx * V3_c[idx].imag());
                double rot2_imag =  (kz * V1_c[idx].real() - kx * V3_c[idx].real());

                double rot3_real = -(kx * V2_c[idx].imag() - ky * V1_c[idx].imag());
                double rot3_imag =  (kx * V2_c[idx].real() - ky * V1_c[idx].real());

                rot1_c[idx] = std::complex<double>(rot1_real, rot1_imag);
                rot2_c[idx] = std::complex<double>(rot2_real, rot2_imag);
                rot3_c[idx] = std::complex<double>(rot3_real, rot3_imag);
            }
        }
    }
}

/**
 * 计算散度 div(V) = ∇ · V (频谱空间)
 */
inline void heffte_compute_div(
    const std::vector<std::complex<double>>& V1_c,
    const std::vector<std::complex<double>>& V2_c,
    const std::vector<std::complex<double>>& V3_c,
    std::vector<std::complex<double>>& div_c,
    int nx, int ny, int nz,
    const heffte::box3d<>& box)
{
    for(int i = box.low[0]; i <= box.high[0]; ++i) {
        for(int j = box.low[1]; j <= box.high[1]; ++j) {
            for(int k = box.low[2]; k <= box.high[2]; ++k) {
                size_t idx = heffte_box_index(box, i, j, k);
                double kx = (i <= nx/2) ? i : i - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                // div = ik · V = i(kx*V1 + ky*V2 + kz*V3)
                double div_real = -(kx * V1_c[idx].imag() + ky * V2_c[idx].imag() + kz * V3_c[idx].imag());
                double div_imag =  (kx * V1_c[idx].real() + ky * V2_c[idx].real() + kz * V3_c[idx].real());

                div_c[idx] = std::complex<double>(div_real, div_imag);
            }
        }
    }
}

/**
 * 计算粘性项 P∆V = -k²V (频谱空间)
 */
inline void heffte_compute_viscous_term(
    const std::vector<std::complex<double>>& V_c,
    std::vector<std::complex<double>>& viscous_c,
    int nx, int ny, int nz,
    const heffte::box3d<>& box)
{
    for(int i = box.low[0]; i <= box.high[0]; ++i) {
        for(int j = box.low[1]; j <= box.high[1]; ++j) {
            for(int k = box.low[2]; k <= box.high[2]; ++k) {
                size_t idx = heffte_box_index(box, i, j, k);
                double kx = (i <= nx/2) ? i : i - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                double k2 = kx*kx + ky*ky + kz*kz;

                // P∆V = -k²V
                viscous_c[idx] = -k2 * V_c[idx];
            }
        }
    }
}

/**
 * 投影到无散空间：V = V - ∇φ，其中 ∇²φ = ∇·V
 */
inline void heffte_make_div_free(
    std::vector<std::complex<double>>& V1_c,
    std::vector<std::complex<double>>& V2_c,
    std::vector<std::complex<double>>& V3_c,
    std::vector<std::complex<double>>& div_c,
    std::vector<std::complex<double>>& phi_c,
    int nx, int ny, int nz,
    const heffte::box3d<>& box)
{
    // 1. 计算散度
    heffte_compute_div(V1_c, V2_c, V3_c, div_c, nx, ny, nz, box);

    // 2. 求解Poisson方程：∇²φ = div → φ = div / (-k²)
    for(int i = box.low[0]; i <= box.high[0]; ++i) {
        for(int j = box.low[1]; j <= box.high[1]; ++j) {
            for(int k = box.low[2]; k <= box.high[2]; ++k) {
                size_t idx = heffte_box_index(box, i, j, k);
                double kx = (i <= nx/2) ? i : i - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                double k2 = kx*kx + ky*ky + kz*kz;

                if (k2 > 1e-10) {
                    phi_c[idx] = div_c[idx] / (-k2);
                } else {
                    phi_c[idx] = 0.0;
                }
            }
        }
    }

    // 3. V = V - ∇φ
    for(int i = box.low[0]; i <= box.high[0]; ++i) {
        for(int j = box.low[1]; j <= box.high[1]; ++j) {
            for(int k = box.low[2]; k <= box.high[2]; ++k) {
                size_t idx = heffte_box_index(box, i, j, k);
                double kx = (i <= nx/2) ? i : i - nx;
                double ky = (j <= ny/2) ? j : j - ny;
                double kz = k;

                // ∇φ = ikφ
                // V -= ikφ → V -= i*phi
                std::complex<double> grad_phi_x(-kx * phi_c[idx].imag(), kx * phi_c[idx].real());
                std::complex<double> grad_phi_y(-ky * phi_c[idx].imag(), ky * phi_c[idx].real());
                std::complex<double> grad_phi_z(-kz * phi_c[idx].imag(), kz * phi_c[idx].real());

                V1_c[idx] -= grad_phi_x;
                V2_c[idx] -= grad_phi_y;
                V3_c[idx] -= grad_phi_z;
            }
        }
    }
}
#include <heffte.h>
