#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3-mpi.h>
#include <iomanip> 
#include <mpi.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <omp.h>

using namespace std;

// Глобальные переменные
fftw_plan plan_r2r_cos;
fftw_plan plan_r2r_sin;

fftw_plan plan_fwd_r2c_cos;
fftw_plan plan_bwd_c2r_cos;

fftw_plan plan_fwd_r2c_sin;
fftw_plan plan_bwd_c2r_sin;


void initialize_r2r_cos(ptrdiff_t local_nx, ptrdiff_t ny, ptrdiff_t nz, double *re_in, double *re_out) 
{   
    //Forward transformation для cos r2r
    const fftw_r2r_kind kind_cos[] = {FFTW_REDFT00};
    const int nz_int = nz/2+1;
    plan_r2r_cos = fftw_plan_many_r2r(1, &nz_int, local_nx*(2*(ny/2+1)),
                            re_in, &nz_int, 1, nz_int,
                            re_out, &nz_int, 1, nz_int,
                            kind_cos, FFTW_PATIENT);        
}

void initialize_r2r_sin(ptrdiff_t local_nx, ptrdiff_t ny, ptrdiff_t nz, double *re_in, double *re_out)  
{   
    //Forward transformation для sin r2r
    const fftw_r2r_kind kind_sin[] = {FFTW_RODFT00};
    const int nz_int = nz/2-1;
    plan_r2r_sin = fftw_plan_many_r2r(1, &nz_int, local_nx*(2*(ny/2+1)),
                            re_in, &nz_int, 1, nz_int,
                            re_out, &nz_int, 1, nz_int,
                            kind_sin, FFTW_PATIENT);       
}

void initialize_fwd_r2c_cos(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, double *re_in, fftw_complex* complex_out)
{   
    //Forward transformation для cos r2c
    const ptrdiff_t nz_int = nz/2+1;                           
    ptrdiff_t nn1[] = {nx, ny};
    plan_fwd_r2c_cos = fftw_mpi_plan_many_dft_r2c(2, nn1, nz_int, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                             re_in, complex_out, MPI_COMM_WORLD, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_OUT);                          
}

void initialize_fwd_r2c_sin(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, double *re_in, fftw_complex* complex_out)
{   
    //Forward transformation для sin r2c
    const ptrdiff_t nz_int = nz/2-1;                           
    ptrdiff_t nn1[] = {nx, ny};
    plan_fwd_r2c_sin = fftw_mpi_plan_many_dft_r2c(2, nn1, nz_int, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                             re_in, complex_out, MPI_COMM_WORLD, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_OUT);                          
}


void initialize_bwd_c2r_cos(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, fftw_complex* complex_out, double *re_in)
{ 
    //Backward transformation для cos c2r
    const ptrdiff_t nz_int = nz/2+1;                           
    ptrdiff_t nn1[] = {nx, ny};
    plan_bwd_c2r_cos = fftw_mpi_plan_many_dft_c2r(2, nn1, nz_int, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                              complex_out, re_in, MPI_COMM_WORLD, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_IN);
}

void initialize_bwd_c2r_sin(ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, fftw_complex* complex_out, double *re_in)
{ 
    //Backward transformation для sin c2r
    const ptrdiff_t nz_int = nz/2-1;                           
    ptrdiff_t nn1[] = {nx, ny};
    plan_bwd_c2r_sin = fftw_mpi_plan_many_dft_c2r(2, nn1, nz_int, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                              complex_out, re_in, MPI_COMM_WORLD, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_IN);
}

void finalize_fft_plans() {
    fftw_destroy_plan(plan_r2r_cos);
    fftw_destroy_plan(plan_r2r_sin);
    fftw_destroy_plan(plan_fwd_r2c_cos);
    fftw_destroy_plan(plan_bwd_c2r_cos);
    fftw_destroy_plan(plan_fwd_r2c_sin);
    fftw_destroy_plan(plan_bwd_c2r_sin);
}

double func_V1(double x, double y, double z, double t) {
    return (t*t+1)*exp(sin(3*x+3*y))*cos(6*z);
}
double func_V2(double x, double y, double z, double t) {
    return (t*t+1)*exp(sin(3*x+3*y))*cos(6*z);
}
double func_V3(double x, double y, double z, double t) {
    return -(t*t+1)*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
}

double func_dV1_dt(double x, double y, double z, double t) {
    return 2*t*exp(sin(3*x+3*y))*cos(6*z);
}

double func_dV2_dt(double x, double y, double z, double t) {
    return 2*t*exp(sin(3*x+3*y))*cos(6*z);
}

double func_dV3_dt(double x, double y, double z, double t) {
    return -2*t*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
}

double func_laplace_V1(double x, double y, double z, double t) {
    double d2v1_dx2 = (t*t+1)*9*exp(sin(3*x+3*y))*((cos(3*x+3*y)*cos(3*x+3*y))-sin(3*x+3*y))*cos(6*z);
    double d2v1_dy2 = (t*t+1)*9*exp(sin(3*x+3*y))*((cos(3*x+3*y)*cos(3*x+3*y))-sin(3*x+3*y))*cos(6*z);
    double d2v1_dz2 = -(t*t+1)*36*exp(sin(3*x+3*y))*cos(6*z);
    return d2v1_dx2 + d2v1_dy2 + d2v1_dz2;
}

double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x,y,z,t);
}

double func_laplace_V3(double x, double y, double z, double t) {
    double d2v3_dx2 = -(t*t+1)*9*exp(sin(3*x+3*y))*cos(3*x+3*y)*((cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))-(2*sin(3*x+3*y)+1))*sin(6*z);
    double d2v3_dy2 = -(t*t+1)*9*exp(sin(3*x+3*y))*cos(3*x+3*y)*((cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))-(2*sin(3*x+3*y)+1))*sin(6*z);
    double d2v3_dz2 = (t*t+1)*36*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
    return d2v3_dx2 + d2v3_dy2 + d2v3_dz2;
}

double func_rot1(double x, double y, double z, double t){
    double dv3_dy = -(t*t+1)*3*exp(sin(3*x+3*y))*(cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))*sin(6*z);
    double dv2_dz = -(t*t+1)*6*exp(sin(3*x+3*y))*sin(6*z);
    return dv3_dy - dv2_dz;
}

double func_rot2(double x, double y, double z, double t){
    double dv1_dz = -(t*t+1)*6*exp(sin(3*x+3*y))*sin(6*z);
    double dv3_dx = -(t*t+1)*3*exp(sin(3*x+3*y))*(cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))*sin(6*z);
    return dv1_dz - dv3_dx;
}

double func_rot3(double x, double y, double z, double t){
    return 0;
}

double func_v_cross_rot1(double x, double y, double z, double t) {
    return func_V2(x,y,z,t)*func_rot3(x,y,z,t)-func_V3(x,y,z,t)*func_rot2(x,y,z,t);
}

double func_v_cross_rot2(double x, double y, double z, double t) {
    return func_V3(x,y,z,t)*func_rot1(x,y,z,t)-func_V1(x,y,z,t)*func_rot3(x,y,z,t);
}

double func_v_cross_rot3(double x, double y, double z, double t) {
    return func_V1(x,y,z,t)*func_rot2(x,y,z,t)-func_V2(x,y,z,t)*func_rot1(x,y,z,t);
}

double func_p(double x, double y, double z, double t) {
    return (t*t+1)*cos(x)*cos(y)*cos(z);
}

double func_grad_p1(double x, double y, double z, double t) {
    return -(t*t+1)*sin(x)*cos(y)*cos(z);
}
double func_grad_p2(double x, double y, double z, double t) {
    return -(t*t+1)*cos(x)*sin(y)*cos(z);
}
double func_grad_p3(double x, double y, double z, double t) {
    return -(t*t+1)*cos(x)*cos(y)*sin(z);
}

double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x,y,z,t)- func_laplace_V1(x,y,z,t)- func_v_cross_rot1(x,y,z,t) + func_grad_p1(x,y,z,t); 
}

double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t)- func_laplace_V2(x,y,z,t)- func_v_cross_rot2(x,y,z,t) + func_grad_p2(x,y,z,t); 
}

double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t)- func_laplace_V3(x,y,z,t)- func_v_cross_rot3(x,y,z,t) + func_grad_p3(x,y,z,t); 
}

void normalization(fftw_complex* V_c_, ptrdiff_t nx,ptrdiff_t local_ny, ptrdiff_t nz, double factor)
{   
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (j * nx + i) * nz + k;
                V_c_[index][0] /= factor;
                V_c_[index][1] /= factor;
            }
        }
    }
}

double calculateEnergy(double* V_r_, ptrdiff_t local_nx,ptrdiff_t ny, ptrdiff_t nz, double factor)
{   
    double sumSquares = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:sumSquares)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < nz; ++k) {
                ptrdiff_t index = (i * (2*(ny/2+1)) + j) * nz + k;
                sumSquares += V_r_[index] * V_r_[index];
            }
        }
    }
    double energy = 0.5 * sumSquares / factor;
    return energy;
}

void compute_rot(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_,
                fftw_complex* rot1_c_, fftw_complex* rot2_c_, fftw_complex* rot3_c_,
                ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                ptrdiff_t local_nx,ptrdiff_t local_ny,ptrdiff_t local_x_start,ptrdiff_t local_y_start)
{
    double alpha = 1;
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                ptrdiff_t global_j = local_y_start + j;
                ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
                ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
                ptrdiff_t k_z = k;
                rot3_c_[index][0] = -(V2_c_[index][1] * k_x - V1_c_[index][1] * k_y); 
                rot3_c_[index][1] = V2_c_[index][0] * k_x - V1_c_[index][0] * k_y;  
            }
        }
    }

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                ptrdiff_t index1 = (j * nx + i) * (nz/2+1) + k + 1;
                ptrdiff_t index2 = (j * nx + i) * (nz/2-1) + k;
                ptrdiff_t global_j = local_y_start + j;

                ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
                ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
                ptrdiff_t k_z = k;
                rot1_c_[index2][0] = (-V3_c_[index2][1] * (k_y)) -(-V2_c_[index1][0] * (k_z+1)*alpha); 
                rot1_c_[index2][1] = (V3_c_[index2][0] * (k_y)) - (-V2_c_[index1][1] * (k_z+1) *alpha); 
                rot2_c_[index2][0] = (-V1_c_[index1][0] * (k_z+1)*alpha) - (-V3_c_[index2][1] * (k_x)); 
                rot2_c_[index2][1] = (-V1_c_[index1][1] * (k_z+1)*alpha) - (V3_c_[index2][0] * (k_x));  
            }
        }
    }
}

void compute_div(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_,fftw_complex* div_c_, 
                ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                ptrdiff_t local_nx,ptrdiff_t local_ny,ptrdiff_t local_x_start,ptrdiff_t local_y_start)
{
    double alpha = 1;
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            ptrdiff_t global_j = local_y_start + j;
            ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
            ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
            for(ptrdiff_t k = 1; k <= (nz/2-1); ++k) {
                ptrdiff_t index1 = (j * nx + i) * (nz/2+1) + k;
                ptrdiff_t index2 = (j * nx + i) * (nz/2-1) + k-1;
                ptrdiff_t k_z = k;
                div_c_[index1][0] = (-V1_c_[index1][1] * k_x) + (-V2_c_[index1][1] * k_y) + (V3_c_[index2][0] * k_z*alpha);
                div_c_[index1][1] = (V1_c_[index1][0] * k_x) +  (V2_c_[index1][0] * k_y ) + (V3_c_[index2][1] * k_z*alpha); 
            }
            // k = 0
            ptrdiff_t index = (j * nx + i) * (nz/2+1) + 0;
            div_c_[index][0] = (-V1_c_[index][1] * k_x) + (-V2_c_[index][1] * k_y);
            div_c_[index][1] = (V1_c_[index][0] * k_x) +  (V2_c_[index][0] * k_y );

            // k = nz/2
            ptrdiff_t index2 = (j * nx + i) * (nz/2+1) + (nz/2+1)-1;
            div_c_[index2][0] = (-V1_c_[index2][1] * k_x) + (-V2_c_[index2][1] * k_y);
            div_c_[index2][1] = (V1_c_[index2][0] * k_x) +  (V2_c_[index2][0] * k_y );
        }
    }
}

void make_div_0(fftw_complex* V1_, fftw_complex* V2_, fftw_complex* V3_, fftw_complex* fi, fftw_complex* div_,
                ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
                ptrdiff_t local_nx,ptrdiff_t local_ny,ptrdiff_t local_x_start,ptrdiff_t local_y_start)
{
    double alpha = 1;
    compute_div(V1_,V2_,V3_,div_,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start);
    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t global_j = local_y_start + j;
                ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
                ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
                ptrdiff_t k_z = k;
                ptrdiff_t index1 = (j * nx + i) * (nz/2+1) + k;

                if (i==0 && global_j==0 && k==0) {
                    fi[index1][0] = 0;
                    fi[index1][1] = 0;
                } else {
                    fi[index1][0] = div_[index1][0]/ -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                    fi[index1][1] = div_[index1][1]/ -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                }
            }
        }
    }

    #pragma omp parallel for collapse(3)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t global_j = local_y_start + j;
                ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
                ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
                ptrdiff_t k_z = k;
                ptrdiff_t index1 = (j * nx + i) * (nz/2+1) + k;
                ptrdiff_t index2 = (j * nx + i) * (nz/2-1) + k-1;

                if (k != 0 && k != (nz/2+1) - 1) 
                {
                    V1_[index1][0] = V1_[index1][0] - (-fi[index1][1] * k_x);
                    V1_[index1][1] = V1_[index1][1] - (fi[index1][0] * k_x);
                    V2_[index1][0] = V2_[index1][0] - (-fi[index1][1] * k_y);
                    V2_[index1][1] = V2_[index1][1] - (fi[index1][0] * k_y);
                    V3_[index2][0] = V3_[index2][0] - (-fi[index1][0]* k_z*alpha);
                    V3_[index2][1] = V3_[index2][1] - (-fi[index1][1]* k_z*alpha);
                } else if (k == 0 || k == (nz/2+1) - 1) 
                {
                    V1_[index1][0] = V1_[index1][0] - (-fi[index1][1] * k_x);
                    V1_[index1][1] = V1_[index1][1] - (fi[index1][0] * k_x);
                    V2_[index1][0] = V2_[index1][0] - (-fi[index1][1] * k_y);
                    V2_[index1][1] = V2_[index1][1] - (fi[index1][0] * k_y);
                }
            }
        }
    }
}



void compute_v_cross_rot(fftw_complex* V1_c, fftw_complex *V2_c, fftw_complex *V3_c,
                        double *V1_r, double *V2_r, double *V3_r,
                        double *V1_xy_r, double *V2_xy_r, double *V3_xy_r,
                        fftw_complex *rotv1_c, fftw_complex *rotv2_c, fftw_complex *rotv3_c,
                        double *rotv1_r, double *rotv2_r, double *rotv3_r,
                        double *rotv1_xy_r, double *rotv2_xy_r, double *rotv3_xy_r,
                        fftw_complex *v_cross_rot1_c, fftw_complex *v_cross_rot2_c, fftw_complex *v_cross_rot3_c,
                        double *v_cross_rot1_r, double *v_cross_rot2_r, double *v_cross_rot3_r,
                        double *v_cross_rot1_z_r, double *v_cross_rot2_z_r, double *v_cross_rot3_z_r,
                        ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, 
                        ptrdiff_t local_nx, ptrdiff_t local_ny, ptrdiff_t local_x_start,ptrdiff_t local_y_start)
{   
    compute_rot(V1_c,V2_c,V3_c,rotv1_c,rotv2_c,rotv3_c,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start);

    // backward fft for rotV
    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_sin,rotv1_c,rotv1_xy_r);
    fftw_execute_r2r(plan_r2r_sin, rotv1_xy_r, rotv1_r);

    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_sin,rotv2_c,rotv2_xy_r);
    fftw_execute_r2r(plan_r2r_sin, rotv2_xy_r, rotv2_r);
    
    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_cos,rotv3_c,rotv3_xy_r);
    fftw_execute_r2r(plan_r2r_cos, rotv3_xy_r, rotv3_r);

    #pragma omp parallel for collapse(2)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                v_cross_rot1_c[index][0] = V1_c[index][0];
                v_cross_rot1_c[index][1] = V1_c[index][1];
                v_cross_rot2_c[index][0] = V2_c[index][0];
                v_cross_rot2_c[index][1] = V2_c[index][1];
            }
            for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                v_cross_rot3_c[index][0] = V3_c[index][0];
                v_cross_rot3_c[index][1] = V3_c[index][1];
            }
        }
    }

    // backward fft for V
    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_cos,v_cross_rot1_c,V1_xy_r);
    fftw_execute_r2r(plan_r2r_cos, V1_xy_r, V1_r);

    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_cos,v_cross_rot2_c,V2_xy_r);
    fftw_execute_r2r(plan_r2r_cos, V2_xy_r, V2_r);
    
    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_sin,v_cross_rot3_c,V3_xy_r);
    fftw_execute_r2r(plan_r2r_sin, V3_xy_r, V3_r);
    
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index1 = (i * (2*(ny/2+1)) + j) * (nz/2+1) + k;
                ptrdiff_t index2 = (i * (2*(ny/2+1)) + j) * (nz/2-1) + k-1;
                if (k != 0 && k != (nz/2+1) - 1) 
                {
                    
                    v_cross_rot1_r[index1] = V2_r[index1]*rotv3_r[index1] - V3_r[index2]*rotv2_r[index2];
                    v_cross_rot2_r[index1] = V3_r[index2]*rotv1_r[index2] - V1_r[index1]*rotv3_r[index1];
                    v_cross_rot3_r[index2] = V1_r[index1]*rotv2_r[index2] - V2_r[index1]*rotv1_r[index2];
                } else 
                {
                    v_cross_rot1_r[index1] = V2_r[index1]*rotv3_r[index1] - 0;
                    v_cross_rot2_r[index1] = 0 - V1_r[index1]*rotv3_r[index1];
                }
            }
        }
    }


    // Forward transformation  
    // Сделал fft для V x rotV, 
    fftw_execute_r2r(plan_r2r_cos, v_cross_rot1_r, v_cross_rot1_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, v_cross_rot1_z_r, v_cross_rot1_c);

    fftw_execute_r2r(plan_r2r_cos, v_cross_rot2_r, v_cross_rot2_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, v_cross_rot2_z_r, v_cross_rot2_c);

    fftw_execute_r2r(plan_r2r_sin, v_cross_rot3_r, v_cross_rot3_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_sin, v_cross_rot3_z_r, v_cross_rot3_c);

    normalization(v_cross_rot1_c, nx,local_ny,(nz/2+1),nx*ny*nz);
    normalization(v_cross_rot2_c, nx,local_ny,(nz/2+1),nx*ny*nz);
    normalization(v_cross_rot3_c, nx,local_ny,(nz/2-1),nx*ny*nz);
}

void compute_f(fftw_complex* f1_c_, fftw_complex* f2_c_, fftw_complex* f3_c_,
            double* f1_z_r_,double* f2_z_r_,double* f3_z_r_,
            double* f1_r_,double* f2_r_,double* f3_r_, 
            ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz, 
            ptrdiff_t local_nx, ptrdiff_t local_ny, ptrdiff_t local_x_start,ptrdiff_t local_y_start,
            double L_x, double L_y, double L_z,int nt, double tau, int it, double c)
{   
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index1 = (i * (2*(ny/2+1)) + j) * (nz/2+1) + k;
                f1_r_[index1] = func_f1(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,it*tau + c);
                f2_r_[index1] = func_f2(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,it*tau + c);
            }
            for(ptrdiff_t k = 1; k <= (nz/2-1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index2 = (i * (2*(ny/2+1)) + j) * (nz/2-1) + k-1;
                f3_r_[index2] = func_f3(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,it*tau + c);
            }
        }
    }
    // f
    fftw_execute_r2r(plan_r2r_cos, f1_r_, f1_z_r_);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, f1_z_r_, f1_c_);

    fftw_execute_r2r(plan_r2r_cos, f2_r_, f2_z_r_);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, f2_z_r_, f2_c_);

    fftw_execute_r2r(plan_r2r_sin, f3_r_, f3_z_r_);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_sin, f3_z_r_, f3_c_);

    normalization(f1_c_, nx,local_ny,(nz/2+1),nx*ny*nz);
    normalization(f2_c_, nx,local_ny,(nz/2+1),nx*ny*nz);
    normalization(f3_c_, nx,local_ny,(nz/2-1),nx*ny*nz);
}

void compute_F(fftw_complex* F1_c_, fftw_complex* F2_c_, fftw_complex* F3_c_,
            fftw_complex* f1_c_,fftw_complex* f2_c_,fftw_complex* f3_c_,
            double* f1_z_r_,double* f2_z_r_,double* f3_z_r_,
            double* f1_r_,double* f2_r_,double* f3_r_,
            double *V1_xy_r, double *V2_xy_r, double *V3_xy_r,
            fftw_complex *rot1_c, fftw_complex *rot2_c, fftw_complex *rot3_c,
            double *rot1_r, double *rot2_r, double *rot3_r,
            double *rot1_xy_r, double *rot2_xy_r, double *rot3_xy_r,
            fftw_complex* cross1_c_, fftw_complex* cross2_c_, fftw_complex* cross3_c_,
            double *cross1_r, double *cross2_r, double *cross3_r,
            double *cross1_z_r, double *cross2_z_r, double *cross3_z_r,
            fftw_complex* p_c,fftw_complex* div_c,
            ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
            ptrdiff_t local_nx,ptrdiff_t local_ny, ptrdiff_t local_x_start,ptrdiff_t local_y_start,
            double L_x, double L_y, double L_z,int nt, double tau, ptrdiff_t it, double c)
{   
    compute_v_cross_rot(F1_c_,F2_c_,F3_c_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                        cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start);
    compute_f(f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau,it,c);
    
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t j = 0; j < local_ny; ++j) {
        for(ptrdiff_t i = 0; i < nx; ++i) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t index1 = (j * nx + i) * (nz/2+1) + k;
                ptrdiff_t global_j = local_y_start + j;
                ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
                ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
                ptrdiff_t k_z = k;
                F1_c_[index1][0] = -F1_c_[index1][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross1_c_[index1][0] + f1_c_[index1][0];
                F1_c_[index1][1] = -F1_c_[index1][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross1_c_[index1][1] + f1_c_[index1][1];
                F2_c_[index1][0] = -F2_c_[index1][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross2_c_[index1][0] + f2_c_[index1][0];
                F2_c_[index1][1] = -F2_c_[index1][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross2_c_[index1][1] + f2_c_[index1][1];
            }
            for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                ptrdiff_t index2 = (j * nx + i) * (nz/2-1) + k;
                ptrdiff_t global_j = local_y_start + j;
                ptrdiff_t k_x = i <= nx/2 ? i : i -nx;
                ptrdiff_t k_y = global_j <= ny/2 ? global_j : global_j -ny;
                ptrdiff_t k_z = k;
                F3_c_[index2][0] = -F3_c_[index2][0] * (k_x * k_x + k_y * k_y + (k_z+1) * (k_z+1)) + cross3_c_[index2][0] + f3_c_[index2][0];
                F3_c_[index2][1] = -F3_c_[index2][1] * (k_x * k_x + k_y * k_y + (k_z+1) * (k_z+1)) + cross3_c_[index2][1] + f3_c_[index2][1];
            }
        }
    }

    make_div_0(F1_c_, F2_c_, F3_c_, p_c,div_c,nx, ny, nz, local_nx, local_ny, local_x_start, local_y_start);
}

void rungeKutta(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_,
            fftw_complex* F1_c_, fftw_complex* F2_c_, fftw_complex* F3_c_,
            fftw_complex* f1_c_,fftw_complex* f2_c_,fftw_complex* f3_c_,
            double* f1_z_r_,double* f2_z_r_,double* f3_z_r_,
            double* f1_r_,double* f2_r_,double* f3_r_,
            double *V1_xy_r, double *V2_xy_r, double *V3_xy_r,
            fftw_complex *rot1_c, fftw_complex *rot2_c, fftw_complex *rot3_c,
            double *rot1_r, double *rot2_r, double *rot3_r,
            double *rot1_xy_r, double *rot2_xy_r, double *rot3_xy_r,
            fftw_complex* cross1_c_, fftw_complex* cross2_c_, fftw_complex* cross3_c_,
            double *cross1_r, double *cross2_r, double *cross3_r,
            double *cross1_z_r, double *cross2_z_r, double *cross3_z_r,
            fftw_complex* p_c,fftw_complex* div_c,
            ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
            ptrdiff_t alloc_local, ptrdiff_t local_nx,ptrdiff_t local_ny,ptrdiff_t local_x_start,ptrdiff_t local_y_start,
            double L_x, double L_y, double L_z,ptrdiff_t nt, double tau)
{   
    fftw_complex *k1_V1 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k2_V1 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k3_V1 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k4_V1 = fftw_alloc_complex(alloc_local*(nz/2+1));

    fftw_complex *k1_V2 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k2_V2 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k3_V2 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k4_V2 = fftw_alloc_complex(alloc_local*(nz/2+1));

    fftw_complex *k1_V3 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k2_V3 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k3_V3 = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *k4_V3 = fftw_alloc_complex(alloc_local*(nz/2+1));

    for (ptrdiff_t l = 0; l < nt; l++) {
        #pragma omp parallel for collapse(2)
        for(ptrdiff_t j = 0; j < local_ny; ++j) {
            for(ptrdiff_t i = 0; i < nx; ++i) {
                for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                    F1_c_[index][0] = V1_c_[index][0];
                    F1_c_[index][1] = V1_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0];
                    F2_c_[index][1] = V2_c_[index][1];
                }
                for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                    F3_c_[index][0] = V3_c_[index][0];
                    F3_c_[index][1] = V3_c_[index][1];
                }
            }
        }

        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau,l,0);
        
        #pragma omp parallel for collapse(2)        
        for(ptrdiff_t j = 0; j < local_ny; ++j) {
            for(ptrdiff_t i = 0; i < nx; ++i) {
                for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                    k1_V1[index][0] = F1_c_[index][0];
                    k1_V1[index][1] = F1_c_[index][1];
                    F1_c_[index][0] = V1_c_[index][0] + tau / 2 * k1_V1[index][0];
                    F1_c_[index][1] = V1_c_[index][1] + tau / 2 * k1_V1[index][1];

                    k1_V2[index][0] = F2_c_[index][0];
                    k1_V2[index][1] = F2_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0] + tau / 2 * k1_V2[index][0];
                    F2_c_[index][1] = V2_c_[index][1] + tau / 2 * k1_V2[index][1];
                }
                for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                    k1_V3[index][0] = F3_c_[index][0];
                    k1_V3[index][1] = F3_c_[index][1];
                    F3_c_[index][0] = V3_c_[index][0] + tau / 2 * k1_V3[index][0];
                    F3_c_[index][1] = V3_c_[index][1] + tau / 2 * k1_V3[index][1];
                }
            }
        }

        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau,l,tau/2);
       
        #pragma omp parallel for collapse(2)                    
        for(ptrdiff_t j = 0; j < local_ny; ++j) {
            for(ptrdiff_t i = 0; i < nx; ++i) {
                for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                    k2_V1[index][0] = F1_c_[index][0];
                    k2_V1[index][1] = F1_c_[index][1];
                    F1_c_[index][0] = V1_c_[index][0] + tau / 2 * k2_V1[index][0];
                    F1_c_[index][1] = V1_c_[index][1] + tau / 2 * k2_V1[index][1];

                    k2_V2[index][0] = F2_c_[index][0];
                    k2_V2[index][1] = F2_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0] + tau / 2 * k2_V2[index][0];
                    F2_c_[index][1] = V2_c_[index][1] + tau / 2 * k2_V2[index][1];
                }
                for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                    k2_V3[index][0] = F3_c_[index][0];
                    k2_V3[index][1] = F3_c_[index][1];
                    F3_c_[index][0] = V3_c_[index][0] + tau / 2 * k2_V3[index][0];
                    F3_c_[index][1] = V3_c_[index][1] + tau / 2 * k2_V3[index][1];
                }
            }
        }

        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau,l,tau/2);
        
        #pragma omp parallel for collapse(2)        
        for(ptrdiff_t j = 0; j < local_ny; ++j) {
            for(ptrdiff_t i = 0; i < nx; ++i) {
                for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                    k3_V1[index][0] = F1_c_[index][0];
                    k3_V1[index][1] = F1_c_[index][1];
                    F1_c_[index][0] = V1_c_[index][0] + tau * k3_V1[index][0];
                    F1_c_[index][1] = V1_c_[index][1] + tau * k3_V1[index][1];

                    k3_V2[index][0] = F2_c_[index][0];
                    k3_V2[index][1] = F2_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0] + tau * k3_V2[index][0];
                    F2_c_[index][1] = V2_c_[index][1] + tau * k3_V2[index][1];
                }
                for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                    k3_V3[index][0] = F3_c_[index][0];
                    k3_V3[index][1] = F3_c_[index][1];
                    F3_c_[index][0] = V3_c_[index][0] + tau * k3_V3[index][0];
                    F3_c_[index][1] = V3_c_[index][1] + tau * k3_V3[index][1];
                }
            }
        }
        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau,l,tau);
        
        #pragma omp parallel for collapse(2)
        for(ptrdiff_t j = 0; j < local_ny; ++j) {
            for(ptrdiff_t i = 0; i < nx; ++i) {
                for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                    k4_V1[index][0] = F1_c_[index][0];
                    k4_V1[index][1] = F1_c_[index][1];

                    k4_V2[index][0] = F2_c_[index][0];
                    k4_V2[index][1] = F2_c_[index][1];
                }
                for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                    k4_V3[index][0] = F3_c_[index][0];
                    k4_V3[index][1] = F3_c_[index][1];
                }
            }
        }
        
        #pragma omp parallel for collapse(2)
        for(ptrdiff_t j = 0; j < local_ny; ++j) {
            for(ptrdiff_t i = 0; i < nx; ++i) {
                for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2+1) + k;
                    V1_c_[index][0] = V1_c_[index][0] + tau/6 *(k1_V1[index][0] + 2*k2_V1[index][0] + 2*k3_V1[index][0] + k4_V1[index][0]);
                    V1_c_[index][1] = V1_c_[index][1] + tau/6 *(k1_V1[index][1] + 2*k2_V1[index][1] + 2*k3_V1[index][1] + k4_V1[index][1]);
                    V2_c_[index][0] = V2_c_[index][0] + tau/6 *(k1_V2[index][0] + 2*k2_V2[index][0] + 2*k3_V2[index][0] + k4_V2[index][0]);
                    V2_c_[index][1] = V2_c_[index][1] + tau/6 *(k1_V2[index][1] + 2*k2_V2[index][1] + 2*k3_V2[index][1] + k4_V2[index][1]);
                }
                for(ptrdiff_t k = 0; k < (nz/2-1); ++k) {
                    ptrdiff_t index = (j * nx + i) * (nz/2-1) + k;
                    V3_c_[index][0] = V3_c_[index][0] + tau/6 *(k1_V3[index][0] + 2*k2_V3[index][0] + 2*k3_V3[index][0] + k4_V3[index][0]);
                    V3_c_[index][1] = V3_c_[index][1] + tau/6 *(k1_V3[index][1] + 2*k2_V3[index][1] + 2*k3_V3[index][1] + k4_V3[index][1]);
                }
            }
        }
    }
    fftw_free(k1_V1);
    fftw_free(k2_V1);
    fftw_free(k3_V1);
    fftw_free(k4_V1);
    fftw_free(k1_V2);
    fftw_free(k2_V2);
    fftw_free(k3_V2);
    fftw_free(k4_V2);
    fftw_free(k1_V3);
    fftw_free(k2_V3);
    fftw_free(k3_V3);
    fftw_free(k4_V3);
}

void NavierStokes(fftw_complex* V1_start, fftw_complex* V2_start, fftw_complex* V3_start, fftw_complex* p_c,
                double* V1_r_,double* V2_r_,double* V3_r_,
                ptrdiff_t nx, ptrdiff_t ny ,ptrdiff_t nz, 
                ptrdiff_t alloc_local,ptrdiff_t local_nx,ptrdiff_t local_ny,ptrdiff_t local_x_start,ptrdiff_t local_y_start,
                double L_x, double L_y, double L_z, int nt, double tau)
{   
    
    fftw_complex *rot1_c = fftw_alloc_complex(alloc_local*(nz/2-1));
    fftw_complex *rot2_c = fftw_alloc_complex(alloc_local*(nz/2-1));
    fftw_complex *rot3_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    
    double *rot1_xy_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *rot2_xy_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *rot3_xy_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *rot1_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *rot2_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *rot3_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V1_xy_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V2_xy_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V3_xy_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *cross1_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *cross2_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *cross3_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *cross1_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *cross2_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *cross3_z_r = fftw_alloc_real(alloc_local*2*(nz/2-1));

    fftw_complex *cross1_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *cross2_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *cross3_c = fftw_alloc_complex(alloc_local*(nz/2-1));
    
    // f
    double *f1_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *f2_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *f3_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    double *f1_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *f2_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *f3_z_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    
    fftw_complex *F1_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *F2_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *F3_c = fftw_alloc_complex(alloc_local*(nz/2-1));

    fftw_complex *f1_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *f2_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *f3_c = fftw_alloc_complex(alloc_local*(nz/2-1));

    fftw_complex *div_c = fftw_alloc_complex(alloc_local*(nz/2+1));

       
    rungeKutta(V1_start,V2_start,V3_start,F1_c,F2_c,F3_c,f1_c,f2_c,f3_c,f1_z_r,f2_z_r,f3_z_r,f1_r,f2_r,f3_r,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c,cross2_c,cross3_c,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz,alloc_local,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau);
    
    fftw_free(rot1_c);
    fftw_free(rot2_c);
    fftw_free(rot3_c);
    fftw_free(rot1_xy_r);
    fftw_free(rot2_xy_r);
    fftw_free(rot3_xy_r);
    fftw_free(rot1_r);
    fftw_free(rot2_r);
    fftw_free(rot3_r);
    fftw_free(V1_xy_r);
    fftw_free(V2_xy_r);
    fftw_free(V3_xy_r);
    fftw_free(cross1_r);
    fftw_free(cross2_r);
    fftw_free(cross3_r);
    fftw_free(cross1_z_r);
    fftw_free(cross2_z_r);
    fftw_free(cross3_z_r);
    fftw_free(cross1_c);
    fftw_free(cross2_c);
    fftw_free(cross3_c);
    fftw_free(F1_c);
    fftw_free(F2_c);
    fftw_free(F3_c);
    fftw_free(f1_r);
    fftw_free(f2_r);
    fftw_free(f3_r);
    fftw_free(f1_z_r);
    fftw_free(f2_z_r);
    fftw_free(f3_z_r);
    fftw_free(div_c);
}

double *re_in2;

int main(int argc, char **argv) {
    const ptrdiff_t nx = 128, ny = 128, nz = 128;
    const double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;

    int max_threads = omp_get_max_threads();

    ptrdiff_t nt = 1;
    double T = 1;
    double tau = T / 20000; 
    int i, j, k, index;

    ptrdiff_t alloc_local,local_nx,local_x_start,local_ny,local_y_start;
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        cout << "Warning: The MPI does not have the thread support level" << endl;
    }

    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);

    fftw_mpi_init();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        cout << "num threads = " << max_threads << endl;

    // распределить данных и вычислить local size
    alloc_local = fftw_mpi_local_size_2d_transposed(
        nx, ny/2+1, MPI_COMM_WORLD,&local_nx, &local_x_start,&local_ny, &local_y_start
    );

    // V
    double *V1_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V2_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V3_r = fftw_alloc_real(alloc_local*2*(nz/2-1));
    
    double *V1_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V2_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V3_z_r = fftw_alloc_real(alloc_local*2*(nz/2-1));

    fftw_complex *V1_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *V2_c = fftw_alloc_complex(alloc_local*(nz/2+1));
    fftw_complex *V3_c = fftw_alloc_complex(alloc_local*(nz/2-1));

    double *V1_out_xy_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V2_out_xy_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V3_out_xy_r = fftw_alloc_real(alloc_local*2*(nz/2-1));

    double *V1_out_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V2_out_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *V3_out_r = fftw_alloc_real(alloc_local*2*(nz/2-1));

    // p
    double *p_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    double *p_z_r = fftw_alloc_real(alloc_local*2*(nz/2+1));
    fftw_complex *p_c = fftw_alloc_complex(alloc_local*(nz/2+1));

    if (rank == 0) {
        cout << "Planner Flag : FFTW_PATIENT"<< endl;
    }
    // time start 
    MPI_Barrier(MPI_COMM_WORLD); 
    double t_start_initplans = MPI_Wtime(); 
    // initialization plans fftw_mpi
    initialize_r2r_cos(local_nx, ny, nz, V1_r, V1_z_r);   
    initialize_r2r_sin(local_nx, ny, nz, V3_r, V3_z_r);   
    initialize_fwd_r2c_cos(nx,ny,nz,V1_z_r,V1_c);
    initialize_bwd_c2r_cos(nx,ny,nz,V1_c,V1_out_xy_r);
    initialize_fwd_r2c_sin(nx,ny,nz,V3_z_r,V3_c);
    initialize_bwd_c2r_sin(nx,ny,nz,V3_c,V3_out_xy_r);
    
    MPI_Barrier(MPI_COMM_WORLD); 
    double t_end_initplans = MPI_Wtime();
    
    if (rank == 0) {
        cout << scientific << setprecision(2);
        cout<< fixed << "Time initialize plans: " << t_end_initplans - t_start_initplans << " seconds" << endl;
    }

    // // initialize input data
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index = (i * (2*(ny/2+1)) + j) * (nz/2+1) + k;
                V1_r[index] = func_V1(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,0);
                V2_r[index] = func_V2(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,0);
                p_r[index] = func_p(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,0);
            }
            for(ptrdiff_t k = 1; k <= (nz/2-1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index = (i * (2*(ny/2+1)) + j) * (nz/2-1) + k-1;
                V3_r[index] = func_V3(global_i * L_x / nx, j * L_y / ny, k * L_z / nz,0);
            }
        }
    }
    
    //
    fftw_execute_r2r(plan_r2r_cos, V1_r, V1_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, V1_z_r, V1_c);

    fftw_execute_r2r(plan_r2r_cos, V2_r, V2_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, V2_z_r, V2_c);

    fftw_execute_r2r(plan_r2r_sin, V3_r, V3_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_sin, V3_z_r, V3_c);

    // p
    fftw_execute_r2r(plan_r2r_cos, p_r, p_z_r);
    fftw_mpi_execute_dft_r2c(plan_fwd_r2c_cos, p_z_r, p_c);

    // // normalization
    normalization(V1_c, nx,local_ny,(nz/2+1),nx*ny*nz);
    normalization(V2_c, nx,local_ny,(nz/2+1),nx*ny*nz);
    normalization(V3_c, nx,local_ny,(nz/2-1),nx*ny*nz);
    normalization(p_c, nx,local_ny,(nz/2+1),nx*ny*nz);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    NavierStokes(V1_c,V2_c,V3_c,p_c,V1_r,V2_r,V3_r,nx,ny,nz,alloc_local,local_nx,local_ny,local_x_start,local_y_start,L_x,L_y,L_z,nt,tau);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    if (rank == 0) { 
        cout<< fixed << "Execution Time NavierStokes : " << t_end - t_start << " seconds" << endl;
    }

    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_cos,V1_c,V1_out_xy_r);
    fftw_execute_r2r(plan_r2r_cos, V1_out_xy_r, V1_out_r);

    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_cos,V2_c,V2_out_xy_r);
    fftw_execute_r2r(plan_r2r_cos, V2_out_xy_r, V2_out_r);
    
    fftw_mpi_execute_dft_c2r(plan_bwd_c2r_sin,V3_c,V3_out_xy_r);
    fftw_execute_r2r(plan_r2r_sin, V3_out_xy_r, V3_out_r);

    

    double local_sumEnergy1 = calculateEnergy(V1_out_r, local_nx, ny, (nz/2+1), nx*ny*nz);
    double global_sumEnergy1;
    MPI_Allreduce(&local_sumEnergy1, &global_sumEnergy1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << setprecision(12);
        std::cout << "Energy1 = " << global_sumEnergy1 << std::endl;
    }
    double local_sumEnergy2 = calculateEnergy(V2_out_r, local_nx, ny, (nz/2+1), nx*ny*nz);
    double global_sumEnergy2;
    MPI_Allreduce(&local_sumEnergy2, &global_sumEnergy2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << setprecision(12);
        std::cout << "Energy2 = " << global_sumEnergy2 << std::endl;
    }
    double local_sumEnergy3 = calculateEnergy(V3_out_r, local_nx, ny, (nz/2-1), nx*ny*nz);
    double global_sumEnergy3;
    MPI_Allreduce(&local_sumEnergy3, &global_sumEnergy3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << setprecision(12);
        std::cout << "Energy3 = " << global_sumEnergy3 << std::endl;
    }

    
    // err
    double err1 = 0.0, err2 = 0.0, err3= 0.0, err1_, err2_, err3_;
    double local_err1 = 0.0, global_err1 = 0.0,local_err2 = 0.0, global_err2 = 0.0,local_err3 = 0.0, global_err3 = 0.0;
    #pragma omp parallel for collapse(2)
    for(ptrdiff_t i = 0; i < local_nx; ++i) {
        for(ptrdiff_t j = 0; j < ny; ++j) {
            for(ptrdiff_t k = 0; k < (nz/2+1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index1 = (i * (2*(ny/2+1)) + j) * (nz/2+1) + k;
                double diff1 = fabs(V1_out_r[index1] - func_V1(global_i*L_x/nx, j*L_y/ny, k*L_z/nz,nt*tau));
                double diff2 = fabs(V2_out_r[index1] - func_V2(global_i*L_x/nx, j*L_y/ny, k*L_z/nz,nt*tau));
                local_err1 += diff1 * diff1;
                local_err2 += diff2 * diff2;
            }

            for(ptrdiff_t k = 1; k <= (nz/2-1); ++k) {
                ptrdiff_t global_i = local_x_start + i; 
                ptrdiff_t index2 = (i * (2*(ny/2+1)) + j) * (nz/2-1) + k-1;
                double diff = fabs(V3_out_r[index2] - func_V3(global_i*L_x/nx, j*L_y/ny, k*L_z/nz,nt*tau));
                local_err3 += diff * diff;
            }
        }
    }
    
    MPI_Reduce(&local_err1, &global_err1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_err3, &global_err3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << scientific << setprecision(2);
        cout << "1 = " << sqrt(global_err1) << endl;
        cout << "2 = " << sqrt(global_err2) << endl;
        cout << "3 = " << sqrt(global_err3) << endl;
    }
    
    // для даления планов
    finalize_fft_plans();
    
    fftw_free(V1_r);
    fftw_free(V2_r);
    fftw_free(V3_r);
    fftw_free(V1_z_r);
    fftw_free(V2_z_r);
    fftw_free(V3_z_r);
    fftw_free(V1_c);
    fftw_free(V2_c);
    fftw_free(V3_c);
    fftw_free(V1_out_xy_r);
    fftw_free(V2_out_xy_r);
    fftw_free(V3_out_xy_r);
    fftw_free(V1_out_r);
    fftw_free(V2_out_r);
    fftw_free(V3_out_r);
    fftw_free(p_r);
    fftw_free(p_z_r);
    fftw_free(p_c);
    MPI_Finalize();
    return 0;
}

