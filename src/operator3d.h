// operator3d.h - the 7-point finite-volume operator, shared by CPU and GPU.
// The same function runs on both, so they compute exactly the same thing.
#pragma once
#include <cstddef>

#if defined(__CUDACC__) || defined(__CUDA__)
#define HD __host__ __device__
#else
#define HD
#endif

// Diagonal of A for cell (i, j, l): total conductance out of the cell.
// Neighbors outside the side walls don't exist (adiabatic), so they add nothing.
HD inline float diag_cell(int i, int j, int nx, int ny, float gx, float gy,
                          float gzd, float gzu) {
    float d = 0.0f;
    if (i > 0) d += gx;
    if (i < nx - 1) d += gx;
    if (j > 0) d += gy;
    if (j < ny - 1) d += gy;
    return d + gzd + gzu;  // the z terms include the links to ambient
}

// (A x) for cell (i, j, l) = diag * x_c - sum over neighbors of G_n * x_n.
// gx, gy, gzd, gzu are the conductances of plane l.
HD inline float apply_cell(const float* x, int i, int j, int l, int nx, int ny,
                           int nz, float gx, float gy, float gzd, float gzu) {
    const size_t sxy = (size_t)nx * ny;
    const size_t id = (size_t)l * sxy + (size_t)j * nx + i;
    float s = 0.0f;
    if (i > 0) s += gx * x[id - 1];
    if (i < nx - 1) s += gx * x[id + 1];
    if (j > 0) s += gy * x[id - nx];
    if (j < ny - 1) s += gy * x[id + nx];
    if (l > 0) s += gzd * x[id - sxy];        // at l = 0, gzd links to ambient (T = 0)
    if (l < nz - 1) s += gzu * x[id + sxy];   // at the top, gzu links to ambient
    return diag_cell(i, j, nx, ny, gx, gy, gzd, gzu) * x[id] - s;
}
