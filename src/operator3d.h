// operator3d.h - the 7-point finite-volume operator, shared by CPU and GPU.
// The same function runs on both, so they compute exactly the same thing.
//
// The operator is (A + S), where A holds the conductances and S is an optional
// diagonal "shift" per plane. Steady state uses S = 0. A backward Euler time
// step uses S = C / dt, the cell heat capacity divided by the time step.
#pragma once
#include <cstddef>

#if defined(__CUDACC__) || defined(__CUDA__)
#define HD __host__ __device__
#else
#define HD
#endif

// Diagonal of (A + S) for cell (i, j, l): total conductance out of the cell plus
// the shift. Neighbors outside the side walls don't exist (adiabatic).
template <class R>
HD inline R diag_cell(int i, int j, int nx, int ny, R gx, R gy, R gzd, R gzu, R shift) {
    R d = 0;
    if (i > 0) d += gx;
    if (i < nx - 1) d += gx;
    if (j > 0) d += gy;
    if (j < ny - 1) d += gy;
    return d + gzd + gzu + shift;  // the z terms include the links to ambient
}

// ((A + S) x) for cell (i, j, l) = diag * x_c - sum over neighbors of G_n * x_n.
// gx, gy, gzd, gzu, shift are the values of plane l.
template <class R>
HD inline R apply_cell(const R* x, int i, int j, int l, int nx, int ny, int nz,
                       R gx, R gy, R gzd, R gzu, R shift) {
    const size_t sxy = (size_t)nx * ny;
    const size_t id = (size_t)l * sxy + (size_t)j * nx + i;
    R s = 0;
    if (i > 0) s += gx * x[id - 1];
    if (i < nx - 1) s += gx * x[id + 1];
    if (j > 0) s += gy * x[id - nx];
    if (j < ny - 1) s += gy * x[id + nx];
    if (l > 0) s += gzd * x[id - sxy];        // at l = 0, gzd links to ambient (T = 0)
    if (l < nz - 1) s += gzu * x[id + sxy];   // at the top, gzu links to ambient
    return diag_cell(i, j, nx, ny, gx, gy, gzd, gzu, shift) * x[id] - s;
}
