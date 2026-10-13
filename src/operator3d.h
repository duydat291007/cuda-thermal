// operator3d.h - the 7-point finite-volume operator, shared by CPU and GPU and
// by every multigrid level.
//
// The operator is (A + C/dt): A holds the face conductances, C the heat
// capacity of each cell. Steady state uses inv_dt = 0. A backward Euler time
// step uses inv_dt = 1 / dt.
//
// Conductances are stored per cell face (type G, float), values in type R:
//   gE[c]  face to the +x neighbor     gN[c]  face to the +y neighbor
//   gU[c]  face to the +z neighbor, or to ambient on the top plane
//   gB[col] bottom plane to ambient    cap[c] heat capacity
// so a cell's -x face is gE[c - 1], its -y face gN[c - nx], its -z face gU[c - nxy].
#pragma once
#include <cstddef>

#if defined(__CUDACC__) || defined(__CUDA__)
#define HD __host__ __device__
#else
#define HD
#endif

template <class G>
struct CellOp {
    int nx, ny, nz;
    const G *gE, *gN, *gU, *gB, *cap;
};

// Sum of the lateral face conductances of cell id = (i, j, l).
template <class G, class R>
HD inline R op_lat_diag(const CellOp<G>& o, size_t id, int i, int j) {
    R d = 0;
    if (i > 0) d += (R)o.gE[id - 1];
    if (i < o.nx - 1) d += (R)o.gE[id];
    if (j > 0) d += (R)o.gN[id - o.nx];
    if (j < o.ny - 1) d += (R)o.gN[id];
    return d;
}

// Lateral neighbor terms: sum of G * x over the 4 lateral neighbors.
template <class G, class R>
HD inline R op_lat_sum(const CellOp<G>& o, const R* x, size_t id, int i, int j) {
    R s = 0;
    if (i > 0) s += (R)o.gE[id - 1] * x[id - 1];
    if (i < o.nx - 1) s += (R)o.gE[id] * x[id + 1];
    if (j > 0) s += (R)o.gN[id - o.nx] * x[id - o.nx];
    if (j < o.ny - 1) s += (R)o.gN[id] * x[id + o.nx];
    return s;
}

// Full diagonal of (A + C/dt) at cell id = (i, j, l).
template <class G, class R>
HD inline R op_diag(const CellOp<G>& o, size_t id, int i, int j, int l, R inv_dt) {
    const size_t sxy = (size_t)o.nx * o.ny;
    R d = op_lat_diag<G, R>(o, id, i, j);
    d += (R)o.gU[id];                                          // up, or ambient on top
    d += (l > 0) ? (R)o.gU[id - sxy] : (R)o.gB[id];            // down, or ambient at bottom
    return d + (R)o.cap[id] * inv_dt;
}

// ((A + C/dt) x) at cell (i, j, l).
template <class G, class R>
HD inline R op_apply(const CellOp<G>& o, const R* x, int i, int j, int l, R inv_dt) {
    const size_t sxy = (size_t)o.nx * o.ny;
    const size_t id = (size_t)l * sxy + (size_t)j * o.nx + i;
    R s = op_lat_sum(o, x, id, i, j);
    if (l > 0) s += (R)o.gU[id - sxy] * x[id - sxy];
    if (l < o.nz - 1) s += (R)o.gU[id] * x[id + sxy];
    return op_diag<G, R>(o, id, i, j, l, inv_dt) * x[id] - s;
}
