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
// so a cell's -x face is gE of its -x neighbor, its -y face gN of its -y
// neighbor, its -z face gU of the cell below.
//
// Memory layout (where cell (i, j, l) lives in every array), a template flag:
//  * natural (SPLIT = false): c = (l * ny + j) * nx + i. Used by PCG and the CPU.
//  * color-split (SPLIT = true): each row is stored as its red columns
//    ((i + j) even) followed by its black ones, c = l * plane + (2 j + color) * hx
//    + i / 2 with hx = ceil(nx / 2). Used by the GPU multigrid levels: a red-black
//    sweep updates one color, and in this layout the cells it reads and writes
//    are contiguous, so every 32-byte memory sector it fetches is fully used.
//    In the natural layout half of each sector belongs to the other color, and
//    the (bandwidth-bound) smoother moved ~1.5x the bytes it needed.
// The arithmetic is written once below and is identical in both layouts.
#pragma once
#include <cstddef>

#if defined(__CUDACC__) || defined(__CUDA__)
#define HD __host__ __device__
#else
#define HD
#endif

template <class G, bool SPLIT = false>
struct CellOp {
    using Elem = G;
    int nx, ny, nz;
    const G *gE, *gN, *gU, *gB, *cap;

    HD int hx() const { return (nx + 1) / 2; }
    // Cells per z-plane in memory (with padding for odd nx in the split layout).
    HD size_t plane() const { return SPLIT ? (size_t)ny * 2 * hx() : (size_t)nx * ny; }
    // Position of column (i, j) within a plane; also the index into gB.
    HD size_t col(int i, int j) const {
        return SPLIT ? ((size_t)j * 2 + ((i + j) & 1)) * hx() + (i >> 1) : (size_t)j * nx + i;
    }
    HD size_t idx(int i, int j, int l) const { return (size_t)l * plane() + col(i, j); }
};

// Sum of the lateral face conductances of cell (i, j, l).
template <class Op, class R>
HD inline R op_lat_diag(const Op& o, int i, int j, int l) {
    const size_t p = (size_t)l * o.plane();
    R d = 0;
    if (i > 0) d += (R)o.gE[p + o.col(i - 1, j)];
    if (i < o.nx - 1) d += (R)o.gE[p + o.col(i, j)];
    if (j > 0) d += (R)o.gN[p + o.col(i, j - 1)];
    if (j < o.ny - 1) d += (R)o.gN[p + o.col(i, j)];
    return d;
}

// Lateral neighbor terms: sum of G * x over the 4 lateral neighbors.
template <class Op, class R>
HD inline R op_lat_sum(const Op& o, const R* x, int i, int j, int l) {
    const size_t p = (size_t)l * o.plane();
    R s = 0;
    if (i > 0) {
        const size_t w = p + o.col(i - 1, j);
        s += (R)o.gE[w] * x[w];
    }
    if (i < o.nx - 1) s += (R)o.gE[p + o.col(i, j)] * x[p + o.col(i + 1, j)];
    if (j > 0) {
        const size_t sth = p + o.col(i, j - 1);
        s += (R)o.gN[sth] * x[sth];
    }
    if (j < o.ny - 1) s += (R)o.gN[p + o.col(i, j)] * x[p + o.col(i, j + 1)];
    return s;
}

// Full diagonal of (A + C/dt) at cell (i, j, l).
template <class Op, class R>
HD inline R op_diag(const Op& o, int i, int j, int l, R inv_dt) {
    const size_t id = o.idx(i, j, l);
    R d = op_lat_diag<Op, R>(o, i, j, l);
    d += (R)o.gU[id];                                                    // up, or ambient on top
    d += (l > 0) ? (R)o.gU[id - o.plane()] : (R)o.gB[o.col(i, j)];       // down, or ambient at bottom
    return d + (R)o.cap[id] * inv_dt;
}

// Off-diagonal part: sum of G * x over all 6 neighbors of cell (i, j, l).
template <class Op, class R>
HD inline R op_offdiag_sum(const Op& o, const R* x, int i, int j, int l) {
    const size_t id = o.idx(i, j, l), sxy = o.plane();
    R s = op_lat_sum(o, x, i, j, l);
    if (l > 0) s += (R)o.gU[id - sxy] * x[id - sxy];
    if (l < o.nz - 1) s += (R)o.gU[id] * x[id + sxy];
    return s;
}

// Same as op_apply, with the diagonal d = op_diag(...) precomputed. On GPUs with
// slow FP64 this halves the double-precision work: the diagonal costs 7 float ->
// double conversions and 7 additions per cell, but only changes with dt.
template <class Op, class R>
HD inline R op_apply_diag(const Op& o, const R* x, R d, int i, int j, int l) {
    return d * x[o.idx(i, j, l)] - op_offdiag_sum(o, x, i, j, l);
}

// ((A + C/dt) x) at cell (i, j, l).
template <class Op, class R>
HD inline R op_apply(const Op& o, const R* x, int i, int j, int l, R inv_dt) {
    return op_diag<Op, R>(o, i, j, l, inv_dt) * x[o.idx(i, j, l)] - op_offdiag_sum(o, x, i, j, l);
}
