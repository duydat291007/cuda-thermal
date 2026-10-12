// mg3d.h - geometric multigrid for the layered stack, shared by CPU and GPU.
//
// Used as the preconditioner inside PCG: one V-cycle per PCG iteration.
//
// Why this design:
//  * In thin silicon the vertical coupling is ~100x stronger than the lateral
//    coupling. Point smoothers can't handle that, so the smoother solves each
//    vertical column (all z-planes at one (i, j)) exactly: a tridiagonal solve.
//    Columns are updated in red-black order, like a checkerboard.
//  * Because the columns are solved exactly, coarsening is only needed in x and
//    y ("semi-coarsening"): every coarse level keeps all z-planes, and merges
//    2 x 2 columns into one. Odd sizes are fine: the last coarse cell then
//    covers a single fine cell, so cells on coarse levels can differ in size.
//  * The stack is only weakly tied to ambient, so the "whole stack warms up"
//    error barely decays under smoothing. Coarsening continues down to a single
//    column, where one tridiagonal solve is exact.
//  * Coarse operators are rediscretized from the geometry (cell sizes dx_i,
//    dy_j per level), so they keep the correct physical conductances.
//  * Restriction sums the residual of the (up to) 4 children; prolongation
//    copies each coarse value to its children. The two are transposes of each
//    other, and the smoothing is symmetric (red, black before; black, red
//    after), so the V-cycle is a symmetric preconditioner, as CG requires.
//
// Performance: everything that depends only on geometry and the time step is
// precomputed per level (mg_factor_column), so the smoother does no divisions.
// FP64 division is very slow on GeForce GPUs; computing it on the fly made the
// smoother ~10x slower than its memory traffic allows.
#pragma once
#include <algorithm>
#include <vector>
#include "operator3d.h"
#include "stack3d.h"

// Per-plane coefficients, per unit area of a cell, so they apply on every level.
template <class R>
struct MgPlanes {
    const R* kdz;      // k * dz: lateral conductance = kdz * (face width) / (center distance)
    const R* gzu_a;    // conductance per area to the plane above (top plane: to ambient)
    const R* gzd_a;    // conductance per area to the plane below (bottom plane: to ambient)
    const R* shift_a;  // diagonal shift per area (C / dt per area for a time step)
    int nz;
};

// One level: cell widths dx[i] (size nx), dy[j] (size ny); inverse distances
// between neighboring cell centers icx[i] = 1 / ((dx[i] + dx[i+1]) / 2) (size
// nx - 1) and icy[j]; and the precomputed column factors (see mg_factor_column).
template <class R>
struct MgGrid {
    int nx, ny;
    const R *dx, *dy, *icx, *icy;
    const R *fcp, *finv;  // per cell: Thomas upper factor, 1 / pivot
};

// Lateral conductances: between columns i and i+1 (gx), and j and j+1 (gy).
template <class R>
HD inline R mg_gx(const MgGrid<R>& g, R kdz, int i, int j) {
    return kdz * g.dy[j] * g.icx[i];
}
template <class R>
HD inline R mg_gy(const MgGrid<R>& g, R kdz, int i, int j) {
    return kdz * g.dx[i] * g.icy[j];
}

// Sum of the lateral conductances of column (i, j) in a plane with this kdz.
template <class R>
HD inline R mg_lateral_diag(const MgGrid<R>& g, R kdz, int i, int j) {
    R d = 0;
    if (i > 0) d += mg_gx(g, kdz, i - 1, j);
    if (i < g.nx - 1) d += mg_gx(g, kdz, i, j);
    if (j > 0) d += mg_gy(g, kdz, i, j - 1);
    if (j < g.ny - 1) d += mg_gy(g, kdz, i, j);
    return d;
}

// Lateral part of row (i, j, l): adds the lateral conductances to d and the
// neighbor terms G_n * x_n to s.
template <class R>
HD inline void mg_lateral(const MgGrid<R>& g, R kdz, const R* x, size_t id, int i, int j,
                          R& d, R& s) {
    R c;
    if (i > 0)        { c = mg_gx(g, kdz, i - 1, j); d += c; s += c * x[id - 1]; }
    if (i < g.nx - 1) { c = mg_gx(g, kdz, i, j);     d += c; s += c * x[id + 1]; }
    if (j > 0)        { c = mg_gy(g, kdz, i, j - 1); d += c; s += c * x[id - g.nx]; }
    if (j < g.ny - 1) { c = mg_gy(g, kdz, i, j);     d += c; s += c * x[id + g.nx]; }
}

// ((A + S) x) at (i, j, l) on one level.
template <class R>
HD inline R mg_apply(const MgGrid<R>& g, const MgPlanes<R>& p, const R* x, int i, int j,
                     int l) {
    const size_t sxy = (size_t)g.nx * g.ny;
    const size_t id = (size_t)l * sxy + (size_t)j * g.nx + i;
    const R area = g.dx[i] * g.dy[j];
    R d = (p.gzd_a[l] + p.gzu_a[l] + p.shift_a[l]) * area, s = 0;
    mg_lateral(g, p.kdz[l], x, id, i, j, d, s);
    if (l > 0) s += p.gzd_a[l] * area * x[id - sxy];
    if (l < p.nz - 1) s += p.gzu_a[l] * area * x[id + sxy];
    return d * x[id] - s;
}

// Factors the tridiagonal matrix of column (i, j) once (Thomas algorithm):
// fcp = modified upper diagonal, finv = 1 / pivot. They depend only on geometry
// and the shift, so they are recomputed only when the time step changes.
template <class R>
HD inline void mg_factor_column(const MgGrid<R>& g, const MgPlanes<R>& p, R* fcp, R* finv,
                                int i, int j) {
    const size_t sxy = (size_t)g.nx * g.ny;
    const size_t col = (size_t)j * g.nx + i;
    const R area = g.dx[i] * g.dy[j];
    R cp_prev = 0;
    for (int l = 0; l < p.nz; ++l) {
        const size_t id = (size_t)l * sxy + col;
        const R d = (p.gzd_a[l] + p.gzu_a[l] + p.shift_a[l]) * area +
                    mg_lateral_diag(g, p.kdz[l], i, j);
        const R lower = (l > 0) ? -p.gzd_a[l] * area : R(0);
        const R upper = (l < p.nz - 1) ? -p.gzu_a[l] * area : R(0);
        const R inv = R(1) / (d - lower * cp_prev);
        finv[id] = inv;
        cp_prev = upper * inv;
        fcp[id] = cp_prev;
    }
}

// Solves column (i, j) exactly for x, holding the neighboring columns fixed:
// a tridiagonal system in z, using the precomputed factors. The forward pass
// writes its intermediate values straight into x, then the back substitution
// finishes in place, so no scratch arrays and no divisions are needed.
template <class R>
HD inline void mg_line_solve(const MgGrid<R>& g, const MgPlanes<R>& p, const R* b, R* x,
                             int i, int j) {
    const size_t sxy = (size_t)g.nx * g.ny;
    const size_t col = (size_t)j * g.nx + i;
    const R area = g.dx[i] * g.dy[j];
    R prev = 0;
    for (int l = 0; l < p.nz; ++l) {
        const size_t id = (size_t)l * sxy + col;
        R d = 0, s = b[id];  // d is not needed: the pivots are precomputed
        mg_lateral(g, p.kdz[l], x, id, i, j, d, s);
        const R lower = (l > 0) ? -p.gzd_a[l] * area : R(0);
        prev = (s - lower * prev) * g.finv[id];
        x[id] = prev;
    }
    for (int l = p.nz - 2; l >= 0; --l) {
        const size_t id = (size_t)l * sxy + col;
        x[id] -= g.fcp[id] * x[id + sxy];
    }
}

// Host side: lateral cell sizes for every level, finest first, down to 1 x 1.
struct MgLevelGeom {
    int nx, ny;
    std::vector<double> dx, dy;
    std::vector<double> icx, icy;  // inverse center distances (at least 1 entry)
};

inline std::vector<double> inverse_center_distances(const std::vector<double>& w) {
    std::vector<double> ic(std::max<size_t>(w.size(), 2) - 1, 0.0);
    for (size_t i = 0; i + 1 < w.size(); ++i) ic[i] = 1.0 / (0.5 * (w[i] + w[i + 1]));
    return ic;
}

inline std::vector<MgLevelGeom> mg_hierarchy(const Model3D& m) {
    std::vector<MgLevelGeom> L(1);
    L[0].nx = m.nx;
    L[0].ny = m.ny;
    L[0].dx.assign(m.nx, m.dx);
    L[0].dy.assign(m.ny, m.dy);
    while (L.back().nx > 1 || L.back().ny > 1) {
        const MgLevelGeom& f = L.back();
        MgLevelGeom c;
        c.nx = (f.nx + 1) / 2;
        c.ny = (f.ny + 1) / 2;
        for (int I = 0; I < c.nx; ++I)
            c.dx.push_back(f.dx[2 * I] + (2 * I + 1 < f.nx ? f.dx[2 * I + 1] : 0.0));
        for (int J = 0; J < c.ny; ++J)
            c.dy.push_back(f.dy[2 * J] + (2 * J + 1 < f.ny ? f.dy[2 * J + 1] : 0.0));
        L.push_back(c);
    }
    for (MgLevelGeom& g : L) {
        g.icx = inverse_center_distances(g.dx);
        g.icy = inverse_center_distances(g.dy);
    }
    return L;
}

// Host side: per-plane coefficients per unit area, for a given diagonal shift
// (shift is per fine cell, as passed to set_shift).
struct MgPlaneCoefs {
    std::vector<double> kdz, gzu_a, gzd_a, shift_a;
};

inline MgPlaneCoefs mg_plane_coefs(const Model3D& m, const std::vector<double>& shift) {
    MgPlaneCoefs c;
    const double A = m.dx * m.dy;
    for (int l = 0; l < m.nz; ++l) {
        c.kdz.push_back(m.k[l] * m.dz[l]);
        c.gzu_a.push_back(m.gzu[l] / A);
        c.gzd_a.push_back(m.gzd[l] / A);
        c.shift_a.push_back(shift[l] / A);
    }
    return c;
}
