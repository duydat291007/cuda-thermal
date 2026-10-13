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
//    covers a single fine cell.
//  * The stack is only weakly tied to ambient, so the "whole stack warms up"
//    error barely decays under smoothing. Coarsening continues down to a single
//    column, where one tridiagonal solve is exact.
//  * Coarse operators are built by aggregating the fine faces (mg_build_levels):
//    vertical faces, ambient links and heat capacities of the children add up
//    (they are in parallel); the fine lateral faces crossing a coarse face add
//    up and are scaled by (fine center distance / coarse center distance). For a
//    uniform material this is exactly the operator of the coarse grid; with TSVs
//    it is a good approximation, which is all a preconditioner needs.
//  * Restriction sums the residual of the (up to) 4 children; prolongation
//    copies each coarse value to its children. The two are transposes of each
//    other, and the smoothing is symmetric (red, black before; black, red
//    after), so the V-cycle is a symmetric preconditioner, as CG requires.
//
// Performance: each column's tridiagonal factors depend only on the operator
// and dt, so they are precomputed (mg_factor_column) and the smoother does no
// divisions; FP64 division is very slow on GeForce GPUs.
#pragma once
#include <algorithm>
#include <vector>
#include "operator3d.h"
#include "stack3d.h"

// One level: its operator (either memory layout, see operator3d.h) plus the
// precomputed column factors, stored in the same layout.
template <class Op, class R>
struct MgGrid {
    Op op;
    const R *fcp, *finv;  // per cell: Thomas upper factor, 1 / pivot
};

// Factors the tridiagonal matrix of column (i, j) (Thomas algorithm).
template <class Op, class R>
HD inline void mg_factor_column(const Op& o, R inv_dt, R* fcp, R* finv, int i, int j) {
    const size_t sxy = o.plane();
    const size_t col = o.col(i, j);
    R cp_prev = 0;
    for (int l = 0; l < o.nz; ++l) {
        const size_t id = (size_t)l * sxy + col;
        const R d = op_diag<Op, R>(o, i, j, l, inv_dt);
        const R lower = (l > 0) ? -(R)o.gU[id - sxy] : R(0);
        const R upper = (l < o.nz - 1) ? -(R)o.gU[id] : R(0);
        const R inv = R(1) / (d - lower * cp_prev);
        finv[id] = inv;
        cp_prev = upper * inv;
        fcp[id] = cp_prev;
    }
}

// Solves column (i, j) exactly for x, holding the neighboring columns fixed,
// using the precomputed factors. The forward pass writes its intermediate
// values straight into x, then the back substitution finishes in place, so no
// scratch arrays and no divisions are needed.
template <class Op, class R>
HD inline void mg_line_solve(const MgGrid<Op, R>& g, const R* b, R* x, int i, int j) {
    const Op& o = g.op;
    const size_t sxy = o.plane();
    const size_t col = o.col(i, j);
    R prev = 0;
    for (int l = 0; l < o.nz; ++l) {
        const size_t id = (size_t)l * sxy + col;
        const R s = b[id] + op_lat_sum(o, x, i, j, l);
        const R lower = (l > 0) ? -(R)o.gU[id - sxy] : R(0);
        prev = (s - lower * prev) * g.finv[id];
        x[id] = prev;
    }
    for (int l = o.nz - 2; l >= 0; --l) {
        const size_t id = (size_t)l * sxy + col;
        x[id] -= g.fcp[id] * x[id + sxy];
    }
}

// Host side: one level's operator arrays (float) and cell widths (for the
// lateral scaling when coarsening).
struct MgLevelHost {
    int nx, ny, nz;
    std::vector<double> dx, dy;
    std::vector<float> gE, gN, gU, gB, cap;
};

// All levels, finest (the model's own operator) first, down to 1 x 1.
inline std::vector<MgLevelHost> mg_build_levels(const Model3D& m) {
    std::vector<MgLevelHost> L(1);
    L[0].nx = m.nx;
    L[0].ny = m.ny;
    L[0].nz = m.nz;
    L[0].dx.assign(m.nx, m.dx);
    L[0].dy.assign(m.ny, m.dy);
    L[0].gE = m.gE;
    L[0].gN = m.gN;
    L[0].gU = m.gU;
    L[0].gB = m.gB;
    L[0].cap = m.cap;
    while (L.back().nx > 1 || L.back().ny > 1) {
        const MgLevelHost& f = L.back();
        MgLevelHost c;
        c.nx = (f.nx + 1) / 2;
        c.ny = (f.ny + 1) / 2;
        c.nz = f.nz;
        for (int I = 0; I < c.nx; ++I)
            c.dx.push_back(f.dx[2 * I] + (2 * I + 1 < f.nx ? f.dx[2 * I + 1] : 0.0));
        for (int J = 0; J < c.ny; ++J)
            c.dy.push_back(f.dy[2 * J] + (2 * J + 1 < f.ny ? f.dy[2 * J + 1] : 0.0));
        const size_t fxy = (size_t)f.nx * f.ny, cxy = (size_t)c.nx * c.ny;
        const size_t cn = cxy * c.nz;
        c.gE.assign(cn, 0.0f);
        c.gN.assign(cn, 0.0f);
        c.gU.assign(cn, 0.0f);
        c.cap.assign(cn, 0.0f);
        c.gB.assign(cxy, 0.0f);
        for (int l = 0; l < c.nz; ++l)
            for (int J = 0; J < c.ny; ++J)
                for (int I = 0; I < c.nx; ++I) {
                    const size_t cid = (size_t)l * cxy + (size_t)J * c.nx + I;
                    const int i0 = 2 * I, i1 = std::min(2 * I + 1, f.nx - 1);
                    const int j0 = 2 * J, j1 = std::min(2 * J + 1, f.ny - 1);
                    double gu = 0, cp = 0, gb = 0;
                    for (int j = j0; j <= j1; ++j)
                        for (int i = i0; i <= i1; ++i) {
                            const size_t fid = (size_t)l * fxy + (size_t)j * f.nx + i;
                            gu += f.gU[fid];
                            cp += f.cap[fid];
                            if (l == 0) gb += f.gB[(size_t)j * f.nx + i];
                        }
                    c.gU[cid] = (float)gu;
                    c.cap[cid] = (float)cp;
                    if (l == 0) c.gB[(size_t)J * c.nx + I] = (float)gb;
                    if (I < c.nx - 1) {  // fine faces between columns 2I+1 and 2I+2
                        double s = 0;
                        for (int j = j0; j <= j1; ++j) s += f.gE[(size_t)l * fxy + (size_t)j * f.nx + 2 * I + 1];
                        double df = 0.5 * (f.dx[2 * I + 1] + f.dx[2 * I + 2]);
                        double dc = 0.5 * (c.dx[I] + c.dx[I + 1]);
                        c.gE[cid] = (float)(s * df / dc);
                    }
                    if (J < c.ny - 1) {  // fine faces between rows 2J+1 and 2J+2
                        double s = 0;
                        for (int i = i0; i <= i1; ++i) s += f.gN[(size_t)l * fxy + (size_t)(2 * J + 1) * f.nx + i];
                        double df = 0.5 * (f.dy[2 * J + 1] + f.dy[2 * J + 2]);
                        double dc = 0.5 * (c.dy[J] + c.dy[J + 1]);
                        c.gN[cid] = (float)(s * df / dc);
                    }
                }
        L.push_back(std::move(c));
    }
    return L;
}
