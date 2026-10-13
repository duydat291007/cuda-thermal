// pcg3d_cpu.cpp - CPU reference: preconditioned conjugate gradient.
// Same algorithm and operator as the GPU version, written as plain loops.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "mg3d.h"
#include "operator3d.h"
#include "pcg3d.h"

static CellOp<float> model_op(const Model3D& m) {
    if (!m.finalized) {
        fprintf(stderr, "Model3D used before finalize()\n");
        exit(1);
    }
    return {m.nx, m.ny, m.nz, m.gE.data(), m.gN.data(), m.gU.data(), m.gB.data(), m.cap.data()};
}

CpuPcg::CpuPcg(const Model3D& m, Precond pc)
    : m_(m), pc_(pc), op_(model_op(m)), P_(to_real(m.P)) {
    size_t n = m.ncells();
    for (auto* v : {&x_, &b_, &r_, &z_, &p_, &Ap_, &invd_}) v->assign(n, 0);
    if (pc_ == Precond::Multigrid)
        for (MgLevelHost& h : mg_build_levels(m)) {
            Level L;
            L.nx = h.nx;
            L.ny = h.ny;
            L.nz = h.nz;
            L.gE = std::move(h.gE);
            L.gN = std::move(h.gN);
            L.gU = std::move(h.gU);
            L.gB = std::move(h.gB);
            L.cap = std::move(h.cap);
            for (auto* v : {&L.x, &L.b, &L.fcp, &L.finv}) v->assign((size_t)h.nx * h.ny * h.nz, 0);
            levels_.push_back(std::move(L));
        }
    set_dt(0.0);
}

void CpuPcg::set_dt(double dt) {
    inv_dt_ = dt > 0 ? (Real)(1.0 / dt) : Real(0);
    for (int l = 0; l < m_.nz; ++l)
        for (int j = 0; j < m_.ny; ++j)
            for (int i = 0; i < m_.nx; ++i) {
                size_t id = m_.idx(i, j, l);
                invd_[id] = Real(1) / op_diag<float, Real>(op_, id, i, j, l, inv_dt_);
            }
    // Multigrid column factors for every level.
    const MgReal mg_inv_dt = (MgReal)inv_dt_;
    for (Level& L : levels_) {
        const CellOp<float> o = L.op();
        for (int j = 0; j < L.ny; ++j)
            for (int i = 0; i < L.nx; ++i)
                mg_factor_column(o, mg_inv_dt, L.fcp.data(), L.finv.data(), i, j);
    }
}

void CpuPcg::set_x_zero() { std::fill(x_.begin(), x_.end(), Real(0)); }

void CpuPcg::set_rhs_steady() { b_ = P_; }

void CpuPcg::set_rhs_transient() {
    for (size_t c = 0; c < b_.size(); ++c) b_[c] = P_[c] + (Real)m_.cap[c] * inv_dt_ * x_[c];
}

// One multigrid V-cycle on level k: approximately solves (A + C/dt) x = b
// there, starting from x = 0. See mg3d.h for the design.
void CpuPcg::vcycle(size_t k) {
    Level& L = levels_[k];
    std::fill(L.x.begin(), L.x.end(), MgReal(0));
    const MgReal inv_dt = (MgReal)inv_dt_;
    const MgGrid<float, MgReal> g{L.op(), L.fcp.data(), L.finv.data()};
    auto sweep = [&](int color) {  // red-black: columns with (i + j) % 2 == color
        for (int j = 0; j < L.ny; ++j)
            for (int i = 0; i < L.nx; ++i)
                if (((i + j) & 1) == color) mg_line_solve(g, L.b.data(), L.x.data(), i, j);
    };

    if (k + 1 == levels_.size()) {  // a single column: one line solve is exact
        mg_line_solve(g, L.b.data(), L.x.data(), 0, 0);
        return;
    }
    sweep(0);
    sweep(1);

    // Restrict the residual: each coarse cell gets the sum over its children.
    Level& C = levels_[k + 1];
    const size_t fxy = (size_t)L.nx * L.ny, cxy = (size_t)C.nx * C.ny;
    for (int l = 0; l < L.nz; ++l)
        for (int J = 0; J < C.ny; ++J)
            for (int I = 0; I < C.nx; ++I) {
                MgReal sum = 0;
                for (int j = 2 * J; j < std::min(2 * J + 2, L.ny); ++j)
                    for (int i = 2 * I; i < std::min(2 * I + 2, L.nx); ++i) {
                        size_t id = (size_t)l * fxy + (size_t)j * L.nx + i;
                        sum += L.b[id] - op_apply(g.op, L.x.data(), i, j, l, inv_dt);
                    }
                C.b[(size_t)l * cxy + (size_t)J * C.nx + I] = sum;
            }

    vcycle(k + 1);

    // Prolongate: each fine cell adds its coarse parent's correction.
    for (int l = 0; l < L.nz; ++l)
        for (int j = 0; j < L.ny; ++j)
            for (int i = 0; i < L.nx; ++i)
                L.x[(size_t)l * fxy + (size_t)j * L.nx + i] +=
                    C.x[(size_t)l * cxy + (size_t)(j / 2) * C.nx + i / 2];

    sweep(1);  // reverse order after the correction keeps the V-cycle symmetric
    sweep(0);
}

void CpuPcg::precondition() {
    if (pc_ == Precond::Jacobi) {
        for (size_t c = 0; c < r_.size(); ++c) z_[c] = r_[c] * invd_[c];
    } else {  // the V-cycle runs in MgReal precision
        Level& L0 = levels_[0];
        for (size_t c = 0; c < r_.size(); ++c) L0.b[c] = (MgReal)r_[c];
        vcycle(0);
        for (size_t c = 0; c < z_.size(); ++c) z_[c] = (Real)L0.x[c];
    }
}

PcgStats CpuPcg::solve(double tol, int maxit) {
    const int nx = m_.nx, ny = m_.ny, nz = m_.nz;
    const size_t n = m_.ncells();
    auto dot = [&](const std::vector<Real>& a, const std::vector<Real>& b) {
        double s = 0;
        for (size_t c = 0; c < n; ++c) s += (double)a[c] * b[c];
        return s;
    };

    // r = b - (A + C/dt) x
    for (int l = 0; l < nz; ++l)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                size_t c = m_.idx(i, j, l);
                r_[c] = b_[c] - op_apply(op_, x_.data(), i, j, l, inv_dt_);
            }
    const double bb = dot(b_, b_);
    double rr = dot(r_, r_);

    PcgStats st;
    if (bb == 0) { set_x_zero(); st.converged = true; return st; }
    st.relres = std::sqrt(rr / bb);
    if (st.relres < tol) { st.converged = true; return st; }  // already converged

    precondition();  // z = M^-1 r
    p_ = z_;
    double rz = dot(r_, z_);

    for (int it = 1; it <= maxit; ++it) {
        // Ap = (A + C/dt) p, and p . Ap
        double pAp = 0;
        for (int l = 0; l < nz; ++l)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = m_.idx(i, j, l);
                    Ap_[c] = op_apply(op_, p_.data(), i, j, l, inv_dt_);
                    pAp += (double)p_[c] * Ap_[c];
                }
        Real alpha = (Real)(rz / pAp);

        // x += alpha p, r -= alpha Ap
        rr = 0;
        for (size_t c = 0; c < n; ++c) {
            x_[c] += alpha * p_[c];
            r_[c] -= alpha * Ap_[c];
            rr += (double)r_[c] * r_[c];
        }

        st.iters = it;
        st.relres = std::sqrt(rr / bb);
        if (st.relres < tol) { st.converged = true; break; }

        precondition();
        double rz_new = dot(r_, z_);
        Real beta = (Real)(rz_new / rz);
        rz = rz_new;
        for (size_t c = 0; c < n; ++c) p_[c] = z_[c] + beta * p_[c];
    }
    return st;
}

SolveResult cpu_pcg(const Model3D& m, double tol, int maxit, Precond pc) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CpuPcg s(m, pc);
    s.set_x_zero();
    s.set_rhs_steady();
    PcgStats st = s.solve(tol, maxit);
    SolveResult res;
    s.get_x(res.T);
    res.iters = st.iters;
    res.relres = st.relres;
    res.converged = st.converged;
    auto t1 = std::chrono::high_resolution_clock::now();
    res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

double true_relres(const Model3D& m, const std::vector<Real>& T) {
    const CellOp<float> o = model_op(m);
    std::vector<double> x(T.begin(), T.end());
    double rr = 0, bb = 0;
    for (int l = 0; l < m.nz; ++l)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                size_t c = m.idx(i, j, l);
                double r = m.P[c] - op_apply(o, x.data(), i, j, l, 0.0);
                rr += r * r;
                bb += m.P[c] * m.P[c];
            }
    return std::sqrt(rr / bb);
}
