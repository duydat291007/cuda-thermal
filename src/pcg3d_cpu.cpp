// pcg3d_cpu.cpp - CPU reference: preconditioned conjugate gradient.
// Same algorithm and operator as the GPU version, written as plain loops.
#include <algorithm>
#include <chrono>
#include <cmath>
#include "operator3d.h"
#include "pcg3d.h"

CpuPcg::CpuPcg(const Model3D& m)
    : m_(m), gx_(to_real(m.gx)), gy_(to_real(m.gy)), gzd_(to_real(m.gzd)),
      gzu_(to_real(m.gzu)), shift_(m.nz, 0), P_(to_real(m.P)) {
    size_t n = m.ncells();
    for (auto* v : {&x_, &b_, &r_, &z_, &p_, &Ap_, &invd_}) v->assign(n, 0);
    set_shift(std::vector<double>(m.nz, 0.0));
}

void CpuPcg::set_shift(const std::vector<double>& shift) {
    shift_ = to_real(shift);
    for (int l = 0; l < m_.nz; ++l)
        for (int j = 0; j < m_.ny; ++j)
            for (int i = 0; i < m_.nx; ++i)
                invd_[m_.idx(i, j, l)] = Real(1) / diag_cell(i, j, m_.nx, m_.ny, gx_[l], gy_[l],
                                                             gzd_[l], gzu_[l], shift_[l]);
}

void CpuPcg::set_x_zero() { std::fill(x_.begin(), x_.end(), Real(0)); }

void CpuPcg::set_rhs_steady() { b_ = P_; }

void CpuPcg::set_rhs_transient() {
    for (int l = 0; l < m_.nz; ++l)
        for (size_t c = 0; c < m_.nxy(); ++c) {
            size_t id = (size_t)l * m_.nxy() + c;
            b_[id] = P_[id] + shift_[l] * x_[id];
        }
}

PcgStats CpuPcg::solve(double tol, int maxit) {
    const int nx = m_.nx, ny = m_.ny, nz = m_.nz;
    const size_t n = m_.ncells();
    auto apply = [&](const std::vector<Real>& v, int i, int j, int l) {
        return apply_cell(v.data(), i, j, l, nx, ny, nz, gx_[l], gy_[l], gzd_[l], gzu_[l],
                          shift_[l]);
    };

    // r = b - (A + S) x, z = M^-1 r, p = z
    double rz = 0, rr = 0, bb = 0;
    for (int l = 0; l < nz; ++l)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                size_t c = m_.idx(i, j, l);
                r_[c] = b_[c] - apply(x_, i, j, l);
                z_[c] = r_[c] * invd_[c];
                p_[c] = z_[c];
                rz += (double)r_[c] * z_[c];
                rr += (double)r_[c] * r_[c];
                bb += (double)b_[c] * b_[c];
            }

    PcgStats st;
    if (bb == 0) { set_x_zero(); st.converged = true; return st; }
    st.relres = std::sqrt(rr / bb);
    if (st.relres < tol) { st.converged = true; return st; }  // already converged

    for (int it = 1; it <= maxit; ++it) {
        // Ap = (A + S) p, and p . Ap
        double pAp = 0;
        for (int l = 0; l < nz; ++l)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = m_.idx(i, j, l);
                    Ap_[c] = apply(p_, i, j, l);
                    pAp += (double)p_[c] * Ap_[c];
                }
        Real alpha = (Real)(rz / pAp);

        // x += alpha p, r -= alpha Ap, z = M^-1 r
        double rz_new = 0;
        rr = 0;
        for (size_t c = 0; c < n; ++c) {
            x_[c] += alpha * p_[c];
            r_[c] -= alpha * Ap_[c];
            z_[c] = r_[c] * invd_[c];
            rz_new += (double)r_[c] * z_[c];
            rr += (double)r_[c] * r_[c];
        }

        st.iters = it;
        st.relres = std::sqrt(rr / bb);
        if (st.relres < tol) { st.converged = true; break; }

        Real beta = (Real)(rz_new / rz);
        rz = rz_new;
        for (size_t c = 0; c < n; ++c) p_[c] = z_[c] + beta * p_[c];
    }
    return st;
}

SolveResult cpu_pcg(const Model3D& m, double tol, int maxit) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CpuPcg s(m);
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
    std::vector<double> x(T.begin(), T.end());
    double rr = 0, bb = 0;
    for (int l = 0; l < m.nz; ++l)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                size_t c = m.idx(i, j, l);
                double r = m.P[c] - apply_cell(x.data(), i, j, l, m.nx, m.ny, m.nz,
                                               m.gx[l], m.gy[l], m.gzd[l], m.gzu[l], 0.0);
                rr += r * r;
                bb += m.P[c] * m.P[c];
            }
    return std::sqrt(rr / bb);
}
