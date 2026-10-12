// pcg3d_cpu.cpp - CPU reference: preconditioned conjugate gradient.
// Same algorithm and operator as the GPU version, written as plain loops.
// Vectors are float like on the GPU; dot products accumulate in double.
#include <chrono>
#include <cmath>
#include "operator3d.h"
#include "pcg3d.h"

SolveResult cpu_pcg(const Model3D& m, double tol, int maxit) {
    auto t0 = std::chrono::high_resolution_clock::now();
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const size_t n = m.ncells();

    std::vector<float> x(n, 0.0f), r(m.P), z(n), p(n), Ap(n), invd(n);
    for (int l = 0; l < nz; ++l)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                invd[m.idx(i, j, l)] =
                    1.0f / diag_cell(i, j, nx, ny, m.gx[l], m.gy[l], m.gzd[l], m.gzu[l]);

    double rz = 0, bb = 0;
    for (size_t c = 0; c < n; ++c) {
        z[c] = r[c] * invd[c];
        p[c] = z[c];
        rz += (double)r[c] * z[c];
        bb += (double)r[c] * r[c];
    }

    SolveResult res;
    if (bb == 0) { res.T = x; res.converged = true; return res; }

    for (int it = 1; it <= maxit; ++it) {
        // Ap = A p, and p . Ap
        double pAp = 0;
        for (int l = 0; l < nz; ++l)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = m.idx(i, j, l);
                    Ap[c] = apply_cell(p.data(), i, j, l, nx, ny, nz, m.gx[l], m.gy[l],
                                       m.gzd[l], m.gzu[l]);
                    pAp += (double)p[c] * Ap[c];
                }
        float alpha = (float)(rz / pAp);

        // x += alpha p, r -= alpha Ap, z = M^-1 r
        double rz_new = 0, rr = 0;
        for (size_t c = 0; c < n; ++c) {
            x[c] += alpha * p[c];
            r[c] -= alpha * Ap[c];
            z[c] = r[c] * invd[c];
            rz_new += (double)r[c] * z[c];
            rr += (double)r[c] * r[c];
        }

        res.iters = it;
        res.relres = std::sqrt(rr / bb);
        if (res.relres < tol) { res.converged = true; break; }

        float beta = (float)(rz_new / rz);
        rz = rz_new;
        for (size_t c = 0; c < n; ++c) p[c] = z[c] + beta * p[c];
    }

    res.T = std::move(x);
    auto t1 = std::chrono::high_resolution_clock::now();
    res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}
