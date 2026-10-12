// pcg3d.h - 3D thermal solvers (CPU reference and GPU).
//
// Both solve (A + S) x = b with the preconditioned conjugate gradient method and
// a Jacobi (diagonal) preconditioner, starting from the current x and iterating
// until ||r|| / ||b|| < tol. S is a per-plane diagonal shift: 0 for steady state,
// C / dt for a backward Euler time step.
//
// Precision: the solver works in double by default. In thin, highly conductive
// layers neighboring cells differ by only ~1e-5 K while T itself is ~50 K, which
// is a few float ulps. In float, PCG then reports convergence while the true
// residual is far from it (0.4 instead of 1e-6 on an 800 x 800 grid).
// Build with -DREAL_FLOAT to reproduce that.
#pragma once
#include <vector>
#include "stack3d.h"

#ifdef REAL_FLOAT
using Real = float;
#else
using Real = double;
#endif

// Copies a double vector into the working precision, with an explicit cast.
inline std::vector<Real> to_real(const std::vector<double>& v) {
    std::vector<Real> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = (Real)v[i];
    return out;
}

struct PcgStats {
    int iters = 0;
    double relres = 0;      // final ||r|| / ||b|| as tracked by the PCG recurrence
    bool converged = false;
};

// Both solver classes have the same interface, so the transient driver
// (transient3d.h) works with either one.
//
//   set_shift(s)    per-plane diagonal shift S (size nz); zeros for steady state
//   set_x_zero()    x = 0
//   set_rhs_steady()     b = P
//   set_rhs_transient()  b = P + S x   (x holds the previous temperature)
//   solve(tol, maxit)    PCG on (A + S) x = b, starting from the current x
//   get_x(T)        copy x to the host

class CpuPcg {
public:
    explicit CpuPcg(const Model3D& m);
    void set_shift(const std::vector<double>& shift);
    void set_x_zero();
    void set_rhs_steady();
    void set_rhs_transient();
    PcgStats solve(double tol, int maxit);
    void get_x(std::vector<Real>& T) const { T = x_; }

private:
    const Model3D& m_;
    std::vector<Real> gx_, gy_, gzd_, gzu_, shift_;
    std::vector<Real> P_, x_, b_, r_, z_, p_, Ap_, invd_;
};

class GpuPcg {
public:
    explicit GpuPcg(const Model3D& m);
    ~GpuPcg();
    GpuPcg(const GpuPcg&) = delete;
    GpuPcg& operator=(const GpuPcg&) = delete;
    void set_shift(const std::vector<double>& shift);
    void set_x_zero();
    void set_rhs_steady();
    void set_rhs_transient();
    PcgStats solve(double tol, int maxit);
    void get_x(std::vector<Real>& T) const;

private:
    const Model3D& m_;
    size_t n_;
    Real *P_ = nullptr, *x_ = nullptr, *b_ = nullptr, *r_ = nullptr, *z_ = nullptr,
         *p_ = nullptr, *Ap_ = nullptr, *invd_ = nullptr;
    double* dots_ = nullptr;
};

// Steady-state solve from T = 0, with timing (GPU time includes allocation and
// transfers).
struct SolveResult {
    std::vector<Real> T;   // temperature rise above ambient, per cell (K)
    int iters = 0;
    double relres = 0;
    double ms = 0;
    bool converged = false;
};
SolveResult cpu_pcg(const Model3D& m, double tol, int maxit);
SolveResult gpu_pcg(const Model3D& m, double tol, int maxit);

// ||P - A T|| / ||P|| for the steady problem, recomputed from scratch in double.
// Unlike the tracked residual, this can't drift, so it shows whether a solve
// really converged.
double true_relres(const Model3D& m, const std::vector<Real>& T);
