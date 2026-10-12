// pcg3d.h - steady-state 3D solvers (CPU reference and GPU).
// Both use the preconditioned conjugate gradient method with a Jacobi
// (diagonal) preconditioner, iterating until ||r|| / ||P|| < tol.
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

struct SolveResult {
    std::vector<Real> T;   // temperature rise above ambient, per cell (K)
    int iters = 0;
    double relres = 0;     // final ||r|| / ||P|| as tracked by the PCG recurrence
    double ms = 0;         // wall time of the solve, including transfers on the GPU
    bool converged = false;
};

SolveResult cpu_pcg(const Model3D& m, double tol, int maxit);
SolveResult gpu_pcg(const Model3D& m, double tol, int maxit);

// ||P - A T|| / ||P|| recomputed from scratch in double. Unlike the tracked
// residual, this can't drift, so it shows whether a solve really converged.
double true_relres(const Model3D& m, const std::vector<Real>& T);
