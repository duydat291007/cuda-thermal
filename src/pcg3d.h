// pcg3d.h - steady-state 3D solvers (CPU reference and GPU).
// Both use the preconditioned conjugate gradient method with a Jacobi
// (diagonal) preconditioner, iterating until ||r|| / ||P|| < tol.
#pragma once
#include <vector>
#include "stack3d.h"

struct SolveResult {
    std::vector<float> T;  // temperature rise above ambient, per cell (K)
    int iters = 0;
    double relres = 0;     // final ||r|| / ||P||
    double ms = 0;         // wall time of the solve, including transfers on the GPU
    bool converged = false;
};

SolveResult cpu_pcg(const Model3D& m, double tol, int maxit);
SolveResult gpu_pcg(const Model3D& m, double tol, int maxit);
