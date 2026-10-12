// problem.h - 2D steady-state heat conduction test problem.
//
// Solves  -k * laplacian(T) = q  on the unit square, with T = 0 on the boundary
// (temperatures are relative to ambient). Grid has nx * ny nodes including the
// boundary, spacing h = 1 / (nx - 1).
//
// The heat source is a manufactured solution:
//     T_exact(x, y) = sin(pi x) sin(pi y)
//     q(x, y)       = 2 pi^2 k sin(pi x) sin(pi y)
// so we can measure the true error of the solver, not just CPU-vs-GPU agreement.
#pragma once
#include <cmath>
#include <vector>

struct Problem {
    int nx, ny;
    float h;        // grid spacing
    float k;        // thermal conductivity
    std::vector<float> q;        // heat source per node
    std::vector<float> T_exact;  // analytic solution per node
};

inline Problem make_problem(int nx, int ny, float k = 1.0f) {
    Problem p;
    p.nx = nx;
    p.ny = ny;
    p.h = 1.0f / (nx - 1);
    p.k = k;
    p.q.assign((size_t)nx * ny, 0.0f);
    p.T_exact.assign((size_t)nx * ny, 0.0f);
    const double pi = 3.14159265358979323846;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = (double)i / (nx - 1);
            double y = (double)j / (ny - 1);
            double s = std::sin(pi * x) * std::sin(pi * y);
            p.T_exact[(size_t)j * nx + i] = (float)s;
            p.q[(size_t)j * nx + i] = (float)(2.0 * pi * pi * k * s);
        }
    }
    return p;
}

// Largest absolute difference between two fields.
inline double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}
