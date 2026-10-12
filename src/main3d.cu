// main3d.cu - verification and benchmark driver for the 3D stack solver.
//
//   thermal3d          run the checks, print the stack temperatures, benchmark 256 x 256
//   (compiled with -DREAL_FLOAT the solver runs in float, to show why it shouldn't)
//   thermal3d <n>      benchmark an n x n grid only (CPU reference skipped above 256)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "pcg3d.h"
#include "stack3d.h"

static const double kTol = 1e-6;   // PCG stops at ||r|| / ||P|| < kTol
static const int kMaxIt = 50000;
// The true residual can't drop much below kTol, so allow a 10x margin.
static const double kTrueTol = 1e-5;
// CPU and GPU run the same arithmetic in a different order.
static const double kMatchTol = sizeof(Real) == 8 ? 1e-6 : 1e-3;

static double max_of(const std::vector<Real>& v) {
    return *std::max_element(v.begin(), v.end());
}

static double max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i]));
    return d;
}

static bool check(const char* what, double value, double limit) {
    bool ok = value < limit;
    printf("  %-34s = %.3e  %s\n", what, value, ok ? "PASS" : "FAIL");
    return ok;
}

// 25 W per tier, with 3 W of the bottom tier packed into a 1 mm x 1 mm hotspot.
static Model3D hotspot_model(int n) {
    Model3D m = default_model(n, n);
    add_uniform_power(m, 0, 22.0);
    add_hotspot(m, 0, 3e-3, 3e-3, 1e-3, 3.0);
    add_uniform_power(m, 2, 25.0);
    return m;
}

static void print_layers(const Model3D& m, const std::vector<Real>& T) {
    printf("  %-20s %10s %10s %10s\n", "layer", "thick(um)", "min(C)", "max(C)");
    for (int L = (int)m.layers.size() - 1; L >= 0; --L) {  // print top to bottom
        double lo = 1e30, hi = -1e30;
        for (int l = 0; l < m.nz; ++l) {
            if (m.plane_layer[l] != L) continue;
            for (size_t c = 0; c < m.nxy(); ++c) {
                double t = T[(size_t)l * m.nxy() + c];
                lo = std::min(lo, t);
                hi = std::max(hi, t);
            }
        }
        printf("  %-20s %10.0f %10.2f %10.2f\n", m.layers[L].name.c_str(),
               m.layers[L].thickness * 1e6, lo + m.T_amb, hi + m.T_amb);
    }
}

static bool verify() {
    bool ok = true;
    const int n = 64;

    // Test 1: uniform power has an exact answer (each column is a 1D resistor chain).
    {
        printf("== verify 1: uniform power, %d x %d x %d, vs exact 1D solution ==\n", n, n,
               default_model(n, n).nz);
        Model3D m = default_model(n, n);
        add_uniform_power(m, 0, 25.0);
        add_uniform_power(m, 2, 25.0);
        SolveResult c = cpu_pcg(m, kTol, kMaxIt);
        SolveResult g = gpu_pcg(m, kTol, kMaxIt);
        std::vector<double> Tz = analytic_1d(m);
        double tmax = *std::max_element(Tz.begin(), Tz.end());
        double ec = 0, eg = 0;
        for (int l = 0; l < m.nz; ++l)
            for (size_t k = 0; k < m.nxy(); ++k) {
                size_t id = (size_t)l * m.nxy() + k;
                ec = std::max(ec, std::fabs(c.T[id] - Tz[l]));
                eg = std::max(eg, std::fabs(g.T[id] - Tz[l]));
            }
        printf("  iterations: CPU %d, GPU %d\n", c.iters, g.iters);
        ok &= c.converged && g.converged;
        ok &= check("CPU true residual", true_relres(m, c.T), kTrueTol);
        ok &= check("GPU true residual", true_relres(m, g.T), kTrueTol);
        ok &= check("CPU vs exact, max err / Tmax", ec / tmax, 1e-4);
        ok &= check("GPU vs exact, max err / Tmax", eg / tmax, 1e-4);
        double P = total_power(m);
        ok &= check("GPU energy balance |out - in| / in", std::fabs(heat_out(m, g.T) - P) / P, 1e-4);
    }

    // Test 2: hotspot. No closed form, so check energy balance and GPU vs CPU.
    {
        printf("\n== verify 2: hotspot in bottom tier, %d x %d ==\n", n, n);
        Model3D m = hotspot_model(n);
        SolveResult c = cpu_pcg(m, kTol, kMaxIt);
        SolveResult g = gpu_pcg(m, kTol, kMaxIt);
        double P = total_power(m);
        printf("  iterations: CPU %d, GPU %d\n", c.iters, g.iters);
        ok &= c.converged && g.converged;
        ok &= check("CPU true residual", true_relres(m, c.T), kTrueTol);
        ok &= check("GPU true residual", true_relres(m, g.T), kTrueTol);
        ok &= check("CPU energy balance |out - in| / in", std::fabs(heat_out(m, c.T) - P) / P, 1e-4);
        ok &= check("GPU energy balance |out - in| / in", std::fabs(heat_out(m, g.T) - P) / P, 1e-4);
        ok &= check("max |GPU - CPU| / Tmax", max_abs_diff(g.T, c.T) / max_of(c.T), kMatchTol);
    }
    return ok;
}

static void benchmark(int n) {
    Model3D m = hotspot_model(n);
    printf("\n== hotspot stack: %d x %d x %d = %zu cells, %.0f W, ambient %.0f C, %s ==\n",
           n, n, m.nz, m.ncells(), total_power(m), m.T_amb,
           sizeof(Real) == 8 ? "double" : "float");

    gpu_pcg(default_model(32, 32), kTol, 10);  // warm-up: pays CUDA context setup

    SolveResult g = gpu_pcg(m, kTol, kMaxIt);
    print_layers(m, g.T);

    printf("\n  %-6s %8s %12s %12s\n", "", "iters", "time (ms)", "ms / iter");
    if (n <= 256) {
        SolveResult c = cpu_pcg(m, kTol, kMaxIt);
        printf("  %-6s %8d %12.1f %12.3f\n", "CPU", c.iters, c.ms, c.ms / c.iters);
        printf("  %-6s %8d %12.1f %12.3f   (%.1fx)\n", "GPU", g.iters, g.ms, g.ms / g.iters,
               c.ms / g.ms);
        printf("  max |GPU - CPU| / Tmax = %.3e\n", max_abs_diff(g.T, c.T) / max_of(c.T));
    } else {
        printf("  %-6s %8d %12.1f %12.3f\n", "GPU", g.iters, g.ms, g.ms / g.iters);
    }

    // Always check the answer, not just the solver's own convergence flag.
    double P = total_power(m), tr = true_relres(m, g.T);
    printf("  GPU true residual = %.2e   energy balance |out - in| / in = %.2e\n", tr,
           std::fabs(heat_out(m, g.T) - P) / P);
    if (!g.converged) printf("  WARNING: GPU solve hit the iteration limit (relres %.2e)\n", g.relres);
    if (tr > kTrueTol) printf("  WARNING: true residual above %.0e - result is not accurate\n", kTrueTol);
}

int main(int argc, char** argv) {
    if (argc == 2) {
        benchmark(atoi(argv[1]));
        return 0;
    }
    if (!verify()) {
        printf("\nVerification FAILED - fix correctness before benchmarking.\n");
        return 1;
    }
    benchmark(256);
    return 0;
}
