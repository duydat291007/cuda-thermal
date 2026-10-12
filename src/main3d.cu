// main3d.cu - verification and benchmark driver for the 3D stack solver.
//
//   thermal3d                  run all checks, then the steady benchmark (256 x 256)
//   thermal3d <n>              steady benchmark on an n x n grid (CPU skipped above 256)
//   thermal3d transient [n]    heat-up after power-on, n x n grid (default 200);
//                              prints a timeline and writes transient.csv
// Compiled with -DREAL_FLOAT the solver runs in float, to show why it shouldn't.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pcg3d.h"
#include "stack3d.h"
#include "transient3d.h"

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

    // Test 3: transient. Every step must conserve energy (power in = heat out +
    // heat stored), the GPU must match the CPU, and after a long time the
    // temperatures must settle to the steady-state solution.
    {
        printf("\n== verify 3: transient heat-up, hotspot stack, %d x %d ==\n", n, n);
        Model3D m = hotspot_model(n);
        const double P = total_power(m);

        TransientOptions shortrun;
        shortrun.t_end = 0.02;
        shortrun.growth = 1.2;
        TransientResult c = run_transient<CpuPcg>(m, shortrun);
        TransientResult g = run_transient<GpuPcg>(m, shortrun);
        printf("  first 20 ms: %zu steps, PCG iterations CPU %ld, GPU %ld\n", g.steps.size(),
               c.total_iters, g.total_iters);
        ok &= c.converged && g.converged;
        ok &= check("max |GPU - CPU| / Tmax at 20 ms", max_abs_diff(g.T, c.T) / max_of(c.T),
                    kMatchTol);

        TransientOptions longrun;
        longrun.t_end = 3.0;  // ~15 thermal time constants of the heat sink
        longrun.dt_max = 0.1;
        TransientResult lg = run_transient<GpuPcg>(m, longrun);
        double eb = 0;
        for (const StepRecord& r : lg.steps)
            eb = std::max(eb, std::fabs(r.q_out + r.q_stored - P) / P);
        ok &= lg.converged;
        ok &= check("GPU energy balance, worst step", eb, 1e-6);
        SolveResult ss = gpu_pcg(m, 1e-10, kMaxIt);
        ok &= check("GPU at 3 s vs steady state / Tmax", max_abs_diff(lg.T, ss.T) / max_of(ss.T),
                    1e-5);
    }
    return ok;
}

// Time at which T_max first reaches `frac` of its steady value, interpolated
// linearly between steps. Returns -1 if never reached.
static double time_to(const std::vector<StepRecord>& steps, double target) {
    double t_prev = 0, T_prev = 0;
    for (const StepRecord& r : steps) {
        if (r.T_max >= target)
            return t_prev + (r.t - t_prev) * (target - T_prev) / (r.T_max - T_prev);
        t_prev = r.t;
        T_prev = r.T_max;
    }
    return -1;
}

static void transient_demo(int n) {
    Model3D m = hotspot_model(n);
    const double P = total_power(m);
    printf("\n== transient: power switched on at t = 0, %d x %d x %d = %zu cells, %.0f W, %s ==\n",
           n, n, m.nz, m.ncells(), P, sizeof(Real) == 8 ? "double" : "float");

    gpu_pcg(default_model(32, 32), kTol, 10);  // warm-up: pays CUDA context setup

    SolveResult ss = gpu_pcg(m, 1e-8, kMaxIt);
    const double Tss = max_of(ss.T);

    TransientOptions o;
    o.t_end = 2.0;
    o.dt_max = 0.1;
    TransientResult tr = run_transient<GpuPcg>(m, o);

    printf("  %10s %10s %10s %8s %12s %12s\n", "t (ms)", "T_max (C)", "% of rise", "iters",
           "heat out(W)", "stored (W)");
    const double marks[] = {0.01e-3, 0.1e-3, 0.3e-3, 1e-3, 3e-3, 10e-3, 30e-3,
                             0.1,     0.3,    1.0,    2.0};
    size_t k = 0;
    for (double mark : marks) {
        while (k < tr.steps.size() && tr.steps[k].t < mark * (1 - 1e-9)) ++k;
        if (k >= tr.steps.size()) break;
        const StepRecord& r = tr.steps[k];
        printf("  %10.3f %10.2f %10.1f %8d %12.3f %12.3f\n", r.t * 1e3, r.T_max + m.T_amb,
               100 * r.T_max / Tss, r.iters, r.q_out, r.q_stored);
    }
    printf("\n  steady state T_max = %.2f C. Hotspot reaches 50%% / 90%% / 99%% of its rise at\n"
           "  %.2f ms / %.0f ms / %.0f ms\n",
           Tss + m.T_amb, 1e3 * time_to(tr.steps, 0.5 * Tss),
           1e3 * time_to(tr.steps, 0.9 * Tss), 1e3 * time_to(tr.steps, 0.99 * Tss));

    double eb = 0;
    for (const StepRecord& r : tr.steps) eb = std::max(eb, std::fabs(r.q_out + r.q_stored - P) / P);
    printf("\n  %zu steps, %ld PCG iterations, %.0f ms total (%.3f ms / iteration)\n",
           tr.steps.size(), tr.total_iters, tr.ms, tr.ms / (double)tr.total_iters);
    printf("  worst energy balance |out + stored - in| / in = %.2e\n", eb);
    if (!tr.converged) printf("  WARNING: some steps hit the iteration limit\n");

    printf("\n  temperatures at t = %.1f s:\n", tr.steps.back().t);
    print_layers(m, tr.T);

    FILE* f = fopen("transient.csv", "w");
    if (!f) { printf("  could not write transient.csv\n"); return; }
    fprintf(f, "t_s,dt_s,pcg_iters,T_max_C,heat_out_W,heat_stored_W\n");
    for (const StepRecord& r : tr.steps)
        fprintf(f, "%.6e,%.6e,%d,%.6f,%.6f,%.6f\n", r.t, r.dt, r.iters, r.T_max + m.T_amb,
                r.q_out, r.q_stored);
    fclose(f);
    printf("\n  wrote transient.csv (%zu rows) - open it in Excel and plot T_max_C vs t_s\n",
           tr.steps.size());
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
    if (argc >= 2 && strcmp(argv[1], "transient") == 0) {
        transient_demo(argc >= 3 ? atoi(argv[2]) : 200);
        return 0;
    }
    if (argc == 2) {
        benchmark(atoi(argv[1]));
        return 0;
    }
    if (!verify()) {
        printf("\nVerification FAILED - fix correctness before benchmarking.\n");
        return 1;
    }
    benchmark(256);
    printf("\nNext: thermal3d.exe transient   (heat-up after power-on)\n");
    return 0;
}
