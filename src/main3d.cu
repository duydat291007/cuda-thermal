// main3d.cu - verification and benchmark driver for the 3D stack solver.
//
//   thermal3d                  run all checks, then the steady benchmark (256 x 256)
//   thermal3d <n>              steady benchmark on an n x n grid, Jacobi vs multigrid
//                              (CPU skipped above 256)
//   thermal3d transient [n] [jacobi]
//                              heat-up after power-on, n x n grid (default 200),
//                              multigrid unless "jacobi" is given;
//                              prints a timeline and writes transient.csv
//   thermal3d case <folder> [n]
//                              steady state of a case (stack.csv, floorplan.csv,
//                              tsv.csv) on an n x n grid (default 200): block
//                              temperatures, TSV effect, tmap_*.csv heat maps
//   thermal3d export3dice <folder> <cell_um> <outdir>
//                              write the case as 3D-ICE input (case.stk, L*.flp)
//   thermal3d compare3dice <folder> <cell_um> <dir>
//                              solve the case on the same grid and compare with
//                              the 3D-ICE results (tmap_L*.txt, tflp_*.txt) in dir
// Compiled with -DREAL_FLOAT the solver runs in float, to show why it shouldn't.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include "case3d.h"
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
    finalize(m);
    return m;
}

// Tiny model for warming up the GPU (the first CUDA call pays context setup).
static Model3D warmup_model() {
    Model3D m = default_model(32, 32);
    add_uniform_power(m, 0, 1.0);
    finalize(m);
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
        printf("== verify 1: uniform power, %d x %d x %d, vs exact 1D solution (Jacobi) ==\n", n, n,
               default_model(n, n).nz);
        Model3D m = default_model(n, n);
        add_uniform_power(m, 0, 25.0);
        add_uniform_power(m, 2, 25.0);
        finalize(m);
        SolveResult c = cpu_pcg(m, kTol, kMaxIt, Precond::Jacobi);
        SolveResult g = gpu_pcg(m, kTol, kMaxIt, Precond::Jacobi);
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
        printf("\n== verify 2: hotspot in bottom tier, %d x %d (Jacobi) ==\n", n, n);
        Model3D m = hotspot_model(n);
        SolveResult c = cpu_pcg(m, kTol, kMaxIt, Precond::Jacobi);
        SolveResult g = gpu_pcg(m, kTol, kMaxIt, Precond::Jacobi);
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
        printf("\n== verify 3: transient heat-up, hotspot stack, %d x %d (Jacobi) ==\n", n, n);
        Model3D m = hotspot_model(n);
        const double P = total_power(m);

        TransientOptions shortrun;
        shortrun.t_end = 0.02;
        shortrun.growth = 1.2;
        shortrun.precond = Precond::Jacobi;
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
        longrun.precond = Precond::Jacobi;
        TransientResult lg = run_transient<GpuPcg>(m, longrun);
        double eb = 0;
        for (const StepRecord& r : lg.steps)
            eb = std::max(eb, std::fabs(r.q_out + r.q_stored - P) / P);
        ok &= lg.converged;
        ok &= check("GPU energy balance, worst step", eb, 1e-6);
        SolveResult ss = gpu_pcg(m, 1e-10, kMaxIt, Precond::Jacobi);
        ok &= check("GPU at 3 s vs steady state / Tmax", max_abs_diff(lg.T, ss.T) / max_of(ss.T),
                    1e-5);
    }

    // Test 4: multigrid preconditioner. Same answers as Jacobi, far fewer iterations.
    {
        printf("\n== verify 4: multigrid preconditioner, hotspot stack, %d x %d ==\n", n, n);
        Model3D m = hotspot_model(n);
        const double P = total_power(m);
        SolveResult c = cpu_pcg(m, 1e-9, kMaxIt, Precond::Multigrid);
        SolveResult g = gpu_pcg(m, 1e-9, kMaxIt, Precond::Multigrid);
        SolveResult gj = gpu_pcg(m, 1e-9, kMaxIt, Precond::Jacobi);
        printf("  steady iterations to 1e-9: multigrid CPU %d, GPU %d; Jacobi GPU %d\n", c.iters,
               g.iters, gj.iters);
        ok &= c.converged && g.converged && gj.converged;
        ok &= check("CPU true residual", true_relres(m, c.T), kTrueTol);
        ok &= check("GPU true residual", true_relres(m, g.T), kTrueTol);
        ok &= check("max |GPU - CPU| / Tmax", max_abs_diff(g.T, c.T) / max_of(c.T), kMatchTol);
        ok &= check("max |multigrid - Jacobi| / Tmax", max_abs_diff(g.T, gj.T) / max_of(gj.T),
                    kMatchTol);

        TransientOptions o;
        o.t_end = 0.02;
        o.growth = 1.2;
        o.precond = Precond::Multigrid;
        TransientResult tm = run_transient<GpuPcg>(m, o);
        o.precond = Precond::Jacobi;
        TransientResult tj = run_transient<GpuPcg>(m, o);
        printf("  transient, first 20 ms: PCG iterations multigrid %ld, Jacobi %ld\n",
               tm.total_iters, tj.total_iters);
        double eb = 0;
        for (const StepRecord& r : tm.steps)
            eb = std::max(eb, std::fabs(r.q_out + r.q_stored - P) / P);
        ok &= tm.converged && tj.converged;
        ok &= check("transient energy balance, worst step", eb, 1e-6);
        ok &= check("transient multigrid vs Jacobi / Tmax", max_abs_diff(tm.T, tj.T) / max_of(tj.T),
                    kMatchTol);

        // CUDA graphs replay the same kernels, so the transient (where 1/dt changes
        // between graph replays) must match launching them directly, step by step.
        gpu_use_graphs(false);
        o.precond = Precond::Multigrid;
        TransientResult tn = run_transient<GpuPcg>(m, o);
        gpu_use_graphs(true);
        printf("  transient PCG iterations: graphs %ld, direct launches %ld\n", tm.total_iters,
               tn.total_iters);
        ok &= tn.converged;
        ok &= check("transient graphs vs direct launches / Tmax",
                    max_abs_diff(tm.T, tn.T) / max_of(tn.T), kMatchTol);
    }

    // Test 5: per-cell materials. Power blocks from a floorplan, and TSVs.
    {
        printf("\n== verify 5: floorplan and TSVs, %d x %d ==\n", n, n);
        const CaseSpec cs = example_case();

        // Power blocks are spread area-weighted over partially covered cells, so the
        // total must be conserved exactly even when edges fall inside cells.
        Model3D mb = default_model(n, n);
        double want = 0;
        for (const PowerBlock& b : cs.blocks) {  // shift off the cell edges where it fits
            double sx = (b.x0 + b.w + 0.037e-3 <= mb.Lx) ? 0.037e-3 : 0.0;
            double sy = (b.y0 + b.h + 0.011e-3 <= mb.Ly) ? 0.011e-3 : 0.0;
            add_power_block(mb, b.layer, b.x0 + sx, b.y0 + sy, b.w, b.h, b.watts);
            want += b.watts;
        }
        ok &= check("floorplan power conserved, rel. err", std::fabs(total_power(mb) - want) / want,
                    1e-12);

        // TSVs over the whole bond layer act like a bond layer with k = k_z(eff):
        // with uniform power that has the exact 1D solution.
        const int bond = find_layer(cs.layers, "bond + BEOL");
        const double f = 0.2;
        Model3D mt = default_model(n, n);
        add_uniform_power(mt, 0, 25.0);
        add_uniform_power(mt, 2, 25.0);
        add_tsv_region(mt, bond, 0, 0, mt.Lx, mt.Ly, f);
        finalize(mt);
        std::vector<Layer> eq_layers = default_stack();
        eq_layers[bond].k = eff_kz(eq_layers[bond].k, mt.tsv_k, f);
        Model3D ma = build_model(eq_layers, n, n, mt.Lx, mt.Ly, mt.h_top, mt.h_bot);
        add_uniform_power(ma, 0, 25.0);
        add_uniform_power(ma, 2, 25.0);
        std::vector<double> Tz = analytic_1d(ma);
        SolveResult gt = gpu_pcg(mt, 1e-10, kMaxIt, Precond::Multigrid);
        double e = 0, tmax = *std::max_element(Tz.begin(), Tz.end());
        for (int l = 0; l < mt.nz; ++l)
            for (size_t k = 0; k < mt.nxy(); ++k)
                e = std::max(e, std::fabs(gt.T[(size_t)l * mt.nxy() + k] - Tz[l]));
        ok &= gt.converged;
        ok &= check("TSV layer vs exact 1D, err / Tmax", e / tmax, 1e-5);

        // The example case: blocks plus TSV fields beside the cores.
        Model3D m = build_case_model(cs, n, n, true);
        const double P = total_power(m);
        SolveResult c = cpu_pcg(m, 1e-9, kMaxIt, Precond::Multigrid);
        SolveResult g = gpu_pcg(m, 1e-9, kMaxIt, Precond::Multigrid);
        SolveResult gj = gpu_pcg(m, 1e-9, kMaxIt, Precond::Jacobi);
        printf("  example case, iterations to 1e-9: multigrid CPU %d, GPU %d; Jacobi GPU %d\n",
               c.iters, g.iters, gj.iters);
        ok &= c.converged && g.converged && gj.converged;
        ok &= check("GPU true residual", true_relres(m, g.T), kTrueTol);
        ok &= check("max |GPU - CPU| / Tmax", max_abs_diff(g.T, c.T) / max_of(c.T), kMatchTol);
        ok &= check("max |multigrid - Jacobi| / Tmax", max_abs_diff(g.T, gj.T) / max_of(gj.T),
                    kMatchTol);
        ok &= check("energy balance |out - in| / in", std::fabs(heat_out(m, g.T) - P) / P, 1e-6);
    }
    return ok;
}

// Max and area-weighted mean temperature rise of a power block, over its
// layer's planes.
static void block_temps(const Model3D& m, const std::vector<Real>& T, const PowerBlock& b,
                        double& tmax, double& tavg) {
    tmax = -1e30;
    double sum = 0, wsum = 0;
    for (int l = 0; l < m.nz; ++l) {
        if (m.plane_layer[l] != b.layer) continue;
        for_each_overlap(m, b.x0, b.y0, b.w, b.h, [&](size_t c, double frac) {
            double t = T[(size_t)l * m.nxy() + c];
            tmax = std::max(tmax, t);
            sum += frac * t;
            wsum += frac;
        });
    }
    tavg = sum / wsum;
}

// Writes the hottest temperature through each active layer's planes as a grid
// (row = y, column = x, deg C) for a heat map in Excel or matplotlib.
static void write_tmaps(const Model3D& m, const std::vector<Real>& T) {
    for (size_t L = 0; L < m.layers.size(); ++L) {
        if (!m.layers[L].active) continue;
        std::string name = "tmap_layer" + std::to_string(L) + ".csv";
        FILE* f = fopen(name.c_str(), "w");
        if (!f) { printf("  could not write %s\n", name.c_str()); continue; }
        for (int j = m.ny - 1; j >= 0; --j) {  // top row = largest y, like a floorplan drawing
            for (int i = 0; i < m.nx; ++i) {
                double t = -1e30;
                for (int l = 0; l < m.nz; ++l)
                    if (m.plane_layer[l] == (int)L) t = std::max(t, (double)T[m.idx(i, j, l)]);
                fprintf(f, i ? ",%.3f" : "%.3f", t + m.T_amb);
            }
            fprintf(f, "\n");
        }
        fclose(f);
        printf("  wrote %s  (%s, %d x %d, deg C)\n", name.c_str(), m.layers[L].name.c_str(),
               m.nx, m.ny);
    }
}

static int run_case(const std::string& dir, int n) {
    CaseSpec cs;
    std::string err;
    if (!load_case(dir, cs, err)) {
        printf("Could not load case '%s': %s\n", dir.c_str(), err.c_str());
        return 1;
    }
    Model3D m = build_case_model(cs, n, n, true);
    const double P = total_power(m);
    printf("\n== case %s: %d layers, %zu power blocks, %zu TSV regions, %.1f W, %d x %d x %d cells ==\n",
           dir.c_str(), (int)cs.layers.size(), cs.blocks.size(), cs.tsvs.size(), P, n, n, m.nz);

    gpu_pcg(warmup_model(), kTol, 10, Precond::Jacobi);
    SolveResult g = gpu_pcg(m, 1e-8, kMaxIt, Precond::Multigrid);
    printf("  solved in %d multigrid PCG iterations, %.1f ms (+ %.0f ms setup); true residual %.1e, energy balance %.1e\n",
           g.iters, g.solve_ms, g.setup_ms, true_relres(m, g.T), std::fabs(heat_out(m, g.T) - P) / P);
    if (!g.converged) printf("  WARNING: did not converge\n");

    std::vector<Real> T0;  // the same case without TSVs, to show their effect
    if (!cs.tsvs.empty()) {
        Model3D m0 = build_case_model(cs, n, n, false);
        T0 = gpu_pcg(m0, 1e-8, kMaxIt, Precond::Multigrid).T;
    }

    printf("\n");
    print_layers(m, g.T);

    printf("\n  %-14s %-20s %7s %9s %9s %9s", "block", "layer", "W", "W/cm^2", "max(C)", "avg(C)");
    if (!T0.empty()) printf(" %13s", "TSVs change");
    printf("\n");
    for (const PowerBlock& b : cs.blocks) {
        double tmax, tavg;
        block_temps(m, g.T, b, tmax, tavg);
        printf("  %-14s %-20s %7.2f %9.1f %9.2f %9.2f", b.name.c_str(),
               cs.layers[b.layer].name.c_str(), b.watts, b.watts / (b.w * b.h * 1e4),
               tmax + m.T_amb, tavg + m.T_amb);
        if (!T0.empty()) {
            double t0max, t0avg;
            block_temps(m, T0, b, t0max, t0avg);
            printf(" %+10.2f C", tmax - t0max);
        }
        printf("\n");
    }

    size_t hot = (size_t)(std::max_element(g.T.begin(), g.T.end()) - g.T.begin());
    int l = (int)(hot / m.nxy()), j = (int)((hot % m.nxy()) / m.nx), i = (int)(hot % m.nx);
    printf("\n  hottest point: %.2f C in %s at x = %.2f mm, y = %.2f mm\n", g.T[hot] + m.T_amb,
           m.layers[m.plane_layer[l]].name.c_str(), (i + 0.5) * m.dx * 1e3, (j + 0.5) * m.dy * 1e3);
    if (!T0.empty())
        printf("  without TSVs:  %.2f C\n", max_of(T0) + m.T_amb);
    printf("\n");
    write_tmaps(m, g.T);
    return 0;
}

// Reads a 3D-ICE Tflp file: returns block name -> temperature (K).
static bool read_3dice_tflp(const std::string& path, std::vector<std::string>& names,
                            std::vector<double>& values) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("% Time", 0) == 0) {
            std::stringstream ss(line.substr(1));
            std::string tok;
            ss >> tok;  // "Time(s)"
            while (ss >> tok) {
                size_t k = tok.find("(K)");
                names.push_back(k == std::string::npos ? tok : tok.substr(0, k));
            }
        } else if (!line.empty() && line[0] != '%') {
            std::stringstream ss(line);
            double t, v;
            ss >> t;  // time
            while (ss >> v) values.push_back(v);
        }
    }
    return !names.empty() && names.size() == values.size();
}

static int export_case_3dice(const std::string& dir, double cell_um, const std::string& outdir) {
    CaseSpec cs;
    std::string err;
    if (!load_case(dir, cs, err) || !export_3dice(cs, cell_um, outdir, err)) {
        printf("Export failed: %s\n", err.c_str());
        return 1;
    }
    printf("Wrote %s/case.stk and floorplans (%g um cells, %d x %d).\n", outdir.c_str(), cell_um,
           (int)std::lround(cs.Lx * 1e6 / cell_um), (int)std::lround(cs.Ly * 1e6 / cell_um));
    printf("Run 3D-ICE there:  3D-ICE-Emulator case.stk\n");
    return 0;
}

static int compare_case_3dice(const std::string& dir, double cell_um, const std::string& icedir) {
    CaseSpec cs;
    std::string err;
    if (!load_case(dir, cs, err)) {
        printf("Could not load case '%s': %s\n", dir.c_str(), err.c_str());
        return 1;
    }
    const int nx = (int)std::lround(cs.Lx * 1e6 / cell_um), ny = (int)std::lround(cs.Ly * 1e6 / cell_um);
    Model3D m = build_case_model(cs, nx, ny, true);
    gpu_pcg(warmup_model(), kTol, 10, Precond::Jacobi);
    SolveResult g = gpu_pcg(m, 1e-10, kMaxIt, Precond::Multigrid);
    printf("\n== compare with 3D-ICE: %s, %d x %d x %d cells, %.1f W ==\n", dir.c_str(), nx, ny,
           m.nz, total_power(m));
    printf("  ours: %d multigrid iterations, %.1f ms (+ %.0f ms setup), true residual %.1e\n",
           g.iters, g.solve_ms, g.setup_ms,
           true_relres(m, g.T));

    int failures = 0;
    for (size_t L = 0; L < cs.layers.size(); ++L) {
        if (!cs.layers[L].active) continue;
        int plane = -1;
        for (int l = 0; l < m.nz; ++l)
            if (m.plane_layer[l] == (int)L) plane = l;
        std::vector<std::vector<double>> ice;
        std::string path = icedir + "/tmap_L" + std::to_string(L) + ".txt";
        if (!read_3dice_tmap(path, ice) || (int)ice.size() != ny || (int)ice[0].size() != nx) {
            printf("  %s: missing, or not %d rows x %d columns\n", path.c_str(), ny, nx);
            ++failures;
            continue;
        }
        // 3D-ICE writes row j = 0 (y = 0) first.
        double maxd = 0, sumd = 0, ours_max = -1e30, ice_max = -1e30;
        int wi = 0, wj = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                double a = g.T[m.idx(i, j, plane)] + m.T_amb, b = ice[j][i] - 273.15;
                double d = a - b;
                sumd += d;
                if (std::fabs(d) > maxd) { maxd = std::fabs(d); wi = i; wj = j; }
                ours_max = std::max(ours_max, a);
                ice_max = std::max(ice_max, b);
            }
        printf("\n  %s (whole map, %d cells):\n", cs.layers[L].name.c_str(), nx * ny);
        printf("    hottest cell: ours %.3f C, 3D-ICE %.3f C\n", ours_max, ice_max);
        printf("    max |difference| = %.4f C at x = %.2f mm, y = %.2f mm; mean difference %+.4f C\n",
               maxd, (wi + 0.5) * m.dx * 1e3, (wj + 0.5) * m.dy * 1e3, sumd / (nx * ny));

        std::vector<std::string> nmax, navg;
        std::vector<double> vmax, vavg;
        bool okf = read_3dice_tflp(icedir + "/tflp_max_L" + std::to_string(L) + ".txt", nmax, vmax) &&
                   read_3dice_tflp(icedir + "/tflp_avg_L" + std::to_string(L) + ".txt", navg, vavg);
        if (!okf) { printf("    (no Tflp files)\n"); continue; }
        printf("    %-14s %10s %10s %8s %10s %10s %8s\n", "block", "max ours", "max 3DICE", "diff",
               "avg ours", "avg 3DICE", "diff");
        for (const PowerBlock& b : cs.blocks) {
            if ((size_t)b.layer != L) continue;
            std::string nm = ice_name(b.name);
            size_t k = std::find(nmax.begin(), nmax.end(), nm) - nmax.begin();
            size_t a = std::find(navg.begin(), navg.end(), nm) - navg.begin();
            if (k >= vmax.size() || a >= vavg.size()) continue;
            double tmax, tavg;
            block_temps(m, g.T, b, tmax, tavg);
            tmax += m.T_amb;
            tavg += m.T_amb;
            double imax = vmax[k] - 273.15, iavg = vavg[a] - 273.15;
            printf("    %-14s %10.3f %10.3f %+8.4f %10.3f %10.3f %+8.4f\n", b.name.c_str(), tmax, imax,
                   tmax - imax, tavg, iavg, tavg - iavg);
        }
    }
    return failures ? 1 : 0;
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

static void transient_demo(int n, Precond pc) {
    Model3D m = hotspot_model(n);
    const double P = total_power(m);
    printf("\n== transient: power switched on at t = 0, %d x %d x %d = %zu cells, %.0f W, %s, %s ==\n",
           n, n, m.nz, m.ncells(), P, sizeof(Real) == 8 ? "double" : "float", precond_name(pc));

    gpu_pcg(warmup_model(), kTol, 10, Precond::Jacobi);  // warm-up: CUDA context setup

    SolveResult ss = gpu_pcg(m, 1e-8, kMaxIt, Precond::Multigrid);
    const double Tss = max_of(ss.T);

    TransientOptions o;
    o.t_end = 2.0;
    o.dt_max = 0.1;
    o.precond = pc;
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
    printf("\n  %zu steps, %ld PCG iterations, %.0f ms total, of which %.0f ms solving\n"
           "  (%.3f ms / iteration; the rest is setup and copying T to the host every step)\n",
           tr.steps.size(), tr.total_iters, tr.ms, tr.solve_ms,
           tr.solve_ms / (double)tr.total_iters);
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
    printf("\n== steady state: %d x %d x %d = %zu cells, %.0f W, ambient %.0f C, %s ==\n",
           n, n, m.nz, m.ncells(), total_power(m), m.T_amb,
           sizeof(Real) == 8 ? "double" : "float");

    gpu_pcg(warmup_model(), kTol, 10, Precond::Jacobi);  // warm-up: CUDA context setup

    SolveResult gm = gpu_pcg(m, kTol, kMaxIt, Precond::Multigrid);
    SolveResult gj = gpu_pcg(m, kTol, kMaxIt, Precond::Jacobi);
    gpu_use_graphs(false);  // same kernels, launched one by one
    SolveResult gmn = gpu_pcg(m, kTol, kMaxIt, Precond::Multigrid);
    gpu_use_graphs(true);
    print_layers(m, gm.T);

    printf("\n  %-26s %6s %10s %10s %10s %10s\n", "", "iters", "setup ms", "solve ms",
           "ms / iter", "speedup");
    auto row = [](const char* name, const SolveResult& r, double ref_ms) {
        printf("  %-26s %6d %10.1f %10.2f %10.3f %9.1fx\n", name, r.iters, r.setup_ms,
               r.solve_ms, r.solve_ms / r.iters, ref_ms / r.solve_ms);
    };
    double ref = 0;
    if (n <= 256) {  // the CPU is too slow beyond this
        SolveResult cj = cpu_pcg(m, kTol, kMaxIt, Precond::Jacobi);
        SolveResult cm = cpu_pcg(m, kTol, kMaxIt, Precond::Multigrid);
        ref = cj.solve_ms;
        printf("  (solve time only; speedup is relative to CPU Jacobi)\n");
        row("CPU Jacobi", cj, ref);
        row("CPU multigrid", cm, ref);
    } else {
        ref = gj.solve_ms;
        printf("  (solve time only; speedup is relative to GPU Jacobi)\n");
    }
    row("GPU Jacobi", gj, ref);
    row("GPU multigrid, no graphs", gmn, ref);
    row("GPU multigrid", gm, ref);
    printf("  CUDA graphs: multigrid solve %.2f -> %.2f ms (%.2fx); setup is done once per solver\n",
           gmn.solve_ms, gm.solve_ms, gmn.solve_ms / gm.solve_ms);
    printf("  max |graphs - no graphs| / Tmax = %.3e\n", max_abs_diff(gm.T, gmn.T) / max_of(gm.T));
    printf("  max |multigrid - Jacobi| / Tmax = %.3e\n", max_abs_diff(gm.T, gj.T) / max_of(gj.T));

    // Always check the answer, not just the solver's own convergence flag.
    double P = total_power(m), tr = true_relres(m, gm.T);
    printf("  GPU multigrid: true residual = %.2e   energy balance |out - in| / in = %.2e\n", tr,
           std::fabs(heat_out(m, gm.T) - P) / P);
    if (!gm.converged || !gj.converged) printf("  WARNING: a solve hit the iteration limit\n");
    if (tr > kTrueTol) printf("  WARNING: true residual above %.0e - result is not accurate\n", kTrueTol);
}

int main(int argc, char** argv) {
    if (argc >= 5 && strcmp(argv[1], "export3dice") == 0)
        return export_case_3dice(argv[2], atof(argv[3]), argv[4]);
    if (argc >= 5 && strcmp(argv[1], "compare3dice") == 0)
        return compare_case_3dice(argv[2], atof(argv[3]), argv[4]);
    if (argc >= 3 && strcmp(argv[1], "case") == 0) {
        return run_case(argv[2], argc >= 4 ? atoi(argv[3]) : 200);
    }
    if (argc >= 2 && strcmp(argv[1], "transient") == 0) {
        int n = argc >= 3 ? atoi(argv[2]) : 200;
        bool jacobi = argc >= 4 && strcmp(argv[3], "jacobi") == 0;
        transient_demo(n, jacobi ? Precond::Jacobi : Precond::Multigrid);
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
