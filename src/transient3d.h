// transient3d.h - time stepping for the 3D stack.
//
// Backward Euler: each step solves
//     (C/dt + A) T_new = C/dt T_old + P
// where C is the heat capacity of each cell. It is unconditionally stable, so dt
// can grow as the stack approaches steady state. Each step is one PCG solve,
// warm-started from T_old.
//
// The stack has very different time scales: heat crosses a 50 um silicon layer
// in ~30 us, spreads around a 1 mm hotspot in ~10 ms, and the 1 mm copper
// spreader plus heat sink settle over ~0.2 s. So the steps start small and grow
// geometrically: dt_{n+1} = growth * dt_n, capped at dt_max.
#pragma once
#include <algorithm>
#include <chrono>
#include <vector>
#include "pcg3d.h"

struct TransientOptions {
    double dt0 = 1e-5;     // first time step (s)
    double growth = 1.1;   // each step is this factor longer than the previous one
    double dt_max = 0.05;  // longest allowed step (s)
    double t_end = 1.0;    // simulated time (s)
    double tol = 1e-9;     // PCG tolerance per step, ||r|| / ||b||
    int maxit = 50000;
    Precond precond = Precond::Multigrid;
};

struct StepRecord {
    double t, dt;       // time at the end of the step, step size (s)
    int iters;          // PCG iterations for this step
    double T_max;       // hottest cell, rise above ambient (K)
    double q_out;       // heat leaving through the boundaries (W)
    double q_stored;    // heat going into storage, sum C dT/dt (W)
};

struct TransientResult {
    std::vector<StepRecord> steps;
    std::vector<Real> T;   // temperature rise at t_end
    long total_iters = 0;
    double ms = 0;         // wall time, including copying T to the host every step
    bool converged = true; // every step's PCG converged
};

// Starts at ambient (T = 0) with the power switched on at t = 0.
// Solver is CpuPcg or GpuPcg.
template <class Solver>
TransientResult run_transient(const Model3D& m, const TransientOptions& o) {
    auto t0 = std::chrono::high_resolution_clock::now();
    TransientResult res;
    Solver s(m, o.precond);
    s.set_x_zero();

    std::vector<Real> T_old(m.ncells(), Real(0)), T_new;
    std::vector<double> shift(m.nz);
    double t = 0, dt = o.dt0, last_dt = -1;

    while (t < o.t_end * (1 - 1e-12)) {
        dt = std::min(dt, o.t_end - t);
        if (dt != last_dt) {  // C/dt changes, so the matrix and preconditioner change
            for (int l = 0; l < m.nz; ++l) shift[l] = m.cap[l] / dt;
            s.set_shift(shift);
            last_dt = dt;
        }
        s.set_rhs_transient();              // b = P + C/dt * T_old (x still holds T_old)
        PcgStats st = s.solve(o.tol, o.maxit);
        s.get_x(T_new);
        t += dt;

        StepRecord rec;
        rec.t = t;
        rec.dt = dt;
        rec.iters = st.iters;
        rec.T_max = (double)*std::max_element(T_new.begin(), T_new.end());
        rec.q_out = heat_out(m, T_new);
        rec.q_stored = heat_stored(m, T_old, T_new, dt);
        res.steps.push_back(rec);
        res.total_iters += st.iters;
        res.converged = res.converged && st.converged;

        T_old.swap(T_new);
        dt = std::min(dt * o.growth, o.dt_max);
    }
    res.T = T_old;
    auto t1 = std::chrono::high_resolution_clock::now();
    res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}
