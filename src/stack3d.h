// stack3d.h - 3D layered chip stack: geometry, materials, TSVs, power maps.
//
// Finite-volume model on a cell-centered grid. Laterally the die is split into
// nx * ny cells of size dx * dy. Vertically each layer is split into ncells
// planes, so planes in different layers can have very different thicknesses.
//
// Materials are stored per cell, so a layer can contain TSV regions. Each cell
// exchanges heat with its 6 neighbors through a face conductance G (W/K): the
// two half-cells on either side of a face are in series,
//     G = area / ( d1 / (2 k1) + d2 / (2 k2) )
// which is the harmonic-mean rule for k at a material change. The top plane
// connects to ambient through half a cell plus the heat sink (1 / h_top), the
// bottom plane through half a cell plus 1 / h_bot (h_bot = 0: adiabatic). The
// side walls are adiabatic.
//
// Temperatures are solved relative to ambient: the energy balance of cell c is
//     C_c dT_c/dt + sum_n G_n (T_c - T_n) + G_amb T_c = P_c
// Steady state drops the first term; a backward Euler step turns it into C/dt.
//
// TSVs (through-silicon vias) are copper columns too small to resolve, so a TSV
// region is modeled as an effective medium with copper area fraction f:
//     vertical:  k_z  = f k_Cu + (1 - f) k        (parallel paths)
//     lateral:   k_xy = k ((1+f) k_Cu + (1-f) k) / ((1-f) k_Cu + (1+f) k)
//                (Rayleigh / Maxwell-Garnett for parallel cylinders)
//     rho*c:     f rho_c_Cu + (1 - f) rho_c
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

constexpr int MAXZ = 64;  // max number of z-planes

struct Layer {
    std::string name;
    double thickness;  // m
    double k;          // W/(m K)
    double rho_c;      // volumetric heat capacity, J/(m^3 K) (density x specific heat)
    int ncells;        // number of z-planes in this layer
    bool active;       // dissipates power
};

struct Model3D {
    int nx = 0, ny = 0, nz = 0;
    double Lx = 0, Ly = 0, dx = 0, dy = 0;  // m
    double h_top = 0, h_bot = 0;            // W/(m^2 K)
    double T_amb = 25.0;                    // deg C
    double tsv_k = 400.0, tsv_rho_c = 3.45e6;  // TSV fill (copper)
    std::vector<Layer> layers;

    // Per z-plane.
    std::vector<int> plane_layer;  // which layer each plane belongs to
    std::vector<double> dz;

    // Per cell (nx * ny * nz, x fastest, then y, then z).
    std::vector<double> tsv_frac;  // copper area fraction, 0 outside TSV regions
    std::vector<double> P;         // power (W)

    // Operator, built by finalize() from the above. Stored in float: they define
    // the discrete model (a 1e-7 relative perturbation of it), they don't carry
    // the solution, and float halves their memory traffic.
    std::vector<float> gE;   // conductance to the +x neighbor (0 on the last column)
    std::vector<float> gN;   // conductance to the +y neighbor (0 on the last row)
    std::vector<float> gU;   // to the +z neighbor; on the top plane: to ambient
    std::vector<float> gB;   // nx * ny: bottom plane to ambient
    std::vector<float> cap;  // heat capacity (J/K)
    bool finalized = false;

    size_t nxy() const { return (size_t)nx * ny; }
    size_t ncells() const { return nxy() * nz; }
    size_t idx(int i, int j, int l) const { return ((size_t)l * ny + j) * nx + i; }
    int layer_index(const std::string& name) const {
        for (size_t L = 0; L < layers.size(); ++L)
            if (layers[L].name == name) return (int)L;
        return -1;
    }
};

inline Model3D build_model(const std::vector<Layer>& layers, int nx, int ny,
                           double Lx, double Ly, double h_top, double h_bot) {
    Model3D m;
    m.layers = layers;
    m.nx = nx; m.ny = ny;
    m.Lx = Lx; m.Ly = Ly;
    m.dx = Lx / nx; m.dy = Ly / ny;
    m.h_top = h_top; m.h_bot = h_bot;
    for (size_t L = 0; L < layers.size(); ++L)
        for (int c = 0; c < layers[L].ncells; ++c) {
            m.plane_layer.push_back((int)L);
            m.dz.push_back(layers[L].thickness / layers[L].ncells);
        }
    m.nz = (int)m.dz.size();
    if (m.nz > MAXZ) {
        fprintf(stderr, "Too many z-planes (%d > %d)\n", m.nz, MAXZ);
        exit(1);
    }
    m.tsv_frac.assign(m.ncells(), 0.0);
    m.P.assign(m.ncells(), 0.0);
    return m;
}

// Effective properties of a cell given its layer's k and TSV fraction f.
inline double eff_kz(double k, double kc, double f) { return f * kc + (1 - f) * k; }
inline double eff_kxy(double k, double kc, double f) {
    return k * ((1 + f) * kc + (1 - f) * k) / ((1 - f) * kc + (1 + f) * k);
}

// Builds the face conductances and capacities. Call after all TSV regions are
// added (power can change at any time; it doesn't enter the operator).
inline void finalize(Model3D& m) {
    const size_t n = m.ncells(), sxy = m.nxy();
    const double A = m.dx * m.dy;
    std::vector<double> kxy(n), kz(n);
    m.cap.assign(n, 0.0f);
    for (int l = 0; l < m.nz; ++l) {
        const Layer& L = m.layers[m.plane_layer[l]];
        for (size_t c = 0; c < sxy; ++c) {
            size_t id = (size_t)l * sxy + c;
            double f = m.tsv_frac[id];
            kxy[id] = eff_kxy(L.k, m.tsv_k, f);
            kz[id] = eff_kz(L.k, m.tsv_k, f);
            m.cap[id] = (float)((f * m.tsv_rho_c + (1 - f) * L.rho_c) * A * m.dz[l]);
        }
    }
    m.gE.assign(n, 0.0f);
    m.gN.assign(n, 0.0f);
    m.gU.assign(n, 0.0f);
    m.gB.assign(sxy, 0.0f);
    for (int l = 0; l < m.nz; ++l)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                size_t id = m.idx(i, j, l);
                double dz = m.dz[l];
                if (i < m.nx - 1)
                    m.gE[id] = (float)(m.dy * dz / (m.dx / (2 * kxy[id]) + m.dx / (2 * kxy[id + 1])));
                if (j < m.ny - 1)
                    m.gN[id] = (float)(m.dx * dz / (m.dy / (2 * kxy[id]) + m.dy / (2 * kxy[id + m.nx])));
                double R_up = (l < m.nz - 1)
                                  ? dz / (2 * kz[id]) + m.dz[l + 1] / (2 * kz[id + sxy])
                                  : dz / (2 * kz[id]) + 1.0 / m.h_top;
                m.gU[id] = (float)(A / R_up);
                if (l == 0 && m.h_bot > 0)
                    m.gB[id] = (float)(A / (dz / (2 * kz[id]) + 1.0 / m.h_bot));
            }
    m.finalized = true;
}

// A two-tier 3D IC on a 10 mm x 10 mm footprint, listed bottom to top.
// The bond layer between the tiers is the thermal bottleneck of 3D stacking:
// the bottom tier's heat has to cross it to reach the heat sink.
inline std::vector<Layer> default_stack() {
    return {
    //  name                 thick    k      rho*c   planes active
        {"Si tier 2 (bottom)", 50e-6, 130.0, 1.66e6, 2, true},
        {"bond + BEOL",        10e-6,   2.0, 1.70e6, 1, false},
        {"Si tier 1 (top)",    50e-6, 130.0, 1.66e6, 2, true},
        {"TIM",                25e-6,   5.0, 2.00e6, 1, false},
        {"Cu spreader",         1e-3, 400.0, 3.45e6, 4, false},
    };
}

// Not finalized: add TSVs if needed, then call finalize().
inline Model3D default_model(int nx, int ny) {
    // h_top = 2e4 W/m^2K on 1 cm^2 is a 0.5 K/W heat sink. Bottom is adiabatic.
    return build_model(default_stack(), nx, ny, 10e-3, 10e-3, 2e4, 0.0);
}

// Overlap length of intervals [a0, a1] and [b0, b1].
inline double overlap(double a0, double a1, double b0, double b1) {
    return std::max(0.0, std::min(a1, b1) - std::max(a0, b0));
}

// Calls fn(cell index within a plane, covered fraction of that cell) for every
// cell overlapping the rectangle [x0, x0 + w] x [y0, y0 + h] (m).
template <class Fn>
void for_each_overlap(const Model3D& m, double x0, double y0, double w, double h, Fn fn) {
    int i0 = std::max(0, (int)std::floor(x0 / m.dx));
    int i1 = std::min(m.nx - 1, (int)std::ceil((x0 + w) / m.dx));
    int j0 = std::max(0, (int)std::floor(y0 / m.dy));
    int j1 = std::min(m.ny - 1, (int)std::ceil((y0 + h) / m.dy));
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i) {
            double a = overlap(i * m.dx, (i + 1) * m.dx, x0, x0 + w) *
                       overlap(j * m.dy, (j + 1) * m.dy, y0, y0 + h);
            // Rounding gives a cell that only touches the edge a tiny positive
            // overlap (~1e-16 of a cell); don't count it as covered.
            if (a > 1e-9 * m.dx * m.dy) fn((size_t)j * m.nx + i, a / (m.dx * m.dy));
        }
}

// Power block: `watts` spread uniformly over the rectangle in active layer
// `layer` (all planes of that layer), area-weighted so partially covered cells
// get their share. Total power is conserved exactly when the block is on the die.
inline void add_power_block(Model3D& m, int layer, double x0, double y0, double w, double h,
                            double watts) {
    const double density = watts / (w * h);  // W per m^2 of footprint
    const int planes = m.layers[layer].ncells;
    for (int l = 0; l < m.nz; ++l) {
        if (m.plane_layer[l] != layer) continue;
        for_each_overlap(m, x0, y0, w, h, [&](size_t c, double frac) {
            m.P[(size_t)l * m.nxy() + c] += density * frac * m.dx * m.dy / planes;
        });
    }
}

// TSV region with copper area fraction f in layer `layer`. Partially covered
// cells get a proportionally smaller fraction. Call finalize() afterwards.
inline void add_tsv_region(Model3D& m, int layer, double x0, double y0, double w, double h,
                           double f) {
    for (int l = 0; l < m.nz; ++l) {
        if (m.plane_layer[l] != layer) continue;
        for_each_overlap(m, x0, y0, w, h, [&](size_t c, double frac) {
            double& t = m.tsv_frac[(size_t)l * m.nxy() + c];
            t = std::min(1.0, t + f * frac);
        });
    }
    m.finalized = false;
}

// Spreads `watts` evenly over every cell of active layer `layer`.
inline void add_uniform_power(Model3D& m, int layer, double watts) {
    add_power_block(m, layer, 0, 0, m.Lx, m.Ly, watts);
}

// Adds `watts` in a square hotspot of side `size` (m) centered at (xc, yc),
// over the cells whose centers fall inside it (kept from earlier milestones so
// results stay comparable).
inline void add_hotspot(Model3D& m, int layer, double xc, double yc, double size,
                        double watts) {
    std::vector<size_t> cells;
    for (int l = 0; l < m.nz; ++l) {
        if (m.plane_layer[l] != layer) continue;
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                double x = (i + 0.5) * m.dx, y = (j + 0.5) * m.dy;
                if (std::fabs(x - xc) <= size / 2 && std::fabs(y - yc) <= size / 2)
                    cells.push_back(m.idx(i, j, l));
            }
    }
    if (cells.empty()) { fprintf(stderr, "Hotspot covers no cells\n"); exit(1); }
    for (size_t c : cells) m.P[c] += watts / (double)cells.size();
}

inline double total_power(const Model3D& m) {
    double s = 0;
    for (double p : m.P) s += p;
    return s;
}

// Heat leaving through the top (heat sink) and bottom boundaries, in W.
// At steady state this must equal the total power: an energy-balance check.
template <class Vec>
double heat_out(const Model3D& m, const Vec& T) {
    double q = 0;
    const size_t top = (size_t)(m.nz - 1) * m.nxy();
    for (size_t c = 0; c < m.nxy(); ++c) {
        q += (double)m.gU[top + c] * (double)T[top + c];
        q += (double)m.gB[c] * (double)T[c];
    }
    return q;
}

// Rate at which heat is being stored, sum of C (T_new - T_old) / dt, in W.
// In a transient step: stored + heat_out = power in (energy balance).
template <class Vec>
double heat_stored(const Model3D& m, const Vec& T_old, const Vec& T_new, double dt) {
    double q = 0;
    for (size_t c = 0; c < m.ncells(); ++c)
        q += (double)m.cap[c] * ((double)T_new[c] - (double)T_old[c]);
    return q / dt;
}

// Exact solution when power is uniform in x and y, the bottom is adiabatic and
// there are no TSVs: no lateral flow, so every column is a 1D resistor chain.
// Returns the temperature rise of each plane. Computed in double from the layer
// data, independently of the (float) operator.
inline std::vector<double> analytic_1d(const Model3D& m) {
    std::vector<double> flux(m.nz), T(m.nz), gz(m.nz);
    const double A = m.dx * m.dy;
    for (int l = 0; l < m.nz; ++l) {
        double k = m.layers[m.plane_layer[l]].k;
        double R = (l < m.nz - 1)
                       ? m.dz[l] / (2 * k) + m.dz[l + 1] / (2 * m.layers[m.plane_layer[l + 1]].k)
                       : m.dz[l] / (2 * k) + 1.0 / m.h_top;
        gz[l] = A / R;
    }
    double F = 0;
    for (int l = 0; l < m.nz; ++l) {
        for (size_t c = 0; c < m.nxy(); ++c) F += m.P[(size_t)l * m.nxy() + c];
        flux[l] = F;  // heat crossing the top face of plane l
    }
    const double nxy = (double)m.nxy();
    T[m.nz - 1] = flux[m.nz - 1] / (gz[m.nz - 1] * nxy);
    for (int l = m.nz - 2; l >= 0; --l) T[l] = T[l + 1] + flux[l] / (gz[l] * nxy);
    return T;
}
