// stack3d.h - 3D layered chip stack: geometry, conductances, power maps.
//
// Finite-volume model on a cell-centered grid. Laterally the die is split into
// nx * ny cells of size dx * dy. Vertically each layer is split into ncells
// planes, so planes in different layers can have very different thicknesses.
//
// Each cell exchanges heat with its 6 neighbors through a thermal conductance
// G (W/K). Between planes l and l+1 the two half-cells are in series:
//     G = A / ( dz_l / (2 k_l) + dz_{l+1} / (2 k_{l+1}) )
// which is the harmonic-mean rule for k at a material interface. The top plane
// connects to ambient through half a cell plus the heat sink (1 / h_top), the
// bottom plane through half a cell plus 1 / h_bot (h_bot = 0 means adiabatic).
// The side walls are adiabatic.
//
// Temperatures are solved relative to ambient: the energy balance of cell c is
//     sum_n G_n (T_c - T_n) + G_amb T_c = P_c
// which is a symmetric positive definite linear system A T = P.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

constexpr int MAXZ = 64;  // max number of z-planes (GPU keeps them in constant memory)

struct Layer {
    std::string name;
    double thickness;  // m
    double k;          // W/(m K)
    int ncells;        // number of z-planes in this layer
    bool active;       // dissipates power
};

struct Model3D {
    int nx = 0, ny = 0, nz = 0;
    double Lx = 0, Ly = 0, dx = 0, dy = 0;  // m
    double h_top = 0, h_bot = 0;            // W/(m^2 K)
    double T_amb = 25.0;                    // deg C
    std::vector<Layer> layers;

    // Per z-plane data.
    std::vector<int> plane_layer;  // which layer each plane belongs to
    std::vector<double> dz, k;
    std::vector<double> gx, gy;    // lateral conductance between neighbor cells (W/K)
    std::vector<double> gzd, gzu;  // conductance to plane below / above (or ambient)

    std::vector<double> P;         // power per cell (W), nx * ny * nz

    size_t nxy() const { return (size_t)nx * ny; }
    size_t ncells() const { return nxy() * nz; }
    size_t idx(int i, int j, int l) const { return ((size_t)l * ny + j) * nx + i; }
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
            m.k.push_back(layers[L].k);
        }
    m.nz = (int)m.dz.size();
    if (m.nz > MAXZ) {
        fprintf(stderr, "Too many z-planes (%d > %d)\n", m.nz, MAXZ);
        exit(1);
    }

    const double A = m.dx * m.dy;
    m.gx.resize(m.nz); m.gy.resize(m.nz);
    m.gzd.resize(m.nz); m.gzu.resize(m.nz);
    for (int l = 0; l < m.nz; ++l) {
        m.gx[l] = m.k[l] * m.dy * m.dz[l] / m.dx;
        m.gy[l] = m.k[l] * m.dx * m.dz[l] / m.dy;
        double half = m.dz[l] / (2.0 * m.k[l]);
        double R_up = (l < m.nz - 1) ? half + m.dz[l + 1] / (2.0 * m.k[l + 1])
                                     : half + 1.0 / h_top;
        m.gzu[l] = A / R_up;
    }
    for (int l = 0; l < m.nz; ++l) {
        if (l > 0) m.gzd[l] = m.gzu[l - 1];
        else m.gzd[l] = (h_bot > 0) ? A / (m.dz[0] / (2.0 * m.k[0]) + 1.0 / h_bot) : 0.0;
    }
    m.P.assign(m.ncells(), 0.0);
    return m;
}

// A two-tier 3D IC on a 10 mm x 10 mm footprint, listed bottom to top.
// The bond layer between the tiers is the thermal bottleneck of 3D stacking:
// the bottom tier's heat has to cross it to reach the heat sink.
inline std::vector<Layer> default_stack() {
    return {
        {"Si tier 2 (bottom)", 50e-6, 130.0, 2, true},
        {"bond + BEOL",        10e-6,   2.0, 1, false},
        {"Si tier 1 (top)",    50e-6, 130.0, 2, true},
        {"TIM",                25e-6,   5.0, 1, false},
        {"Cu spreader",         1e-3, 400.0, 4, false},
    };
}

inline Model3D default_model(int nx, int ny) {
    // h_top = 2e4 W/m^2K on 1 cm^2 is a 0.5 K/W heat sink. Bottom is adiabatic.
    return build_model(default_stack(), nx, ny, 10e-3, 10e-3, 2e4, 0.0);
}

// Spreads `watts` evenly over every cell of active layer `layer`.
inline void add_uniform_power(Model3D& m, int layer, double watts) {
    int planes = m.layers[layer].ncells;
    double w = watts / ((double)m.nxy() * planes);
    for (int l = 0; l < m.nz; ++l)
        if (m.plane_layer[l] == layer)
            for (size_t c = 0; c < m.nxy(); ++c) m.P[(size_t)l * m.nxy() + c] += w;
}

// Adds `watts` in a square hotspot of side `size` (m) centered at (xc, yc).
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
    for (size_t c = 0; c < m.nxy(); ++c) {
        q += m.gzu[m.nz - 1] * (double)T[(size_t)(m.nz - 1) * m.nxy() + c];
        q += m.gzd[0] * (double)T[c];
    }
    return q;
}

// Exact solution of the discrete model when power is uniform in x and y and the
// bottom is adiabatic: no lateral flow, so every column is a 1D resistor chain.
// Returns the temperature rise of each plane.
inline std::vector<double> analytic_1d(const Model3D& m) {
    std::vector<double> flux(m.nz), T(m.nz);
    double F = 0;
    for (int l = 0; l < m.nz; ++l) {
        for (size_t c = 0; c < m.nxy(); ++c) F += m.P[(size_t)l * m.nxy() + c];
        flux[l] = F;  // heat crossing the top face of plane l
    }
    const double nxy = (double)m.nxy();
    T[m.nz - 1] = flux[m.nz - 1] / (m.gzu[m.nz - 1] * nxy);
    for (int l = m.nz - 2; l >= 0; --l) T[l] = T[l + 1] + flux[l] / (m.gzu[l] * nxy);
    return T;
}
