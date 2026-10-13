// case3d.h - load a chip stack, floorplan and TSV layout from CSV files.
//
// A case is a folder with:
//   stack.csv      layers, bottom to top:  name, thickness_um, k, rho_c, planes, active
//                  plus optional settings lines:
//                    # die_mm, 10, 10       # h_top, 2e4      (W/m^2K, heat sink)
//                    # h_bottom, 0          # ambient_C, 25
//   floorplan.csv  power blocks:  name, layer, x_mm, y_mm, w_mm, h_mm, power_W
//                  (x, y = lower-left corner; layer = a name from stack.csv,
//                   which must be active)
//   tsv.csv        optional TSV regions:  layer, x_mm, y_mm, w_mm, h_mm, cu_fraction
// The first line of each file is a header and is skipped. Lines starting with
// '#' are comments (or settings, in stack.csv).
#pragma once
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "stack3d.h"

struct PowerBlock {
    std::string name;
    int layer;
    double x0, y0, w, h, watts;  // m, W
};

struct TsvRegion {
    int layer;
    double x0, y0, w, h, fraction;  // m
};

struct CaseSpec {
    std::vector<Layer> layers;
    double Lx = 10e-3, Ly = 10e-3, h_top = 2e4, h_bot = 0.0, T_amb = 25.0;
    std::vector<PowerBlock> blocks;
    std::vector<TsvRegion> tsvs;
};

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n\"");
    size_t b = s.find_last_not_of(" \t\r\n\"");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(trim(item));
    return out;
}

// Reads the data rows of a CSV file (header skipped). Comment / settings lines
// starting with '#' go to `settings` with the '#' removed. Returns false if the
// file can't be opened.
inline bool read_csv(const std::string& path, std::vector<std::vector<std::string>>& rows,
                     std::vector<std::vector<std::string>>* settings = nullptr) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    bool header_done = false;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;
        if (t[0] == '#') {
            if (settings) settings->push_back(split_csv(trim(t.substr(1))));
            continue;
        }
        if (!header_done) { header_done = true; continue; }
        rows.push_back(split_csv(t));
    }
    return true;
}

inline int find_layer(const std::vector<Layer>& layers, const std::string& name) {
    for (size_t L = 0; L < layers.size(); ++L)
        if (layers[L].name == name) return (int)L;
    return -1;
}

// Loads dir/stack.csv, dir/floorplan.csv and (if present) dir/tsv.csv.
// On error returns false with a message in `err`.
inline bool load_case(const std::string& dir, CaseSpec& cs, std::string& err) {
    std::vector<std::vector<std::string>> rows, settings;
    try {
        if (!read_csv(dir + "/stack.csv", rows, &settings)) {
            err = "cannot open " + dir + "/stack.csv";
            return false;
        }
        for (auto& s : settings) {
            if (s.empty()) continue;
            if (s[0] == "die_mm" && s.size() >= 3) {
                cs.Lx = std::stod(s[1]) * 1e-3;
                cs.Ly = std::stod(s[2]) * 1e-3;
            } else if (s[0] == "h_top" && s.size() >= 2) {
                cs.h_top = std::stod(s[1]);
            } else if (s[0] == "h_bottom" && s.size() >= 2) {
                cs.h_bot = std::stod(s[1]);
            } else if (s[0] == "ambient_C" && s.size() >= 2) {
                cs.T_amb = std::stod(s[1]);
            }
        }
        for (auto& r : rows) {
            if (r.size() < 6) { err = "stack.csv: expected 6 columns"; return false; }
            cs.layers.push_back({r[0], std::stod(r[1]) * 1e-6, std::stod(r[2]), std::stod(r[3]),
                                 std::stoi(r[4]), std::stoi(r[5]) != 0});
        }
        if (cs.layers.empty()) { err = "stack.csv has no layers"; return false; }

        rows.clear();
        if (!read_csv(dir + "/floorplan.csv", rows)) {
            err = "cannot open " + dir + "/floorplan.csv";
            return false;
        }
        for (auto& r : rows) {
            if (r.size() < 7) { err = "floorplan.csv: expected 7 columns"; return false; }
            int L = find_layer(cs.layers, r[1]);
            if (L < 0) { err = "floorplan.csv: unknown layer '" + r[1] + "'"; return false; }
            if (!cs.layers[L].active) {
                err = "floorplan.csv: layer '" + r[1] + "' is not active";
                return false;
            }
            PowerBlock b{r[0], L, std::stod(r[2]) * 1e-3, std::stod(r[3]) * 1e-3,
                         std::stod(r[4]) * 1e-3, std::stod(r[5]) * 1e-3, std::stod(r[6])};
            if (b.x0 < -1e-12 || b.y0 < -1e-12 || b.x0 + b.w > cs.Lx * (1 + 1e-9) ||
                b.y0 + b.h > cs.Ly * (1 + 1e-9)) {
                err = "floorplan.csv: block '" + b.name + "' extends outside the die";
                return false;
            }
            cs.blocks.push_back(b);
        }

        rows.clear();
        if (read_csv(dir + "/tsv.csv", rows))  // optional
            for (auto& r : rows) {
                if (r.size() < 6) { err = "tsv.csv: expected 6 columns"; return false; }
                int L = find_layer(cs.layers, r[0]);
                if (L < 0) { err = "tsv.csv: unknown layer '" + r[0] + "'"; return false; }
                TsvRegion t{L, std::stod(r[1]) * 1e-3, std::stod(r[2]) * 1e-3,
                            std::stod(r[3]) * 1e-3, std::stod(r[4]) * 1e-3, std::stod(r[5])};
                if (t.fraction < 0 || t.fraction > 1) {
                    err = "tsv.csv: cu_fraction must be between 0 and 1";
                    return false;
                }
                cs.tsvs.push_back(t);
            }
    } catch (const std::exception& e) {
        err = std::string("could not parse a number: ") + e.what();
        return false;
    }
    return true;
}

// Builds a finalized model on an nx x ny grid, with or without the TSVs.
inline Model3D build_case_model(const CaseSpec& cs, int nx, int ny, bool with_tsv) {
    Model3D m = build_model(cs.layers, nx, ny, cs.Lx, cs.Ly, cs.h_top, cs.h_bot);
    m.T_amb = cs.T_amb;
    for (const PowerBlock& b : cs.blocks) add_power_block(m, b.layer, b.x0, b.y0, b.w, b.h, b.watts);
    if (with_tsv)
        for (const TsvRegion& t : cs.tsvs) add_tsv_region(m, t.layer, t.x0, t.y0, t.w, t.h, t.fraction);
    finalize(m);
    return m;
}

// The example in cases/example, built in code for the self-checks: a logic die
// (4 cores, cache, I/O) under a memory die, with TSV fields beside the cores.
inline CaseSpec example_case() {
    CaseSpec cs;
    cs.layers = default_stack();
    auto L = [&](const char* n) { return find_layer(cs.layers, n); };
    const double mm = 1e-3;
    cs.blocks = {
        {"core0", L("Si tier 2 (bottom)"), 0.5 * mm, 0.5 * mm, 2.0 * mm, 2.0 * mm, 6.0},
        {"core1", L("Si tier 2 (bottom)"), 3.0 * mm, 0.5 * mm, 2.0 * mm, 2.0 * mm, 6.0},
        {"core2", L("Si tier 2 (bottom)"), 0.5 * mm, 3.0 * mm, 2.0 * mm, 2.0 * mm, 6.0},
        {"core3", L("Si tier 2 (bottom)"), 3.0 * mm, 3.0 * mm, 2.0 * mm, 2.0 * mm, 6.0},
        {"l2_cache", L("Si tier 2 (bottom)"), 5.5 * mm, 0.5 * mm, 4.0 * mm, 4.5 * mm, 3.0},
        {"noc_io", L("Si tier 2 (bottom)"), 0.5 * mm, 5.5 * mm, 9.0 * mm, 4.0 * mm, 3.0},
        {"dram_banks", L("Si tier 1 (top)"), 0.0 * mm, 0.0 * mm, 10.0 * mm, 10.0 * mm, 12.0},
        {"dram_ctrl", L("Si tier 1 (top)"), 4.0 * mm, 4.0 * mm, 2.0 * mm, 2.0 * mm, 3.0},
    };
    for (const char* layer : {"bond + BEOL", "Si tier 1 (top)"}) {
        cs.tsvs.push_back({L(layer), 2.55 * mm, 0.5 * mm, 0.40 * mm, 4.5 * mm, 0.15});  // between cores
        cs.tsvs.push_back({L(layer), 0.5 * mm, 2.55 * mm, 4.5 * mm, 0.40 * mm, 0.15});
    }
    return cs;
}

// ---------------------------------------------------------------------------
// Export to 3D-ICE (EPFL, https://github.com/esl-epfl/3d-ice) for validation.
//
// Writes <outdir>/case.stk and one floorplan file per active layer, describing
// the same stack on the same lateral grid: every plane of our model becomes one
// 3D-ICE layer, and every active layer one die with a single source layer.
// Units: 3D-ICE uses um, W/(um K), J/(um^3 K), W/(um^2 K) and kelvin.
// Requirements (checked): active layers have 1 plane (3D-ICE puts a die's power
// in one source layer), TSVs only in non-active layers, no bottom heat sink,
// no overlapping power blocks within a layer, and the cell size divides the
// die size.
//
// TSVs: 3D-ICE 4.0 lets a layer have a "layout" file listing rectangles of
// other materials (but not a die's source layer, hence the restriction above).
// Each distinct TSV fraction in a plane becomes one anisotropic material (the
// same effective kxy, kz and rho*c that our model uses), and its cells are
// written as rectangles, one per run of equal cells in a row. So both tools see
// exactly the same per-cell materials, including partially covered cells.
//
// Known modeling difference: 3D-ICE couples the bottom layer of the stack
// to the layer above with its full height (its node sits at the adiabatic
// bottom face) instead of half its height. To compare like with like, give
// the stack a thin, unpowered bottom layer.

inline std::string ice_name(const std::string& s) {
    std::string out;
    for (char c : s) out += (std::isalnum((unsigned char)c) ? c : '_');
    return out;
}

// Writes the 3D-ICE layout file for plane l: one material per distinct TSV
// fraction, and its cells as rectangles (runs of equal cells along each row).
inline bool export_3dice_layout(const Model3D& m, int l, const Layer& ly, double cell_um,
                                const std::string& path, std::string& err) {
    auto key = [&](int i, int j) {  // TSV fraction, rounded so equal fractions group together
        return std::llround(m.tsv_frac[(size_t)l * m.nxy() + (size_t)j * m.nx + i] * 1e9);
    };
    std::map<long long, std::vector<std::string>> rects;  // fraction key -> rectangles
    char buf[160];
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx;) {
            long long k = key(i, j);
            int i1 = i + 1;
            while (i1 < m.nx && key(i1, j) == k) ++i1;
            if (k != 0) {
                snprintf(buf, sizeof buf, "   rectangle (%.6f, %.6f, %.6f, %.6f) ;\n",
                         i * cell_um, j * cell_um, (i1 - i) * cell_um, cell_um);
                rects[k].push_back(buf);
            }
            i = i1;
        }
    FILE* g = fopen(path.c_str(), "w");
    if (!g) { err = "cannot write " + path; return false; }
    fprintf(g, "// TSV layout for %s (generated by thermal3d export3dice)\n\n", ly.name.c_str());
    int n = 0;
    for (const auto& kv : rects) {
        double fr = kv.first * 1e-9;
        fprintf(g, "material TSV%d :   // copper fraction %.6f\n"
                   "   thermal conductivity     %.10e, %.10e, %.10e ;\n"
                   "   volumetric heat capacity %.10e ;\n\n", n, fr,
                eff_kxy(ly.k, m.tsv_k, fr) * 1e-6, eff_kxy(ly.k, m.tsv_k, fr) * 1e-6,
                eff_kz(ly.k, m.tsv_k, fr) * 1e-6,
                (fr * m.tsv_rho_c + (1 - fr) * ly.rho_c) * 1e-18);
        ++n;
    }
    n = 0;
    for (const auto& kv : rects) {
        fprintf(g, "TSV%d :\n", n++);
        for (const std::string& r : kv.second) fputs(r.c_str(), g);
        fprintf(g, "\n");
    }
    fclose(g);
    return true;
}

inline bool export_3dice(const CaseSpec& cs, double cell_um, const std::string& outdir,
                         std::string& err) {
    const double Lx_um = cs.Lx * 1e6, Ly_um = cs.Ly * 1e6;
    if (std::fabs(Lx_um / cell_um - std::round(Lx_um / cell_um)) > 1e-9 ||
        std::fabs(Ly_um / cell_um - std::round(Ly_um / cell_um)) > 1e-9) {
        err = "cell size must divide the die size";
        return false;
    }
    const int nx = (int)std::lround(Lx_um / cell_um), ny = (int)std::lround(Ly_um / cell_um);
    const Model3D m = build_case_model(cs, nx, ny, true);  // per-cell TSV fractions on this grid
    std::vector<int> first_plane(cs.layers.size(), 0);
    for (int l = m.nz - 1; l >= 0; --l) first_plane[m.plane_layer[l]] = l;
    auto plane_has_tsv = [&](int l) {
        for (size_t c = 0; c < m.nxy(); ++c)
            if (m.tsv_frac[(size_t)l * m.nxy() + c] > 0) return true;
        return false;
    };
    for (size_t L = 0; L < cs.layers.size(); ++L)
        if (cs.layers[L].active && plane_has_tsv(first_plane[L])) {
            err = "TSVs in active layer '" + cs.layers[L].name +
                  "' can't be exported (3D-ICE has no material layouts in source layers)";
            return false;
        }
    if (cs.h_bot > 0) { err = "a bottom heat sink is not exported"; return false; }
    for (size_t L = 0; L < cs.layers.size(); ++L)
        if (cs.layers[L].active && cs.layers[L].ncells != 1) {
            err = "active layer '" + cs.layers[L].name + "' must have 1 plane for 3D-ICE";
            return false;
        }
    for (size_t a = 0; a < cs.blocks.size(); ++a)
        for (size_t b = a + 1; b < cs.blocks.size(); ++b) {
            const PowerBlock &p = cs.blocks[a], &q = cs.blocks[b];
            if (p.layer == q.layer &&
                overlap(p.x0, p.x0 + p.w, q.x0, q.x0 + q.w) * overlap(p.y0, p.y0 + p.h, q.y0, q.y0 + q.h) > 0) {
                err = "power blocks '" + p.name + "' and '" + q.name + "' overlap (3D-ICE needs disjoint blocks)";
                return false;
            }
        }

    FILE* f = fopen((outdir + "/case.stk").c_str(), "w");
    if (!f) { err = "cannot write " + outdir + "/case.stk (does the folder exist?)"; return false; }
    fprintf(f, "// Generated by thermal3d export3dice. Units: um, W/(um K), J/(um^3 K), K.\n\n");
    for (size_t L = 0; L < cs.layers.size(); ++L)
        fprintf(f, "material M%zu :   // %s\n   thermal conductivity     %.10e ;\n"
                   "   volumetric heat capacity %.10e ;\n\n",
                L, cs.layers[L].name.c_str(), cs.layers[L].k * 1e-6, cs.layers[L].rho_c * 1e-18);
    fprintf(f, "top heat sink :\n   heat transfer coefficient %.10e ;\n   temperature %.4f ;\n\n",
            cs.h_top * 1e-12, cs.T_amb + 273.15);
    fprintf(f, "dimensions :\n   chip length %.6f, width %.6f ;\n   cell length %.6f, width %.6f ;\n"
               "   non-uniform false ;\n\n", Lx_um, Ly_um, cell_um, cell_um);

    // One 3D-ICE layer type per plane, then one die per active layer (3D-ICE
    // wants all layer definitions before the die definitions).
    struct Elem { bool die; size_t layer; int plane; };
    std::vector<Elem> elems;  // bottom to top
    for (size_t L = 0; L < cs.layers.size(); ++L) {
        const Layer& ly = cs.layers[L];
        if (ly.active) {
            elems.push_back({true, L, 0});
            continue;
        }
        for (int p = 0; p < ly.ncells; ++p) {
            fprintf(f, "layer LT%zu_%d :   // %s, plane %d\n   height %.6f ;\n   material M%zu ;\n",
                    L, p, ly.name.c_str(), p, ly.thickness * 1e6 / ly.ncells, L);
            const int l = first_plane[L] + p;
            if (plane_has_tsv(l)) {
                std::string lay = "LT" + std::to_string(L) + "_" + std::to_string(p) + ".lay";
                if (!export_3dice_layout(m, l, ly, cell_um, outdir + "/" + lay, err)) {
                    fclose(f);
                    return false;
                }
                fprintf(f, "   layout \"./%s\" ;\n", lay.c_str());
            }
            fprintf(f, "\n");
            elems.push_back({false, L, p});
        }
    }
    for (size_t L = 0; L < cs.layers.size(); ++L)
        if (cs.layers[L].active)
            fprintf(f, "die DIE%zu :   // %s\n   source %.6f M%zu ;\n\n", L,
                    cs.layers[L].name.c_str(), cs.layers[L].thickness * 1e6, L);
    fprintf(f, "stack:\n");
    for (size_t e = elems.size(); e-- > 0;) {  // 3D-ICE lists the stack top to bottom
        const Elem& el = elems[e];
        if (el.die)
            fprintf(f, "   die   D%zu DIE%zu floorplan \"./L%zu.flp\" ;\n", el.layer, el.layer, el.layer);
        else
            fprintf(f, "   layer P%zu_%d LT%zu_%d ;\n", el.layer, el.plane, el.layer, el.plane);
    }
    fprintf(f, "\nsolver:\n   steady ;\n   initial temperature %.4f ;\n   numofcores 1 ;\n\noutput:\n",
            cs.T_amb + 273.15);
    for (size_t L = 0; L < cs.layers.size(); ++L) {
        if (!cs.layers[L].active) continue;
        fprintf(f, "   Tmap ( D%zu, \"tmap_L%zu.txt\", final ) ;\n", L, L);
        fprintf(f, "   Tflp ( D%zu, \"tflp_max_L%zu.txt\", maximum, final ) ;\n", L, L);
        fprintf(f, "   Tflp ( D%zu, \"tflp_avg_L%zu.txt\", average, final ) ;\n", L, L);
    }
    fclose(f);

    for (size_t L = 0; L < cs.layers.size(); ++L) {
        if (!cs.layers[L].active) continue;
        FILE* g = fopen((outdir + "/L" + std::to_string(L) + ".flp").c_str(), "w");
        if (!g) { err = "cannot write floorplan file"; return false; }
        int count = 0;
        for (const PowerBlock& b : cs.blocks) {
            if ((size_t)b.layer != L) continue;
            fprintf(g, "%s :\n   position %.6f, %.6f ;\n   dimension %.6f, %.6f ;\n   power values %.10e ;\n\n",
                    ice_name(b.name).c_str(), b.x0 * 1e6, b.y0 * 1e6, b.w * 1e6, b.h * 1e6, b.watts);
            ++count;
        }
        if (count == 0)  // 3D-ICE needs at least one element per die
            fprintf(g, "none :\n   position 0, 0 ;\n   dimension %.6f, %.6f ;\n   power values 0 ;\n\n",
                    Lx_um, Ly_um);
        fclose(g);
    }
    return true;
}

// Reads a 3D-ICE Tmap file (lines starting with '%' are comments) into a
// row-major grid. Returns false if it can't be opened.
inline bool read_3dice_tmap(const std::string& path, std::vector<std::vector<double>>& rows) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::stringstream ss(line);
        std::vector<double> r;
        double v;
        while (ss >> v) r.push_back(v);
        if (!r.empty()) rows.push_back(r);
    }
    return true;
}
