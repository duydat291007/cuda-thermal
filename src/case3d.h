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
#include <fstream>
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
