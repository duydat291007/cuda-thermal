// viewer.cu - live view: watch the chip heat up while you switch blocks on and off.
//
//   thermal3d_view [case folder] [n]     (default: the built-in example, 256 x 256)
//
// Every frame runs one backward Euler time step on the GPU (multigrid PCG,
// warm-started), then a kernel turns the temperature field into colors and
// finds each die's hottest cell, and CUDA-OpenGL interop copies the colors
// straight into the OpenGL textures that are drawn: the temperatures never
// leave the GPU. If interop is unavailable (the window ended up on the
// integrated GPU), the colors go through the host instead.
//
// The active dies are shown side by side, bottom die on the left (up to two at
// a time; up / down scroll if a case has more). A cross marks each die's
// hottest cell; the triangles on the color bar show each die's maximum, in the
// color of that die's frame.
//
// Controls:
//   1 - 9        switch power block 1 - 9 on / off (listed in the console)
//   left click   add a 2 W hotspot (0.5 mm square) on the die you click
//   right click  remove the hotspots on the die you click
//   up / down    scroll through the dies (cases with more than two)
//   + / -        simulated time runs faster / slower
//   space        pause      R  reset to ambient (t = 0)      Esc  quit
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// Ask laptop drivers to run this program (and its OpenGL context) on the NVIDIA
// GPU, so CUDA and OpenGL share one device and interop works.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
}
#endif
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "case3d.h"
#include "cuda_check.h"
#include "pcg3d.h"

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F  // OpenGL 1.2; Windows' gl.h stops at 1.1
#endif

// ---- colors -----------------------------------------------------------------

// Black -> purple -> red -> orange -> pale yellow (close to "inferno").
HD inline uchar4 colormap(float s) {
    const float stops[5][3] = {{0, 0, 4}, {87, 16, 110}, {188, 55, 84}, {249, 142, 9}, {252, 255, 164}};
    s = s < 0 ? 0 : (s > 1 ? 1 : s);
    float f = s * 4;
    int k = (int)f;
    if (k > 3) k = 3;
    f -= k;
    uchar4 c;
    c.x = (unsigned char)(stops[k][0] + f * (stops[k + 1][0] - stops[k][0]));
    c.y = (unsigned char)(stops[k][1] + f * (stops[k + 1][1] - stops[k][1]));
    c.z = (unsigned char)(stops[k][2] + f * (stops[k + 1][2] - stops[k][2]));
    c.w = 255;
    return c;
}

// Die map: the hottest of layer planes [l0, l1] at each (i, j), as a color on
// the scale [0, rise_hi] (temperature rise above ambient). Also finds the die's
// hottest cell: key = (float bits << 32) | cell, so one 64-bit max gives both
// the value and where it is (non-negative floats order like unsigned ints).
__global__ void k_colorize(const Real* __restrict__ T, int nx, int ny, int l0, int l1,
                           float rise_hi, uchar4* out, unsigned long long* max_key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float v = 0.0f;
    unsigned long long key = 0;
    if (i < nx && j < ny) {
        for (int l = l0; l <= l1; ++l)
            v = fmaxf(v, (float)T[((size_t)l * ny + j) * nx + i]);
        out[(size_t)j * nx + i] = colormap(v / rise_hi);
        key = ((unsigned long long)__float_as_uint(v) << 32) | (unsigned)(j * nx + i);
    }
    for (int off = 16; off > 0; off >>= 1)  // warp max, then one atomic per warp
        key = max(key, __shfl_down_sync(0xffffffffu, key, off));
    if ((threadIdx.x + threadIdx.y * blockDim.x) % 32 == 0) atomicMax(max_key, key);
}

// ---- the live simulation ----------------------------------------------------

constexpr int PANELS = 2;  // dies shown at once

struct Panel {               // one die on screen
    GLuint tex = 0;
    cudaGraphicsResource* res = nullptr;
    double x = 0, y = 0, size = 1;  // in framebuffer pixels, origin bottom left
    float frame[3];                  // frame and color-bar marker color
    float max_rise = 0;              // hottest cell (K above ambient)
    int max_i = 0, max_j = 0;
};

struct App {
    CaseSpec cs;
    Model3D m;
    GpuPcg* solver = nullptr;
    std::vector<bool> on;               // per floorplan block
    std::vector<PowerBlock> hotspots;   // added with the mouse
    std::vector<int> dies;              // active layers, bottom to top
    int first = 0;                      // first die shown (left panel)
    Panel panel[PANELS];
    double t = 0, speed = 0.25;         // simulated s, simulated s per real s
    bool paused = false;
    float rise_hi = 1;                  // color scale top (K above ambient)
};
static App app;

static int shown() { return std::min(PANELS, (int)app.dies.size()); }
static int die_of(int p) { return app.dies[(app.first + p) % app.dies.size()]; }

// Rebuilds the power map from the switched-on blocks and hotspots and sends it
// to the GPU. Every change of the power inputs goes through here (keyboard,
// mouse, and later possibly an external controller such as an FPGA board).
static void apply_power() {
    std::fill(app.m.P.begin(), app.m.P.end(), 0.0);
    double total = 0;
    for (size_t b = 0; b < app.cs.blocks.size(); ++b)
        if (app.on[b]) {
            const PowerBlock& p = app.cs.blocks[b];
            add_power_block(app.m, p.layer, p.x0, p.y0, p.w, p.h, p.watts);
            total += p.watts;
        }
    for (const PowerBlock& p : app.hotspots) {
        add_power_block(app.m, p.layer, p.x0, p.y0, p.w, p.h, p.watts);
        total += p.watts;
    }
    app.solver->set_power(app.m.P);
    printf("  power: %.1f W  (", total);
    for (size_t b = 0; b < app.cs.blocks.size(); ++b)
        printf("%s%s %s", b ? ", " : "", app.cs.blocks[b].name.c_str(), app.on[b] ? "on" : "off");
    printf("; %zu hotspots)\n", app.hotspots.size());
}

static void plane_range(int layer, int& l0, int& l1) {
    l0 = -1;
    l1 = -1;
    for (int l = 0; l < app.m.nz; ++l)
        if (app.m.plane_layer[l] == layer) {
            if (l0 < 0) l0 = l;
            l1 = l;
        }
}

static void on_key(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9 && action == GLFW_PRESS) {
        size_t b = (size_t)(key - GLFW_KEY_1);
        if (b < app.on.size()) {
            app.on[b] = !app.on[b];
            apply_power();
        }
    }
    const int nd = (int)app.dies.size();
    if (nd > PANELS && key == GLFW_KEY_UP) app.first = (app.first + 1) % nd;
    if (nd > PANELS && key == GLFW_KEY_DOWN) app.first = (app.first + nd - 1) % nd;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) app.paused = !app.paused;
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        app.solver->set_x_zero();
        app.t = 0;
    }
    if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) app.speed = std::min(app.speed * 2, 16.0);
    if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) app.speed = std::max(app.speed / 2, 1.0 / 256);
}

static void on_mouse(GLFWwindow* w, int button, int action, int) {
    if (action != GLFW_PRESS) return;
    double cx, cy;
    glfwGetCursorPos(w, &cx, &cy);
    // Cursor is in screen coordinates (top left origin); convert to framebuffer pixels.
    int ww, wh, fw, fh;
    glfwGetWindowSize(w, &ww, &wh);
    glfwGetFramebufferSize(w, &fw, &fh);
    const double px = cx * fw / std::max(ww, 1), py = (wh - cy) * fh / std::max(wh, 1);
    for (int p = 0; p < shown(); ++p) {
        const Panel& P = app.panel[p];
        const double u = (px - P.x) / P.size, v = (py - P.y) / P.size;
        if (u < 0 || u > 1 || v < 0 || v > 1) continue;
        const int layer = die_of(p);
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            const double s = 0.5e-3;
            app.hotspots.push_back({"hotspot", layer, u * app.m.Lx - s / 2, v * app.m.Ly - s / 2, s, s, 2.0});
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            app.hotspots.erase(std::remove_if(app.hotspots.begin(), app.hotspots.end(),
                                              [&](const PowerBlock& h) { return h.layer == layer; }),
                               app.hotspots.end());
        }
        apply_power();
    }
}

static void rect(const Panel& P, double x0, double y0, double w, double h) {  // chip m -> px
    const double sx = P.size / app.m.Lx, sy = P.size / app.m.Ly;
    glBegin(GL_LINE_LOOP);
    glVertex2d(P.x + x0 * sx, P.y + y0 * sy);
    glVertex2d(P.x + (x0 + w) * sx, P.y + y0 * sy);
    glVertex2d(P.x + (x0 + w) * sx, P.y + (y0 + h) * sy);
    glVertex2d(P.x + x0 * sx, P.y + (y0 + h) * sy);
    glEnd();
}

int main(int argc, char** argv) {
    // ---- the case ----
    int n = 256;
    if (argc >= 2) {
        std::string err;
        if (!load_case(argv[1], app.cs, err)) {
            printf("Could not load case '%s': %s\n", argv[1], err.c_str());
            return 1;
        }
        if (argc >= 3) n = atoi(argv[2]);
    } else {
        app.cs = example_case();
    }
    app.m = build_case_model(app.cs, n, n, true);
    app.on.assign(app.cs.blocks.size(), true);
    for (size_t L = 0; L < app.cs.layers.size(); ++L)
        if (app.cs.layers[L].active) app.dies.push_back((int)L);
    if (app.dies.empty()) { printf("The case has no active layer.\n"); return 1; }

    printf("thermal3d_view: %d x %d x %d cells. Blocks:\n", app.m.nx, app.m.ny, app.m.nz);
    for (size_t b = 0; b < app.cs.blocks.size(); ++b)
        printf("  [%zu] %-12s %5.1f W  on %s%s\n", b + 1, app.cs.blocks[b].name.c_str(),
               app.cs.blocks[b].watts, app.cs.layers[app.cs.blocks[b].layer].name.c_str(),
               b < 9 ? "" : "   (no key)");

    // Color scale: the hottest steady-state rise with every block on (+10%).
    SolveResult ss = gpu_pcg(app.m, 1e-8, 2000, Precond::Multigrid);
    app.rise_hi = 1.1f * (float)*std::max_element(ss.T.begin(), ss.T.end());
    printf("  steady state, all on: hottest %.1f C; color scale %.0f - %.0f C\n",
           app.m.T_amb + app.rise_hi / 1.1, app.m.T_amb, app.m.T_amb + app.rise_hi);

    GpuPcg solver(app.m, Precond::Multigrid);
    app.solver = &solver;
    solver.set_x_zero();
    apply_power();

    // ---- window and OpenGL ----
    if (!glfwInit()) { printf("glfwInit failed\n"); return 1; }
    GLFWwindow* win = glfwCreateWindow(1280, 720, "thermal3d_view", nullptr, nullptr);
    if (!win) { printf("could not open a window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);  // vsync
    glfwSetKeyCallback(win, on_key);
    glfwSetMouseButtonCallback(win, on_mouse);
    printf("  OpenGL renderer: %s\n", (const char*)glGetString(GL_RENDERER));

    const int nx = app.m.nx, ny = app.m.ny;
    const float frames[PANELS][3] = {{0.35f, 0.85f, 1.0f}, {0.55f, 1.0f, 0.45f}};  // cyan, green
    bool interop = true;
    for (int p = 0; p < PANELS; ++p) {
        Panel& P = app.panel[p];
        memcpy(P.frame, frames[p], sizeof P.frame);
        glGenTextures(1, &P.tex);
        glBindTexture(GL_TEXTURE_2D, P.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        if (interop && cudaGraphicsGLRegisterImage(&P.res, P.tex, GL_TEXTURE_2D,
                                                   cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
            cudaGetLastError();  // clear the error
            interop = false;
        }
    }
    printf(interop ? "  CUDA-OpenGL interop: colors go straight from CUDA into the textures\n"
                   : "  CUDA-OpenGL interop unavailable (window not on the NVIDIA GPU?): "
                     "copying colors through the host instead\n");
    uchar4* d_rgba = nullptr;
    unsigned long long* d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgba, (size_t)PANELS * nx * ny * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_max, PANELS * sizeof(unsigned long long)));
    std::vector<uchar4> h_rgba(interop ? 0 : (size_t)nx * ny);
    printf("Controls: 1-9 blocks, left click hotspot, right click clear, +/- speed, "
           "space pause, R reset, Esc quit%s\n", app.dies.size() > PANELS ? ", up/down dies" : "");

    // ---- main loop ----
    double last_dt = -1, step_ms = 0;
    int iters = 0, frame = 0;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // One implicit time step per frame (60 frames per real second).
        if (!app.paused) {
            const double dt = app.speed / 60.0;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (dt != last_dt) { solver.set_dt(dt); last_dt = dt; }
            solver.set_rhs_transient();
            iters = solver.solve(1e-8, 500).iters;
            step_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();
            app.t += dt;
        }

        // Colors and hottest cells on the GPU, then into the textures.
        CUDA_CHECK(cudaMemset(d_max, 0, PANELS * sizeof(unsigned long long)));
        const dim3 blk(32, 8), grd((nx + 31) / 32, (ny + 7) / 8);
        for (int p = 0; p < shown(); ++p) {
            int l0, l1;
            plane_range(die_of(p), l0, l1);
            uchar4* rgba = d_rgba + (size_t)p * nx * ny;
            k_colorize<<<grd, blk>>>(solver.device_x(), nx, ny, l0, l1, app.rise_hi, rgba, d_max + p);
            KERNEL_CHECK();
            Panel& P = app.panel[p];
            if (interop) {
                cudaArray_t arr;
                CUDA_CHECK(cudaGraphicsMapResources(1, &P.res));
                CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&arr, P.res, 0, 0));
                CUDA_CHECK(cudaMemcpy2DToArray(arr, 0, 0, rgba, nx * sizeof(uchar4),
                                               nx * sizeof(uchar4), ny, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaGraphicsUnmapResources(1, &P.res));
            } else {
                CUDA_CHECK(cudaMemcpy(h_rgba.data(), rgba, h_rgba.size() * sizeof(uchar4),
                                      cudaMemcpyDeviceToHost));
                glBindTexture(GL_TEXTURE_2D, P.tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_UNSIGNED_BYTE, h_rgba.data());
            }
        }
        unsigned long long keys[PANELS];
        CUDA_CHECK(cudaMemcpy(keys, d_max, sizeof keys, cudaMemcpyDeviceToHost));
        for (int p = 0; p < shown(); ++p) {
            Panel& P = app.panel[p];
            unsigned bits = (unsigned)(keys[p] >> 32), cell = (unsigned)(keys[p] & 0xffffffffu);
            memcpy(&P.max_rise, &bits, sizeof(float));
            P.max_i = (int)(cell % (unsigned)nx);
            P.max_j = (int)(cell / (unsigned)nx);
        }

        // Layout: the dies as squares side by side, then the color bar.
        int fw, fh;
        glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, fw, 0, fh, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glClearColor(0.08f, 0.08f, 0.1f, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        const int np = shown();
        const double margin = 0.04 * std::min(fw, fh), barw = 0.35 * margin;
        const double size = std::min(fh - 2 * margin, (fw - (np + 2) * margin - barw) / np);
        for (int p = 0; p < np; ++p) {
            Panel& P = app.panel[p];
            P.size = size;
            P.x = margin + p * (size + margin);
            P.y = (fh - size) / 2;
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, P.tex);
            glColor3f(1, 1, 1);
            glBegin(GL_QUADS);  // texture row 0 = y = 0 at the bottom
            glTexCoord2f(0, 0); glVertex2d(P.x, P.y);
            glTexCoord2f(1, 0); glVertex2d(P.x + size, P.y);
            glTexCoord2f(1, 1); glVertex2d(P.x + size, P.y + size);
            glTexCoord2f(0, 1); glVertex2d(P.x, P.y + size);
            glEnd();
            glDisable(GL_TEXTURE_2D);

            const int layer = die_of(p);
            glLineWidth(2);
            for (size_t b = 0; b < app.cs.blocks.size(); ++b) {  // floorplan blocks of this die
                const PowerBlock& B = app.cs.blocks[b];
                if (B.layer != layer) continue;
                if (app.on[b]) glColor3f(0.9f, 0.9f, 0.9f); else glColor3f(0.35f, 0.35f, 0.4f);
                rect(P, B.x0, B.y0, B.w, B.h);
            }
            glColor3f(0.3f, 0.9f, 1.0f);
            for (const PowerBlock& h : app.hotspots)
                if (h.layer == layer) rect(P, h.x0, h.y0, h.w, h.h);
            glLineWidth(3);  // die frame, in the die's color
            glColor3fv(P.frame);
            glBegin(GL_LINE_LOOP);
            glVertex2d(P.x - 4, P.y - 4);
            glVertex2d(P.x + size + 4, P.y - 4);
            glVertex2d(P.x + size + 4, P.y + size + 4);
            glVertex2d(P.x - 4, P.y + size + 4);
            glEnd();
            // Cross on the hottest cell.
            const double hx = P.x + (P.max_i + 0.5) * size / nx, hy = P.y + (P.max_j + 0.5) * size / ny;
            const double arm = 0.02 * size;
            glLineWidth(2);
            glColor3f(1, 1, 1);
            glBegin(GL_LINES);
            glVertex2d(hx - arm, hy); glVertex2d(hx + arm, hy);
            glVertex2d(hx, hy - arm); glVertex2d(hx, hy + arm);
            glEnd();
        }

        const double bx = margin + np * (size + margin), by = (fh - size) / 2;  // color bar
        glBegin(GL_QUAD_STRIP);
        for (int k = 0; k <= 32; ++k) {
            uchar4 c = colormap(k / 32.0f);
            glColor3ub(c.x, c.y, c.z);
            const double y = by + size * k / 32.0;
            glVertex2d(bx, y);
            glVertex2d(bx + barw, y);
        }
        glEnd();
        for (int p = 0; p < np; ++p) {  // each die's maximum, in its frame color
            const Panel& P = app.panel[p];
            const double my = by + size * std::min(1.0f, P.max_rise / app.rise_hi);
            const double x0 = p == 0 ? bx - 2 : bx + barw + 2, dir = p == 0 ? -1 : 1;
            glColor3fv(P.frame);
            glBegin(GL_TRIANGLES);
            glVertex2d(x0, my);
            glVertex2d(x0 + dir * 14, my + 8);
            glVertex2d(x0 + dir * 14, my - 8);
            glEnd();
        }
        glfwSwapBuffers(win);

        if (frame++ % 10 == 0) {  // status in the title bar
            char title[512];
            int len = snprintf(title, sizeof title, "t = %.3f s%s | ", app.t, app.paused ? " (paused)" : "");
            for (int p = 0; p < np; ++p)
                len += snprintf(title + len, sizeof title - len, "%s %s: max %.1f C | ",
                                p == 0 ? "left" : "right", app.cs.layers[die_of(p)].name.c_str(),
                                app.m.T_amb + app.panel[p].max_rise);
            snprintf(title + len, sizeof title - len,
                     "scale %.0f-%.0f C | %d iters, %.1f ms/step | speed %.3gx", app.m.T_amb,
                     app.m.T_amb + app.rise_hi, iters, step_ms, app.speed);
            glfwSetWindowTitle(win, title);
        }
    }

    for (Panel& P : app.panel)
        if (P.res) cudaGraphicsUnregisterResource(P.res);
    cudaFree(d_rgba);
    cudaFree(d_max);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
