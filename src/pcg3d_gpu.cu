// pcg3d_gpu.cu - GPU preconditioned conjugate gradient, matrix-free.
//
// One PCG iteration with the Jacobi preconditioner is three kernels:
//   k_apply    Ap = (A + C/dt) p     and the dot product p . Ap
//   k_update   x += a p, r -= a Ap, z = r / diag   and r . z, r . r
//   k_update_p p = z + b p
// With the multigrid preconditioner, z = M^-1 r is one V-cycle instead (mg3d.h):
// k_update_xr updates x and r and writes a float copy of r for the V-cycle, the
// V-cycle runs in MgReal (float by default), and k_mg_dot / k_update_p_mg read
// its result directly (converting to double) to form r . z and the new p.
//
// Dot products are reduced inside each block (warp shuffles, then shared
// memory) and added across blocks with one double-precision atomicAdd per block.
//
// Scalars stay on the GPU: the dot products, alpha, beta and 1/dt live in a
// small device array (sc, slots S_*), and one-thread kernels compute alpha and
// beta from them. So kernel arguments never change between iterations, and the
// host only reads ||r||^2 once per iteration, for the convergence test.
//
// CUDA graphs: an iteration is split into part A (Ap, alpha, x and r update,
// copy ||r||^2 to the host) and part B (preconditioner, beta, p update). With
// multigrid, each part is recorded once into a CUDA graph when the solver is
// built, then replayed with a single launch: part B alone is ~70 kernels
// (sweeps, restrictions, prolongations on every level), which would otherwise
// be launched one by one from the CPU every iteration.
//
// The operator is stored per cell face (operator3d.h), in float: the fine one in
// the natural layout for PCG, and every multigrid level (the fine one again) in
// the color-split layout, where each red-black sweep reads contiguous memory.
// Every buffer stays on the GPU between solves, so a transient run only moves
// the temperature field back to the host when it asks for it. All work runs on
// the solver's own stream (graphs can only be recorded from a non-default one).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <type_traits>
#include "cuda_check.h"
#include "mg3d.h"
#include "operator3d.h"
#include "pcg3d.h"

using Op = CellOp<float>;         // natural layout: PCG on the fine grid
using SOp = CellOp<float, true>;  // color-split layout: every multigrid level

// Index in the color-split layout of the fine-grid cell with natural index c
// (for the element-wise kernels that hand vectors between PCG and multigrid).
__device__ __forceinline__ size_t split_of(const SOp& s, size_t c) {
    const unsigned nx = (unsigned)s.nx, nxy = nx * (unsigned)s.ny, cc = (unsigned)c;
    const unsigned l = cc / nxy, rem = cc - l * nxy, j = rem / nx, i = rem - j * nx;
    return s.idx((int)i, (int)j, (int)l);
}

constexpr int TBX = 32, TBY = 8;  // 2D tile per plane for stencil kernels
constexpr int TB1 = 256;          // block size for element-wise kernels

__device__ __forceinline__ double warp_sum(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Sums v over the block and adds the result to *out.
// Every thread of the block must call it (no early returns before this).
// Block size must be a multiple of 32. scratch is a __shared__ double[32].
__device__ void block_sum_to(double v, double* scratch, double* out) {
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    const int nwarps = (blockDim.x * blockDim.y) >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < nwarps) ? scratch[lane] : 0.0;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---- PCG kernels (fine grid, Real precision) --------------------------------

// Slots of the device scalar array sc.
enum { S_RZ, S_RZNEW, S_RR, S_PAP, S_BB, S_ALPHA, S_BETA, S_INVDT, NSC };

// Graph replay on (default) or off; see gpu_use_graphs() in pcg3d.h.
static bool g_use_graphs = true;
void gpu_use_graphs(bool on) { g_use_graphs = on; }

static GpuSetupTimes g_last_setup;
GpuSetupTimes gpu_last_setup() { return g_last_setup; }

// Diagonal of (A + C/dt), stored once per time step; for Jacobi also its
// inverse (one FP64 division per cell, so multigrid skips it: invd == nullptr).
__global__ void k_diag(Op o, Real inv_dt, Real* diag, Real* invd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= o.nx || j >= o.ny) return;
    size_t id = ((size_t)l * o.ny + j) * o.nx + i;
    Real d = op_diag<Op, Real>(o, i, j, l, inv_dt);
    diag[id] = d;
    if (invd) invd[id] = Real(1) / d;
}

// b = P + C/dt x  (x holds the previous temperature)
__global__ void k_rhs_transient(const float* __restrict__ cap, Real inv_dt,
                                const Real* __restrict__ P, const Real* __restrict__ x,
                                Real* __restrict__ b, size_t n) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) b[c] = P[c] + (Real)cap[c] * inv_dt * x[c];
}

__global__ void k_set_scalar(double* sc, int slot, double v) { sc[slot] = v; }

// alpha = rz / pAp, and clear the sums the rest of the iteration accumulates.
__global__ void k_alpha(double* sc) {
    sc[S_ALPHA] = sc[S_RZ] / sc[S_PAP];
    sc[S_RR] = 0.0;
    sc[S_RZNEW] = 0.0;
}

// beta = rz_new / rz, rz = rz_new, and clear pAp for the next iteration.
__global__ void k_beta(double* sc) {
    sc[S_BETA] = sc[S_RZNEW] / sc[S_RZ];
    sc[S_RZ] = sc[S_RZNEW];
    sc[S_PAP] = 0.0;
}

// After the first preconditioner application: rz = r . z.
__global__ void k_start(double* sc) {
    sc[S_RZ] = sc[S_RZNEW];
    sc[S_PAP] = 0.0;
}

// Jacobi: r = b - (A + C/dt) x, z = M^-1 r, p = z;  sums r . z, r . r, b . b
__global__ void k_init(Op o, const Real* __restrict__ diag, const Real* __restrict__ b,
                       const Real* __restrict__ x, const Real* __restrict__ invd,
                       Real* __restrict__ r, Real* __restrict__ z, Real* __restrict__ p,
                       double* sc) {
    __shared__ double s0[32], s1[32], s2[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double rz = 0.0, rr = 0.0, bb = 0.0;
    if (i < o.nx && j < o.ny) {
        size_t id = ((size_t)l * o.ny + j) * o.nx + i;
        Real bi = b[id];
        Real ri = bi - op_apply_diag(o, x, diag[id], i, j, l);
        Real zi = ri * invd[id];
        r[id] = ri; z[id] = zi; p[id] = zi;
        rz = (double)ri * zi;
        rr = (double)ri * ri;
        bb = (double)bi * bi;
    }
    block_sum_to(rz, s0, &sc[S_RZ]);
    block_sum_to(rr, s1, &sc[S_RR]);
    block_sum_to(bb, s2, &sc[S_BB]);
}

// Multigrid: r = b - (A + C/dt) x, b0 = r in MgReal;  sums r . r, b . b
__global__ void k_residual(Op o, const Real* __restrict__ diag, const Real* __restrict__ b,
                           const Real* __restrict__ x, Real* __restrict__ r, SOp s0,
                           MgReal* __restrict__ b0, double* sc) {
    __shared__ double s1[32], s2[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double rr = 0.0, bb = 0.0;
    if (i < o.nx && j < o.ny) {
        size_t id = ((size_t)l * o.ny + j) * o.nx + i;
        Real bi = b[id];
        Real ri = bi - op_apply_diag(o, x, diag[id], i, j, l);
        r[id] = ri;
        b0[s0.idx(i, j, l)] = (MgReal)ri;
        rr = (double)ri * ri;
        bb = (double)bi * bi;
    }
    block_sum_to(rr, s1, &sc[S_RR]);
    block_sum_to(bb, s2, &sc[S_BB]);
}

// Ap = (A + C/dt) p;  sums p . Ap
__global__ void k_apply(Op o, const Real* __restrict__ diag, const Real* __restrict__ p,
                        Real* __restrict__ Ap, double* sc) {
    __shared__ double s[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double v = 0.0;
    if (i < o.nx && j < o.ny) {
        size_t id = ((size_t)l * o.ny + j) * o.nx + i;
        Real a = op_apply_diag(o, p, diag[id], i, j, l);
        Ap[id] = a;
        v = (double)p[id] * a;
    }
    block_sum_to(v, s, &sc[S_PAP]);
}

// Jacobi: x += alpha p, r -= alpha Ap, z = M^-1 r;  sums r . z (new), r . r
__global__ void k_update(const double* __restrict__ sc_in, const Real* __restrict__ p,
                         const Real* __restrict__ Ap, const Real* __restrict__ invd,
                         Real* x, Real* r, Real* z, size_t n, double* sc) {
    __shared__ double s0[32], s1[32];
    const Real alpha = (Real)sc_in[S_ALPHA];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double rz = 0.0, rr = 0.0;
    if (c < n) {
        x[c] += alpha * p[c];
        Real ri = r[c] - alpha * Ap[c];
        Real zi = ri * invd[c];
        r[c] = ri; z[c] = zi;
        rz = (double)ri * zi;
        rr = (double)ri * ri;
    }
    block_sum_to(rz, s0, &sc[S_RZNEW]);
    block_sum_to(rr, s1, &sc[S_RR]);
}

// Multigrid: x += alpha p, r -= alpha Ap, b0 = r in MgReal (split layout);  sums r . r
__global__ void k_update_xr(const double* __restrict__ sc_in, const Real* __restrict__ p,
                            const Real* __restrict__ Ap, Real* x, Real* r, SOp s0,
                            MgReal* __restrict__ b0, size_t n, double* sc) {
    __shared__ double s[32];
    const Real alpha = (Real)sc_in[S_ALPHA];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (c < n) {
        x[c] += alpha * p[c];
        Real ri = r[c] - alpha * Ap[c];
        r[c] = ri;
        b0[split_of(s0, c)] = (MgReal)ri;
        v = (double)ri * ri;
    }
    block_sum_to(v, s, &sc[S_RR]);
}

// Multigrid: z = M^-1 r is the V-cycle result x0, converted to Real where it is
// used. It is never stored as a separate double array: that saved 8 bytes of
// writes per cell here and 4 bytes of reads per cell in k_update_p_mg.
// Sums r . z (new).
__global__ void k_mg_dot(SOp s0, const MgReal* __restrict__ x0, const Real* __restrict__ r,
                         size_t n, double* sc) {
    __shared__ double s[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (c < n) v = (double)r[c] * (Real)x0[split_of(s0, c)];
    block_sum_to(v, s, &sc[S_RZNEW]);
}

// Multigrid: p = z + beta p, with z = x0 (first: p = z).
__global__ void k_update_p_mg(const double* __restrict__ sc, SOp s0,
                              const MgReal* __restrict__ x0, Real* p, size_t n, int first) {
    const Real beta = (Real)sc[S_BETA];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const Real z = (Real)x0[split_of(s0, c)];
    p[c] = first ? z : z + beta * p[c];
}

// p = z + beta p
__global__ void k_update_p(const double* __restrict__ sc, const Real* __restrict__ z, Real* p,
                           size_t n) {
    const Real beta = (Real)sc[S_BETA];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) p[c] = z[c] + beta * p[c];
}

// ---- multigrid kernels (MgReal precision, color-split layout) ---------------
//
// Same-color threads: x index t, color c -> column i = 2 t + ((j + c) & 1), so
// the threads of a warp touch consecutive entries of the color-split arrays.

// Column factors for every column of a level (blockIdx.z = color).
__global__ void k_mg_factor(SOp o, MgReal inv_dt, MgReal* fcp, MgReal* finv) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = 2 * t + ((j + (int)blockIdx.z) & 1);
    if (i >= o.nx || j >= o.ny) return;
    mg_factor_column(o, inv_dt, fcp, finv, i, j);
}

// Red-black line smoother: solves every column with (i + j) % 2 == color.
__global__ void k_mg_sweep(MgGrid<SOp, MgReal> g, const MgReal* __restrict__ b, MgReal* x,
                           int color) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = 2 * t + ((j + color) & 1);
    if (i >= g.op.nx || j >= g.op.ny) return;
    mg_line_solve(g, b, x, i, j);
}

// Coarse b = sum over the (up to 4) children of the fine residual b - A x.
// One thread per coarse cell, so the sum has a fixed order (no atomics).
__global__ void k_mg_restrict(SOp f, const double* __restrict__ sc, const MgReal* __restrict__ fb,
                              const MgReal* __restrict__ fx, SOp co, MgReal* __restrict__ cb) {
    const MgReal inv_dt = (MgReal)sc[S_INVDT];
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int J = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (I >= co.nx || J >= co.ny) return;
    MgReal sum = 0;
    for (int j = 2 * J; j < min(2 * J + 2, f.ny); ++j)
        for (int i = 2 * I; i < min(2 * I + 2, f.nx); ++i)
            sum += fb[f.idx(i, j, l)] - op_apply(f, fx, i, j, l, inv_dt);
    cb[co.idx(I, J, l)] = sum;
}

// Fine x += coarse x of the parent cell.
__global__ void k_mg_prolong(SOp f, MgReal* fx, SOp co, const MgReal* __restrict__ cx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= f.nx || j >= f.ny) return;
    fx[f.idx(i, j, l)] += cx[co.idx(i / 2, j / 2, l)];
}

// ---- setup kernels: multigrid levels built on the GPU ------------------------

// Coarse level from the fine one, natural layout: one thread per coarse cell,
// the same code as the CPU builder (mg_coarse_cell in mg3d.h).
__global__ void k_coarsen(Op f, const double* __restrict__ fdx, const double* __restrict__ fdy,
                          const double* __restrict__ cdx, const double* __restrict__ cdy,
                          int cnx, int cny, float* cgE, float* cgN, float* cgU, float* ccap,
                          float* cgB) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int J = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (I >= cnx || J >= cny) return;
    mg_coarse_cell(f, fdx, fdy, cdx, cdy, cnx, cny, I, J, l, cgE, cgN, cgU, ccap, cgB);
}

// Natural -> color-split copy of one array with `planes` z-planes.
__global__ void k_to_split(SOp s, int planes, const float* __restrict__ in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= s.nx || j >= s.ny || l >= planes) return;
    out[s.idx(i, j, l)] = in[((size_t)l * s.ny + j) * s.nx + i];
}

// ---------------------------------------------------------------------------

static dim3 block2d() { return dim3(TBX, TBY); }
static dim3 grid2d(int nx, int ny, int nz) {
    return dim3((nx + TBX - 1) / TBX, (ny + TBY - 1) / TBY, nz);
}

GpuPcg::GpuPcg(const Model3D& m, Precond pc) : pc_(pc), n_(m.ncells()) {
    if (!m.finalized) {
        fprintf(stderr, "Model3D used before finalize()\n");
        exit(1);
    }
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](clk::time_point t) {
        return std::chrono::duration<double, std::milli>(clk::now() - t).count();
    };
    auto t0 = clk::now();

    // A blocking stream: it waits for the plain cudaMemcpy uploads below, which
    // run on the default stream.
    CUDA_CHECK(cudaStreamCreate(&s_));
    const size_t bytes = n_ * sizeof(Real);
    for (Real** ptr : {&P_, &x_, &b_, &r_, &p_, &Ap_, &diag_})
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    if (pc_ == Precond::Jacobi)  // z and 1 / diag are only needed by Jacobi
        for (Real** ptr : {&z_, &invd_}) CUDA_CHECK(cudaMalloc(ptr, bytes));
    CUDA_CHECK(cudaMalloc(&sc_, NSC * sizeof(double)));
    CUDA_CHECK(cudaMemset(sc_, 0, NSC * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_sc_, NSC * sizeof(double)));  // pinned: async copies, graphs
    CUDA_CHECK(cudaDeviceSynchronize());
    setup_.alloc_ms = ms_since(t0);
    t0 = clk::now();

    // The fine operator, natural layout, straight from the model (used by PCG).
    auto upload = [](const std::vector<float>& v, float** dst) {
        CUDA_CHECK(cudaMalloc(dst, v.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(*dst, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    };
    fine_.nx = m.nx;
    fine_.ny = m.ny;
    fine_.nz = m.nz;
    upload(m.gE, &fine_.gE);
    upload(m.gN, &fine_.gN);
    upload(m.gU, &fine_.gU);
    upload(m.gB, &fine_.gB);
    upload(m.cap, &fine_.cap);
    // Real is double by default, like m.P: upload it directly, without a copy.
    if constexpr (std::is_same<Real, double>::value) {
        CUDA_CHECK(cudaMemcpy(P_, m.P.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        std::vector<Real> P = to_real(m.P);
        CUDA_CHECK(cudaMemcpy(P_, P.data(), bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    setup_.upload_ms = ms_since(t0);
    t0 = clk::now();

    // Multigrid levels, built on the GPU: coarsen in the natural layout (level 0
    // is fine_), then copy every level into the color-split layout.
    if (pc_ == Precond::Multigrid) {
        Level nat = fine_;                       // current level, natural layout
        std::vector<double> dx(m.nx, m.dx), dy(m.ny, m.dy);
        double *d_dx = nullptr, *d_dy = nullptr;
        auto upload_d = [](const std::vector<double>& v, double** dst) {
            CUDA_CHECK(cudaMalloc(dst, v.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(*dst, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice));
        };
        upload_d(dx, &d_dx);
        upload_d(dy, &d_dy);
        while (true) {
            Level L;                             // the split copy of `nat`
            L.nx = nat.nx;
            L.ny = nat.ny;
            L.nz = nat.nz;
            const SOp so = L.sop();
            const size_t lc = L.cells(), lcol = so.plane();
            for (float** ptr : {&L.gE, &L.gN, &L.gU, &L.cap}) {
                CUDA_CHECK(cudaMalloc(ptr, lc * sizeof(float)));
                CUDA_CHECK(cudaMemset(*ptr, 0, lc * sizeof(float)));  // padding stays 0
            }
            CUDA_CHECK(cudaMalloc(&L.gB, lcol * sizeof(float)));
            CUDA_CHECK(cudaMemset(L.gB, 0, lcol * sizeof(float)));
            const dim3 g3 = grid2d(L.nx, L.ny, L.nz), g1 = grid2d(L.nx, L.ny, 1);
            k_to_split<<<g3, block2d()>>>(so, L.nz, nat.gE, L.gE);
            k_to_split<<<g3, block2d()>>>(so, L.nz, nat.gN, L.gN);
            k_to_split<<<g3, block2d()>>>(so, L.nz, nat.gU, L.gU);
            k_to_split<<<g3, block2d()>>>(so, L.nz, nat.cap, L.cap);
            k_to_split<<<g1, block2d()>>>(so, 1, nat.gB, L.gB);
            KERNEL_CHECK();
            for (MgReal** ptr : {&L.fcp, &L.finv, &L.b, &L.x}) {
                CUDA_CHECK(cudaMalloc(ptr, lc * sizeof(MgReal)));
                CUDA_CHECK(cudaMemset(*ptr, 0, lc * sizeof(MgReal)));  // padding stays 0
            }
            levels_.push_back(L);
            if (nat.nx == 1 && nat.ny == 1) break;

            // Next coarser level, natural layout (temporary).
            Level c;
            c.nx = (nat.nx + 1) / 2;
            c.ny = (nat.ny + 1) / 2;
            c.nz = nat.nz;
            std::vector<double> cdx = mg_coarse_widths(dx), cdy = mg_coarse_widths(dy);
            double *d_cdx = nullptr, *d_cdy = nullptr;
            upload_d(cdx, &d_cdx);
            upload_d(cdy, &d_cdy);
            const size_t cn = (size_t)c.nx * c.ny * c.nz, cxy = (size_t)c.nx * c.ny;
            for (float** ptr : {&c.gE, &c.gN, &c.gU, &c.cap}) CUDA_CHECK(cudaMalloc(ptr, cn * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.gB, cxy * sizeof(float)));
            k_coarsen<<<grid2d(c.nx, c.ny, c.nz), block2d()>>>(nat.op(), d_dx, d_dy, d_cdx, d_cdy,
                                                               c.nx, c.ny, c.gE, c.gN, c.gU,
                                                               c.cap, c.gB);
            KERNEL_CHECK();
            if (nat.gE != fine_.gE)  // free the previous temporary level (never fine_)
                for (float* ptr : {nat.gE, nat.gN, nat.gU, nat.gB, nat.cap}) cudaFree(ptr);
            CUDA_CHECK(cudaFree(d_dx));
            CUDA_CHECK(cudaFree(d_dy));
            nat = c;
            dx.swap(cdx);
            dy.swap(cdy);
            d_dx = d_cdx;
            d_dy = d_cdy;
        }
        if (nat.gE != fine_.gE)
            for (float* ptr : {nat.gE, nat.gN, nat.gU, nat.gB, nat.cap}) cudaFree(ptr);
        CUDA_CHECK(cudaFree(d_dx));
        CUDA_CHECK(cudaFree(d_dy));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    setup_.levels_ms = ms_since(t0);
    t0 = clk::now();

    set_dt(0.0);
    set_x_zero();
    CUDA_CHECK(cudaDeviceSynchronize());
    setup_.factor_ms = ms_since(t0);
    t0 = clk::now();

    // Record the two halves of an iteration as graphs. They read 1/dt, alpha and
    // beta from sc_, so they stay valid when the time step changes. Multigrid
    // only: a Jacobi iteration is just 4 kernels, and measured on an RTX 4060
    // (Windows) replaying them as graphs was no faster, sometimes slower.
    if (g_use_graphs && pc_ == Precond::Multigrid) {
        for (int part = 0; part < 2; ++part) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
            if (part == 0) iter_a(); else iter_b();
            CUDA_CHECK(cudaStreamEndCapture(s_, &g));
            CUDA_CHECK(cudaGraphInstantiate(part == 0 ? &graph_a_ : &graph_b_, g, 0));
            CUDA_CHECK(cudaGraphDestroy(g));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(s_));
    setup_.graphs_ms = ms_since(t0);
    g_last_setup = setup_;
}

GpuPcg::~GpuPcg() {
    // No CUDA_CHECK here: a destructor must not exit or throw.
    cudaStreamSynchronize(s_);
    if (graph_a_) cudaGraphExecDestroy(graph_a_);
    if (graph_b_) cudaGraphExecDestroy(graph_b_);
    for (Real* ptr : {P_, x_, b_, r_, z_, p_, Ap_, diag_, invd_}) cudaFree(ptr);  // null is fine
    cudaFree(sc_);
    cudaFreeHost(h_sc_);
    for (float* ptr : {fine_.gE, fine_.gN, fine_.gU, fine_.gB, fine_.cap}) cudaFree(ptr);
    for (const Level& L : levels_) {
        for (float* ptr : {L.gE, L.gN, L.gU, L.gB, L.cap}) cudaFree(ptr);
        for (MgReal* ptr : {L.fcp, L.finv, L.b, L.x}) cudaFree(ptr);
    }
    cudaStreamDestroy(s_);
}

void GpuPcg::set_dt(double dt) {
    inv_dt_ = dt > 0 ? (Real)(1.0 / dt) : Real(0);
    k_set_scalar<<<1, 1, 0, s_>>>(sc_, S_INVDT, (double)inv_dt_);
    KERNEL_CHECK();
    const Op fine = fine_.op();
    k_diag<<<grid2d(fine.nx, fine.ny, fine.nz), block2d(), 0, s_>>>(fine, inv_dt_, diag_, invd_);
    KERNEL_CHECK();
    if (pc_ == Precond::Multigrid)
        for (const Level& L : levels_) {  // multigrid column factors, both colors
            k_mg_factor<<<grid2d((L.nx + 1) / 2, L.ny, 2), block2d(), 0, s_>>>(
                L.sop(), (MgReal)inv_dt_, L.fcp, L.finv);
            KERNEL_CHECK();
        }
}

void GpuPcg::set_x_zero() { CUDA_CHECK(cudaMemsetAsync(x_, 0, n_ * sizeof(Real), s_)); }

void GpuPcg::set_rhs_steady() {
    CUDA_CHECK(cudaMemcpyAsync(b_, P_, n_ * sizeof(Real), cudaMemcpyDeviceToDevice, s_));
}

void GpuPcg::set_rhs_transient() {
    k_rhs_transient<<<(unsigned)((n_ + TB1 - 1) / TB1), TB1, 0, s_>>>(fine_.cap, inv_dt_,
                                                                       P_, x_, b_, n_);
    KERNEL_CHECK();
}

void GpuPcg::set_power(const std::vector<double>& P) {
    CUDA_CHECK(cudaStreamSynchronize(s_));  // the previous P may still be in use
    if constexpr (std::is_same<Real, double>::value) {
        CUDA_CHECK(cudaMemcpy(P_, P.data(), n_ * sizeof(Real), cudaMemcpyHostToDevice));
    } else {
        std::vector<Real> Pr = to_real(P);
        CUDA_CHECK(cudaMemcpy(P_, Pr.data(), n_ * sizeof(Real), cudaMemcpyHostToDevice));
    }
}

void GpuPcg::get_x(std::vector<Real>& T) const {
    T.resize(n_);
    CUDA_CHECK(cudaMemcpyAsync(T.data(), x_, n_ * sizeof(Real), cudaMemcpyDeviceToHost, s_));
    CUDA_CHECK(cudaStreamSynchronize(s_));
}

// Launches below may be recorded into a graph, where synchronizing is not
// allowed, so they only check the launch itself. Callers check (and, with
// SYNC_CHECK, synchronize) after the whole sequence.
#define LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

// One multigrid V-cycle on level k (see mg3d.h): approximately solves
// (A + C/dt) x = b there, starting from x = 0.
void GpuPcg::vcycle(size_t k) {
    const Level& L = levels_[k];
    const MgGrid<SOp, MgReal> g{L.sop(), L.fcp, L.finv};
    const dim3 blk = block2d();
    const dim3 g2 = grid2d((L.nx + 1) / 2, L.ny, 1);  // sweeps: one thread per same-color column
    CUDA_CHECK(cudaMemsetAsync(L.x, 0, L.cells() * sizeof(MgReal), s_));

    if (k + 1 == levels_.size()) {  // a single column: one line solve is exact
        k_mg_sweep<<<g2, blk, 0, s_>>>(g, L.b, L.x, 0);
        LAUNCH_CHECK();
        return;
    }
    k_mg_sweep<<<g2, blk, 0, s_>>>(g, L.b, L.x, 0);
    LAUNCH_CHECK();
    k_mg_sweep<<<g2, blk, 0, s_>>>(g, L.b, L.x, 1);
    LAUNCH_CHECK();

    const Level& C = levels_[k + 1];
    k_mg_restrict<<<grid2d(C.nx, C.ny, C.nz), blk, 0, s_>>>(L.sop(), sc_, L.b, L.x, C.sop(),
                                                           C.b);
    LAUNCH_CHECK();
    vcycle(k + 1);
    k_mg_prolong<<<grid2d(L.nx, L.ny, L.nz), blk, 0, s_>>>(L.sop(), L.x, C.sop(), C.x);
    LAUNCH_CHECK();

    k_mg_sweep<<<g2, blk, 0, s_>>>(g, L.b, L.x, 1);  // reverse order keeps it symmetric
    LAUNCH_CHECK();
    k_mg_sweep<<<g2, blk, 0, s_>>>(g, L.b, L.x, 0);
    LAUNCH_CHECK();
}

// Part A of an iteration: Ap, alpha, x += alpha p, r -= alpha Ap (and, for
// Jacobi, z); then copy r . r to the host for the convergence test.
void GpuPcg::iter_a() {
    const Op o = fine_.op();
    const unsigned egrid = (unsigned)((n_ + TB1 - 1) / TB1);
    k_apply<<<grid2d(o.nx, o.ny, o.nz), block2d(), 0, s_>>>(o, diag_, p_, Ap_, sc_);
    LAUNCH_CHECK();
    k_alpha<<<1, 1, 0, s_>>>(sc_);
    LAUNCH_CHECK();
    if (pc_ == Precond::Multigrid)
        k_update_xr<<<egrid, TB1, 0, s_>>>(sc_, p_, Ap_, x_, r_, levels_[0].sop(),
                                           levels_[0].b, n_, sc_);
    else
        k_update<<<egrid, TB1, 0, s_>>>(sc_, p_, Ap_, invd_, x_, r_, z_, n_, sc_);
    LAUNCH_CHECK();
    CUDA_CHECK(cudaMemcpyAsync(&h_sc_[S_RR], &sc_[S_RR], sizeof(double), cudaMemcpyDeviceToHost, s_));
}

// Part B: z = M^-1 r (multigrid), beta, p = z + beta p.
void GpuPcg::iter_b() {
    const unsigned egrid = (unsigned)((n_ + TB1 - 1) / TB1);
    if (pc_ == Precond::Multigrid) {
        const SOp s0 = levels_[0].sop();
        vcycle(0);
        k_mg_dot<<<egrid, TB1, 0, s_>>>(s0, levels_[0].x, r_, n_, sc_);
        LAUNCH_CHECK();
        k_beta<<<1, 1, 0, s_>>>(sc_);
        LAUNCH_CHECK();
        k_update_p_mg<<<egrid, TB1, 0, s_>>>(sc_, s0, levels_[0].x, p_, n_, 0);
        LAUNCH_CHECK();
        return;
    }
    k_beta<<<1, 1, 0, s_>>>(sc_);
    LAUNCH_CHECK();
    k_update_p<<<egrid, TB1, 0, s_>>>(sc_, z_, p_, n_);
    LAUNCH_CHECK();
}

void GpuPcg::run_a() {
    if (graph_a_) CUDA_CHECK(cudaGraphLaunch(graph_a_, s_)); else iter_a();
    KERNEL_CHECK();
}

void GpuPcg::run_b() {
    if (graph_b_) CUDA_CHECK(cudaGraphLaunch(graph_b_, s_)); else iter_b();
    KERNEL_CHECK();
}

PcgStats GpuPcg::solve(double tol, int maxit) {
    const Op o = fine_.op();
    const dim3 sgrid = grid2d(o.nx, o.ny, o.nz), sblock = block2d();
    const unsigned egrid = (unsigned)((n_ + TB1 - 1) / TB1);
    PcgStats st;

    // Start: r = b - A x, z = M^-1 r, p = z, rz = r . z. Clear every sum (not 1/dt).
    CUDA_CHECK(cudaMemsetAsync(sc_, 0, S_INVDT * sizeof(double), s_));
    if (pc_ == Precond::Multigrid)
        k_residual<<<sgrid, sblock, 0, s_>>>(o, diag_, b_, x_, r_, levels_[0].sop(),
                                             levels_[0].b, sc_);
    else
        k_init<<<sgrid, sblock, 0, s_>>>(o, diag_, b_, x_, invd_, r_, z_, p_, sc_);
    KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpyAsync(h_sc_, sc_, NSC * sizeof(double), cudaMemcpyDeviceToHost, s_));
    CUDA_CHECK(cudaStreamSynchronize(s_));
    const double bb = h_sc_[S_BB];
    if (bb == 0) { set_x_zero(); st.converged = true; return st; }
    st.relres = std::sqrt(h_sc_[S_RR] / bb);
    if (st.relres < tol) { st.converged = true; return st; }

    if (pc_ == Precond::Multigrid) {
        const SOp s0 = levels_[0].sop();
        vcycle(0);
        k_mg_dot<<<egrid, TB1, 0, s_>>>(s0, levels_[0].x, r_, n_, sc_);
        LAUNCH_CHECK();
        k_start<<<1, 1, 0, s_>>>(sc_);
        LAUNCH_CHECK();
        k_update_p_mg<<<egrid, TB1, 0, s_>>>(sc_, s0, levels_[0].x, p_, n_, 1);
        KERNEL_CHECK();
    }

    // The only synchronization per iteration: wait for part A, test ||r||.
    for (int it = 1; it <= maxit; ++it) {
        run_a();
        CUDA_CHECK(cudaStreamSynchronize(s_));
        st.iters = it;
        st.relres = std::sqrt(h_sc_[S_RR] / bb);
        if (st.relres < tol) { st.converged = true; break; }
        run_b();
    }
    return st;
}

// Steady solve from x = 0, timed in three parts: setup (levels, allocation,
// upload, graph recording), solve, and total (including download and freeing).
SolveResult gpu_pcg(const Model3D& m, double tol, int maxit, Precond pc) {
    cudaEvent_t start, mid, end, stop;
    for (cudaEvent_t* e : {&start, &mid, &end, &stop}) CUDA_CHECK(cudaEventCreate(e));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    SolveResult res;
    {
        GpuPcg s(m, pc);  // builds the levels, allocates, uploads, records the graphs
        s.set_x_zero();
        s.set_rhs_steady();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(mid));
        PcgStats st = s.solve(tol, maxit);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(end));
        s.get_x(res.T);
        res.iters = st.iters;
        res.relres = st.relres;
        res.converged = st.converged;
    }  // frees the GPU buffers before the stop event, as before

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // also reports kernel execution errors
    float a = 0, b = 0, c = 0;
    CUDA_CHECK(cudaEventElapsedTime(&a, start, mid));
    CUDA_CHECK(cudaEventElapsedTime(&b, mid, end));
    CUDA_CHECK(cudaEventElapsedTime(&c, start, stop));
    res.setup_ms = a;
    res.solve_ms = b;
    res.ms = c;
    for (cudaEvent_t e : {start, mid, end, stop}) CUDA_CHECK(cudaEventDestroy(e));
    return res;
}
