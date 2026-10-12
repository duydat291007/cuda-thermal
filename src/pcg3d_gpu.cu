// pcg3d_gpu.cu - GPU preconditioned conjugate gradient, matrix-free.
//
// One PCG iteration is three kernels:
//   k_apply    Ap = (A + S) p     and the dot product p . Ap
//   k_update   x += a p, r -= a Ap, z = r / diag   and r . z, r . r
//   k_update_p p = z + b p
// Dot products are reduced inside each block (warp shuffles, then shared
// memory) and added across blocks with one double-precision atomicAdd per block.
// The host reads the scalars back each iteration to compute alpha and beta.
//
// With the multigrid preconditioner, z = M^-1 r is one V-cycle instead (mg3d.h),
// so the loop uses unfused kernels: update x and r, run the V-cycle, then r . z.
//
// The GpuPcg object keeps every buffer on the GPU between solves, so a transient
// run only moves the temperature field back to the host when it asks for it.
#include <algorithm>
#include <cmath>
#include "cuda_check.h"
#include "mg3d.h"
#include "operator3d.h"
#include "pcg3d.h"

// Per-plane values. Every thread in a block works on the same plane, so
// constant memory serves each read to the whole warp at once.
struct PlaneParams {
    Real gx[MAXZ], gy[MAXZ], gzd[MAXZ], gzu[MAXZ], shift[MAXZ];  // fine grid, per cell
    Real kdz[MAXZ], gzu_a[MAXZ], gzd_a[MAXZ], shift_a[MAXZ];     // multigrid, per area
};
__constant__ PlaneParams c_pp;  // one GpuPcg at a time uses this

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

__device__ __forceinline__ Real apply_dev(const Real* v, int i, int j, int l, int nx,
                                          int ny, int nz) {
    return apply_cell(v, i, j, l, nx, ny, nz, c_pp.gx[l], c_pp.gy[l], c_pp.gzd[l],
                      c_pp.gzu[l], c_pp.shift[l]);
}

__global__ void k_invdiag(Real* invd, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= nx || j >= ny) return;
    invd[((size_t)l * ny + j) * nx + i] =
        Real(1) / diag_cell(i, j, nx, ny, c_pp.gx[l], c_pp.gy[l], c_pp.gzd[l], c_pp.gzu[l],
                            c_pp.shift[l]);
}

// b = P + S x  (x holds the previous temperature)
__global__ void k_rhs_transient(const Real* __restrict__ P, const Real* __restrict__ x,
                                Real* __restrict__ b, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= nx || j >= ny) return;
    size_t id = ((size_t)l * ny + j) * nx + i;
    b[id] = P[id] + c_pp.shift[l] * x[id];
}

// r = b - (A + S) x, z = M^-1 r, p = z
// dots[0] += r . z, dots[1] += r . r, dots[2] += b . b
__global__ void k_init(const Real* __restrict__ b, const Real* __restrict__ x,
                       const Real* __restrict__ invd, Real* __restrict__ r,
                       Real* __restrict__ z, Real* __restrict__ p, int nx, int ny, int nz,
                       double* dots) {
    __shared__ double s0[32], s1[32], s2[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double rz = 0.0, rr = 0.0, bb = 0.0;
    if (i < nx && j < ny) {
        size_t id = ((size_t)l * ny + j) * nx + i;
        Real bi = b[id];
        Real ri = bi - apply_dev(x, i, j, l, nx, ny, nz);
        Real zi = ri * invd[id];
        r[id] = ri; z[id] = zi; p[id] = zi;
        rz = (double)ri * zi;
        rr = (double)ri * ri;
        bb = (double)bi * bi;
    }
    block_sum_to(rz, s0, &dots[0]);
    block_sum_to(rr, s1, &dots[1]);
    block_sum_to(bb, s2, &dots[2]);
}

// Ap = (A + S) p;  *pAp += p . Ap
__global__ void k_apply(const Real* __restrict__ p, Real* __restrict__ Ap,
                        int nx, int ny, int nz, double* pAp) {
    __shared__ double s[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double v = 0.0;
    if (i < nx && j < ny) {
        size_t id = ((size_t)l * ny + j) * nx + i;
        Real a = apply_dev(p, i, j, l, nx, ny, nz);
        Ap[id] = a;
        v = (double)p[id] * a;
    }
    block_sum_to(v, s, pAp);
}

// x += alpha p, r -= alpha Ap, z = M^-1 r;  dots[0] += r . z, dots[1] += r . r
__global__ void k_update(Real alpha, const Real* __restrict__ p,
                         const Real* __restrict__ Ap, const Real* __restrict__ invd,
                         Real* x, Real* r, Real* z, size_t n, double* dots) {
    __shared__ double s0[32], s1[32];
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
    block_sum_to(rz, s0, &dots[0]);
    block_sum_to(rr, s1, &dots[1]);
}

// p = z + beta p
__global__ void k_update_p(Real beta, const Real* __restrict__ z, Real* p, size_t n) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) p[c] = z[c] + beta * p[c];
}

// ---- kernels used with the multigrid preconditioner ------------------------

// r = b - (A + S) x;  dots[1] += r . r, dots[2] += b . b
__global__ void k_residual(const Real* __restrict__ b, const Real* __restrict__ x,
                           Real* __restrict__ r, int nx, int ny, int nz, double* dots) {
    __shared__ double s1[32], s2[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double rr = 0.0, bb = 0.0;
    if (i < nx && j < ny) {
        size_t id = ((size_t)l * ny + j) * nx + i;
        Real bi = b[id];
        Real ri = bi - apply_dev(x, i, j, l, nx, ny, nz);
        r[id] = ri;
        rr = (double)ri * ri;
        bb = (double)bi * bi;
    }
    block_sum_to(rr, s1, &dots[1]);
    block_sum_to(bb, s2, &dots[2]);
}

// x += alpha p, r -= alpha Ap;  *rr += r . r
__global__ void k_update_xr(Real alpha, const Real* __restrict__ p,
                            const Real* __restrict__ Ap, Real* x, Real* r, size_t n,
                            double* rr) {
    __shared__ double s[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (c < n) {
        x[c] += alpha * p[c];
        Real ri = r[c] - alpha * Ap[c];
        r[c] = ri;
        v = (double)ri * ri;
    }
    block_sum_to(v, s, rr);
}

// *out += a . b
__global__ void k_dot(const Real* __restrict__ a, const Real* __restrict__ b, size_t n,
                      double* out) {
    __shared__ double s[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = (c < n) ? (double)a[c] * b[c] : 0.0;
    block_sum_to(v, s, out);
}

__device__ __forceinline__ MgPlanes<Real> mg_planes_dev(int nz) {
    return MgPlanes<Real>{c_pp.kdz, c_pp.gzu_a, c_pp.gzd_a, c_pp.shift_a, nz};
}

// Red-black line smoother: solves every column with (i + j) % 2 == color.
__global__ void k_mg_sweep(MgGrid<Real> g, const Real* __restrict__ b, Real* x, int nz,
                           int color) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= g.nx || j >= g.ny || ((i + j) & 1) != color) return;
    mg_line_solve(g, mg_planes_dev(nz), b, x, i, j);
}

// Coarse b = sum over the (up to 4) children of the fine residual b - A x.
// One thread per coarse cell, so the sum has a fixed order (no atomics).
__global__ void k_mg_restrict(MgGrid<Real> f, const Real* __restrict__ fb,
                              const Real* __restrict__ fx, int cnx, int cny,
                              Real* __restrict__ cb, int nz) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int J = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (I >= cnx || J >= cny) return;
    const MgPlanes<Real> p = mg_planes_dev(nz);
    const size_t fxy = (size_t)f.nx * f.ny;
    Real sum = 0;
    for (int j = 2 * J; j < min(2 * J + 2, f.ny); ++j)
        for (int i = 2 * I; i < min(2 * I + 2, f.nx); ++i) {
            size_t id = (size_t)l * fxy + (size_t)j * f.nx + i;
            sum += fb[id] - mg_apply(f, p, fx, i, j, l);
        }
    cb[((size_t)l * cny + J) * cnx + I] = sum;
}

// Fine x += coarse x of the parent cell.
__global__ void k_mg_prolong(int fnx, int fny, Real* fx, int cnx, int cny,
                             const Real* __restrict__ cx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= fnx || j >= fny) return;
    fx[((size_t)l * fny + j) * fnx + i] += cx[((size_t)l * cny + j / 2) * cnx + i / 2];
}

// ---------------------------------------------------------------------------

static dim3 stencil_block() { return dim3(TBX, TBY); }
static dim3 stencil_grid(const Model3D& m) {
    return dim3((m.nx + TBX - 1) / TBX, (m.ny + TBY - 1) / TBY, m.nz);
}

static dim3 grid2d(int nx, int ny, int nz) {
    return dim3((nx + TBX - 1) / TBX, (ny + TBY - 1) / TBY, nz);
}

GpuPcg::GpuPcg(const Model3D& m, Precond pc) : m_(m), pc_(pc), n_(m.ncells()) {
    const size_t bytes = n_ * sizeof(Real);
    for (Real** ptr : {&P_, &x_, &b_, &r_, &z_, &p_, &Ap_, &invd_})
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    CUDA_CHECK(cudaMalloc(&dots_, 3 * sizeof(double)));
    if (pc_ == Precond::Multigrid) {
        for (const MgLevelGeom& g : mg_hierarchy(m)) {
            Level L;
            L.nx = g.nx;
            L.ny = g.ny;
            std::vector<Real> dx = to_real(g.dx), dy = to_real(g.dy);
            CUDA_CHECK(cudaMalloc(&L.dx, dx.size() * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&L.dy, dy.size() * sizeof(Real)));
            CUDA_CHECK(cudaMemcpy(L.dx, dx.data(), dx.size() * sizeof(Real), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(L.dy, dy.data(), dy.size() * sizeof(Real), cudaMemcpyHostToDevice));
            if (levels_.empty()) {   // finest level works directly on PCG's r and z
                L.b = r_;
                L.x = z_;
            } else {
                size_t lb = (size_t)g.nx * g.ny * m.nz * sizeof(Real);
                CUDA_CHECK(cudaMalloc(&L.b, lb));
                CUDA_CHECK(cudaMalloc(&L.x, lb));
            }
            levels_.push_back(L);
        }
    }
    std::vector<Real> P = to_real(m.P);
    CUDA_CHECK(cudaMemcpy(P_, P.data(), bytes, cudaMemcpyHostToDevice));
    set_shift(std::vector<double>(m.nz, 0.0));
    set_x_zero();
}

GpuPcg::~GpuPcg() {
    // No CUDA_CHECK here: a destructor must not exit or throw.
    for (Real* ptr : {P_, x_, b_, r_, z_, p_, Ap_, invd_}) cudaFree(ptr);
    cudaFree(dots_);
    for (size_t k = 0; k < levels_.size(); ++k) {
        cudaFree(levels_[k].dx);
        cudaFree(levels_[k].dy);
        if (k > 0) {  // level 0 borrows r_ and z_
            cudaFree(levels_[k].b);
            cudaFree(levels_[k].x);
        }
    }
}

void GpuPcg::set_shift(const std::vector<double>& shift) {
    PlaneParams pp = {};
    for (int l = 0; l < m_.nz; ++l) {
        pp.gx[l] = (Real)m_.gx[l];
        pp.gy[l] = (Real)m_.gy[l];
        pp.gzd[l] = (Real)m_.gzd[l];
        pp.gzu[l] = (Real)m_.gzu[l];
        pp.shift[l] = (Real)shift[l];
    }
    MgPlaneCoefs c = mg_plane_coefs(m_, shift);
    for (int l = 0; l < m_.nz; ++l) {
        pp.kdz[l] = (Real)c.kdz[l];
        pp.gzu_a[l] = (Real)c.gzu_a[l];
        pp.gzd_a[l] = (Real)c.gzd_a[l];
        pp.shift_a[l] = (Real)c.shift_a[l];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_pp, &pp, sizeof(pp)));
    k_invdiag<<<stencil_grid(m_), stencil_block()>>>(invd_, m_.nx, m_.ny);
    KERNEL_CHECK();
}

void GpuPcg::set_x_zero() { CUDA_CHECK(cudaMemset(x_, 0, n_ * sizeof(Real))); }

void GpuPcg::set_rhs_steady() {
    CUDA_CHECK(cudaMemcpy(b_, P_, n_ * sizeof(Real), cudaMemcpyDeviceToDevice));
}

void GpuPcg::set_rhs_transient() {
    k_rhs_transient<<<stencil_grid(m_), stencil_block()>>>(P_, x_, b_, m_.nx, m_.ny);
    KERNEL_CHECK();
}

void GpuPcg::get_x(std::vector<Real>& T) const {
    T.resize(n_);
    CUDA_CHECK(cudaMemcpy(T.data(), x_, n_ * sizeof(Real), cudaMemcpyDeviceToHost));
}

// One multigrid V-cycle on level k (see mg3d.h): approximately solves
// (A + S) x = b there, starting from x = 0.
void GpuPcg::vcycle(size_t k) {
    const Level& L = levels_[k];
    const int nz = m_.nz;
    const MgGrid<Real> g{L.nx, L.ny, L.dx, L.dy};
    const dim3 blk = stencil_block(), g2 = grid2d(L.nx, L.ny, 1);
    CUDA_CHECK(cudaMemset(L.x, 0, (size_t)L.nx * L.ny * nz * sizeof(Real)));

    if (k + 1 == levels_.size()) {  // a single column: one line solve is exact
        k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, nz, 0);
        KERNEL_CHECK();
        return;
    }
    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, nz, 0);
    KERNEL_CHECK();
    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, nz, 1);
    KERNEL_CHECK();

    const Level& C = levels_[k + 1];
    k_mg_restrict<<<grid2d(C.nx, C.ny, nz), blk>>>(g, L.b, L.x, C.nx, C.ny, C.b, nz);
    KERNEL_CHECK();
    vcycle(k + 1);
    k_mg_prolong<<<grid2d(L.nx, L.ny, nz), blk>>>(L.nx, L.ny, L.x, C.nx, C.ny, C.x);
    KERNEL_CHECK();

    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, nz, 1);  // reverse order keeps it symmetric
    KERNEL_CHECK();
    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, nz, 0);
    KERNEL_CHECK();
}

void GpuPcg::precondition() { vcycle(0); }  // reads r_ (level 0 b), writes z_ (level 0 x)

PcgStats GpuPcg::solve(double tol, int maxit) {
    if (pc_ == Precond::Multigrid) {
        const int nx = m_.nx, ny = m_.ny, nz = m_.nz;
        const dim3 sgrid = stencil_grid(m_), sblock = stencil_block();
        const unsigned egrid = (unsigned)((n_ + TB1 - 1) / TB1);
        double h[3];

        CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
        k_residual<<<sgrid, sblock>>>(b_, x_, r_, nx, ny, nz, dots_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h, dots_, 3 * sizeof(double), cudaMemcpyDeviceToHost));
        const double bb = h[2];
        PcgStats st;
        if (bb == 0) { set_x_zero(); st.converged = true; return st; }
        st.relres = std::sqrt(h[1] / bb);
        if (st.relres < tol) { st.converged = true; return st; }

        precondition();
        CUDA_CHECK(cudaMemcpy(p_, z_, n_ * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(dots_, 0, sizeof(double)));
        k_dot<<<egrid, TB1>>>(r_, z_, n_, &dots_[0]);
        KERNEL_CHECK();
        double rz;
        CUDA_CHECK(cudaMemcpy(&rz, &dots_[0], sizeof(double), cudaMemcpyDeviceToHost));

        for (int it = 1; it <= maxit; ++it) {
            CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
            k_apply<<<sgrid, sblock>>>(p_, Ap_, nx, ny, nz, &dots_[2]);
            KERNEL_CHECK();
            double pAp;
            CUDA_CHECK(cudaMemcpy(&pAp, &dots_[2], sizeof(double), cudaMemcpyDeviceToHost));
            Real alpha = (Real)(rz / pAp);

            k_update_xr<<<egrid, TB1>>>(alpha, p_, Ap_, x_, r_, n_, &dots_[1]);
            KERNEL_CHECK();
            double rr;
            CUDA_CHECK(cudaMemcpy(&rr, &dots_[1], sizeof(double), cudaMemcpyDeviceToHost));
            st.iters = it;
            st.relres = std::sqrt(rr / bb);
            if (st.relres < tol) { st.converged = true; break; }

            precondition();
            k_dot<<<egrid, TB1>>>(r_, z_, n_, &dots_[0]);
            KERNEL_CHECK();
            double rz_new;
            CUDA_CHECK(cudaMemcpy(&rz_new, &dots_[0], sizeof(double), cudaMemcpyDeviceToHost));
            Real beta = (Real)(rz_new / rz);
            rz = rz_new;
            k_update_p<<<egrid, TB1>>>(beta, z_, p_, n_);
            KERNEL_CHECK();
        }
        return st;
    }

    // Jacobi preconditioner: fused kernels, z = r / diag computed inline.
    const int nx = m_.nx, ny = m_.ny, nz = m_.nz;
    const dim3 sgrid = stencil_grid(m_), sblock = stencil_block();
    const unsigned egrid = (unsigned)((n_ + TB1 - 1) / TB1);

    double h[3];
    CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
    k_init<<<sgrid, sblock>>>(b_, x_, invd_, r_, z_, p_, nx, ny, nz, dots_);
    KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h, dots_, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    double rz = h[0];
    const double bb = h[2];

    PcgStats st;
    if (bb == 0) { set_x_zero(); st.converged = true; return st; }
    st.relres = std::sqrt(h[1] / bb);
    if (st.relres < tol) { st.converged = true; return st; }  // already converged

    for (int it = 1; it <= maxit; ++it) {
        CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));

        k_apply<<<sgrid, sblock>>>(p_, Ap_, nx, ny, nz, &dots_[2]);
        KERNEL_CHECK();
        double pAp;
        CUDA_CHECK(cudaMemcpy(&pAp, &dots_[2], sizeof(double), cudaMemcpyDeviceToHost));
        Real alpha = (Real)(rz / pAp);

        k_update<<<egrid, TB1>>>(alpha, p_, Ap_, invd_, x_, r_, z_, n_, dots_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h, dots_, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        double rz_new = h[0], rr = h[1];

        st.iters = it;
        st.relres = std::sqrt(rr / bb);
        if (st.relres < tol) { st.converged = true; break; }

        Real beta = (Real)(rz_new / rz);
        rz = rz_new;
        k_update_p<<<egrid, TB1>>>(beta, z_, p_, n_);
        KERNEL_CHECK();
    }
    return st;
}

SolveResult gpu_pcg(const Model3D& m, double tol, int maxit, Precond pc) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    SolveResult res;
    {
        GpuPcg s(m, pc);
        s.set_x_zero();
        s.set_rhs_steady();
        PcgStats st = s.solve(tol, maxit);
        s.get_x(res.T);
        res.iters = st.iters;
        res.relres = st.relres;
        res.converged = st.converged;
    }  // frees the GPU buffers before the stop event, as before

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // also reports kernel execution errors
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    res.ms = ms;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}
