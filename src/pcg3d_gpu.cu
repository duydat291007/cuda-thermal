// pcg3d_gpu.cu - GPU preconditioned conjugate gradient, matrix-free.
//
// One PCG iteration with the Jacobi preconditioner is three kernels:
//   k_apply    Ap = (A + C/dt) p     and the dot product p . Ap
//   k_update   x += a p, r -= a Ap, z = r / diag   and r . z, r . r
//   k_update_p p = z + b p
// With the multigrid preconditioner, z = M^-1 r is one V-cycle instead (mg3d.h):
// k_update_xr updates x and r and writes a float copy of r for the V-cycle, the
// V-cycle runs in MgReal (float by default), and k_from_mg_dot converts its
// result back to double while computing r . z.
//
// Dot products are reduced inside each block (warp shuffles, then shared
// memory) and added across blocks with one double-precision atomicAdd per block.
// The host reads the scalars back each iteration to compute alpha and beta.
//
// The operator is stored per cell face (operator3d.h), in float. Multigrid
// level 0 holds the fine operator, which PCG uses in both preconditioner modes.
// Every buffer stays on the GPU between solves, so a transient run only moves
// the temperature field back to the host when it asks for it.
#include <algorithm>
#include <cmath>
#include "cuda_check.h"
#include "mg3d.h"
#include "operator3d.h"
#include "pcg3d.h"

using Op = CellOp<float>;

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

__global__ void k_invdiag(Op o, Real inv_dt, Real* invd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= o.nx || j >= o.ny) return;
    size_t id = ((size_t)l * o.ny + j) * o.nx + i;
    invd[id] = Real(1) / op_diag<float, Real>(o, id, i, j, l, inv_dt);
}

// b = P + C/dt x  (x holds the previous temperature)
__global__ void k_rhs_transient(const float* __restrict__ cap, Real inv_dt,
                                const Real* __restrict__ P, const Real* __restrict__ x,
                                Real* __restrict__ b, size_t n) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) b[c] = P[c] + (Real)cap[c] * inv_dt * x[c];
}

// Jacobi: r = b - (A + C/dt) x, z = M^-1 r, p = z
// dots[0] += r . z, dots[1] += r . r, dots[2] += b . b
__global__ void k_init(Op o, Real inv_dt, const Real* __restrict__ b,
                       const Real* __restrict__ x, const Real* __restrict__ invd,
                       Real* __restrict__ r, Real* __restrict__ z, Real* __restrict__ p,
                       double* dots) {
    __shared__ double s0[32], s1[32], s2[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double rz = 0.0, rr = 0.0, bb = 0.0;
    if (i < o.nx && j < o.ny) {
        size_t id = ((size_t)l * o.ny + j) * o.nx + i;
        Real bi = b[id];
        Real ri = bi - op_apply(o, x, i, j, l, inv_dt);
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

// Multigrid: r = b - (A + C/dt) x, b0 = r in MgReal;  dots[1] += r . r, dots[2] += b . b
__global__ void k_residual(Op o, Real inv_dt, const Real* __restrict__ b,
                           const Real* __restrict__ x, Real* __restrict__ r,
                           MgReal* __restrict__ b0, double* dots) {
    __shared__ double s1[32], s2[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double rr = 0.0, bb = 0.0;
    if (i < o.nx && j < o.ny) {
        size_t id = ((size_t)l * o.ny + j) * o.nx + i;
        Real bi = b[id];
        Real ri = bi - op_apply(o, x, i, j, l, inv_dt);
        r[id] = ri;
        b0[id] = (MgReal)ri;
        rr = (double)ri * ri;
        bb = (double)bi * bi;
    }
    block_sum_to(rr, s1, &dots[1]);
    block_sum_to(bb, s2, &dots[2]);
}

// Ap = (A + C/dt) p;  *pAp += p . Ap
__global__ void k_apply(Op o, Real inv_dt, const Real* __restrict__ p,
                        Real* __restrict__ Ap, double* pAp) {
    __shared__ double s[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double v = 0.0;
    if (i < o.nx && j < o.ny) {
        size_t id = ((size_t)l * o.ny + j) * o.nx + i;
        Real a = op_apply(o, p, i, j, l, inv_dt);
        Ap[id] = a;
        v = (double)p[id] * a;
    }
    block_sum_to(v, s, pAp);
}

// Jacobi: x += alpha p, r -= alpha Ap, z = M^-1 r;  dots[0] += r . z, dots[1] += r . r
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

// Multigrid: x += alpha p, r -= alpha Ap, b0 = r in MgReal;  *rr += r . r
__global__ void k_update_xr(Real alpha, const Real* __restrict__ p,
                            const Real* __restrict__ Ap, Real* x, Real* r,
                            MgReal* __restrict__ b0, size_t n, double* rr) {
    __shared__ double s[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (c < n) {
        x[c] += alpha * p[c];
        Real ri = r[c] - alpha * Ap[c];
        r[c] = ri;
        b0[c] = (MgReal)ri;
        v = (double)ri * ri;
    }
    block_sum_to(v, s, rr);
}

// z = V-cycle result x0 converted to Real;  *rz += r . z
__global__ void k_from_mg_dot(const MgReal* __restrict__ x0, const Real* __restrict__ r,
                              Real* __restrict__ z, size_t n, double* rz) {
    __shared__ double s[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (c < n) {
        Real zi = (Real)x0[c];
        z[c] = zi;
        v = (double)r[c] * zi;
    }
    block_sum_to(v, s, rz);
}

// p = z + beta p
__global__ void k_update_p(Real beta, const Real* __restrict__ z, Real* p, size_t n) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) p[c] = z[c] + beta * p[c];
}

// ---- multigrid kernels (MgReal precision) -----------------------------------

// Column factors for every column of a level.
__global__ void k_mg_factor(Op o, MgReal inv_dt, MgReal* fcp, MgReal* finv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= o.nx || j >= o.ny) return;
    mg_factor_column(o, inv_dt, fcp, finv, i, j);
}

// Red-black line smoother: solves every column with (i + j) % 2 == color.
// Threads are launched only for that color (x index t covers every other
// column), so no thread sits idle.
__global__ void k_mg_sweep(MgGrid<float, MgReal> g, const MgReal* __restrict__ b, MgReal* x,
                           int color) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = 2 * t + ((j + color) & 1);
    if (i >= g.op.nx || j >= g.op.ny) return;
    mg_line_solve(g, b, x, i, j);
}

// Coarse b = sum over the (up to 4) children of the fine residual b - A x.
// One thread per coarse cell, so the sum has a fixed order (no atomics).
__global__ void k_mg_restrict(Op f, MgReal inv_dt, const MgReal* __restrict__ fb,
                              const MgReal* __restrict__ fx, int cnx, int cny,
                              MgReal* __restrict__ cb) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int J = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (I >= cnx || J >= cny) return;
    const size_t fxy = (size_t)f.nx * f.ny;
    MgReal sum = 0;
    for (int j = 2 * J; j < min(2 * J + 2, f.ny); ++j)
        for (int i = 2 * I; i < min(2 * I + 2, f.nx); ++i) {
            size_t id = (size_t)l * fxy + (size_t)j * f.nx + i;
            sum += fb[id] - op_apply(f, fx, i, j, l, inv_dt);
        }
    cb[((size_t)l * cny + J) * cnx + I] = sum;
}

// Fine x += coarse x of the parent cell.
__global__ void k_mg_prolong(int fnx, int fny, MgReal* fx, int cnx, int cny,
                             const MgReal* __restrict__ cx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= fnx || j >= fny) return;
    fx[((size_t)l * fny + j) * fnx + i] += cx[((size_t)l * cny + j / 2) * cnx + i / 2];
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
    const size_t bytes = n_ * sizeof(Real);
    for (Real** ptr : {&P_, &x_, &b_, &r_, &z_, &p_, &Ap_, &invd_})
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    CUDA_CHECK(cudaMalloc(&dots_, 3 * sizeof(double)));

    // Operator levels: just the fine one for Jacobi, the whole hierarchy for multigrid.
    std::vector<MgLevelHost> hosts = mg_build_levels(m);
    if (pc_ == Precond::Jacobi) hosts.resize(1);
    for (size_t k = 0; k < hosts.size(); ++k) {
        const MgLevelHost& h = hosts[k];
        Level L;
        L.nx = h.nx;
        L.ny = h.ny;
        L.nz = h.nz;
        auto upload = [](const std::vector<float>& v, float** dst) {
            CUDA_CHECK(cudaMalloc(dst, v.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(*dst, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        };
        upload(h.gE, &L.gE);
        upload(h.gN, &L.gN);
        upload(h.gU, &L.gU);
        upload(h.gB, &L.gB);
        upload(h.cap, &L.cap);
        if (pc_ == Precond::Multigrid) {
            const size_t lb = (size_t)h.nx * h.ny * h.nz * sizeof(MgReal);
            for (MgReal** ptr : {&L.fcp, &L.finv, &L.b, &L.x}) CUDA_CHECK(cudaMalloc(ptr, lb));
        }
        levels_.push_back(L);
    }

    std::vector<Real> P = to_real(m.P);
    CUDA_CHECK(cudaMemcpy(P_, P.data(), bytes, cudaMemcpyHostToDevice));
    set_dt(0.0);
    set_x_zero();
}

GpuPcg::~GpuPcg() {
    // No CUDA_CHECK here: a destructor must not exit or throw.
    for (Real* ptr : {P_, x_, b_, r_, z_, p_, Ap_, invd_}) cudaFree(ptr);
    cudaFree(dots_);
    for (const Level& L : levels_) {
        for (float* ptr : {L.gE, L.gN, L.gU, L.gB, L.cap}) cudaFree(ptr);
        for (MgReal* ptr : {L.fcp, L.finv, L.b, L.x}) cudaFree(ptr);  // null is fine
    }
}

void GpuPcg::set_dt(double dt) {
    inv_dt_ = dt > 0 ? (Real)(1.0 / dt) : Real(0);
    const Op fine = levels_[0].op();
    k_invdiag<<<grid2d(fine.nx, fine.ny, fine.nz), block2d()>>>(fine, inv_dt_, invd_);
    KERNEL_CHECK();
    if (pc_ == Precond::Multigrid)
        for (const Level& L : levels_) {  // multigrid column factors
            k_mg_factor<<<grid2d(L.nx, L.ny, 1), block2d()>>>(L.op(), (MgReal)inv_dt_, L.fcp,
                                                              L.finv);
            KERNEL_CHECK();
        }
}

void GpuPcg::set_x_zero() { CUDA_CHECK(cudaMemset(x_, 0, n_ * sizeof(Real))); }

void GpuPcg::set_rhs_steady() {
    CUDA_CHECK(cudaMemcpy(b_, P_, n_ * sizeof(Real), cudaMemcpyDeviceToDevice));
}

void GpuPcg::set_rhs_transient() {
    k_rhs_transient<<<(unsigned)((n_ + TB1 - 1) / TB1), TB1>>>(levels_[0].cap, inv_dt_, P_, x_,
                                                                b_, n_);
    KERNEL_CHECK();
}

void GpuPcg::get_x(std::vector<Real>& T) const {
    T.resize(n_);
    CUDA_CHECK(cudaMemcpy(T.data(), x_, n_ * sizeof(Real), cudaMemcpyDeviceToHost));
}

// One multigrid V-cycle on level k (see mg3d.h): approximately solves
// (A + C/dt) x = b there, starting from x = 0.
void GpuPcg::vcycle(size_t k) {
    const Level& L = levels_[k];
    const MgReal inv_dt = (MgReal)inv_dt_;
    const MgGrid<float, MgReal> g{L.op(), L.fcp, L.finv};
    const dim3 blk = block2d();
    const dim3 g2 = grid2d((L.nx + 1) / 2, L.ny, 1);  // sweeps: one thread per same-color column
    CUDA_CHECK(cudaMemset(L.x, 0, (size_t)L.nx * L.ny * L.nz * sizeof(MgReal)));

    if (k + 1 == levels_.size()) {  // a single column: one line solve is exact
        k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, 0);
        KERNEL_CHECK();
        return;
    }
    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, 0);
    KERNEL_CHECK();
    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, 1);
    KERNEL_CHECK();

    const Level& C = levels_[k + 1];
    k_mg_restrict<<<grid2d(C.nx, C.ny, C.nz), blk>>>(L.op(), inv_dt, L.b, L.x, C.nx, C.ny, C.b);
    KERNEL_CHECK();
    vcycle(k + 1);
    k_mg_prolong<<<grid2d(L.nx, L.ny, L.nz), blk>>>(L.nx, L.ny, L.x, C.nx, C.ny, C.x);
    KERNEL_CHECK();

    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, 1);  // reverse order keeps it symmetric
    KERNEL_CHECK();
    k_mg_sweep<<<g2, blk>>>(g, L.b, L.x, 0);
    KERNEL_CHECK();
}

PcgStats GpuPcg::solve(double tol, int maxit) {
    const Op o = levels_[0].op();
    const dim3 sgrid = grid2d(o.nx, o.ny, o.nz), sblock = block2d();
    const unsigned egrid = (unsigned)((n_ + TB1 - 1) / TB1);
    double h[3];
    PcgStats st;

    if (pc_ == Precond::Multigrid) {
        MgReal* b0 = levels_[0].b;
        CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
        k_residual<<<sgrid, sblock>>>(o, inv_dt_, b_, x_, r_, b0, dots_);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h, dots_, 3 * sizeof(double), cudaMemcpyDeviceToHost));
        const double bb = h[2];
        if (bb == 0) { set_x_zero(); st.converged = true; return st; }
        st.relres = std::sqrt(h[1] / bb);
        if (st.relres < tol) { st.converged = true; return st; }

        vcycle(0);  // z = M^-1 r
        CUDA_CHECK(cudaMemset(dots_, 0, sizeof(double)));
        k_from_mg_dot<<<egrid, TB1>>>(levels_[0].x, r_, z_, n_, &dots_[0]);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(p_, z_, n_ * sizeof(Real), cudaMemcpyDeviceToDevice));
        double rz;
        CUDA_CHECK(cudaMemcpy(&rz, &dots_[0], sizeof(double), cudaMemcpyDeviceToHost));

        for (int it = 1; it <= maxit; ++it) {
            CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
            k_apply<<<sgrid, sblock>>>(o, inv_dt_, p_, Ap_, &dots_[2]);
            KERNEL_CHECK();
            double pAp;
            CUDA_CHECK(cudaMemcpy(&pAp, &dots_[2], sizeof(double), cudaMemcpyDeviceToHost));
            Real alpha = (Real)(rz / pAp);

            k_update_xr<<<egrid, TB1>>>(alpha, p_, Ap_, x_, r_, b0, n_, &dots_[1]);
            KERNEL_CHECK();
            double rr;
            CUDA_CHECK(cudaMemcpy(&rr, &dots_[1], sizeof(double), cudaMemcpyDeviceToHost));
            st.iters = it;
            st.relres = std::sqrt(rr / bb);
            if (st.relres < tol) { st.converged = true; break; }

            vcycle(0);
            k_from_mg_dot<<<egrid, TB1>>>(levels_[0].x, r_, z_, n_, &dots_[0]);
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
    CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
    k_init<<<sgrid, sblock>>>(o, inv_dt_, b_, x_, invd_, r_, z_, p_, dots_);
    KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h, dots_, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    double rz = h[0];
    const double bb = h[2];
    if (bb == 0) { set_x_zero(); st.converged = true; return st; }
    st.relres = std::sqrt(h[1] / bb);
    if (st.relres < tol) { st.converged = true; return st; }

    for (int it = 1; it <= maxit; ++it) {
        CUDA_CHECK(cudaMemset(dots_, 0, 3 * sizeof(double)));
        k_apply<<<sgrid, sblock>>>(o, inv_dt_, p_, Ap_, &dots_[2]);
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
