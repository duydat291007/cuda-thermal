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
// The GpuPcg object keeps every buffer on the GPU between solves, so a transient
// run only moves the temperature field back to the host when it asks for it.
#include <cmath>
#include "cuda_check.h"
#include "operator3d.h"
#include "pcg3d.h"

// Per-plane values. Every thread in a block works on the same plane, so
// constant memory serves each read to the whole warp at once.
struct PlaneParams {
    Real gx[MAXZ], gy[MAXZ], gzd[MAXZ], gzu[MAXZ], shift[MAXZ];
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

// ---------------------------------------------------------------------------

static dim3 stencil_block() { return dim3(TBX, TBY); }
static dim3 stencil_grid(const Model3D& m) {
    return dim3((m.nx + TBX - 1) / TBX, (m.ny + TBY - 1) / TBY, m.nz);
}

GpuPcg::GpuPcg(const Model3D& m) : m_(m), n_(m.ncells()) {
    const size_t bytes = n_ * sizeof(Real);
    for (Real** ptr : {&P_, &x_, &b_, &r_, &z_, &p_, &Ap_, &invd_})
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    CUDA_CHECK(cudaMalloc(&dots_, 3 * sizeof(double)));
    std::vector<Real> P = to_real(m.P);
    CUDA_CHECK(cudaMemcpy(P_, P.data(), bytes, cudaMemcpyHostToDevice));
    set_shift(std::vector<double>(m.nz, 0.0));
    set_x_zero();
}

GpuPcg::~GpuPcg() {
    // No CUDA_CHECK here: a destructor must not exit or throw.
    for (Real* ptr : {P_, x_, b_, r_, z_, p_, Ap_, invd_}) cudaFree(ptr);
    cudaFree(dots_);
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

PcgStats GpuPcg::solve(double tol, int maxit) {
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

SolveResult gpu_pcg(const Model3D& m, double tol, int maxit) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    SolveResult res;
    {
        GpuPcg s(m);
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
