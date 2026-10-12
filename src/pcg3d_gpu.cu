// pcg3d_gpu.cu - GPU preconditioned conjugate gradient, matrix-free.
//
// One PCG iteration is three kernels:
//   k_apply    Ap = A p           and the dot product p . Ap
//   k_update   x += a p, r -= a Ap, z = r / diag   and r . z, r . r
//   k_update_p p = z + b p
// Dot products are reduced inside each block (warp shuffles, then shared
// memory) and added across blocks with one double-precision atomicAdd per block.
// The host reads the scalars back each iteration to compute alpha and beta.
#include <cmath>
#include "cuda_check.h"
#include "operator3d.h"
#include "pcg3d.h"

// Per-plane conductances. Every thread in a block works on the same plane, so
// constant memory serves each read to the whole warp at once.
struct PlaneParams {
    float gx[MAXZ], gy[MAXZ], gzd[MAXZ], gzu[MAXZ];
};
__constant__ PlaneParams c_pp;

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

__global__ void k_invdiag(float* invd, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    if (i >= nx || j >= ny) return;
    invd[((size_t)l * ny + j) * nx + i] =
        1.0f / diag_cell(i, j, nx, ny, c_pp.gx[l], c_pp.gy[l], c_pp.gzd[l], c_pp.gzu[l]);
}

// r = b, z = M^-1 r, p = z, x = 0;  dots[0] += r . z, dots[1] += b . b
__global__ void k_init(const float* __restrict__ b, const float* __restrict__ invd,
                       float* x, float* r, float* z, float* p, size_t n, double* dots) {
    __shared__ double s0[32], s1[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double rz = 0.0, bb = 0.0;
    if (c < n) {
        float bi = b[c], zi = bi * invd[c];
        x[c] = 0.0f; r[c] = bi; z[c] = zi; p[c] = zi;
        rz = (double)bi * zi;
        bb = (double)bi * bi;
    }
    block_sum_to(rz, s0, &dots[0]);
    block_sum_to(bb, s1, &dots[1]);
}

// Ap = A p;  *pAp += p . Ap
__global__ void k_apply(const float* __restrict__ p, float* __restrict__ Ap,
                        int nx, int ny, int nz, double* pAp) {
    __shared__ double s[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.z;
    double v = 0.0;
    if (i < nx && j < ny) {
        size_t id = ((size_t)l * ny + j) * nx + i;
        float a = apply_cell(p, i, j, l, nx, ny, nz,
                             c_pp.gx[l], c_pp.gy[l], c_pp.gzd[l], c_pp.gzu[l]);
        Ap[id] = a;
        v = (double)p[id] * a;
    }
    block_sum_to(v, s, pAp);
}

// x += alpha p, r -= alpha Ap, z = M^-1 r;  dots[0] += r . z, dots[1] += r . r
__global__ void k_update(float alpha, const float* __restrict__ p,
                         const float* __restrict__ Ap, const float* __restrict__ invd,
                         float* x, float* r, float* z, size_t n, double* dots) {
    __shared__ double s0[32], s1[32];
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double rz = 0.0, rr = 0.0;
    if (c < n) {
        x[c] += alpha * p[c];
        float ri = r[c] - alpha * Ap[c];
        float zi = ri * invd[c];
        r[c] = ri; z[c] = zi;
        rz = (double)ri * zi;
        rr = (double)ri * ri;
    }
    block_sum_to(rz, s0, &dots[0]);
    block_sum_to(rr, s1, &dots[1]);
}

// p = z + beta p
__global__ void k_update_p(float beta, const float* __restrict__ z, float* p, size_t n) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) p[c] = z[c] + beta * p[c];
}

SolveResult gpu_pcg(const Model3D& m, double tol, int maxit) {
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const size_t n = m.ncells();
    const size_t bytes = n * sizeof(float);

    PlaneParams pp = {};
    for (int l = 0; l < nz; ++l) {
        pp.gx[l] = m.gx[l]; pp.gy[l] = m.gy[l];
        pp.gzd[l] = m.gzd[l]; pp.gzu[l] = m.gzu[l];
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpyToSymbol(c_pp, &pp, sizeof(pp)));
    float *b, *x, *r, *z, *p, *Ap, *invd;
    double* dots;  // [0] r.z  [1] r.r (or b.b at init)  [2] p.Ap
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&r, bytes));
    CUDA_CHECK(cudaMalloc(&z, bytes));
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMalloc(&Ap, bytes));
    CUDA_CHECK(cudaMalloc(&invd, bytes));
    CUDA_CHECK(cudaMalloc(&dots, 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(b, m.P.data(), bytes, cudaMemcpyHostToDevice));

    dim3 sblock(TBX, TBY);
    dim3 sgrid((nx + TBX - 1) / TBX, (ny + TBY - 1) / TBY, nz);
    unsigned egrid = (unsigned)((n + TB1 - 1) / TB1);

    k_invdiag<<<sgrid, sblock>>>(invd, nx, ny);
    KERNEL_CHECK();

    double h[3];
    CUDA_CHECK(cudaMemset(dots, 0, 3 * sizeof(double)));
    k_init<<<egrid, TB1>>>(b, invd, x, r, z, p, n, dots);
    KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h, dots, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    double rz = h[0];
    const double bb = h[1];

    SolveResult res;
    if (bb == 0) res.converged = true;
    for (int it = 1; it <= maxit && !res.converged; ++it) {
        CUDA_CHECK(cudaMemset(dots, 0, 3 * sizeof(double)));

        k_apply<<<sgrid, sblock>>>(p, Ap, nx, ny, nz, &dots[2]);
        KERNEL_CHECK();
        double pAp;
        CUDA_CHECK(cudaMemcpy(&pAp, &dots[2], sizeof(double), cudaMemcpyDeviceToHost));
        float alpha = (float)(rz / pAp);

        k_update<<<egrid, TB1>>>(alpha, p, Ap, invd, x, r, z, n, dots);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h, dots, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        double rz_new = h[0], rr = h[1];

        res.iters = it;
        res.relres = std::sqrt(rr / bb);
        if (res.relres < tol) { res.converged = true; break; }

        float beta = (float)(rz_new / rz);
        rz = rz_new;
        k_update_p<<<egrid, TB1>>>(beta, z, p, n);
        KERNEL_CHECK();
    }

    res.T.resize(n);
    CUDA_CHECK(cudaMemcpy(res.T.data(), x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // also reports kernel execution errors
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    res.ms = ms;

    for (float* ptr : {b, x, r, z, p, Ap, invd}) CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaFree(dots));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}
