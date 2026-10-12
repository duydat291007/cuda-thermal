// gpu_jacobi.cu - CUDA Jacobi solvers.
//   naive: one thread per node, every neighbor read goes to global memory
//          (in practice mostly served by the L1/L2 caches).
//   tiled: each block loads its tile plus a 1-node halo into shared memory once,
//          then computes the stencil from shared memory.
#include "gpu_jacobi.h"
#include "cuda_check.h"
#include <utility>

constexpr int BX = 32;  // block width: one warp reads a contiguous row segment
constexpr int BY = 8;   // block height

__global__ void jacobi_naive(const float* __restrict__ T, float* __restrict__ Tn,
                             const float* __restrict__ q, int nx, int ny,
                             float src_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || j < 1 || i >= nx - 1 || j >= ny - 1) return;  // skip boundary
    int id = j * nx + i;
    Tn[id] = 0.25f * (T[id - 1] + T[id + 1] + T[id - nx] + T[id + nx]
                      + src_scale * q[id]);
}

__global__ void jacobi_tiled(const float* __restrict__ T, float* __restrict__ Tn,
                             const float* __restrict__ q, int nx, int ny,
                             float src_scale) {
    // Tile of BY x BX nodes plus a 1-node halo on every side.
    __shared__ float tile[BY + 2][BX + 2];

    int i = blockIdx.x * BX + threadIdx.x;
    int j = blockIdx.y * BY + threadIdx.y;
    int tx = threadIdx.x + 1;  // position inside the tile (offset by the halo)
    int ty = threadIdx.y + 1;

    // Every thread loads its own node.
    if (i < nx && j < ny) tile[ty][tx] = T[j * nx + i];

    // Threads on the block's edges also load the halo next to them.
    if (threadIdx.x == 0 && i > 0 && j < ny)
        tile[ty][0] = T[j * nx + i - 1];
    if (threadIdx.x == BX - 1 && i + 1 < nx && j < ny)
        tile[ty][BX + 1] = T[j * nx + i + 1];
    if (threadIdx.y == 0 && j > 0 && i < nx)
        tile[0][tx] = T[(j - 1) * nx + i];
    if (threadIdx.y == BY - 1 && j + 1 < ny && i < nx)
        tile[BY + 1][tx] = T[(j + 1) * nx + i];

    __syncthreads();  // the whole tile must be loaded before anyone reads it

    if (i < 1 || j < 1 || i >= nx - 1 || j >= ny - 1) return;  // after the barrier
    int id = j * nx + i;
    // Same summation order as the CPU and naive kernels, so results match exactly.
    Tn[id] = 0.25f * (tile[ty][tx - 1] + tile[ty][tx + 1] + tile[ty - 1][tx]
                      + tile[ty + 1][tx] + src_scale * q[id]);
}

GpuResult gpu_jacobi(const Problem& p, int iters, Kernel kernel) {
    const int nx = p.nx, ny = p.ny;
    const size_t n = (size_t)nx * ny;
    const size_t bytes = n * sizeof(float);
    const float src_scale = p.h * p.h / p.k;

    cudaEvent_t start, stop, k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));

    CUDA_CHECK(cudaEventRecord(start));  // total time includes transfers

    float *dT, *dTn, *dq;
    CUDA_CHECK(cudaMalloc(&dT, bytes));
    CUDA_CHECK(cudaMalloc(&dTn, bytes));
    CUDA_CHECK(cudaMalloc(&dq, bytes));
    CUDA_CHECK(cudaMemset(dT, 0, bytes));   // initial guess and boundary T = 0
    CUDA_CHECK(cudaMemset(dTn, 0, bytes));
    CUDA_CHECK(cudaMemcpy(dq, p.q.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BX, BY);
    dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY);

    CUDA_CHECK(cudaEventRecord(k_start));
    for (int it = 0; it < iters; ++it) {
        if (kernel == Kernel::Naive)
            jacobi_naive<<<grid, block>>>(dT, dTn, dq, nx, ny, src_scale);
        else
            jacobi_tiled<<<grid, block>>>(dT, dTn, dq, nx, ny, src_scale);
        KERNEL_CHECK();
        std::swap(dT, dTn);
    }
    CUDA_CHECK(cudaEventRecord(k_stop));

    GpuResult r;
    r.T.resize(n);
    CUDA_CHECK(cudaMemcpy(r.T.data(), dT, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    // Also reports any execution error from the kernels when SYNC_CHECK=0.
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&r.kernel_ms, k_start, k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&r.total_ms, start, stop));

    CUDA_CHECK(cudaFree(dT));
    CUDA_CHECK(cudaFree(dTn));
    CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));
    return r;
}
