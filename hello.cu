#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 8>>>();  // Launch one block containing eight threads.

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}