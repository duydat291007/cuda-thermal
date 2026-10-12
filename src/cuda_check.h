// cuda_check.h - error checking for CUDA calls and kernel launches.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA API call: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t e_ = (x);                                                \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n  in: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(e_), #x);                   \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// SYNC_CHECK=1 (default, debug): synchronize after every launch so errors
// that happen while a kernel runs are reported at the launch that caused them.
// SYNC_CHECK=0 (benchmark builds): only launch-configuration errors are caught
// here; execution errors surface at the next CUDA_CHECK(cudaEventSynchronize(...)).
#ifndef SYNC_CHECK
#define SYNC_CHECK 1
#endif

// Put right after every kernel launch: kernel<<<g, b>>>(...); KERNEL_CHECK();
#define KERNEL_CHECK()                                   \
    do {                                                 \
        CUDA_CHECK(cudaGetLastError());                  \
        if (SYNC_CHECK) CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
