// gpu_jacobi.h - GPU solver interface (no CUDA types, so .cpp files can include it).
#pragma once
#include <vector>
#include "problem.h"

enum class Kernel { Naive, Tiled };

inline const char* kernel_name(Kernel k) {
    return k == Kernel::Naive ? "naive" : "tiled";
}

struct GpuResult {
    std::vector<float> T;
    float kernel_ms = 0.0f;  // Jacobi iterations only
    float total_ms = 0.0f;   // including cudaMalloc, copies to and from the GPU
};

GpuResult gpu_jacobi(const Problem& p, int iters, Kernel kernel);
