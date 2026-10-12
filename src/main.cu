// main.cu - verification and benchmark driver.
//
//   thermal2d                 verify both kernels, then benchmark 1024 x 1024, 500 iters
//   thermal2d <n> <iters>     benchmark an n x n grid for <iters> iterations
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "cpu_jacobi.h"
#include "gpu_jacobi.h"
#include "problem.h"

static const Kernel kKernels[] = {Kernel::Naive, Kernel::Tiled};

// Small grid, iterated to convergence. Checks each GPU kernel against the CPU,
// and everything against the analytic solution.
// 67 is deliberately not a multiple of the block size, so partial edge blocks
// get tested too.
static bool verify() {
    const int n = 67, iters = 20000;
    printf("== verify: %d x %d grid, %d iterations ==\n", n, n, iters);
    Problem p = make_problem(n, n);

    std::vector<float> Tc = cpu_jacobi(p, iters);
    double cpu_vs_exact = max_abs_diff(Tc, p.T_exact);
    // The 5-point stencil has O(h^2) error, about 2e-4 here, so 1e-3 is a safe bound.
    bool ok = cpu_vs_exact < 1e-3;
    printf("  max |CPU - analytic|   = %.3e  %s\n", cpu_vs_exact, ok ? "PASS" : "FAIL");

    for (Kernel k : kKernels) {
        GpuResult g = gpu_jacobi(p, iters, k);
        double vs_cpu = max_abs_diff(g.T, Tc);
        bool pass = vs_cpu < 1e-4;
        printf("  max |%s - CPU|      = %.3e  %s\n", kernel_name(k), vs_cpu,
               pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    return ok;
}

static void benchmark(int n, int iters) {
    printf("\n== benchmark: %d x %d grid, %d iterations ==\n", n, n, iters);
    Problem p = make_problem(n, n);

    gpu_jacobi(p, 10, Kernel::Naive);  // warm-up: the first CUDA call pays context setup

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> Tc = cpu_jacobi(p, iters);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  CPU (1 thread)       : %10.2f ms\n", cpu_ms);

    // Ideal bytes moved per node per iteration: read T + read q + write Tn.
    double gb = 12.0 * n * n * iters / 1e9;
    for (Kernel k : kKernels) {
        GpuResult g = gpu_jacobi(p, iters, k);
        printf("  %-5s kernels only   : %10.2f ms  (%.1fx vs CPU, ~%.0f GB/s effective)\n",
               kernel_name(k), g.kernel_ms, cpu_ms / g.kernel_ms, gb / (g.kernel_ms / 1e3));
        printf("  %-5s incl. transfers: %10.2f ms   max |GPU - CPU| = %.3e\n",
               kernel_name(k), g.total_ms, max_abs_diff(g.T, Tc));
    }
}

int main(int argc, char** argv) {
    if (argc == 3) {
        benchmark(atoi(argv[1]), atoi(argv[2]));
        return 0;
    }
    if (!verify()) {
        printf("\nVerification FAILED - fix correctness before benchmarking.\n");
        return 1;
    }
    benchmark(1024, 500);
    return 0;
}
