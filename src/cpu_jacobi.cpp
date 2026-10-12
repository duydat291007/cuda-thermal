#include "cpu_jacobi.h"
#include <utility>

// One Jacobi sweep of the 5-point stencil:
//   T_new(i,j) = ( T(i-1,j) + T(i+1,j) + T(i,j-1) + T(i,j+1) + h^2 q / k ) / 4
// Boundary nodes are never written, so they stay at T = 0.
std::vector<float> cpu_jacobi(const Problem& p, int iters) {
    const int nx = p.nx, ny = p.ny;
    const float src_scale = p.h * p.h / p.k;
    std::vector<float> T((size_t)nx * ny, 0.0f);
    std::vector<float> Tn((size_t)nx * ny, 0.0f);

    for (int it = 0; it < iters; ++it) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                size_t id = (size_t)j * nx + i;
                Tn[id] = 0.25f * (T[id - 1] + T[id + 1] + T[id - nx] + T[id + nx]
                                  + src_scale * p.q[id]);
            }
        }
        std::swap(T, Tn);
    }
    return T;
}
