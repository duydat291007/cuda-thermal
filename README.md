# cuda-thermal

GPU thermal solver for 3D-stacked ICs. **Current milestone:** 2D steady-state
heat conduction, solved by Jacobi iteration on the CPU (reference) and on the GPU
(naive and shared-memory tiled CUDA kernels).

## Test problem
`-k ∇²T = q` on the unit square, `T = 0` on the edges, with a manufactured
solution `T = sin(πx) sin(πy)`. Because the exact answer is known, the solver
reports its true error, not just whether CPU and GPU agree.

## Build and run (x64 Native Tools Command Prompt)
```
build.bat            # debug build, SYNC_CHECK=1
thermal2d.exe        # verify, then benchmark 1024 x 1024
build.bat bench      # benchmark build, SYNC_CHECK=0
thermal2d.exe 2048 1000
```
CMake alternative:
```
cmake -B build -DSYNC_CHECK=0
cmake --build build --config Release
```

## Files
| File | Purpose |
|---|---|
| `src/problem.h` | Grid, heat source, analytic solution |
| `src/cpu_jacobi.cpp` | CPU reference solver |
| `src/gpu_jacobi.cu` | CUDA kernel, launch loop, CUDA-event timing |
| `src/cuda_check.h` | `CUDA_CHECK` and `KERNEL_CHECK` macros |
| `src/main.cu` | Verification and benchmark driver |

## Roadmap
1. ~~2D CPU Jacobi reference~~
2. ~~Naive CUDA kernel~~ → shared-memory tiling (here)
3. 3D with per-layer k and ρc, convective top boundary
4. Transient, explicit time stepping
5. Implicit solver: matrix-free preconditioned conjugate gradient
6. Power maps from file, TSVs, anisotropic layers
7. Validate and benchmark against 3D-ICE
