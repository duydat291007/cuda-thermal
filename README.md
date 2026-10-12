# cuda-thermal

GPU thermal solver for 3D-stacked ICs, written in CUDA C++.

**Current milestone:** steady-state temperature of a layered 3D chip stack,
solved with a matrix-free preconditioned conjugate gradient (PCG) method on the
CPU (reference) and the GPU.

## Programs
| Program | What it solves |
|---|---|
| `thermal3d` | 3D layered stack: per-layer conductivity, heat sink on top, power maps with hotspots |
| `thermal2d` | 2D model problem used to learn stencil kernels (Jacobi, naive vs shared-memory tiled) |

## The 3D model
A two-tier stack on a 10 mm x 10 mm footprint, bottom to top:

| Layer | Thickness | k (W/m K) | Power |
|---|---|---|---|
| Si tier 2 (bottom) | 50 um | 130 | 25 W (incl. 3 W in a 1 mm hotspot) |
| bond + BEOL | 10 um | 2 | |
| Si tier 1 (top) | 50 um | 130 | 25 W |
| TIM | 25 um | 5 | |
| Cu spreader | 1 mm | 400 | |

The top connects to a heat sink (h = 2e4 W/m²K, i.e. 0.5 K/W); the sides and
bottom are adiabatic. Cells exchange heat through conductances; across material
interfaces the two half-cells are in series (harmonic mean of k). Layers can use
different cell thicknesses.

**Checks:** with uniform power the exact answer is a 1D resistor chain, so the
solver is compared against it. With a hotspot it must conserve energy (heat out
through the sink equals power in) and the GPU must match the CPU.

## Build and run (x64 Native Tools Command Prompt)
```
build.bat            # debug build, SYNC_CHECK=1
thermal3d.exe        # checks, stack temperatures, benchmark 256 x 256
build.bat bench      # benchmark build, SYNC_CHECK=0
thermal3d.exe 512    # benchmark a 512 x 512 grid (GPU only above 256)
```
CMake alternative:
```
cmake -B build -DSYNC_CHECK=0
cmake --build build --config Release
```

## Files
| File | Purpose |
|---|---|
| `src/stack3d.h` | Layers, conductances, power maps, exact 1D solution, energy balance |
| `src/operator3d.h` | 7-point operator shared by CPU and GPU |
| `src/pcg3d_cpu.cpp` | CPU reference PCG |
| `src/pcg3d_gpu.cu` | GPU PCG: stencil, update and reduction kernels |
| `src/main3d.cu` | 3D verification and benchmark driver |
| `src/cuda_check.h` | `CUDA_CHECK` and `KERNEL_CHECK` macros |
| `src/problem.h`, `src/*jacobi*`, `src/main.cu` | 2D Jacobi model problem |

## Roadmap
1. ~~2D CPU Jacobi reference~~
2. ~~CUDA Jacobi kernels: naive and shared-memory tiled~~
3. ~~3D layered stack, steady state, PCG solver~~ (here). Jacobi needed ~90x more
   iterations than PCG on this stack, so the implicit solver moved up.
4. Transient: backward Euler time stepping with PCG
5. Faster PCG: keep scalars on the GPU, fuse kernels, better preconditioner
6. Power maps from file, TSVs, anisotropic layers
7. Validate and benchmark against 3D-ICE
