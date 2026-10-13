# cuda-thermal

GPU thermal solver for 3D-stacked ICs, written in CUDA C++.

**Current milestone:** temperature of a layered 3D chip stack with real
floorplans and TSVs, steady state and transient (heat-up after power-on), solved
with a matrix-free preconditioned conjugate gradient (PCG) method on the CPU
(reference) and the GPU, with a geometric multigrid preconditioner.

## Programs
| Program | What it solves |
|---|---|
| `thermal3d` | 3D layered stack: per-layer conductivity, heat sink on top, power maps with hotspots |
| `thermal2d` | 2D model problem used to learn stencil kernels (Jacobi, naive vs shared-memory tiled) |

## The 3D model
A two-tier stack on a 10 mm x 10 mm footprint, bottom to top:

| Layer | Thickness | k (W/m K) | rho*c (J/m³K) | Power |
|---|---|---|---|---|
| Si tier 2 (bottom) | 50 um | 130 | 1.66e6 | 25 W (incl. 3 W in a 1 mm hotspot) |
| bond + BEOL | 10 um | 2 | 1.70e6 | |
| Si tier 1 (top) | 50 um | 130 | 1.66e6 | 25 W |
| TIM | 25 um | 5 | 2.00e6 | |
| Cu spreader | 1 mm | 400 | 3.45e6 | |

The top connects to a heat sink (h = 2e4 W/m²K, i.e. 0.5 K/W); the sides and
bottom are adiabatic. Cells exchange heat through conductances; across material
interfaces the two half-cells are in series (harmonic mean of k). Layers can use
different cell thicknesses.

**Checks:** with uniform power the exact answer is a 1D resistor chain, so the
solver is compared against it. Every solve also recomputes the true residual
`||P - A T|| / ||P||` from scratch, and must conserve energy (heat out through
the sink equals power in). The GPU must match the CPU.

**Transient:** backward Euler time stepping, `(C/dt + A) T_new = C/dt T_old + P`,
one PCG solve per step warm-started from the previous temperature. Time steps
start at 10 us and grow 10% per step, because the stack has very different time
scales: the hotspot reaches half its final rise in ~4 ms (silicon heating up
locally), then climbs slowly for ~0.5 s while the copper spreader and heat sink
warm up. Each step must conserve energy (power in = heat out + heat stored), and
a long run must settle to the steady-state solution.

**Multigrid preconditioner** (`src/mg3d.h`): one V-cycle per PCG iteration.
PCG then converges in ~10 iterations at every grid size (64 x 64 to 800 x 800),
instead of the 400-3200 that the Jacobi preconditioner needs. The design follows
from the stack's physics:
- vertical coupling in thin silicon is ~100x the lateral coupling, so the
  smoother solves each vertical column exactly (tridiagonal solve, red-black order);
- so coarse levels only coarsen in x and y, merging 2 x 2 columns, all the way
  down to a single column, which handles the weakly damped "whole stack warms up"
  mode exactly;
- coarse operators are rebuilt from the coarse cell sizes (odd sizes allowed), and
  restriction is the transpose of prolongation, so the V-cycle stays symmetric.

Profiling (Nsight Systems, 800 x 800) showed the smoother taking ~6.6 ms per
pass, ~10x its memory-bandwidth limit: it did ~6 FP64 divisions per cell (FP64
runs at 1/64 speed on GeForce) and half its threads idled in red-black order. Now
the conductances and each column's tridiagonal factors are precomputed per level
whenever the time step changes, the smoother only multiplies, and threads are
launched only for the active color. Result: smoother 6.6 -> 1.4 ms per pass,
800 x 800 steady solve 488 -> 236 ms.

**Mixed precision:** PCG runs in double, the multigrid V-cycle in float. The
V-cycle only proposes a search direction and PCG measures its effect in double,
so float rounding in the preconditioner never accumulates in the solution (unlike
the all-float solver above). Measured on 200-800 grids: identical iteration counts
down to 1e-10, temperatures equal to 1e-13 K. Float halves the V-cycle's memory
traffic and runs ~64x faster arithmetic on GeForce. Build with `-DMG_DOUBLE` to
run the V-cycle in double for comparison.

**Precision:** the solver runs in double. Neighboring cells in the thin silicon
layers differ by ~1e-5 K while T is ~50 K, only a few float ulps apart. In float,
PCG's tracked residual reaches 1e-6 while the true residual stays at 1e-3 (64 x 64)
to 0.4 (800 x 800), and the 800 x 800 temperatures came out 0.6 C too low.
`thermal3d_float.exe` is the same solver in float, kept to show this.

**Grid convergence** (double, bottom-tier peak): 79.80 C at 200 x 200,
79.81 C at 400 x 400, 79.82 C at 800 x 800. Use sizes where the 1 mm hotspot
is a whole number of cells (200, 400, 800).

## Cases: your own stack, floorplan and TSVs
A case is a folder of CSV files (see `src/case3d.h` for the format):
- `stack.csv`: layers bottom to top (thickness, k, rho*c, planes, active), plus die
  size, heat-sink h and ambient temperature
- `floorplan.csv`: power blocks (name, layer, position, size, W)
- `tsv.csv` (optional): TSV regions (layer, position, size, copper fraction)

`thermal3d case <folder> [n]` prints per-block max/average temperatures, the
change each block sees from the TSVs (by also solving without them), the hottest
point, and writes `tmap_layer<k>.csv` heat maps of each active layer.

Materials are stored per cell, so the operator uses per-face conductances
(harmonic mean across material changes). TSV regions are an effective medium:
vertical k = f k_Cu + (1-f) k (parallel), lateral k from the Rayleigh / Maxwell
Garnett formula, rho*c mixed by volume. Multigrid builds coarse levels by adding
up the fine faces (scaled by center distances for lateral faces); for uniform
layers this equals the coarse-grid operator, and with TSVs it still converges in
~11 iterations.

Two examples: `cases/example` (logic die with 4 cores at 150 W/cm^2 under a DRAM
die, TSV fields *beside* the cores) and `cases/tsv_under_core0` (the same copper
fraction *under* core0). Beside the cores the TSVs change core temperatures by
~0.1 C, because heat can't spread sideways far through 50 um of silicon; under
core0 they cool it by ~6.6 C, by bypassing the 10 um bond layer (k = 2) that
otherwise drops ~7.5 K at that power density, while warming the DRAM above by
0.2 C.

## Build and run (x64 Native Tools Command Prompt)
```
build.bat            # debug build, SYNC_CHECK=1
thermal3d.exe        # checks, stack temperatures, benchmark 256 x 256
build.bat bench      # benchmark build, SYNC_CHECK=0
thermal3d.exe 400    # steady benchmark, Jacobi vs multigrid (GPU only above 256)
thermal3d.exe transient       # heat-up timeline on 200 x 200, writes transient.csv
thermal3d.exe transient 400   # same on a finer grid
thermal3d.exe transient 200 jacobi   # same with the Jacobi preconditioner
thermal3d.exe case cases\example 200          # block temperatures, TSV effect, heat maps
thermal3d.exe case cases\tsv_under_core0 200
thermal3d_float.exe 800   # the float version, to see the precision failure
```
CMake alternative:
```
cmake -B build -DSYNC_CHECK=0
cmake --build build --config Release
```

## Files
| File | Purpose |
|---|---|
| `src/stack3d.h` | Layers, per-cell materials, TSVs, face conductances, power blocks, energy balance |
| `src/operator3d.h` | 7-point per-face operator shared by CPU, GPU and all multigrid levels |
| `src/case3d.h` | Case loader (stack / floorplan / TSV CSV files) and the built-in example |
| `cases/` | Example cases |
| `src/pcg3d.h` | Solver interface: `CpuPcg` and `GpuPcg` objects, preconditioner choice |
| `src/mg3d.h` | Multigrid: level geometry, column (line) smoother, coarse operators |
| `src/pcg3d_cpu.cpp` | CPU reference PCG |
| `src/pcg3d_gpu.cu` | GPU PCG and multigrid: stencil, smoother, transfer and reduction kernels |
| `src/transient3d.h` | Backward Euler time stepping, works with either solver |
| `src/main3d.cu` | 3D verification and benchmark driver |
| `src/cuda_check.h` | `CUDA_CHECK` and `KERNEL_CHECK` macros |
| `src/problem.h`, `src/*jacobi*`, `src/main.cu` | 2D Jacobi model problem |

## Roadmap
1. ~~2D CPU Jacobi reference~~
2. ~~CUDA Jacobi kernels: naive and shared-memory tiled~~
3. ~~3D layered stack, steady state, PCG solver in double~~. Jacobi needed
   ~90x more iterations than PCG on this stack, so the implicit solver moved up.
4. ~~Transient: backward Euler time stepping with PCG~~
5. ~~Multigrid preconditioner~~, ~~profiled and optimized~~, ~~float V-cycle in
   double PCG~~. Next for speed: pinned host memory, keep PCG scalars on
   the GPU, batch the small coarse-level launches (CUDA graphs)
6. ~~Power maps from files, TSVs (per-cell anisotropic materials)~~ (here)
7. Validate and benchmark against 3D-ICE
