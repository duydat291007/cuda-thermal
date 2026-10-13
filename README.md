# cuda-thermal

GPU thermal solver for 3D-stacked ICs, written in CUDA C++.

**Current milestone:** validated against 3D-ICE to its output precision
(0.0005 C), including TSV regions. Solves the temperature of a layered 3D chip stack with real
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
800 x 800 steady solve 488 -> 236 ms (these two include setup, see below).

**Launch overhead: scalars on the GPU, CUDA graphs.** PCG used to read 3 dot
products back to the CPU every iteration (each a blocking copy), and a multigrid
iteration is ~80 kernel launches, most of them tiny ones on the coarse levels.
Now alpha, beta, 1/dt and the dot products live in a small device array, one-thread
kernels compute alpha and beta, and the CPU waits once per iteration, for
||r||^2. Since no kernel argument changes between iterations any more, each half
of a multigrid iteration is recorded once as a CUDA graph and replayed with a
single launch. The benchmark now times setup (building levels, allocating,
uploading, recording graphs) separately from the solve. RTX 4060 laptop, Windows:

| grid | setup | solve, direct launches | solve, CUDA graphs | gain |
|---|---|---|---|---|
| 200 x 200 x 10 | 8.6 ms | 7.34 ms | 5.79 ms (8 iterations) | 1.27x |
| 256 x 256 x 10 | 14.2 ms | 9.09 ms | 6.80 ms (8 iterations) | 1.34x |
| 800 x 800 x 10 | 101 ms | 99.4 ms | 102.8 ms (9 iterations) | none |

Graphs help where launch overhead is a large share of an iteration, and not at
800 x 800, where the kernels themselves take ~11 ms per iteration. The table also
shows that half of the earlier "236 ms" at 800 x 800 was setup; the solve is
~100 ms. For the Jacobi preconditioner (4 kernels per iteration) graphs were no
faster, so only multigrid uses them. Graph replay matches direct launches to
1e-13 (verify 4 checks it through a transient, where 1/dt changes between replays).
Transient, 200 x 200, 107 steps to 2 s: 842 PCG iterations, 522 ms of solving
(0.62 ms per iteration).

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

## Validation against 3D-ICE
[3D-ICE](https://github.com/esl-epfl/3d-ice) (EPFL) is an established 3D-IC
thermal simulator. `thermal3d export3dice` writes a case in its input format on
the same grid (each of our planes becomes a 3D-ICE layer, each active layer a die),
and `thermal3d compare3dice` solves the case here and compares cell by cell.

`cases/validate_3dice` is the example stack set up so both tools solve the same
discrete problem: 1 plane per active layer (3D-ICE puts a die's power in one
source layer) and disjoint power blocks (3D-ICE requires them). Results:

| grid | max difference, both full dies | every block max / avg | 3D-ICE time, memory |
|---|---|---|---|
| 100 x 100 x 9 | 0.0005 C | within 0.0005 C | 6.6 s, 0.5 GB |
| 200 x 200 x 9 | 0.0005 C | within 0.0005 C | 66 s, 2.7 GB |

0.0005 C is the rounding of 3D-ICE's output (3 decimals), so the two agree to the
last digit it prints. (3D-ICE timings on a 2-core cloud machine with its
default single-threaded direct solver. On the RTX 4060 laptop, 3D-ICE takes 2.8 s
for the 100 x 100 case; ours takes 16 ms including setup.)

**TSVs.** 3D-ICE 4.0 can give a layer a layout file of rectangles made of other
(anisotropic) materials. The exporter writes one material per distinct TSV
fraction, with the same effective kxy, kz and rho*c as our model, and every TSV
cell as a rectangle, so partially covered cells and overlapping regions carry over
exactly. `cases/validate_tsv` adds three TSV regions to the bond layer: dense TSVs
under core0, and two crossing strips between the cores that don't line up with the
100 um cells. Result at 100 x 100 x 9: max difference 0.0005 C on both dies, every
block within 0.0005 C, and both tools show the TSVs cooling core0 from 73.98 C to
67.54 C. Limitation: 3D-ICE can't put materials in a die's source layer, so TSVs
can only be validated in inactive layers (the exporter refuses others).

One modeling difference was found and quantified: 3D-ICE couples the *bottom*
layer of the stack to its neighbor with its full height instead of half (its node
sits on the adiabatic bottom face). Without the 1 um base layer in
`validate_3dice`, 3D-ICE runs 0.25 C hotter under core0, as expected from that
extra half layer of silicon under 150 W/cm^2. The comparison also caught a bug
here: block statistics counted cells that only touched a block's edge (a ~1e-16
rounding overlap), inflating two block maxima by 0.5-0.7 C.

Building 3D-ICE (Ubuntu / WSL):
```
sudo apt install -y build-essential git unzip bison flex libopenblas-dev
git clone https://github.com/esl-epfl/3d-ice.git && cd 3d-ice
bash install-superlumt.sh && make -j8
```

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
mkdir cases\validate_3dice\ice100
thermal3d.exe export3dice cases\validate_3dice 100 cases\validate_3dice\ice100
   (run 3D-ICE-Emulator case.stk in that folder, from WSL)
thermal3d.exe compare3dice cases\validate_3dice 100 cases\validate_3dice\ice100
   (same three steps with cases\validate_tsv to validate the TSV regions)
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
| `src/case3d.h` | Case loader (stack / floorplan / TSV CSV files), built-in example, 3D-ICE export |
| `cases/` | Example cases |
| `src/pcg3d.h` | Solver interface: `CpuPcg` and `GpuPcg` objects, preconditioner choice |
| `src/mg3d.h` | Multigrid: level geometry, column (line) smoother, coarse operators |
| `src/pcg3d_cpu.cpp` | CPU reference PCG |
| `src/pcg3d_gpu.cu` | GPU PCG and multigrid: stencil, smoother, transfer and reduction kernels, device scalars, CUDA graphs |
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
   double PCG~~, ~~PCG scalars on the GPU, CUDA graphs~~. Next for speed: the
   kernels themselves at large grids (fewer or fused sweeps), and building the
   multigrid levels on the GPU (setup equals the solve time at 800 x 800)
6. ~~Power maps from files, TSVs (per-cell anisotropic materials)~~
7. ~~Validate against 3D-ICE~~, ~~including TSV regions~~ (here): agreement to
   its output precision
