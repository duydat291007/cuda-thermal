cmake_minimum_required(VERSION 3.24)
project(cuda_thermal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4060 (Ada Lovelace, sm_89)

# Debug-style error checking by default; configure with -DSYNC_CHECK=0 to benchmark.
set(SYNC_CHECK 1 CACHE STRING "1 = sync after every kernel launch, 0 = benchmark mode")

add_executable(thermal2d
    src/main.cu
    src/gpu_jacobi.cu
    src/cpu_jacobi.cpp)
target_compile_definitions(thermal2d PRIVATE SYNC_CHECK=${SYNC_CHECK})

add_executable(thermal3d
    src/main3d.cu
    src/pcg3d_gpu.cu
    src/pcg3d_cpu.cpp)
target_compile_definitions(thermal3d PRIVATE SYNC_CHECK=${SYNC_CHECK})

# Same solver in float, kept to demonstrate why the solver uses double.
add_executable(thermal3d_float
    src/main3d.cu
    src/pcg3d_gpu.cu
    src/pcg3d_cpu.cpp)
target_compile_definitions(thermal3d_float PRIVATE SYNC_CHECK=${SYNC_CHECK} REAL_FLOAT)
