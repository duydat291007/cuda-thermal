@echo off
rem Build without CMake. Run from the x64 Native Tools Command Prompt.
rem   build.bat          debug build  (SYNC_CHECK=1)
rem   build.bat bench    benchmark build (SYNC_CHECK=0)
set SYNC=1
if /I "%1"=="bench" set SYNC=0
set FLAGS=-O3 -std=c++17 -arch=sm_89 -DSYNC_CHECK=%SYNC%

nvcc %FLAGS% src\main.cu src\gpu_jacobi.cu src\cpu_jacobi.cpp -o thermal2d.exe
if errorlevel 1 exit /b 1
nvcc %FLAGS% src\main3d.cu src\pcg3d_gpu.cu src\pcg3d_cpu.cpp -o thermal3d.exe
if errorlevel 1 exit /b 1
rem Same solver in float, kept to demonstrate why the solver uses double.
nvcc %FLAGS% -DREAL_FLOAT src\main3d.cu src\pcg3d_gpu.cu src\pcg3d_cpu.cpp -o thermal3d_float.exe
if errorlevel 1 exit /b 1
echo Built thermal2d.exe, thermal3d.exe and thermal3d_float.exe with SYNC_CHECK=%SYNC%

rem Live viewer (needs GLFW, https://www.glfw.org, "64-bit Windows binaries").
rem Set GLFW_DIR to where you unpacked it, or keep it at the default below.
if not defined GLFW_DIR set GLFW_DIR=C:\libs\glfw-3.5.1.bin.WIN64
if not exist "%GLFW_DIR%\include\GLFW\glfw3.h" (
    echo GLFW not found in %GLFW_DIR%: skipping thermal3d_view.exe
    exit /b 0
)
set GLFW_LIB=%GLFW_DIR%\lib-vc2026
if not exist "%GLFW_LIB%\glfw3dll.lib" set GLFW_LIB=%GLFW_DIR%\lib-vc2022
nvcc %FLAGS% -DGLFW_DLL -I"%GLFW_DIR%\include" src\viewer.cu src\pcg3d_gpu.cu src\pcg3d_cpu.cpp -o thermal3d_view.exe -L"%GLFW_LIB%" -lglfw3dll -lopengl32 -luser32 -lgdi32 -lshell32
if errorlevel 1 exit /b 1
copy /Y "%GLFW_LIB%\glfw3.dll" . >nul
echo Built thermal3d_view.exe (live viewer)
