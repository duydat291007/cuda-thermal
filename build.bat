@echo off
rem Build without CMake. Run from the x64 Native Tools Command Prompt.
rem   build.bat          debug build  (SYNC_CHECK=1)
rem   build.bat bench    benchmark build (SYNC_CHECK=0)
set SYNC=1
if /I "%1"=="bench" set SYNC=0
nvcc -O3 -std=c++17 -arch=sm_89 -DSYNC_CHECK=%SYNC% ^
     src\main.cu src\gpu_jacobi.cu src\cpu_jacobi.cpp -o thermal2d.exe
if errorlevel 1 exit /b 1
echo Built thermal2d.exe with SYNC_CHECK=%SYNC%
