// cpu_jacobi.h - CPU reference solver (the correctness oracle).
#pragma once
#include <vector>
#include "problem.h"

// Runs `iters` Jacobi sweeps starting from T = 0 and returns the final field.
std::vector<float> cpu_jacobi(const Problem& p, int iters);
