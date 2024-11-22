#!/usr/bin/env python

import subprocess

import numpy as np

from helpers import compute_mean_and_ci

SOLUTION_FILENAME_TPL = "_output/N={:04d}/solution-{:s}.txt"
RUNTIME_FILENAME_TPL = "_output/N={:04d}/runtime-{:s}.txt"

N = 3200

run_1 = subprocess.run(
    ["./call_from_c", "jl_diffeq", SOLUTION_FILENAME_TPL.format(N, "c")],
    encoding="utf-8",
    capture_output=True,
)
print(run_1.stdout)
print(run_1.stderr)
assert run_1.returncode == 0

run_2 = subprocess.run(
    ["julia", "call_ivp_julia.jl", SOLUTION_FILENAME_TPL.format(N, "julia")],
    encoding="utf-8",
)
assert run_2.returncode == 0

result_1 = np.loadtxt(SOLUTION_FILENAME_TPL.format(N, "c"))
result_2 = np.loadtxt(SOLUTION_FILENAME_TPL.format(N, "julia"))

np.testing.assert_allclose(result_1, result_2, rtol=1e-12, atol=1e-12)


runtimes_1 = np.loadtxt(RUNTIME_FILENAME_TPL, "c")
runtimes_2 = np.loadtxt(RUNTIME_FILENAME_TPL, "julia")

mean_1, ci_1 = compute_mean_and_ci(runtimes_1)
mean_2, ci_2 = compute_mean_and_ci(runtimes_2)

print(f"C mean runtime, sec: {mean_1:.3f} ± {ci_1:.3f}")
print(f"Julia mean runtime, sec: {mean_2:.3f} ± {ci_2:.3f}")
