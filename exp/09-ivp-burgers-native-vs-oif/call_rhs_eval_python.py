import time

import numpy as np
import numpy.testing as npt
from rhsversions import (
    compute_rhs_oif_numba_v1,
    compute_rhs_oif_numba_v2,
    compute_rhs_oif_numba_v3,
    compute_rhs_oif_numba_v4,
)

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci

N_TRIALS = 30
N_RUNS = 100_000
N = 3200


def print_runtime(prefix, mean, ci):
    print(f"{prefix:32s} {mean:.3f} ± {ci:.3f}")


problem = BurgersEquationProblem(N=N)

u = np.random.random((N_RUNS, len(problem.u0)))
for i in range(N_RUNS):
    u[i] = problem.u0

udot = np.empty_like(problem.u0)

print(
    f"Python, accumulated runtime of {N_RUNS:n} RHS evals, "
    f"statistics from {N_TRIALS:n} trials"
)
print(f"Problem size is {len(udot):n}")

# Timing plain version
udot_test_plain = np.empty_like(problem.u0)
problem.compute_rhs(0.0, u[0], udot_test_plain, None)
values_plain = []
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        problem.compute_rhs(0.0, u[j], udot, None)
    toc = time.perf_counter()
    values_plain.append(toc - tic)
mean, ci = compute_mean_and_ci(values_plain)
print_runtime("Python + NumPy", mean, ci)

# Timing optim version

values_optim = []
p = (problem.dx,)
udot_test_numba_1 = np.empty_like(u[0])
compute_rhs_oif_numba_v3(0.0, u[0], udot_test_numba_1, p)
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        compute_rhs_oif_numba_v3(0.0, u[j], udot, p)
    toc = time.perf_counter()
    values_optim.append(toc - tic)
mean, ci = compute_mean_and_ci(values_optim)
print_runtime("Python + Numba v3", mean, ci)

# Timing optim version without signature.
values_optim = []
p = (problem.dx,)
udot_test_numba_2 = np.empty_like(u[0])
compute_rhs_oif_numba_v4(0.0, u[0], udot_test_numba_2, p)
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        compute_rhs_oif_numba_v4(0.0, u[j], udot, p)
    toc = time.perf_counter()
    values_optim.append(toc - tic)
mean, ci = compute_mean_and_ci(values_optim)
print_runtime("Python + Numba v4", mean, ci)

npt.assert_allclose(udot_test_plain, udot_test_numba_1, rtol=1e-14, atol=1e-14)
npt.assert_allclose(udot_test_plain, udot_test_numba_2, rtol=1e-14, atol=1e-14)
print(f"Leftmost udot value: {udot[0]:.16f}")
