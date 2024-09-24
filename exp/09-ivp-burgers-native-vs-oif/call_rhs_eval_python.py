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


def get_wrapper_for_compute_rhs_oif():
    def compute_rhs_wrapper(t, u_matrix, udot, p):
        compute_rhs_oif_numba_v4(t, u_matrix, udot, p)

    return compute_rhs_wrapper


def print_runtime(prefix, mean, ci):
    print(f"{prefix:32s} {mean:.3f} ± {ci:.3f}")


def benchmark_this(version_name, func, udot, u_matrix, p):
    func(0.0, u_matrix[0], udot, p)
    values = []
    for k in range(N_TRIALS):
        tic = time.perf_counter()
        for j in range(N_RUNS):
            func(0.0, u_matrix[j], udot, p)
        toc = time.perf_counter()
        values.append(toc - tic)
    mean, ci = compute_mean_and_ci(values)
    print_runtime(version_name, mean, ci)


def main():
    problem = BurgersEquationProblem(N=N)

    u_matrix = np.random.random((N_RUNS, len(problem.u0)))
    for i in range(N_RUNS):
        u_matrix[i] = problem.u0

    udot_v0 = np.empty_like(problem.u0)
    udot_v1 = np.empty_like(u_matrix[0])
    udot_v2 = np.empty_like(u_matrix[0])
    udot_v3 = np.empty_like(u_matrix[0])
    udot_v4 = np.empty_like(u_matrix[0])

    print(
        f"Python, accumulated runtime of {N_RUNS:n} RHS evals, "
        f"statistics from {N_TRIALS:n} trials"
    )
    print(f"Problem size is {len(udot_v0):n}")

    p = (problem.dx,)
    compute_rhs_oif_numba_v0 = problem.compute_rhs

    benchmark_this("Python + NumPy   ", compute_rhs_oif_numba_v0, udot_v0, u_matrix, p)
    benchmark_this("Python + Numba v1", compute_rhs_oif_numba_v1, udot_v1, u_matrix, p)
    benchmark_this("Python + Numba v2", compute_rhs_oif_numba_v2, udot_v2, u_matrix, p)
    benchmark_this("Python + Numba v3", compute_rhs_oif_numba_v3, udot_v3, u_matrix, p)
    benchmark_this("Python + Numba v4", compute_rhs_oif_numba_v4, udot_v4, u_matrix, p)

    npt.assert_allclose(udot_v0, udot_v1, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(udot_v0, udot_v2, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(udot_v0, udot_v3, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(udot_v0, udot_v4, rtol=1e-14, atol=1e-14)

    print(f"Leftmost udot value: {udot_v0[0]:.16f}")


if __name__ == "__main__":
    main()
