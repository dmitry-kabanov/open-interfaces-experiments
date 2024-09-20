import time

import numba as nb
import numpy as np
import numpy.testing as npt

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci

N_TRIALS = 30
N_RUNS = 100_000


def print_runtime(prefix, mean, ci):
    print(f"{prefix:32s} {mean:.3f} ± {ci:.3f}")


# Note that `nopython=True` is default since Numba 0.59.
@nb.jit(
    # nb.types.void(nb.float64, nb.float64[:], nb.float64[:], nb.typeof((3.14,))),
    boundscheck=False,
    nogil=True,
)
def compute_rhs_oif_numba_v3(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p
    N = u.shape[0]

    local_ss = 0.0
    for i in range(N - 1):
        cand = abs(u[i])
        if cand > local_ss:
            local_ss = cand
    local_ss_rb = max(abs(u[0]), abs(u[-1]))

    dx_inv = 1.0 / dx

    f_cur = 0.5 * u[0] ** 2
    f_hat_lb = 0.5 * (f_cur + 0.5 * u[-1] ** 2) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_hat_prev = f_hat_lb
    for i in range(N - 1):
        f_next = 0.5 * u[i + 1] ** 2
        f_hat_cur = 0.5 * ((f_cur + f_next) - local_ss * (u[i + 1] - u[i]))
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur)
        f_hat_prev, f_cur = f_hat_cur, f_next

    f_hat_rb = f_hat_lb
    udot[-1] = dx_inv * (f_hat_prev - f_hat_rb)


problem = BurgersEquationProblem(N=4000)

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
optim_1 = nb.jit(
    nb.types.void(nb.float64, nb.float64[:], nb.float64[:], nb.typeof((3.14,))),
    # nb.types.void(
    #     nb.float64, nb.float64[:], nb.float64[:], nb.types.UniTuple(nb.float64, 1)
    # ),
    boundscheck=False,
    nogil=True,
)(compute_rhs_oif_numba_v3)

values_optim = []
p = (problem.dx,)
udot_test_numba_1 = np.empty_like(u[0])
optim_1(0.0, u[0], udot_test_numba_1, p)
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        optim_1(0.0, u[j], udot, p)
    toc = time.perf_counter()
    values_optim.append(toc - tic)
mean, ci = compute_mean_and_ci(values_optim)
print_runtime("Python + Numba v3, signature=yes", mean, ci)

# Timing optim version without signature.
optim_2 = nb.jit(
    boundscheck=False,
    nogil=True,
)(compute_rhs_oif_numba_v3)

values_optim = []
p = (problem.dx,)
udot_test_numba_2 = np.empty_like(u[0])
optim_2(0.0, u[0], udot_test_numba_2, p)
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        optim_2(0.0, u[j], udot, p)
    toc = time.perf_counter()
    values_optim.append(toc - tic)
mean, ci = compute_mean_and_ci(values_optim)
print_runtime("Python + Numba v3, signature=no", mean, ci)

npt.assert_allclose(udot_test_plain, udot_test_numba_1, rtol=1e-14, atol=1e-14)
npt.assert_allclose(udot_test_plain, udot_test_numba_2, rtol=1e-14, atol=1e-14)
print(f"Leftmost udot value: {udot[0]:.16f}")
