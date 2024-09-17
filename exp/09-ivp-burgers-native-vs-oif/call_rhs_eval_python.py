import time

import numba as nb
import numpy as np

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci

N_TRIALS = 30
N_RUNS = 100_000


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
    # local_ss = np.amax(np.abs(u))
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
udot = np.empty_like(problem.u0)

print(f"Python, accumulated {N_RUNS} RHS evals, averaged over {N_TRIALS} trials")

# Timing plain version
problem.compute_rhs(0.0, u[0], udot, None)
values_plain = []
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        problem.compute_rhs(0.0, u[j], udot, None)
    toc = time.perf_counter()
    values_plain.append(toc - tic)
mean, ci = compute_mean_and_ci(values_plain)
print(f"Python + NumPy: {mean:.3f} ± {ci:.3f}")

# Timing optim version
values_optim = []
p = (problem.dx,)
compute_rhs_oif_numba_v3(0.0, u[0], udot, p)
for k in range(N_TRIALS):
    tic = time.perf_counter()
    for j in range(N_RUNS):
        compute_rhs_oif_numba_v3(0.0, u[j], udot, p)
    toc = time.perf_counter()
    values_optim.append(toc - tic)
mean, ci = compute_mean_and_ci(values_optim)
print(f"Python + Numba: {mean:.3f} ± {ci:.3f}")
