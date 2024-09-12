import time

import numpy as np

from common import BurgersEquationProblem

N_RUNS = 100_000


problem = BurgersEquationProblem(N=4000)

u = np.random.random((N_RUNS, len(problem.u0)))
udot = np.empty_like(problem.u0)

tic = time.perf_counter()
for j in range(N_RUNS):
    problem.compute_rhs(0.0, u[j], udot, None)
toc = time.perf_counter()

print(f"Python, Burgers RHS evaluation {N_RUNS} times: {toc - tic:.4f}")
