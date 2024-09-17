"""Run jl_diffeq via OIF from Python to assess its runtime."""

import argparse
import csv
import dataclasses
import os
import subprocess
import time

import numba as nb
import numpy as np
from oif.interfaces.ivp import IVP
from scipy import integrate

from common import BurgersEquationProblem
from helpers import get_outdir

RESOLUTIONS_LIST = [800, 1600, 3200]
N_RUNS = 30

OUTDIR = get_outdir()
RESULT_JL_DIFFEQ_PYTHON_FILENAME = (
    OUTDIR / "runtime_vs_resolution_python_jl_diffeq_numba.csv"
)


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


@dataclasses.dataclass
class Args:
    impl: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["jl_diffeq", "native"],
        default="jl_diffeq",
        nargs="?",
    )
    args = p.parse_args()
    return Args(**vars(args))


def measure_perf_once(N):
    args = _parse_args()
    impl = args.impl
    print(f"Implementation: {impl}")
    problem = BurgersEquationProblem(N=N)
    t0 = 0.0
    y0 = problem.u0
    p = (problem.dx,)

    oif_rhs_numba = compute_rhs_oif_numba_v3

    if impl == "native":
        s = integrate.ode(problem.compute_rhs_scipy_ode)
        s.set_initial_value(y0, t0)
        s.set_integrator("dopri5", rtol=1e-6, atol=1e-12)
    else:
        s = IVP(impl)
        s.set_initial_value(y0, t0)
        s.set_user_data(p)
        s.set_rhs_fn(oif_rhs_numba)
        s.set_tolerances(1e-6, 1e-12)
        assert impl == "jl_diffeq"
        s.set_integrator("DP5")

    times = np.linspace(problem.t0, problem.tfinal, num=101)

    s.set_initial_value(y0, t0)
    tic = time.perf_counter()
    for t in times[1:]:
        s.integrate(t)
    toc = time.perf_counter()
    runtime = toc - tic
    solution_last = s.y

    return runtime, problem.x, problem.u0, solution_last


def main():
    if not data_are_present():
        compute()
    else:
        process()


def data_are_present():
    if os.path.isfile(RESULT_JL_DIFFEQ_PYTHON_FILENAME):
        return True

    return False


def compute():
    # Run once to warm up the Julia's interpreter
    measure_perf_once(RESOLUTIONS_LIST[0])

    table = {}
    for N in RESOLUTIONS_LIST:
        print()
        print(f"Resolution N = {N}")
        print(f"Measure performance {N_RUNS} times")

        elapsed_times = []
        solution_last = None

        for k in range(N_RUNS):
            (runtime, grid, solution_init, solution_last) = measure_perf_once(N)
            elapsed_times.append(runtime)

        mean = np.mean(elapsed_times)
        sem = np.std(elapsed_times, ddof=1) / np.sqrt(len(elapsed_times))
        ci = 2.0 * sem  # Coefficient corresponds to the 95% Confidence Interval.
        print(f"Runtime, sec: {mean:.4f} ± {ci:.4f}")
        print(f"Solution second point from the left value: {solution_last[1]:.16f}")

        table[N] = f"{mean:.2f} ± {ci:.2f}"

    with open(RESULT_JL_DIFFEQ_PYTHON_FILENAME, "w", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method/resolution"] + RESOLUTIONS_LIST)
        runtimes = []
        for N in RESOLUTIONS_LIST:
            runtimes.append(table[N])
        writer.writerow(["numba-julia-oif-from-python"] + runtimes)

    print(f"Data are written to {RESULT_JL_DIFFEQ_PYTHON_FILENAME}")
    print("Finished")


def process():
    subprocess.run(
        ["column", "-s,", "-t"], stdin=open(RESULT_JL_DIFFEQ_PYTHON_FILENAME)
    )


if __name__ == "__main__":
    main()
