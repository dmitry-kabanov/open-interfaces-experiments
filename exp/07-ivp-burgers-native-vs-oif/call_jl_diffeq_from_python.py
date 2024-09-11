"""Run jl_diffeq via OIF from Python to assess its runtime."""

import argparse
import csv
import dataclasses
import os
import subprocess
import time

import numpy as np
from oif.interfaces.ivp import IVP
from scipy import integrate

from common import BurgersEquationProblem
from helpers import get_outdir

RESOLUTIONS_LIST = [800, 1600, 3200]
N_RUNS = 30

OUTDIR = get_outdir()
RESULT_JL_DIFFEQ_PYTHON_FILENAME = OUTDIR / "runtime_vs_resolution_python_jl_diffeq.csv"


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

    if impl == "native":
        s = integrate.ode(problem.compute_rhs_scipy_ode)
        s.set_initial_value(y0, t0)
        s.set_integrator("dopri5", rtol=1e-6, atol=1e-12)
    else:
        s = IVP(impl)
        s.set_initial_value(y0, t0)
        s.set_rhs_fn(problem.compute_rhs)
        s.set_tolerances(1e-6, 1e-12)
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

        table[N] = f"{mean:.4f} ± {ci:.4f}"

    with open(RESULT_JL_DIFFEQ_PYTHON_FILENAME, "w", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method/resolution"] + RESOLUTIONS_LIST)
        runtimes = []
        for N in RESOLUTIONS_LIST:
            runtimes.append(table[N])
        writer.writerow(["oif from python"] + runtimes)

    print(f"Data are written to {RESULT_JL_DIFFEQ_PYTHON_FILENAME}")
    print("Finished")


def process():
    subprocess.run(
        ["column", "-s,", "-t"], stdin=open(RESULT_JL_DIFFEQ_PYTHON_FILENAME)
    )


if __name__ == "__main__":
    main()
