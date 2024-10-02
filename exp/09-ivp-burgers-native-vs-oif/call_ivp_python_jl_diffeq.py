"""Run jl_diffeq via OIF from Python to assess its runtime."""

import argparse
import csv
import dataclasses
import os
import subprocess
import time

import numba as nb
import numpy as np
import oif.core as core
from oif.interfaces.ivp import IVP
from rhsversions import compute_rhs_oif_numba_v4
from scipy import integrate

from common import BurgersEquationProblem
from helpers import get_outdir

RTOL = 1e-6
ATOL = 1e-12

RESOLUTIONS_LIST = [200, 400, 800, 1600, 3200]
N_RUNS = 30

OUTDIR = get_outdir()
RESULT_JL_DIFFEQ_PYTHON_FILENAME = OUTDIR / "runtime_vs_resolution_python_jl_diffeq.csv"
RESULT_PYTHON_NATIVE_FILENAME = OUTDIR / "runtime_vs_resolution_python_native.csv"
RESULT_PYTHON_SOLVE_IVP_FILENAME = OUTDIR / "runtime_vs_resolution_python_solve_ivp.csv"


class ComputeRHSScipyWrapper:
    def __init__(self, dx, N, func):
        self.p = (dx,)
        self.udot = np.empty(N)
        self.rhs_evals = 0
        self.func = func

    def compute_rhs_ode_wrapper(self, t, u):
        self.rhs_evals += 1
        self.func(t, u, self.udot, self.p)
        return self.udot

    def compute_rhs_solve_ivp_wrapper(self, t, u):
        self.rhs_evals += 1
        udot = np.empty_like(u)
        self.func(t, u, udot, self.p)
        return udot

    def compute_rhs_oif(self, t, u, udot, p):
        self.rhs_evals += 1
        self.func(t, u, udot, p)


@dataclasses.dataclass
class Args:
    impl: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["jl_diffeq", "native", "solve_ivp"],
        default="jl_diffeq",
        nargs="?",
    )
    args = p.parse_args()
    return Args(**vars(args))


def measure_perf_once(N):
    args = _parse_args()
    impl = args.impl
    print(f"---Implementation: {impl}")
    problem = BurgersEquationProblem(N=N)
    t0 = 0.0
    y0 = problem.u0
    p = (problem.dx,)

    wrapper = ComputeRHSScipyWrapper(problem.dx, len(y0), compute_rhs_oif_numba_v4)

    compute_rhs_oif_numba_v4(0.0, y0, np.empty_like(y0), p)

    if impl == "native":
        s = integrate.ode(wrapper.compute_rhs_ode_wrapper)
        s.set_initial_value(y0, t0)
        s.set_integrator("dopri5", rtol=RTOL, atol=ATOL)
    elif impl == "jl_diffeq":
        s = IVP(impl)
        s.set_initial_value(y0, t0)
        s.set_user_data(p)
        s.set_rhs_fn(wrapper.compute_rhs_oif)
        s.set_tolerances(RTOL, ATOL)
        s.set_integrator("DP5")
    elif impl == "solve_ivp":
        pass
    else:
        raise ValueError("Shouldn't have come here")

    times = np.linspace(problem.t0, problem.tfinal, num=101)

    if impl != "solve_ivp":
        n_rhs = 0
        n_acc = 0
        n_rej = 0
        s.set_initial_value(y0, t0)
        tic = time.perf_counter()
        for t in times[1:]:
            s.integrate(t)
            if impl == "native":
                # Read about this magic here:
                # https://github.com/scipy/scipy/blob/main/scipy/integrate/dop/dopri5.f#L182
                n_rhs += s._integrator.iwork[16]
                n_acc += s._integrator.iwork[18]
                n_rej += s._integrator.iwork[19]
        toc = time.perf_counter()
        runtime = toc - tic
        solution_last = s.y
    else:
        tic = time.perf_counter()
        solution = integrate.solve_ivp(
            wrapper.compute_rhs_solve_ivp_wrapper,
            (problem.t0, problem.tfinal),
            y0,
            t_eval=times,
            method="RK45",
            rtol=RTOL,
            atol=ATOL,
        )
        toc = time.perf_counter()
        runtime = toc - tic
        if not solution.success:
            print(solution)
        solution_last = solution.y[:, -1]
    print("Leftmost point:", solution_last[0])

    # if impl != "solve_ivp":
    #     if hasattr(s, "print_stats") and impl != "native":
    #         s.print_stats()
    #     else:
    #         assert impl == "native"
    #         print("    No. of RHS evaluations: ", n_rhs)
    #         print("    No. of  accepted steps: ", n_acc)
    #         print("    No. of  rejected steps: ", n_rej)

    # print("Manual  RHS evals: ", wrapper.rhs_evals)

    return runtime, problem.x, problem.u0, solution_last


def main():
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

        table[N] = f"{mean:.2f} ± {ci:.2f}"

    args = _parse_args()
    if args.impl == "jl_diffeq":
        filename = RESULT_JL_DIFFEQ_PYTHON_FILENAME
        desc = "jl-openif-numba-v4"
    elif args.impl == "native":
        filename = RESULT_PYTHON_NATIVE_FILENAME
        desc = "py-native-numba-v4"
    elif args.impl == "solve_ivp":
        filename = RESULT_PYTHON_SOLVE_IVP_FILENAME
        desc = "py-solve_ivp-numba-v4"

    with open(filename, "w", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method/resolution"] + RESOLUTIONS_LIST)
        runtimes = []
        for N in RESOLUTIONS_LIST:
            runtimes.append(table[N])
        writer.writerow(["{:30s}".format(desc)] + runtimes)

    print(f"Data are written to {filename}")
    subprocess.run(["column", "-s,", "-t"], stdin=open(filename))

    if hasattr(core, "elapsed"):
        print(f"Elapsed time in wrapper: {core.elapsed:.3f} sec")
        print(f"Elapsed time average: {core.elapsed / N_RUNS:.3f} sec")
    print("Finished")


if __name__ == "__main__":
    main()
