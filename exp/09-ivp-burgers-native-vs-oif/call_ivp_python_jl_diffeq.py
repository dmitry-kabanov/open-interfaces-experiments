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

RTOL = 1e-6
ATOL = 1e-12

RESOLUTIONS_LIST = [800, 1600, 3200]
N_RUNS = 30

OUTDIR = get_outdir()
RESULT_JL_DIFFEQ_PYTHON_FILENAME = OUTDIR / "runtime_vs_resolution_python_jl_diffeq.csv"
RESULT_PYTHON_NATIVE_FILENAME = OUTDIR / "runtime_vs_resolution_python_native.csv"


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


class ComputeRHSScipyWrapper:
    def __init__(self, dx, N):
        self.p = (dx,)
        self.udot = np.empty(N)
        self.rhs_evals = 0

    def compute_rhs_ode_wrapper(self, t, u):
        self.rhs_evals += 1
        compute_rhs_oif_numba_v3(t, u, self.udot, self.p)
        return self.udot

    def compute_rhs_oif(self, t, u, udot, p):
        self.rhs_evals += 1
        compute_rhs_oif_numba_v3(t, u, udot, p)


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
    print(f"---Implementation: {impl}")
    problem = BurgersEquationProblem(N=N)
    t0 = 0.0
    y0 = problem.u0
    p = (problem.dx,)

    wrapper = ComputeRHSScipyWrapper(problem.dx, len(y0))

    compute_rhs_oif_numba_v3(0.0, y0, np.empty_like(y0), p)

    if impl == "native":
        s = integrate.ode(wrapper.compute_rhs_ode_wrapper)
        s.set_initial_value(y0, t0)
        s.set_integrator("dopri5", rtol=RTOL, atol=ATOL)
    else:
        s = IVP(impl)
        s.set_initial_value(y0, t0)
        s.set_user_data(p)
        s.set_rhs_fn(wrapper.compute_rhs_oif)
        s.set_tolerances(RTOL, ATOL)
        assert impl == "jl_diffeq"
        s.set_integrator("DP5")

    times = np.linspace(problem.t0, problem.tfinal, num=101)

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
    print("Leftmost point:", solution_last[0])

    if hasattr(s, "print_stats") and impl != "native":
        s.print_stats()
    else:
        assert impl == "native"
        print("    No. of RHS evaluations: ", n_rhs)
        print("    No. of  accepted steps: ", n_acc)
        print("    No. of  rejected steps: ", n_rej)

    print("Manual  RHS evals: ", wrapper.rhs_evals)

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
        desc = "jl-openif-numba-v3"
    else:
        assert args.impl == "native"
        filename = RESULT_PYTHON_NATIVE_FILENAME
        desc = "py-native-numba-v3"

    with open(filename, "w", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method/resolution"] + RESOLUTIONS_LIST)
        runtimes = []
        for N in RESOLUTIONS_LIST:
            runtimes.append(table[N])
        writer.writerow([desc] + runtimes)

    print(f"Data are written to {filename}")
    subprocess.run(["column", "-s,", "-t"], stdin=open(filename))
    print("Finished")


if __name__ == "__main__":
    main()
