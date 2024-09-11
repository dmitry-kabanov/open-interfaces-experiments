""" Assess performance of Open Interfaces in Python.

Compare solution of Burgers' equation using IVP interface versus solving
directly via SciPy.
"""

import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from oif.interfaces.ivp import IVP
from scipy import integrate

from common import BurgersEquationProblem
from helpers import get_outdir

RESOLUTIONS_LIST = [800, 1600, 3200]
NUMBER_OF_RUNS = 30

OUTDIR = get_outdir()
RESULT_PERF_FILENAME = OUTDIR / "runtime_vs_resolution_python.csv"
RESULT_FIG_FILENAME = OUTDIR / "solution_python.pdf"


# @dataclasses.dataclass
# class Args:
#     impl: str


# def _parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument(
#         "impl",
#         choices=["jl_diffeq", "native"],
#         default="jl_diffeq",
#         nargs="?",
#     )
#     args = p.parse_args()
#     return Args(**vars(args))


def measure_perf_once(N):
    # args = _parse_args()
    # impl = args.impl
    # print(f"Implementation: {impl}")
    problem = BurgersEquationProblem(N=N)
    t0 = problem.t0
    y0 = problem.u0

    s = IVP("scipy_ode")
    s.set_initial_value(problem.u0, problem.t0)
    s.set_rhs_fn(problem.compute_rhs)
    s.set_integrator("dopri5")
    s.set_tolerances(rtol=1e-6, atol=1e-12)

    times = np.linspace(t0, problem.tfinal, num=101)
    # times = np.arange(problem.t0, problem.tfinal + problem.dt_max, step=problem.dt_max)

    oif_solution = [y0]
    tic = time.perf_counter()
    for t in times[1:]:
        s.integrate(t)
    toc = time.perf_counter()
    oif_time = toc - tic
    oif_solution.append(s.y)

    solver_ode = integrate.ode(problem.compute_rhs_scipy_ode)
    solver_ode.set_integrator("dopri5", rtol=1e-6, atol=1e-12)
    solver_ode.set_initial_value(problem.u0, problem.t0)

    native_solution = [problem.u0]
    ode_tic = time.perf_counter()
    for t in times[1:]:
        solver_ode.integrate(t)
    ode_toc = time.perf_counter()
    native_time = ode_toc - ode_tic
    native_solution.append(solver_ode.y)

    npt.assert_allclose(oif_solution[-1], native_solution[-1], rtol=1e-14, atol=1e-14)

    return (oif_time, native_time), (problem.x, oif_solution, native_solution)


def main() -> None:
    print("Comparing performance of Open Interfaces for IVP interface from Python")

    table = {}

    grid: np.ndarray
    oif_solution: List[np.ndarray]
    native_solution: List[np.ndarray]
    for N in RESOLUTIONS_LIST:
        oif_time_list = []
        native_time_list = []
        for k in range(NUMBER_OF_RUNS):
            (oif_time, native_time), (grid, oif_solution, native_solution) = (
                measure_perf_once(N)
            )
            oif_time_list.append(oif_time)
            native_time_list.append(native_time)
        table[("oif", N)] = oif_time_list
        table[("native", N)] = native_time_list

        std_est = []
        k_values = range(3, len(oif_time_list))
        for k in k_values:
            std_est.append(np.std(oif_time_list[:k], ddof=1))

    with open(RESULT_PERF_FILENAME, "w") as fh:
        fh.write(
            "# method, resolution, "
            + ", ".join([f"runtime{k:d}" for k in range(NUMBER_OF_RUNS)])
            + "\n"
        )
        for method in ["oif", "native"]:
            for N in RESOLUTIONS_LIST:
                runtimes = ", ".join([f"{t:3f}" for t in table[method, N]])
                fh.write(f"{method}, {N}, " + runtimes + "\n")

    plt.figure()
    plt.plot(grid, oif_solution[-1], "-", label="OIF")
    plt.plot(grid, native_solution[-1], "--", label="Native")
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel("Solution")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_FIG_FILENAME)


if __name__ == "__main__":
    main()
