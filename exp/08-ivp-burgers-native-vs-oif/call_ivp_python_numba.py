""" Assess performance of Open Interfaces in Python.

Compare solution of Burgers' equation using IVP interface versus solving
directly via SciPy.
"""

import time
from typing import List

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.testing as npt
from oif.interfaces.ivp import IVP
from scipy import integrate

from common import BurgersEquationProblem
from helpers import get_outdir

RESOLUTIONS_LIST = [800, 1600, 3200]
NUMBER_OF_RUNS = 30

OUTDIR = get_outdir()
RESULT_PERF_FILENAME = OUTDIR / "runtime_vs_resolution_python_numba.csv"


def compute_rhs_oif_numpy(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p

    f = 0.5 * u**2
    local_ss = np.max(np.abs(u))

    f_hat = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
    f_plus = f_hat[1:]
    f_minus = f_hat[0:-1]
    udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

    local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])


def compute_rhs_oif(__, u, udot, p):
    compute_rhs_numba(__, u, udot, p)


@numba.jit
def compute_rhs_numba(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p

    f = 0.5 * u**2
    local_ss = np.max(np.abs(u))

    f_hat = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
    f_plus = f_hat[1:]
    f_minus = f_hat[0:-1]
    udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

    local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])

    # N = u.shape[0]

    # f = np.empty(N)
    # for i in range(N):
    #     f[i] = 0.5 * u[i] ** 2

    # local_ss = 0.0
    # for i in range(N - 1):
    #     cand = max(abs(u[i]), abs(u[i + 1]))
    #     if cand > local_ss:
    #         local_ss = cand

    # f_hat = np.empty(N - 1)
    # for i in range(N - 1):
    #     f_hat[i] = 0.5 * (f[i] + f[i + 1]) - 0.5 * local_ss * (u[i + 1] - u[i])

    # for i in range(1, N - 1):
    #     udot[i] = -1.0 / dx * (f_hat[i] - f_hat[i - 1])

    # local_ss_rb = max(abs(u[0]), abs(u[-1]))
    # f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    # f_lb = f_rb

    # udot[0] = -1.0 / dx * (f_hat[0] - f_lb)
    # udot[-1] = -1.0 / dx * (f_rb - f_hat[-1])


def get_wrapper_for_compute_rhs_ode(dx):
    def compute_rhs_ode_wrapper(t, u):
        return compute_rhs_ode_numba(t, u, dx)

    return compute_rhs_ode_wrapper


@numba.jit
def compute_rhs_ode_numba(t: float, u: np.ndarray, dx: float) -> np.ndarray:
    N = u.shape[0]

    f = np.empty(N)
    for i in range(N):
        f[i] = 0.5 * u[i] ** 2

    local_ss = 0.0
    for i in range(N - 1):
        cand = max(abs(u[i]), abs(u[i + 1]))
        if cand > local_ss:
            local_ss = cand

    f_hat = np.empty(N - 1)
    for i in range(N - 1):
        f_hat[i] = 0.5 * (f[i] + f[i + 1]) - 0.5 * local_ss * (u[i + 1] - u[i])

    udot = np.empty_like(u)
    for i in range(1, N - 1):
        udot[i] = -1.0 / dx * (f_hat[i] - f_hat[i - 1])

    local_ss_rb = max(abs(u[0]), abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[0] = -1.0 / dx * (f_hat[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_hat[-1])

    return udot


@numba.jit
def compute_rhs_ode_2(t, u: np.ndarray, dx: float) -> np.ndarray:
    f = 0.5 * u**2
    local_ss = np.max(np.abs(u))
    f_hat = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
    f_plus = f_hat[1:]
    f_minus = f_hat[0:-1]
    udot = np.empty_like(u)
    udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

    local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])

    return udot


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
    p = (problem.dx,)
    dx = problem.dx

    compute_rhs_ode = get_wrapper_for_compute_rhs_ode(dx)
    # compute_rhs_ode = compute_rhs_ode_numba

    # Sanity check: Numba functions must return the same values as the NumPy one.
    # result_1 = np.empty_like(y0)
    # result_2 = np.empty_like(y0)
    # compute_rhs_oif_numpy(t0, y0, result_1, p)
    # compute_rhs_numba(t0, y0, result_2, p)
    # result_3 = compute_rhs_ode(t0, y0)
    # npt.assert_allclose(result_1, result_2, rtol=1e-14, atol=1e-14)
    # npt.assert_allclose(result_1, result_3, rtol=1e-14, atol=1e-14)

    s = IVP("scipy_ode")
    s.set_initial_value(problem.u0, problem.t0)
    s.set_user_data(p)
    s.set_rhs_fn(compute_rhs_oif)
    s.set_integrator("dopri5")
    s.set_tolerances(rtol=1e-6, atol=1e-12)

    times = np.linspace(t0, problem.tfinal, num=101)
    # # times = np.arange(t0, problem.tfinal + problem.dt_max, step=problem.dt_max)

    oif_solution = [y0]
    tic = time.perf_counter()
    for t in times[1:]:
        s.integrate(t)
    toc = time.perf_counter()
    oif_time = toc - tic
    oif_solution.append(s.y)

    solver_ode = integrate.ode(compute_rhs_ode)
    # solver_ode.set_f_params(dx)
    solver_ode.set_integrator("dopri5", rtol=1e-6, atol=1e-12)
    solver_ode.set_initial_value(problem.u0, problem.t0)

    native_solution = [problem.u0]
    ode_tic = time.perf_counter()
    for t in times[1:]:
        solver_ode.integrate(t)
    ode_toc = time.perf_counter()
    native_time = ode_toc - ode_tic
    native_solution.append(solver_ode.y)

    npt.assert_allclose(oif_solution[-1], native_solution[-1], rtol=1e-10, atol=1e-10)

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
        table[("numba-python-oif", N)] = oif_time_list
        table[("numba-python-native", N)] = native_time_list

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
        for method in ["numba-python-oif", "numba-python-native"]:
            for N in RESOLUTIONS_LIST:
                runtimes = ", ".join([f"{t:3f}" for t in table[method, N]])
                fh.write(f"{method}, {N}, " + runtimes + "\n")


if __name__ == "__main__":
    main()
