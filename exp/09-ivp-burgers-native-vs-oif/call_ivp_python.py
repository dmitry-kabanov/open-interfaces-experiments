"""Assess performance of Open Interfaces in Python.

Compare solution of Burgers' equation using IVP interface versus solving
directly via SciPy.
"""

import csv
import time

import numpy as np
import numpy.testing as npt
from oif.interfaces.ivp import IVP
from rhsversions import (
    compute_rhs_oif_numba_v1,
    compute_rhs_oif_numba_v2,
    compute_rhs_oif_numba_v3,
    compute_rhs_oif_numba_v4,
    compute_rhs_oif_numpy,
)
from scipy import integrate

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci, get_outdir

RTOL = 1e-6
ATOL = 1e-12

RESOLUTIONS_LIST = [800, 1600, 3200]
RESOLUTIONS_LIST = [800, 1600]
N_RUNS = 2
VERSIONS = ["v1", "v2", "v3", "v4", "v4+wrapper"]

OUTDIR = get_outdir()
RESULT_PERF_FILENAME = OUTDIR / "runtime_vs_resolution_python.csv"


def get_wrapper_for_compute_rhs_oif():
    def compute_rhs_wrapper(t, u, udot, p):
        compute_rhs_oif_numba_v4(t, u, udot, p)

    return compute_rhs_wrapper


def get_wrapper_for_compute_rhs_native(dx, N):
    p = (dx,)
    udot = np.empty(N)

    def compute_rhs_ode_wrapper(t, u):
        compute_rhs_oif_numba_v4(t, u, udot, p)
        return udot

    return compute_rhs_ode_wrapper


def measure_perf_once(N):
    problem = BurgersEquationProblem(N=N)
    t0 = problem.t0
    y0 = problem.u0
    p = (problem.dx,)
    dx = problem.dx

    compute_rhs_ode = get_wrapper_for_compute_rhs_native(dx, len(y0))

    # Sanity check: Numba functions must return the same values as the NumPy one.
    result_0 = np.empty_like(y0)
    compute_rhs_oif_numpy(t0, y0, result_0, p)

    result_1 = np.empty_like(y0)
    result_2 = np.empty_like(y0)
    result_3 = np.empty_like(y0)
    result_4 = np.empty_like(y0)
    compute_rhs_oif_numba_v1(t0, y0, result_1, p)
    compute_rhs_oif_numba_v2(t0, y0, result_2, p)
    compute_rhs_oif_numba_v3(t0, y0, result_3, p)
    compute_rhs_oif_numba_v4(t0, y0, result_4, p)

    npt.assert_allclose(result_0, result_1, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_2, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_3, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_4, rtol=1e-14, atol=1e-14)

    result_scipy_3 = compute_rhs_ode(t0, y0)
    npt.assert_allclose(result_0, result_scipy_3, rtol=1e-14, atol=1e-14)

    runtimes = {}
    for version in VERSIONS + ["native"]:
        if version.startswith("v"):
            s = IVP("scipy_ode")
            s.set_initial_value(problem.u0, problem.t0)
            s.set_user_data(p)
            if version == "v1":
                s.set_rhs_fn(compute_rhs_oif_numba_v1)
            elif version == "v2":
                s.set_rhs_fn(compute_rhs_oif_numba_v2)
            elif version == "v3":
                s.set_rhs_fn(compute_rhs_oif_numba_v3)
            elif version == "v4":
                s.set_rhs_fn(compute_rhs_oif_numba_v4)
            elif version == "v4+wrapper":
                s.set_rhs_fn(get_wrapper_for_compute_rhs_oif())
            s.set_integrator("dopri5")
            s.set_tolerances(RTOL, ATOL)
        elif version == "native":
            s = integrate.ode(compute_rhs_ode)
            s.set_integrator("dopri5", rtol=RTOL, atol=ATOL)
            s.set_initial_value(problem.u0, problem.t0)

        times = np.linspace(t0, problem.tfinal, num=101)

        oif_solution_1 = [y0]
        tic = time.perf_counter()
        for t in times[1:]:
            s.integrate(t)
        toc = time.perf_counter()
        runtimes[version] = toc - tic
        oif_solution_1.append(s.y)
        print(f"RHS {version:>6s}: leftmost point = {s.y[0]:.16f}")

    return runtimes


def main():
    print("Comparing performance of Open Interfaces for IVP interface from Python")

    numba_format_template = "py-openif-numba-{v}"
    table = {}
    methods = {}
    for v in VERSIONS + ["native"]:
        if v.startswith("v"):
            methods[v] = numba_format_template.format(v=v)
        else:
            methods[v] = "py-native-numba-v3"
        table[methods[v]] = []

    for N in RESOLUTIONS_LIST:
        print()
        print(f"Resolution N = {N}")
        print(f"Measure performance {N_RUNS} times")
        elapsed_times = {}
        for v in VERSIONS + ["native"]:
            elapsed_times[v] = []
        for k in range(N_RUNS):
            runtimes = measure_perf_once(N)
            for v in VERSIONS + ["native"]:
                elapsed_times[v].append(runtimes[v])

        print()
        for v in VERSIONS + ["native"]:
            runtime_mean, ci = compute_mean_and_ci(elapsed_times[v])
            print(f"Runtime {v:>6s}, sec: {runtime_mean:.3f} ± {ci:.3f}")
            val = f"{runtime_mean:.2f} ± {ci:.2f}"
            table[methods[v]].append(val)

    with open(RESULT_PERF_FILENAME, "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# method"] + RESOLUTIONS_LIST)
        for method in methods.values():
            writer.writerow([method] + table[method])


if __name__ == "__main__":
    main()
