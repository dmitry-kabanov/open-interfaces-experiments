"""Assess performance of Open Interfaces in Python.

Compare solution of Burgers' equation using IVP interface versus solving
directly via SciPy.
"""

import csv
import time

import numba as nb
import numpy as np
import numpy.testing as npt
from oif.interfaces.ivp import IVP
from scipy import integrate

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci, get_outdir

RTOL = 1e-6
ATOL = 1e-12

RESOLUTIONS_LIST = [800, 1600, 3200]
RESOLUTIONS_LIST = [800, 1600]
N_RUNS = 2
VERSIONS = ["v1", "v2", "v3"]

OUTDIR = get_outdir()
RESULT_PERF_FILENAME = OUTDIR / "runtime_vs_resolution_python.csv"


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
    compute_rhs_oif_numba_v1(__, u, udot, p)


@nb.jit
def compute_rhs_oif_numba_v1(__, u: np.ndarray, udot: np.ndarray, p) -> None:
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


@nb.jit
def compute_rhs_oif_numba_v2(__, u: np.ndarray, udot: np.ndarray, p) -> None:
    (dx,) = p

    N = u.shape[0]

    f = np.empty(N)
    for i in range(N):
        f[i] = 0.5 * u[i] ** 2

    local_ss = 0.0
    for i in range(N - 1):
        cand = abs(u[i])
        if cand > local_ss:
            local_ss = cand

    f_hat = np.empty(N - 1)
    for i in range(N - 1):
        f_hat[i] = 0.5 * (f[i] + f[i + 1]) - 0.5 * local_ss * (u[i + 1] - u[i])

    for i in range(1, N - 1):
        udot[i] = -1.0 / dx * (f_hat[i] - f_hat[i - 1])

    local_ss_rb = max(abs(u[0]), abs(u[-1]))
    f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
    f_lb = f_rb

    udot[0] = -1.0 / dx * (f_hat[0] - f_lb)
    udot[-1] = -1.0 / dx * (f_rb - f_hat[-1])


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


def get_wrapper_for_compute_rhs_native(dx, N):
    p = (dx,)
    udot = np.empty(N)

    def compute_rhs_ode_wrapper(t, u):
        return compute_rhs_native_numba_v3(t, u, udot, p)

    return compute_rhs_ode_wrapper


@nb.jit(
    boundscheck=False,
    nogil=True,
)
def compute_rhs_native_numba_v3(
    t: float, u: np.ndarray, udot: np.ndarray, p
) -> np.ndarray:
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

    return udot


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
    compute_rhs_oif_numba_v1(t0, y0, result_1, p)
    compute_rhs_oif_numba_v2(t0, y0, result_2, p)
    compute_rhs_oif_numba_v3(t0, y0, result_3, p)

    npt.assert_allclose(result_0, result_1, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_2, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_3, rtol=1e-14, atol=1e-14)

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

        for v in VERSIONS + ["native"]:
            runtime_mean, ci = compute_mean_and_ci(elapsed_times[v])
            print(f"Runtime, sec: {runtime_mean:.3f} ± {ci:.3f}")
            val = f"{runtime_mean:.2f} ± {ci:.2f}"
            table[methods[v]].append(val)

    with open(RESULT_PERF_FILENAME, "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# method"] + RESOLUTIONS_LIST)
        for method in methods.values():
            writer.writerow([method] + table[method])


if __name__ == "__main__":
    main()
