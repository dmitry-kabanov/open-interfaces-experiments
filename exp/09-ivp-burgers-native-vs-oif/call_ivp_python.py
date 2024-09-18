""" Assess performance of Open Interfaces in Python.

Compare solution of Burgers' equation using IVP interface versus solving
directly via SciPy.
"""

import time

import numba as nb
import numpy as np
import numpy.testing as npt
from oif.interfaces.ivp import IVP
from scipy import integrate

from common import BurgersEquationProblem
from helpers import get_outdir

RESOLUTIONS_LIST = [800, 1600, 3200]
RESOLUTIONS_LIST = [800, 1600]
NUMBER_OF_RUNS = 2
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
    udot = np.empty(N)

    def compute_rhs_ode_wrapper(t, u):
        return compute_rhs_native_numba_v3(t, u, dx, udot)

    return compute_rhs_ode_wrapper


@nb.jit(
    boundscheck=False,
    nogil=True,
)
def compute_rhs_native_numba_v3(
    t: float, u: np.ndarray, dx: float, udot: np.ndarray
) -> np.ndarray:
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
    # compute_rhs_ode = compute_rhs_ode_numba

    # Sanity check: Numba functions must return the same values as the NumPy one.
    result_0 = np.empty_like(y0)
    compute_rhs_oif_numpy(t0, y0, result_0, p)

    result_1 = np.empty_like(y0)
    result_2 = np.empty_like(y0)
    result_3 = np.empty_like(y0)
    # result_4 = np.empty_like(y0)
    compute_rhs_oif_numba_v1(t0, y0, result_1, p)
    compute_rhs_oif_numba_v2(t0, y0, result_2, p)
    compute_rhs_oif_numba_v3(t0, y0, result_3, p)
    # compute_rhs_oif_numba_v4(t0, y0, result_4, p)

    npt.assert_allclose(result_0, result_1, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_2, rtol=1e-14, atol=1e-14)
    npt.assert_allclose(result_0, result_3, rtol=1e-14, atol=1e-14)
    # npt.assert_allclose(result_0, result_4, rtol=1e-14, atol=1e-14)

    result_scipy_3 = compute_rhs_ode(t0, y0)
    npt.assert_allclose(result_0, result_scipy_3, rtol=1e-14, atol=1e-14)

    runtimes = {}
    for version in VERSIONS:
        s = IVP("scipy_ode")
        s.set_initial_value(problem.u0, problem.t0)
        s.set_user_data(p)
        if version == "v1":
            s.set_rhs_fn(compute_rhs_oif_numba_v1)
        elif version == "v2":
            s.set_rhs_fn(compute_rhs_oif_numba_v2)
        elif version == "v3":
            s.set_rhs_fn(compute_rhs_oif_numba_v3)
        # elif version == "v4":
        #     s.set_rhs_fn(compute_rhs_oif_numba_v4)
        s.set_integrator("dopri5")
        s.set_tolerances(rtol=1e-6, atol=1e-12)

        times = np.linspace(t0, problem.tfinal, num=101)
        # # times = np.arange(t0, problem.tfinal + problem.dt_max, step=problem.dt_max)

        oif_solution_1 = [y0]
        tic = time.perf_counter()
        for t in times[1:]:
            s.integrate(t)
        toc = time.perf_counter()
        runtimes[version] = toc - tic
        oif_solution_1.append(s.y)

    solver_ode = integrate.ode(compute_rhs_ode)
    solver_ode.set_integrator("dopri5", rtol=1e-6, atol=1e-12)
    solver_ode.set_initial_value(problem.u0, problem.t0)

    native_solution = [problem.u0]
    ode_tic = time.perf_counter()
    for t in times[1:]:
        solver_ode.integrate(t)
    ode_toc = time.perf_counter()
    native_time = ode_toc - ode_tic
    native_solution.append(solver_ode.y)

    runtimes["native"] = native_time

    npt.assert_allclose(oif_solution_1[-1], native_solution[-1], rtol=1e-10, atol=1e-10)

    return runtimes


def main():
    print("Comparing performance of Open Interfaces for IVP interface from Python")

    table = {}

    numba_format_template = "py-openif-numba-{v}"

    for N in RESOLUTIONS_LIST:
        oif_time_list = {}
        for v in VERSIONS:
            oif_time_list[v] = []
        native_time_list = []
        for k in range(NUMBER_OF_RUNS):
            runtimes = measure_perf_once(N)
            for v in VERSIONS:
                oif_time_list[v].append(runtimes[v])
            native_time_list.append(runtimes["native"])

        for v in VERSIONS:
            table[(numba_format_template.format(v=v), N)] = oif_time_list[v]
        table[("py-native-numba-v3", N)] = native_time_list

    with open(RESULT_PERF_FILENAME, "w") as fh:
        fh.write(
            "# method, resolution, "
            + ", ".join([f"runtime{k:d}" for k in range(NUMBER_OF_RUNS)])
            + "\n"
        )
        for method in [numba_format_template.format(v=v) for v in VERSIONS]:
            for N in RESOLUTIONS_LIST:
                runtimes = ", ".join([f"{t:3f}" for t in table[method, N]])
                fh.write(f"{method}, {N}, " + runtimes + "\n")

        for method in ["py-native-numba-v3"]:
            for N in RESOLUTIONS_LIST:
                runtimes = ", ".join([f"{t:3f}" for t in table[method, N]])
                fh.write(f"{method}, {N}, " + runtimes + "\n")


if __name__ == "__main__":
    main()
