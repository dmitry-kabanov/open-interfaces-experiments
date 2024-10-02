"""We wrap a C version of Burgers' eqn with ctypes and invoke OrdinaryDiffEq.jl"""

import ctypes
import sys
import time

import numpy as np
import numpy.testing as npt
from juliacall import Main as jl
from juliacall import VectorValue
from juliacall import convert as jlconvert
from line_profiler import profile

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci, get_outdir

RTOL = 1e-6
ATOL = 1e-12

RESOLUTIONS_LIST = [200, 400, 800, 1600, 3200]
N_RUNS = 30
RESOLUTIONS_LIST = [3200]
N_RUNS = 2

OUTDIR = get_outdir()
RESULT_PERF_FILENAME = OUTDIR / "runtime_vs_resolution_python.csv"

ELAPSED_TIME = 0.0


def get_wrapper_for_burgers_c_func():
    lib = ctypes.CDLL("./burgers.so")
    compute_rhs = lib.rhs_carray
    compute_rhs.restype = None
    compute_rhs.argtypes = [
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    @profile
    def compute_rhs_wrapper(udot, u, p, t):
        # Load C function
        # Call it with arguments (t, u, udot, p)
        global ELAPSED_TIME
        tic = time.perf_counter()
        if isinstance(u, VectorValue):
            np_u = u.to_numpy(copy=False)
            np_udot = np.asarray(udot)
        else:
            print("Numpy")
            # sys.exit()
            np_u = u
            np_udot = udot
        # c_u = np_u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # c_udot = np_udot.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # c_u = np_u.__array_interface__["data"][0]
        # c_udot = np_udot.__array_interface__["data"][0]
        # c_u = np_u.ctypes.data_as(ctypes.c_void_p)
        # c_udot = np_udot.ctypes.data_as(ctypes.c_void_p)
        c_u = np_u.ctypes.data
        c_udot = np_udot.ctypes.data
        x = ctypes.pointer(ctypes.c_double(p[0]))
        toc = time.perf_counter()
        ELAPSED_TIME += toc - tic
        compute_rhs(t, c_u, c_udot, x, len(u))

    return compute_rhs_wrapper


def measure_perf_once(N):
    problem = BurgersEquationProblem(N=N)
    t0 = problem.t0
    y0 = problem.u0
    p = (problem.dx,)

    compute_rhs = get_wrapper_for_burgers_c_func()

    result_0 = np.empty_like(y0)
    problem.compute_rhs(t0, y0, result_0, p)

    result_1 = np.empty_like(y0)
    compute_rhs(result_1, y0, p, t0)

    npt.assert_allclose(result_0, result_1, rtol=1e-14, atol=1e-14)

    tspan = (problem.t0, problem.tfinal)
    ode_problem = jl.ODEProblem(compute_rhs, y0, tspan, p)
    solver = jl.init(
        ode_problem,
        jl.DP5(),
        reltol=RTOL,
        abstol=ATOL,
        save_everystep=False,
    )

    times = np.linspace(problem.t0, problem.tfinal, num=101)
    tic = time.perf_counter()
    for t in times[2:]:
        jl.step_b(solver, t - solver.t, True)
    toc = time.perf_counter()
    runtime = toc - tic

    return runtime


def warmup():
    print("BEGIN warmup")
    problem = BurgersEquationProblem(N=101)
    y0 = problem.u0
    p = (problem.dx,)
    tspan = (problem.t0, problem.tfinal)
    compute_rhs = get_wrapper_for_burgers_c_func()

    jl.seval("using OrdinaryDiffEq")
    ode_problem = jl.ODEProblem(compute_rhs, y0, tspan, p)
    solver = jl.init(
        ode_problem,
        jl.DP5(),
        reltol=RTOL,
        abstol=ATOL,
        save_everystep=False,
    )
    jl.step_b(solver, 0.01 - 0.0, True)
    print("END warmup")


def main():
    print("Comparing performance of Open Interfaces for IVP interface from Python")

    # numba_format_template = "py-pycall-cfunc-odejl"
    # table = {}
    # methods = {}

    # Because Python does not allow using `!` in function name,s
    # we need to introduce a function alias.

    warmup()

    for N in RESOLUTIONS_LIST:
        print()
        print(f"Resolution N = {N}")
        print(f"Measure performance {N_RUNS} times")
        elapsed_times = []
        for k in range(N_RUNS):
            runtime = measure_perf_once(N)
            elapsed_times.append(runtime)

        runtime_mean, ci = compute_mean_and_ci(elapsed_times)
        print(f"Runtime, sec: {runtime_mean:.3f} ± {ci:.3f}")

    # print(f"ELAPSED_TIME: {ELAPSED_TIME:.3f}")
    print(f"ELAPSED_TIME mean: {ELAPSED_TIME / N_RUNS:.3f}")


if __name__ == "__main__":
    main()
