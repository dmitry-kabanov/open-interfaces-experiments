"""We wrap a C version of Burgers' eqn with ctypes and invoke OrdinaryDiffEq.jl"""

import argparse
import csv
import ctypes
import sys
import time
from dataclasses import dataclass

import numpy as np
import numpy.testing as npt
from juliacall import Main as jl
from juliacall import VectorValue
from line_profiler import profile

from common import BurgersEquationProblem
from helpers import compute_mean_and_ci, get_outdir

RTOL = 1e-6
ATOL = 1e-12

RESOLUTIONS_LIST = [200, 400, 800, 1600, 3200, 6400]
N_RUNS = 30

OUTDIR = get_outdir()
RESULT_PERF_FILENAME = OUTDIR / "runtime_python.csv"


@dataclass
class Args:
    nruns: int


def parse_args(args=None):
    if args is None:
        args = sys.argv

    p = argparse.ArgumentParser()
    p.add_argument("--nruns", "-n", type=int, default=N_RUNS, help="Number of runs")

    args = p.parse_args()

    return Args(**vars(args))


def get_wrapper_for_burgers_c_func():
    lib = ctypes.CDLL("./burgers.so")
    # nd_pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    double_p_t = ctypes.POINTER(ctypes.c_double)

    compute_rhs = lib.rhs_carray
    compute_rhs.restype = None
    compute_rhs.argtypes = [
        ctypes.c_double,
        # nd_pointer,
        # nd_pointer,
        # double_p_t,
        ctypes.c_void_p,
        double_p_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    @profile
    def compute_rhs_wrapper(udot, u, p, t):
        # Load C function
        # Call it with arguments (t, u, udot, p)
        if isinstance(u, VectorValue):
            np_u = u.to_numpy(dtype=np.float64, copy=False)
            np_udot = udot.to_numpy(dtype=np.float64, copy=False)
            # np_udot = np.asarray(udot)
        else:
            np_u = u
            np_udot = udot
        # c_u = np_u.ctypes.data_as(double_p_t)
        # c_udot = np_udot.ctypes.data_as(double_p_t)
        # c_u = np_u.ctypes._as_parameter_
        # c_udot = np_udot.ctypes.data_as(double_p_t)
        c_u = ctypes.cast(np.ctypeslib.as_ctypes(np_u), double_p_t)
        c_udot = ctypes.cast(np.ctypeslib.as_ctypes(np_udot), double_p_t)
        # c_u = ctypes.cast(np_u.__array_interface__["data"][0], double_p_t)
        # c_udot = np_udot.__array_interface__["data"][0], double_p_t)
        # c_u = np_u.ctypes.data_as(ctypes.c_void_p)
        # c_udot = np_udot.ctypes.data_as(ctypes.c_void_p)
        # c_u = np_u.ctypes.data
        # c_udot = np_udot.ctypes.data
        # c_u = ctypes.cast(memoryview(np_u)[:], double_p_t)
        # c_udot = ctypes.cast(memoryview(np_udot)[:], double_p_t)
        x = ctypes.pointer(ctypes.c_double(p[0]))
        compute_rhs(t, c_u, c_udot, x, len(u))
        # compute_rhs(t, np_u, np_udot, x, len(u))

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


def main():
    args = parse_args()

    if args.nruns:
        N_RUNS = args.nruns

    print("Calling OrdinaryDiffEq.jl from Python with RHS written in C")
    print(f"N_RUNS = {N_RUNS}")

    print("BEGIN warmup")
    jl.seval("using OrdinaryDiffEq")
    measure_perf_once(N=101)
    print("END warmup")

    results = ["OrdinaryDiffEq.jl from Python"]

    for N in RESOLUTIONS_LIST:
        print()
        print(f"Resolution N = {N}")
        print(f"Measure performance {N_RUNS} times")
        elapsed_times = []
        for k in range(N_RUNS):
            runtime = measure_perf_once(N)
            elapsed_times.append(runtime)

        runtime_mean, ci = compute_mean_and_ci(elapsed_times)
        result = f"{runtime_mean:.3f} ± {ci:.3f}"
        print(f"Runtime, sec: {result:s}")
        results.append(result)

    header = ["method/resolution"] + RESOLUTIONS_LIST

    with open(RESULT_PERF_FILENAME, "w") as fh:
        writer = csv.writer(fh, delimiter=",")
        writer.writerow(header)
        writer.writerow(results)


if __name__ == "__main__":
    main()
