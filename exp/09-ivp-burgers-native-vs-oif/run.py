"""Experiment 09.

Compare performance between native implementations and using OIF
in Python and Julia.
We analyze also different optimizations that can be done to the right-hand
side evaluations via Numba in Python and writing code smartly in Julia.
"""

import csv
import os
import subprocess

import numpy as np

from helpers import get_expdir, get_outdir

EXPDIR = get_expdir()
OUTDIR = get_outdir()
STYLES = ["-", "--", ":"]

DATA_1 = OUTDIR / "runtime_vs_resolution_python.csv"

PROG_1 = ["python", (EXPDIR / "call_ivp_python.py")]

DATA_2_1 = OUTDIR / "runtime_vs_resolution_julia.csv"
# DATA_2_2 = OUTDIR / "runtime_vs_resolution_julia_noinline.csv"
DATA_2_3 = OUTDIR / "runtime_vs_resolution_python_jl_diffeq.csv"

PROG_2_1 = ["julia", EXPDIR / "call_ivp_julia.jl"]
PROG_2_3 = ["python", EXPDIR / "call_jl_diffeq_from_python.py", "jl_diffeq"]

DATA_3 = OUTDIR / "runtime_vs_resolution_julia_sundials.csv"

RESULT_PYTHON_STATS = OUTDIR / "runtime_vs_resolution_python_stats.csv"


def main():
    process_1()
    process_2()
    process_3()


def process_1():
    methods = []
    resolutions = []
    table = {}

    with open(DATA_1, "r") as fh:
        for line in fh.readlines():
            if line.startswith("#"):
                continue
            chunks = line.split(",")
            assert (
                len(chunks) >= 3
            ), "At least method, resolution and one runtime value are required"
            method = chunks[0].strip()
            N = int(chunks[1])
            runtimes = [float(v) for v in chunks[2:]]

            methods.append(method)
            resolutions.append(N)
            table[(method, N)] = runtimes

    methods = list(set(methods))
    methods.sort()
    resolutions = list(set(resolutions))
    resolutions.sort()
    stats = {}

    for method in methods:
        runtimes = []
        for N in resolutions:
            values = table[(method, N)]
            mean = np.mean(values)
            err = 2 * np.std(values, ddof=1) / np.sqrt(len(values))
            runtimes.append(f"{mean:.2f} ± {err:.2f}")

        stats[method] = runtimes

    with open(RESULT_PYTHON_STATS, "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# method"] + resolutions)
        for method in methods:
            writer.writerow([method] + stats[method])

    print()
    print("Python native and via OIF: Scipy.integrate.ode.dopri5")
    subprocess.run(["column", "-s,", "-t"], stdin=open(RESULT_PYTHON_STATS, "r"))


def process_2():
    print()
    print("Julia native")
    subprocess.run(["column", "-s,", "-t"], stdin=open(DATA_2_1, "r"))


def process_3():
    print()
    print("Python via OIF call to `jl_diffeq` (Julia OrdinaryDiffEq.jl)")
    subprocess.run(["column", "-s,", "-t"], stdin=open(DATA_2_3, "r"))


if __name__ == "__main__":
    main()
