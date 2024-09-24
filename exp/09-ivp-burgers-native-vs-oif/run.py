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


def main():
    process_1()
    process_2()
    process_3()


def process_1():
    print()
    print("Python native and via OIF: Scipy.integrate.ode.dopri5")
    subprocess.run(["column", "-s,", "-t"], stdin=open(DATA_1, "r"))


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
