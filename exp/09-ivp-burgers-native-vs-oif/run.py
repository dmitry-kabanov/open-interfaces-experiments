"""Experiment 07.

Compare runtime performance between invoking IVP solvers natively
and via Open Interfaces.
Example IVP is based on the inviscid Burgers' equation.
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
DATA_2_3 = OUTDIR / "runtime_vs_resolution_python_jl_diffeq_numba.csv"

PROG_2_1 = ["julia", EXPDIR / "call_ivp_julia.jl"]
PROG_2_3 = ["python", EXPDIR / "call_jl_diffeq_from_python.py", "jl_diffeq"]

DATA_3 = OUTDIR / "runtime_vs_resolution_julia_sundials.csv"

RESULT_PYTHON_STATS = OUTDIR / "runtime_vs_resolution_python_stats.csv"
RESULT_JULIA_STATS = OUTDIR / "runtime_vs_resolution_julia_stats.csv"


def main():
    if not data1_are_present():
        print("Data1 are not present, running compute_1()")
        compute_1()
    else:
        print("Data1 are present, running process_1()")
        process_1()

    if not data2_are_present():
        print("Data2 are not present, running compute_2()")
        compute_2()
    else:
        print("Data2 are present, running process_2()")
        process_2()

    # if not data3_are_present():
    #     print("Data3 are not present, running compute_3()")
    #     # compute_3()
    #     raise NotImplementedError()
    # else:
    #     print("Data3 are present, running process_3()")
    #     process_3()

    # if not data4_are_present():
    #     print("Data4 are not present, running compute_4()")
    #     raise NotImplementedError()
    # else:
    #     process_4()


def data1_are_present():
    are_present = True
    if not os.path.isfile(DATA_1):
        are_present = False

    return are_present


def data2_are_present():
    for f in [DATA_2_1, DATA_2_3]:
        if not os.path.isfile(f):
            return False

    return True


def data3_are_present():
    are_present = True
    if not os.path.isfile(DATA_3):
        are_present = False

    return are_present


def compute_1():
    p1 = subprocess.run(
        PROG_1,
        encoding="utf-8",
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
        check=False,
    )
    assert p1.returncode == 0


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
    print("Python OIF vs Native")
    subprocess.run(["column", "-s,", "-t"], stdin=open(RESULT_PYTHON_STATS, "r"))
    # print()
    # print("To view the results next time, you can use the following command:")
    # print("    column -s, -t ", RESULT_PYTHON_STATS)


def compute_2():
    p2_1 = subprocess.run(PROG_2_1, encoding="utf-8")
    assert p2_1.returncode == 0

    p2_3 = subprocess.run(PROG_2_3, encoding="utf-8")
    assert p2_3.returncode == 0


def process_2():
    f_2_1 = open(DATA_2_1, "r")
    # f_2_2 = open(DATA_2_2, "r")
    f_2_3 = open(DATA_2_3, "r")

    f_2_out = open(RESULT_JULIA_STATS, "w")
    for line in f_2_1.readlines():
        f_2_out.write(line)

    # noinline in Julia
    # f_2_2.readline()  # skip header
    # for line in f_2_2.readlines():
    #     f_2_out.write(line)

    f_2_3.readline()  # skip header
    f_2_out.write(f_2_3.readline())

    f_2_1.close()
    # f_2_2.close()
    f_2_out.close()

    print()
    print("Julia OIF from Python vs Native")
    subprocess.run(["column", "-s,", "-t"], stdin=open(RESULT_JULIA_STATS, "r"))


def process_3():
    print()
    print("Julia DP5 versus Sundials.jl CVODE_Adams")
    subprocess.run(["column", "-s,", "-t", DATA_3])


if __name__ == "__main__":
    main()
