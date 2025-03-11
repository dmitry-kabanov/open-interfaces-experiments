#!/usr/bin/env python

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from helpers import compute_mean_and_ci

IC_FILENAME_TPL = "_output/N={:04d}/ic-{:s}.txt"
SOLUTION_FILENAME_TPL = "_output/N={:04d}/solution-{:s}.txt"
RUNTIME_FILENAME_TPL = "_output/N={:04d}/runtimes-{:s}.txt"

RESOLUTION_LIST = [1600, 6400, 25_600]
# RESOLUTION_LIST = [1600]
N_TRIALS = 30


def run(N: int) -> None:
    os.makedirs("_output/N={:04d}".format(N), exist_ok=True)

    ic_file_1 = IC_FILENAME_TPL.format(N, "c")
    ic_file_2 = IC_FILENAME_TPL.format(N, "julia")

    solution_file_1 = SOLUTION_FILENAME_TPL.format(N, "c")
    solution_file_2 = SOLUTION_FILENAME_TPL.format(N, "julia")

    result_file_1 = RUNTIME_FILENAME_TPL.format(N, "c")
    result_file_2 = RUNTIME_FILENAME_TPL.format(N, "julia")

    if (
        os.path.isfile(ic_file_1)
        and os.path.isfile(ic_file_2)
        and os.path.isfile(solution_file_1)
        and os.path.isfile(solution_file_2)
        and os.path.isfile(result_file_1)
        and os.path.isfile(result_file_2)
    ):
        return

    run_1 = subprocess.run(
        ["bin/call_from_c", str(N), str(N_TRIALS)],
        encoding="utf-8",
    )
    assert run_1.returncode == 0

    run_2 = subprocess.run(
        ["julia", "call_ivp_julia.jl", str(N), str(N_TRIALS)],
        encoding="utf-8",
    )
    assert run_2.returncode == 0

    ic_1 = np.loadtxt(ic_file_1)
    ic_2 = np.loadtxt(ic_file_2)
    np.testing.assert_allclose(ic_1, ic_2, rtol=1e-15, atol=1e-15)

    result_1 = np.loadtxt(solution_file_1)
    result_2 = np.loadtxt(solution_file_2)
    # For resolution 25_600, there is a mismatch of 0.2 % of array elements
    # if I use 1e-6 for rtol and atol, hence use of 1e-5.
    np.testing.assert_allclose(result_1, result_2, rtol=1e-5, atol=1e-5)


def main():
    for N in RESOLUTION_LIST:
        run(N)

    runtimes_mean_c = []
    runtimes_mean_julia = []
    runtimes_ci_c = []
    runtimes_ci_julia = []
    row_header = ["Method/N"]
    row_c = ["C + OIF"]
    row_julia = ["Julia native"]

    for N in RESOLUTION_LIST:
        runtimes_c = np.loadtxt(RUNTIME_FILENAME_TPL.format(N, "c"))
        runtimes_julia = np.loadtxt(RUNTIME_FILENAME_TPL.format(N, "julia"))
        assert len(runtimes_c) == len(runtimes_julia)

        mean_c, ci_c = compute_mean_and_ci(runtimes_c)
        mean_julia, ci_julia = compute_mean_and_ci(runtimes_julia)

        runtimes_mean_c.append(mean_c)
        runtimes_mean_julia.append(mean_julia)

        runtimes_ci_c.append(ci_c)
        runtimes_ci_julia.append(ci_julia)

        print(f"N={N:04d}, C mean runtime, sec: {mean_c:.3f} ± {ci_c:.3f}")
        print(f"N={N:04d}, J mean runtime, sec: {mean_julia:.3f} ± {ci_julia:.3f}")
        print()

        row_header.append(f"{N:9d}")
        row_c.append(f"{mean_c:.3f} ± {ci_c:.3f}")
        row_julia.append(f"{mean_julia:.3f} ± {ci_julia:.3f}")

    print(" | ".join(row_header) + " |")
    print(" | ".join(row_c) + " |")
    print(" | ".join(row_julia) + " |")

    plt.figure()
    plt.errorbar(
        RESOLUTION_LIST, runtimes_mean_c, yerr=runtimes_ci_c, fmt="o", label="C + OIF"
    )
    plt.errorbar(
        RESOLUTION_LIST,
        runtimes_mean_julia,
        yerr=runtimes_ci_julia,
        fmt="s",
        label="Julia",
    )
    plt.gca().set_xticks(RESOLUTION_LIST)
    plt.xlabel("Resolution")
    plt.ylabel("Runtime, sec")
    plt.tight_layout(pad=0.1)

    plt.savefig("_assets/perf-c-vs-julia.pdf")
    plt.savefig("_assets/perf-c-vs-julia.png")


if __name__ == "__main__":
    main()
