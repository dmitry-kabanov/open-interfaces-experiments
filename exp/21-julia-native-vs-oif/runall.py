#!/usr/bin/env python

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

from helpers import compute_mean_and_ci

IC_FILENAME_TPL = "_output/N={:04d}/ic-{:s}.txt"
SOLUTION_FILENAME_TPL = "_output/N={:04d}/solution-{:s}.txt"
RUNTIME_FILENAME_TPL = "_output/N={:04d}/runtimes-{:s}.txt"

RESOLUTION_LIST = [1600, 6400, 25_600]
# RESOLUTION_LIST = [1600]
N_TRIALS = 30
# N_TRIALS = 2


def run(N: int) -> None:
    os.makedirs("_output/N={:04d}".format(N), exist_ok=True)

    ic_file_1 = IC_FILENAME_TPL.format(N, "julia-native")
    ic_file_2 = IC_FILENAME_TPL.format(N, "julia-oif")

    solution_file_1 = SOLUTION_FILENAME_TPL.format(N, "julia-native")
    solution_file_2 = SOLUTION_FILENAME_TPL.format(N, "julia-oif")

    result_file_1 = RUNTIME_FILENAME_TPL.format(N, "julia-native")
    result_file_2 = RUNTIME_FILENAME_TPL.format(N, "julia-oif")

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
        ["julia", "call_ivp_julia.jl", str(N), str(N_TRIALS)],
        encoding="utf-8",
    )
    assert run_1.returncode == 0

    #     run_2 = subprocess.run(
    #         ["julia", "call_ivp_julia.jl", str(N), str(N_TRIALS)],
    #         encoding="utf-8",
    #     )
    #     assert run_2.returncode == 0

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

    runtimes_mean_julia_native = []
    runtimes_mean_julia_oif = []
    runtimes_ci_julia_native = []
    runtimes_ci_julia_oif = []
    row_header = ["Method/N"]
    row_julia_native = ["Julia native"]
    row_julia_oif = ["Julia OIF"]

    for N in RESOLUTION_LIST:
        runtimes_julia_native = loadtxt(RUNTIME_FILENAME_TPL.format(N, "julia-native"))
        runtimes_julia_oif = loadtxt(RUNTIME_FILENAME_TPL.format(N, "julia-oif"))
        assert len(runtimes_julia_native) == len(runtimes_julia_oif)

        mean_julia_native, ci_julia_native = compute_mean_and_ci(runtimes_julia_native)
        mean_julia_oif, ci_julia_oif = compute_mean_and_ci(runtimes_julia_oif)

        runtimes_mean_julia_native.append(mean_julia_native)
        runtimes_mean_julia_oif.append(mean_julia_oif)

        runtimes_ci_julia_native.append(ci_julia_native)
        runtimes_ci_julia_oif.append(ci_julia_oif)

        print(
            f"N={N:04d}, Native mean runtime, sec: {mean_julia_native:.3f} ± {ci_julia_native:.3f}"
        )
        print(
            f"N={N:04d}, OIFFFF mean runtime, sec: {mean_julia_oif:.3f} ± {ci_julia_oif:.3f}"
        )
        print()

        row_header.append(f"{N:9d}")
        row_julia_native.append(f"{mean_julia_native:.3f} ± {ci_julia_native:.3f}")
        row_julia_oif.append(f"{mean_julia_oif:.3f} ± {ci_julia_oif:.3f}")

    print(" | ".join(row_header) + " |")
    print(" | ".join(row_julia_native) + " |")
    print(" | ".join(row_julia_oif) + " |")

    plt.figure()
    plt.errorbar(
        RESOLUTION_LIST,
        runtimes_mean_julia_native,
        yerr=runtimes_ci_julia_native,
        fmt="o",
        label="Julia native",
    )
    plt.errorbar(
        RESOLUTION_LIST,
        runtimes_mean_julia_oif,
        yerr=runtimes_ci_julia_oif,
        fmt="s",
        label="Julia OIF",
    )
    plt.gca().set_xticks(RESOLUTION_LIST)
    plt.xlabel("Resolution")
    plt.ylabel("Runtime, sec")
    plt.tight_layout(pad=0.1)

    plt.savefig("_assets/perf-julia-native-vs-oif.pdf")
    plt.savefig("_assets/perf-julia-native-vs-oif.png")


if __name__ == "__main__":
    main()
