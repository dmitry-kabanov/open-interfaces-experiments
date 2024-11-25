#!/usr/bin/env python

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from helpers import compute_mean_and_ci

SOLUTION_FILENAME_TPL = "_output/N={:04d}/solution-{:s}.txt"
RUNTIME_FILENAME_TPL = "_output/N={:04d}/runtimes-{:s}.txt"

RESOLUTION_LIST = [100, 200, 400, 800, 1600, 3200, 6400]
RESOLUTION_LIST = [6400]


def run(N: int) -> None:
    os.makedirs("_output/N={:04d}".format(N), exist_ok=True)

    solution_file_1 = SOLUTION_FILENAME_TPL.format(N, "c")
    solution_file_2 = SOLUTION_FILENAME_TPL.format(N, "julia")

    result_file_1 = RUNTIME_FILENAME_TPL.format(N, "c")
    result_file_2 = RUNTIME_FILENAME_TPL.format(N, "julia")

    if (
        os.path.isfile(solution_file_1)
        and os.path.isfile(solution_file_2)
        and os.path.isfile(result_file_1)
        and os.path.isfile(result_file_2)
    ):
        return

    run_1 = subprocess.run(
        ["./call_from_c", str(N)],
        encoding="utf-8",
    )
    assert run_1.returncode == 0

    run_2 = subprocess.run(
        ["julia", "call_ivp_julia.jl", str(N)],
        encoding="utf-8",
    )
    assert run_2.returncode == 0

    result_1 = np.loadtxt(solution_file_1)
    result_2 = np.loadtxt(solution_file_2)

    np.testing.assert_allclose(result_1, result_2, rtol=1e-6, atol=1e-6)

    runtimes_1 = np.loadtxt(result_file_1)
    runtimes_2 = np.loadtxt(result_file_2)
    assert len(runtimes_1) == len(runtimes_2)

    mean_1, ci_1 = compute_mean_and_ci(runtimes_1)
    mean_2, ci_2 = compute_mean_and_ci(runtimes_2)

    print(f"N={N:04d}, C mean runtime, sec: {mean_1:.3f} ± {ci_1:.3f}")
    print(f"N={N:04d}, J mean runtime, sec: {mean_2:.3f} ± {ci_2:.3f}")


def main():
    for N in RESOLUTION_LIST:
        run(N)

    runtimes_mean_c = []
    runtimes_mean_julia = []
    runtimes_ci_c = []
    runtimes_ci_julia = []
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


if __name__ == "__main__":
    main()
