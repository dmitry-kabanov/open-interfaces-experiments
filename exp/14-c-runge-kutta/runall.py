#!/usr/bin/env python
"""Run OIF and raw versions of Burgers' solver multiple time and compute statistics."""
import csv
import os
import subprocess

import numpy as np

from helpers import compute_mean_and_ci, get_expdir, get_outdir

OUTDIR = get_outdir()
EXPDIR = get_expdir()

FLAG_COMPLETED = OUTDIR / "completed"
RESULTS_PERF_FILENAME = "_assets/c-oif-vs-raw.csv"

RESOLUTIONS_LIST = [1600, 6400, 25_600]

N_RUNS = 30
VERSIONS = ["oif", "raw"]


def _get_result_filename(N: int, version: str) -> str:
    dirname = OUTDIR / f"N={N:04d}"
    return str(dirname / f"{version}.txt")


def main():
    if not os.path.isfile(FLAG_COMPLETED):
        _compute()
        open(FLAG_COMPLETED, "a")
    else:
        _process()


def _compute():
    runtimes_table_oif = _get_runtimes("oif")
    runtimes_table_raw = _get_runtimes("raw")

    for N in RESOLUTIONS_LIST:
        print(f"Resolution N = {N:04d}")
        os.makedirs(OUTDIR / "N={N:04d}".format(N=N), exist_ok=True)

        fn = _get_result_filename(N, "oif")
        np.savetxt(fn, runtimes_table_oif[N])
        print(f"Saved runtime results for OIF in {fn}")

        fn = _get_result_filename(N, "raw")
        np.savetxt(fn, runtimes_table_raw[N])
        print(f"Saved runtime results for RAW in {fn}")


def _get_runtimes(version: str) -> dict:
    assert version in VERSIONS

    runtimes_table = {}
    for N in RESOLUTIONS_LIST:
        runtimes = []
        for _ in range(N_RUNS):
            prog = subprocess.run(
                [f"bin/run_burgers_{version}", str(N)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            print(prog.stdout)
            assert prog.returncode == 0
            for line in prog.stdout.split("\n"):
                if line.startswith("Elapsed time ="):
                    chunks = line.split()
                    runtime = float(chunks[-2])
                    assert runtime > 0
                    break
            runtimes.append(runtime)
            print()

        runtimes_table[N] = runtimes

    return runtimes_table


def _process():
    table = {}
    for v in VERSIONS:
        table[v] = []

    for N in RESOLUTIONS_LIST:
        print()
        print(f"Resolution N = {N}")

        runtimes = {}
        runtimes["oif"] = np.loadtxt(_get_result_filename(N, "oif"))
        runtimes["raw"] = np.loadtxt(_get_result_filename(N, "raw"))

        print()
        for v in VERSIONS:
            mean, ci = compute_mean_and_ci(runtimes[v])
            val = f"{mean:.2f} ± {ci:.2f}"
            table[v].append(val)

    with open(RESULTS_PERF_FILENAME, "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# method"] + RESOLUTIONS_LIST)
        for v in VERSIONS:
            writer.writerow(["{:30s}".format(v)] + table[v])

    print(f"Results are written to file '{RESULTS_PERF_FILENAME}'")
    subprocess.run(["column", "-s,", "-t", RESULTS_PERF_FILENAME])


if __name__ == "__main__":
    main()
