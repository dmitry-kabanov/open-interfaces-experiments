#!/usr/bin/env python
"""Run OIF and raw versions of Burgers' solver multiple time and compute statistics."""
import subprocess

import numpy as np

from helpers import get_expdir, get_outdir

OUTDIR = get_outdir()
EXPDIR = get_expdir()

RESULTS_OIF_FILENAME = OUTDIR / "oif.txt"
RESULTS_RAW_FILENAME = OUTDIR / "raw.txt"

RESOLUTION_LIST = [200, 400, 800, 1600, 3200, 6400]


def _get_runtimes(version: str) -> list[float]:
    assert version in ["oif", "raw"]

    runtimes = []
    for N in RESOLUTION_LIST:
        p_oif = subprocess.run(
            [f"./run_burgers_{version}", str(N)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        print(p_oif.stdout)
        assert p_oif.returncode == 0
        for line in p_oif.stdout.split("\n"):
            if line.startswith("Elapsed time ="):
                chunks = line.split()
                runtime = float(chunks[-2])
                assert runtime > 0
                break
        runtimes.append(runtime)
        print()

    return runtimes


def _compute()
    runtimes_oif = _get_runtimes("oif")
    runtimes_raw = _get_runtimes("raw")

    np.savetxt(RESULTS_OIF_FILENAME, runtimes_oif)
    print(f"Saved runtime results for OIF in {RESULTS_OIF_FILENAME}")

    np.savetxt(RESULTS_RAW_FILENAME, runtimes_raw)
    print(f"Saved runtime results for RAW in {RESULTS_RAW_FILENAME}")

def _process():
    runtimes_oif = np.loadtxt(RESULTS_OIF_FILENAME)
    runtimes_raw = np.loadtxt(RESULTS_RAW_FILENAME)

    mean_oif, ci_oif
