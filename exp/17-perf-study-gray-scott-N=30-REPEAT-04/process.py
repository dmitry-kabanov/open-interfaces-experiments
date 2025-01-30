#!/usr/bin/env python
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import src.compare_performance_ivp_cvode_gray_scott as gs
from params import RESULT_DATA_PICKLE, RESULT_PERF_FILENAME

from helpers import FIGSIZE_TWO_SUBPLOTS_TWO_ROWS

HUMAN_LABELS = {
    "sundials_cvode": "Open Interfaces",
    "native_sundials_cvode": "Scikit ODES",
}

STYLES = ["-o", "--s", ".^"]

RESULT_DATA_PICKLE = os.path.join("_output.latest", RESULT_DATA_PICKLE)


def process():
    with open(RESULT_DATA_PICKLE, "rb") as fh:
        tts_list = pickle.load(fh)

    tts_stats = gs.compute_stats(tts_list)

    gs.print_stats(tts_stats)

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=FIGSIZE_TWO_SUBPLOTS_TWO_ROWS
    )
    for i, impl in enumerate(gs.IMPL_LIST):
        tts_ave = [tts_stats[impl][N]["tts_ave"] for N in gs.RESOLUTIONS]
        tts_std = [tts_stats[impl][N]["tts_std"] for N in gs.RESOLUTIONS]
        axes[0].errorbar(
            gs.RESOLUTIONS,
            tts_ave,
            fmt=STYLES[i],
            yerr=tts_std,
            label=HUMAN_LABELS[impl],
        )
    axes[0].set_ylabel("Run time, seconds")
    axes[0].legend(loc="best")

    # Plot relative times (normalized by native performance).
    impl = gs.IMPL_LIST[-1]
    assert impl.startswith("native_")
    assert len(gs.IMPL_LIST) == 2
    tts_ave_native = np.array([tts_stats[impl][N]["tts_ave"] for N in gs.RESOLUTIONS])
    tts_std_native = np.array([tts_stats[impl][N]["tts_std"] for N in gs.RESOLUTIONS])

    for impl in gs.IMPL_LIST[:-1]:
        tts_ave = np.array([tts_stats[impl][N]["tts_ave"] for N in gs.RESOLUTIONS])
        tts_std = np.array([tts_stats[impl][N]["tts_std"] for N in gs.RESOLUTIONS])
        tts_std_normalized = np.sqrt(
            np.square(tts_std / tts_ave) + np.square(tts_std_native / tts_ave_native)
        )
        axes[1].errorbar(
            gs.RESOLUTIONS,
            tts_ave / tts_ave_native,
            yerr=tts_std_normalized,
            fmt=STYLES[0],
            label=HUMAN_LABELS[impl],
        )
    axes[1].set_xlabel(r"Resolution $N$")
    axes[1].set_ylabel("Normalized run time")
    axes[1].set_xticks(gs.RESOLUTIONS)
    plt.tight_layout(pad=0.1)

    plt.savefig(RESULT_PERF_FILENAME)


if __name__ == "__main__":
    process()
