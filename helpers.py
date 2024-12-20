"""Auxiliary module to simplify plotting."""

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# Figure size for a single-plot figure that takes 50 % of text width.
FIGSIZE_NORMAL = (3.0, 2)
# Figure size for a single-plot figure that takes about 75 % of text width.
FIGSIZE_LARGER = (4.5, 3)
# Figure size for a two-subplots figure.
FIGSIZE_TWO_SUBPLOTS_TWO_ROWS = (3.0, 4.0)
# Figure size for a figure with two subplots in one row.
FIGSIZE_TWO_SUBPLOTS_ONE_ROW = (6.0, 2)
# Figure size for a figure with two subplots in two rows.
FIGSIZE_WIDE_TWO_SUBPLOTS_TWO_ROWS = (4.5, 4)


def get_expdir() -> pathlib.Path:
    """Return the directory of the current experiment."""
    expdir = pathlib.Path(sys.argv[0]).parent.resolve()
    return expdir


def get_outdir() -> pathlib.Path:
    """Return the directory of the output for the current experiment."""
    outdir = pathlib.Path(sys.argv[0]).parent.resolve() / "_output"
    assert os.path.isdir(outdir)
    outdir = outdir.relative_to(os.getcwd())
    assert os.path.isdir(outdir)
    return outdir


def savefig(filename, dirname="", **kwargs):
    """Save figure if the environment variable SAVE_FIGURES is set."""
    cur_fig = plt.gcf()

    if dirname or "SAVE_FIGURES" in os.environ:
        if os.path.isdir(dirname):
            filename = os.path.join(dirname, filename)
            cur_fig.savefig(filename, **kwargs)
        else:
            raise RuntimeError("Directory `%s` does not exist" % dirname)
    else:
        plt.show()


def compute_mean_and_ci(values: np.ndarray | list) -> tuple[float, float]:
    """Returns mean and err, so that mean ْ± err defines 95% confidence interval."""
    mean = np.mean(values)
    if len(values) > 1:
        dev = np.std(values, ddof=1)
    else:
        print("WARNING: computing deviation without correction")
        dev = np.std(values)
    err = 2 * dev / np.sqrt(len(values))

    return mean, err
