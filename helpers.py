"""Auxiliary module to simplify plotting."""

import atexit
import functools
import json
import os
import pathlib
import shutil
import sys
from datetime import datetime

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


def exp(main_fn):
    @functools.wraps(main_fn)
    def wrapper_exp(args=sys.argv):
        now = datetime.now()
        outdir = ("_output." + now.isoformat(timespec="seconds")).replace(":", ".")

        if os.path.isdir(outdir):
            print(f"ERROR: directory `{outdir}` already exists", file=sys.stderr)
            sys.exit(1)
        os.makedirs(outdir)

        status = 2
        begin_time = now
        try:
            status = main_fn(args, outdir=pathlib.Path(outdir))
            if status is None:
                status = 0
        except KeyboardInterrupt:
            print("ERROR: experiment was interrupted", file=sys.stderr)
            status = 1
        except Exception as e:
            status = 2
            raise e
        finally:
            if status != 0:
                print("ERROR: experiment was not successful", file=sys.stderr)
                shutil.move(outdir, outdir + ".fail")
                outdir = outdir + ".fail"
            else:
                if os.path.exists("_output.latest"):
                    os.remove("_output.latest")
                os.symlink(outdir, "_output.latest")

            end_time = datetime.now()

            summary = {
                "begin_time": begin_time.isoformat(timespec="seconds"),
                "end_time": end_time.isoformat(timespec="seconds"),
                "args": " ".join(args),
                "status": status,
                "outdir": outdir,
            }
            summary_json = json.dumps(summary, indent=4)
            with open(os.path.join(outdir, "summary.json"), "w") as fh:
                fh.write(summary_json)
            os.chmod(outdir, mode=0o555)

            atexit.register(
                lambda summary: print(f"\n=== Experiment summary:\n{summary}"),
                summary_json,
            )

        return status

    return wrapper_exp
