#!/usr/bin/env python
"""Driver script for performance study based on the Gray--Scott problem."""

import os
import pickle
import shutil

import src.compare_performance_ivp_cvode_gray_scott as gs

from helpers import exp


@exp
def main(args, outdir=""):
    args = gs.parse_args(args)
    run(args)
    move_files_to_outdir(outdir)
    print("Computations are done")


def run(args):
    gs.run_all_impl(args)


def move_files_to_outdir(outdir):
    files = os.listdir("_assets")
    for f in files:
        if f.startswith("ivp_cvode_gs"):
            filename = os.path.join("_assets", f)
            new_filename = os.path.join(outdir, f)
            shutil.move(filename, new_filename)


if __name__ == "__main__":
    args = ["all", "--n_runs", "30"]
    main(args)
