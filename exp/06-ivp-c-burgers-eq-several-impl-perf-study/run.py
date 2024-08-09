"""Plot solutions of Burgers' equation obtained with 3 implementations."""

import os
import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers import FIGSIZE_NORMAL, get_output_dir

IMPL_LIST = ["scipy_ode", "jl_diffeq"]

OUTDIR = get_output_dir()
PROG = Path(os.environ["OIF_DIR"]) / "build/examples/call_ivp_from_c_burgers_eq"
STYLES = ["-", "--", ":"]
DATA_FILENAME_TPL = "ivp_c_burgers_eq_{:s}_N={:d}.txt"
RESOLUTIONS_LIST = [101, 201, 401, 801, 1601]
RESULT_FIG_FILENAME = OUTDIR / "ivp_c_burgers_eq.pdf"
RESULT_STATS = OUTDIR / "stats.pickle"


def main():
    if not data_are_present():
        print("Data are not present, running compute()")
        compute()
    else:
        print("Data are present, running plot()")
        plot()


def data_are_present():
    for impl in IMPL_LIST:
        for N in RESOLUTIONS_LIST:
            fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl, N))
            if not os.path.isfile(fn):
                print("'{:s}' is not a file".format(fn))
                return False
    if not os.path.isfile(RESULT_STATS):
        return False
    return True


def compute():
    os.makedirs(OUTDIR, exist_ok=True)
    stats = {}
    for impl in IMPL_LIST:
        for N in RESOLUTIONS_LIST:
            outfile = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl, N))
            p = subprocess.run(
                [PROG, impl, outfile, str(N)],
                check=True,
                capture_output=True,
            )
            output_lines = p.stdout.decode()
            print("===")
            print(f"Implementation {impl}, N = {N}")
            print(output_lines)

            runtime = 0.0
            n_rhs_evals = 0
            for line in output_lines.split("\n"):
                print(line)
                if line.startswith("Elapsed time"):
                    print("HEERE 1")
                    chunks = line.split(" ")
                    runtime = float(chunks[-2])
                if line.startswith("Number of right-hand"):
                    print("HEERE 2")
                    chunks = line.split(" ")
                    n_rhs_evals = int(chunks[-1])

            assert runtime > 0.0
            assert n_rhs_evals > 0

            stats[(impl, N)] = (runtime, n_rhs_evals)

    with open(RESULT_STATS, "wb") as f:
        pickle.dump(stats, f)


def plot():
    plt.figure(figsize=FIGSIZE_NORMAL)

    N = 101
    for i, impl in enumerate(IMPL_LIST):
        fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl, N))
        data = np.loadtxt(fn)
        x, soln = data[:, 0], data[:, 1]
        plt.plot(x, soln, STYLES[i], label=impl)

    plt.xlabel("$x$")
    plt.ylabel("Solution")
    plt.legend(loc="upper right")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_FIG_FILENAME)
    plt.show()

    with open(RESULT_STATS, "rb") as f:
        stats = pickle.load(f)

        for N in RESOLUTIONS_LIST:
            for impl in IMPL_LIST:
                datum = stats[(impl, N)]
                print(f"{N}, {impl}, {datum[0]}, {datum[1]}")


if __name__ == "__main__":
    main()
