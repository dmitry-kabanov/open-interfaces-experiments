"""Plot solutions of Burgers' equation obtained with 3 implementations."""

import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers import FIGSIZE_NORMAL, get_output_dir

IMPL_LIST = ["scipy_ode", "jl_diffeq"]

OUTDIR = get_output_dir()
PROG = Path(os.environ["OIF_DIR"]) / "build/examples/call_ivp_from_c_burgers_eq"
STYLES = ["-", "--", ":"]
DATA_FILENAME_TPL = "ivp_c_burgers_eq_{:s}.txt"
RESULT_FIG_FILENAME = OUTDIR / "ivp_c_burgers_eq.pdf"


def main():
    if not data_are_present():
        print("Data are not present, running compute()")
        compute()
    else:
        print("Data are present, running plot()")
        plot()


def data_are_present():
    for impl in IMPL_LIST:
        fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        if not os.path.isfile(fn):
            print("'{:s}' is not a file".format(fn))
            return False
    return True


def compute():
    os.makedirs(OUTDIR, exist_ok=True)
    for impl in IMPL_LIST:
        outfile = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        subprocess.run([PROG, impl, outfile], check=True)


def plot():
    plt.figure(figsize=FIGSIZE_NORMAL)

    for i, impl in enumerate(IMPL_LIST):
        fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        data = np.loadtxt(fn)
        x, soln = data[:, 0], data[:, 1]
        plt.plot(x, soln, STYLES[i], label=impl)

    plt.xlabel("$x$")
    plt.ylabel("Solution")
    plt.legend(loc="upper right")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_FIG_FILENAME)
    plt.show()


if __name__ == "__main__":
    main()
