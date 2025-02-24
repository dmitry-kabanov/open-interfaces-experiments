"""Process simulation results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import params

from helpers import FIGSIZE_NORMAL, get_outdir

OUTDIR = get_outdir()

fig, axes = plt.subplots(figsize=FIGSIZE_NORMAL, nrows=1, ncols=1)

fn = os.path.join(OUTDIR, params.FILENAME_SOLN_BURGERS)
data = np.loadtxt(fn)
x, soln = data[:, 0], data[:, 1]
axes.plot(x, soln, "-")
axes.set_xlabel("$x$")
axes.set_ylabel("Solution")
plt.tight_layout(pad=0.1)
plt.savefig("_assets/ivp-burgers.pdf")
plt.show()

# Van der Pol
fig, axes = plt.subplots(figsize=FIGSIZE_NORMAL, nrows=1, ncols=1)
fn = os.path.join(OUTDIR, "ivp_py_vdp_eq_jl_diffeq.txt")
data = np.loadtxt(fn)
t, y1 = data[:, 0], data[:, 1]
axes.plot(t, y1, "-")
axes.set_xlabel("$t$")
axes.set_ylabel("Solution")
plt.tight_layout(pad=0.1)
plt.savefig("_assets/ivp-vdp.pdf")
plt.show()

open("_assets/.done", "wa").close()
