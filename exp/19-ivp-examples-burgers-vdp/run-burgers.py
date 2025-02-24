"""Plot solutions of Burgers' equation obtained with 3 implementations."""

import os
import subprocess
from pathlib import Path

import params

from helpers import get_outdir

OUTDIR = get_outdir()
PROG = Path(os.environ["OIF_DIR"]) / "build/examples/call_ivp_from_c_burgers_eq"


outfile = os.path.join(OUTDIR, params.FILENAME_SOLN_BURGERS)
subprocess.run([PROG, "scipy_ode", outfile], check=True)

prog = Path(os.environ["OIF_DIR"]) / "examples/call_ivp_from_python_vdp.py"
subprocess.run(["python", prog, "jl_diffeq", "--outdir", OUTDIR])
