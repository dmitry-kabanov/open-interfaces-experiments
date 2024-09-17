import subprocess

from helpers import get_expdir, get_outdir

OUTDIR = get_outdir()
EXPDIR = get_expdir()

p1 = subprocess.run(["python", EXPDIR / "call_rhs_eval_python.py"])
assert p1.returncode == 0

p2 = subprocess.run(["julia", EXPDIR / "call_rhs_eval_julia.jl"])
assert p2.returncode == 0
