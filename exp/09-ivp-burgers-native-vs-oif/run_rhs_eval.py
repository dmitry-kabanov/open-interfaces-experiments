import subprocess

from helpers import get_expdir, get_outdir

OUTDIR = get_outdir()
EXPDIR = get_expdir()

RESULT_RHS_EVALS_FILENAME = OUTDIR / "rhs_evals.txt"

p1 = subprocess.run(
    ["python", EXPDIR / "call_rhs_eval_python.py"],
    encoding="utf-8",
    capture_output=True,
)
assert p1.returncode == 0
print(p1.stdout)
print(p1.stderr)

print()

p2 = subprocess.run(
    ["julia", EXPDIR / "call_rhs_eval_julia.jl"], encoding="utf-8", capture_output=True
)
assert p2.returncode == 0
print(p2.stdout)
print(p2.stderr)

with open(RESULT_RHS_EVALS_FILENAME, "w") as fh:
    fh.write(p1.stdout)
    fh.write(p2.stdout)
