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
print(p1.stdout)
if p1.stderr:
    print(p1.stderr)
assert p1.returncode == 0

print()

p2 = subprocess.run(
    ["julia", EXPDIR / "call_rhs_eval_julia.jl"], encoding="utf-8", capture_output=True
)
print(p2.stdout)
if p2.stderr:
    print(p2.stderr)
assert p2.returncode == 0

with open(RESULT_RHS_EVALS_FILENAME, "w") as fh:
    fh.write(p1.stdout)
    fh.write("\n")
    fh.write(p2.stdout)
