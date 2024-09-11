import argparse
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("dirname", help="Dirname of the experiment to copy")
args = p.parse_args()

dirname = Path(args.dirname)
note = dirname / "note.md"

if not os.path.isdir(dirname):
    raise RuntimeError("Argument must be a directory for computational experiment")

if not os.path.isfile(note):
    raise RuntimeError(f"Note.md is not present where should be: '{note:s}'")


rootdir = os.getcwd()
bookdir = Path("oif-notebook")

os.makedirs(bookdir / dirname, exist_ok=True)
os.symlink(rootdir / note, bookdir / note)
