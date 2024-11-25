import argparse
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("dirname", help="Dirname of the experiment to copy")
args = p.parse_args()

rootdir = os.getcwd()
bookdir = Path("oif-notebook")

dirname = Path(args.dirname)
note = dirname / "note.md"
assetsdir = dirname / "_assets"

if not os.path.isdir(dirname):
    raise RuntimeError("Argument must be a directory for computational experiment")

if not os.path.isfile(note):
    raise RuntimeError(f"File '{note.as_posix():s}' does not exist")

if not os.path.isdir(assetsdir):
    raise RuntimeError(f"Directory '{assetsdir.as_posix():s}' does not exist")


os.makedirs(bookdir / dirname, exist_ok=True)
os.symlink(rootdir / note, bookdir / note)
os.symlink(rootdir / assetsdir, bookdir / assetsdir)
