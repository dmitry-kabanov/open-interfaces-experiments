#!/bin/bash
# Define the target server and the base path
SERVER="myri"
dirname=${PWD/#$HOME\/}

# Exclude patterns
EXCLUDE=(
    "--exclude=_assets/"
    "--exclude=bin/"
    "--exclude=code/build*"
    "--exclude=_output/"
    "--exclude=*.so"
    "--exclude=.mypy_cache"
    "--exclude=__pycache__"
    "--exclude=code/.cache"
)

# Run rsync
if rsync -avz --delete --progress "${EXCLUDE[@]}" ./ "${SERVER}:${dirname}" ; then
    echo "Files successfully copied to ${SERVER}:${dirname}."
    echo "Do not forget to do \`make clean\` and \`make\` to recompile"
else
    echo "Error during file sync."
fi
