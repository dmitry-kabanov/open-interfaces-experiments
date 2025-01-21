#!/bin/bash
# Define the target server and the base path
SERVER="myri"
dirname=${PWD/#$HOME\/}

# Exclude patterns
EXCLUDE=(
    "--exclude=_output/"
    "--exclude=_assets/"
    "--exclude=code/build*/"
)

# Run rsync
if rsync -avz --delete --progress "${EXCLUDE[@]}" ./ "${SERVER}:${dirname}" ; then
    echo "Files successfully copied to ${SERVER}:${dirname}."
    echo "Do not forget to do \`make clean\` and \`make\` to recompile"
else
    echo "Error during file sync."
fi
