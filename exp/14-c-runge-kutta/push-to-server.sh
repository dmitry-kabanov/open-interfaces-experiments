#!/bin/bash

# Check if rsync is installed
if ! command -v rsync &> /dev/null; then
    echo "Error: rsync is not installed."
    exit 1
fi

# Define the target server and base path
SERVER="myri"
TARGET_BASE_PATH="~/target/directory" # Update this path to the base directory on the server

dirname=${PWD/#$HOME/}

# Exclude patterns
EXCLUDE=(
    "--exclude=_output/"
    "--exclude=_assets/"
    "--exclude=*.sh"
)

# Run rsync
rsync -avz --progress "${EXCLUDE[@]}" ./ "${SERVER}:${dirname}"

# Confirm completion
if [ $? -eq 0 ]; then
    echo "Files successfully copied to ${SERVER}:${dirname}."
    echo "Do not forget to do \`make clean\` and \`make\` to recompile"
else
    echo "Error during file sync."
fi
