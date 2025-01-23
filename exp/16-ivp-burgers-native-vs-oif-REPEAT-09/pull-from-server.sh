#!/bin/bash
# Define the target server and the base path
SERVER="myri"
dirname=${PWD/#$HOME\/}

# Run rsync
if rsync -avz --delete --progress "${SERVER}:${dirname}/_output*" . ; then
    echo "Files are successfully pulled from ${SERVER}:${dirname}"
else
    echo "Error during pulling files from server"
fi
