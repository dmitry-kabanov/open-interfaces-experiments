#!/bin/bash

# Monitor changes to note.md files in exp/*/ directories
inotifywait -e modify -e create -e delete --monitor --recursive exp |
while read -r directory action file; do
    if [[ "${file}" == "note.md" ]]; then
        make notebook
    fi
done
