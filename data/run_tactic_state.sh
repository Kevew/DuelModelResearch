#!/bin/bash

for i in $(seq -w 1 25); do
    filename="dataset_0${i}.jsonl"
    echo "Processing $filename..."
    python get_tactic_state.py "$filename"
done
