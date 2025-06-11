#!/bin/bash
#SBATCH --account=def-papyan
#SBATCH --job-name=lean_repl_d4
#SBATCH --time=3-12:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --output=logs/%x-%j.out

cd ..
source trainENV/bin/activate
cd data

# Run the command
python get_tactic_state.py dataset_004.jsonl