#!/bin/bash
#SBATCH --account=def-papyan
#SBATCH --job-name=lean_repl_d1
#SBATCH --time=2-12:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --output=logs/%x-%j.out

source trainENV/bin/activate

# Run the command
python get_tactic_state.py dataset_001.jsonl