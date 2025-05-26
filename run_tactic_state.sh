#!/bin/bash
#SBATCH --account=def-papyan
#SBATCH --job-name=lean_repl_d6
#SBATCH --time=0-12:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/%x-%j.out

source trainENV/bin/activate

# Run the command
python get_tactic_state.py dataset_006.jsonl