#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --job-name=testing_batch

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run script
python hello.py
