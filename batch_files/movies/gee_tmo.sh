#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=mattole_overlay
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts/movies

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

# Run script
python -u gee_tmo.py
