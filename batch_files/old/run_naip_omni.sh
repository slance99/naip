#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err:
#SBATCH --job-name=2003_2025_naip
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts
#SBATCH --gres=gpu:1


# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

# Run script
python naip_omni.py
