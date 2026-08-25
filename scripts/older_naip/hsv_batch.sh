#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=hsv
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts/older_naip

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

python -u hsv_testing_v2.py
