#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=clean_masks
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts/processing

RIVER=$1
OUTPUTS=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

python -u cleaning_watermasks.py $RIVER $OUTPUTS
