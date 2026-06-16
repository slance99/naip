#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=sc_yearly_naip
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts
#SBATCH --gres=shard:4

nvidia-smi

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

# Run script
python sac_omni_yearly.py
