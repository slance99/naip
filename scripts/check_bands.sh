#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=check_bands
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

python -u check_bands.py /home/geomorph/california_rivers/naip/naip_omni_tiles/smith/smith_1/naip_2010.tif
