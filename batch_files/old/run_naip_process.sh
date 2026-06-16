#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --job-name=naip_download
#SBATCH --mail-type=END                
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate naip_env

# Run script
python naip_processing.py
