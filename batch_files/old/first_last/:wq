#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/geomorph/california_rivers/naip/scripts/first_last/logs/rm_images-%j.out
#SBATCH --error=/home/geomorph/california_rivers/naip/scripts/first_last/logs/rm_images-%j.err
#SBATCH --job-name=rm_images 
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts/first_last

RIVER=$1
OUTPUTS=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env


python -u first_last_doubles_mile.py $1 $2
