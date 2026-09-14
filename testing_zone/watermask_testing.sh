#!/bin/bash

#SBATCH --job-name=watermask_test
#SBATCH --output=logs/test_watermask_cleaning_%j.out
#SBATCH --error=logs/test_watermask_cleaning_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/testing_zone

export OPT_NUM_THREADS=8
export MKL_NUM_THREADS=8

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

python -u watermask_testing.py

