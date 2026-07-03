#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=mattole_naip
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts/processing
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hpc-12.grit.ucsb.edu

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni_env

python -u gee_processor.py
