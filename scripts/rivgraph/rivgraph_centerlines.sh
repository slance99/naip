#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=centerlines
#SBATCH --mail-type=END
#SBATCH --mail-user=slance@ucsb.edu
#SBATCH --chdir=/home/geomorph/california_rivers/naip/scripts/rivgraph/

# conda info --base prints a bare path (unlike `mamba info --base`, which
# prints a labeled line like "base environment : /path (writable)" that
# breaks naive command substitution).
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/slance/miniconda3")

source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Checking environment..."
conda run -n rivgraph_yml python check_env.py

echo "Running batch job..."
conda run -n rivgraph_yml python -u rivgraph_centerlines.py

echo "Job finished at $(date)"
