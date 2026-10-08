#!/bin/bash
# Where one batched closed-loop call spends its time on a Della GPU: host against device
# time, and the functions that dominate under CuPy.
#
#   cd <repository>; pixi install -e cuda      # on the login node: compute nodes are offline
#   sbatch benchmarks/della_profile.slurm
#SBATCH --job-name=pb-profile
#SBATCH --account=rse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=00:30:00
#SBATCH --output=pb-profile-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
git log --oneline -1
run() { pixi run --frozen -e cuda "$@"; }
for case in ngl_deethanizer ngl_demethanizer; do
    run python benchmarks/profile_call.py --case "$case" --device cuda --states 64
done
run python benchmarks/scaling.py --multiples 1 64 --states 16 64 256 --cpu-max 1
