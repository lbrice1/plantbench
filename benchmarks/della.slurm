#!/bin/bash
# The GPU tests and the benchmarks on one Della GPU (A100 40 GB, not MIG).
#
#   cd <repository>; pixi install -e cuda      # on the login node: compute nodes are offline
#   sbatch benchmarks/della.slurm
#
# One hour or less runs under the gpu-test QOS.  The job is single-threaded on the host,
# hence one core.
#SBATCH --job-name=pb-bench
#SBATCH --account=rse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig&gpu40
#SBATCH --time=01:00:00
#SBATCH --output=pb-bench-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run() { pixi run --frozen -e cuda "$@"; }
run test-gpu
run python benchmarks/scaling.py --states 16 64 256
run python benchmarks/jacobian.py --cases ngl_deethanizer
# The largest batches again, on the GPU only, with a numerical warning an error: the
# traceback names the solve that overflowed, if one does.
echo "warnings as errors, 64 Jacobians per call:"
run python -W error::RuntimeWarning benchmarks/scaling.py --multiples 64 --cpu-max 0 --repeats 1
