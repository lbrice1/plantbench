#!/bin/bash
# The GPU tests and the benchmarks on one GPU of a Slurm cluster.
#
#   cd <repository>; pixi install -e cuda      # on the login node: compute nodes may be offline
#   sbatch --account=<account> --constraint=<feature> benchmarks/cluster.slurm
#
# The account and the GPU depend on the cluster; the script names neither.  Both come from
# the command line, or from SBATCH_ACCOUNT and SBATCH_CONSTRAINT in the environment.  The
# constraint selects the GPU by node feature, for instance a 40 GB A100 that is not
# partitioned into MIG slices; without it the job takes any GPU.  `sinfo -o "%P %f %D %t"`
# lists the partitions, the features of their nodes and the nodes' state.  The job is
# single-threaded on the host, hence one core, and takes less than an hour.
#SBATCH --job-name=pb-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=pb-bench-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# pixi's cache on node-local disk: the environment was installed on the login node, and a
# cache on the network file system draws a warning on every call.
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run() { pixi run --frozen -e cuda "$@"; }
# A failed test is reported and the benchmarks run all the same.
run test-gpu || echo "test-gpu FAILED"
run python benchmarks/scaling.py --states 16 64 256
run python benchmarks/jacobian.py --cases ngl_deethanizer
# Ensembles in lockstep against their number of members, on the GPU; the host only alone.
run python benchmarks/ensemble.py --members 1 4 16 64 --cpu-max 1 --reference 1
# The largest batches again, on the GPU only, with a numerical warning an error: the
# traceback names the solve that overflowed, if one does.
echo "warnings as errors, 64 Jacobians per call:"
run python -W error::RuntimeWarning benchmarks/scaling.py --multiples 64 --cpu-max 0 --repeats 1
