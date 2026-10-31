#!/bin/bash
# Two commits of plantbench timed against each other on one cluster node, alternating, so
# that both see the same hardware and load.
#
# On the login node, from the checkout of the newer commit with `pixi install` done (and
# `pixi install -e cuda` for a GPU run):
#
#   git worktree add --detach ../pb-old <older commit>
#   sbatch --account=<account> benchmarks/cluster_ab.slurm                 # CPU node
#   sbatch --account=<account> --gres=gpu:1 --constraint=<feature> \
#       --export=ALL,ENV=cuda benchmarks/cluster_ab.slurm                  # GPU node
#
# The account and the GPU are given as for cluster.slurm.
# The older commit runs from inside its worktree with the checkout's pixi environment:
# the current directory comes first on sys.path, ahead of the editable install of the
# checkout.  The first lines of the output name the commit each side imported; a run whose
# two lines name the same commit compared nothing.  `OLD` overrides the worktree's path,
# `T_END` the simulated minutes of jacobian.py.  The serial cost of a change is measured on
# a CPU node; the scaling of the batched Jacobian needs the GPU form.
#SBATCH --job-name=pb-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=pb-ab-%j.out
set -euo pipefail
NEW=${SLURM_SUBMIT_DIR:-$PWD}
OLD=${OLD:-$NEW/../pb-old}
ENV=${ENV:-default}
T_END=${T_END:-20}
OLD=$(cd "$OLD" && pwd)
# pixi's cache on node-local disk: the environment was installed on the login node, and a
# cache on the network file system draws a warning on every call.
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER

hostname
lscpu | grep 'Model name' || sysctl -n machdep.cpu.brand_string
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true

# Each side runs from its own tree; PYTHONPATH also covers scripts that change directory.
side() {
    local tree=$1; shift
    (cd "$tree" && PYTHONPATH=$tree pixi run --frozen --manifest-path "$NEW/pyproject.toml" \
        -e "$ENV" "$@")
}
check='import plantbench, subprocess; print(plantbench.__file__, subprocess.run(["git", "-C", plantbench.__path__[0], "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())'
echo "== imports"
side "$OLD" python -c "$check"
side "$NEW" python -c "$check"

for round in 1 2; do
    for tree in "$OLD" "$NEW"; do
        echo "== jacobian.py, round $round, $(git -C "$tree" log --oneline -1 | cut -c1-7)"
        side "$tree" python benchmarks/jacobian.py --cases ngl_deethanizer --t-end "$T_END"
    done
done
for tree in "$OLD" "$NEW"; do
    echo "== profile_call.py, cpu, one state, $(git -C "$tree" log --oneline -1 | cut -c1-7)"
    side "$tree" python benchmarks/profile_call.py --case ngl_deethanizer --device cpu --states 1 --top 15
done
if [ "$ENV" = cuda ]; then
    for tree in "$OLD" "$NEW"; do
        echo "== scaling.py, $(git -C "$tree" log --oneline -1 | cut -c1-7)"
        side "$tree" python benchmarks/scaling.py --states 16 64 256 --cpu-max 1
    done
fi
echo "== done"
