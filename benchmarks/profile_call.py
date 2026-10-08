"""Where the time of one batched closed-loop call goes, on the host or a GPU.

    python benchmarks/profile_call.py --case ngl_demethanizer --device cuda --states 64

Times the call with `cupyx.profiler.benchmark` on the GPU, which separates the time the
host spends issuing work (CPU) from the time the device spends on it (GPU): a GPU time far
below the CPU time means the call is bound by issuing kernels and waiting on the device,
not by arithmetic.  Then profiles one call with cProfile and prints the functions with the
most time of their own and the most calls.  On the host only the cProfile part runs.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np

import plantbench as pb
from plantbench import backend
from plantbench.core import control as ctl

sys.path.insert(0, str(Path(__file__).parent))
from scaling import _states  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="ngl_demethanizer")
    p.add_argument("--device", default="cuda", choices=backend.DEVICES)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()
    case = pb.load_case(args.case)
    setup = pb.build(case, case.config())
    d, S = setup.design, setup.structure
    y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    batches = [backend.to_device(_states(y0, args.states, seed), args.device)
               for seed in range(12)]

    def call(Y):
        return backend.to_host(ctl.closed_loop_rhs_batch(0.0, Y, S, d.u, d.pp, None, d.rhs))

    call(batches[0])  # warm-up
    print(f"{args.case}, {args.states} states, {args.device}")
    if args.device == "cuda":
        from cupyx.profiler import benchmark

        # Fresh states on every repeat, as the plant holds its last result.
        it = iter(batches[1:])
        print(benchmark(lambda: call(next(it)), n_repeat=8, n_warmup=0))
    prof = cProfile.Profile()
    prof.enable()
    call(batches[-1])
    prof.disable()
    stats = pstats.Stats(prof)
    print(f"\nthe {args.top} functions with the most time of their own")
    stats.sort_stats("tottime").print_stats(args.top)
    print(f"\nthe {args.top} functions called most often")
    stats.sort_stats("ncalls").print_stats(args.top)


if __name__ == "__main__":
    main()
