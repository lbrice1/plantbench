"""Wall time of one batched closed-loop evaluation against the size of the batch.

    python benchmarks/scaling.py [--cases ...] [--multiples 1 4 16 64] [--cpu-max 4]
                                 [--states 16 64 256]

A batched Jacobian evaluates the closed loop at n + 1 states, n the number of states; an
ensemble of k members, integrated in lockstep, would evaluate it at k (n + 1).  For each
case and device this times one `closed_loop_rhs_batch` call at m = k (n + 1) states,
k in `--multiples`, the states drawn about the design point with a relative spread of
1e-6.  A time per call that grows more slowly than m means the device is not yet full,
and that integrating ensembles in lockstep would raise the throughput.  `--states` adds
batches of a given number of states, such as the M states at which an ensemble of M
members evaluates each stage of a step.

Each time is the median of `--repeats` calls after one warm-up call, each at fresh
states, the result copied to the host so that the device has finished.  On the GPU the
size of CuPy's memory pool after the calls is also printed, an upper bound on the memory
one call needs.  The last column counts the rows, over the warm-up and the repeats, whose
derivative is not finite: any is a failure of the model at that batch size.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import plantbench as pb
from plantbench import backend
from plantbench.core import control as ctl

CASES = ("ngl_deethanizer", "ngl_demethanizer")


def _states(y0: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y0 * (1.0 + 1e-6 * rng.standard_normal((m, len(y0))))


def _time(setup, y0: np.ndarray, m: int, device: str, repeats: int) -> tuple[float, int]:
    """The median time of one call, each call at fresh states (the plant holds the result
    of its last call at the same states, `Plant.temperatures`, which a repeat would hit),
    and the number of rows, over all the calls, with a derivative that is not finite."""
    d, S = setup.design, setup.structure
    times, bad = [], 0
    for seed in range(repeats + 1):  # the first is the warm-up
        Y = backend.to_device(_states(y0, m, seed), device)
        start = time.perf_counter()
        F = backend.to_host(ctl.closed_loop_rhs_batch(0.0, Y, S, d.u, d.pp, None, d.rhs))
        times.append(time.perf_counter() - start)
        bad += int(np.sum(~np.isfinite(F).all(axis=1)))
    return statistics.median(times[1:]), bad


def _pool_bytes(device: str) -> int | None:
    if device != "cuda":
        return None
    return backend.module("cuda").get_default_memory_pool().total_bytes()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--multiples", nargs="+", type=int, default=[1, 4, 16, 64],
                   help="batch sizes, in Jacobians of n + 1 states (default 1 4 16 64)")
    p.add_argument("--states", nargs="*", type=int, default=[],
                   help="further batch sizes, in states (default none)")
    p.add_argument("--cpu-max", type=int, default=4,
                   help="the largest multiple timed on the CPU (default 4)")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    devices = ["cpu"] + (["cuda"] if backend.available("cuda") else [])
    print(f"devices: {', '.join(devices)}")
    for case_id in args.cases:
        case = pb.load_case(case_id)
        setup = pb.build(case, case.config())
        if not ctl.supports_batch(setup.design):
            print(f"\n{case_id}: no batched right-hand side; skipped")
            continue
        d, S = setup.design, setup.structure
        y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
        n = len(y0)
        print(f"\n{case_id}: {n} states, {n + 1} per Jacobian")
        print(f"  {'device':6s} {'Jacobians':>9s} {'states':>8s} {'s per call':>11s} "
              f"{'us per state':>13s} {'memory pool':>12s} {'not finite':>10s}")
        for device in devices:
            if device == "cuda":
                backend.module("cuda").get_default_memory_pool().free_all_blocks()
            sizes = sorted(set(args.states) | {k * (n + 1) for k in args.multiples})
            for m in sizes:
                if device == "cpu" and m > args.cpu_max * (n + 1):
                    continue
                t, bad = _time(setup, y0, m, device, args.repeats)
                pool = _pool_bytes(device)
                pool = "" if pool is None else f"{pool / 2**30:9.2f} GiB"
                print(f"  {device:6s} {m / (n + 1):9.2f} {m:8d} {t:11.3f} {1e6 * t / m:13.1f} "
                      f"{pool:>12s} {bad:10d}", flush=True)


if __name__ == "__main__":
    main()
