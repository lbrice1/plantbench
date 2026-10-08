"""Wall time of the closed-loop simulation under each way of forming the Jacobian.

    python benchmarks/jacobian.py [--t-end 20] [--cases ngl_deethanizer ngl_demethanizer]

For each case that declares a batched right-hand side: one Jacobian, and a run of
`--t-end` minutes with a 10 % step in the feed's ethane at one minute, with LSODA's own
Jacobian, with the batched Jacobian on the host, and on the GPU when CuPy finds one.  The
runs are compared against the first for the largest relative difference in the states.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import plantbench as pb
from plantbench import backend
from plantbench.core import control as ctl

CASES = ("ngl_deethanizer", "ngl_demethanizer")


def _jacobian_time(setup, device: str, repeats: int = 3) -> float:
    """The mean time of one Jacobian, each at a fresh state: the plant holds the result of
    its last call at the same states (`Plant.temperatures`), which a repeat would hit."""
    d, S = setup.design, setup.structure
    y = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    rng = np.random.default_rng(0)
    states = [backend.to_device(y * (1.0 + 1e-6 * rng.standard_normal(len(y))), device)
              for _ in range(repeats + 1)]
    args = (S, d.u, d.pp, None, d.rhs)
    backend.to_host(ctl.jacobian_batched(0.0, states[0], *args))  # warm-up: caches, kernels
    start = time.perf_counter()
    for y0 in states[1:]:
        backend.to_host(ctl.jacobian_batched(0.0, y0, *args))
    return (time.perf_counter() - start) / repeats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--t-end", type=float, default=20.0, help="minutes simulated (default 20)")
    p.add_argument("--cases", nargs="+", default=list(CASES))
    args = p.parse_args()
    devices = ["cpu"] + (["cuda"] if backend.available("cuda") else [])
    print(f"devices: {', '.join(devices)}")
    for case_id in args.cases:
        case = pb.load_case(case_id)
        config = case.config(disturbance={"name": "feed ethane", "fraction": 0.1, "t": 1.0})
        setup = pb.build(case, config)
        n = len(setup.design.x) + ctl.n_augmented(setup.structure)
        print(f"\n{case_id}: {n} states")
        for device in devices:
            print(f"  one batched Jacobian, {device:4s}  {_jacobian_time(setup, device):8.3f} s")
        ref = None
        for jacobian, device in [("internal", "cpu")] + [("batched", d) for d in devices]:
            tr = pb.run(case, config, t_end=args.t_end, dt=1.0, jacobian=jacobian, device=device)
            line = (f"  run {jacobian:8s} {device:4s}  {tr.wall_time:8.1f} s  "
                    f"{60.0 * tr.wall_time / args.t_end:8.1f} s per simulated hour  "
                    f"nfev {tr.nfev}")
            if ref is None:
                ref = tr
            else:
                diff = np.max(np.abs(tr.x - ref.x) / (1e-6 + np.abs(ref.x)))
                line += f"  largest relative state difference {diff:.1e}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
