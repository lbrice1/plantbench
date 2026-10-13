"""Throughput of ensembles in lockstep against the number of members.

    python benchmarks/ensemble.py [--case ngl_deethanizer] [--members 1 4 16 64]
                                  [--t-end 20] [--cpu-max 4] [--reference 2]

Member i steps the feed's ethane by a fraction and at a time of its own and has its own
stage 4 temperature gain, so that the members' step sizes differ as a dataset's would.
For each number of members M and each device (the GPU when CuPy finds one, the host up to
`--cpu-max` members), the ensemble is run once and reported as its wall time, runs per
hour and the mean number of rows evaluated per member.  The first `--reference`
members are run alone with LSODA's own Jacobian (`pb.run`), for the time of a serial run
and the largest difference in the states, in units of the tolerance, |dx| / (atol + rtol
|x|): a relative difference would be dominated by states that are zero to within atol, such
as the fraction of a component absent from a stage, where LSODA's own error is of order
atol.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import plantbench as pb
from plantbench import backend

RTOL, ATOL = 1e-8, 1e-10  # those of both integrations
GAIN = {"ngl_deethanizer": "stage 4 temperature", "ngl_demethanizer": None}


def configs(case, m: int) -> list:
    out = []
    for i in range(m):
        sections = {"disturbance": {"name": "feed ethane", "fraction": 0.05 + 0.05 * (i % 4),
                                    "t": 1.0 + 0.5 * (i % 5)}}
        loop = GAIN.get(case.id)
        if loop:
            sections["tuning"] = {loop: {"Kc": -0.25 * (0.6 + 0.8 * ((i * 7) % 11) / 10)}}
        out.append(case.config(**sections))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="ngl_deethanizer")
    p.add_argument("--members", nargs="+", type=int, default=[1, 4, 16, 64])
    p.add_argument("--t-end", type=float, default=20.0, help="minutes simulated (default 20)")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--cpu-max", type=int, default=4,
                   help="largest ensemble run on the host (default 4)")
    p.add_argument("--reference", type=int, default=2,
                   help="members also run alone with LSODA (default 2)")
    args = p.parse_args()
    case = pb.load_case(args.case)
    devices = ["cpu"] + (["cuda"] if backend.available("cuda") else [])
    print(f"{case.id}, {args.t_end:g} min, devices: {', '.join(devices)}", flush=True)

    all_configs = configs(case, max(args.members))
    refs = []
    for config in all_configs[:args.reference]:
        tr = pb.run(case, config, t_end=args.t_end, dt=args.dt)
        refs.append(tr)
        print(f"  LSODA alone: {tr.wall_time:8.1f} s  nfev {tr.nfev}", flush=True)

    for m in args.members:
        for device in devices:
            if device == "cpu" and m > args.cpu_max:
                continue
            start = time.perf_counter()
            res = pb.run_ensemble(case, all_configs[:m], t_end=args.t_end, dt=args.dt,
                                  device=device)
            wall = time.perf_counter() - start
            ok = [r for r in res if r.error is None]
            line = (f"  M {m:3d} {device:4s}  {wall:8.1f} s  {3600.0 * m / wall:8.1f} runs/h  "
                    f"{wall / m:7.2f} s/run  {len(ok)}/{m} ok  "
                    f"nfev/member {np.mean([r.trajectory.nfev for r in ok]):.0f}")
            diffs = [np.max(np.abs(r.trajectory.x - ref.x) / (ATOL + RTOL * np.abs(ref.x)))
                     for r, ref in zip(res, refs) if r.error is None]
            if diffs:
                line += f"  largest difference from LSODA {max(diffs):.2g} tolerances"
            print(line, flush=True)


if __name__ == "__main__":
    main()
