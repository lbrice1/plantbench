"""Throughput of ensembles in lockstep against the number of members.

    python benchmarks/ensemble.py [--case ngl_deethanizer] [--members 1 4 16 64]
                                  [--t-end 20] [--cpu-max 4] [--reference 2] [--repeats 1]

Member i steps the feed's ethane by a fraction and at a time of its own and, on the
deethanizer, has its own stage 4 temperature gain, so that the members' step sizes differ
as a dataset's would.  The members share one design, solved once before anything is
timed.  For each number of members M and each device (the GPU when CuPy finds one, the
host up to `--cpu-max` members), a warm-up integration of two output intervals compiles
the kernels and fills the memory pool for that M; the ensemble is then integrated
`--repeats` times and reported as the median wall time with the range over the repeats,
runs per hour and seconds per run at the median, and the mean over the members of the
accepted and rejected steps, Jacobians and rows evaluated (`EnsembleRun`).  The first
`--reference` members are run alone with LSODA's own Jacobian (`pb.run`), for the time of
a serial run and the largest difference in the states, in units of the tolerance,
|dx| / (atol + rtol |x|): a relative difference would be dominated by states that are zero
to within atol, such as the fraction of a component absent from a stage, where LSODA's own
error is of order atol.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import plantbench as pb
from plantbench import backend
from plantbench.core.case import build_on, design_key
from plantbench.core.ensemble import integrate_ensemble

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


def _integrate(design, setups, t_end: float, n_points: int, device: str):
    return integrate_ensemble(design, [s.structure for s in setups],
                              [s.disturbance for s in setups], t_end, n_points, RTOL, ATOL,
                              device=device, starts=[s.x0 for s in setups])


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
    p.add_argument("--repeats", type=int, default=1,
                   help="timed integrations of each ensemble (default 1)")
    args = p.parse_args()
    case = pb.load_case(args.case)
    devices = ["cpu"] + (["cuda"] if backend.available("cuda") else [])
    print(f"{case.id}, {args.t_end:g} min, devices: {', '.join(devices)}, "
          f"{args.repeats} repeats", flush=True)

    all_configs = [case.validate(c) for c in configs(case, max(args.members))]
    assert len({design_key(c) for c in all_configs}) == 1
    design = case.make_design(all_configs[0])
    setups = [build_on(case, design, c) for c in all_configs]
    n_points = int(round(args.t_end / args.dt)) + 1
    n = len(design.x)

    refs = []
    for config in all_configs[:args.reference]:
        tr = pb.run(case, config, t_end=args.t_end, dt=args.dt)
        refs.append(tr)
        print(f"  LSODA alone: {tr.wall_time:8.1f} s  nfev {tr.nfev}", flush=True)

    for m in args.members:
        for device in devices:
            if device == "cpu" and m > args.cpu_max:
                continue
            _integrate(design, setups[:m], 2 * args.dt, 3, device)  # warm-up
            walls = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                r = _integrate(design, setups[:m], args.t_end, n_points, device)
                walls.append(time.perf_counter() - start)
            wall = statistics.median(walls)
            ok = np.array([s == "ok" for s in r.status])
            mean = lambda v: float(np.mean(v[ok])) if ok.any() else float("nan")  # noqa: E731
            line = (f"  M {m:3d} {device:4s}  {wall:8.1f} s ({min(walls):.1f} to "
                    f"{max(walls):.1f})  {3600.0 * m / wall:8.1f} runs/h  {wall / m:7.2f} s/run  "
                    f"{int(ok.sum())}/{m} ok  steps {mean(r.steps):.0f}  "
                    f"rejected {mean(r.rejected):.1f}  jacobians {mean(r.jacobians):.0f}  "
                    f"nfev/member {mean(r.nfev):.0f}")
            diffs = [np.max(np.abs(r.Y[j, :, :n].T - ref.x) / (ATOL + RTOL * np.abs(ref.x)))
                     for j, ref in enumerate(refs[:m]) if ok[j]]
            if diffs:
                line += f"  largest difference from LSODA {max(diffs):.2g} tolerances"
            print(line, flush=True)


if __name__ == "__main__":
    main()
