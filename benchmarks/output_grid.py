"""Accepted steps of an ensemble against the output grid its steps end on.

    python benchmarks/output_grid.py [--case ngl_deethanizer] [--members 4] [--t-end 20]
                                     [--dt 1.0 0.25 0.1] [--device cpu]

`integrate_ensemble` ends every step of a member on the next output time rather than
interpolating to it.  For the members of `benchmarks/ensemble.py` this integrates the
ensemble once on a grid of the end time alone, where only the members' breakpoints end
steps, then on grids of spacing `--dt` and on the 1200 points `run_ensemble` takes when no
`dt` is given, and reports the mean accepted and rejected steps per member and the extra
accepted steps against the coarse grid.  The counts do not depend on the device beyond
round-off; `--device cuda` measures them where the ensembles are timed.
"""

from __future__ import annotations

import argparse

import numpy as np
from ensemble import _integrate, configs

import plantbench as pb
from plantbench.core.case import build_on


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="ngl_deethanizer")
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--t-end", type=float, default=20.0, help="minutes simulated (default 20)")
    p.add_argument("--dt", nargs="+", type=float, default=[1.0, 0.25, 0.1])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    case = pb.load_case(args.case)
    all_configs = [case.validate(c) for c in configs(case, args.members)]
    design = case.make_design(all_configs[0])
    setups = [build_on(case, design, c) for c in all_configs]
    print(f"{case.id}, {args.members} members, {args.t_end:g} min, {args.device}", flush=True)
    grids = [("end time only", 2)]
    grids += [(f"dt {dt:g}", int(round(args.t_end / dt)) + 1) for dt in args.dt]
    grids += [("1200 points", 1200)]
    base = None
    for label, n_points in grids:
        r = _integrate(design, setups, args.t_end, n_points, args.device)
        assert all(s == "ok" for s in r.status), r.message
        steps = float(np.mean(r.steps))
        base = steps if base is None else base
        print(f"  {label:14s} {n_points:5d} points  steps {steps:7.1f}  rejected "
              f"{np.mean(r.rejected):5.1f}  nfev/member {np.mean(r.nfev):8.0f}  "
              f"extra steps {100.0 * (steps - base) / base:5.1f} %", flush=True)


if __name__ == "__main__":
    main()
