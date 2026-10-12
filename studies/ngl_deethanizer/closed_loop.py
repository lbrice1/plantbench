"""Indicative closed-loop runs of the composition cascade: the two disturbances of the
case's operator panel, the feed flow (FIC-100) and the feed temperature, and the feed's
ethane.

    .venv/bin/python -m studies.ngl_deethanizer.closed_loop

Each run starts at the design, steps the disturbance at 10 min and runs to 120 min.  For
each controlled variable and the products it records the design value, the value at the
end and the largest deviation, with the run's cost.  Writes `results/closed_loop.txt`,
and the trajectories of the measurements shown to `results/closed_loop.npz` for the
report's figure; the case card reports these runs as indicative, not as reference
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_deethanizer.definition import CASE

HERE = Path(__file__).resolve().parent
RUNS = {
    "feed flow +10 %": {"name": "step", "field": "F_feed", "fraction": 0.10, "t": 10.0},
    "feed temperature +5 K": {"name": "step", "field": "T_feed", "value": 20.0 + 273.15,
                              "t": 10.0},
    "feed ethane +10 %": {"name": "feed ethane", "fraction": 0.10, "t": 10.0},
}
SHOWN = ("ethane in LPG", "stage 28 temperature", "stage 4 temperature", "feed temperature",
         "overhead pressure", "condenser level", "reboiler level", "ethane recovery",
         "propane in distillate", "reflux flow", "reboiler duty")
T_END = 120.0


def main() -> None:
    lines = []
    say = lines.append
    say("INDICATIVE CLOSED-LOOP RUNS, COMPOSITION CASCADE")
    say(datasets.stamp_lines(datasets.stamp())[-1])
    saved = {}
    for label, disturbance in RUNS.items():
        config = CASE.config(structure="composition cascade", disturbance=disturbance)
        tr = pb.run(CASE, config, t_end=T_END, dt=0.5)
        say("")
        say(f"{label}: {tr.wall_time:.0f} s, {tr.nfev} evaluations")
        say(f"  {'':26s}{'design':>12s}{'at 120 min':>12s}{'max |dev|':>12s}")
        for name in SHOWN:
            v = np.asarray(tr.y[name])
            say(f"  {name:26s}{v[0]:12.5g}{v[-1]:12.5g}{np.max(np.abs(v - v[0])):12.4g}")
        print("\n".join(lines[-len(SHOWN) - 2:]), flush=True)
        saved[f"{label}/t"] = tr.t
        for name in SHOWN:
            saved[f"{label}/{name}"] = np.asarray(tr.y[name])
    (HERE / "results" / "closed_loop.txt").write_text("\n".join(lines) + "\n")
    np.savez_compressed(HERE / "results" / "closed_loop.npz", **saved)


if __name__ == "__main__":
    main()
