"""The closed-loop spectra of the case's control structures, at the design point.

    .venv/bin/python -m studies.ngl_deethanizer.tuning

Each structure is linearized at its design point (`Setup.spectrum`, a central-difference
Jacobian of the plant with its integral states) and summarized as the case records a
tuning: the smallest damping ratio among the oscillatory modes, the rightmost eigenvalue,
and the slowest mode.  The tunings are those of `ngl_deethanizer.definition`, `TUNING`
and `COMPOSITION`.  Writes `results/tuning.txt`, from
which the damping ratios in the case card and in `tests/test_structures.py` are taken.
"""

from __future__ import annotations

import time
from pathlib import Path

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_deethanizer.definition import CASE
from plantbench.core import control as ctl

HERE = Path(__file__).resolve().parent
STRUCTURES = ("open loop", "basic", "composition cascade")


def main() -> None:
    lines = []
    say = lines.append
    say("THE CLOSED-LOOP SPECTRA OF THE CONTROL STRUCTURES")
    say(datasets.stamp_lines(datasets.stamp())[-1])
    say("")
    say(f"{'structure':22s}{'damping':>8s}{'rightmost':>12s}{'slowest, min':>14s}{'s':>6s}")
    for structure in STRUCTURES:
        start = time.perf_counter()
        setup = pb.build(CASE, CASE.config(structure=structure))
        ev = setup.spectrum()
        zeta = ctl.damping_ratio(ev)
        right = float(ev.real.max())
        # The open loop has the integrating modes of the levels and the pressure.
        slowest = -1.0 / right if right < -1e-9 else float("inf")
        say(f"{structure:22s}{zeta:8.3f}{right:12.3e}{slowest:14.1f}"
            f"{time.perf_counter() - start:6.0f}")
        print(lines[-1], flush=True)

    out = HERE / "results" / "tuning.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
