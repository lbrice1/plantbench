"""The closed-loop spectra of the case's control structures, on every scheme and basis.

    .venv/bin/python -m studies.ngl_demethanizer.tuning

Each structure is linearized at its design point (`Setup.spectrum`, a central-difference
Jacobian of the plant with its integral states) and summarized as the case records a
tuning: the smallest damping ratio among the oscillatory modes, the rightmost eigenvalue,
and the slowest mode.  The tunings are those of `ngl_demethanizer.definition`, `TUNING`,
`RECOVERY_ON_TEMPERATURE` and `RECOVERY_ON_RATIO`.  Writes `results/tuning.txt`, from
which the damping ratios in the case card and in `tests/test_structures.py` are taken.
"""

from __future__ import annotations

import time
from pathlib import Path

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_demethanizer.definition import CASE
from plantbench.core import control as ctl

HERE = Path(__file__).resolve().parent
PLANTS = (("conventional", "chebeir2019"), ("gsp", "chebeir2019"), ("crr", "chebeir2019"),
          ("crr", "hysys"))
STRUCTURES = ("basic", "recovery cascade", "recovery cascade on ratio")


def main() -> None:
    lines = []
    say = lines.append
    say("THE CLOSED-LOOP SPECTRA OF THE CONTROL STRUCTURES")
    say(datasets.stamp_lines(datasets.stamp())[-1])
    say("")
    say(f"{'scheme':13s}{'basis':12s}{'structure':27s}{'damping':>8s}{'rightmost':>12s}"
        f"{'slowest, min':>14s}{'s':>6s}")
    for scheme, basis in PLANTS:
        for structure in STRUCTURES:
            if structure == "recovery cascade on ratio" and scheme == "conventional":
                continue
            start = time.perf_counter()
            setup = pb.build(CASE, CASE.config(structure=structure,
                                               options={"scheme": scheme, "basis": basis}))
            ev = setup.spectrum()
            zeta = ctl.damping_ratio(ev)
            right = float(ev.real.max())
            say(f"{scheme:13s}{basis:12s}{structure:27s}{zeta:8.3f}{right:12.3e}"
                f"{-1.0 / right:14.1f}{time.perf_counter() - start:6.0f}")
            print(lines[-1], flush=True)

    out = HERE / "results" / "tuning.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
