"""The closed-loop spectra of the case's control structures, at the design point.

    .venv/bin/python -m studies.ngl_deethanizer.tuning

Each structure is linearized at its design point (`Setup.spectrum`, a central-difference
Jacobian of the plant with its integral states) and summarized as the case records a
tuning: the smallest damping ratio among the oscillatory modes, the rightmost eigenvalue,
and the slowest mode.  The tunings are those of `ngl_deethanizer.definition`, `TUNING`
and `COMPOSITION`.  A second table repeats the summary for the HYSYS tuning converted
unchanged and for other gains of TIC-101 and XIC-100, as fractions of their converted
gains, with the least damped oscillatory mode.  Writes `results/tuning.txt`, from which
the damping ratios in the case card, in `definition.py` and in `tests/test_structures.py`
are taken.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_deethanizer.definition import CASE, COMPOSITION, TUNING
from plantbench.core import control as ctl

HERE = Path(__file__).resolve().parent
STRUCTURES = ("open loop", "basic", "composition cascade")
TIC101, XIC100 = "stage 4 temperature", "ethane in LPG"
# The HYSYS tuning converted (README, Control structures): Kc in %/% times the OP span
# over the PV span, the integral time from seconds to minutes.  TIC-101: 13 %/% of
# 1000 kmol/h over 200 K, 210 s; XIC-100: 20 %/% of 145 K over a mole fraction of 1, 66 s.
CONVERTED = {TIC101: dict(Kc=-13.0 * (1000.0 / 60) / 200.0, tau_I=210.0 / 60),
             XIC100: dict(Kc=-20.0 * 145.0 / 1.0, tau_I=66.0 / 60)}
# (structure, TIC-101 gain, XIC-100 gain): a fraction of the converted gain, "converted"
# for the HYSYS gain and integral time, or None for the case's tuning.
VARIANTS = (
    ("basic", "converted", None),
    ("basic", 0.25, None),
    ("basic", 0.15, None),
    ("basic", 0.10, None),
    ("basic", 0.05, None),
    ("composition cascade", "converted", "converted"),
    ("composition cascade", None, "converted"),
    ("composition cascade", None, 1 / 12),
)


def _tuning(loop: str, gain) -> dict:
    if gain is None:
        return {}
    if gain == "converted":
        return {loop: CONVERTED[loop]}
    return {loop: dict(Kc=gain * CONVERTED[loop]["Kc"])}


def _label(gain) -> str:
    return "case" if gain is None else gain if isinstance(gain, str) else f"{gain:.4f}"


def least_damped(ev: np.ndarray) -> complex:
    """The oscillatory eigenvalue of the smallest damping ratio, as `damping_ratio` finds it."""
    osc = ev[np.abs(ev.imag) > 1e-9]
    return complex(osc[np.argmin(-osc.real / np.abs(osc))])


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

    say("")
    say("OTHER GAINS OF TIC-101 AND XIC-100")
    say(f"case tuning: TIC-101 {TUNING[TIC101]['Kc']} kmol/min per K, "
        f"{TUNING[TIC101]['Kc'] / CONVERTED[TIC101]['Kc']:.4f} of the converted "
        f"{CONVERTED[TIC101]['Kc']:.4f}; XIC-100 {COMPOSITION['Kc']} K, "
        f"{COMPOSITION['Kc'] / CONVERTED[XIC100]['Kc']:.4f} of the converted "
        f"{CONVERTED[XIC100]['Kc']:.1f}")
    say("gains as fractions of the converted gain; least damped mode in rad/min")
    say("")
    say(f"{'structure':22s}{'TIC-101':>10s}{'XIC-100':>10s}{'damping':>9s}"
        f"{'least damped mode':>22s}{'slowest, min':>14s}")
    for structure, tic, xic in VARIANTS:
        tuning = {**_tuning(TIC101, tic), **_tuning(XIC100, xic)}
        ev = pb.build(CASE, CASE.config(structure=structure, tuning=tuning)).spectrum()
        mode = least_damped(ev)
        # `basic` has no XIC-100.
        xic_label = "-" if structure == "basic" else _label(xic)
        say(f"{structure:22s}{_label(tic):>10s}{xic_label:>10s}"
            f"{ctl.damping_ratio(ev):9.3f}{f'{mode.real:.3f} ± {abs(mode.imag):.3f}j':>22s}"
            f"{-1.0 / float(ev.real.max()):14.1f}")
        print(lines[-1], flush=True)

    out = HERE / "results" / "tuning.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
