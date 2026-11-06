"""The HYSYS case "Deethanizer - Tuned": its stream table, column profile and controllers,
and a starting state for the design solve taken from them.

`hysys_reference.json` is transcribed from four reports of the case exported from Aspen
HYSYS 12 (workbook, streams, column profiles, heat profiles).  Stream
names are HYSYS's: "Feed" through VLV-100 to "1", E-100 to "2", the column's distillate "3"
through VLV-101 to "Ethane Production", its bottoms "4" through VLV-102 to "LPG".
Compositions are mole fractions in the order of `components`, flows kmol/h, temperatures C,
pressures kPa.  The column profile gives the liquid composition of nitrogen, methane, ethane
and propane only.  The streams inside the column carry no composition in the report.

The `inspection` section holds a later reading of the same case through the HYSYS COM
interface, with the integrator stopped: the products, internal flows, duties, pressures and
vessel holdups of its saved state, to more figures than the reports give.  `config` is the
configuration that compares the case with it: the design at the HYSYS set-points under
`composition cascade`, the structure the HYSYS case runs, and with `tuning="hysys"` the
HYSYS tuning of the two loops this case detunes (README, Control structures).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from plantbench.core.case import Config

HERE = Path(__file__).resolve().parent
C = 273.15


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads((HERE / "hysys_reference.json").read_text())


def stream(name: str) -> dict:
    return load()["streams"][name]


def inspection() -> dict:
    return load()["inspection"]


# The loops this case detunes, by the HYSYS tag whose tuning they would take.
DETUNED = {"stage 4 temperature": "TIC-101", "ethane in LPG": "XIC-100"}


def _hysys_tuning() -> dict[str, dict[str, float]]:
    """The HYSYS tuning of the detuned loops in this case's units: the gain in percent of
    output span per percent of PV span times the output span over the PV span, negative
    for a "Direct" controller; the integral time from seconds to minutes.  The output spans
    are the loops' lo and hi, which are the HYSYS ranges (definition.py, TUNING)."""
    from .definition import COMPOSITION, TUNING

    loops = {**TUNING, "ethane in LPG": COMPOSITION}
    out = {}
    for name, tag in DETUNED.items():
        c, loop = load()["controllers"][tag], loops[name]
        sign = -1.0 if c["action"] == "Direct" else 1.0
        Kc = sign * c["Kc"] * (loop["hi"] - loop["lo"]) / (c["PV_max"] - c["PV_min"])
        out[name] = {"Kc": Kc, "tau_I": c["Ti_s"] / 60.0}
    return out


HYSYS_TUNING = _hysys_tuning()
TUNINGS = ("plantbench", "hysys")


def config(tuning: str = "plantbench") -> Config:
    """The configuration of the comparison with the HYSYS case: `composition cascade` at
    the design of the HYSYS set-points, with this case's tuning or, with `tuning="hysys"`,
    the HYSYS tuning of TIC-101 and XIC-100.  The design is the same under either."""
    from .definition import CASE

    if tuning not in TUNINGS:
        raise ValueError(f"tuning must be one of {TUNINGS}, not {tuning!r}")
    overrides = {k: dict(v) for k, v in HYSYS_TUNING.items()} if tuning == "hysys" else {}
    return CASE.config(structure="composition cascade", tuning=overrides)


def initial_state(plant) -> np.ndarray:
    """A state near the HYSYS steady state: the stage holdups from the reported liquid
    flows, the stage liquids from the reported light components with the remainder in the
    proportions of the LPG's butanes and heavier, and the vessels at their design levels."""
    ref, pp = load(), plant.pp
    lpg = np.array(stream("4")["z"])
    heavy = np.where(np.arange(len(lpg)) >= 4, lpg, 0.0)
    heavy /= heavy.sum()
    light = np.array(ref["column"]["x_N2_C1_C2_C3"])
    x = np.zeros((len(light), len(lpg)))
    x[:, :4] = light
    x += np.clip(1.0 - light.sum(axis=1), 0.0, None)[:, None] * heavy
    x /= x.sum(axis=1, keepdims=True)
    L = np.array(ref["column"]["L_kmol_h"]) / 60.0
    M = pp.holdup_weir + (L / pp.weir_coeff) ** (2 / 3)
    return plant.layout.pack(
        T_E100=stream("2")["T_C"] + C,
        M_D=pp.level_drum_design * pp.M_drum_full, x_D=np.array(stream("3")["z"]),
        M=M, x=x, M_B=pp.level_reboiler_design * pp.M_reboiler_full, x_B=lpg,
        P_top=pp.P_top)
