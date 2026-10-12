"""The HYSYS case "Deethanizer - Tuned": its stream table, column profile and controllers,
and a starting state for the design solve taken from them.

`hysys_reference.json` is transcribed from four reports of the case exported from Aspen
HYSYS 12 (workbook, streams, column profiles, heat profiles).  Stream
names are HYSYS's: "Feed" through VLV-100 to "1", E-100 to "2", the column's distillate "3"
through VLV-101 to "Ethane Production", its bottoms "4" through VLV-102 to "LPG".
Compositions are mole fractions in the order of `components`, flows kmol/h, temperatures C,
pressures kPa.  The column profile gives the liquid composition of nitrogen, methane, ethane
and propane only.  The streams inside the column carry no composition in the report.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = 273.15


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads((HERE / "hysys_reference.json").read_text())


def stream(name: str) -> dict:
    return load()["streams"][name]


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
