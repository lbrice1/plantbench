"""The HYSYS case "Demethanizer - Tuned": its stream table and column profile, and a
starting state for the design solve taken from them.

`hysys_reference.json` is transcribed from four reports of the case exported from Aspen
HYSYS 12 (workbook, streams, column profiles, heat profiles).  The case
is a CRR plant on its own feed (the `hysys` basis); stream names are HYSYS's.  Compositions
are mole fractions in the order of `components`, flows kmol/h, temperatures C, pressures
kPa.  The column profile gives the liquid composition of nitrogen, carbon dioxide, methane
and ethane only.

Correspondence of tags, HYSYS to the paper: E-101 is E-100, E-102 is E-102, E-103 is
E-104, Separator is TK-100, Turboexpander and Booster are TE-100, Recompressor is K-101,
Cryogenic Compressor is K-102, the column's reboiler is E-103.

The `inspection` section holds a later reading of the same case through the HYSYS COM
interface, with the integrator stopped: its recovery and ratio set-points, NGL flow, duties
and machine powers, and ERIC-100 and RIC-100.  The `hysys` basis designs at the TK-100
temperature of the reports, which recovers 0.8296; the HYSYS case's ERIC-100 holds 0.82.
`config` is the configuration that compares the case with it: `recovery cascade` on the
`hysys` basis, designed at the HYSYS recovery set-point, the TK-100 temperature a result;
with `tuning="hysys"`, ERIC-100 at the HYSYS tuning.  The inspection gives ERIC-100's gain,
integral time and output range, but neither its PV span nor its action, so both are
choices here (`ERIC_PV_SPAN`, `ERIC_SIGN`).
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


# ERIC-100's PV span, the whole of a fraction as for this case's own tuning, and its sign,
# this case's: raising the TK-100 set-point lowers the recovery [choice].  The inspection
# gives neither.
ERIC_PV_SPAN = (0.0, 1.0)
ERIC_SIGN = -1.0


def _hysys_tuning() -> dict[str, dict[str, float]]:
    """ERIC-100's HYSYS tuning in this case's units: the gain in percent of output span per
    percent of PV span times the output span over `ERIC_PV_SPAN`, with `ERIC_SIGN`; the
    integral time from seconds to minutes; the output range, the TIC-100 set-point's."""
    c = inspection()["ERIC-100"]
    lo, hi = c["OP_min_C"], c["OP_max_C"]
    Kc = ERIC_SIGN * c["Kc_percent"] * (hi - lo) / (ERIC_PV_SPAN[1] - ERIC_PV_SPAN[0])
    return {"ethane recovery": {"Kc": Kc, "tau_I": c["Ti_s"] / 60.0, "lo": lo, "hi": hi}}


HYSYS_TUNING = _hysys_tuning()
TUNINGS = ("plantbench", "hysys")


def config(tuning: str = "plantbench") -> Config:
    """The configuration of the comparison with the HYSYS case: the crr scheme on the
    `hysys` basis under `recovery cascade`, designed at ERIC-100's set-point, with this
    case's tuning or, with `tuning="hysys"`, ERIC-100's HYSYS tuning.  The design is the
    same under either."""
    from .definition import CASE

    if tuning not in TUNINGS:
        raise ValueError(f"tuning must be one of {TUNINGS}, not {tuning!r}")
    overrides = {k: dict(v) for k, v in HYSYS_TUNING.items()} if tuning == "hysys" else {}
    return CASE.config(
        structure="recovery cascade", options={"scheme": "crr", "basis": "hysys"},
        params={"separator_spec": "recovery",
                "recovery_design": inspection()["ERIC-100"]["setpoint"]},
        tuning=overrides)


def hysys_ratio(measurements: dict) -> float:
    """RIC-100's measurement, F18 / F17: the branch, the bypass vapor and the branch
    liquid, over the expander flow.  The input `split` is the expander over the bypass
    vapor alone, so its reciprocal is not this ratio."""
    return measurements["branch flow"] / measurements["expander flow"]


def initial_state(plant) -> np.ndarray:
    """A state near the HYSYS steady state, for `model.settle` on the hysys basis and the
    crr scheme: exchanger cells between the reported inlet and outlet temperatures, the
    stage holdups from the reported liquid flows, and the stage liquids from the reported
    light components, the remainder in the proportions of the NGL's propane and heavier."""
    from .model import transmitters_at_zero

    if plant.basis != "hysys" or plant.scheme != "crr":
        raise ValueError("the HYSYS case is the crr scheme on the hysys basis")
    ref, pp, n = load(), plant.pp, plant.pp.n_cells

    def T(name):
        return stream(name)["T_C"] + C

    def cells(a, b):
        return np.linspace(T(a), T(b), n)

    ngl = np.array(stream("NGL")["z"])
    heavy = np.where(np.arange(len(ngl)) >= 4, ngl, 0.0)
    heavy /= heavy.sum()
    # The trays are stages 2 to 30; stage 1 is the mixing point (model.py).
    light = np.array(ref["column"]["x_N2_CO2_C1_C2"])[1:]
    x = np.zeros((len(light), len(ngl)))
    x[:, :4] = light
    x += (1.0 - light.sum(axis=1))[:, None] * heavy
    L = np.array(ref["column"]["L_kmol_h"])[1:] / 60.0
    M = pp.holdup_weir + (L / pp.weir_coeff) ** (2 / 3)
    return plant.layout.pack(
        # Hot sides from cell 0 at the inlet, cold sides from cell n - 1 at the inlet.
        T_E100h=cells("2", "4"), T_E100c=cells("5", "3"),
        T_E102h=cells("18", "20"), T_E102c=cells("3", "27"),
        T_E104h=cells("23", "24"), T_E104c=np.full(n, T("21")),
        T_TK100=T("8"), M_TK100=pp.M_TK100_design, x_TK100=np.array(stream("8")["z"]),
        M=M, x=x, M_B=pp.M_reboiler_design, x_B=ngl, P_top=pp.P_top,
        **transmitters_at_zero(plant))
