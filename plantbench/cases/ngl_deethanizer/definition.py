"""`ngl_deethanizer` as a `Case`: the deethanizer of an NGL fractionation train, separating
ethane from the propane and heavier, constructed from an unpublished HYSYS case.

The design point is the steady state at the HYSYS case's set-points: the feed heated to
33 C, the top at 2453 kPa, stage 4 at 9.536 C (TIC-101), 2 % ethane in the LPG
(XIC-100) and both vessels at 45 % level.  The reflux, the duties and the stage 28
temperature, TIC-100's set-point, are results.

The structures are those of the HYSYS case: `basic`, its regulatory layer with the
stage 28 temperature at a fixed set-point, and `composition cascade`, with XIC-100
writing that set-point, the configuration HYSYS runs.  The tunings are chosen for damping
and their damping ratios are recorded in the case card.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from plantbench.core import control as ctl
from plantbench.core.case import Case, Config, override
from plantbench.core.casekit import cached_design, measured

from .model import Design, ethane_recovery, solve_design
from .parameters import DeethanizerParameters, parameters

C = 273.15
OPTIONS: dict = {}
CONTINUATION = (0.25, 0.5, 0.75, 1.0)


@cached_design
def _design(pp: DeethanizerParameters) -> Design:
    default = parameters()
    if pp == default:
        return solve_design(pp)
    try:
        return solve_design(pp)
    except ValueError:
        pass
    # Continuation from the defaults to the overrides, in the numerical parameters.
    changed = {f.name: getattr(pp, f.name) for f in dataclasses.fields(pp)
               if isinstance(getattr(pp, f.name), float)
               and getattr(pp, f.name) != getattr(default, f.name)}
    design = solve_design(default)
    for lam in CONTINUATION:
        step = dataclasses.replace(pp, **{k: getattr(default, k) + lam * (v - getattr(default, k))
                                          for k, v in changed.items()})
        design = solve_design(step, start=(design.x, design.u))
    return design


def make_design(config: Config) -> Design:
    return _design(override(parameters(), config.params))


def measurements(design: Design) -> dict:
    """The measures, each of one state or of a batch (`casekit.measured`)."""
    plant, pp = design.plant, design.pp
    lay = plant.layout
    i = {name: lay.slices[name] for name in lay.shapes}
    names = plant.eos.c.names
    C2, C3 = names.index("ethane"), names.index("propane")

    def streams(x, u):
        return plant.streams(x, u)

    def level(name, full):
        k = i[name].start
        return lambda x, u, pp: measured(100.0 * x[..., k] / full)

    def x_D(x):
        return x[..., i["x_D"]]

    def x_B(x):
        return x[..., i["x_B"]]

    m = {
        # The controller PVs, in the units of the HYSYS case.
        "feed flow": lambda x, u, pp: 60.0 * u.F_feed,  # kmol/h, FIC-100
        "feed temperature": lambda x, u, pp: measured(x[..., i["T_E100"].start] - C),  # TIC-102
        "overhead pressure": lambda x, u, pp: measured(x[..., i["P_top"].start]),  # kPa, PIC-100
        "condenser level": level("M_D", pp.M_drum_full),  # %, LIC-100
        "reboiler level": level("M_B", pp.M_reboiler_full),  # %, LIC-101
        "reflux flow": lambda x, u, pp: 60.0 * u.L_reflux,  # kmol/h, FIC-101
        "ethane in LPG": lambda x, u, pp: measured(x_B(x)[..., C2]),  # mole fraction, XIC-100
        # The products.
        "distillate flow": lambda x, u, pp: 60.0 * streams(x, u).D,  # kmol/h
        "LPG flow": lambda x, u, pp: 60.0 * streams(x, u).B,  # kmol/h
        "ethane in distillate": lambda x, u, pp: measured(x_D(x)[..., C2]),
        "propane in distillate": lambda x, u, pp: measured(x_D(x)[..., C3]),
        "ethane recovery": lambda x, u, pp: ethane_recovery(plant, u, streams(x, u), x_D(x)),
        "propane recovery": lambda x, u, pp: measured(
            streams(x, u).B * x_B(x)[..., C3] / (u.F_feed * u.z_feed[..., C3])),
        "condenser temperature": lambda x, u, pp: streams(x, u).T_drum - C,
        "reboiler temperature": lambda x, u, pp: measured(streams(x, u).column.T[..., -1] - C),
        "condenser duty": lambda x, u, pp: u.Q_condenser,  # kW
        "reboiler duty": lambda x, u, pp: u.Q_reboiler,  # kW
        "E-100 duty": lambda x, u, pp: u.Q_E100,  # kW
    }
    # TIC-101 reads stage 4 and TIC-100 stage 28.
    for n in range(pp.n_stages):
        m[f"stage {n + 1} temperature"] = (
            lambda x, u, pp, k=n: measured(streams(x, u).column.T[..., k] - C))
    return m


# The loops, named by what they control; the tags of the HYSYS case in the comments.
# Gains in units of the manipulated variable per unit of the measurement, negative where
# raising the manipulated variable lowers the measurement; integral times in minutes.  The
# starting point was the HYSYS case's own tuning, in percent of its ranges, converted to
# these units: a "Direct" HYSYS controller has a negative gain here, a "Reverse" one a
# positive gain.
TUNING = {
    # TIC-102: E-100 duty, kW per K.  [H] 6.33 %/%, PV 0 to 80 C, OP 0 to 833 kW, 101 s
    "feed temperature": dict(Kc=66.0, tau_I=1.68, lo=0.0, hi=833.3),
    # PIC-100: condenser duty, kW per kPa.  [H] 17.9 %/%, PV 0 to 3500 kPa, OP 0 to
    # 2500 kW, 304 s
    "overhead pressure": dict(Kc=-12.8, tau_I=5.06, lo=0.0, hi=2500.0),
    # LIC-100: VLV-101 opening per % level.  [H] 5 %/%, 42 s
    "condenser level": dict(Kc=-0.05, tau_I=0.7, lo=0.0, hi=1.0),
    # LIC-101: VLV-102 opening per % level.  [H] 3 %/%, 42 s
    "reboiler level": dict(Kc=-0.03, tau_I=0.7, lo=0.0, hi=1.0),
    # TIC-101, cascaded on FIC-101, which is ideal here: reflux, kmol/min per K.  [H]
    # 13 %/%, PV 0 to 200 C, FIC-101 set-point 0 to 1000 kmol/h, 210 s: -1.08 kmol/min
    # per K, which damps the closed loop at 0.235; a quarter of it, 0.445.
    "stage 4 temperature": dict(Kc=-0.25, tau_I=3.5, lo=0.0, hi=1000.0 / 60),
    # TIC-100: reboiler duty, kW per K.  [H] 10.5 %/%, PV -5 to 140 C, OP 0 to 5556 kW, 88 s
    "stage 28 temperature": dict(Kc=402.0, tau_I=1.47, lo=0.0, hi=5555.6),
}
# XIC-100, the composition cascade's primary: the TIC-100 set-point, K per unit mole
# fraction.  [H] 20 %/%, PV 0 to 1, OP the TIC-100 range, -5 to 140 C, 66 s: -2900 K
# and 1.1 min, which with TIC-101 retuned damps the cascade at 0.398.  Its proportional
# action passes through TIC-100's to the reboiler duty, and a twelfth of the gain over
# five minutes gives 0.427.
COMPOSITION = dict(Kc=-250.0, tau_I=5.0, lo=-5.0, hi=140.0)


def _loop(m: dict, name: str, mv: str) -> ctl.Loop:
    return ctl.Loop(name, m[name], mv, **TUNING[name])


def basic(design: Design) -> ctl.Structure:
    """The regulatory layer of the HYSYS case, the stage 28 temperature at a fixed
    set-point; without FIC-100 and FIC-101, whose flows are inputs (README, Control
    structures)."""
    m = measurements(design)
    return [_loop(m, "feed temperature", "Q_E100"),
            _loop(m, "overhead pressure", "Q_condenser"),
            _loop(m, "condenser level", "v_VLV101"),
            _loop(m, "reboiler level", "v_VLV102"),
            _loop(m, "stage 4 temperature", "L_reflux"),
            _loop(m, "stage 28 temperature", "Q_reboiler")]


def composition_cascade(design: Design) -> ctl.Structure:
    """`basic` with XIC-100 writing the stage 28 temperature set-point."""
    m = measurements(design)
    return basic(design) + [ctl.Loop("ethane in LPG", m["ethane in LPG"],
                                     "sp:stage 28 temperature", **COMPOSITION)]


def feed_ethane(design: Design, config: Config, fraction: float, t: float = 30.0):
    """The feed's ethane mole fraction stepped by `fraction` of itself at time t, the
    other components scaled to keep the composition summing to one."""
    C2 = design.plant.eos.c.names.index("ethane")

    def d(time: float, u0):
        if time < t:
            return u0
        z = np.array(u0.z_feed, dtype=float)
        new = z[C2] * (1.0 + fraction)
        if not 0.0 < new < 1.0:
            raise ValueError(f"the stepped ethane fraction {new:.4g} is outside (0, 1)")
        z *= (1.0 - new) / (1.0 - z[C2])
        z[C2] = new
        return dataclasses.replace(u0, z_feed=z)

    d.times = (float(t),)
    return d


DISTURBANCES = {"feed ethane": feed_ethane}


STRUCTURES = {
    "open loop": lambda design: [],
    "basic": basic,
    "composition cascade": composition_cascade,
}


def make_structure(design: Design, config: Config) -> ctl.Structure:
    return ctl.bias_from_design(STRUCTURES[config.structure](design), design)


CASE = Case(
    id="ngl_deethanizer",
    title="Deethanizer of an NGL fractionation train",
    summary=(
        "A feed heater and a 30-stage deethanizer with a total condenser and a reboiler, "
        "separating an ethane product from an LPG of propane and heavier: constructed from "
        "an unpublished HYSYS case, on its ten-component feed. Peng-Robinson thermodynamics "
        "throughout; some 350 states. The HYSYS case's regulatory structure, with the "
        "stage 28 temperature at a fixed set-point or cascaded from the ethane in the LPG."
    ),
    options=OPTIONS,
    structures=tuple(STRUCTURES),
    default_structure="basic",
    make_design=make_design,
    make_structure=make_structure,
    disturbances=DISTURBANCES,
    measurements=measurements,
    state_names=lambda design: design.plant.state_names(),
)
