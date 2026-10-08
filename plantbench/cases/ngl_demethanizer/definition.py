"""`ngl_demethanizer` as a `Case`: the demethanizer section of a cryogenic NGL recovery
plant, in three recovery schemes.

The options select the scheme (`conventional`, `gsp`, `crr`) and the basis.  On the
`chebeir2019` basis the design point of each scheme is a steady state at the recovery
Chebeir, Salas and Romagnoli (2019) compare the schemes at, 0.82 of the feed's ethane
(ERIC-100), with 1 % methane in the NGL (XIC-100), the column top at 1010 kPa and the
vessels half full; the TK-100 temperature is then a result, not the -29.3 C the paper
states for the conventional scheme.  On the `hysys` basis it is the steady state of the
HYSYS case "Demethanizer - Tuned", a CRR plant on its own ten-component feed: TK-100 at
-38.71 C, the top at 1010 kPa, 1 % methane in the NGL.

The structures are the paper's: `basic`, its regulatory layer (Fig. 4, section 3.2), and
the recovery cascade on the TK-100 temperature or, under gsp and crr, on the
expander-to-branch ratio.  The tunings are chosen for damping and their damping ratios
are recorded in the case card.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from plantbench.backend import namespace
from plantbench.core import control as ctl
from plantbench.core.case import Case, Config, override
from plantbench.core.casekit import cached_design, measured

from . import design_starts
from .model import SCHEMES, Design, Plant, solve_design
from .parameters import BASES, NGLParameters, parameters

C = 273.15
OPTIONS = {"scheme": "conventional", "basis": "chebeir2019"}
CONTINUATION = (0.25, 0.5, 0.75, 1.0)


@cached_design
def _design(pp: NGLParameters, scheme: str, basis: str) -> Design:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, not {scheme!r}")
    if basis not in BASES:
        raise ValueError(f"basis must be one of {tuple(BASES)}, not {basis!r}")
    default = parameters(basis)
    start = design_starts.start(Plant(pp, scheme, basis))
    if start is None:
        return solve_design(pp, scheme, basis)
    if pp == default:
        return solve_design(pp, scheme, basis, start=start)
    try:
        return solve_design(pp, scheme, basis, start=start)
    except ValueError:
        pass
    # Continuation from the defaults to the overrides, in the numerical parameters.
    changed = {f.name: getattr(pp, f.name) for f in dataclasses.fields(pp)
               if isinstance(getattr(pp, f.name), float)
               and getattr(pp, f.name) != getattr(default, f.name)}
    design = None
    for lam in CONTINUATION:
        step = dataclasses.replace(pp, **{k: getattr(default, k) + lam * (v - getattr(default, k))
                                          for k, v in changed.items()})
        design = solve_design(step, scheme, basis, start=start)
        start = (design.x, design.u)
    return design


def make_design(config: Config) -> Design:
    basis = config.options["basis"]
    return _design(override(parameters(basis), config.params), config.options["scheme"],
                   basis)


def measurements(design: Design) -> dict:
    """The measures, each of one state or of a batch (`casekit.measured`)."""
    plant, pp = design.plant, design.pp
    lay = plant.layout
    i = {name: lay.slices[name] for name in lay.shapes}
    labels = plant.eos.c.names
    C1, C2 = labels.index("methane"), labels.index("ethane")

    def streams(x, u):
        return plant.streams(x, u)

    def state(name):
        k = i[name].start
        return lambda x, u, pp: measured(x[..., k])

    def state_kmol_h(name):
        k = i[name].start
        return lambda x, u, pp: measured(60.0 * x[..., k])

    def level(name, full):
        k = i[name].start
        return lambda x, u, pp: measured(100.0 * x[..., k] / full)

    m = {
        "feed flow": lambda x, u, pp: 60.0 * u.F_feed,  # kmol/h
        "TK-100 temperature": lambda x, u, pp: measured(x[..., i["T_TK100"].start] - C),
        "TK-100 level": level("M_TK100", pp.M_TK100_design / pp.level_TK100_design),  # %
        "reboiler level": level("M_B", 2 * pp.M_reboiler_design),
        "overhead pressure": state("P_top"),  # kPa
        "methane in NGL": lambda x, u, pp: measured(100.0 * x[..., i["x_B"]][..., C1]),  # mol %
        # ERIC-100: the NGL flow transmitter and the analysers on the feed and the NGL
        "ethane recovery": lambda x, u, pp: measured(
            x[..., i["FT_NGL"].start] * x[..., i["x_B"]][..., C2]
            / (u.F_feed * u.z_feed[..., C2])),
        "NGL flow": state_kmol_h("FT_NGL"),
        "NGL temperature": lambda x, u, pp: measured(streams(x, u).column.T[..., -1] - C),
        "expander flow": lambda x, u, pp: 60.0 * streams(x, u).F_expander,  # kmol/h
        "expander power": lambda x, u, pp: streams(x, u).W_expander / 60.0,  # kW
        "E-100 hot outlet temperature":
            lambda x, u, pp: measured(x[..., i["T_E100h"]][..., -1] - C),
        "E-102 hot outlet temperature":
            lambda x, u, pp: measured(x[..., i["T_E102h"]][..., -1] - C),
        "overhead temperature leaving E-100":
            lambda x, u, pp: measured(x[..., i["T_E100c"]][..., 0] - C),
        "booster discharge pressure": lambda x, u, pp: streams(x, u).P_booster_out,  # kPa
        "K-101 discharge temperature": lambda x, u, pp: streams(x, u).T_K101_out - C,
        "residue gas flow": lambda x, u, pp: 60.0 * streams(x, u).F_comp,  # kmol/h
    }
    if plant.scheme == "conventional":
        m["TK-101 temperature"] = lambda x, u, pp: measured(x[..., i["T_TK101"].start] - C)
        m["TK-101 level"] = level("M_TK101", 2 * pp.M_TK101_design)
    if plant.scheme != "conventional":
        m["branch flow"] = lambda x, u, pp: 60.0 * streams(x, u).F_branch  # kmol/h
    if plant.scheme == "crr":
        m["recycle flow"] = lambda x, u, pp: 60.0 * streams(x, u).F_recycle  # kmol/h
        # RFIC-100: the recycle over the residue gas, the specification of the design
        m["recycle ratio"] = lambda x, u, pp: measured(x[..., i["FT_recycle"].start]
                                                       / x[..., i["FT_residue"].start])
        m["E-104 hot outlet temperature"] = \
            lambda x, u, pp: measured(x[..., i["T_E104h"]][..., -1] - C)
        m["E-104 cold outlet temperature"] = \
            lambda x, u, pp: measured(x[..., i["T_E104c"]][..., 0] - C)
    # Under crr, stage 1 is the mixing point above the trays and its temperature that of
    # the overhead.
    skip = pp.n_stages - len(plant.layout.unpack(design.x)["M"])
    if skip:
        m["stage 1 temperature"] = lambda x, u, pp: overhead_temperature(plant, x, u) - C
    for n in range(skip, pp.n_stages):
        m[f"stage {n + 1} temperature"] = (
            lambda x, u, pp, k=n - skip: measured(streams(x, u).column.T[..., k] - C))
    return m


def overhead_temperature(plant: Plant, x, u) -> float:
    """The overhead's temperature, K: under crr, that of the stage-1 mixture."""
    st = plant.streams(x, u)
    if plant.scheme != "crr":
        return measured(st.column.T[..., 0])
    if x.ndim == 1:
        P_top = x[plant.layout.slices["P_top"]] * 1e3  # Pa
        f = plant.eos.flash_PH(P_top, np.array([st.h_overhead]), st.column.y[:1],
                               T0=st.column.T[:1])
        return float(f.T[0])
    P_top = x[:, plant.layout.slices["P_top"].start] * 1e3
    f = plant._eos(namespace(x), batch=True).flash_PH(
        P_top, st.h_overhead, st.column.y[:, 0], T0=st.column.T[:, 0])
    return f.T


# The loops, named by what they control; the tags of Chebeir et al. (2019) Fig. 4 and of
# the HYSYS case in the comments.  Gains in units of the manipulated variable per unit of
# the measurement, negative where raising the manipulated variable lowers the measurement;
# integral times in minutes.  The starting point was the HYSYS case's own tuning, in
# percent of its ranges (README, Control structures), converted to these units.
TUNING = {
    # TIC-100: E-101 duty, kW per K.  [H] 1.5 %/%, PV -50 to 0 C, OP 0 to 5556 kW, 300 s
    "TK-100 temperature": dict(Kc=-167.0, tau_I=5.0, lo=0.0, hi=5556.0),
    # LIC-100: TK-100 liquid valve per % level.  [H] 1.7 %/%, 480 s
    "TK-100 level": dict(Kc=-0.017, tau_I=8.0, lo=0.0, hi=1.0),
    # LIC-101 under conventional (Fig. 4): TK-101 liquid valve per % level
    "TK-101 level": dict(Kc=-0.017, tau_I=8.0, lo=0.0, hi=1.0),
    # LIC-102 under conventional, LIC-101 under gsp and crr: NGL valve per % level.
    # [H] 1.7 %/%, 480 s
    "reboiler level": dict(Kc=-0.017, tau_I=8.0, lo=0.0, hi=1.0),
    # PIC-100: K-101 power, kW per kPa.  [H] 1.5 %/%, PV 500 to 4000 kPa, OP 0 to
    # 11944 kW, 240 s
    "overhead pressure": dict(Kc=-5.12, tau_I=4.0, lo=0.0, hi=11944.0),
    # XIC-100: E-103 duty, kW per mol % methane.  [H] 1.0 %/%, PV 0 to 1, OP 0 to
    # 2722 kW, 120 s
    "methane in NGL": dict(Kc=-27.2, tau_I=2.0, lo=0.0, hi=2722.0),
    # RFIC-100, crr: K-102 power, kW per unit recycle ratio.  [H] 0.5 %/%, PV 0 to 1, OP
    # 0 to 50 kW, 30 s
    "recycle ratio": dict(Kc=25.0, tau_I=0.5, lo=0.0, hi=50.0),
}
# ERIC-100, the recovery cascade's primary: on the TK-100 temperature set-point, K per
# unit recovery, or on the expander-to-branch ratio R, per unit recovery.
RECOVERY_ON_TEMPERATURE = dict(Kc=-5.0, tau_I=3.0, lo=-60.0, hi=-20.0)
# A larger R sends less of the TK-100 vapour to the branch and raises the recovery: at a
# fixed TK-100 temperature, 0.0064 for R 5 % above its design value, gsp and crr alike.
RECOVERY_ON_RATIO = dict(Kc=1.0, tau_I=3.0, lo=0.5, hi=4.0)


def _loop(m: dict, name: str, mv: str, tuning: dict) -> ctl.Loop:
    return ctl.Loop(name, m[name], mv, **tuning)


def basic(design: Design) -> ctl.Structure:
    """The regulatory layer of Chebeir et al. (2019) Fig. 4 and section 3.2, without
    FIC-100 and RIC-100 (README, Control structures)."""
    m = measurements(design)
    scheme = design.plant.scheme
    loops = [_loop(m, "TK-100 temperature", "Q_chiller", TUNING["TK-100 temperature"]),
             _loop(m, "TK-100 level", "v_LCV100", TUNING["TK-100 level"])]
    if scheme == "conventional":
        loops.append(_loop(m, "TK-101 level", "v_LCV101", TUNING["TK-101 level"]))
    loops += [_loop(m, "reboiler level", "v_LCV102", TUNING["reboiler level"]),
              _loop(m, "overhead pressure", "W_recompressor", TUNING["overhead pressure"]),
              _loop(m, "methane in NGL", "Q_reboiler", TUNING["methane in NGL"])]
    if scheme == "crr":
        loops.append(_loop(m, "recycle ratio", "W_K102", TUNING["recycle ratio"]))
    return loops


def recovery_cascade(design: Design) -> ctl.Structure:
    """`basic` with ERIC-100 writing the TK-100 temperature set-point."""
    m = measurements(design)
    return basic(design) + [_loop(m, "ethane recovery", "sp:TK-100 temperature",
                                  RECOVERY_ON_TEMPERATURE)]


def recovery_cascade_on_ratio(design: Design) -> ctl.Structure:
    """`basic` with ERIC-100 writing the expander-to-branch ratio, gsp and crr only."""
    if design.plant.scheme == "conventional":
        raise ValueError("the recovery cascade on ratio needs the expander bypass of the "
                         "gsp and crr schemes; the conventional scheme has none")
    m = measurements(design)
    return basic(design) + [_loop(m, "ethane recovery", "split", RECOVERY_ON_RATIO)]


def feed_ethane(design: Design, config: Config, fraction: float, t: float = 30.0):
    """The feed's ethane mole fraction stepped by `fraction` of itself at time t, the
    other components scaled to keep the composition summing to one (Chebeir et al. 2019,
    section 4.3, +10 %)."""
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
    "recovery cascade": recovery_cascade,
    "recovery cascade on ratio": recovery_cascade_on_ratio,
}


def make_structure(design: Design, config: Config) -> ctl.Structure:
    return ctl.bias_from_design(STRUCTURES[config.structure](design), design)


CASE = Case(
    id="ngl_demethanizer",
    title="Demethanizer section of a cryogenic NGL recovery plant",
    summary=(
        "Feed conditioning, cold separation, turboexpansion, a 30-stage demethanizer with "
        "a reboiler and no condenser, and residue-gas recompression: a modified version of "
        "the plant of Chebeir, Salas and Romagnoli (2019), in the conventional, gas "
        "subcooled (gsp) and cold residue recycle (crr) schemes, on the paper's basis or "
        "on that of an unpublished HYSYS case of the cold residue recycle plant. Peng-Robinson thermodynamics throughout; some 300 states. The "
        "paper's regulatory structure and its two recovery cascades."
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
