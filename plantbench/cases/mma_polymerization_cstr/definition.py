"""`mma_polymerization_cstr` as a `Case`: the isothermal MMA polymerization reactor of
Doyle, Ogunnaike and Pearson (1995), with the initiator flow setting the product grade.

The design is the paper's operating point, F_I = 0.016783 m3/h, where the
number-average molecular weight is 25 000.5 kg/kmol.  The grades of `GRADES` are a
choice: three NAMW targets between which a grade-change study moves the set-point.

`ENGINEERING` is the `mismatch` of an engineering model of the reactor: the same balances
without chain transfer to monomer, and the initiator efficiency fitted so that the model
gives the reactor's NAMW at the design flow, as a model tuned on a plant's main grade
would be.
"""

from __future__ import annotations

from dataclasses import replace

from scipy.optimize import brentq

from plantbench.core import control as ctl
from plantbench.core.case import Case, Config, override
from plantbench.core.casekit import (
    cached_design,
    limits_from,
    measured,
    parameter_mismatch,
    state_measurement,
)
from plantbench.core.control import Loop
from plantbench.core.limits import CHOICE

from .model import LAYOUT, Design, MMAParameters, namw, solve_design, steady_state

OPTIONS: dict = {}
# NAMW targets of three product grades, kg/kmol [choice].
GRADES = {"A": 25_000.0, "B": 40_000.0, "C": 18_000.0}
# The initiator pump's range, m3/h [choice].
F_I_RANGE = (0.002, 0.08)

_design = cached_design(solve_design)


def make_design(config: Config) -> Design:
    return _design(override(MMAParameters(), config.params))


def grade_flow(target: float, p: MMAParameters | None = None) -> float:
    """The initiator flow at which the steady NAMW is `target`; NAMW falls with F_I."""
    p = p or MMAParameters()
    return brentq(lambda f: float(namw(steady_state(p, f))) - target, 1e-5, 0.5, xtol=1e-12)


def engineering_mismatch(p: MMAParameters | None = None) -> dict[str, float]:
    """The engineering model as a `mismatch`: no chain transfer to monomer, and the
    initiator efficiency at which it gives the reactor's NAMW at the design flow."""
    p = p or MMAParameters()
    target = float(namw(steady_state(p, p.F_I)))
    f = brentq(lambda f: float(namw(steady_state(replace(p, Z_fm=0.0, f_star=f), p.F_I)))
               - target, 0.01, 10.0, xtol=1e-12)
    return {"Z_fm": 0.0, "f_star": f}


ENGINEERING = engineering_mismatch()


def m_namw(x, u, pp):
    return measured(namw(x))


def m_conversion(x, u, pp):
    return measured(1.0 - x[..., 0] / pp.C_min)


MEASUREMENTS = {
    "NAMW": m_namw,
    "conversion": m_conversion,
    **{name: state_measurement(name, LAYOUT) for name in LAYOUT.names()},
}
# Spans [choice]: NAMW from zero to above the heaviest grade; conversion a fraction.
SPANS = {"NAMW": (0.0, 60_000.0), "conversion": (0.0, 1.0)}

# A fixed tuning across grades whose gain to F_I changes twelvefold: -2.76e6 kg/kmol per
# m3/h at grade B, -6.08e5 at A, -2.31e5 at C.  This one keeps the loop damped at B, with
# a 5 min analyzer as well (0.833, and 0.618 with it), at the cost of a slow loop at C,
# whose slowest mode is 214 min (README, Control structures) [choice].
NAMW_TUNING = {"Kc": -2.0e-7, "tau_I": 10.0}  # m3/h per kg/kmol, min


def namw_loop() -> Loop:
    """NAMW on the initiator flow; raising the flow lowers the NAMW, so the gain is
    negative."""
    return Loop("NAMW", m_namw, "F_I", **NAMW_TUNING, lo=F_I_RANGE[0], hi=F_I_RANGE[1],
                pv_range=SPANS["NAMW"])


STRUCTURES = {
    "open loop": lambda: [],
    "NAMW PI": lambda: [namw_loop()],
}


def make_structure(design: Design, config: Config) -> ctl.Structure:
    return ctl.bias_from_design(STRUCTURES[config.structure](), design)


def limits(design: Design) -> dict:
    return limits_from({name: {"span": (*span, "the measurement's range")}
                        for name, span in SPANS.items()}, CHOICE)


CASE = Case(
    id="mma_polymerization_cstr",
    title="Isothermal MMA polymerization CSTR",
    summary=(
        "Free-radical polymerization of methyl methacrylate in an isothermal CSTR, as "
        "Doyle, Ogunnaike and Pearson (1995) give it: four states, the initiator flow "
        "setting the number-average molecular weight. The process gain changes twelvefold "
        "between grades. Structures open loop and a NAMW PI loop; an engineering model "
        "without chain transfer as a mismatch."
    ),
    options=OPTIONS,
    structures=tuple(STRUCTURES),
    default_structure="NAMW PI",
    make_design=make_design,
    make_structure=make_structure,
    disturbances={},
    measurements=lambda design: dict(MEASUREMENTS),
    state_names=lambda design: LAYOUT.names(),
    apply_mismatch=parameter_mismatch,
    limits=limits,
)
