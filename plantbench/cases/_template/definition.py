"""Template case, the definition: what the library needs to know about the model.

`plantbench new-case <id>` copies this template into a package of its own,
which declares `CASE` as an entry point so that plantbench finds it by its id.  A case
can also be made available for a session with `@plantbench.case` or
`plantbench.register(CASE)`, and a case in the library is listed in
`plantbench/cases/__init__.py`.  This template is none of these; the contract tests load
it directly so that it stays a working example.
"""

from __future__ import annotations

from plantbench.core import control as ctl
from plantbench.core.case import Case, Config, override
from plantbench.core.casekit import state_measurement
from plantbench.core.control import Loop

from .model import LAYOUT, Design, TankParameters, solve_design


def make_design(config: Config) -> Design:
    # Apply parameter overrides, then design.  Cache this if the design solve is slow:
    # a sweep over tuning or disturbances then solves it once per worker process.
    return solve_design(override(TankParameters(), config.params))


m_level = state_measurement("h", LAYOUT)


def m_outflow(x, u, pp):
    return float(pp.Cv * u.valve * max(x[0], 0.0) ** 0.5)


# Structures are functions returning a list of loops, before biasing.  Opening the valve
# lowers the level, so the level loop's gain is negative.
STRUCTURES = {
    "open loop": lambda: [],
    "level PI": lambda: [Loop("level", m_level, "valve", Kc=-0.5, tau_I=10.0, lo=0.0, hi=1.0)],
}


def make_structure(design: Design, config: Config) -> ctl.Structure:
    # bias_from_design starts every loop at rest at the design point.
    return ctl.bias_from_design(STRUCTURES[config.structure](), design)


CASE = Case(
    id="template",
    title="Gravity-drained tank (template)",
    summary="The smallest working case: one state, one level loop. Copy it to start a case.",
    options={},
    structures=tuple(STRUCTURES),
    default_structure="level PI",
    make_design=make_design,
    make_structure=make_structure,
    # The generic "step" and "ramp" act on any input; add named disturbances here when a
    # change is not a change of one input.
    disturbances={},
    measurements=lambda design: {"level": m_level, "outflow": m_outflow},
    state_names=lambda design: LAYOUT.names(),
)
