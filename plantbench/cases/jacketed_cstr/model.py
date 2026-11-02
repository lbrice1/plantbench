"""The reference CSTR on its own: three states, constant volume, one reaction A -> B.

The model is `units.cstr.rhs_constant_volume`, the constant-volume model of Romagnoli and
Palazoglu (2020), with its four external
quantities made inputs so that they can be manipulated or disturbed:

    F0   feed flow, m3/min            CA0  feed concentration, kmol/m3
    T0   feed temperature, K          Fj   jacket coolant flow, m3/min

State vector: [C_A kmol/m3, T K, T_j K].  The reactor has three steady states at the
reference parameters; the design point is the middle one, which is open-loop unstable,
unless another branch is asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plantbench.core.casekit import StateLayout
from plantbench.units import cstr
from plantbench.units.parameters import ReactorParameters

BRANCHES = ("low", "middle", "high")
LAYOUT = StateLayout([("C_A", ()), ("T", ()), ("Tj", ())])


@dataclass
class Inputs:
    F0: float
    CA0: float
    T0: float
    Fj: float


@dataclass
class Design:
    """The reactor at a steady state, ready for the closed-loop simulator."""

    pp: ReactorParameters
    x: np.ndarray
    u: Inputs
    branch: str

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: ReactorParameters) -> np.ndarray:
        return cstr.rhs_constant_volume(t, x, pp, T0=u.T0, CA0=u.CA0, Fj=u.Fj, F0=u.F0)


def solve_design(p: ReactorParameters, branch: str = "middle") -> Design:
    """The steady state on the named branch of the multiplicity."""
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, not {branch!r}")
    roots = cstr.steady_state_branches(p)
    if len(roots) != 3:
        raise ValueError(f"these parameters give {len(roots)} steady states, not 3; "
                         "the branch cannot be chosen")
    x = cstr.steady_state(p, guess=cstr.branch_start(roots[BRANCHES.index(branch)], p))
    return Design(pp=p, x=x, u=Inputs(F0=p.F0, CA0=p.CA0, T0=p.T0, Fj=p.Fj), branch=branch)
