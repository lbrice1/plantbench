"""Template case, the model: a gravity-drained tank.

`plantbench new-case <id>` copies it to start a case.  Everything a case needs is here in its smallest
form: frozen parameters, an inputs dataclass, a design at steady state with an `rhs`
method, and nothing else.  Replace the tank with your plant.

    A dh/dt = F_in - Cv * valve * sqrt(h)

State: [h, m].  Inputs: `F_in` m3/min, `valve` 0-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plantbench.core.casekit import StateLayout

# The state vector by name; `LAYOUT.names()` names every state, and `unpack` and `pack`
# convert between the vector and its blocks once a plant has more than a few states.
LAYOUT = StateLayout([("h", ())])


@dataclass(frozen=True)
class TankParameters:
    """Every constant of the model, with its unit; nothing is read from inside a function."""

    A: float = 2.0  # m2, cross-section
    Cv: float = 1.0  # m3/min per m**0.5, outlet valve fully open
    h_design: float = 1.5  # m, the level the plant is designed to run at
    F_design: float = 1.0  # m3/min, the design throughput


@dataclass
class Inputs:
    F_in: float  # m3/min, inflow
    valve: float  # outlet valve opening, 0-1


@dataclass
class Design:
    """The plant at its design steady state: what the closed-loop simulator needs."""

    pp: TankParameters
    x: np.ndarray
    u: Inputs

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: TankParameters) -> np.ndarray:
        h = max(x[0], 0.0)
        return np.array([(u.F_in - pp.Cv * u.valve * np.sqrt(h)) / pp.A])


def solve_design(pp: TankParameters) -> Design:
    """The valve opening that holds the design level at the design throughput."""
    valve = pp.F_design / (pp.Cv * np.sqrt(pp.h_design))
    if not 0.0 < valve <= 1.0:
        raise ValueError(f"the design needs a valve opening of {valve:.3f}, outside (0, 1]")
    return Design(pp=pp, x=np.array([pp.h_design]), u=Inputs(F_in=pp.F_design, valve=valve))
