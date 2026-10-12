"""A two-tank toy plant written with `x[..., k]`, so that one function serves one state
and a batch, and a structure on it with a cascade, a ratio station, a lagged instrument
with a second-order Pade deadtime and a rate-limited actuator, so that every branch of
`control.apply` is evaluated on a batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from plantbench.core import control as ctl
from plantbench.core.casekit import measured
from plantbench.core.control import Loop, Ratio
from plantbench.core.instruments import Actuator, Instrument


@dataclass
class Inputs:
    q1: float = 1.0
    q2: float = 0.5
    d: float = 0.0
    z: np.ndarray = None


@dataclass
class Params:
    a1: float = 0.8
    a2: float = 0.4


@dataclass
class Tanks:
    x: np.ndarray
    u: Inputs
    pp: Params
    batched: ClassVar[bool] = True

    def rhs(self, t, x, u, pp):
        h1, h2 = x[..., 0], x[..., 1]
        d1 = u.q1 + u.d + 0.1 * u.z[..., 0] - pp.a1 * np.sqrt(np.abs(h1))
        d2 = pp.a1 * np.sqrt(np.abs(h1)) + u.q2 - pp.a2 * np.sqrt(np.abs(h2))
        return np.stack([d1, d2], axis=-1)


def tanks_design() -> Tanks:
    u = Inputs(z=np.array([0.2, 0.8]))
    pp = Params()
    h1 = ((u.q1 + 0.1 * u.z[0]) / pp.a1) ** 2
    h2 = ((pp.a1 * np.sqrt(h1) + u.q2) / pp.a2) ** 2
    return Tanks(x=np.array([h1, h2]), u=u, pp=pp)


def tanks_structure(design):
    level2 = lambda x, u, pp: measured(x[..., 1])  # noqa: E731
    level1 = lambda x, u, pp: measured(x[..., 0])  # noqa: E731
    loops = [
        Loop("level 2", level2, "sp:level 1", Kc=-0.3, tau_I=20.0, lo=0.0, hi=50.0,
             instrument=Instrument(tau=0.5, deadtime=1.0, order=2)),
        Loop("level 1", level1, "q1", Kc=-0.4, tau_I=8.0, lo=0.0, hi=5.0,
             actuator=Actuator(tau=0.3, rate_limit=0.5)),
        Ratio("q2 ratio", "q2", "q1"),
    ]
    return ctl.bias_from_design(loops, design)
