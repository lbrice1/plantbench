"""Measurements of `reactor_separator_recycle`: functions of the plant state and inputs a loop can control.

Each takes (x, u, pp) and returns a float, which is the signature `core.control.Loop`
expects of its `measure`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from plantbench.core.casekit import measured
from plantbench.units import column as col
from plantbench.units import thermo

from .parameters import PlantParameters
from .plant import N_REACTOR_STATES, Inputs

I, A, B = 0, 1, 2




# Each measure reads the last axis, so that it serves one state and a batch alike
# (`casekit.measured`).  The column's state follows the reactor's: the drum holdup and
# composition, the tray holdups, the tray compositions row by row, the base holdup and
# composition (`units.column.unpack`).


def _base(pp) -> int:
    """Where the base holdup sits in the plant state."""
    nt = pp.column.n_trays
    return N_REACTOR_STATES + 1 + col.N_SPECIES + nt + nt * col.N_SPECIES


def m_reactor_volume(x, u, pp):
    return measured(x[..., 0])


def m_reactor_T(x, u, pp):
    return measured(x[..., 4])


def m_reactor_CA(x, u, pp):
    return measured(x[..., 1 + A])


def m_drum_level(x, u, pp):
    return measured(x[..., N_REACTOR_STATES])


def m_base_level(x, u, pp):
    return measured(x[..., _base(pp)])


def m_bottoms_A(x, u, pp):
    return measured(x[..., _base(pp) + 1 + A])


def m_tray_T(tray: int, P: float) -> Callable[[np.ndarray, Inputs, PlantParameters], float]:
    """Temperature of `tray` (1 = top) at column pressure P: the bubble point of its liquid."""

    def measure(x, u, pp):
        k = N_REACTOR_STATES + 1 + col.N_SPECIES + pp.column.n_trays + (tray - 1) * col.N_SPECIES
        return measured(thermo.bubble_point(x[..., k:k + col.N_SPECIES], P, pp.thermo,
                                            pp.column))

    return measure


def m_recycle(x, u, pp):
    return measured(u.D - u.P)


def inert_inventory(x, pp) -> tuple[float, float]:
    """Moles of inert held in the reactor and in the column.

    The inert has no reaction term and, with the purge shut, no exit, so its total is
    the quantity Step 7 asks to be checked.  Watching its mole fraction instead is
    misleading once a holdup starts to run away, because the fraction can fall while
    the inventory rises.
    """
    reactor = x[0] * x[1 + I]  # m3 * kmol/m3
    MD, xD, M, xs, MB, xB = col.unpack(x[N_REACTOR_STATES:], pp.column)
    column = MD * xD[I] + float((M * xs[:, I]).sum()) + MB * xB[I]
    return reactor, column
