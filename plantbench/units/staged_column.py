"""A staged column with any number of components, feeds on any stage, a stage energy
balance and a reboiler, for thermodynamics supplied from outside.

`plantbench.units.column` assumes three components, constant relative volatility, constant
molar overflow, one feed and a total condenser.  This model drops each assumption: the
equilibrium and the enthalpies come from a `Thermo` object (an equation of state, or the
constant-volatility model of `IdealThermo`, which reduces it to the other column), the
vapour flows from a stage energy balance, and liquid, vapour or two-phase feeds enter any
stage.  It has no condenser; the liquid on the top stage comes from the feeds: the reflux
of a condenser the caller models, or a cold feed, as in a demethanizer whose top feed is
the expander outlet.

Assumptions:

  * ideal stages, the vapour leaving a stage in equilibrium with its liquid, at the stage
    temperature, which is the bubble point of the liquid at the stage pressure
  * no vapour holdup; the pressure of stage n is the top pressure plus n pressure drops
  * Francis weir hydraulics, the form of `units.column.weir_flow`
  * the energy balance of a stage is quasi-steady in its specific enthalpy: d(M h)/dt is
    taken as h dM/dt, so that the vapour leaving each stage follows explicitly from the
    stage below; the reboiler likewise, its duty an input

States, holdups in kmol and compositions as mole fractions, stage 0 at the top:

    M[n_stages], x[n_stages, n_c]     stages
    M_B, x_B[n_c]                     reboiler, which is the bottom equilibrium stage

Flows in kmol/min, enthalpies in kJ/kmol (J/mol), duties in kJ/min, pressures in Pa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .column import weir_flow


class Thermo(Protocol):
    def bubble_T(self, P: np.ndarray, x: np.ndarray, T0=None) -> tuple[np.ndarray, np.ndarray]: ...

    def h(self, T: np.ndarray, P: np.ndarray, x: np.ndarray, phase: str) -> np.ndarray: ...


@dataclass(frozen=True)
class ColumnSpec:
    """The geometry and hydraulics of a column.  `weir_coeff` and `holdup_weir` have the
    meaning they have in `units.column.weir_flow`."""

    n_stages: int
    n_components: int
    weir_coeff: float  # kmol/min per kmol**1.5
    holdup_weir: float  # kmol held below the weir crest
    dP_stage: float  # Pa, pressure drop per stage, downward
    # Pa, the pressure of each stage and the reboiler above the top, when not uniform;
    # overrides dP_stage.
    P_offsets: tuple | None = None

    def pressures(self, P_top: float) -> np.ndarray:
        """The stage and reboiler pressures, Pa."""
        if self.P_offsets is not None:
            return P_top + np.asarray(self.P_offsets, dtype=float)
        return P_top + self.dP_stage * np.arange(self.n_stages + 1)


@dataclass(frozen=True)
class Feed:
    """A stream entering a stage: flow kmol/min, composition, enthalpy kJ/kmol."""

    stage: int
    F: float
    z: np.ndarray
    h: float


@dataclass
class Profile:
    """What `evaluate` computes besides the derivatives, for measurements and for the
    streams leaving the column."""

    T: np.ndarray  # stages then reboiler, K
    P: np.ndarray
    y: np.ndarray
    h_L: np.ndarray
    h_V: np.ndarray
    L: np.ndarray  # liquid leaving each stage, kmol/min
    V: np.ndarray  # vapour leaving each stage and the reboiler, kmol/min


def weir_constants(L_weir: float, A_tray: float, h_weir: float,
                   rho: float) -> tuple[float, float]:
    """`weir_coeff` and `holdup_weir` of a tray from its geometry: the Francis weir,
    Q = 1.84 L_w h_ow**1.5 m3/s, with the crest h_ow = (M - M_weir) / (rho A), in the form
    L = weir_coeff (M - holdup_weir)**1.5 kmol/min of `units.column.weir_flow`.

    `L_weir` weir length m, `A_tray` tray area m2, `h_weir` weir height m, `rho` liquid
    molar density kmol/m3."""
    return 60 * 1.84 * L_weir * rho**-0.5 * A_tray**-1.5, rho * A_tray * h_weir


def equilibrium(spec: ColumnSpec, thermo: Thermo, x, x_B, P_top: float, T0=None):
    """Stage and reboiler pressures, temperatures and vapour compositions: the part of
    `evaluate` that depends on the state alone, for a caller that needs the overhead
    vapour before it can compute the feeds."""
    P = spec.pressures(P_top)
    T, y = thermo.bubble_T(P, np.vstack([x, x_B[None]]), T0=T0)
    return P, T, y


def evaluate(spec: ColumnSpec, thermo: Thermo, M, x, M_B, x_B, P_top: float,
             feeds: Sequence[Feed], Q_R: float, B: float, T0=None, eq=None):
    """The derivatives of the column states and its profile.

    `P_top` is the pressure of stage 0, `Q_R` the reboiler duty and `B` the bottoms flow
    drawn from the reboiler.  `T0`, the stage and reboiler temperatures of a previous
    call, starts the bubble-point solve; the result does not depend on it beyond the
    solver's tolerance.

    `eq`, the result of `equilibrium` for this state, saves solving it again.

    Returns (dM, dx, dM_B, dx_B, profile).
    """
    n, nc = spec.n_stages, spec.n_components
    P, T, y = eq if eq is not None else equilibrium(spec, thermo, x, x_B, P_top, T0)
    x_all = np.vstack([x, x_B[None]])
    h_L = thermo.h(T, P, x_all, "liquid")
    h_V = thermo.h(T, P, y, "vapour")

    L = weir_flow(M, spec)  # leaving stage n for stage n + 1, the last into the reboiler
    F = np.zeros(n)
    Fz = np.zeros((n, nc))
    Fh = np.zeros(n)
    for f in feeds:
        F[f.stage] += f.F
        Fz[f.stage] += f.F * np.asarray(f.z)
        Fh[f.stage] += f.F * f.h

    # Vapour, from the reboiler up.  At each stage, with d(M h)/dt = h dM/dt,
    #   V_n (H_n - h_n) = V_{n+1} (H_{n+1} - h_n) + L_{n-1} (h_{n-1} - h_n) + sum F (h_F - h_n)
    V = np.empty(n + 1)
    V[n] = (Q_R + L[n - 1] * (h_L[n - 1] - h_L[n])) / (h_V[n] - h_L[n])
    for k in range(n - 1, -1, -1):
        above = L[k - 1] * (h_L[k - 1] - h_L[k]) if k > 0 else 0.0
        V[k] = (V[k + 1] * (h_V[k + 1] - h_L[k]) + above + Fh[k] - F[k] * h_L[k]) \
            / (h_V[k] - h_L[k])

    L_in = np.concatenate([[0.0], L[:-1]])
    x_in = np.vstack([np.zeros((1, nc)), x[:-1]])
    dM = L_in + V[1:] + F - L - V[:n]
    dMx = (L_in[:, None] * x_in + V[1:, None] * y[1:] + Fz
           - L[:, None] * x - V[:n, None] * y[:n])
    dx = (dMx - x * dM[:, None]) / M[:, None]
    dM_B = L[-1] - V[n] - B
    dx_B = (L[-1] * x[-1] - V[n] * y[n] - B * x_B - x_B * dM_B) / M_B
    return dM, dx, dM_B, dx_B, Profile(T=T, P=P, y=y, h_L=h_L, h_V=h_V, L=L, V=V)


class IdealThermo:
    """Constant relative volatility and constant latent heat: saturated liquid at zero
    enthalpy and saturated vapour at `latent`, independent of temperature.  Under it the
    energy balance gives constant molar overflow, and the column reduces to the model of
    `units.column`.  The temperature returned is a placeholder."""

    def __init__(self, alpha: Sequence[float], latent: float):
        self.alpha = np.asarray(alpha, dtype=float)
        self.latent = float(latent)

    def bubble_T(self, P, x, T0=None):
        num = self.alpha * x
        return np.full(len(P), 300.0), num / num.sum(axis=1, keepdims=True)

    def h(self, T, P, x, phase):
        return np.full(len(T), 0.0 if phase == "liquid" else self.latent)
