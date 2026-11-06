"""The isothermal free-radical polymerization of methyl methacrylate (MMA) in a CSTR, as
Doyle, Ogunnaike and Pearson (1995), Section 5.1, give it.

Monomer is fed in toluene and initiator (azobisisobutyronitrile) in its own stream; the
reactor is isothermal, perfectly mixed and of constant volume, with no gel effect, no
polymer in the feed and an initiator flow small beside the monomer flow.  The live
radicals are at a quasi-steady state.  The state is

    C_m   monomer concentration, kmol/m3
    C_I   initiator concentration, kmol/m3
    D_0   zeroth moment of the dead-polymer distribution, kmol/m3
    D_1   first moment, kg/m3

and the product grade is the number-average molecular weight NAMW = D_1 / D_0, kg/kmol,
which the initiator flow F_I sets: more initiator, more and shorter chains.

The constants are the paper's, in its units (`MMAParameters`): flows in m3/h, time in h.
The library's time is the minute, so `rhs` is the paper's Eq. (76) divided by 60, and the
flows stay in m3/h.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from plantbench.backend import namespace
from plantbench.core.casekit import StateLayout

LAYOUT = StateLayout([("C_m", ()), ("C_I", ()), ("D_0", ()), ("D_1", ())])
PER_MINUTE = 1.0 / 60.0  # the paper's rates are per hour


@dataclass(frozen=True)
class MMAParameters:
    """The constants of Tables 1 and 2 of the paper [source]."""

    # Table 1: pre-exponential factors and activation energies, kJ/kmol.
    Z_Tc: float = 3.8223e10  # m3/(kmol h), termination by combination
    E_Tc: float = 2.9442e3
    Z_Td: float = 3.1457e11  # m3/(kmol h), termination by disproportionation
    E_Td: float = 2.9442e3
    Z_I: float = 3.7920e18  # 1/h, initiator decomposition
    E_I: float = 1.2550e5
    Z_P: float = 1.7700e9  # m3/(kmol h), propagation
    E_P: float = 1.8283e4
    Z_fm: float = 1.0067e15  # m3/(kmol h), chain transfer to monomer
    E_fm: float = 7.4478e4
    f_star: float = 0.58  # initiator efficiency
    # Table 2.  F is printed in m3; the balances make it a flow, m3/h.
    F: float = 1.0  # m3/h, monomer and solvent feed
    V: float = 0.1  # m3
    C_Iin: float = 8.0  # kmol/m3, initiator in its feed
    C_min: float = 6.0  # kmol/m3, monomer in its feed
    M_m: float = 100.12  # kg/kmol, monomer
    R: float = 8.314  # kJ/(kmol K)
    T: float = 335.0  # K
    F_I: float = 0.016783  # m3/h, the initiator flow of the operating point


def rate_constants(p: MMAParameters) -> dict[str, float]:
    """The Arrhenius constants at the reactor temperature, per hour."""
    def k(Z, E):
        return Z * np.exp(-E / (p.R * p.T))

    return {"Tc": k(p.Z_Tc, p.E_Tc), "Td": k(p.Z_Td, p.E_Td), "I": k(p.Z_I, p.E_I),
            "P": k(p.Z_P, p.E_P), "fm": k(p.Z_fm, p.E_fm)}


@dataclass
class Inputs:
    F_I: float  # m3/h, initiator flow
    C_Iin: float  # kmol/m3, initiator in its feed


def live_radicals(C_I, p: MMAParameters, k: dict[str, float]):
    """P_0, kmol/m3, at the quasi-steady state of the radicals."""
    xp = namespace(C_I)
    return xp.sqrt(2.0 * p.f_star * xp.maximum(C_I, 0.0) * k["I"] / (k["Td"] + k["Tc"]))


def rhs_per_hour(x, u: Inputs, p: MMAParameters):
    """Eq. (76) of the paper: the derivatives of [C_m, C_I, D_0, D_1] per hour, for one
    state or a batch on a leading axis."""
    xp = namespace(x)
    k = rate_constants(p)
    C_m, C_I, D_0, D_1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    P_0 = live_radicals(C_I, p, k)
    growth = (k["P"] + k["fm"]) * C_m * P_0  # kmol/(m3 h) of monomer into chains
    return xp.stack([
        -growth + p.F * (p.C_min - C_m) / p.V,
        -k["I"] * C_I + (u.F_I * u.C_Iin - p.F * C_I) / p.V,
        (0.5 * k["Tc"] + k["Td"]) * P_0 ** 2 + k["fm"] * C_m * P_0 - p.F * D_0 / p.V,
        p.M_m * growth - p.F * D_1 / p.V,
    ], axis=-1)


def namw(x):
    return x[..., 3] / x[..., 2]


def steady_state(p: MMAParameters, F_I: float, C_Iin: float | None = None) -> np.ndarray:
    """The steady state at initiator flow F_I, in closed form: the initiator balance gives
    C_I, the radicals P_0, the monomer balance C_m, and the moments follow."""
    C_Iin = p.C_Iin if C_Iin is None else C_Iin
    k = rate_constants(p)
    C_I = F_I * C_Iin / (k["I"] * p.V + p.F)
    P_0 = live_radicals(np.asarray(C_I), p, k)
    C_m = p.F * p.C_min / p.V / ((k["P"] + k["fm"]) * P_0 + p.F / p.V)
    D_0 = p.V / p.F * ((0.5 * k["Tc"] + k["Td"]) * P_0 ** 2 + k["fm"] * C_m * P_0)
    D_1 = p.V / p.F * p.M_m * (k["P"] + k["fm"]) * C_m * P_0
    return np.array([C_m, C_I, D_0, D_1], dtype=float)


@dataclass
class Design:
    """The reactor at a steady state, ready for the closed-loop simulator."""

    pp: MMAParameters
    x: np.ndarray
    u: Inputs
    # The right-hand side and the measures take a batch of states (`control.supports_batch`).
    batched: ClassVar[bool] = True

    def rhs(self, t: float, x, u: Inputs, pp: MMAParameters):
        return rhs_per_hour(x, u, pp) * PER_MINUTE


def solve_design(p: MMAParameters) -> Design:
    """The steady state at the parameters' initiator flow and feed."""
    return Design(pp=p, x=steady_state(p, p.F_I), u=Inputs(F_I=p.F_I, C_Iin=p.C_Iin))
