"""Reactor / column / recycle flowsheet built on the reference CSTR.

The reactor is the non-isothermal CSTR of Romagnoli and Palazoglu (2020).

Fresh feed, containing reactant A and a light inert I, joins the recycle and enters the
CSTR, where A -> B runs first-order and exothermic against a cooling jacket.  The
reactor effluent feeds a distillation column.  The inert and the unreacted reactant
leave overhead; the distillate is split into a recycle returning to the reactor and a
purge; the product B leaves as bottoms.

The flowsheet is designed so that the reactor sits exactly on the reference operating
point: the mixed reactor inlet carries C_A0 = 5 kmol/m3 at F0 = 3.2 m3/min and
T0 = 336.97 K, which reproduces C_A = 2.356 kmol/m3, T = 362.4 K, T_j = 345.8 K, and
with them the open-loop instability of the reference reactor.  Everything
established about the reactor on its own therefore carries over unchanged, and the
question this example asks is what closing a recycle around it does.

State vector, length 6 + column states:

    0        V        reactor liquid volume, m3
    1:4      C        reactor concentrations [I, A, B], kmol/m3
    4        T        reactor temperature, K
    5        T_j      jacket temperature, K
    6:       column   see `column.unpack`

Pressure is not a state.  Constant molar overflow presumes the column pressure loop is
perfect, which removes one degree of freedom from the plantwide problem; the reader
should count nine, not ten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np
from scipy.optimize import fsolve

from plantbench.backend import namespace
from plantbench.units import column as col
from plantbench.units import cstr

from .parameters import PlantParameters

N_REACTOR_STATES = 6
I, A, B = 0, 1, 2

# Signals that are functions of the inputs rather than inputs themselves, recorded by the
# closed-loop simulator alongside them.
DERIVED: dict[str, Callable[["Inputs"], float]] = {"recycle": lambda u: u.D - u.P}


@dataclass
class Inputs:
    """Manipulated variables and disturbances acting on the plant."""

    F_fresh: float  # fresh feed, kmol/min
    z_fresh: np.ndarray  # fresh feed composition [I, A, B]
    Fj: float  # jacket coolant flow, m3/min
    F_out: float  # reactor effluent, m3/min
    V_boil: float  # reboiler boil-up, kmol/min
    L_reflux: float  # reflux, kmol/min
    D: float  # total distillate, kmol/min
    P: float  # purge, kmol/min (P <= D; recycle is D - P)
    Bm: float  # bottoms, kmol/min
    T_fresh: float  # reactor inlet temperature in the base plant, K
    # Used only by the heat-integrated plants of the design comparison.
    Q_feed_trim: float = 0.0  # kJ/min, feed trim heater (negative: preheat bypassed)
    T_storage: float | None = None  # K, fresh feed as delivered; None takes the default


@dataclass
class Design:
    """A converged plantwide steady state."""

    pp: PlantParameters
    x: np.ndarray  # full state vector
    u: Inputs
    rho_molar: float
    C_I0: float  # inert concentration at the reactor inlet, kmol/m3
    C_B0: float  # product concentration at the reactor inlet, kmol/m3
    xD: np.ndarray
    xB: np.ndarray
    residual: float
    extras: dict = field(default_factory=dict)
    derived: ClassVar[dict] = DERIVED

    @property
    def recycle(self) -> float:
        return self.u.D - self.u.P

    # The right-hand side and the measures take a batch of states (`control.supports_batch`);
    # the heat-integrated plants (`integrated.IntegratedDesign`) do not.
    batched: ClassVar[bool] = True

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: PlantParameters) -> np.ndarray:
        """The plant's right-hand side, which the closed-loop simulator integrates: `rhs`
        for one state, `rhs_batch` for a batch."""
        return rhs(t, x, u, pp) if x.ndim == 1 else rhs_batch(t, x, u, pp)

    def summary(self) -> str:
        u = self.u
        prod = self.extras["A_consumed"]
        loss = self.extras["A_lost"]
        lines = [
            f"reactor      C_A {self.x[1 + A]:.4f} kmol/m3   T {self.x[4]:.2f} K   "
            f"T_j {self.x[5]:.2f} K   V {self.x[0]:.2f} m3",
            f"reactor in   F0 {u.F_out:.3f} m3/min   C_A0 {self.extras['CA0']:.4f} "
            f"kmol/m3   T0 {self.extras['T0']:.2f} K",
            f"molar        rho {self.rho_molar:.4f} kmol/m3   column feed "
            f"{self.extras['F_col']:.3f} kmol/min",
            f"flows        fresh {u.F_fresh:.3f}   recycle {self.recycle:.3f}   "
            f"purge {u.P:.4f}   product {u.Bm:.3f} kmol/min",
            f"column       V {u.V_boil:.3f}   L {u.L_reflux:.3f}   D {u.D:.3f}   "
            f"reflux ratio {u.L_reflux / u.D:.3f}",
            f"distillate   I {self.xD[0]:.5f}   A {self.xD[1]:.5f}   B {self.xD[2]:.6f}",
            f"bottoms      I {self.xB[0]:.2e}   A {self.xB[1]:.2e}   B {self.xB[2]:.6f}",
            f"reactant     consumed {prod:.4f}   lost {loss:.4f} kmol/min "
            f"({loss / (prod + loss):.2%} of feed)",
            f"recycle ratio {self.recycle / u.F_fresh:.3f}   "
            # Round-off below 1e-12 is printed as a bound, so a report does not depend on
            # the last bit of the arithmetic.
            "residual " + ("< 1e-12" if self.residual < 1e-12 else f"{self.residual:.1e}"),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------------
# Steady-state design
# --------------------------------------------------------------------------------


# The inert level at the reactor inlet, kmol/m3, of the published design point.  The
# balances leave it free (see `solve_design`), and this is the value the reference results
# were produced at.
C_I0_REFERENCE = 0.5485242267723835
# The column-base holdup, kmol, of the published design point.  No steady-state balance
# involves it (see `solve_design`); this is where the integration that first produced the
# design left it, and the reference results were produced at it.  The column's nominal
# `base_holdup` parameter, 18 kmol, seeds the integration only.
BASE_HOLDUP_REFERENCE = 6.895913559879565


def solve_design(
    pp: PlantParameters,
    V_boil: float,
    z_inert: float = 0.005,
    C_I0: float = C_I0_REFERENCE,
    base_holdup: float = BASE_HOLDUP_REFERENCE,
    guess: tuple[float, float] | None = None,
) -> Design:
    """Design the flowsheet around the reference reactor operating point.

    The reactor inlet is pinned to `C_A0 = 5 kmol/m3` at `F0 = 3.2 m3/min`, so the
    reactor states are the reference's and the conversion, and with it the rate at which
    reactant must be fed and product withdrawn, are fixed before the separation is
    considered at all.

    Two unknowns remain, the reflux and the recycle flow, and two balances close them:

      * the product B entering the reactor arrives only in the recycle
      * the inert fed must equal the inert purged plus the inert lost in the bottoms

    The product balance, product made against product leaving in the purge and the
    bottoms, follows from the first of these and the column's own product balance, so it
    determines nothing and is reported as a check.  The reactant balance closes
    identically for the same reason: the mixer and the column balances force it.

    `V_boil` sets the separation quality, `z_inert` the inert content of the fresh feed,
    and `C_I0` the inert level at the reactor inlet, which is equivalent to choosing the
    purge.  All three are design choices rather than consequences.  Before `C_I0` was a
    specification the product balance stood in its place, and the inert level was then
    left to the solver's path.

    The reflux drum and column base inventories are free in the same way: the column
    fixes the distillate and bottoms flows, so no steady-state balance involves either
    holdup, and a solve leaves them wherever its path ends.  They are set after the solve,
    the drum to the column's nominal `drum_holdup` and the base to `base_holdup`.
    """
    rp = pp.reactor
    cp = pp.column
    rho_m = pp.rho_molar

    # The reactor is the reference's, so solve it once and be done.
    reactor_ss = cstr.steady_state(rp)
    CA_ss = reactor_ss[0]
    conv = rp.CA0 - CA_ss  # kmol/m3 of A converted
    A_consumed = rp.F0 * conv  # kmol/min

    F_col = rp.F0 * rho_m  # column feed, kmol/min
    cache: dict = {}

    def residual(u):
        L_reflux, R = u
        C_B0 = rho_m - rp.CA0 - C_I0
        D = V_boil - L_reflux
        P = D - R
        F_fresh = F_col - R
        Bm = F_col - D

        C_out = np.array([C_I0, CA_ss, C_B0 + conv])
        zF = C_out / rho_m
        z = col.steady_state(
            cp, F_col, zF, 1.0, V_boil, L_reflux,
            guess=cache.get("z"), warm="z" in cache,
        )
        cache["z"] = z
        _, xD, _, _, _, xB = col.unpack(z, cp)
        cache.update(
            xD=xD, xB=xB, D=D, P=P, F_fresh=F_fresh, Bm=Bm, zF=zF,
            C_I0=C_I0, C_B0=C_B0, L_reflux=L_reflux, R=R,
        )
        cache["check"] = A_consumed - (P * xD[B] + Bm * xB[B])  # product made = removed
        return [
            C_B0 - R * xD[B] / rp.F0,  # product at the reactor inlet is recycled only
            F_fresh * z_inert - (P * xD[I] + Bm * xB[I]),  # inert in = inert out
        ]

    if guess is None:
        # Production sets the bottoms flow, the bottoms flow sets the distillate, and
        # the distillate sets the reflux at the chosen boil-up.
        D0 = F_col - A_consumed
        guess = (V_boil - D0, D0 - 0.25)

    sol, _, ier, msg = fsolve(residual, guess, full_output=True, xtol=1e-12)
    res = float(np.max(np.abs([*residual(sol), cache["check"]])))
    if res > 1e-8:
        raise RuntimeError(f"plantwide design did not converge ({res:.1e}): {msg}")
    _, xD_col, M_trays, x_trays, _, xB_col = col.unpack(cache["z"], cp)
    cache["z"] = col.pack(cp.drum_holdup, xD_col, M_trays, x_trays, base_holdup, xB_col)

    xD, xB = cache["xD"], cache["xB"]
    z_fresh = np.array([z_inert, 1.0 - z_inert, 0.0])
    u = Inputs(
        F_fresh=cache["F_fresh"],
        z_fresh=z_fresh,
        Fj=rp.Fj,
        F_out=rp.F0,
        V_boil=V_boil,
        L_reflux=cache["L_reflux"],
        D=cache["D"],
        P=cache["P"],
        Bm=cache["Bm"],
        T_fresh=rp.T0,
    )
    C_in = np.array([cache["C_I0"], rp.CA0, cache["C_B0"]])
    x = np.concatenate(
        [
            [rp.V],
            [cache["C_I0"], CA_ss, cache["C_B0"] + conv],
            [reactor_ss[1], reactor_ss[2]],
            cache["z"],
        ]
    )
    A_lost = cache["P"] * xD[A] + cache["Bm"] * xB[A]
    return Design(
        pp=pp,
        x=x,
        u=u,
        rho_molar=rho_m,
        C_I0=cache["C_I0"],
        C_B0=cache["C_B0"],
        xD=xD,
        xB=xB,
        residual=res,
        extras=dict(
            CA0=rp.CA0,
            T0=rp.T0,
            C_in=C_in,
            F_col=F_col,
            A_consumed=A_consumed,
            A_lost=A_lost,
            zF=cache["zF"],
            reactor_ss=reactor_ss,
        ),
    )


# --------------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------------


def mixer(F_fresh: float, z_fresh: np.ndarray, R: float, xD: np.ndarray, rho_m: float):
    """Combine fresh feed and recycle.  Returns (F_in m3/min, C_in kmol/m3)."""
    n = F_fresh + R  # kmol/min
    if n <= 0:
        return 0.0, np.zeros(3)
    z = (F_fresh * z_fresh + R * xD) / n
    return n / rho_m, z * rho_m


def rhs(t: float, x: np.ndarray, u: Inputs, pp: PlantParameters) -> np.ndarray:
    """Right-hand side of the whole plant."""
    rp, cp, rho_m = pp.reactor, pp.column, pp.rho_molar

    R = u.D - u.P
    _, xD, _, _, _, _ = col.unpack(x[N_REACTOR_STATES:], cp)
    F_in, C_in = mixer(u.F_fresh, u.z_fresh, R, xD, rho_m)

    reactor = cstr.rhs_multicomponent(
        x[:N_REACTOR_STATES], rp, F_in, C_in, u.T_fresh, u.F_out, u.Fj
    )

    # The reactor effluent is the column feed.
    C_react = x[1 : 1 + 3]
    F_col = u.F_out * rho_m
    zF = C_react / rho_m

    column = col.rhs(
        x[N_REACTOR_STATES:], cp, F_col, zF, 1.0, u.V_boil, u.L_reflux, u.D, u.Bm
    )
    return np.concatenate([reactor, column])


def rhs_batch(t: float, X, u: Inputs, pp: PlantParameters):
    """`rhs` at a batch of m states, X (m, n), with every field of `u` carrying the leading
    axis (`control.batch_inputs`): the same flowsheet from the batched reactor and column
    blocks, on NumPy or CuPy.  `rhs` itself is kept as it is, since the frozen results are
    computed with it."""
    xp = namespace(X)
    rp, cp, rho_m = pp.reactor, pp.column, pp.rho_molar
    N = N_REACTOR_STATES

    # The mixer, row by row: fresh feed and recycle, nothing when both are shut.
    R = u.D - u.P
    xD = X[:, N + 1:N + 4]
    n = u.F_fresh + R
    flowing = n > 0
    z = (u.F_fresh[:, None] * u.z_fresh + R[:, None] * xD) / xp.where(flowing, n, 1.0)[:, None]
    F_in = xp.where(flowing, n / rho_m, 0.0)
    C_in = xp.where(flowing[:, None], z * rho_m, 0.0)

    reactor = cstr.rhs_multicomponent_batch(X[:, :N], rp, F_in, C_in, u.T_fresh, u.F_out, u.Fj)
    zF = X[:, 1:1 + 3] / rho_m
    column = col.rhs_batch(X[:, N:], cp, u.F_out * rho_m, zF, 1.0, u.V_boil, u.L_reflux, u.D,
                           u.Bm)
    return xp.concatenate([reactor, column], axis=1)
