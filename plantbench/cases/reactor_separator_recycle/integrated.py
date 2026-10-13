"""Stage 2 plants: the base plant with a heat-exchanger network inside its loops.

Each design's exchangers are sized at the design operating point (`hen.design`) and
rated continuously as the plant moves (`hen.rate_exchangers`).  Three things connect the
network to the plant:

  * the reactor inlet temperature is no longer fixed.  Fresh feed and recycle mix at a
    temperature set by the reflux drum, the preheat exchangers raise it, and a trim
    heater under temperature control makes up the difference.  The trim can go negative,
    which stands for bypassing the preheat exchangers, so one loop covers both
    directions;
  * heat recovered in the reboiler reduces the steam the tray temperature loop has to
    ask for.  If recovered heat exceeds the demand, boil-up exceeds it until the bypass
    acts;
  * a reactor pump-around to the reboiler removes heat from the reactor alongside the
    jacket, so the column's bottoms temperature enters the reactor energy balance.

Exchanger duties follow their rated values through a first-order lag, which stands for
wall and fluid holdup and also keeps the model free of algebraic loops.  The column feed
is taken to be conditioned to its bubble point by a fast trim exchanger, so the column
sees a saturated liquid feed in every design.

The base-case design D1 is modeled the same way, with a trim heater carrying the whole
preheat duty, so that designs are compared with the same feed-temperature control.  The
§19.1-19.3 results use the base plant, where the reactor inlet temperature is simply held.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from plantbench.core import control as ctl
from plantbench.core.control import Loop

from . import hen, plant, streams
from . import scenarios as sc
from .parameters import PlantParameters
from .plant import Design, Inputs

TAU_EXCHANGER = 2.0  # min
TAU_INLET = 1.0  # min
# Spare trim heater capacity, expressed as the feed temperature rise it can add.  Sized
# in temperature rather than as a share of preheat duty so that every design can absorb
# the same feed-temperature upset: a share would leave D1, whose trim already carries the
# whole preheat, with far less room in kelvin than a design whose exchangers carry it.
TRIM_SPARE_K = 15.0


@dataclass
class IntegratedDesign:
    """A Stage 2 plant at its design steady state, ready for `controllers.simulate`."""

    name: str
    base: Design
    sized: hen.NetworkResult
    pp: PlantParameters
    x: np.ndarray
    u: Inputs
    n_base: int
    feed: list[int]  # exchangers heating the reactor feed
    reboiler: list[int]  # exchangers heating the reboiler
    pump_around: int | None
    trim_range: tuple[float, float]  # kJ/min
    CP_feed: float  # kJ/(min K) at design
    derived: ClassVar[dict] = plant.DERIVED

    @property
    def pressure(self) -> float:
        return self.sized.network.pressure

    # -- state access ---------------------------------------------------------------
    def T_inlet(self, x: np.ndarray) -> float:
        return float(x[self.n_base])

    def duties(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[self.n_base + 1 :], dtype=float)

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: PlantParameters) -> np.ndarray:
        rp = pp.reactor
        nb = self.n_base
        xb, T_in, Q = x[:nb], x[nb], x[nb + 1 :]
        if u.T_storage is not None:
            pp = replace(pp, thermo=replace(pp.thermo, T_storage=u.T_storage))

        Q_pa = Q[self.pump_around] if self.pump_around is not None else 0.0
        op = streams.operating_point("dynamic", xb, u, pp)
        # The reactor must shed its heat through the jacket and the pump-around together.
        op = replace(op, Q_jacket=op.Q_jacket + Q_pa)
        table = streams.stream_table(op, self.pressure, pp)
        rated, _ = hen.rate_exchangers(self.sized, table, pp)
        dQ = (np.array([e.duty for e in rated]) - Q) / TAU_EXCHANGER

        temps = streams.temperatures(op, self.pressure, pp)
        T_mix = temps.T_mix
        CP_feed = (u.F_fresh + u.D - u.P) * pp.cp_molar
        preheat = sum(Q[k] for k in self.feed)
        dT_in = (T_mix + (preheat + u.Q_feed_trim) / CP_feed - T_in) / TAU_INLET

        # Heat recovered into the reboiler, less the sensible heat the column also takes,
        # sets a floor on boil-up: the column cannot refuse it.
        recovered = sum(Q[k] for k in self.reboiler)
        sensible = streams.column_sensible_heat(op, temps, pp)
        u_base = replace(
            u, T_fresh=T_in, V_boil=max(u.V_boil, (recovered - sensible) / pp.column.lambda_vap)
        )
        d = plant.rhs(t, xb, u_base, pp)
        d[4] -= Q_pa / (xb[0] * rp.rho * rp.cp)
        return np.concatenate([d, [dT_in], dQ])


def build(design: Design, network: str | hen.Network) -> IntegratedDesign:
    """The Stage 2 plant for a network ("D1", "D3", "D4", "D5", or a Network) at rest."""
    pp, rp = design.pp, design.pp.reactor
    if isinstance(network, str):
        network = hen.DESIGNS[network]
    name = network.name
    op = streams.design_point(design)
    sized = hen.design(network, op, pp)

    feed = [k for k, e in enumerate(sized.exchangers) if e.match.cold == "reactor feed"]
    reboiler = [k for k, e in enumerate(sized.exchangers) if e.match.cold == "reboiler"]
    pa = [k for k, e in enumerate(sized.exchangers) if e.match.hot == "reactor heat"]
    pump_around = pa[0] if pa else None
    Q = np.array([e.duty for e in sized.exchangers])

    x_base = design.x.copy()
    u = replace(design.u)
    if pump_around is not None:
        # The jacket now carries only what the pump-around leaves.  Re-solve its steady
        # state for the same reactor temperature.
        Q_jacket = op.Q_jacket - Q[pump_around]
        UA = rp.U * rp.A_ht
        Tj = op.T_reactor - Q_jacket / UA
        u.Fj = Q_jacket / (rp.rhoj * rp.cpj * (Tj - rp.Tj0))
        x_base[5] = Tj

    T_mix = streams.temperatures(op, network.pressure, pp).T_mix
    CP_feed = (op.F_fresh + op.recycle) * pp.cp_molar
    full_preheat = CP_feed * (rp.T0 - T_mix)
    recovered = float(sum(Q[k] for k in feed))
    u.Q_feed_trim = full_preheat - recovered
    u.T_fresh = rp.T0
    trim_range = (-recovered, u.Q_feed_trim + TRIM_SPARE_K * CP_feed)

    x = np.concatenate([x_base, [rp.T0], Q])
    return IntegratedDesign(
        name,
        design,
        sized,
        pp,
        x,
        u,
        len(design.x),
        feed,
        reboiler,
        pump_around,
        trim_range,
        CP_feed,
    )


# ------------------------------------------------------------------------------------
# Stage 2 control structures
# ------------------------------------------------------------------------------------


def feed_temperature_loop(integ: IntegratedDesign) -> Loop:
    """Reactor inlet temperature on the trim heater and preheat bypass (split range)."""
    lo, hi = integ.trim_range
    nb = integ.n_base
    return Loop(
        "feed T",
        lambda x, u, pp: float(x[nb]),
        "Q_feed_trim",
        Kc=2.0 * integ.CP_feed,
        tau_I=2.0,
        lo=lo,
        hi=hi,
        pv_range=sc.PV_RANGES["feed T"],
    )


def stage2(
    integ: IntegratedDesign,
    cascade: bool = True,
    T_setpoint_step: tuple[float, float] | None = None,
) -> ctl.Structure:
    """The reference structure at the network's column pressure, with its own loop added.

    With `cascade=False` the composition primary is left in manual, so a set-point step
    on the reactor temperature loop reaches the plant unaltered, as in the dynamic
    validation of the design comparison.
    """
    loops = sc.plantwide_loops(integ.pressure)
    if not cascade:
        loops = [lp for lp in loops if lp.name != "reactor C_A"]
    if T_setpoint_step is not None:
        loops = [
            replace(lp, sp_schedule=(T_setpoint_step,)) if lp.name == "reactor T" else lp
            for lp in loops
        ]
    loops.append(feed_temperature_loop(integ))
    return ctl.bias_from_design(loops, integ)
