"""The control structures compared in the example, and the disturbances applied.

The structures differ in the decisions of Step 6 of the plantwide procedure of Luyben et al. (1998) -- whether a flow is fixed in the
recycle loop, and which one -- and in whether the reactor conditions move with the
throughput.  The column is controlled the same way in every structure where the choice
exists: a stripping-section tray temperature on the boil-up, with the reflux ratioed to
the column feed.  The comparison therefore isolates the recycle-loop decisions.

The reference structure is `plantwide`: the reactor effluent flow is fixed and sets
the production rate, the fresh feed holds the reactor level, and a composition
controller moves the reactor temperature set-point.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
from plantbench.core.control import Loop, Ratio, with_instruments
from plantbench.core.instruments import Instrument

from . import measurements as meas
from .plant import Design, Inputs

I, A, B = 0, 1, 2

# One tuning for the reactor temperature loop, shared by every structure.  Chosen for
# damping, not merely stability: the earlier K_c = -2, tau_I = 5 min stabilized the
# reactor but left its temperature mode at a damping ratio of 0.06.  K_c = -1 and
# tau_I = 20 min raise that to 0.16 in the base plant.
REACTOR_T_TUNING = {"Kc": -1.0, "tau_I": 20.0}

# Product quality is held through a tray temperature, not a bottoms analyzer: the design
# bottoms carry 4e-7 mole fraction of A, which no on-line analyzer resolves.  Tray 17
# (1 = top, feed on 15) is chosen by the slope criterion: below the feed, the largest
# tray-to-tray temperature change is between trays 16 and 17, 4.3 K at 0.8 bar (3.7 K at
# 0.4 bar, 3.1 K at 0.15 bar).  Raising the boil-up heats the tray, so the gain is
# positive.  The tray loop barely touches the reactor temperature mode (damping 0.161
# without it, 0.158 with it).
TEMPERATURE_TRAY = 17
TRAY_T_TUNING = {"Kc": 2.0, "tau_I": 10.0}  # kmol/min per K, min

# The measurement spans, which serve only to express the signals in percent of span
# (`control.percent_signals`).  The reference gives no transmitter ranges, so each is a
# [choice]: holdups from empty to twice their design value, the reactor temperature over
# the range the composition primary moves its set-point in, the reactor concentration up
# to that of the inlet, C_A0 = 5 kmol/m3, and the tray and feed temperatures over 50 K
# about their design values of 386 K and 337 K.
PV_RANGES = {
    "reactor T": (340.0, 385.0),  # K
    "reactor C_A": (0.0, 5.0),  # kmol/m3
    "reactor level": (0.0, 45.64),  # m3, twice V_nominal = 22.82
    "drum level": (0.0, 24.0),  # kmol, twice the 12 kmol drum holdup
    "base level": (0.0, 13.8),  # kmol, twice the 6.896 kmol base holdup
    "tray T": (360.0, 410.0),  # K
    "feed T": (310.0, 360.0),  # K
}


def column_loops(P: float) -> ctl.Structure:
    """Tray temperature on the boil-up, and the reflux ratioed to the column feed.

    The ratio is what lets a stripping-tray temperature hold the product: with the
    reflux fixed instead, a throughput increase raises the boil-up but not the reflux,
    the rectifying section loses its separation, and product B is pushed overhead into
    the recycle.
    """
    return [
        Loop("tray T", meas.m_tray_T(TEMPERATURE_TRAY, P), "V_boil", **TRAY_T_TUNING,
             lo=10.0, hi=60.0, pv_range=PV_RANGES["tray T"]),
        # The reflux valve has the same 40 kmol/min limit as in the other structures.
        Ratio("reflux/feed", "L_reflux", "F_out", lo=1.0, hi=40.0),
    ]


def _reactor_T() -> Loop:
    return Loop("reactor T", meas.m_reactor_T, "Fj", **REACTOR_T_TUNING, lo=0.05, hi=6.0,
                pv_range=PV_RANGES["reactor T"])


def _column_levels() -> ctl.Structure:
    # Both levels are held by a stream leaving the vessel, so the gains are negative.
    return [
        Loop("drum level", meas.m_drum_level, "D", Kc=-2.0, tau_I=20.0, lo=0.5, hi=25.0,
             pv_range=PV_RANGES["drum level"]),
        Loop("base level", meas.m_base_level, "Bm", Kc=-2.0, tau_I=20.0, lo=0.5, hi=25.0,
             pv_range=PV_RANGES["base level"]),
    ]


def recycle_free(design: Design) -> ctl.Structure:
    """Fresh feed on flow control, and no flow fixed anywhere in the recycle loop.

    Reactor level is held by the effluent valve.  The recycle is left to absorb whatever
    the rest of the plant does not, and the reactor temperature stays at its set-point
    whatever the throughput.
    """
    P = design.pp.column.P_base
    loops = [
        _reactor_T(),
        # Level held by the outflow: negative gain.
        Loop("reactor level", meas.m_reactor_volume, "F_out", Kc=-1.0, tau_I=30.0,
             lo=0.5, hi=8.0, pv_range=PV_RANGES["reactor level"]),
        *_column_levels(),
        *column_loops(P),
    ]
    return ctl.bias_from_design(loops, design)


def _effluent_fixed_loops(P: float) -> ctl.Structure:
    return [
        _reactor_T(),
        # Level held by an inflow: positive gain.
        Loop("reactor level", meas.m_reactor_volume, "F_fresh", Kc=1.5, tau_I=30.0,
             lo=0.5, hi=25.0, pv_range=PV_RANGES["reactor level"]),
        *_column_levels(),
        *column_loops(P),
    ]


def effluent_fixed(design: Design) -> ctl.Structure:
    """Step 6 applied literally: the reactor effluent flow is fixed, nothing else changes.

    The effluent is also the column feed, so it is a flow in the loop reactor -> column
    -> recycle -> reactor.  Holding it stops flow disturbances traveling round the loop.
    The fresh feed takes over reactor level, and the throughput is set at the effluent
    flow.  At fixed reactor temperature and holdup, though, more flow through the reactor
    still means less conversion per pass, and the recycle grows as fast as before.
    """
    return ctl.bias_from_design(_effluent_fixed_loops(design.pp.column.P_base), design)


def plantwide_loops(P: float) -> ctl.Structure:
    """The reference structure's loops at column pressure P, before biases are set from a design."""
    loops = _effluent_fixed_loops(P)
    loops.insert(1, Loop("reactor C_A", meas.m_reactor_CA, "sp:reactor T", Kc=-6.0,
                         tau_I=120.0, lo=340.0, hi=385.0, pv_range=PV_RANGES["reactor C_A"]))
    return loops


def plantwide(design: Design) -> ctl.Structure:
    """The effluent flow fixed, and the reactor conditions moved with the throughput.

    The reference reactor closes with a concentration controller that adjusts the set-point of
    the reactor temperature controller.  Inside the recycle loop it does what Step 6
    alone cannot: holding the reactor outlet composition holds the conversion per pass,
    so that more throughput is met by a hotter reactor rather than by more recycle.
    """
    return ctl.bias_from_design(plantwide_loops(design.pp.column.P_base), design)


def distillate_fixed(design: Design) -> ctl.Structure:
    """Fix the distillate instead, and watch the reflux drum fill.

    The distillate is the only way unconverted A leaves the column.  With it fixed, any
    increase in the A reaching the column stays inside the column, the drum level
    controller raises the reflux to its limit, and the drum fills without bound.  Not
    every flow in the recycle loop will do for Step 6.
    """
    P = design.pp.column.P_base
    loops = [
        _reactor_T(),
        Loop("reactor level", meas.m_reactor_volume, "F_out", Kc=-1.0, tau_I=30.0,
             lo=0.5, hi=8.0, pv_range=PV_RANGES["reactor level"]),
        Loop("drum level", meas.m_drum_level, "L_reflux", Kc=-2.0, tau_I=20.0,
             lo=1.0, hi=40.0, pv_range=PV_RANGES["drum level"]),
        Loop("base level", meas.m_base_level, "Bm", Kc=-2.0, tau_I=20.0, lo=0.5, hi=25.0,
             pv_range=PV_RANGES["base level"]),
        column_loops(P)[0],  # the reflux is on drum level, so there is no ratio
    ]
    return ctl.bias_from_design(loops, design)


STRUCTURES = {
    "recycle free": recycle_free,
    "effluent fixed": effluent_fixed,
    "effluent fixed + cascade": plantwide,
    "distillate fixed": distillate_fixed,
}

# Which input sets the throughput in each structure.  Where the fresh feed is a
# manipulated variable, the throughput is moved at the effluent flow instead.
THROUGHPUT_HANDLE = {
    "recycle free": "F_fresh",
    "effluent fixed": "F_out",
    "effluent fixed + cascade": "F_out",
    "distillate fixed": "F_fresh",
}


# --------------------------------------------------------------------------------
# Disturbances
# --------------------------------------------------------------------------------


def throughput_step(fraction: float, handle: str, t_step: float = 30.0):
    """Raise the plant throughput by `fraction` through whichever handle sets it."""
    return dist.step(handle, fraction=fraction, t=t_step)


def feed_step(fraction: float, t_step: float = 30.0):
    """A step change in the fresh feed rate."""
    return throughput_step(fraction, "F_fresh", t_step)


def purge_closed(t_step: float = 30.0):
    """Close the purge valve, so the inert has no way out of the plant."""

    def d(t: float, u0: Inputs) -> Inputs:
        return u0 if t < t_step else replace(u0, P=0.0)

    d.times = (float(t_step),)
    return d


def inert_step(fraction: float, t_step: float = 30.0):
    """A step change in the inert content of the fresh feed."""

    def d(t: float, u0: Inputs) -> Inputs:
        if t < t_step:
            return u0
        zI = u0.z_fresh[0] * (1.0 + fraction)
        return replace(u0, z_fresh=np.array([zI, 1.0 - zI, 0.0]))

    d.times = (float(t_step),)
    return d


# ------------------------------------------------------------------------------------
# Non-idealities
# ------------------------------------------------------------------------------------
#
# Structures are data, so a non-ideal plant is the same structure with an instrument or
# an actuator attached to named loops rather than a different structure; the general
# form is `core.control.with_instruments`.  The published results use none of these; the
# helper below returns a new structure and leaves the one it was given untouched.


def analyzer_deadtime(structure: ctl.Structure, minutes: float,
                      order: int = 1) -> ctl.Structure:
    """The composition cascade measured by an analyzer with a cycle time.

    The reactor composition loop is the one place in this flowsheet where the measurement
    would be a chromatograph rather than a thermocouple, and it is also the slowest and
    most consequential loop, so it is the first non-ideality worth studying.
    """
    return with_instruments(
        structure, instruments={"reactor C_A": Instrument(deadtime=minutes, order=order)}
    )
