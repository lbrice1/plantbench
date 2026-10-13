"""PI and PID control loops, the closed-loop simulator and its linearization, for any case.

Control structures are declared as data.  A `Structure` is a list of `Loop`s, each
naming a measurement, a manipulated variable and a tuning, so that comparing two
structures means comparing two lists rather than rewiring the model.

Integral action is carried as extra states appended to the plant state vector, which
keeps the whole closed loop a single ODE system that a stiff integrator can handle and
makes the closed-loop Jacobian available for the stability arguments.

A loop may also carry an `Instrument` on its measurement and an `Actuator` on its output,
and those have states of their own, as has the filter of a loop with derivative action.
The augmented vector is laid out with every integral first, in structure order, and any
instrument, actuator and filter states after them.  An ideal PI element contributes none,
so a structure of ideal PI elements has exactly one augmented state per element and the
layout is the one this model has always had --- which is what lets the published results
be reproduced digit for digit while the seams exist.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from plantbench.backend import check as check_device
from plantbench.backend import clamp, module, namespace, to_device, to_host

from .instruments import NEGLIGIBLE, Actuator, Instrument, stack_actuators, stack_instruments

# How the Jacobian of the closed loop is formed; see `integrate`.
JACOBIANS = ("internal", "batched")


@dataclass
class Loop:
    """A single PI or PID loop.

    `measure` maps the plant state, the current inputs and the parameters to the
    controlled variable.  `mv` names the field of the inputs the loop writes, or
    `"sp:<loop>"` for a primary loop that writes the set-point of another.  `Kc` carries
    the sign: a negative gain means raising the manipulated variable lowers the
    measurement.  A set-point of None means the design value, which `bias_from_design`
    fills in.

    `pv_range` is the transmitter's span, (PV_min, PV_max) in the units of the measure,
    and `(lo, hi)` the output's.  They serve only to express the signals in percent of
    span, as a simulator such as HYSYS reports them (`percent_signals`, `percent_gain`).
    The measurement is not clamped to its span: that would make the right-hand side
    non-smooth at the limits and change the dynamics of every published result.

    The control law is the ideal (ISA standard) form, Kc multiplying every term, with the
    derivative acting on the filtered measurement rather than on the error, so that a
    set-point change gives no derivative kick:

        OP = bias + Kc (e + I / tau_I - tau_D dz/dt),
        dz/dt = (pv - z) / (alpha tau_D),

    with e = SP - pv and dI/dt = e.  The filter state z is one augmented state, and with it
    tau_D dz/dt = (pv - z) / alpha: the derivative of the measurement through a first-order
    filter of time constant alpha tau_D, formed without differentiating the plant.  The
    right-hand side stays continuous.  A loop with tau_D = 0, the default, is a PI loop
    and has no filter state.
    """

    name: str
    measure: Callable[[np.ndarray, Any, Any], float]
    mv: str
    Kc: float
    tau_I: float
    setpoint: float | None = None
    bias: float = 0.0
    lo: float = 0.0
    hi: float = np.inf
    # Set-point changes during a run, as (time, change) pairs; the changes in force at
    # time t are summed onto the set-point.  A primary loop's writes take precedence.
    sp_schedule: tuple[tuple[float, float], ...] = ()
    # Measurement and valve non-idealities.  None means ideal, which is the default and
    # the configuration every published result was produced with.
    instrument: Instrument | None = None
    actuator: Actuator | None = None
    pv_range: tuple[float, float] | None = None
    # Derivative time, min, and the derivative filter's time constant as a fraction of it.
    tau_D: float = 0.0
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.tau_D < 0.0:
            raise ValueError(f"loop {self.name!r}: tau_D must not be negative")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"loop {self.name!r}: alpha must be in (0, 1]")

    @property
    def n_filter_states(self) -> int:
        """1 for a loop with derivative action, which filters its measurement; else 0."""
        return 1 if self.tau_D > NEGLIGIBLE else 0

    def output(self, pv: float, integral: float, z=None) -> tuple[float, float]:
        """PI output with clamping, or PID output given the filter state z.  Returns the
        clamped value and the unclamped one."""
        if self.setpoint is None:
            raise ValueError(f"loop {self.name!r} has no set-point; build the structure "
                             "with bias_from_design, or set one explicitly")
        e = self.setpoint - pv
        if z is None:
            raw = self.bias + self.Kc * (e + integral / self.tau_I)
        else:
            raw = self.bias + self.Kc * (e + integral / self.tau_I - (pv - z) / self.alpha)
        return clamp(raw, self.lo, self.hi), raw


@dataclass
class _StackedLoop(Loop):
    """A loop of an ensemble (`stack`): its fields hold one value per member or per row,
    and its layout is the members' common one, so that it is not recomputed from arrays."""

    n_filter: int = 0

    def __post_init__(self) -> None:  # the members were validated when constructed
        pass

    @property
    def n_filter_states(self) -> int:
        return self.n_filter


@dataclass
class Ratio:
    """A ratio station: `mv` is held at `ratio` times the input `source`, without feedback.

    It is evaluated after every loop, so it follows whatever a loop has just written to
    its source.  It keeps an integral state like a loop, held at zero, so that a
    structure has exactly one such state per element.
    """

    name: str
    mv: str
    source: str
    ratio: float | None = None  # None: the design ratio, filled in by bias_from_design
    lo: float = 0.0
    hi: float = np.inf
    actuator: Actuator | None = None


Structure = list[Loop | Ratio]


# --------------------------------------------------------------------------------
# Closed-loop simulation
# --------------------------------------------------------------------------------


def state_layout(structure: Structure) -> tuple[list[tuple[slice, slice, slice]], int]:
    """Where each element's instrument, actuator and derivative-filter states live in the
    augmented vector.

    Integrals occupy the first `len(structure)` entries, in structure order, and the
    instrument, actuator and filter states of each element follow.  An ideal PI element
    contributes none, so such a structure gives `(empty slices, len(structure))`.
    """
    off = len(structure)
    slots = []
    for element in structure:
        inst = getattr(element, "instrument", None)
        act = getattr(element, "actuator", None)
        ni = inst.n_states if inst is not None else 0
        na = act.n_states if act is not None else 0
        nf = getattr(element, "n_filter_states", 0)
        slots.append((slice(off, off + ni), slice(off + ni, off + ni + na),
                      slice(off + ni + na, off + ni + na + nf)))
        off += ni + na + nf
    return slots, off


def n_augmented(structure: Structure) -> int:
    """Length of the augmented state vector this structure adds to the plant."""
    return state_layout(structure)[1]


def initial_augmented(structure: Structure, design) -> np.ndarray:
    """Augmented states that leave the closed loop at rest at the design point.

    Integrals start at zero, as they always have.  Instrument states are initialized so
    that the reported measurement equals the true one, and actuator states so that the
    valve already sits at its design position; otherwise a non-ideal structure would begin
    every run with a transient that has nothing to do with the disturbance under study.
    A derivative filter starts at the measurement, so the derivative term starts at zero.
    """
    slots, n_aug = state_layout(structure)
    aug = np.zeros(n_aug)
    if n_aug == len(structure):
        return aug
    for k, element in enumerate(structure):
        i_slot, a_slot, f_slot = slots[k]
        inst = getattr(element, "instrument", None)
        if inst is not None and inst.n_states:
            aug[i_slot] = inst.initial(element.measure(design.x, design.u, design.pp))
        if getattr(element, "n_filter_states", 0):
            # An instrument starts at rest, reporting the true value.
            aug[f_slot] = element.measure(design.x, design.u, design.pp)
        act = getattr(element, "actuator", None)
        if act is not None and act.n_states:
            aug[a_slot] = act.initial(getattr(design.u, element.mv))
    return aug


def apply(structure: Structure, x: np.ndarray, aug: np.ndarray, u0,
          pp, t: float = 0.0, signals: dict | None = None) -> tuple[Any, np.ndarray]:
    """Evaluate every loop and return the resulting inputs and the augmented rates.

    For a batch, `x` is (m, n), `aug` (m, n_aug) and the fields of `u0` carry the same
    leading axis (`batch_inputs`); the measures must then accept a batch as well.

    Given a dict as `signals`, `apply` also records each loop's signals in it, under
    "<loop>.SP", the live set-point, after the schedule and any primary's write;
    "<loop>.PV", the measurement the controller sees, after its instrument; and
    "<loop>.OP", the output it commands, before its actuator, which for a primary is the
    secondary's set-point.  A ratio station records its output only.  This is the only
    place a secondary's live set-point exists."""
    u = replace(u0)
    slots, n_aug = state_layout(structure)
    d_integ = namespace(x).zeros(x.shape[:-1] + (n_aug,))
    # A primary loop writes the set-point of a secondary rather than a valve, so the
    # structure is evaluated with a local copy whose set-points can be overwritten.
    live = [_scheduled(lp, t) for lp in structure]
    # Primaries first, so that a secondary is evaluated against the set-point its
    # primary has just written rather than against the previous one.
    loops = [k for k, lp in enumerate(live) if isinstance(lp, Loop)]
    order = sorted(loops, key=lambda k: not live[k].mv.startswith("sp:"))
    for k in order:
        loop = live[k]
        true_pv = loop.measure(x, u, pp)
        # What the controller sees, which is the true value only for an ideal instrument.
        inst = loop.instrument
        if inst is None:
            pv = true_pv
        else:
            i_states = aug[..., slots[k][0]]
            pv = inst.output(i_states, true_pv)
            if inst.n_states:
                d_integ[..., slots[k][0]] = inst.derivatives(i_states, true_pv)
        f_slot = slots[k][2]
        if loop.n_filter_states:
            z = aug[..., f_slot.start]
            value, raw = loop.output(pv, aug[..., k], z)
            d_integ[..., f_slot.start] = (pv - z) / (loop.alpha * loop.tau_D)
        else:
            value, raw = loop.output(pv, aug[..., k])
        if signals is not None:
            signals[f"{loop.name}.SP"] = loop.setpoint
            signals[f"{loop.name}.PV"] = pv
            signals[f"{loop.name}.OP"] = value
        if loop.mv.startswith("sp:"):
            target = loop.mv[3:]
            for j, other in enumerate(live):
                if other.name == target:
                    live[j] = replace(other, setpoint=value)
                    break
            else:
                raise KeyError(f"cascade target {target!r} is not in the structure")
        else:
            # A valve reaches the commanded position only as fast as it can travel.
            act = loop.actuator
            if act is None:
                setattr(u, loop.mv, value)
            else:
                a_states = aug[..., slots[k][1]]
                setattr(u, loop.mv, act.output(a_states, value))
                if act.n_states:
                    d_integ[..., slots[k][1]] = act.derivatives(a_states, value)
        # Back-calculation anti-windup with tracking time tau_I: on a limit, the integral
        # is pulled back until the unclamped output sits at the limit.  Unlike switching
        # integration off, this keeps the right-hand side continuous, so the integrator does
        # not chatter across a limit.
        d_integ[..., k] = (loop.setpoint - pv) + (value - raw) / loop.Kc
    for k, station in enumerate(live):
        if isinstance(station, Ratio):
            value = station.ratio * getattr(u, station.source)
            value = clamp(value, station.lo, station.hi)
            if signals is not None:
                signals[f"{station.name}.OP"] = value
            act = station.actuator
            if act is None:
                setattr(u, station.mv, value)
            else:
                a_states = aug[..., slots[k][1]]
                setattr(u, station.mv, act.output(a_states, value))
                if act.n_states:
                    d_integ[..., slots[k][1]] = act.derivatives(a_states, value)
    return u, d_integ


def _scheduled(element, t: float):
    """The element with the set-point changes in force at time t applied."""
    if not isinstance(element, Loop) or not element.sp_schedule:
        return element
    if isinstance(element.sp_schedule, StackedSchedule):
        return replace(element, setpoint=element.setpoint + element.sp_schedule.at(t))
    active = [change for time, change in element.sp_schedule if t >= time]
    return replace(element, setpoint=element.setpoint + sum(active)) if active else element


def closed_loop_rhs(t, y, structure, u0, pp, disturbance, model):
    """Closed-loop derivative.  `model` is the plant's right-hand side, taking
    (t, x, u, pp); a design supplies it as `design.rhs`."""
    n = len(y) - n_augmented(structure)
    x, aug = y[:n], y[n:]
    base = u0 if disturbance is None else disturbance(t, u0)
    u, d_integ = apply(structure, x, aug, base, pp, t)
    return np.concatenate([model(t, x, u, pp), d_integ])


# --------------------------------------------------------------------------------
# The batched closed loop
# --------------------------------------------------------------------------------
#
# A design whose right-hand side accepts a batch of states, (m, n) with inputs whose fields
# carry the same leading axis, declares `batched = True`; so must the measures its loops
# read.  The closed loop can then be evaluated at many states in one call, which is what
# a finite-difference Jacobian needs: one perturbed state per column.  On a GPU the whole
# Jacobian is one batch.


def supports_batch(design) -> bool:
    """Whether the design's right-hand side and measures accept a batch of states."""
    return bool(getattr(design, "batched", False))


def batch_inputs(u, m: int, xp=np):
    """A copy of the inputs `u` with every field repeated along a new leading axis of m:
    a scalar becomes (m,), a vector (m, k).  A field that is None stays None."""
    fields = {}
    for f in dataclasses.fields(u):
        v = getattr(u, f.name)
        if v is not None:
            v = xp.asarray(v, dtype=float)
            v = xp.broadcast_to(v, (m,) + v.shape).copy()
        fields[f.name] = v
    return replace(u, **fields)


def closed_loop_rhs_batch(t: float, Y, structure, u0, pp, disturbance, model):
    """`closed_loop_rhs` at m states at once, Y (m, n + n_aug), all at time t.

    The disturbance is applied once, to the nominal inputs, and the result repeated for
    every state; on the device, Y is a CuPy array and the result is one too."""
    xp = namespace(Y)
    n = Y.shape[-1] - n_augmented(structure)
    X, aug = Y[:, :n], Y[:, n:]
    base = u0 if disturbance is None else disturbance(t, u0)
    u, d_integ = apply(structure, X, aug, batch_inputs(base, Y.shape[0], xp), pp, t)
    return xp.concatenate([model(t, X, u, pp), d_integ], axis=1)


# The relative increment of the forward-difference Jacobian, the square root of the
# machine epsilon, with a floor of one unit for states that sit near zero (integrals,
# deadtime states).
_FD_STEP = float(np.sqrt(np.finfo(float).eps))


def jacobian_batched(t: float, y, structure, u0, pp, disturbance, model):
    """The Jacobian of the closed loop at (t, y) by forward differences, every column in
    one batched evaluation of n + 1 states.  `y` may be a CuPy array; so is the result."""
    xp = namespace(y)
    h = _FD_STEP * xp.maximum(xp.abs(y), 1.0)
    h = (y + h) - y  # an increment the arithmetic represents exactly
    Y = xp.concatenate([y[None], y[None] + xp.diag(h)])
    F = closed_loop_rhs_batch(t, Y, structure, u0, pp, disturbance, model)
    return ((F[1:] - F[0]) / h[:, None]).T


def _jacobian_callback(design, structure, disturbance, device: str):
    """The `jac` callable LSODA takes, evaluating on `device` and returning to the host."""
    if not supports_batch(design):
        raise ValueError(f"{type(design).__name__} does not declare a batched right-hand "
                         "side (`batched = True`); use jacobian='internal'")
    args = (structure, design.u, design.pp, disturbance, design.rhs)

    def jac(t, y, *_):  # solve_ivp passes the right-hand side's arguments as well
        return to_host(jacobian_batched(t, to_device(y, device), *args))

    return jac


# --------------------------------------------------------------------------------
# The closed loop of an ensemble
# --------------------------------------------------------------------------------
#
# The members of an ensemble share a design and the layout of their structures, and
# differ in tunings, set-points, instruments, actuators and disturbances, and in where
# they are in time.  `stack` holds the members' parameters as arrays of one value per
# member; `take` repeats them over the rows of an evaluation, one row per member for a
# step and n + 1 per member for a Jacobian.  The design's right-hand side must not depend
# on t, which is then an array of one time per row.

# The fields of a loop or a ratio station that may differ between members.
_LOOP_FIELDS = ("Kc", "tau_I", "setpoint", "bias", "lo", "hi", "tau_D", "alpha")
_RATIO_FIELDS = ("ratio", "lo", "hi")


@dataclass(frozen=True)
class StackedSchedule:
    """The set-point changes of several members, (members, k) arrays of times and
    changes, padded with changes of zero at infinite times."""

    times: Any
    changes: Any

    def at(self, t):
        """The sum of the changes in force at t, one time per row."""
        return (self.changes * (t[..., None] >= self.times)).sum(axis=-1)

    def take(self, rows) -> StackedSchedule:
        return StackedSchedule(self.times[rows], self.changes[rows])


def layout_signature(structure: Structure) -> tuple:
    """What members must share to be stacked: the elements' kinds, names and manipulated
    variables, and the augmented-state layout."""
    slots, n_aug = state_layout(structure)
    kinds = tuple((type(e).__name__, e.name, e.mv) for e in structure)
    return kinds, tuple(tuple(part.stop - part.start for part in s) for s in slots), n_aug


def stack(structures: list[Structure], xp=np) -> Structure:
    """One structure for the members `structures`, each field that may differ held as an
    array of one value per member, on the array module `xp`.

    The members must have been built from one design by one case, so that a measure of a
    given loop name is the same function of the state in every member; the first
    member's is kept.  Raises when their layouts differ (`layout_signature`)."""
    first = structures[0]
    signature = layout_signature(first)
    for s in structures[1:]:
        if layout_signature(s) != signature:
            raise ValueError("the members' control structures differ in their elements or "
                             "their state layout and cannot be stacked")
    out = []
    for k, element in enumerate(first):
        members = [s[k] for s in structures]
        names = _LOOP_FIELDS if isinstance(element, Loop) else _RATIO_FIELDS
        changes = {f: xp.asarray([getattr(m, f) for m in members], dtype=float) for f in names}
        if isinstance(element, Loop):
            changes["instrument"] = stack_instruments([m.instrument for m in members], xp)
            changes["sp_schedule"] = _stack_schedules([m.sp_schedule for m in members], xp)
            changes["actuator"] = stack_actuators([m.actuator for m in members], xp)
            fields = {f.name: getattr(element, f.name) for f in dataclasses.fields(Loop)}
            out.append(_StackedLoop(**{**fields, **changes},
                                    n_filter=element.n_filter_states))
            continue
        changes["actuator"] = stack_actuators([m.actuator for m in members], xp)
        out.append(replace(element, **changes))
    return out


def _stack_schedules(schedules, xp):
    k = max(len(s) for s in schedules)
    if k == 0:
        return ()
    times = np.full((len(schedules), k), np.inf)
    changes = np.zeros((len(schedules), k))
    for i, s in enumerate(schedules):
        for j, (t, dv) in enumerate(s):
            times[i, j], changes[i, j] = t, dv
    return StackedSchedule(xp.asarray(times), xp.asarray(changes))


def take(stacked: Structure, rows) -> Structure:
    """The stacked structure with each member's values repeated over its rows: `rows` is
    the member of each row, an integer array on the structure's device."""
    out = []
    for element in stacked:
        names = _LOOP_FIELDS if isinstance(element, Loop) else _RATIO_FIELDS
        changes = {f: getattr(element, f)[rows] for f in names}
        if isinstance(element, Loop):
            if element.instrument is not None:
                changes["instrument"] = element.instrument.take(rows)
            if element.sp_schedule:
                changes["sp_schedule"] = element.sp_schedule.take(rows)
        if element.actuator is not None:
            changes["actuator"] = element.actuator.take(rows)
        out.append(replace(element, **changes))
    return out


def ensemble_inputs(t_members: np.ndarray, disturbances, u0, rows, xp=np):
    """The inputs of every row: each member's disturbance at its own time, applied to the
    nominal inputs `u0` on the host, then repeated over the member's `rows` (a host integer
    array) and moved to `xp`.  A member whose disturbance is None takes `u0`."""
    members = [u0 if d is None else d(float(t), u0) for t, d in zip(t_members, disturbances)]
    fields = {}
    for f in dataclasses.fields(u0):
        values = [getattr(u, f.name) for u in members]
        if all(v is None for v in values):
            fields[f.name] = None
            continue
        fields[f.name] = xp.asarray(np.asarray(values, dtype=float)[rows])
    return replace(u0, **fields)


def ensemble_rhs(t_rows, Y, stacked_rows: Structure, inputs, pp, model):
    """The closed loop at m states, one time per row, Y (m, n + n_aug), with the structure
    and the inputs already taken over the rows (`take`, `ensemble_inputs`)."""
    xp = namespace(Y)
    n = Y.shape[-1] - n_augmented(stacked_rows)
    X, aug = Y[:, :n], Y[:, n:]
    u, d_integ = apply(stacked_rows, X, aug, inputs, pp, t_rows)
    return xp.concatenate([model(t_rows, X, u, pp), d_integ], axis=1)


def ensemble_jacobian(t_rows, y, stacked_rows: Structure, inputs, pp, model):
    """The Jacobians of k members at states y (k, n) by forward differences, as
    `jacobian_batched` forms each, in one evaluation of k (n + 1) rows.  The structure,
    the inputs and `t_rows` are taken over those rows, n + 1 per member in member order.
    Returns (k, n, n)."""
    xp = namespace(y)
    k, n = y.shape
    h = _FD_STEP * xp.maximum(xp.abs(y), 1.0)
    h = (y + h) - y
    Y = xp.empty((k, n + 1, n))
    Y[:, 0] = y
    Y[:, 1:] = y[:, None, :] + h[:, :, None] * xp.eye(n)
    F = ensemble_rhs(t_rows, Y.reshape(k * (n + 1), n), stacked_rows, inputs, pp, model)
    F = F.reshape(k, n + 1, n)
    return ((F[:, 1:] - F[:, :1]) / h[:, :, None]).transpose(0, 2, 1)


def batch_outputs(design, structure: Structure, disturbance, t: np.ndarray, Y: np.ndarray,
                  measurements: dict[str, Callable]) -> tuple[dict, dict, dict]:
    """The inputs, the measurements and the controller signals of one run at its grid
    points, as `integrate` records them, but in one batched evaluation: `Y`
    (n_times, n + n_aug) holds the closed-loop states at times `t`.  The design and the
    measures must accept a batch.
    Rows that are not finite, those after a member stopped, are NaN."""
    n = len(design.x)
    ok = np.flatnonzero(np.isfinite(Y).all(axis=1))
    m = ok.size
    S = take(stack([structure]), np.zeros(m, dtype=int))
    inputs = ensemble_inputs(t[ok], [disturbance] * m, design.u, np.arange(m))
    signals: dict = {}
    u, _ = apply(S, Y[ok, :n], Y[ok, n:], inputs, design.pp, t[ok], signals)

    def full(v):
        out = np.full(len(t), np.nan)
        out[ok] = np.broadcast_to(to_host(v), (m,))
        return out

    U: dict[str, np.ndarray] = {}
    for f in dataclasses.fields(u):
        v = getattr(u, f.name)
        if v is None:
            U[f.name] = np.full(len(t), np.nan)
        elif np.ndim(v) <= 1:
            U[f.name] = full(v)
        else:
            for k in range(v.shape[1]):
                U[f"{f.name}[{k}]"] = full(v[:, k])
    U.update({name: full(g(u)) for name, g in getattr(design, "derived", {}).items()})
    Ym = {name: full(g(Y[ok, :n], u, design.pp)) for name, g in measurements.items()}
    C = {name: full(v) for name, v in signals.items()}
    return U, Ym, percent_signals(structure, C)


class WallTimeExceeded(RuntimeError):
    """A run stopped because it exceeded its wall-clock budget."""


@dataclass
class Run:
    """What `integrate` returns: states, inputs, measurements, the solver's cost and the
    controller signals (`apply`, `percent_signals`)."""

    t: np.ndarray
    X: np.ndarray
    U: dict[str, np.ndarray]
    Y: dict[str, np.ndarray]
    nfev: int
    C: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)


def integrate(
    design,
    structure: Structure,
    t_end: float,
    disturbance=None,
    n_points: int = 1200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    measurements: dict[str, Callable] | None = None,
    wall_budget: float | None = None,
    breakpoints=None,
    jacobian: str = "internal",
    device: str = "cpu",
) -> Run:
    """Integrate the closed-loop plant.

    `X` holds the plant states only.  `U` is a dict of arrays, reconstructed after the
    fact so that the control action can be plotted alongside the states: one per scalar
    input (an array-valued input contributes one entry per element, `name[i]`, and an
    input that is None is recorded as NaN), plus any derived signals the design declares
    in `design.derived`, a mapping from name to a function of the inputs.  `Y` holds the
    named `measurements`, each a function of (x, u, pp), evaluated on the same grid, and
    `C` the signals of every loop, as `apply` records them, with their percentages of span
    where the loop declares one (`percent_signals`).

    A `wall_budget` in seconds raises `WallTimeExceeded` once the integration has run
    that long.  The check is a terminal event, which the solver evaluates between its
    steps: the right-hand side it integrates is unchanged, and nothing is raised from
    inside the compiled solver, which older SciPy releases do not survive.

    `breakpoints` are times at which the inputs change discontinuously, a step
    disturbance or a set-point step.  The integration is restarted at each of them, so
    that the solver never takes a step across one.  Without them a plant at rest lets the
    solver grow its step until a single step spans the discontinuity, and the trial state
    it lands on can be unphysical.  `breakpoints_of` collects them from a disturbance and a
    structure.  With none given the integration is a single call, as it has always been.

    A trajectory with a non-finite state raises rather than being returned: the solver
    does not always reject a step that produced one.

    `jacobian` is "internal", LSODA's own forward differences, one right-hand side call per
    state, which is what every published result was produced with; or "batched", the same
    differences evaluated as one batch by `jacobian_batched` and handed to LSODA, on
    `device` ("cpu" or "cuda").  The two agree to the accuracy of the differences, so the
    trajectories agree to the solver's tolerance but not to the last bit.  "batched" needs
    a design that declares `batched = True`.
    """
    if jacobian not in JACOBIANS:
        raise ValueError(f"jacobian must be one of {JACOBIANS}, not {jacobian!r}")
    check_device(device)
    if device != "cpu" and jacobian != "batched":
        raise ValueError("a device other than the CPU needs jacobian='batched'")
    pp = design.pp
    y0 = np.concatenate([design.x, initial_augmented(structure, design)])
    t_eval = np.linspace(0.0, t_end, n_points)
    events = None if wall_budget is None else _deadline(time.perf_counter() + wall_budget)
    args = (structure, design.u, pp, disturbance, design.rhs)
    jac = (None if jacobian == "internal"
           else {"jac": _jacobian_callback(design, structure, disturbance, device)})
    cuts = sorted({float(b) for b in (breakpoints or ()) if 0.0 < b < t_end})
    if not cuts:
        sol = solve_ivp(closed_loop_rhs, (0.0, t_end), y0, args=args, method="LSODA",
                        t_eval=t_eval, rtol=rtol, atol=atol, events=events, **(jac or {}))
        _stop_on_deadline(sol, wall_budget)
        if not sol.success:
            raise RuntimeError(f"closed-loop integration failed: {sol.message}")
        t_out, y_out, nfev = sol.t, sol.y, int(sol.nfev)
    else:
        t_out, y_out, nfev = _integrate_piecewise(closed_loop_rhs, y0, args,
                                                   [0.0, *cuts, t_end], t_eval, rtol, atol,
                                                   events, wall_budget, jac)
    finite = np.isfinite(y_out).all(axis=0)
    if not finite.all():
        raise RuntimeError(f"closed-loop integration produced a non-finite state at "
                           f"t = {t_out[np.argmin(finite)]:.2f}")
    sol = _Solution(t_out, y_out)

    n = len(design.x)
    derived = getattr(design, "derived", {})
    measurements = measurements or {}
    U: dict[str, np.ndarray] = {}
    Y = {name: np.empty(sol.t.size) for name in measurements}
    C: dict[str, np.ndarray] = {}
    for i, ti in enumerate(sol.t):
        base = design.u if disturbance is None else disturbance(ti, design.u)
        x = sol.y[:n, i]
        signals: dict = {}
        u, _ = apply(structure, x, sol.y[n:, i], base, pp, ti, signals)
        for name, v in signals.items():
            C.setdefault(name, np.empty(sol.t.size))[i] = v
        values = input_values(u)
        values.update({name: float(f(u)) for name, f in derived.items()})
        for name, v in values.items():
            U.setdefault(name, np.full(sol.t.size, np.nan))[i] = v
        for name, f in measurements.items():
            Y[name][i] = f(x, u, pp)
    return Run(sol.t, sol.y[:n, :], U, Y, nfev, percent_signals(structure, C))


@dataclass
class _Solution:
    t: np.ndarray
    y: np.ndarray


def _deadline(deadline: float):
    """A terminal event that fires at the first step taken after `deadline`.

    The event is zero at the time of that step and positive before it, so the solver
    places the stop at the step itself rather than searching for a crossing that a wall
    clock does not have.
    """
    stop: list[float] = []

    def event(t, y, *args):
        if not stop:
            if time.perf_counter() <= deadline:
                return 1.0
            stop.append(t)
        return stop[0] - t

    event.terminal = True
    event.direction = -1
    return event


def _stop_on_deadline(sol, wall_budget) -> None:
    if sol.status == 1:
        raise WallTimeExceeded(f"stopped at t = {sol.t_events[0][0]:.1f} after "
                               f"{wall_budget:.0f} s")


def _integrate_piecewise(rhs, y0, args, edges, t_eval, rtol, atol, events=None,
                         wall_budget=None, jac=None):
    """Integrate segment by segment between `edges`, restarting the solver at each."""
    ts, ys, nfev = [], [], 0
    y = y0
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = k == len(edges) - 2
        inside = t_eval[(t_eval >= a) & ((t_eval <= b) if last else (t_eval < b))]
        # The segment end is always evaluated, to start the next segment from it.
        grid = np.union1d(inside, [b])
        sol = solve_ivp(rhs, (a, b), y, args=args, method="LSODA", t_eval=grid,
                        rtol=rtol, atol=atol, events=events, **(jac or {}))
        _stop_on_deadline(sol, wall_budget)
        if not sol.success:
            raise RuntimeError(f"closed-loop integration failed on [{a:g}, {b:g}]: "
                               f"{sol.message}")
        nfev += int(sol.nfev)
        keep = np.isin(sol.t, inside)
        ts.append(sol.t[keep])
        ys.append(sol.y[:, keep])
        y = sol.y[:, -1]
    return np.concatenate(ts), np.concatenate(ys, axis=1), nfev


def breakpoints_of(disturbance, structure: Structure) -> tuple[float, ...]:
    """Times at which a disturbance or a scheduled set-point change acts discontinuously.

    A disturbance declares them as an attribute `times`; the disturbances of
    `core.disturbances` do.  Scheduled set-point changes come from the loops.
    """
    times = set(getattr(disturbance, "times", ()) if disturbance is not None else ())
    for element in structure:
        if isinstance(element, Loop):
            times.update(t for t, _ in element.sp_schedule)
    return tuple(sorted(times))


def simulate(
    design,
    structure: Structure,
    t_end: float,
    disturbance=None,
    n_points: int = 1200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Integrate the closed-loop plant and return (t, X, U); see `integrate`."""
    r = integrate(design, structure, t_end, disturbance, n_points, rtol, atol)
    return r.t, r.X, r.U


def input_values(u) -> dict[str, float]:
    """The inputs as a flat name -> float mapping, as `simulate` records them."""
    out = {}
    for f in dataclasses.fields(u):
        v = getattr(u, f.name)
        if v is None:
            out[f.name] = np.nan
        elif np.ndim(v) == 0:
            out[f.name] = float(v)
        else:
            for k, vk in enumerate(np.ravel(v)):
                out[f"{f.name}[{k}]"] = float(vk)
    return out


def percent_signals(structure: Structure, C: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The signals `C` with their percentages of span added, as a simulator such as HYSYS
    reports them: "<loop>.PV%" where the loop declares a `pv_range`, and "<loop>.OP%"
    where its output range `(lo, hi)` is finite.  The set-point is in the measurement's
    span, so it is not repeated in percent."""
    out = dict(C)
    for e in structure:
        pv_range = getattr(e, "pv_range", None)
        if pv_range is not None and f"{e.name}.PV" in C:
            out[f"{e.name}.PV%"] = _percent(C[f"{e.name}.PV"], *pv_range)
        if np.isfinite(e.hi) and f"{e.name}.OP" in C:
            out[f"{e.name}.OP%"] = _percent(C[f"{e.name}.OP"], e.lo, e.hi)
    return out


def _percent(v, lo: float, hi: float):
    return 100.0 * (v - lo) / (hi - lo)


def percent_gain(loop: Loop) -> float | None:
    """The loop's gain in percent of output span per percent of measurement span, the
    form in which HYSYS states a tuning, with the action as its sign: negative for a
    direct-acting loop.  None without both spans.  For display: a loop is tuned in
    engineering units."""
    if loop.pv_range is None or not np.isfinite(loop.hi):
        return None
    pv_lo, pv_hi = loop.pv_range
    return float(loop.Kc * (pv_hi - pv_lo) / (loop.hi - loop.lo))


def bias_from_design(structure: Structure, design) -> Structure:
    """Set each loop's bias and set-point from the converged design.

    A loop whose bias is the design value of its manipulated variable and whose
    set-point is the design value of its measurement starts with zero error and zero
    output change, so the plant begins the simulation exactly at rest.
    """
    out = []
    for loop in structure:
        if isinstance(loop, Ratio):
            # A ratio station starts at the design ratio of its two inputs.
            ratio = (loop.ratio if loop.ratio is not None
                     else getattr(design.u, loop.mv) / getattr(design.u, loop.source))
            out.append(replace(loop, ratio=ratio))
            continue
        pv = loop.measure(design.x, design.u, design.pp)
        if loop.mv.startswith("sp:"):
            # A primary loop's bias is the secondary's design set-point, which is the
            # secondary's design measurement.
            target = loop.mv[3:]
            sec = next(s for s in structure if s.name == target)
            bias = sec.measure(design.x, design.u, design.pp)
        else:
            bias = getattr(design.u, loop.mv)
        out.append(
            replace(
                loop,
                setpoint=pv if loop.setpoint is None else loop.setpoint,
                bias=bias,
            )
        )
    return out


# --------------------------------------------------------------------------------
# Changing a structure by loop name
# --------------------------------------------------------------------------------
#
# Every helper returns a new structure and leaves the one it was given untouched, and
# raises on a name that is not in the structure, so that a typo in a study configuration
# fails immediately rather than silently leaving the loop as it was.


def _check_names(structure: Structure, names) -> None:
    known = {element.name for element in structure}
    for name in names:
        if name not in known:
            raise KeyError(f"{name!r} is not a loop in this structure; have {sorted(known)}")


def _loop(structure: Structure, name: str) -> Loop:
    element = next(e for e in structure if e.name == name)
    if not isinstance(element, Loop):
        raise TypeError(f"{name!r} is a ratio station, not a feedback loop")
    return element


def with_setpoints(structure: Structure, setpoints: dict[str, float]) -> Structure:
    """A copy of `structure` with the named loops' set-points replaced.

    Applied after `bias_from_design`, a set-point away from the design value starts the
    run with an error, which is how a set-point change at time zero is expressed.  A
    primary loop overwrites its secondary's set-point, so setting a cascade secondary
    has no effect while its primary is in the structure.
    """
    _check_names(structure, setpoints)
    for name in setpoints:
        _loop(structure, name)
    return [replace(e, setpoint=float(setpoints[e.name])) if e.name in setpoints else e
            for e in structure]


def with_setpoint_steps(structure: Structure,
                        steps: dict[str, list[tuple[float, float]]]) -> Structure:
    """A copy of `structure` with set-point changes scheduled on the named loops.

    `steps` maps a loop name to (time, change) pairs, added to any schedule the loop
    already carries.
    """
    _check_names(structure, steps)
    out = []
    for e in structure:
        if e.name in steps:
            loop = _loop(structure, e.name)
            extra = tuple((float(t), float(dv)) for t, dv in steps[e.name])
            e = replace(loop, sp_schedule=loop.sp_schedule + extra)
        out.append(e)
    return out


TUNABLE = ("Kc", "tau_I", "lo", "hi", "tau_D", "alpha")


def with_tuning(structure: Structure, tuning: dict[str, dict[str, float]]) -> Structure:
    """A copy of `structure` with the named loops retuned.

    `tuning` maps a loop name to any of Kc, tau_I, lo, hi, tau_D and alpha.  The bias is
    not changed, so a retuned loop still starts at rest at the design point.  A tau_D
    above zero gives the loop derivative action and a filter state (`Loop`).
    """
    _check_names(structure, tuning)
    for name, values in tuning.items():
        _loop(structure, name)
        unknown = set(values) - set(TUNABLE)
        if unknown:
            raise KeyError(f"cannot tune {sorted(unknown)} on {name!r}; tunable: {TUNABLE}")
    return [replace(e, **{k: float(v) for k, v in tuning[e.name].items()})
            if e.name in tuning else e for e in structure]


# --------------------------------------------------------------------------------
# Attaching non-idealities
# --------------------------------------------------------------------------------


def with_instruments(structure: Structure,
                     instruments: dict[str, Instrument] | None = None,
                     actuators: dict[str, Actuator] | None = None) -> Structure:
    """A copy of `structure` with instruments and actuators attached by loop name.

    Raises if a name does not match, so that a typo in a study configuration fails
    immediately rather than silently leaving the loop ideal.
    """
    instruments = instruments or {}
    actuators = actuators or {}
    _check_names(structure, (*instruments, *actuators))
    out = []
    for element in structure:
        changes = {}
        if element.name in instruments:
            if isinstance(element, Ratio):
                raise TypeError(f"{element.name!r} is a ratio station and has no measurement")
            changes["instrument"] = instruments[element.name]
        if element.name in actuators:
            changes["actuator"] = actuators[element.name]
        out.append(replace(element, **changes) if changes else element)
    return out


# --------------------------------------------------------------------------------
# Closed-loop spectrum
# --------------------------------------------------------------------------------

# Every element carries an integral state, so a ratio station, which has no integral,
# contributes a state held at zero and with it an exact zero eigenvalue.  It says nothing
# about stability, so it is set aside: the eigenvalue nearest zero for every ratio station
# in the structure, none for the structures without one.  Each must be smaller than
# STRUCTURAL_ZERO; a slow mode of the plant itself is kept.  Instrument and actuator states
# are not structural in this sense and their eigenvalues are kept.
STRUCTURAL_ZERO = 1e-5


def closed_loop_spectrum(design, structure: Structure, jacobian: str = "internal",
                         device: str = "cpu") -> np.ndarray:
    """Eigenvalues of the closed-loop plant linearized at its design steady state.

    Structural zeros are removed.  The Jacobian is by central differences, a column at a
    time, or with `jacobian="batched"` every column in one batch of 2n states on `device`,
    which agrees to the accuracy of the differences.
    """
    y0 = np.concatenate([design.x, initial_augmented(structure, design)])
    n = len(y0)
    if jacobian == "batched":
        if not supports_batch(design):
            raise ValueError(f"{type(design).__name__} does not declare a batched "
                             "right-hand side (`batched = True`)")
        xp = module(device)
        y = to_device(y0, device)
        h = 1e-6 * xp.maximum(1.0, xp.abs(y))
        E = xp.diag(h)
        F = closed_loop_rhs_batch(0.0, xp.concatenate([y + E, y - E]), structure,
                                  design.u, design.pp, None, design.rhs)
        J = to_host(((F[:n] - F[n:]) / (2 * h[:, None])).T)
    elif jacobian == "internal":
        f = lambda y: closed_loop_rhs(0.0, y, structure, design.u, design.pp, None,
                                      design.rhs)
        J = np.empty((n, n))
        for j in range(n):
            h = 1e-6 * max(1.0, abs(y0[j]))
            yp, ym = y0.copy(), y0.copy()
            yp[j] += h
            ym[j] -= h
            J[:, j] = (f(yp) - f(ym)) / (2 * h)
    else:
        raise ValueError(f"jacobian must be one of {JACOBIANS}, not {jacobian!r}")
    ev = np.linalg.eigvals(J)
    n_ratio = sum(isinstance(e, Ratio) for e in structure)
    nearest = np.argsort(np.abs(ev))[:n_ratio]
    if np.any(np.abs(ev[nearest]) > STRUCTURAL_ZERO):
        raise RuntimeError(f"{n_ratio} ratio station(s), but only "
                           f"{int((np.abs(ev) <= STRUCTURAL_ZERO).sum())} eigenvalue(s) "
                           f"within {STRUCTURAL_ZERO:g} of zero")
    return np.delete(ev, nearest)


def damping_ratio(eigenvalues: np.ndarray, min_frequency: float = 0.05) -> float:
    """Smallest damping ratio among the oscillatory modes (1.0 if there are none)."""
    osc = eigenvalues[np.abs(eigenvalues.imag) > min_frequency]
    return float(np.min(-osc.real / np.abs(osc))) if osc.size else 1.0
