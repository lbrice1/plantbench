"""PI control loops, the closed-loop simulator and its linearization, for any case.

Control structures are declared as data.  A `Structure` is a list of `Loop`s, each
naming a measurement, a manipulated variable and a tuning, so that comparing two
structures means comparing two lists rather than rewiring the model.

Integral action is carried as extra states appended to the plant state vector, which
keeps the whole closed loop a single ODE system that a stiff integrator can handle and
makes the closed-loop Jacobian available for the stability arguments.

A loop may also carry an `Instrument` on its measurement and an `Actuator` on its output,
and those have states of their own.  The augmented vector is laid out with every integral
first, in structure order, and any instrument and actuator states after them.  An ideal
element contributes none, so a structure of ideal elements has exactly one augmented state
per element and the layout is the one this model has always had --- which is what lets the
published results be reproduced digit for digit while the seams exist.
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

from .instruments import Actuator, Instrument

# How the Jacobian of the closed loop is formed; see `integrate`.
JACOBIANS = ("internal", "batched")


@dataclass
class Loop:
    """A single PI loop.

    `measure` maps the plant state, the current inputs and the parameters to the
    controlled variable.  `mv` names the field of the inputs the loop writes, or
    `"sp:<loop>"` for a primary loop that writes the set-point of another.  `Kc` carries
    the sign: a negative gain means raising the manipulated variable lowers the
    measurement.  A set-point of None means the design value, which `bias_from_design`
    fills in.
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

    def output(self, pv: float, integral: float) -> tuple[float, float]:
        """PI output with clamping.  Returns the clamped value and the unclamped one."""
        if self.setpoint is None:
            raise ValueError(f"loop {self.name!r} has no set-point; build the structure "
                             "with bias_from_design, or set one explicitly")
        e = self.setpoint - pv
        raw = self.bias + self.Kc * (e + integral / self.tau_I)
        return clamp(raw, self.lo, self.hi), raw


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


def state_layout(structure: Structure) -> tuple[list[tuple[slice, slice]], int]:
    """Where each element's instrument and actuator states live in the augmented vector.

    Integrals occupy the first `len(structure)` entries, in structure order, and the
    instrument and actuator states of each element follow.  An ideal element contributes
    neither, so an ideal structure gives `(empty slices, len(structure))`.
    """
    off = len(structure)
    slots = []
    for element in structure:
        inst = getattr(element, "instrument", None)
        act = getattr(element, "actuator", None)
        ni = inst.n_states if inst is not None else 0
        na = act.n_states if act is not None else 0
        slots.append((slice(off, off + ni), slice(off + ni, off + ni + na)))
        off += ni + na
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
    """
    slots, n_aug = state_layout(structure)
    aug = np.zeros(n_aug)
    if n_aug == len(structure):
        return aug
    for k, element in enumerate(structure):
        i_slot, a_slot = slots[k]
        inst = getattr(element, "instrument", None)
        if inst is not None and inst.n_states:
            aug[i_slot] = inst.initial(element.measure(design.x, design.u, design.pp))
        act = getattr(element, "actuator", None)
        if act is not None and act.n_states:
            aug[a_slot] = act.initial(getattr(design.u, element.mv))
    return aug


def apply(structure: Structure, x: np.ndarray, aug: np.ndarray, u0,
          pp, t: float = 0.0) -> tuple[Any, np.ndarray]:
    """Evaluate every loop and return the resulting inputs and the augmented rates.

    For a batch, `x` is (m, n), `aug` (m, n_aug) and the fields of `u0` carry the same
    leading axis (`batch_inputs`); the measures must then accept a batch as well."""
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
        value, raw = loop.output(pv, aug[..., k])
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


class WallTimeExceeded(RuntimeError):
    """A run stopped because it exceeded its wall-clock budget."""


@dataclass
class Run:
    """What `integrate` returns: states, inputs, measurements and the solver's cost."""

    t: np.ndarray
    X: np.ndarray
    U: dict[str, np.ndarray]
    Y: dict[str, np.ndarray]
    nfev: int


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
    named `measurements`, each a function of (x, u, pp), evaluated on the same grid.

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
    for i, ti in enumerate(sol.t):
        base = design.u if disturbance is None else disturbance(ti, design.u)
        x = sol.y[:n, i]
        u, _ = apply(structure, x, sol.y[n:, i], base, pp, ti)
        values = input_values(u)
        values.update({name: float(f(u)) for name, f in derived.items()})
        for name, v in values.items():
            U.setdefault(name, np.full(sol.t.size, np.nan))[i] = v
        for name, f in measurements.items():
            Y[name][i] = f(x, u, pp)
    return Run(sol.t, sol.y[:n, :], U, Y, nfev)


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


TUNABLE = ("Kc", "tau_I", "lo", "hi")


def with_tuning(structure: Structure, tuning: dict[str, dict[str, float]]) -> Structure:
    """A copy of `structure` with the named loops retuned.

    `tuning` maps a loop name to any of Kc, tau_I, lo and hi.  The bias is not changed,
    so a retuned loop still starts at rest at the design point.
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
