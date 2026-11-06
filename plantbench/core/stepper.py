"""A configured plant advanced one interval at a time, for a controller outside the library.

`case.run` integrates a whole trajectory with the controllers given as data.  A policy
that acts at each control interval needs the plant advanced one interval at a time: it
chooses an action at t, the action is held over [t, t + dt] (zero-order hold), and the
closed loop of the plant and the loops that remain is integrated over that interval.  The
plant state, the augmented state of the remaining loops and the time are kept between
steps.

The policy owns inputs, set-points, or both:

- an owned input is removed from the structure with every element that writes it
  (`control.without_inputs`, as a configuration's `hold` removes it), and is set by the
  action before the remaining loops are evaluated, so that a ratio station following it
  follows the action;
- an owned set-point leaves its loop in place and replaces the loop's set-point by the
  action.

With nothing owned, the stepper integrates the closed loop `run` integrates, restarted at
every step boundary as well as at the breakpoints of the disturbance and the set-point
schedules; the two agree to the solver's tolerance.

A step that stalls inside the solver cannot be interrupted from Python: the solver is
compiled, and a signal reaches only the main thread of a POSIX process.  A stepper given a
time limit therefore integrates in a worker process of its own, spawned once and kept for
the stepper's life, and terminates it when a step runs over; the stepper is then
invalidated.  The worker rebuilds the configured plant from the case's importable module
and the plain configuration, so a case defined in a script, or changed after import, can
be stepped only without a limit.
"""

from __future__ import annotations

import importlib
import multiprocessing
import pickle
import sys
import time
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from . import control as ctl
from .case import Case, Setup, _check_inputs, _check_undisturbed, build


class StepFailed(RuntimeError):
    """The integration over an interval failed or reached a non-finite state.  The stepper
    is left where the interval began."""


@dataclass
class Step:
    """The plant at the end of an interval: time, plant state, inputs (as `control.
    input_values` flattens them), named measurements as a sampled instrument reports them,
    the signals of the remaining loops (`control.apply`), the right-hand-side evaluations
    the interval took, and the true values of the measurements.  `y` and `y_true` differ
    only under a configuration's `noise`.  `y_min` and `y_max` hold, for the measurements
    a stepper was asked for, the smallest and the largest true value found over the
    interval (`Stepper`, `extrema`)."""

    t: float
    x: np.ndarray
    u: dict[str, float]
    y: dict[str, float]
    c: dict[str, float]
    nfev: int
    y_true: dict[str, float]
    y_min: dict[str, float] = field(default_factory=dict)
    y_max: dict[str, float] = field(default_factory=dict)


class Stepper:
    """A configured plant (`plantbench.build`) stepped at a fixed interval `dt`, with the
    `inputs` and the loops' `setpoints` it names owned by the caller.

    The run starts where the configuration starts it, at `setup.x0` with the augmented
    states of `control.initial_augmented`, at time 0.  `displaced` holds the elements the
    owned inputs removed, as `Loop` and `Ratio` objects, and `structure` the elements that
    remain.  Refused: an input the design does not have or that is not a scalar, an input
    the configuration holds or a disturbance changes, a set-point of a loop that is removed
    or whose set-point a cascade primary writes, and a set-point that carries a schedule.

    `step_budget` limits the wall-clock seconds of one step, and `episode_budget` those of
    all the steps since the start or the last `reset`; the time a caller spends between
    steps does not count.  With either, the stepper integrates in a worker process
    (`close` ends it, as does leaving a `with` block), and a step that runs over raises
    `control.WallTimeExceeded` after the worker has been terminated; the stepper cannot be
    used again.  `startup_budget` limits the worker's start, which builds the plant again;
    None waits for it.  The setup must be one `build` makes from its configuration.

    A configuration's `noise` adds to each named measurement, once per sample (the start
    and the end of each step), a draw of zero mean and the given standard deviation, from
    a generator seeded by the section's `seed` and used for nothing else.  The loops that
    remain see the true values, as `run`'s do; `Step.y_true` holds them beside `Step.y`.

    `extrema` names measurements whose smallest and largest true values over each interval
    a step reports (`Step.y_min`, `Step.y_max`).  They are taken over `refine` equal parts
    of every segment of the interval between breakpoints, on the solver's dense output,
    and on both sides of each breakpoint, with the interval's ends included.  They are
    numerical estimates that converge as `refine` grows, not bounds: an excursion shorter
    than a part can be missed.  The refinement does not change the integration.
    """

    def __init__(self, setup: Setup, dt: float, inputs=(), setpoints=(),
                 rtol: float = 1e-8, atol: float = 1e-10, step_budget: float | None = None,
                 episode_budget: float | None = None, startup_budget: float | None = None,
                 extrema=(), refine: int = 10):
        if not dt > 0.0:
            raise ValueError("the interval dt must be positive")
        for name, budget in (("step_budget", step_budget), ("episode_budget", episode_budget),
                             ("startup_budget", startup_budget)):
            if budget is not None and not budget > 0.0:
                raise ValueError(f"{name} must be positive seconds, or None")
        self.setup, self.dt, self.rtol, self.atol = setup, float(dt), rtol, atol
        self.design = setup.design
        self.inputs, self.setpoints = tuple(inputs), tuple(setpoints)
        _check_inputs(self.design.u, self.inputs, "own", "owned")
        held = sorted(set(self.inputs) & set(setup.config.hold))
        if held:
            raise ValueError(f"{held} are held by the configuration and cannot be owned")
        self.disturbance = setup.disturbance
        _check_undisturbed(self.disturbance, self.design.u, self.inputs, "owned")
        self.structure, self.displaced = ctl.without_inputs(setup.structure, self.inputs)
        self._owned_loops = self._check_setpoints()
        self.breakpoints = ctl.breakpoints_of(self.disturbance, self.structure)
        self.measurements = setup.case.measurements(self.design)
        self.extrema = tuple(extrema)
        unknown = sorted(set(self.extrema) - set(self.measurements))
        if unknown:
            raise KeyError(f"no measurements {unknown}; measurements: "
                           f"{sorted(self.measurements)}")
        if int(refine) != refine or refine < 1:
            raise ValueError("refine is a whole number of parts per segment, at least 1")
        self.refine = int(refine)
        noise = setup.config.noise
        self.noise_std = {k: float(noise["std"][k]) for k in sorted(noise.get("std", {}))}
        self.noise_seed = int(noise.get("seed", 0))
        self.step_budget, self.episode_budget = step_budget, episode_budget
        self._invalid: str | None = None
        self._worker: _Worker | None = None
        self.reset()
        if step_budget is not None or episode_budget is not None:
            module, attr = _importable(setup.case)
            self._worker = _Worker(self, _serve, (
                module, attr, setup.config.to_dict(), self.dt, self.inputs, self.setpoints,
                rtol, atol, self.extrema, self.refine), startup_budget)
            x, x0 = self._worker.ready
            if not (np.array_equal(x, self.design.x) and np.array_equal(x0, setup.x0)):
                self.close()
                raise ValueError("the worker's plant differs from the setup's: a stepper "
                                 "with a time limit needs a setup that `build` makes from "
                                 "its configuration")

    def _check_setpoints(self) -> dict[str, int]:
        """The position in the structure of each loop whose set-point is owned."""
        names = {e.name: k for k, e in enumerate(self.structure)}
        removed = {e.name for e in self.displaced}
        written = {e.mv[3:]: e.name for e in self.structure
                   if isinstance(e, ctl.Loop) and e.mv.startswith("sp:")}
        out = {}
        for name in self.setpoints:
            if name in removed:
                raise ValueError(f"loop {name!r} is removed with an owned input; its "
                                 "set-point cannot be owned")
            if name not in names:
                raise KeyError(f"no loop {name!r} in the structure; loops are "
                               f"{sorted(names)}")
            loop = self.structure[names[name]]
            if not isinstance(loop, ctl.Loop):
                raise TypeError(f"{name!r} is a ratio station and has no set-point")
            if name in written:
                raise ValueError(f"the set-point of {name!r} is written by the cascade "
                                 f"primary {written[name]!r}")
            if loop.sp_schedule:
                raise ValueError(f"the set-point of {name!r} carries a schedule; an owned "
                                 "set-point is set by the action alone")
            out[name] = names[name]
        return out

    # ------------------------------------------------------------------------------

    def reset(self, noise_seed: int | None = None) -> None:
        """Back to the start of the run, at time 0, with the nominal action, and the noise
        generator seeded again: by the configuration's seed, or by `noise_seed`, so that an
        episode is reproducible from its seed and episodes may differ in their noise."""
        x0 = np.asarray(self.setup.x0, dtype=float)
        self._y = np.concatenate([x0, ctl.initial_augmented(self.structure, self.design, x0)])
        self.t = 0.0
        self.nfev = 0
        self.spent = 0.0  # wall-clock seconds of the steps since the start
        self.action = self.nominal_action()
        self._range: tuple[dict, dict] = ({}, {})
        self._rng = np.random.default_rng(self.noise_seed if noise_seed is None else noise_seed)
        self._noise = self._draw()

    def _draw(self) -> dict[str, float]:
        """The noise of one sample, one draw per noisy measurement in name order."""
        if not self.noise_std:
            return {}
        z = self._rng.standard_normal(len(self.noise_std))
        return {k: s * float(v) for (k, s), v in zip(self.noise_std.items(), z)}

    @property
    def x(self) -> np.ndarray:
        """The plant state now."""
        return self._y[: len(self.design.x)]

    def nominal_action(self) -> dict[str, float]:
        """The action that keeps the plant as configured: the inputs at their values in the
        design, the set-points at those of the structure."""
        action = {f: float(getattr(self.design.u, f)) for f in self.inputs}
        action.update({name: float(self.structure[k].setpoint)
                       for name, k in self._owned_loops.items()})
        return action

    def _structure_with(self, action: Mapping[str, float]) -> ctl.Structure:
        if not self._owned_loops:
            return self.structure
        S = list(self.structure)
        for name, k in self._owned_loops.items():
            S[k] = replace(S[k], setpoint=float(action[name]))
        return S

    def _evaluate(self, t: float, y: np.ndarray, S: ctl.Structure, owned: dict,
                  signals: dict | None = None):
        """The inputs and the augmented rates at (t, y) under structure S and the owned
        inputs' values."""
        n = len(self.design.x)
        base = self.design.u if self.disturbance is None else self.disturbance(t, self.design.u)
        if owned:
            base = replace(base, **owned)
        return ctl.apply(S, y[:n], y[n:], base, self.design.pp, t, signals)

    def _rhs(self, t, y, S, owned):
        u, d_aug = self._evaluate(t, y, S, owned)
        n = len(self.design.x)
        return np.concatenate([self.design.rhs(t, y[:n], u, self.design.pp), d_aug])

    def _check_action(self, action: Mapping[str, float]) -> dict[str, float]:
        expected = {*self.inputs, *self.setpoints}
        if set(action) != expected:
            raise KeyError(f"an action names exactly {sorted(expected)}, not "
                           f"{sorted(action)}")
        return {k: float(v) for k, v in action.items()}

    def step(self, action: Mapping[str, float] | None = None) -> Step:
        """Hold `action` (the nominal one when None) over one interval and advance the
        closed loop, restarting the integration at every breakpoint inside the interval.

        Raises `StepFailed` if the solver fails or a state is non-finite; the stepper is
        then left where the interval began."""
        if self._invalid is not None:
            raise RuntimeError(self._invalid)
        action = self.nominal_action() if action is None else self._check_action(action)
        start = time.perf_counter()
        try:
            if self._worker is None:
                y, nfev, extremes = self._advance(self._y, self.t, action)
            else:
                y, nfev, extremes = self._advance_in_worker(action)
        finally:
            self.spent += time.perf_counter() - start
        self._y, self.t, self.action, self._range = y, self.t + self.dt, action, extremes
        self.nfev += nfev
        self._noise = self._draw()
        return self._observe(nfev)

    def _advance(self, y: np.ndarray, t: float, action: dict, rtol: float | None = None,
                 atol: float | None = None, extrema: tuple | None = None
                 ) -> tuple[np.ndarray, int, tuple[dict, dict]]:
        """The closed-loop state after one interval from (t, y) under `action`, the
        evaluations it took, and the extremes of the `extrema` measurements over it; the
        tolerances and the measurements are the stepper's unless given
        (`core.stepmap`)."""
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        extrema = self.extrema if extrema is None else tuple(extrema)
        S = self._structure_with(action)
        owned = {f: action[f] for f in self.inputs}
        a, b = t, t + self.dt
        edges = [a, *(c for c in self.breakpoints if a < c < b), b]
        nfev = 0
        lo_values: dict[str, float] = {}
        hi_values: dict[str, float] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            grid = np.linspace(lo, hi, self.refine + 1) if extrema else [hi]
            sol = solve_ivp(self._rhs, (lo, hi), y, args=(S, owned), method="LSODA",
                            t_eval=grid, rtol=rtol, atol=atol)
            if not sol.success:
                raise StepFailed(f"integration failed on [{lo:g}, {hi:g}]: {sol.message}")
            nfev += int(sol.nfev)
            if extrema:
                if not np.isfinite(sol.y).all():
                    raise StepFailed(f"non-finite state on [{lo:g}, {hi:g}]")
                # The segment's end is evaluated with the inputs in force just before it,
                # its start with those from it on: both sides of each breakpoint.
                times = [*sol.t[:-1], np.nextafter(hi, lo)]
                for tk, yk in zip(times, sol.y.T):
                    u, _ = self._evaluate(tk, yk, S, owned)
                    for name in extrema:
                        v = float(self.measurements[name](yk[:len(self.design.x)], u,
                                                          self.design.pp))
                        lo_values[name] = min(lo_values.get(name, v), v)
                        hi_values[name] = max(hi_values.get(name, v), v)
            y = sol.y[:, -1]
            if not np.isfinite(y).all():
                raise StepFailed(f"non-finite state at t = {hi:g}")
        if extrema:
            # The interval's end as the next interval sees it, with the inputs at b.
            u, _ = self._evaluate(b, y, S, owned)
            for name in extrema:
                v = float(self.measurements[name](y[:len(self.design.x)], u, self.design.pp))
                lo_values[name] = min(lo_values[name], v)
                hi_values[name] = max(hi_values[name], v)
        return y, nfev, (lo_values, hi_values)

    def _advance_in_worker(self, action: dict) -> tuple[np.ndarray, int, tuple[dict, dict]]:
        limits = [b for b in (self.step_budget,
                              None if self.episode_budget is None
                              else self.episode_budget - self.spent) if b is not None]
        limit = min(limits)
        try:
            if limit <= 0.0:
                raise ctl.WallTimeExceeded(f"the episode's {self.episode_budget:g} s are "
                                           f"spent at t = {self.t:g}")
            return self._worker.call(("step", self._y, self.t, action), limit,
                                     f"the step from t = {self.t:g}")
        except (ctl.WallTimeExceeded, _WorkerLost) as exc:
            self._invalid = (f"this stepper was stopped ({exc}) and cannot step again; "
                             "build a new one")
            self.close()
            if isinstance(exc, _WorkerLost):
                raise RuntimeError(str(exc)) from None
            raise

    def close(self) -> None:
        """End the worker process of a stepper with a time limit; nothing otherwise."""
        if self._worker is not None:
            self._worker.close()

    def __enter__(self) -> Stepper:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def observe(self) -> Step:
        """The plant now, under the action last held (the nominal one before any step),
        with the noise drawn for this sample: observing again draws nothing new."""
        return self._observe(0)

    def _observe(self, nfev: int) -> Step:
        S = self._structure_with(self.action)
        signals: dict = {}
        u, _ = self._evaluate(self.t, self._y, S, {f: self.action[f] for f in self.inputs},
                              signals)
        x = self.x.copy()
        values = ctl.input_values(u)
        values.update({name: float(f(u))
                       for name, f in getattr(self.design, "derived", {}).items()})
        y = {name: float(f(x, u, self.design.pp)) for name, f in self.measurements.items()}
        measured = {name: v + self._noise.get(name, 0.0) for name, v in y.items()}
        return Step(self.t, x, values, measured, {k: float(v) for k, v in signals.items()},
                    nfev, y, dict(self._range[0]), dict(self._range[1]))


# ------------------------------------------------------------------------------------
# The worker of a stepper with a time limit
# ------------------------------------------------------------------------------------


class _WorkerLost(RuntimeError):
    """The worker process ended without answering."""


def _importable(case: Case) -> tuple[str, str]:
    """The module and the global name under which an importable module holds `case`."""
    for fn in (case.make_design, case.make_structure):
        name = getattr(fn, "__module__", None)
        module = sys.modules.get(name) if name not in (None, "__main__") else None
        if module is None:
            continue
        for attr, value in vars(module).items():
            if value is case:
                return name, attr
    raise ValueError(f"case {case.id} cannot be rebuilt in a worker process: no importable "
                     "module holds it as a global.  A case defined in a script, or changed "
                     "after import, can be stepped without a time limit only")


def _portable(exc: BaseException) -> BaseException:
    """The exception itself if it survives pickling, else a RuntimeError naming it."""
    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:  # noqa: BLE001 -- any exception that does not travel is described
        return RuntimeError(f"{type(exc).__name__}: {exc}")


def _serve(conn, module: str, attr: str, config: dict, dt: float, inputs, setpoints,
           rtol: float, atol: float, extrema, refine: int) -> None:
    """The worker: build the plant, report its design and start, then advance intervals
    on request until told to stop or the connection closes."""
    try:
        case = getattr(importlib.import_module(module), attr)
        stepper = Stepper(build(case, config), dt, inputs, setpoints, rtol, atol,
                          extrema=extrema, refine=refine)
    except BaseException as exc:  # noqa: BLE001 -- reported to the parent
        conn.send(("error", _portable(exc)))
        return
    conn.send(("ready", (stepper.design.x, stepper.setup.x0)))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message[0] == "close":
            return
        _, y, t, action = message
        try:
            reply = ("ok", stepper._advance(y, t, action))
        except Exception as exc:  # noqa: BLE001 -- reported to the parent
            reply = ("error", _portable(exc))
        conn.send(reply)


class _Worker:
    """A spawned process running `target(conn, *args)`, which builds its plant, sends what
    it built as `ready` and then advances intervals on request; terminated on a timeout,
    and when `owner` is collected."""

    def __init__(self, owner, target, args: tuple, startup_budget: float | None):
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=target, daemon=True, args=(child, *args))
        self.process.start()
        child.close()
        self._finalizer = weakref.finalize(owner, _stop, self.process, self.conn)
        try:
            self.ready = self.call(None, startup_budget, "the worker's start")
        except BaseException:
            self.close()
            raise

    def call(self, message, limit: float | None, what: str) -> Any:
        """Send `message` (None to wait for the start) and return the reply's value, or
        raise what the worker raised; after `limit` seconds, terminate the worker and
        raise `WallTimeExceeded`."""
        if message is not None:
            try:
                self.conn.send(message)
            except OSError:  # the worker has gone and its end of the pipe with it
                self.process.join(1.0)
                raise _WorkerLost(f"the stepper's worker ended before {what} (exit code "
                                  f"{self.process.exitcode})") from None
        if not self.conn.poll(limit):
            self.close()
            raise ctl.WallTimeExceeded(f"{what} exceeded {limit:g} s; its worker was "
                                       "terminated")
        try:
            kind, value = self.conn.recv()
        except (EOFError, OSError):
            self.process.join(1.0)
            raise _WorkerLost(f"the stepper's worker ended during {what} (exit code "
                              f"{self.process.exitcode})") from None
        if kind == "error":
            raise value
        return value

    def close(self) -> None:
        self._finalizer()


def _stop(process, conn) -> None:
    """End a worker: ask it to stop, then terminate it, then kill it, and join it."""
    try:
        conn.send(("close",))
    except (OSError, ValueError):
        pass
    process.join(0.5)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
    if process.is_alive():
        process.kill()
        process.join()
    conn.close()
