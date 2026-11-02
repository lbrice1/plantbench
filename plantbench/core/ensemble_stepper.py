"""Configured plants advanced one interval at a time, in lockstep, for a controller that
acts on all of them at once.

A policy that acts on many closed loops, the members of a set of starts, scenarios or
training environments, evaluates its action once on the batch of their observations.
`EnsembleStepper` advances the members over each interval together, by the lockstep
Rodas5P integrator of `core.ensemble`, from the state each member reached: one batch of
right-hand sides per stage for every member still running, each member with its own
step size and error control.

Each member is a `Stepper` of its own, which decides what the scalar stepper decides:
which inputs and set-points are owned and how, the nominal action, the noise of its
samples, from the generator its configuration seeds, and its observations.  Only the
integration of the interval differs: Rodas5P rather than LSODA, so that a member agrees
with a scalar stepper of the same configuration to the tolerance, not to the last bit.
The members must share a design object and the layout of their structures, as the
members of an ensemble run do (`core.case.run_setups`); `from_configs` builds them so.

A member whose integration fails is reported with its error and left where its interval
began; it takes no further steps until `reset`.  The others go on.  With a time limit,
the members are integrated in one worker process, and a step that runs over terminates
it: every member of that worker is then lost, and the ensemble stepper invalidated.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from plantbench.backend import check as check_device

from . import control as ctl
from .case import Case, Setup, build_on, design_key
from .ensemble import integrate_ensemble
from .stepper import Step, StepFailed, Stepper, _importable, _portable, _Worker, _WorkerLost


@dataclass
class EnsembleStep:
    """The members at the end of an interval, or at a sample (`EnsembleStepper.observe`):
    `steps[i]` the `Step` of member i, None for a member that has failed, and `errors[i]`
    its failure, None for a member still running.  `y`, `y_true`, `u`, `y_min` and
    `y_max` hold the same values as arrays over the members, NaN for a failed one, for a
    policy that acts on the batch."""

    t: float
    steps: list[Step | None]
    errors: list[Exception | None]

    def _stack(self, attr: str) -> dict[str, np.ndarray]:
        names = next((list(getattr(s, attr)) for s in self.steps if s is not None), [])
        return {k: np.array([np.nan if s is None else getattr(s, attr)[k]
                             for s in self.steps]) for k in names}

    @property
    def y(self) -> dict[str, np.ndarray]:
        return self._stack("y")

    @property
    def y_true(self) -> dict[str, np.ndarray]:
        return self._stack("y_true")

    @property
    def u(self) -> dict[str, np.ndarray]:
        return self._stack("u")

    @property
    def y_min(self) -> dict[str, np.ndarray]:
        return self._stack("y_min")

    @property
    def y_max(self) -> dict[str, np.ndarray]:
        return self._stack("y_max")

    @property
    def running(self) -> np.ndarray:
        """Whether each member is still running."""
        return np.array([e is None for e in self.errors])


class _Owned:
    """A member's disturbance with its owned inputs set over it, as `Stepper._evaluate`
    sets them, in the form the ensemble integrator takes; it declares the disturbance's
    breakpoints."""

    def __init__(self, disturbance, owned: dict[str, float]):
        self.disturbance, self.owned = disturbance, owned
        self.times = tuple(getattr(disturbance, "times", ()))

    def __call__(self, t, u):
        base = u if self.disturbance is None else self.disturbance(t, u)
        return replace(base, **self.owned)


class EnsembleStepper:
    """Configured plants (`setups`) stepped together at a fixed interval `dt`, each with
    the `inputs` and the loops' `setpoints` it names owned by the caller.

    Every member is a `Stepper` with the same arguments (`members`), which checks the
    ownership, holds the member's state, time and noise, and makes its observations.  The
    setups must share one design object and the layout of their structures once the
    owned inputs are removed; `from_configs` builds them so.  The design must declare a
    batched right-hand side (`control.supports_batch`).

    An action maps each owned name to one value per member, or to one value for all; a
    failed member's value is not read.  The integration of an interval is the lockstep
    Rodas5P of `core.ensemble`, on `device`, which restarts at every breakpoint of a
    member's disturbance and schedules.  `extrema` and `refine` are those of `Stepper`,
    with one difference: the values are taken at `refine` equal parts of the whole
    interval, the same for every member so that a member's integration does not depend on
    the others', and on both sides of each of the member's breakpoints.

    `step_budget` and `episode_budget` are those of `Stepper`, for the whole ensemble: one
    worker process integrates every member, and a step that runs over terminates it.  All
    the members are then lost, and the ensemble stepper cannot be used again.
    """

    def __init__(self, setups: Sequence[Setup], dt: float, inputs=(), setpoints=(),
                 rtol: float = 1e-8, atol: float = 1e-10, extrema=(), refine: int = 10,
                 device: str = "cpu", step_budget: float | None = None,
                 episode_budget: float | None = None, startup_budget: float | None = None):
        setups = list(setups)
        if not setups:
            raise ValueError("an ensemble stepper needs at least one member")
        design = setups[0].design
        if any(s.design is not design for s in setups):
            raise ValueError("the members must share one design object; build them with "
                             "`EnsembleStepper.from_configs` or `core.case.build_on`")
        if not ctl.supports_batch(design):
            raise ValueError(f"{type(design).__name__} does not declare a batched "
                             "right-hand side (`batched = True`), which an ensemble needs")
        self.members = [Stepper(s, dt, inputs, setpoints, rtol, atol, extrema=extrema,
                                refine=refine) for s in setups]
        if len({ctl.layout_signature(m.structure) for m in self.members}) > 1:
            raise ValueError("the members' structures differ in their elements; an ensemble "
                             "steps members of one layout (`control.layout_signature`)")
        for name, budget in (("step_budget", step_budget), ("episode_budget", episode_budget),
                             ("startup_budget", startup_budget)):
            if budget is not None and not budget > 0.0:
                raise ValueError(f"{name} must be positive seconds, or None")
        first = self.members[0]
        self.design, self.dt, self.rtol, self.atol = design, first.dt, rtol, atol
        self.inputs, self.setpoints = first.inputs, first.setpoints
        self.extrema, self.refine = first.extrema, first.refine
        self.device = check_device(device)
        self.step_budget, self.episode_budget = step_budget, episode_budget
        self._invalid: str | None = None
        self._worker: _Worker | None = None
        self.reset()
        if step_budget is not None or episode_budget is not None:
            case = setups[0].case
            module, attr = _importable(case)
            self._worker = _Worker(self, _serve, (
                module, attr, [s.config.to_dict() for s in setups], self.dt, self.inputs,
                self.setpoints, rtol, atol, self.extrema, self.refine, self.device),
                startup_budget)
            built = self._worker.ready
            if not all(np.array_equal(x, design.x) and np.array_equal(x0, s.x0)
                       for (x, x0), s in zip(built, setups)):
                self.close()
                raise ValueError("the worker's plants differ from the setups: an ensemble "
                                 "stepper with a time limit needs setups that "
                                 "`from_configs` makes from their configurations")

    @classmethod
    def from_configs(cls, case: Case, configs: Sequence, dt: float, **kwargs) -> EnsembleStepper:
        """Members built from `configs` of `case` on one design, which their design keys
        must share (`core.case.design_key`); keyword arguments go to the constructor."""
        configs = [case.validate(c) for c in configs]
        keys = {design_key(c) for c in configs}
        if len(keys) > 1:
            raise ValueError("the configurations need different designs (their structures, "
                             "options or params differ); an ensemble steps members of one")
        design = case.make_design(configs[0])
        return cls([build_on(case, design, c) for c in configs], dt, **kwargs)

    def __len__(self) -> int:
        return len(self.members)

    @property
    def t(self) -> float:
        return self._t

    def reset(self, noise_seeds: Sequence[int | None] | None = None) -> None:
        """Every member back to its start (`Stepper.reset`), the failed ones included, its
        noise seeded by its configuration or by `noise_seeds[i]`."""
        seeds = [None] * len(self.members) if noise_seeds is None else list(noise_seeds)
        if len(seeds) != len(self.members):
            raise ValueError("one noise seed (or None) is needed per member")
        for m, seed in zip(self.members, seeds):
            m.reset(seed)
        self.errors: list[Exception | None] = [None] * len(self.members)
        self._t = 0.0
        self.spent = 0.0

    def nominal_action(self) -> dict[str, np.ndarray]:
        """Each member's nominal action (`Stepper.nominal_action`), as arrays over them."""
        actions = [m.nominal_action() for m in self.members]
        return {k: np.array([a[k] for a in actions]) for k in actions[0]}

    def _actions(self, action: Mapping[str, Any] | None) -> list[dict[str, float]]:
        if action is None:
            return [m.nominal_action() for m in self.members]
        expected = {*self.inputs, *self.setpoints}
        if set(action) != expected:
            raise KeyError(f"an action names exactly {sorted(expected)}, not {sorted(action)}")
        M = len(self.members)
        values = {}
        for k, v in action.items():
            v = np.asarray(v, dtype=float)
            if v.ndim == 0:
                v = np.full(M, float(v))
            if v.shape != (M,):
                raise ValueError(f"the action on {k!r} has shape {v.shape}; it is one value "
                                 f"per member, ({M},), or one for all")
            values[k] = v
        return [{k: float(v[i]) for k, v in values.items()} for i in range(M)]

    def step(self, action: Mapping[str, Any] | None = None) -> EnsembleStep:
        """Hold each member's action over one interval and advance the members that are
        still running together.  A member whose integration fails gets a `StepFailed` in
        `errors` and stays where the interval began."""
        if self._invalid is not None:
            raise RuntimeError(self._invalid)
        actions = self._actions(action)
        rows = [i for i, e in enumerate(self.errors) if e is None]
        start = time.perf_counter()
        try:
            if not rows:
                results = []
            elif self._worker is None:
                results = _advance(self.members, rows, self._t, actions, self.device)
            else:
                results = self._advance_in_worker(rows, actions)
        finally:
            self.spent += time.perf_counter() - start
        done: dict[int, int] = {}
        for i, r in zip(rows, results):
            m = self.members[i]
            if isinstance(r, Exception):
                self.errors[i] = r
                continue
            y, nfev, extremes = r
            m._y, m.t, m.action, m._range = y, m.t + m.dt, actions[i], extremes
            m.nfev += nfev
            m._noise = m._draw()
            done[i] = nfev
        self._t += self.dt
        return EnsembleStep(self._t, [m._observe(done[i]) if i in done else None
                                      for i, m in enumerate(self.members)],
                            list(self.errors))

    def observe(self) -> EnsembleStep:
        """Every member now (`Stepper.observe`), the failed ones where they stopped."""
        return EnsembleStep(self._t, [m.observe() for m in self.members], list(self.errors))

    def _advance_in_worker(self, rows: list[int], actions: list[dict]) -> list:
        limits = [b for b in (self.step_budget,
                              None if self.episode_budget is None
                              else self.episode_budget - self.spent) if b is not None]
        limit = min(limits)
        try:
            if limit <= 0.0:
                raise ctl.WallTimeExceeded(f"the episode's {self.episode_budget:g} s are "
                                           f"spent at t = {self._t:g}")
            return self._worker.call(("step", [self.members[i]._y for i in rows], self._t,
                                      actions, rows), limit, f"the step from t = {self._t:g}")
        except (ctl.WallTimeExceeded, _WorkerLost) as exc:
            self._invalid = (f"this ensemble stepper was stopped ({exc}); every member of "
                             "its worker is lost and it cannot step again; build a new one")
            self.close()
            if isinstance(exc, _WorkerLost):
                raise RuntimeError(str(exc)) from None
            raise

    def close(self) -> None:
        """End the worker process of an ensemble stepper with a time limit."""
        if self._worker is not None:
            self._worker.close()

    def __enter__(self) -> EnsembleStepper:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _advance(members: list[Stepper], rows: list[int], t: float, actions: list[dict],
             device: str) -> list:
    """One interval from t for the members `rows`, in lockstep: per member, its state at
    the end, the rows of right-hand side it evaluated and the extremes of its `extrema`
    measurements, or the `StepFailed` that stopped it."""
    first = members[rows[0]]
    dt, extrema, refine = first.dt, first.extrema, first.refine
    a, b = t, t + dt
    S = [members[i]._structure_with(actions[i]) for i in rows]
    owned = [{f: actions[i][f] for f in members[i].inputs} for i in rows]
    D = [members[i].disturbance if not o else _Owned(members[i].disturbance, o)
         for i, o in zip(rows, owned)]
    r = integrate_ensemble(first.design, S, D, b, refine + 1 if extrema else 2, first.rtol,
                           first.atol, device=device, t0=a,
                           y0=[members[i]._y for i in rows])
    out: list = []
    for j, i in enumerate(rows):
        if r.status[j] != "ok":
            out.append(StepFailed(f"integration failed on [{a:g}, {b:g}]: {r.message[j]}"))
            continue
        Y = r.Y[j]
        if not np.isfinite(Y).all():
            out.append(StepFailed(f"non-finite state on [{a:g}, {b:g}]"))
            continue
        extremes: tuple[dict, dict] = ({}, {})
        if extrema:
            extremes = _extremes(members[i], S[j], owned[j], r.t, Y, r.at_breaks[j], a, b)
        out.append((Y[-1].copy(), int(r.nfev[j]), extremes))
    return out


def _extremes(m: Stepper, S, owned: dict, grid, Y, at_breaks: dict, a: float, b: float
              ) -> tuple[dict, dict]:
    """The least and the largest true value of each of `m.extrema` over [a, b]: at the
    grid's points before b, on both sides of each breakpoint inside the interval, and at
    b with the inputs just before it and at it."""
    points = [(tk, yk) for tk, yk in zip(grid[:-1], Y[:-1])]
    for c, yc in at_breaks.items():
        if a < c < b:
            points += [(np.nextafter(c, a), yc), (c, yc)]
    points += [(np.nextafter(b, a), Y[-1]), (b, Y[-1])]
    n = len(m.design.x)
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for tk, yk in points:
        u, _ = m._evaluate(tk, yk, S, owned)
        for name in m.extrema:
            v = float(m.measurements[name](yk[:n], u, m.design.pp))
            lo[name] = min(lo.get(name, v), v)
            hi[name] = max(hi.get(name, v), v)
    return lo, hi


def _serve(conn, module: str, attr: str, configs: list, dt: float, inputs, setpoints,
           rtol: float, atol: float, extrema, refine: int, device: str) -> None:
    """The worker: build the members on one design, report their design and starts, then
    advance intervals on request until told to stop or the connection closes."""
    try:
        case = getattr(importlib.import_module(module), attr)
        ens = EnsembleStepper.from_configs(case, configs, dt, inputs=inputs,
                                           setpoints=setpoints, rtol=rtol, atol=atol,
                                           extrema=extrema, refine=refine, device=device)
    except BaseException as exc:  # noqa: BLE001 -- reported to the parent
        conn.send(("error", _portable(exc)))
        return
    conn.send(("ready", [(m.design.x, m.setup.x0) for m in ens.members]))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message[0] == "close":
            return
        _, ys, t, actions, rows = message
        try:
            for i, y in zip(rows, ys):
                ens.members[i]._y = y
            reply = ("ok", [r if not isinstance(r, Exception) else _portable(r)
                            for r in _advance(ens.members, rows, t, actions, device)])
        except Exception as exc:  # noqa: BLE001 -- reported to the parent
            reply = ("error", _portable(exc))
        conn.send(reply)
