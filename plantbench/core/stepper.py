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
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp

from . import control as ctl
from .case import Setup, _check_inputs, _check_undisturbed


class StepFailed(RuntimeError):
    """The integration over an interval failed or reached a non-finite state.  The stepper
    is left where the interval began."""


@dataclass
class Step:
    """The plant at the end of an interval: time, plant state, inputs (as `control.
    input_values` flattens them), named measurements, the signals of the remaining loops
    (`control.apply`), and the right-hand-side evaluations the interval took."""

    t: float
    x: np.ndarray
    u: dict[str, float]
    y: dict[str, float]
    c: dict[str, float]
    nfev: int


class Stepper:
    """A configured plant (`plantbench.build`) stepped at a fixed interval `dt`, with the
    `inputs` and the loops' `setpoints` it names owned by the caller.

    The run starts where the configuration starts it, at `setup.x0` with the augmented
    states of `control.initial_augmented`, at time 0.  `displaced` holds the elements the
    owned inputs removed, as `Loop` and `Ratio` objects, and `structure` the elements that
    remain.  Refused: an input the design does not have or that is not a scalar, an input
    the configuration holds or a disturbance changes, a set-point of a loop that is removed
    or whose set-point a cascade primary writes, and a set-point that carries a schedule.
    """

    def __init__(self, setup: Setup, dt: float, inputs=(), setpoints=(),
                 rtol: float = 1e-8, atol: float = 1e-10):
        if not dt > 0.0:
            raise ValueError("the interval dt must be positive")
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
        self.reset()

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

    def reset(self) -> None:
        """Back to the start of the run, at time 0, with the nominal action."""
        x0 = np.asarray(self.setup.x0, dtype=float)
        self._y = np.concatenate([x0, ctl.initial_augmented(self.structure, self.design, x0)])
        self.t = 0.0
        self.nfev = 0
        self.action = self.nominal_action()

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
        action = self.nominal_action() if action is None else self._check_action(action)
        S = self._structure_with(action)
        owned = {f: action[f] for f in self.inputs}
        a, b = self.t, self.t + self.dt
        edges = [a, *(c for c in self.breakpoints if a < c < b), b]
        y, nfev = self._y, 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            sol = solve_ivp(self._rhs, (lo, hi), y, args=(S, owned), method="LSODA",
                            t_eval=[hi], rtol=self.rtol, atol=self.atol)
            if not sol.success:
                raise StepFailed(f"integration failed on [{lo:g}, {hi:g}]: {sol.message}")
            nfev += int(sol.nfev)
            y = sol.y[:, -1]
            if not np.isfinite(y).all():
                raise StepFailed(f"non-finite state at t = {hi:g}")
        self._y, self.t, self.action = y, b, action
        self.nfev += nfev
        return self._observe(nfev)

    def observe(self) -> Step:
        """The plant now, under the action last held (the nominal one before any step)."""
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
        return Step(self.t, x, values, y, {k: float(v) for k, v in signals.items()}, nfev)
