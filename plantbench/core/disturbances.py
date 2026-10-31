"""Disturbances as functions of time acting on the inputs.

A disturbance is a function `d(t, u0) -> u` that returns the inputs in force at time t
given the nominal inputs u0.  It must return u0 itself, or a modified copy, and never
change u0 in place, because the simulator calls it with the same nominal inputs at every
evaluation.  A disturbance that changes the inputs discontinuously declares the times it
does so as an attribute `times`, and the simulator restarts the integration there.  Everything here works on any inputs dataclass, so a case needs its own
disturbance functions only where the change is not a change of one input.

`combine` hands each disturbance the inputs the earlier ones produced, so two fractional
steps of one input compound.  A `sequence` does not: each of its levels is taken relative
to the inputs the sequence is handed, which in a run are the nominal ones.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

import numpy as np

Disturbance = Callable[[float, Any], Any]


def _target(u0, field: str, fraction: float | None, value: float | None) -> float:
    if (fraction is None) == (value is None):
        raise ValueError("give exactly one of fraction and value")
    if not hasattr(u0, field):
        raise AttributeError(f"the inputs have no field {field!r}")
    return getattr(u0, field) * (1.0 + fraction) if value is None else value


def step(field: str, fraction: float | None = None, value: float | None = None,
         t: float = 30.0) -> Disturbance:
    """Move one input at time t, by a fraction of its nominal value or to a value."""

    def d(time: float, u0):
        if time < t:
            return u0
        return replace(u0, **{field: _target(u0, field, fraction, value)})

    d.times = (float(t),)
    return d


def ramp(field: str, fraction: float | None = None, value: float | None = None,
         t: float = 30.0, duration: float = 60.0) -> Disturbance:
    """Move one input linearly from its nominal value over [t, t + duration]."""
    if duration <= 0:
        raise ValueError("a ramp needs a positive duration; use step for an instant change")

    def d(time: float, u0):
        if time < t:
            return u0
        start = getattr(u0, field)
        end = _target(u0, field, fraction, value)
        w = min((time - t) / duration, 1.0)
        return replace(u0, **{field: start + w * (end - start)})

    d.times = (float(t), float(t + duration))  # the ramp's slope changes at both ends
    return d


def combine(*disturbances: Disturbance) -> Disturbance:
    """Apply several disturbances in order; later ones see the earlier ones' changes."""

    def d(time: float, u0):
        u = u0
        for dist in disturbances:
            u = dist(time, u)
        return u

    d.times = tuple(sorted({t for dist in disturbances for t in getattr(dist, "times", ())}))
    return d


def sequence(parts: Sequence[Disturbance], times: Sequence[float]) -> Disturbance:
    """Each of `parts` in force from its time in `times` until the next one's, applied to
    the inputs the sequence is handed, so that consecutive levels do not compound; the
    inputs unchanged before the first time.  Each part is typically a step at its own time.
    The breakpoints are those the parts declare."""
    times = tuple(float(t) for t in times)
    if len(parts) != len(times) or not parts:
        raise ValueError("a sequence needs one time per part, and at least one part")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("the times of a sequence must increase")

    def d(time: float, u0):
        k = int(np.searchsorted(times, time, side="right"))
        return u0 if k == 0 else parts[k - 1](time, u0)

    d.times = tuple(sorted({*times, *(t for p in parts for t in getattr(p, "times", ()))}))
    return d


def step_sequence(field: str, times: Sequence[float], levels: Sequence[float],
                  kind: str = "fraction") -> Disturbance:
    """One input moved at each of `times` to the matching level, a fraction of its nominal
    value (`kind="fraction"`) or a value (`kind="value"`)."""
    if kind not in ("fraction", "value"):
        raise ValueError(f"kind must be 'fraction' or 'value', not {kind!r}")
    if len(levels) != len(times):
        raise ValueError("a sequence needs one level per time")
    d = sequence([step(field, t=t, **{kind: float(v)}) for t, v in zip(times, levels)], times)
    d.levels = tuple(float(v) for v in levels)
    return d


LEVEL_KINDS = ("uniform", "binary")


def random_levels(seed: int, low: float, high: float, dwell: Sequence[float],
                  start: float, end: float, kind: str = "uniform"
                  ) -> tuple[list[float], list[float]]:
    """Switching times and levels of a random piecewise-constant signal, as identification
    data are excited with: the first switch at `start`, each next one after a dwell drawn
    uniformly in [dwell[0], dwell[1]], the last before `end`.  Each level is drawn uniformly
    in [low, high] (`kind="uniform"`, an amplitude-modulated pseudo-random binary signal)
    or is one of the two bounds (`kind="binary"`).  The same seed gives the same signal."""
    if kind not in LEVEL_KINDS:
        raise ValueError(f"kind must be one of {LEVEL_KINDS}, not {kind!r}")
    lo, hi = float(dwell[0]), float(dwell[1])
    if not 0.0 < lo <= hi:
        raise ValueError("dwell is [shortest, longest], with 0 < shortest <= longest")
    if not low <= high or not start < end:
        raise ValueError("a random sequence needs low <= high and start < end")
    rng = np.random.default_rng(seed)
    times, levels = [], []
    t = float(start)
    while t < end:
        times.append(t)
        levels.append(float(rng.uniform(low, high)) if kind == "uniform"
                      else float((low, high)[rng.integers(2)]))
        t += float(rng.uniform(lo, hi))
    return times, levels
