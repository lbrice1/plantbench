"""Operating limits of a case's measurements, and a monitor of the outcomes they define.

A case declares, for each measurement it bounds, the transmitter's span, alarm and trip
limits, and the measurement's values at the plant's open-loop steady states where it
has several (`Case.limits`).  Each number is tagged as the case cards tag theirs:
"source" when the case's reference gives it, "choice" when the case decides it, with the
reason.

What makes a limit an outcome is data as well.  A `Rule` names a measurement, a level,
the side of it that counts and a dwell, the time the measurement must stay beyond the
level without a break; `limit_rules` makes rules from a case's limits.  A `Monitor`
applies rules to the samples of a run, the steps of a `Stepper` or the grid of a
`Trajectory` (`check`), and records the first time each rule is met.

The samples are all a monitor sees, unless a step reports the extremes of the interval
before it (`Stepper`, `extrema`).  Two consequences follow:

- The time beyond a level is counted from the last sample within it, or from the first
  sample of the run.  The crossing lies after that sample, so the count is an upper
  bound of the time beyond, and a sampled check of a dwell reports the outcome no later
  than the plant reaches it.  An interval whose extremes show the measurement back
  within the level restarts the count at the interval's start.
- A rule without a dwell is met between samples when an interval's extreme lies beyond
  its level while the samples do not.  The violation is then an estimate, made from the
  refined dense output of the step, and is reported as such (`Violation.sampled`); the
  crossing lies in the interval, which the violation gives, not at a time of its own.

The monitor reads true values: `Step.y_true`, not the measured `Step.y` that a
configuration's noise perturbs, since an outcome is a state of the plant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

SOURCE, CHOICE = "source", "choice"
BASES = (SOURCE, CHOICE)
SIDES = ("above", "below")
KINDS = ("alarm", "trip")


def _check_basis(basis: str, what: str) -> None:
    if basis not in BASES:
        raise ValueError(f"the basis of {what} is one of {BASES}, not {basis!r}")


@dataclass(frozen=True)
class Limit:
    """One limit of a measurement, in its units, with its basis: "source", from the case's
    reference, or "choice", decided by the case; `why` says what it rests on."""

    value: float
    basis: str
    why: str = ""

    def __post_init__(self) -> None:
        _check_basis(self.basis, "a limit")
        if not math.isfinite(self.value):
            raise ValueError("a limit is a finite value")


@dataclass(frozen=True)
class Span:
    """A transmitter's span, lo to hi in the units of the measurement, with its basis."""

    lo: float
    hi: float
    basis: str
    why: str = ""

    def __post_init__(self) -> None:
        _check_basis(self.basis, "a span")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo < self.hi):
            raise ValueError(f"a span runs from lo to a larger hi, not {self.lo} to {self.hi}")

    def at(self, fraction: float) -> float:
        """The value at `fraction` of the span, 0 at lo and 1 at hi."""
        return self.lo + fraction * (self.hi - self.lo)

    def fraction(self, value: float) -> float:
        """Where `value` lies in the span, 0 at lo and 1 at hi."""
        return (value - self.lo) / (self.hi - self.lo)


@dataclass(frozen=True)
class Limits:
    """What a case declares about one measurement: its span, its alarm and trip limits
    below and above, any of them None, and the measurement's values at the plant's
    open-loop steady states where the plant has several, in the same order for every
    measurement of the case, which the case states.  A trip limit lies outside the alarm
    limit on the same side."""

    span: Span | None = None
    alarm_lo: Limit | None = None
    alarm_hi: Limit | None = None
    trip_lo: Limit | None = None
    trip_hi: Limit | None = None
    steady_states: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        lo = [x.value for x in (self.trip_lo, self.alarm_lo) if x is not None]
        hi = [x.value for x in (self.alarm_hi, self.trip_hi) if x is not None]
        if lo != sorted(lo) or hi != sorted(hi) or (lo and hi and max(lo) >= min(hi)):
            raise ValueError("limits are ordered trip_lo <= alarm_lo < alarm_hi <= trip_hi")
        object.__setattr__(self, "steady_states",
                           tuple(float(v) for v in self.steady_states))

    def bounds(self, kind: str) -> tuple[Limit | None, Limit | None]:
        """The low and the high limit of `kind`, "alarm" or "trip"."""
        if kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}, not {kind!r}")
        return getattr(self, f"{kind}_lo"), getattr(self, f"{kind}_hi")

    def within(self, value: float, kind: str) -> bool:
        """Whether `value` lies within the limits of `kind`, inclusive."""
        lo, hi = self.bounds(kind)
        return (lo is None or value >= lo.value) and (hi is None or value <= hi.value)


# ------------------------------------------------------------------------------------
# Rules and the monitor
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """The outcome `outcome` is reached when `measurement` lies `side` ("above" or
    "below") `level` for at least `dwell` time units without a break; strictly beyond,
    so a value at the level is within it.  A dwell of zero is met by one sample."""

    outcome: str
    measurement: str
    side: str
    level: float
    dwell: float = 0.0

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side is one of {SIDES}, not {self.side!r}")
        if not math.isfinite(self.level):
            raise ValueError("a rule's level is a finite value")
        if not self.dwell >= 0.0:
            raise ValueError("a rule's dwell is a non-negative time")

    def beyond(self, value: float) -> bool:
        return value > self.level if self.side == "above" else value < self.level


def limit_rules(limits: Mapping[str, Limits], kind: str, outcome: str | None = None,
                dwell: float = 0.0) -> list[Rule]:
    """A rule for each limit of `kind` ("alarm" or "trip") the measurements declare:
    below the low one and above the high one, as `outcome` (`kind` by default) after
    `dwell`."""
    rules = []
    for name, lim in limits.items():
        lo, hi = lim.bounds(kind)
        if lo is not None:
            rules.append(Rule(outcome or kind, name, "below", lo.value, dwell))
        if hi is not None:
            rules.append(Rule(outcome or kind, name, "above", hi.value, dwell))
    return rules


@dataclass(frozen=True)
class Violation:
    """The first time a rule was met: at the sample at `t`, with the measurement at
    `value`, beyond the level since no later than `since`; or, when `sampled` is False,
    estimated from the extremes of the interval from `since` to `t`, `value` being the
    extreme beyond the level."""

    rule: Rule
    t: float
    value: float
    since: float
    sampled: bool = True

    @property
    def outcome(self) -> str:
        return self.rule.outcome


class Monitor:
    """Applies `rules` to the samples of one run, in order of time, and keeps the first
    violation of each rule (`violations`), the earliest of them as `first`."""

    def __init__(self, rules: Iterable[Rule]):
        self.rules = tuple(rules)
        if len(set(self.rules)) != len(self.rules):
            raise ValueError("a rule is given twice")
        self.reset()

    def reset(self) -> None:
        """Back to before the first sample, for a new run."""
        self.violations: dict[Rule, Violation] = {}
        self._since: dict[Rule, float | None] = {r: None for r in self.rules}
        self._t: float | None = None

    @property
    def first(self) -> Violation | None:
        """The earliest violation, ties in the order of the rules; None if none."""
        found = [self.violations[r] for r in self.rules if r in self.violations]
        return min(found, key=lambda v: v.t) if found else None

    @property
    def outcome(self) -> str | None:
        """The outcome of the earliest violation, None if no rule was met."""
        first = self.first
        return None if first is None else first.outcome

    def observe(self, t: float, y: Mapping[str, float],
                y_min: Mapping[str, float] | None = None,
                y_max: Mapping[str, float] | None = None) -> list[Violation]:
        """Take the sample of true values `y` at `t`, with the smallest and largest values
        over the interval since the previous sample where they are known, and return the
        violations met for the first time."""
        t = float(t)
        if self._t is not None and not t > self._t:
            raise ValueError(f"samples come in order of time: {t:g} after {self._t:g}")
        previous, self._t = self._t, t
        start = t if previous is None else previous
        new = []
        for rule in self.rules:
            if rule.measurement not in y:
                raise KeyError(f"rule {rule.outcome!r} reads {rule.measurement!r}, which "
                               f"the sample does not hold; it holds {sorted(y)}")
            value = float(y[rule.measurement])
            lowest = None if y_min is None else y_min.get(rule.measurement)
            highest = None if y_max is None else y_max.get(rule.measurement)
            # The extreme on the side of the level, and the one on the other side.
            far, near = (highest, lowest) if rule.side == "above" else (lowest, highest)
            if not rule.beyond(value):
                self._since[rule] = None
            elif self._since[rule] is None or (near is not None and not rule.beyond(near)):
                self._since[rule] = start
            if rule in self.violations:
                continue
            since = self._since[rule]
            if since is not None and t - since >= rule.dwell:
                found = Violation(rule, t, value, since)
            elif rule.dwell == 0.0 and far is not None and rule.beyond(far):
                found = Violation(rule, t, float(far), start, sampled=False)
            else:
                continue
            self.violations[rule] = found
            new.append(found)
        return new

    def step(self, step) -> list[Violation]:
        """`observe` a `Step` of a stepper: its time, true values and extremes."""
        return self.observe(step.t, step.y_true, step.y_min or None, step.y_max or None)


def check(trajectory, rules: Iterable[Rule]) -> Monitor:
    """A monitor that has observed every point of a `Trajectory`'s grid, each a sample."""
    monitor = Monitor(rules)
    names = sorted({r.measurement for r in monitor.rules})
    missing = [n for n in names if n not in trajectory.y]
    if missing:
        raise KeyError(f"the trajectory has no measurements {missing}; it has "
                       f"{sorted(trajectory.y)}")
    for k, t in enumerate(trajectory.t):
        monitor.observe(t, {n: trajectory.y[n][k] for n in names})
    return monitor
