"""Measurement and actuator non-idealities, ideal by default.

The plant of `plant.py` measures instantaneously and exactly, and its valves move to
whatever the controller asks for in the same instant.  Neither is true of a plant.  This
module supplies the two objects that sit between a controller and the process --- an
`Instrument` on the way in and an `Actuator` on the way out --- so that deadtime, sensor
lag, noise and valve dynamics can be added without rewriting the control layer.

Both are ideal when default-constructed, and an ideal element carries no state, so a
structure built entirely from defaults has exactly one augmented state per element, as
before, and reproduces the published results digit for digit.  Adding a non-ideality is
a deliberate act.

Deadtime is represented by a Pade approximation rather than a history buffer.  A buffer
is exact, but it makes the closed loop a delay-differential system: `solve_ivp` no longer
applies, and the finite-difference Jacobian behind `controllers.closed_loop_spectrum` has
nothing to differentiate, so the damping ratios that the design comparison rests on stop
being computable.  A Pade approximation is a few linear states, which the existing stiff
integrator and the existing linearization both handle unchanged.  The price is the
undershoot every Pade approximation shows on a step, and `method="lags"` offers a chain
of first-order lags instead for cases where that matters more than phase accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plantbench.backend import clamp, namespace

# Below this, a time constant or a deadtime is treated as absent rather than as a very
# fast state, which would make the system needlessly stiff.
NEGLIGIBLE = 1e-12


@dataclass(frozen=True)
class Instrument:
    """The measurement chain between the process and the controller.

    The signal passes through the lag first, then the deadtime.

    `noise_std`, `bias` and `quantum` are declared and rejected rather than silently
    ignored.  Noise is the one that needs a decision rather than an afternoon: to change
    anything it has to enter the control law, but an adaptive-step solver evaluates the
    right-hand side at trial times it goes on to reject, so drawing a random number inside
    it makes the trajectory a function of the solver's internal stepping.  Doing it
    properly means a controller sampled at a fixed interval, which is a different control
    layer from the continuous one here.  They are named now so that the state layout does
    not change when they are implemented.
    """

    tau: float = 0.0  # first-order sensor lag, min
    deadtime: float = 0.0  # transport or analyzer delay, min
    method: str = "pade"  # "pade" or "lags", how the deadtime is represented
    order: int = 1  # Pade order (1 or 2), or the number of lags in the chain
    noise_std: float = 0.0  # not yet implemented
    bias: float = 0.0  # not yet implemented
    quantum: float = 0.0  # not yet implemented

    def __post_init__(self) -> None:
        if self.method not in ("pade", "lags"):
            raise ValueError(f"unknown deadtime method {self.method!r}")
        if self.method == "pade" and self.deadtime > NEGLIGIBLE and self.order not in (1, 2):
            raise ValueError("Pade deadtime is implemented for order 1 and 2")
        if self.order < 1:
            raise ValueError("order must be at least 1")
        if self.noise_std != 0.0 or self.bias != 0.0 or self.quantum != 0.0:
            raise NotImplementedError(
                "noise, bias and quantization are declared but not implemented; see the "
                "class docstring for what has to be decided first"
            )

    @property
    def is_ideal(self) -> bool:
        """True when the instrument reports the process value unchanged."""
        return (
            self.tau <= NEGLIGIBLE
            and self.deadtime <= NEGLIGIBLE
            and self.noise_std == 0.0
            and self.bias == 0.0
            and self.quantum == 0.0
        )

    @property
    def n_lag_states(self) -> int:
        return 1 if self.tau > NEGLIGIBLE else 0

    @property
    def n_delay_states(self) -> int:
        if self.deadtime <= NEGLIGIBLE:
            return 0
        return self.order

    @property
    def n_states(self) -> int:
        return self.n_lag_states + self.n_delay_states

    def initial(self, pv: float) -> np.ndarray:
        """States that hold the reported value at `pv`, so a run starts at rest.

        The Pade realizations carry a direct feedthrough, so their states are not simply
        the steady output; they are the values for which the derivative vanishes and the
        output equals the input.
        """
        s = []
        if self.n_lag_states:
            s.append(pv)
        if self.n_delay_states:
            th = self.deadtime
            if self.method == "lags":
                s.extend([pv] * self.order)
            elif self.order == 1:
                s.append(pv * th / 2.0)  # z = v theta / 2 gives y = v
            else:
                s.extend([pv * th * th / 12.0, 0.0])  # z1 = v theta^2/12, z2 = 0
        return np.asarray(s, dtype=float)

    def derivatives(self, states: np.ndarray, pv: float) -> np.ndarray:
        """Time derivatives of the instrument states given the true process value.

        For a batch, `states` is (m, n_states) and `pv` (m,)."""
        xp = namespace(states)
        if self.n_states == 0:
            return xp.zeros(np.shape(pv) + (0,))
        d = xp.empty(np.shape(pv) + (self.n_states,))
        i = 0
        v = pv
        if self.n_lag_states:
            d[..., 0] = (pv - states[..., 0]) / self.tau
            v = states[..., 0]  # the deadtime sees the lagged signal
            i = 1
        if self.n_delay_states:
            z = states[..., i:]
            th = self.deadtime
            if self.method == "lags":
                k = self.order / th
                prev = v
                for j in range(self.order):
                    d[..., i + j] = (prev - z[..., j]) * k
                    prev = z[..., j]
            elif self.order == 1:
                d[..., i] = -(2.0 / th) * z[..., 0] + v
            else:
                d[..., i] = z[..., 1]
                d[..., i + 1] = -(12.0 / (th * th)) * z[..., 0] - (6.0 / th) * z[..., 1] + v
        return d

    def output(self, states: np.ndarray, pv: float) -> float:
        """The value the controller sees."""
        if self.n_states == 0:
            return pv
        i = 0
        v = pv
        if self.n_lag_states:
            v = states[..., 0]
            i = 1
        if self.n_delay_states:
            z = states[..., i:]
            th = self.deadtime
            if self.method == "lags":
                v = z[..., -1]
            elif self.order == 1:
                v = -v + (4.0 / th) * z[..., 0]
            else:
                v = v - (12.0 / th) * z[..., 1]
        return float(v) if np.ndim(v) == 0 else v


@dataclass(frozen=True)
class Actuator:
    """The valve between the controller output and the process input.

    A first-order travel with an optional slew limit,

        dm/dt = clip((command - m) / tau, -rate_limit, +rate_limit),

    which is continuous in the state, so the stiff integrator and the linearization both
    keep working.  `stiction` and `deadband` are declared but not implemented: both are
    discontinuous, and the model deliberately keeps its right-hand side continuous, so
    adding them means deciding first whether to smooth them or to carry events through
    the integrator.  They are named here so that the state layout does not change when
    that decision is made.
    """

    tau: float = 0.0  # positioner travel time constant, min
    rate_limit: float = np.inf  # maximum |dm/dt| in units of the manipulated variable per min
    stiction: float = 0.0  # not yet implemented
    deadband: float = 0.0  # not yet implemented

    def __post_init__(self) -> None:
        if self.stiction != 0.0 or self.deadband != 0.0:
            raise NotImplementedError(
                "stiction and deadband are declared but not implemented; see the module "
                "docstring for what has to be decided first"
            )
        if np.isfinite(self.rate_limit) and self.tau <= NEGLIGIBLE:
            raise ValueError("a rate limit needs a travel time constant: set tau as well")

    @property
    def is_ideal(self) -> bool:
        """True when the valve takes the controller output immediately."""
        return self.tau <= NEGLIGIBLE and not np.isfinite(self.rate_limit)

    @property
    def n_states(self) -> int:
        return 0 if self.is_ideal else 1

    def initial(self, command: float) -> np.ndarray:
        return np.zeros(0) if self.n_states == 0 else np.asarray([command], dtype=float)

    def derivatives(self, states: np.ndarray, command: float) -> np.ndarray:
        xp = namespace(states)
        if self.n_states == 0:
            return xp.zeros(np.shape(command) + (0,))
        rate = (command - states[..., 0]) / self.tau
        if np.isfinite(self.rate_limit):
            rate = clamp(rate, -self.rate_limit, self.rate_limit)
        return xp.asarray(rate)[..., None]

    def output(self, states: np.ndarray, command: float) -> float:
        if self.n_states == 0:
            return command
        return float(states[0]) if states.ndim == 1 else states[..., 0]


IDEAL_INSTRUMENT = Instrument()
IDEAL_ACTUATOR = Actuator()
