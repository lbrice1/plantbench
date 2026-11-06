"""The stepped map of a configured plant: its fixed point and its linearization.

A stepper holding one action defines a map over one interval, y_{k+1} = F(y_k), on the
plant state and the augmented states of the loops that remain (`control.state_layout`).
An equilibrium of the sampled closed loop is a fixed point y* = F(y*), locally
asymptotically stable when every eigenvalue of dF/dy at y* lies inside the unit circle.
The fixed point is found by `scipy.optimize.fsolve`, a Newton-type method, which finds
unstable fixed points as well as stable ones; the Jacobian is taken by central
differences of F.

F is the stepper's own integration of an interval, at tighter tolerances than a step's
so that the differences resolve its derivative.  A configuration's noise enters the
samples and never the plant, so it has no part in F; the stepper itself is left as it
was, its time, state and noise generator untouched.  F depends on the time unless the
configuration has neither a disturbance nor a set-point schedule, so a stepper with
either is refused.  The integrals of ratio stations, which never change, are left out of
F's coordinates, as `control.closed_loop_spectrum` removes their structural zeros.

With nothing owned, or with owned inputs and set-points held at constant values, the
loops that remain act continuously within the interval, and the eigenvalues of F are
exp(lambda dt) of the continuous closed loop's eigenvalues lambda (`Setup.spectrum` when
nothing is owned).  A controller sampled at dt that keeps a memory of its own, a sampled
PI loop say, makes the state of the map larger than the plant's and the loops', and the
equality does not hold for it; analyzing such a controller needs its memory in the map,
which this module does not provide.  At a fixed point where a valve sits at a limit the
closed loop is not differentiable: the differences then average its two sides, and the
eigenvalues describe neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import fsolve

from . import control as ctl


class StepMap:
    """F, the closed loop of `stepper` over one interval with `action` held (the nominal
    one when None), on the coordinates `names`; `start` is the design point in them, the
    plant at its design state and the loops at rest there."""

    def __init__(self, stepper, action: Mapping[str, float] | None = None,
                 rtol: float = 1e-11, atol: float = 1e-12):
        if stepper.disturbance is not None or stepper.breakpoints:
            raise ValueError("the stepped map depends on time under a disturbance or a "
                             "set-point schedule; it is analyzed without either")
        self.stepper, self.dt, self.rtol, self.atol = stepper, stepper.dt, rtol, atol
        self.action = (stepper.nominal_action() if action is None
                       else stepper._check_action(action))
        design, S = stepper.design, stepper.structure
        n = len(design.x)
        self._full = np.concatenate([design.x, ctl.initial_augmented(S, design)])
        ratios = [n + k for k, e in enumerate(S) if isinstance(e, ctl.Ratio)]
        self.free = np.setdiff1d(np.arange(len(self._full)), ratios)
        names = [*stepper.setup.case.state_names(design), *ctl.augmented_names(S)]
        self.names = [names[k] for k in self.free]
        self.start = self._full[self.free].copy()

    def __call__(self, y: np.ndarray) -> np.ndarray:
        full = self._full.copy()
        full[self.free] = y
        end, _, _ = self.stepper._advance(full, 0.0, self.action, rtol=self.rtol,
                                          atol=self.atol, extrema=())
        return end[self.free]


def fixed_point(F: StepMap, y0: np.ndarray | None = None,
                tol: float = 1e-8) -> tuple[np.ndarray, float]:
    """A fixed point of F from `y0` (the design point by default), and its residual, the
    largest |F(y) - y| relative to max(|y|, 1).  Raises if the residual exceeds `tol`."""
    y0 = F.start if y0 is None else np.asarray(y0, dtype=float)
    scale = np.maximum(np.abs(y0), 1.0)
    w, _, _, message = fsolve(lambda w: (F(w * scale) - w * scale) / scale, y0 / scale,
                              xtol=1e-12, full_output=True)
    y = w * scale
    residual = float(np.max(np.abs(F(y) - y) / np.maximum(np.abs(y), 1.0)))
    if not residual <= tol:
        raise RuntimeError(f"no fixed point found: the residual is {residual:.3g}, above "
                           f"{tol:g} ({message})")
    return y, residual


def jacobian(F: StepMap, y: np.ndarray, rel: float = 1e-5) -> np.ndarray:
    """dF/dy at y by central differences, each of rel * max(|y_i|, 1)."""
    h = rel * np.maximum(np.abs(y), 1.0)
    J = np.empty((len(y), len(y)))
    for i in range(len(y)):
        dy = np.zeros_like(y)
        dy[i] = h[i]
        J[:, i] = (F(y + dy) - F(y - dy)) / (2.0 * h[i])
    return J


@dataclass
class Linearization:
    """F linearized at its fixed point `y`, in the coordinates `names`.

    `eigenvalues` are those of the Jacobian `J`, largest modulus first, and `vectors` their
    eigenvectors as columns, in the same order.  `residual` is that of the fixed point."""

    y: np.ndarray
    names: list[str]
    residual: float
    J: np.ndarray
    eigenvalues: np.ndarray
    vectors: np.ndarray
    dt: float

    @property
    def spectral_radius(self) -> float:
        return float(np.abs(self.eigenvalues).max())

    @property
    def stable(self) -> bool:
        """Whether every eigenvalue lies inside the unit circle."""
        return self.spectral_radius < 1.0

    @property
    def continuous(self) -> np.ndarray:
        """log(mu) / dt for each eigenvalue mu: the continuous-time eigenvalue it would be
        exp(lambda dt) of, on the principal branch of the logarithm."""
        return np.log(self.eigenvalues.astype(complex)) / self.dt

    def slowest_stable(self) -> int | None:
        """The index of the stable mode of largest modulus, None if no mode is stable."""
        stable = np.flatnonzero(np.abs(self.eigenvalues) < 1.0)
        return int(stable[0]) if stable.size else None

    @property
    def decay_time(self) -> float:
        """-dt / ln|mu| of the slowest stable mode: the time in which it decays by a
        factor e.  NaN if no mode is stable."""
        k = self.slowest_stable()
        return float("nan") if k is None else float(-self.dt / np.log(abs(self.eigenvalues[k])))

    def shares(self, mode: int = 0, scale: Mapping[str, float] | None = None
               ) -> dict[str, float]:
        """The share of each coordinate in the eigenvector of `mode` (by default the one
        of largest modulus), |v_i / s_i|^2 over their sum.  The shares depend on the
        scales s_i: those of `scale` by name, a span say, and max(|y_i|, 1) for the
        others, the scale of the differences."""
        s = np.maximum(np.abs(self.y), 1.0)
        for name, value in (scale or {}).items():
            s[self.names.index(name)] = float(value)
        w = np.abs(self.vectors[:, mode] / s) ** 2
        return dict(zip(self.names, (w / w.sum()).tolist()))


def linearize(stepper, action: Mapping[str, float] | None = None,
              y0: np.ndarray | None = None, rel: float = 1e-5, rtol: float = 1e-11,
              atol: float = 1e-12) -> Linearization:
    """The fixed point of the stepped map of `stepper` under `action`, from `y0`, and the
    map linearized there (`StepMap`, `fixed_point`, `jacobian`)."""
    F = StepMap(stepper, action, rtol, atol)
    y, residual = fixed_point(F, y0)
    J = jacobian(F, y, rel)
    mu, V = np.linalg.eig(J)
    order = np.argsort(-np.abs(mu), kind="stable")
    return Linearization(y, F.names, residual, J, mu[order], V[:, order], F.dt)
