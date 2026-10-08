"""The batched closed loop: one evaluation at many states, and the Jacobian built from it.

The plant is a two-tank toy written with `x[..., k]`, so that one function serves one
state and a batch; its structure has a cascade, a ratio station, a lagged instrument with
a second-order Pade deadtime and a rate-limited actuator, so that every branch of
`control.apply` is evaluated on a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from _toy import toy_loop

from plantbench import backend
from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
from plantbench.core.casekit import StateLayout, measured
from plantbench.core.control import Loop, Ratio
from plantbench.core.instruments import Actuator, Instrument


@dataclass
class Inputs:
    q1: float = 1.0
    q2: float = 0.5
    d: float = 0.0
    z: np.ndarray = None


@dataclass
class Params:
    a1: float = 0.8
    a2: float = 0.4


@dataclass
class Tanks:
    x: np.ndarray
    u: Inputs
    pp: Params
    batched: ClassVar[bool] = True

    def rhs(self, t, x, u, pp):
        h1, h2 = x[..., 0], x[..., 1]
        d1 = u.q1 + u.d + 0.1 * u.z[..., 0] - pp.a1 * np.sqrt(np.abs(h1))
        d2 = pp.a1 * np.sqrt(np.abs(h1)) + u.q2 - pp.a2 * np.sqrt(np.abs(h2))
        return np.stack([d1, d2], axis=-1)


def _design() -> Tanks:
    u = Inputs(z=np.array([0.2, 0.8]))
    pp = Params()
    h1 = ((u.q1 + 0.1 * u.z[0]) / pp.a1) ** 2
    h2 = ((pp.a1 * np.sqrt(h1) + u.q2) / pp.a2) ** 2
    return Tanks(x=np.array([h1, h2]), u=u, pp=pp)


def _structure(design):
    level2 = lambda x, u, pp: measured(x[..., 1])  # noqa: E731
    level1 = lambda x, u, pp: measured(x[..., 0])  # noqa: E731
    loops = [
        Loop("level 2", level2, "sp:level 1", Kc=-0.3, tau_I=20.0, lo=0.0, hi=50.0,
             instrument=Instrument(tau=0.5, deadtime=1.0, order=2)),
        Loop("level 1", level1, "q1", Kc=-0.4, tau_I=8.0, lo=0.0, hi=5.0,
             actuator=Actuator(tau=0.3, rate_limit=0.5)),
        Ratio("q2 ratio", "q2", "q1"),
    ]
    return ctl.bias_from_design(loops, design)


@pytest.fixture
def plant():
    design = _design()
    return design, _structure(design)


def _states(design, S, m=6, seed=0):
    y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
    rng = np.random.default_rng(seed)
    return y0, y0 + 0.05 * (1.0 + np.abs(y0)) * rng.standard_normal((m, len(y0)))


def test_a_batch_evaluates_as_its_rows_do(plant):
    design, S = plant
    _, Y = _states(design, S)
    Fb = ctl.closed_loop_rhs_batch(3.0, Y, S, design.u, design.pp, None, design.rhs)
    Fs = np.array([ctl.closed_loop_rhs(3.0, y, S, design.u, design.pp, None, design.rhs)
                   for y in Y])
    np.testing.assert_allclose(Fb, Fs, rtol=1e-14, atol=1e-15)


def test_a_batch_sees_the_disturbance_in_force(plant):
    design, S = plant
    _, Y = _states(design, S)
    d = dist.step("d", value=0.2, t=1.0)
    for t in (0.5, 2.0):
        Fb = ctl.closed_loop_rhs_batch(t, Y, S, design.u, design.pp, d, design.rhs)
        Fs = np.array([ctl.closed_loop_rhs(t, y, S, design.u, design.pp, d, design.rhs)
                       for y in Y])
        np.testing.assert_allclose(Fb, Fs, rtol=1e-14, atol=1e-15)


def test_the_batched_jacobian_is_the_forward_difference(plant):
    design, S = plant
    y0, _ = _states(design, S)
    args = (S, design.u, design.pp, None, design.rhs)
    J = ctl.jacobian_batched(0.0, y0, *args)
    f0 = ctl.closed_loop_rhs(0.0, y0, *args)
    for j in range(len(y0)):
        h = ctl._FD_STEP * max(abs(y0[j]), 1.0)
        h = (y0[j] + h) - y0[j]
        y = y0.copy()
        y[j] += h
        np.testing.assert_allclose(J[:, j], (ctl.closed_loop_rhs(0.0, y, *args) - f0) / h,
                                   rtol=1e-12, atol=1e-12)


def test_a_run_with_the_batched_jacobian_agrees_with_lsoda_s_own(plant):
    design, S = plant
    S = ctl.with_setpoint_steps(S, {"level 2": [(5.0, 0.5)]})
    kw = dict(n_points=101, breakpoints=ctl.breakpoints_of(None, S))
    ref = ctl.integrate(design, S, 100.0, **kw)
    new = ctl.integrate(design, S, 100.0, jacobian="batched", **kw)
    assert np.max(np.abs(new.X - ref.X) / (1.0 + np.abs(ref.X))) < 1e-6
    for k in ref.U:
        np.testing.assert_allclose(new.U[k], ref.U[k], rtol=1e-6, atol=1e-8)


def test_the_batched_spectrum_agrees(plant):
    design, S = plant
    a = np.sort_complex(ctl.closed_loop_spectrum(design, S))
    b = np.sort_complex(ctl.closed_loop_spectrum(design, S, jacobian="batched"))
    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-9)


def test_a_design_that_does_not_declare_a_batch_is_refused(design):
    """The toy plant of `_toy` reads `x[0]`, which on a batch is the first member."""
    S = ctl.bias_from_design([toy_loop()], design)
    with pytest.raises(ValueError, match="batched"):
        ctl.integrate(design, S, 10.0, jacobian="batched")
    with pytest.raises(ValueError, match="batched"):
        ctl.closed_loop_spectrum(design, S, jacobian="batched")


def test_options_are_checked(plant):
    design, S = plant
    with pytest.raises(ValueError, match="jacobian"):
        ctl.integrate(design, S, 1.0, jacobian="sparse")
    with pytest.raises(ValueError, match="device"):
        ctl.integrate(design, S, 1.0, jacobian="batched", device="tpu")
    with pytest.raises(ValueError, match="needs jacobian='batched'"):
        ctl.integrate(design, S, 1.0, device="cuda")


def test_batch_inputs_repeats_every_field():
    u = ctl.batch_inputs(Inputs(z=np.array([0.2, 0.8])), 3)
    assert u.q1.shape == (3,) and u.z.shape == (3, 2)
    assert np.all(u.z == [0.2, 0.8])
    u.z[0, 0] = 9.0  # a copy, not a view of one row
    assert u.z[1, 0] == 0.2


def test_a_layout_unpacks_and_packs_a_batch():
    lay = StateLayout([("a", ()), ("b", 3), ("c", (2, 3))])
    X = np.arange(20.0).reshape(2, 10)
    blocks = lay.unpack(X)
    assert blocks["a"].shape == (2,) and blocks["c"].shape == (2, 2, 3)
    np.testing.assert_array_equal(lay.pack(**blocks), X)
    blocks["a"] = 0.0  # a scalar block given once for the whole batch
    np.testing.assert_array_equal(lay.pack(**blocks)[:, 0], [0.0, 0.0])
    assert isinstance(lay.unpack(X[0])["a"], float)


def test_clamp_keeps_the_scalar_path_and_clips_arrays():
    assert backend.clamp(3.0, 0.0, 1.0) == 1.0 and isinstance(backend.clamp(0.5, 0, 1), float)
    np.testing.assert_array_equal(backend.clamp(np.array([-1.0, 0.5, 2.0]), 0.0, 1.0),
                                  [0.0, 0.5, 1.0])


def test_the_host_is_always_available():
    assert backend.available("cpu")
    assert backend.module("cpu") is np
    with pytest.raises(ValueError):
        backend.available("gpu")


@pytest.mark.skipif(backend.available("cuda"), reason="CuPy and a GPU are present")
def test_cuda_without_cupy_says_how_to_get_it():
    if backend._cupy() is None:
        with pytest.raises(RuntimeError, match="plantbench\\[gpu\\]"):
            backend.module("cuda")
