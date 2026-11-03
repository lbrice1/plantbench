"""The closed loop of an ensemble, members with their own tunings, set-points,
instruments, actuators, disturbances and times evaluated as one batch, and its
integration by Rodas5P.

The plant is the two-tank toy of `tests/_tanks.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from _tanks import tanks_design, tanks_structure
from scipy.integrate import solve_ivp

from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
from plantbench.core.ensemble import _A, _C, _H, GAMMA, STAGES, _c, _d, _dense, integrate_ensemble
from plantbench.core.instruments import Actuator, Instrument


@pytest.fixture
def design():
    return tanks_design()


def _members(design):
    """Four structures that differ in everything a member may change."""
    base = tanks_structure(design)
    return [
        base,
        ctl.with_tuning(base, {"level 1": {"Kc": -0.8, "hi": 4.0}, "level 2": {"tau_I": 12.0}}),
        ctl.with_setpoint_steps(ctl.with_setpoints(base, {"level 2": 6.0}),
                                {"level 2": [(2.0, 0.5), (4.0, -0.2)]}),
        ctl.with_instruments(base, instruments={"level 2": Instrument(tau=0.2, deadtime=2.0,
                                                                      order=2)},
                             actuators={"level 1": Actuator(tau=0.6, rate_limit=np.inf)}),
    ]


def _disturbances():
    return [None, dist.step("d", value=0.2, t=1.0), dist.step("q2", fraction=0.1, t=3.0),
            dist.combine(dist.step("d", value=-0.1, t=0.5), dist.step("d", value=0.3, t=2.5))]


def _states(design, structures, seed=0):
    rng = np.random.default_rng(seed)
    Y = []
    for S in structures:
        y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
        Y.append(y0 + 0.05 * (1.0 + np.abs(y0)) * rng.standard_normal(len(y0)))
    return np.array(Y)


def test_equal_members_evaluate_as_the_serial_closed_loop(design):
    S = tanks_structure(design)
    m = 5
    rows = np.arange(m)
    Y = _states(design, [S] * m)
    t = np.full(m, 3.0)
    u = ctl.ensemble_inputs(t, [None] * m, design.u, rows)
    F = ctl.ensemble_rhs(t, Y, ctl.take(ctl.stack([S] * m), rows), u, design.pp, design.rhs)
    for i in range(m):
        np.testing.assert_array_equal(
            F[i], ctl.closed_loop_rhs(3.0, Y[i], S, design.u, design.pp, None, design.rhs))


def test_each_member_evaluates_as_its_own_closed_loop(design):
    structures, ds = _members(design), _disturbances()
    Y = _states(design, structures)
    t = np.array([0.5, 1.5, 3.0, 4.5])
    rows = np.arange(len(structures))
    stacked = ctl.stack(structures)
    F = ctl.ensemble_rhs(t, Y, ctl.take(stacked, rows),
                         ctl.ensemble_inputs(t, ds, design.u, rows), design.pp, design.rhs)
    for i, (S, d) in enumerate(zip(structures, ds)):
        np.testing.assert_array_equal(
            F[i], ctl.closed_loop_rhs(t[i], Y[i], S, design.u, design.pp, d, design.rhs))


def test_rows_repeat_their_member(design):
    structures, ds = _members(design), _disturbances()
    Y = _states(design, structures)
    t = np.array([0.5, 1.5, 3.0, 4.5])
    rows = np.array([3, 3, 0, 2, 1, 3])
    F = ctl.ensemble_rhs(t[rows], Y[rows], ctl.take(ctl.stack(structures), rows),
                         ctl.ensemble_inputs(t, ds, design.u, rows), design.pp, design.rhs)
    for r, i in enumerate(rows):
        np.testing.assert_array_equal(
            F[r], ctl.closed_loop_rhs(t[i], Y[i], structures[i], design.u, design.pp, ds[i],
                                      design.rhs))


def test_the_jacobians_of_the_members_are_their_own(design):
    structures, ds = _members(design), _disturbances()
    Y = _states(design, structures)
    t = np.array([0.5, 1.5, 3.0, 4.5])
    n = Y.shape[1]
    rows = np.repeat(np.arange(len(structures)), n + 1)
    J = ctl.ensemble_jacobian(t[rows], Y, ctl.take(ctl.stack(structures), rows),
                              ctl.ensemble_inputs(t, ds, design.u, rows), design.pp, design.rhs)
    assert J.shape == (len(structures), n, n)
    for i, (S, d) in enumerate(zip(structures, ds)):
        np.testing.assert_array_equal(
            J[i], ctl.jacobian_batched(t[i], Y[i], S, design.u, design.pp, d, design.rhs))


def test_members_with_different_layouts_are_refused(design):
    base = tanks_structure(design)
    lagged = ctl.with_instruments(base, instruments={"level 1": Instrument(tau=0.5)})
    with pytest.raises(ValueError, match="state layout"):
        ctl.stack([base, lagged])
    with pytest.raises(ValueError, match="state layout"):
        ctl.stack([base, base[:2]])


# ------------------------------------------------------------------------------------
# The integrator
# ------------------------------------------------------------------------------------


@dataclass
class _NoInputs:
    pass


@dataclass
class _VanDerPol:
    x: np.ndarray
    u: _NoInputs
    pp: None = None
    batched: ClassVar[bool] = True

    def rhs(self, t, x, u, pp):
        return np.stack([x[..., 1], (1.0 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]], axis=-1)


def test_rodas5p_converges_at_fifth_order():
    """At a tolerance every step passes and with the steps ending on the grid, the steps
    grow to the grid spacing and stay there, so the error at the end measures the order of
    the method."""
    design = _VanDerPol(x=np.array([2.0, 0.0]), u=_NoInputs())
    ref = solve_ivp(lambda t, y: design.rhs(t, y, None, None), (0.0, 1.0), design.x,
                    method="DOP853", rtol=1e-14, atol=1e-14).y[:, -1]
    errors = []
    for n in (10, 20, 40):
        r = integrate_ensemble(design, [[]], [None], 1.0, n_points=n + 1, rtol=1.0, atol=1.0,
                               dense=False)
        errors.append(np.max(np.abs(r.Y[0, -1] - ref)))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 4.5), (errors, rates)


def _rodas5p_step(lam, t, y, h):
    """One Rodas5P step of y' = lam (y - sin t) + cos t, whose solution from y = sin t is
    sin t: the step's end and its kd = H k."""
    def f(t, y):
        return lam * (y - np.sin(t)) + np.cos(t)
    f_t = -lam * np.cos(t) - np.sin(t)
    W = 1.0 / (h * GAMMA) - lam
    k: list = []
    for i in range(STAGES):
        yi = y + sum(_A[i, j] * k[j] for j in range(i))
        b = f(t + _c[i] * h, yi) + h * _d[i] * f_t + sum(_C[i, j] / h * k[j] for j in range(i))
        k.append(b / W)
    y1 = y + sum(_A[STAGES - 1, j] * k[j] for j in range(STAGES - 1)) + k[-1]
    return y1, _H @ np.array(k)


@pytest.mark.parametrize("lam", [-1.0, -1e4])
def test_the_dense_output_converges_at_the_order_of_the_step(lam):
    """The error of the dense output inside a step falls as h^5, as that of the step's end
    does, on a mild and on a stiff problem (Prothero and Robinson, 1974)."""
    theta = np.array([0.2, 0.5, 0.8])
    errors = []
    for h in (0.2, 0.1, 0.05):
        y1, kd = _rodas5p_step(lam, 0.0, 0.0, h)
        y = _dense(np.array([0.0]), np.array([y1]), kd[:, None], theta)[:, 0]
        errors.append(np.max(np.abs(y - np.sin(theta * h))))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 4.8), (errors, rates)


@pytest.mark.parametrize("n_points", [11, 101, 1001])
def test_the_dense_output_is_as_close_to_the_solution_as_the_steps(n_points):
    """Van der Pol over 5 time units: the grid values of the dense output and those of
    steps ending on the grid are both within 1.5 tolerances of the solution, scaled by the
    size of each state; the dense output takes the same steps on any grid."""
    design = _VanDerPol(x=np.array([2.0, 0.0]), u=_NoInputs())
    t = np.linspace(0.0, 5.0, n_points)
    ref = solve_ivp(lambda t, y: design.rhs(t, y, None, None), (0.0, 5.0), design.x,
                    method="DOP853", rtol=1e-13, atol=1e-13, t_eval=t).y.T
    rtol, atol = 1e-8, 1e-10
    scale = atol + rtol * np.max(np.abs(ref), axis=0)
    runs = [integrate_ensemble(design, [[]], [None], 5.0, n_points=n_points, rtol=rtol,
                               atol=atol, dense=dense) for dense in (True, False)]
    for r in runs:
        np.testing.assert_array_equal(r.t, t)
        assert np.max(np.abs(r.Y[0] - ref) / scale) < 1.5
    coarse = integrate_ensemble(design, [[]], [None], 5.0, n_points=2, rtol=rtol, atol=atol)
    assert runs[0].steps[0] == coarse.steps[0] <= runs[1].steps[0]
    np.testing.assert_array_equal(runs[0].Y[0, -1], coarse.Y[0, -1])


def _lsoda(design, S, d, t_end, n_points):
    return ctl.integrate(design, S, t_end, d, n_points=n_points,
                         breakpoints=ctl.breakpoints_of(d, S))


def test_members_agree_with_their_lsoda_runs(design):
    structures = _members(design)
    ds = _disturbances()
    ds[0] = dist.ramp("d", value=0.3, t=1.5, duration=2.0)
    r = integrate_ensemble(design, structures, ds, 10.0, n_points=41)
    assert r.status == ["ok"] * 4
    n = len(design.x)
    for i, (S, d) in enumerate(zip(structures, ds)):
        ref = _lsoda(design, S, d, 10.0, 41)
        np.testing.assert_array_equal(r.t, ref.t)
        X = r.Y[i, :, :n].T
        assert np.max(np.abs(X - ref.X) / (1.0 + np.abs(ref.X))) < 1e-6, i
    assert np.all(r.jacobians == r.steps)


def test_a_failing_member_stops_and_the_others_finish(design):
    S = tanks_structure(design)
    poison = dist.step("d", value=np.nan, t=2.0)
    r = integrate_ensemble(design, [S, S, S], [None, poison, dist.step("d", value=0.1, t=1.0)],
                           5.0, n_points=11)
    assert r.status == ["ok", "failed", "ok"]
    assert "round-off" in r.message[1]
    assert np.isnan(r.Y[1, -1]).all() and np.isfinite(r.Y[1, :5]).all()
    assert np.isfinite(r.Y[[0, 2]]).all()


def test_the_wall_budget_stops_every_member(design):
    S = tanks_structure(design)
    r = integrate_ensemble(design, [S, S], [None, None], 5.0, n_points=11, wall_budget=0.0)
    assert r.status == ["timeout", "timeout"]


def test_the_design_must_declare_a_batch(design):
    from _toy import toy_design, toy_loop
    toy = toy_design()
    with pytest.raises(ValueError, match="batched"):
        integrate_ensemble(toy, [ctl.bias_from_design([toy_loop()], toy)], [None], 1.0)
