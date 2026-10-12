"""The closed loop of an ensemble: members with their own tunings, set-points,
instruments, actuators, disturbances and times, evaluated as one batch.

The plant is the two-tank toy of `tests/_tanks.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from _tanks import tanks_design, tanks_structure

from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
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
