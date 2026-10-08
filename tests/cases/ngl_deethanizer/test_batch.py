"""The deethanizer on the batched path: its Jacobian and a run made with it, against the
one-state-at-a-time path that the published results use."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.core import control as ctl

CASE = pb.load_case("ngl_deethanizer")


@pytest.fixture(scope="module")
def setup():
    return pb.build(CASE, CASE.config())


def test_the_batched_jacobian_is_the_forward_difference(setup):
    """Sampled columns against single evaluations with the same increments: within the
    tolerance of the nested solves, which a batch may iterate further (`contract.BATCH_TOL`)."""
    d, S = setup.design, setup.structure
    y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    args = (S, d.u, d.pp, None, d.rhs)
    J = ctl.jacobian_batched(0.0, y0, *args)
    f0 = ctl.closed_loop_rhs(0.0, y0, *args)
    names = d.plant.state_names()
    for j in (names.index("P_top"), names.index("M_B"), names.index("x[3,C2]"), len(y0) - 1):
        h = ctl._FD_STEP * max(abs(y0[j]), 1.0)
        h = (y0[j] + h) - y0[j]
        y = y0.copy()
        y[j] += h
        col = (ctl.closed_loop_rhs(0.0, y, *args) - f0) / h
        assert np.max(np.abs(J[:, j] - col)) < 1e-5 * max(1.0, np.max(np.abs(col)))


@pytest.mark.slow
def test_a_run_with_the_batched_jacobian_agrees():
    config = CASE.config(disturbance={"name": "feed ethane", "fraction": 0.1, "t": 0.5})
    ref = pb.run(CASE, config, t_end=3.0, dt=0.5)
    new = pb.run(CASE, config, t_end=3.0, dt=0.5, jacobian="batched")
    assert np.max(np.abs(new.x - ref.x) / (1e-6 + np.abs(ref.x))) < 1e-6
    for k in ref.y:
        np.testing.assert_allclose(new.y[k], ref.y[k], rtol=1e-6, atol=1e-8)
