"""The deethanizer run as an ensemble, against the same configurations run one by one."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.core import control as ctl
from plantbench.core.case import build_on, run_ensemble

T_END = 2.0
STEP = {"name": "feed ethane", "fraction": 0.1, "t": 0.5}


@pytest.fixture(scope="module")
def case():
    return pb.load_case("ngl_deethanizer")


def _configs(case):
    return [case.config(),
            case.config(disturbance=STEP, tuning={"stage 4 temperature": {"Kc": -0.4}}),
            case.config(setpoint_steps={"stage 28 temperature": [(1.0, 0.5)]}),
            {"structure": "no such structure"}]


@pytest.fixture(scope="module")
def results(case):
    return run_ensemble(case, _configs(case), T_END, dt=0.25)


def test_members_agree_with_their_serial_runs(case, results):
    assert isinstance(results[-1].error, KeyError)
    for config, res in zip(_configs(case)[:-1], results[:-1]):
        assert res.error is None and res.ensemble_size == 3
        ref = pb.run(case, config, t_end=T_END, dt=0.25)
        tr = res.trajectory
        np.testing.assert_array_equal(tr.t, ref.t)
        assert tr.state_names == ref.state_names and list(tr.u) == list(ref.u)
        assert np.max(np.abs(tr.x - ref.x) / (1e-6 + np.abs(ref.x))) < 1e-6
        for k in ref.y:
            np.testing.assert_allclose(tr.y[k], ref.y[k], rtol=1e-6, atol=1e-9, err_msg=k)
        for k in ref.u:
            np.testing.assert_allclose(tr.u[k], ref.u[k], rtol=1e-6, atol=1e-9, err_msg=k)


def test_batched_outputs_are_the_serial_ones(case):
    """The outputs of one run, evaluated as one batch, against `integrate`'s loop."""
    config = case.validate(case.config(disturbance=STEP))
    setup = build_on(case, case.make_design(config), config)
    d, S = setup.design, setup.structure
    y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    rng = np.random.default_rng(0)
    t = np.array([0.0, 0.5, 1.0, 1.5])
    Y = y0 + 1e-4 * np.abs(y0) * rng.standard_normal((len(t), len(y0)))
    Y[2] = np.nan  # a member that stopped
    measures = case.measurements(d)
    U, Ym = ctl.batch_outputs(d, S, setup.disturbance, t, Y, measures)
    n = len(d.x)
    for i in (0, 1, 3):
        base = setup.disturbance(t[i], d.u)
        u, _ = ctl.apply(S, Y[i, :n], Y[i, n:], base, d.pp, t[i])
        values = ctl.input_values(u)
        for k, v in values.items():
            np.testing.assert_allclose(U[k][i], v, rtol=1e-12, atol=1e-12, err_msg=k)
        for k, f in measures.items():
            np.testing.assert_allclose(Ym[k][i], f(Y[i, :n], u, d.pp), rtol=1e-10,
                                       atol=1e-12, err_msg=k)
    assert all(np.isnan(v[2]) for v in (*U.values(), *Ym.values()))
