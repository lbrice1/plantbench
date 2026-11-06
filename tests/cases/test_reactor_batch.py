"""The batched right-hand sides of `jacketed_cstr`, `reactor_separator_recycle` and
`mma_polymerization_cstr`.

A batch row is the one state evaluated alone, whatever the batch around it; the batched
balances conserve what the scalar ones do; and an ensemble, which needs a batched
right-hand side, agrees with `run`."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.reactor_separator_recycle.plant import N_REACTOR_STATES
from plantbench.core import control as ctl
from plantbench.units import column as col

CSTR = pb.load_case("jacketed_cstr")
RSR = pb.load_case("reactor_separator_recycle")
MMA = pb.load_case("mma_polymerization_cstr")


def _states(setup, m, seed=0, spread=1e-2):
    d = setup.design
    rng = np.random.default_rng(seed)
    return d.x * (1.0 + spread * rng.standard_normal((m, len(d.x))))


@pytest.fixture(scope="module", params=["jacketed_cstr", "reactor_separator_recycle",
                                                 "mma_polymerization_cstr"])
def setup(request):
    case = pb.load_case(request.param)
    return pb.build(case, case.config("open loop") if "open loop" in case.structures
                    else case.config())


def test_a_row_of_a_batch_is_the_state_alone(setup):
    """Open-loop derivatives and every measurement at 16 states 1 % from the design, one
    at a time and as a batch, agree to round-off."""
    d = setup.design
    X = _states(setup, 16)
    u = ctl.batch_inputs(d.u, len(X))
    batch = d.rhs(0.0, X, u, d.pp)
    single = np.array([d.rhs(0.0, x, d.u, d.pp) for x in X])
    scale = np.maximum(np.abs(single).max(axis=0), 1e-12)
    assert np.max(np.abs(batch - single) / scale) < 1e-12
    for name, f in setup.case.measurements(d).items():
        np.testing.assert_allclose(f(X, u, d.pp), [f(x, d.u, d.pp) for x in X], rtol=1e-13,
                                   err_msg=name)


def test_a_row_does_not_depend_on_the_batch_around_it(setup):
    d = setup.design
    X = _states(setup, 16, seed=1)
    whole = d.rhs(0.0, X, ctl.batch_inputs(d.u, 16), d.pp)
    for size in (1, 4):
        parts = np.vstack([d.rhs(0.0, X[k:k + size], ctl.batch_inputs(d.u, size), d.pp)
                           for k in range(0, 16, size)])
        assert np.array_equal(parts, whole)
    reversed_ = d.rhs(0.0, X[::-1], ctl.batch_inputs(d.u, 16), d.pp)
    assert np.array_equal(reversed_[::-1], whole)


def test_the_batched_plant_conserves_volume_and_species():
    """At states 1 % from the design: the reactor volume changes by the flow in less the
    flow out, and the column's holdup of each species by its feed less its products."""
    setup = pb.build(RSR, RSR.config())
    d = setup.design
    X = _states(setup, 8, seed=2)
    u = ctl.batch_inputs(d.u, len(X))
    F = d.rhs(0.0, X, u, d.pp)
    cp, rho = d.pp.column, d.pp.rho_molar
    R = u.D - u.P
    n_in = u.F_fresh + R
    assert np.allclose(F[:, 0], n_in / rho - u.F_out, rtol=1e-12, atol=1e-12)
    for k in range(len(X)):
        z, dz = X[k, N_REACTOR_STATES:], F[k, N_REACTOR_STATES:]
        MD, xD, M, x, MB, xB = col.unpack(z, cp)
        dMD, dxD, dM, dx, dMB, dxB = col.unpack(dz, cp)
        held = (dMD * xD + MD * dxD + (dM[:, None] * x + M[:, None] * dx).sum(axis=0)
                + dMB * xB + MB * dxB)
        fed = u.F_out[k] * rho * X[k, 1:4] / rho
        out = u.D[k] * xD + u.Bm[k] * xB
        np.testing.assert_allclose(held, fed - out, rtol=1e-9, atol=1e-11)


def test_the_heat_integrated_plants_are_not_batched():
    setup = pb.build(RSR, RSR.config(options={"network": "D4"}))
    assert not ctl.supports_batch(setup.design)
    with pytest.raises(ValueError, match="batched"):
        pb.run_ensemble(RSR, [RSR.config(options={"network": "D4"})], t_end=1.0)


@pytest.mark.parametrize("case, configs, t_end, dt", [
    (CSTR, [CSTR.config("temperature PI", disturbance={"name": "feed_temperature",
                                                       "value": v, "t": 5.0})
            for v in (330.0, 340.0)], 200.0, 0.5),
    (RSR, [RSR.config(disturbance={"name": "throughput", "fraction": f, "t": 5.0})
           for f in (0.05, 0.1)], 60.0, 1.0),
    (MMA, [MMA.config("NAMW PI", setpoint_steps={"NAMW": [[5.0, step]]})
           for step in (15_000.0, -7_000.0)], 120.0, 1.0),
])
def test_an_ensemble_of_a_reactor_case_agrees_with_run(case, configs, t_end, dt):
    """Rodas5P against LSODA, to the tolerance: 1e-6 of each measurement, relative."""
    results = pb.run_ensemble(case, configs, t_end=t_end, dt=dt)
    for config, r in zip(configs, results):
        assert r.error is None
        ref = pb.run(case, config, t_end=t_end, dt=dt)
        for name in ref.y:
            np.testing.assert_allclose(r.trajectory.y[name], ref.y[name], rtol=1e-6,
                                       atol=1e-6 * np.abs(ref.y[name]).max(), err_msg=name)
