"""The batched path on an NVIDIA GPU, against the same path on the host.

Skipped unless CuPy and a GPU are present: `pixi run -e cuda test-gpu`."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench import backend
from plantbench.core import control as ctl

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not backend.available("cuda"), reason="no CuPy or no GPU")]


@pytest.fixture(scope="module")
def setup():
    case = pb.load_case("ngl_deethanizer")
    return case, pb.build(case, case.config())


def test_the_equation_of_state_on_the_device_is_the_one_on_the_host(setup):
    _, s = setup
    plant = s.design.plant
    host, dev = plant.eos.on("cpu"), plant.eos.on("cuda")
    st = plant.layout.unpack(s.design.x)
    x = np.vstack([st["x"], st["x_B"][None]])
    P = plant.column.pressures(st["P_top"] * 1e3)
    T, y = host.bubble_T(P, x)
    Td, yd = dev.bubble_T(backend.to_device(P, "cuda"), backend.to_device(x, "cuda"))
    np.testing.assert_allclose(backend.to_host(Td), T, rtol=1e-12)
    np.testing.assert_allclose(backend.to_host(yd), y, rtol=1e-10, atol=1e-14)
    for phase, comp in (("liquid", x), ("vapor", y)):
        np.testing.assert_allclose(
            backend.to_host(dev.h(Td, backend.to_device(P, "cuda"), backend.to_device(comp, "cuda"), phase)),
            host.h(T, P, comp, phase), rtol=1e-11, atol=1e-8)


# Derivative action on the reflux loop, which adds a filter state to every member.
PID = {"stage 4 temperature": {"tau_D": 0.5}}


@pytest.mark.parametrize("tuning", [{}, PID], ids=["PI", "PID"])
def test_the_jacobian_on_the_device_is_the_one_on_the_host(setup, tuning):
    _, s = setup
    d, S = s.design, ctl.with_tuning(s.structure, tuning)
    y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    args = (S, d.u, d.pp, None, d.rhs)
    J = ctl.jacobian_batched(0.0, y0, *args)
    Jd = backend.to_host(ctl.jacobian_batched(0.0, backend.to_device(y0, "cuda"), *args))
    assert np.max(np.abs(Jd - J)) < 1e-5 * max(1.0, np.max(np.abs(J)))


def test_a_run_on_the_device_agrees_with_one_on_the_host(setup):
    case, _ = setup
    config = case.config(disturbance={"name": "feed ethane", "fraction": 0.1, "t": 0.5})
    host = pb.run(case, config, t_end=3.0, dt=0.5, jacobian="batched")
    dev = pb.run(case, config, t_end=3.0, dt=0.5, jacobian="batched", device="cuda")
    assert np.max(np.abs(dev.x - host.x) / (1e-6 + np.abs(host.x))) < 1e-6


@pytest.mark.parametrize("tuning", [{}, PID], ids=["PI", "PID"])
def test_an_ensemble_on_the_device_agrees_with_one_on_the_host(setup, tuning):
    from plantbench.core.ensemble import integrate_ensemble
    case, s = setup
    d = s.design
    base = ctl.with_tuning(s.structure, tuning)
    S = [base, ctl.with_tuning(base, {"stage 4 temperature": {"Kc": -0.4}})]
    ds = [None, pb.build(case, case.config(disturbance={"name": "feed ethane",
                                                        "fraction": 0.1, "t": 0.5})).disturbance]
    rtol, atol = 1e-8, 1e-10
    host = integrate_ensemble(d, S, ds, 2.0, n_points=5, rtol=rtol, atol=atol)
    dev = integrate_ensemble(d, S, ds, 2.0, n_points=5, rtol=rtol, atol=atol, device="cuda")
    assert host.status == dev.status == ["ok", "ok"]
    assert np.array_equal(host.steps, dev.steps) and np.array_equal(host.rejected, dev.rejected)
    # In units of the tolerance: a relative difference would be dominated by states that
    # are zero to within atol, which neither device resolves further.  Before the device
    # factorized by LU, its batched inverse, reproduced on the host, moved a controller
    # state by 0.065 of the tolerance; the kernels' 2e-11 relative difference in the
    # states amounts to at most 0.002.
    err = np.abs(dev.Y - host.Y) / (atol + rtol * np.abs(host.Y))
    m, k, i = np.unravel_index(np.argmax(err), err.shape)
    assert err[m, k, i] < 0.25, (f"member {m}, t = {host.t[k]}, state {i}: host "
                                f"{host.Y[m, k, i]!r}, device {dev.Y[m, k, i]!r}")


@pytest.mark.parametrize("case_id", ["ngl_deethanizer", "ngl_demethanizer"])
def test_the_batched_lu_on_the_device_solves_as_the_host(case_id):
    """The step's systems I / (h gamma) - J of three members of an NGL closed loop, N above
    the 256 of CuPy's batched solve, factorized once and solved twice: residuals at
    round-off, and the host's LU solution to the conditioning of the systems.  A member
    whose matrix is singular gets NaN, and the others their solutions."""
    from plantbench.core.ensemble import GAMMA, _factor
    case = pb.load_case(case_id)
    s = pb.build(case, case.config())
    d, S = s.design, s.structure
    y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    J = ctl.jacobian_batched(0.0, y0, S, d.u, d.pp, None, d.rhs)
    N = len(y0)
    assert N > 256
    h = np.array([1e-3, 1e-2, 1e-1])
    W = np.eye(N) / (h * GAMMA)[:, None, None] - J
    rng = np.random.default_rng(0)
    host = _factor(W, np)
    dev = _factor(backend.to_device(W, "cuda"), backend.module("cuda"))
    for _ in range(2):
        b = rng.standard_normal((3, N))
        x = backend.to_host(dev(backend.to_device(b, "cuda")))
        residual = np.abs(np.einsum("kij,kj->ki", W, x) - b).max(axis=1)
        scale = np.abs(W).max(axis=(1, 2)) * np.abs(x).max(axis=1)
        assert np.all(residual < 1e-12 * scale), residual / scale
        ref = host(b)
        cond = np.linalg.cond(W)
        assert np.all(np.abs(x - ref).max(axis=1) < 1e-14 * cond * np.abs(ref).max(axis=1))
    W[1, :, 5] = 0.0
    x = backend.to_host(_factor(backend.to_device(W, "cuda"), backend.module("cuda"))(
        backend.to_device(b, "cuda")))
    assert np.isnan(x[1]).all() and np.isfinite(x[[0, 2]]).all()


@pytest.mark.parametrize("case_id", ["jacketed_cstr", "reactor_separator_recycle",
                                     "mma_polymerization_cstr"])
def test_a_reactor_case_on_the_device_is_the_one_on_the_host(case_id):
    """The batched right-hand side and measurements of the reactor cases, at 8 states 1 %
    from the design, on the device and on the host."""
    case = pb.load_case(case_id)
    d = pb.build(case, case.config()).design
    X = d.x * (1.0 + 1e-2 * np.random.default_rng(0).standard_normal((8, len(d.x))))
    u = ctl.batch_inputs(d.u, len(X))
    ud = ctl.batch_inputs(d.u, len(X), backend.module("cuda"))
    Xd = backend.to_device(X, "cuda")
    F = d.rhs(0.0, X, u, d.pp)
    np.testing.assert_allclose(backend.to_host(d.rhs(0.0, Xd, ud, d.pp)), F, rtol=1e-12,
                               atol=1e-12 * np.abs(F).max())
    for name, f in case.measurements(d).items():
        np.testing.assert_allclose(backend.to_host(f(Xd, ud, d.pp)), f(X, u, d.pp),
                                   rtol=1e-12, err_msg=name)


def test_an_ensemble_stepper_on_the_device_steps_as_on_the_host():
    from plantbench.core.ensemble_stepper import EnsembleStepper
    case = pb.load_case("jacketed_cstr")
    configs = [case.config("temperature PI", disturbance={"name": "feed_temperature",
                                                          "value": v, "t": 1.25})
               for v in (330.0, 340.0)]
    host = EnsembleStepper.from_configs(case, configs, 0.5)
    dev = EnsembleStepper.from_configs(case, configs, 0.5, device="cuda")
    # In units of the tolerance, as for the ensemble above, but without equal steps: the
    # device's arithmetic can change an accepted step, after which the two integrations
    # agree to the order of the tolerance rather than in the last bits.  On an
    # A100-SXM4-80GB, member 0 takes 275 evaluations in its fifth step to the host's 259
    # and then differs by 0.328 of the tolerance; at rtol 1e-10 and atol 1e-12, by 0.020.
    # Member 1, whose steps do not change, differs by at most 5.99e-14 relative.
    for _ in range(10):
        a, b = host.step(), dev.step()
        for i in range(2):
            x = a.steps[i].x
            err = np.abs(b.steps[i].x - x) / (host.atol + host.rtol * np.abs(x))
            assert np.max(err) < 1.0, f"t = {a.steps[i].t}, member {i}: {err}"
