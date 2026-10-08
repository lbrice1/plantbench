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
    for phase, comp in (("liquid", x), ("vapour", y)):
        np.testing.assert_allclose(
            backend.to_host(dev.h(Td, backend.to_device(P, "cuda"), backend.to_device(comp, "cuda"), phase)),
            host.h(T, P, comp, phase), rtol=1e-11, atol=1e-8)


def test_the_jacobian_on_the_device_is_the_one_on_the_host(setup):
    _, s = setup
    d, S = s.design, s.structure
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
