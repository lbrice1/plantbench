"""The NumPy Peng-Robinson of `units.peng_robinson` against `thermo` on the same constants,
over the temperatures and pressures of a natural-gas plant: 35 to -116 C, 1000 to 6015 kPa,
on a seven-component natural gas."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("thermo")

from plantbench.units.peng_robinson import Components, PengRobinson  # noqa: E402

C = 273.15
COMPOSITIONS = {
    "feed": None,  # filled from the component set
    "top liquid": [0.002, 0.60, 0.25, 0.10, 0.03, 0.01, 0.008],
    "bottom liquid": [0.0, 0.02, 0.35, 0.30, 0.18, 0.08, 0.07],
}
NAMES = ("nitrogen", "methane", "ethane", "propane", "n-butane", "n-pentane", "n-hexane")
FEED = (0.010, 0.930, 0.030, 0.015, 0.009, 0.003, 0.003)
GRID = [(t, p) for t in (35.0, -4.8, -29.3, -59.6, -90.0, -116.3)
        for p in (1000e3, 1265e3, 4672e3, 6015e3)]


@pytest.fixture(scope="module")
def comps():
    return Components.from_names(NAMES)


@pytest.fixture(scope="module")
def pr(comps):
    return PengRobinson(comps)


@pytest.fixture(scope="module")
def phases(comps):
    from thermo import PRMIX, CEOSGas, CEOSLiquid

    c = comps
    consts, props = c._thermo
    kw = dict(Tcs=list(c.Tc), Pcs=list(c.Pc), omegas=list(c.omega), kijs=c.kij.tolist())
    return {"liquid": CEOSLiquid(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases),
            "vapour": CEOSGas(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases)}


def _x(name):
    x = np.array(FEED) if COMPOSITIONS[name] is None else np.array(COMPOSITIONS[name])
    return x / x.sum()


@pytest.mark.parametrize("phase", ["liquid", "vapour"])
@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_fugacity_and_enthalpy_agree_with_thermo_at_every_state(pr, phases, name, phase):
    x = _x(name)
    T = np.array([C + t for t, _ in GRID])
    P = np.array([p for _, p in GRID])
    X = np.tile(x, (len(GRID), 1))
    ln_phi = pr.ln_phi(T, P, X, phase)
    h = pr.h(T, P, X, phase)
    for n in range(len(GRID)):
        ref = phases[phase].to(T=T[n], P=P[n], zs=list(x))
        assert np.allclose(ln_phi[n], ref.lnphis(), rtol=0, atol=1e-10), GRID[n]
        assert h[n] == pytest.approx(ref.H(), rel=1e-9, abs=1e-6), GRID[n]


def test_bubble_points_agree_with_thermo(pr, comps):
    flasher = comps.flasher()
    x = np.array([_x("top liquid"), _x("bottom liquid"), _x("top liquid")])
    P = np.array([1010e3, 1000e3, 4672e3])
    T, y = pr.bubble_T(P, x)
    for n in range(len(P)):
        ref = flasher.flash(P=P[n], VF=0, zs=list(x[n]))
        assert T[n] == pytest.approx(ref.T, abs=1e-6)
        assert np.allclose(y[n], ref.gas.zs, atol=1e-5)
    K = pr.K(T, P, x, y)
    assert np.allclose((K * x).sum(axis=1), 1.0, atol=1e-12)


def test_the_bubble_point_does_not_depend_on_its_start(pr):
    """The result is smooth enough to difference: two starts agree to round-off."""
    x = np.tile(_x("top liquid"), (2, 1))
    P = np.full(2, 1010e3)
    T1, _ = pr.bubble_T(P, x)
    T2, _ = pr.bubble_T(P, x, T0=T1 + np.array([3.0, -3.0]))
    assert np.allclose(T1, T2, rtol=0, atol=1e-9)


def test_the_enthalpy_table_is_exact_to_a_millijoule(comps):
    assert comps.h_table_error < 1e-3


def test_a_temperature_outside_the_table_is_refused(pr):
    with pytest.raises(ValueError, match="outside the enthalpy table"):
        pr.h(np.array([30.0]), np.array([1e5]), np.array([_x("feed")]), "vapour")
