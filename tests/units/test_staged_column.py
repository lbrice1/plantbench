"""The staged column: reduces to `plantbench.units.column` under constant volatility and
latent heat, and closes its balances under the equation of state."""

from __future__ import annotations

import numpy as np
import pytest

from plantbench.units import column as ref
from plantbench.units.parameters import ColumnParameters
from plantbench.units.staged_column import ColumnSpec, Feed, IdealThermo, evaluate, weir_constants


def test_it_reduces_to_the_constant_volatility_column():
    """With the reflux given as a liquid feed on the top stage and the boil-up as a duty,
    the tray and base derivatives are those of `units.column.rhs`."""
    cp = ColumnParameters()
    rng = np.random.default_rng(0)
    z = ref.initial_state(cp, np.array([0.3, 0.6, 0.1]), np.array([0.01, 0.1, 0.89]))
    z = z + rng.normal(0.0, 1e-3, z.size) * np.abs(z)  # off steady state
    MD, xD, M, x, MB, xB = ref.unpack(z, cp)
    F, zF, V, L_reflux, D, Bm = 1.2, np.array([0.1, 0.5, 0.4]), 3.0, 2.4, 0.6, 0.6
    expected = ref.unpack(ref.rhs(z, cp, F, zF, 1.0, V, L_reflux, D, Bm), cp)

    spec = ColumnSpec(n_stages=cp.n_trays, n_components=3, weir_coeff=cp.weir_coeff,
                      holdup_weir=cp.holdup_weir, dP_stage=0.0)
    thermo = IdealThermo(cp.alpha, cp.lambda_vap)
    feeds = [Feed(0, L_reflux, xD, 0.0), Feed(cp.feed_tray - 1, F, zF, 0.0)]
    dM, dx, dM_B, dx_B, profile = evaluate(spec, thermo, M, x, MB, xB, 1e5, feeds,
                                           Q_R=V * cp.lambda_vap, B=Bm)
    assert np.allclose(profile.V, V, rtol=1e-14)
    assert np.allclose(dM, expected[2], rtol=1e-12, atol=1e-14)
    assert np.allclose(dx, expected[3], rtol=1e-12, atol=1e-14)
    assert dM_B == pytest.approx(expected[4], rel=1e-12, abs=1e-14)
    assert np.allclose(dx_B, expected[5], rtol=1e-12, atol=1e-14)


@pytest.fixture(scope="module")
def column():
    pytest.importorskip("thermo")
    from plantbench.units.peng_robinson import Components, PengRobinson

    pr = PengRobinson(Components.from_names(
        ("nitrogen", "methane", "ethane", "propane", "n-butane", "n-pentane", "n-hexane")))
    n, nc = 12, pr.c.n
    spec = ColumnSpec(n_stages=n, n_components=nc, weir_coeff=20.0, holdup_weir=1.0,
                      dP_stage=640.0)
    top = np.array([0.005, 0.80, 0.12, 0.05, 0.015, 0.005, 0.005])
    bottom = np.array([0.0, 0.02, 0.35, 0.30, 0.18, 0.08, 0.07])
    f = np.linspace(0, 1, n + 1)[:, None]
    x = (1 - f) * top + f * bottom
    x /= x.sum(axis=1, keepdims=True)
    M = np.full(n, 1.5)
    T_feed = np.array([160.0])
    zf = np.array([[0.01, 0.93, 0.03, 0.015, 0.009, 0.003, 0.003]])
    h_feed = float(pr.flash_PT(T_feed, np.array([1.01e6]), zf).h[0])
    feeds = [Feed(0, 10.0, zf[0], h_feed), Feed(7, 3.0, bottom, float(
        pr.h(np.array([250.0]), np.array([1.02e6]), bottom[None], "liquid")[0]))]
    return pr, spec, M, x[:n], 20.0, x[n], feeds


def test_the_balances_close(column):
    """Whatever the state, total and component moles and energy are conserved: what
    the stages and reboiler gain is what enters less what leaves."""
    pr, spec, M, x, M_B, x_B, feeds = column
    Q_R, B = 2.0e4, 2.5
    dM, dx, dM_B, dx_B, p = evaluate(spec, pr, M, x, M_B, x_B, 1.01e6, feeds, Q_R, B)
    F_in = sum(f.F for f in feeds)
    assert dM.sum() + dM_B == pytest.approx(F_in - B - p.V[0], rel=1e-12)
    d_moles = (dx * M[:, None] + x * dM[:, None]).sum(axis=0) + dx_B * M_B + x_B * dM_B
    comp_in = sum(f.F * f.z for f in feeds)
    assert np.allclose(d_moles, comp_in - B * x_B - p.V[0] * p.y[0], atol=1e-10)
    # Energy, under the quasi-steady closure: sum h dM equals the enthalpy balance.
    h_in = sum(f.F * f.h for f in feeds) + Q_R
    h_out = B * p.h_L[-1] + p.V[0] * p.h_V[0]
    stored = (p.h_L[:-1] * dM).sum() + p.h_L[-1] * dM_B
    assert stored == pytest.approx(h_in - h_out, rel=1e-10)


def test_temperatures_rise_down_the_column(column):
    pr, spec, M, x, M_B, x_B, feeds = column
    *_, p = evaluate(spec, pr, M, x, M_B, x_B, 1.01e6, feeds, 3.0e5, 2.5)
    assert np.all(np.diff(p.T) > 0)
    assert np.all(p.V > 0)


def test_weir_constants_give_the_francis_flow():
    """At a crest h_ow above the weir, the flow is 1.84 L_w h_ow**1.5 m3/s."""
    L_w, A, h_w, rho, h_ow = 2.238, 4.397, 0.05, 12.0, 0.02
    coeff, held = weir_constants(L_w, A, h_w, rho)
    assert held == pytest.approx(rho * A * h_w)
    M = held + rho * A * h_ow
    assert coeff * (M - held) ** 1.5 == pytest.approx(60 * rho * 1.84 * L_w * h_ow**1.5)
