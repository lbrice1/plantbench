"""Plantwide steady state, conservation, and the two structural results."""

import numpy as np
import pytest
import scipy.optimize

from plantbench.cases.reactor_separator_recycle import measurements as meas
from plantbench.cases.reactor_separator_recycle import plant
from plantbench.cases.reactor_separator_recycle import scenarios as sc
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.core import control as ctl
from plantbench.units import column as col
from plantbench.units import cstr
from plantbench.units.parameters import REFERENCE_STEADY_STATE

A = 1


@pytest.fixture(scope="module")
def design():
    return plant.solve_design(PlantParameters(), V_boil=28.0)


# -- the column on its own ------------------------------------------------------


def test_column_balances_close():
    cp = PlantParameters().column
    F, zF, q, V, Lr = 18.0, np.array([0.10, 0.42, 0.48]), 1.0, 28.0, 18.5
    z = col.steady_state(cp, F, zF, q, V, Lr)
    D, Bm = col.product_flows(cp, F, q, V, Lr)
    assert np.max(np.abs(col.rhs(z, cp, F, zF, q, V, Lr, D, Bm))) < 1e-10
    _, xD, _, _, _, xB = col.unpack(z, cp)
    assert D + Bm == pytest.approx(F, rel=1e-12)
    assert np.max(np.abs(F * zF - D * xD - Bm * xB)) < 1e-9
    assert xD.sum() == pytest.approx(1.0, abs=1e-10)
    assert xB.sum() == pytest.approx(1.0, abs=1e-10)


def test_internal_liquid_steps_up_below_the_feed():
    """Nothing prescribes the internal liquid rate; the weir hydraulics produce it."""
    cp = PlantParameters().column
    F, zF, q, V, Lr = 18.0, np.array([0.10, 0.42, 0.48]), 1.0, 28.0, 18.5
    z = col.steady_state(cp, F, zF, q, V, Lr)
    _, _, M, _, _, _ = col.unpack(z, cp)
    L = col.weir_flow(M, cp)
    assert L[0] == pytest.approx(Lr, rel=1e-6)
    assert L[-1] == pytest.approx(Lr + q * F, rel=1e-6)


def test_a_root_with_a_negative_holdup_is_never_returned(monkeypatch):
    """No steady-state balance involves the drum or base holdup, so Newton can move either
    to a negative value at no cost in the residual.  A warm start in the plantwide design
    did so with some of OpenBLAS's x86 kernels, the base reaching -7.4e7 kmol."""
    cp = PlantParameters().column
    F, zF, q, V, Lr = 18.0, np.array([0.10, 0.42, 0.48]), 1.0, 28.0, 18.5
    z = col.steady_state(cp, F, zF, q, V, Lr)
    _, xD, M, x, _, xB = col.unpack(z, cp)
    root = col.pack(-455.0, xD, M, x, -7.4e7, xB)
    D, Bm = col.product_flows(cp, F, q, V, Lr)
    assert np.max(np.abs(col.rhs(root, cp, F, zF, q, V, Lr, D, Bm))) < 1e-10
    # Every Newton solve, the warm start and the polish after the settle, finds that root.
    monkeypatch.setattr(scipy.optimize, "fsolve", lambda f, x0, **kw: (root, {}, 1, ""))
    MD, _, M, _, MB, _ = col.unpack(col.steady_state(cp, F, zF, q, V, Lr, guess=z, warm=True), cp)
    assert MD > 0 and MB > 0 and np.all(M > 0)


def test_a_settle_stops_at_its_evaluation_limit():
    cp = PlantParameters().column
    F, zF, q, V, Lr = 18.0, np.array([0.10, 0.42, 0.48]), 1.0, 28.0, 18.5
    with pytest.raises(col.SettleLimitExceeded, match="after 1[0-9]{2} right-hand-side "
                                                      "evaluations, the limit being 100"):
        col.steady_state(cp, F, zF, q, V, Lr, max_evaluations=100)


# -- the plantwide design -------------------------------------------------------


def test_design_converges(design):
    assert design.residual < 1e-8


def test_reactor_sits_on_reference_operating_point(design):
    """The design constraint that ties the plant to the reference reactor."""
    assert design.extras["CA0"] == pytest.approx(5.0, abs=1e-12)
    assert design.u.F_out == pytest.approx(3.2, abs=1e-12)
    assert design.x[1 + A] == pytest.approx(REFERENCE_STEADY_STATE["CA"], rel=0.005)
    assert design.x[4] == pytest.approx(REFERENCE_STEADY_STATE["T"], abs=0.01)
    assert design.x[5] == pytest.approx(REFERENCE_STEADY_STATE["Tj"], abs=0.15)


def test_reactor_inside_the_plant_is_still_unstable(design):
    """The recycle does not remove the instability of the reference reactor."""
    ss = np.array([design.x[1 + A], design.x[4], design.x[5]])
    lam = cstr.eigenvalues(ss, design.pp.reactor)
    assert (lam.real > 1e-6).sum() == 1


def test_design_overall_balances(design):
    u, d = design.u, design
    assert u.F_fresh == pytest.approx(u.P + u.Bm, rel=1e-9)
    made = d.extras["A_consumed"]
    assert u.P * d.xD[2] + u.Bm * d.xB[2] == pytest.approx(made, rel=1e-7)
    z_in = design.pp.z_inert
    assert u.F_fresh * z_in == pytest.approx(
        u.P * d.xD[0] + u.Bm * d.xB[0], rel=1e-7
    )


def test_plant_is_at_rest_at_the_design_point(design):
    for name, build in sc.STRUCTURES.items():
        S = build(design)
        y0 = np.concatenate([design.x, np.zeros(len(S))])
        r = ctl.closed_loop_rhs(0.0, y0, S, design.u, design.pp, None, design.rhs)
        assert np.max(np.abs(r)) < 1e-9, name


def test_mole_fractions_are_conserved_under_dynamics(design):
    """Catches a sign error in the recycle connection."""
    S = sc.recycle_free(design)
    t, X, U = ctl.simulate(
        design, S, t_end=400.0, disturbance=sc.feed_step(0.10, 30.0), n_points=50
    )
    for i in range(X.shape[1]):
        MD, xD, M, x, MB, xB = col.unpack(X[6:, i], design.pp.column)
        assert xD.sum() == pytest.approx(1.0, abs=1e-7)
        assert xB.sum() == pytest.approx(1.0, abs=1e-7)
        assert np.allclose(x.sum(axis=1), 1.0, atol=1e-7)
        assert MD > 0 and MB > 0 and M.min() > 0


# -- the structural results -----------------------------------------------------


def test_temperature_loop_has_a_stabilizing_gain_threshold(design):
    """The plantwide echo of the reference root-locus result."""
    import dataclasses

    def rightmost(Kc):
        S = sc.recycle_free(design)
        S[0] = dataclasses.replace(S[0], Kc=Kc, tau_I=5.0)
        y0 = np.concatenate([design.x, np.zeros(len(S))])
        f = lambda y: ctl.closed_loop_rhs(0.0, y, S, design.u, design.pp, None, design.rhs)
        n = len(y0)
        J = np.empty((n, n))
        for j in range(n):
            h = 1e-7 * max(1.0, abs(y0[j]))
            yp, ym = y0.copy(), y0.copy()
            yp[j] += h
            ym[j] -= h
            J[:, j] = (f(yp) - f(ym)) / (2 * h)
        return np.linalg.eigvals(J).real.max()

    assert rightmost(-0.10) > 1e-6  # too little gain: still unstable
    assert rightmost(-2.00) < 1e-6  # enough gain: stabilized


def test_fixing_the_distillate_fills_the_reflux_drum(design):
    """Not every flow in the recycle loop may be fixed."""
    S = sc.distillate_fixed(design)
    t, X, U = ctl.simulate(
        design, S, t_end=6000.0, disturbance=sc.feed_step(0.10, 30.0), n_points=300
    )
    MD = np.array(
        [col.unpack(X[6:, i], design.pp.column)[0] for i in range(X.shape[1])]
    )
    assert MD[-1] > 20 * design.pp.column.drum_holdup
    assert MD[-1] > MD[-2]  # still rising: it never settles


def test_closing_the_purge_accumulates_inert(design):
    """The inert has no exit but the purge, so its inventory rises at the rate it is fed.

    Asserted on the inventory rather than on the mole fraction, which falls late in the
    run as the reflux drum runs away.
    """
    S = sc.plantwide(design)
    # 600 min keeps the test short; the inventory claim holds at any horizon.
    t, X, U = ctl.simulate(
        design, S, t_end=600.0, disturbance=sc.purge_closed(30.0), n_points=200
    )
    inv = np.array([sum(meas.inert_inventory(X[:, i], design.pp))
                    for i in range(X.shape[1])])
    # The fresh feed holds reactor level in this structure, so it moves: integrate it
    # from the moment the purge shuts.
    after = t >= 30.0
    fed = np.trapezoid(U["F_fresh"][after] * design.pp.z_inert, t[after])
    assert inv[-1] - inv[0] == pytest.approx(fed, rel=0.01)
    assert np.all(np.diff(inv) > -1e-6)  # monotone: nothing removes it


def test_purge_open_reaches_a_new_inert_level(design):
    S = sc.plantwide(design)
    t, X, U = ctl.simulate(
        design, S, t_end=5000.0, disturbance=sc.inert_step(0.5, 30.0), n_points=200
    )
    xI = np.array([col.unpack(X[6:, i], design.pp.column)[1][0]
                   for i in range(X.shape[1])])
    assert xI[-1] > xI[0]
    assert abs(xI[-1] - xI[-2]) < 1e-4  # settled


def test_fixing_the_effluent_needs_the_cascade(design):
    """Fixing a flow in the recycle loop does not by itself change the per-pass
    conversion; holding the reactor composition does, and the amplification goes with it."""
    out = {}
    for name in ("recycle free", "effluent fixed", "effluent fixed + cascade"):
        S = sc.STRUCTURES[name](design)
        t, X, U = ctl.simulate(
            design, S, t_end=5000.0, n_points=200,
            disturbance=sc.throughput_step(0.10, sc.THROUGHPUT_HANDLE[name], 30.0),
        )
        prod = U["Bm"][-1] / U["Bm"][0] - 1
        assert prod > 0.03, name
        out[name] = (U["recycle"][-1] / U["recycle"][0] - 1) / prod
    assert out["recycle free"] > 3.0
    assert out["effluent fixed"] > 3.0
    assert out["effluent fixed + cascade"] < 1.6


def test_tray_temperature_holds_the_product(design):
    """The tray loop returns its tray to set-point, and the bottoms purity that follows
    stays within a factor of three of the design value through a +25% throughput step."""
    S = sc.plantwide(design)
    measure = next(lp.measure for lp in S if lp.name == "tray T")
    t, X, U = ctl.simulate(
        design, S, t_end=5000.0, n_points=400,
        disturbance=sc.throughput_step(0.25, sc.THROUGHPUT_HANDLE["effluent fixed + cascade"], 30.0),
    )
    T = [measure(X[:, i], None, design.pp) for i in (0, -1)]
    assert T[-1] == pytest.approx(T[0], abs=1e-3)
    xB = np.array([col.unpack(X[6:, i], design.pp.column)[5][A] for i in range(X.shape[1])])
    assert xB.max() < 3 * xB[0]
