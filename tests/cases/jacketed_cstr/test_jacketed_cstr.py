"""`jacketed_cstr`, the reference CSTR: its published values and its loops."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.jacketed_cstr.definition import CASE
from plantbench.units import cstr
from plantbench.units.parameters import (
    REFERENCE_EIGENVALUES,
    REFERENCE_STEADY_STATE,
    reference_parameters,
)


def test_design_is_the_published_steady_state():
    d = pb.build(CASE, CASE.config("open loop")).design
    assert np.allclose(d.x, cstr.steady_state(reference_parameters()), rtol=0, atol=1e-9)
    assert d.x[1] == pytest.approx(REFERENCE_STEADY_STATE["T"], abs=0.01)


def test_open_loop_spectrum_is_the_published_one():
    ev = pb.build(CASE, CASE.config("open loop")).spectrum()
    assert np.sort(ev.real) == pytest.approx(np.sort(REFERENCE_EIGENVALUES), abs=1e-3)


def test_the_input_form_is_equation_17_10():
    """With the inputs at their nominal values the case's rhs is rhs_constant_volume exactly."""
    d = pb.build(CASE, CASE.config("open loop")).design
    x = d.x + np.array([0.1, 1.0, -0.5])
    assert np.array_equal(d.rhs(0.0, x, d.u, d.pp), cstr.rhs_constant_volume(0.0, x, d.pp))


@pytest.mark.parametrize("branch, sign", [("low", -1), ("middle", +1), ("high", -1)])
def test_only_the_middle_branch_is_open_loop_unstable(branch, sign):
    ev = pb.build(CASE, CASE.config("open loop", options={"branch": branch})).spectrum()
    assert np.sign(ev.real.max()) == sign


@pytest.mark.parametrize("structure", ["temperature PI", "composition cascade"])
def test_feedback_stabilizes_the_middle_branch(structure):
    ev = pb.build(CASE, CASE.config(structure)).spectrum()
    assert ev.real.max() < 0


def test_open_loop_leaves_for_the_high_branch_on_a_hotter_feed():
    """Figure 17.4b, through the case interface."""
    tr = pb.run(CASE, CASE.config("open loop", disturbance={
        "name": "feed_temperature", "value": 340.0, "t": 0.0}), t_end=600.0, n_points=7)
    assert tr.state("T")[-1] > REFERENCE_STEADY_STATE["T"] + 15


def test_temperature_loop_holds_the_reactor_on_the_same_disturbance():
    tr = pb.run(CASE, CASE.config("temperature PI", disturbance={
        "name": "feed_temperature", "value": 340.0, "t": 0.0}), t_end=600.0, n_points=7)
    assert tr.state("T")[-1] == pytest.approx(tr.state("T")[0], abs=1e-3)


def test_cascade_returns_the_concentration_after_a_feed_upset():
    tr = pb.run(CASE, CASE.config("composition cascade", disturbance={
        "name": "step", "field": "CA0", "fraction": 0.05, "t": 10.0}), t_end=1500.0, n_points=7)
    assert tr.state("C_A")[-1] == pytest.approx(tr.state("C_A")[0], abs=1e-4)
    assert tr.state("T")[-1] != pytest.approx(tr.state("T")[0], abs=1e-2)


@pytest.mark.parametrize("structure, least", [("temperature PI", 0.25),
                                              ("composition cascade", 0.2)])
def test_default_tunings_are_damped(structure, least):
    from plantbench.core import control as ctl
    assert ctl.damping_ratio(pb.build(CASE, CASE.config(structure)).spectrum()) > least


def test_the_simulink_tuning_is_poorly_damped_on_this_reactor():
    """Recorded in the case's docstring; it is tuning for the Simulink parameter set."""
    from plantbench.core import control as ctl
    from plantbench.units.parameters import SIMULINK_TUNING
    t, c = SIMULINK_TUNING["T_loop"], SIMULINK_TUNING["C_loop"]
    tuning = {"reactor T": {"Kc": -abs(t["Kc"]), "tau_I": t["tau_I"]},
              "reactor C_A": {"Kc": -abs(c["Kc"]), "tau_I": c["tau_I"]}}
    ev = pb.build(CASE, CASE.config("composition cascade", tuning=tuning)).spectrum()
    assert ctl.damping_ratio(ev) == pytest.approx(0.014, abs=0.001)


def test_a_mismatch_needs_no_design_for_the_changed_parameters():
    """k0 lowered by a fifth leaves one steady state, so `params` has no middle branch to
    design on; as a mismatch the loop stays as designed and the plant changes under it."""
    k0 = CASE.make_design(CASE.config()).pp.k0
    with pytest.raises(ValueError, match="steady states"):
        pb.build(CASE, CASE.config(params={"k0": 0.8 * k0}))
    s = pb.build(CASE, CASE.config(mismatch={"k0": 0.8 * k0}))
    assert s.structure == pb.build(CASE, CASE.config()).structure
    assert s.design.pp.k0 == 0.8 * k0


@pytest.mark.parametrize("name", ["CA0", "T0", "F0"])
def test_a_mismatch_of_a_feed_condition_is_refused(name):
    """The design turns the feed conditions into inputs, which a mismatch leaves as designed."""
    p = CASE.make_design(CASE.config()).pp
    with pytest.raises(ValueError, match="changes neither"):
        pb.build(CASE, CASE.config(mismatch={name: 1.1 * getattr(p, name)}))


def test_holding_the_coolant_under_the_cascade_is_the_open_loop_reactor():
    """Both loops go, the cascade primary with the loop it writes to, and the run is the
    open-loop one to the last bit: the 340 K feed carries the reactor to the high branch."""
    Fj = CASE.make_design(CASE.config()).u.Fj
    dist = {"name": "feed_temperature", "value": 340.0, "t": 10.0}
    held = pb.run(CASE, CASE.config("composition cascade", disturbance=dist, hold={"Fj": Fj}),
                  t_end=200.0, dt=0.5)
    open_loop = pb.run(CASE, CASE.config("open loop", disturbance=dist), t_end=200.0, dt=0.5)
    assert np.array_equal(held.x, open_loop.x)
    assert held.state("T")[-1] > 370.0
