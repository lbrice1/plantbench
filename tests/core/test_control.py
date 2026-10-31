"""The control layer on the toy plant: set-points, schedules, tuning, input recording."""

from __future__ import annotations

import numpy as np
import pytest
from _toy import K, measure_x, toy_design, toy_loop

from plantbench.core import control as ctl
from plantbench.core.control import Loop, Ratio


def test_a_biased_structure_starts_at_rest(design):
    S = ctl.bias_from_design([toy_loop()], design)
    y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
    r = ctl.closed_loop_rhs(0.0, y0, S, design.u, design.pp, None, design.rhs)
    assert np.max(np.abs(r)) < 1e-12


def test_a_zero_setpoint_is_kept():
    """A deliberate set-point of zero must not be read as 'unset' and replaced."""
    design = toy_design(m=1.0)  # design value of x is 2
    S = ctl.bias_from_design([toy_loop(setpoint=0.0)], design)
    assert S[0].setpoint == 0.0
    S = ctl.bias_from_design([toy_loop()], design)
    assert S[0].setpoint == pytest.approx(K * 1.0)


def test_an_unset_setpoint_is_refused_rather_than_guessed(design):
    with pytest.raises(ValueError, match="no set-point"):
        toy_loop().output(1.0, 0.0)


def test_with_setpoints_moves_the_plant(design):
    S = ctl.with_setpoints(ctl.bias_from_design([toy_loop()], design), {"level": 3.0})
    t, X, U = ctl.simulate(design, S, t_end=200.0, n_points=50)
    assert X[0, -1] == pytest.approx(3.0, abs=1e-5)


def test_setpoint_schedule_sums_the_changes_in_force(design):
    S = ctl.bias_from_design([toy_loop()], design)
    S = ctl.with_setpoint_steps(S, {"level": [(10.0, 0.5), (100.0, -0.2)]})
    t, X, U = ctl.simulate(design, S, t_end=300.0, n_points=301)
    at = lambda time: X[0, np.searchsorted(t, time)]
    assert at(9.0) == pytest.approx(2.0, abs=1e-9)
    assert at(99.0) == pytest.approx(2.5, abs=1e-3)
    assert X[0, -1] == pytest.approx(2.3, abs=1e-5)


def test_with_tuning_replaces_only_what_is_named(design):
    S = ctl.bias_from_design([toy_loop()], design)
    T = ctl.with_tuning(S, {"level": {"Kc": 1.5}})
    assert T[0].Kc == 1.5 and T[0].tau_I == S[0].tau_I and T[0].bias == S[0].bias
    assert S[0].Kc == 0.5  # the original is untouched


@pytest.mark.parametrize("helper, arg", [
    (ctl.with_setpoints, {"nope": 1.0}),
    (ctl.with_tuning, {"nope": {"Kc": 1.0}}),
    (ctl.with_setpoint_steps, {"nope": [(1.0, 1.0)]}),
])
def test_unknown_loop_names_are_rejected(design, helper, arg):
    with pytest.raises(KeyError, match="nope"):
        helper([toy_loop()], arg)


def test_only_tuning_fields_can_be_tuned():
    with pytest.raises(KeyError, match="setpoint"):
        ctl.with_tuning([toy_loop()], {"level": {"setpoint": 1.0}})


def test_a_ratio_station_has_no_setpoint():
    S = [toy_loop(), Ratio("r", "d", "m")]
    with pytest.raises(TypeError, match="ratio station"):
        ctl.with_setpoints(S, {"r": 1.0})


def test_simulate_records_every_input_and_the_derived_signals(design):
    S = ctl.bias_from_design([toy_loop()], design)
    t, X, U = ctl.simulate(design, S, t_end=10.0, n_points=5)
    assert set(U) == {"m", "d", "z[0]", "z[1]", "note", "m plus d"}
    assert np.all(np.isnan(U["note"]))
    assert np.allclose(U["z[1]"], 0.75)
    assert np.allclose(U["m plus d"], U["m"] + U["d"])


def test_integrate_records_measurements_on_the_same_grid(design):
    S = ctl.bias_from_design([toy_loop()], design)
    r = ctl.integrate(design, S, 10.0, n_points=7, measurements={"x": measure_x})
    assert np.allclose(r.Y["x"], r.X[0]) and r.nfev > 0


def test_a_wall_budget_stops_the_run(design):
    S = ctl.bias_from_design([toy_loop()], design)
    with pytest.raises(ctl.WallTimeExceeded):
        ctl.integrate(design, S, 1e4, n_points=10, wall_budget=0.0)
    with pytest.raises(ctl.WallTimeExceeded):  # and when the run restarts at breakpoints
        ctl.integrate(design, S, 1e4, n_points=10, wall_budget=0.0, breakpoints=[5.0])


def test_a_budget_that_is_not_reached_leaves_the_run_unchanged(design):
    S = ctl.bias_from_design([toy_loop()], design)
    free = ctl.integrate(design, S, 50.0, n_points=11, breakpoints=[5.0])
    budgeted = ctl.integrate(design, S, 50.0, n_points=11, breakpoints=[5.0],
                             wall_budget=600.0)
    assert np.array_equal(free.X, budgeted.X) and free.nfev == budgeted.nfev


def test_a_primary_loop_writes_its_secondary_setpoint(design):
    """A cascade: the primary's output is the secondary's set-point, not a valve."""
    secondary = toy_loop()
    primary = Loop("outer", measure_x, "sp:level", Kc=0.2, tau_I=50.0, lo=0.0, hi=10.0)
    S = ctl.bias_from_design([secondary, primary], design)
    S = ctl.with_setpoints(S, {"outer": 2.4})
    # The outer error decays with a time constant of about tau_I (1 + Kc) / Kc = 300 min.
    t, X, U = ctl.simulate(design, S, t_end=6000.0, n_points=50)
    assert X[0, -1] == pytest.approx(2.4, abs=1e-6)


def test_a_non_finite_state_is_an_error_not_a_result(design):
    """The solver can accept a step that produced NaN; integrate must not return it."""

    class Breaks(type(design)):
        def rhs(self, t, x, u, pp):
            return np.array([np.nan]) if t > 5.0 else super().rhs(t, x, u, pp)

    broken = Breaks(x=design.x, u=design.u, pp=design.pp)
    S = ctl.bias_from_design([toy_loop()], broken)
    with pytest.raises(RuntimeError, match="non-finite"):
        ctl.integrate(broken, S, 20.0, n_points=11)


def test_breakpoints_restart_the_solver_without_changing_the_answer(design):
    from plantbench.core import disturbances as dist
    S = ctl.with_setpoint_steps(ctl.bias_from_design([toy_loop()], design),
                                {"level": [(700.0, 0.5)]})
    d = dist.step("d", value=0.3, t=400.0)
    cuts = ctl.breakpoints_of(d, S)
    assert cuts == (400.0, 700.0)
    whole = ctl.integrate(design, S, 1000.0, d, n_points=101, rtol=1e-10, atol=1e-12)
    pieces = ctl.integrate(design, S, 1000.0, d, n_points=101, rtol=1e-10, atol=1e-12,
                           breakpoints=cuts)
    assert np.array_equal(whole.t, pieces.t)
    assert np.allclose(whole.X, pieces.X, atol=1e-6)


def test_disturbances_declare_their_discontinuities():
    from plantbench.core import disturbances as dist
    assert dist.step("m", value=1.0, t=5.0).times == (5.0,)
    assert dist.ramp("m", value=1.0, t=5.0, duration=10.0).times == (5.0, 15.0)
    both = dist.combine(dist.step("m", value=1.0, t=5.0), dist.step("d", value=1.0, t=2.0))
    assert both.times == (2.0, 5.0)


def test_without_inputs_removes_the_writers_and_their_primaries():
    """On the two tanks, holding q1 removes its level loop and the cascade primary over it;
    the ratio station that follows q1 stays."""
    from _tanks import tanks_design, tanks_structure

    S = tanks_structure(tanks_design())
    kept, removed = ctl.without_inputs(S, ["q1"])
    assert [e.name for e in kept] == ["q2 ratio"]
    assert [e.name for e in removed] == ["level 2", "level 1"]
    assert ctl.without_inputs(S, ["d"]) == (S, [])


def test_augmented_states_start_at_rest_at_a_given_start(design):
    """Instrument and filter states start at the measurement at the start, the actuator at
    the design position of its valve, the integral at zero."""
    from plantbench.core.instruments import Actuator, Instrument

    loop = toy_loop(setpoint=2.0, bias=1.0, tau_D=1.0, instrument=Instrument(tau=1.0),
                    actuator=Actuator(tau=0.5))
    aug = ctl.initial_augmented([loop], design, np.array([3.0]))
    slots, n = ctl.state_layout([loop])
    i_slot, a_slot, f_slot = slots[0]
    assert aug[0] == 0.0 and np.all(aug[i_slot] == 3.0) and aug[f_slot][0] == 3.0
    assert np.all(aug[a_slot] == design.u.m)
    assert np.array_equal(ctl.initial_augmented([loop], design),
                          ctl.initial_augmented([loop], design, design.x))
