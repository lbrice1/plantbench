"""The stepper against `run`: with nothing owned, or with owned inputs and set-points at
the values a configuration would give them, it integrates the same closed loop and gives
the same trajectory to within the integrator's tolerance."""

from __future__ import annotations

import numpy as np
import pytest
from _tanks import TANKS_CASE

import plantbench as pb
from plantbench.cases import load_case
from plantbench.cases.jacketed_cstr.definition import TEMPERATURE_TUNING
from plantbench.cases.template import CASE as TEMPLATE
from plantbench.core import stepper as stp
from plantbench.core.control import Loop

RTOL = 1e-5  # the restarts at every step change the solver's path, not the closed loop
CSTR = load_case("jacketed_cstr")
FEED = {"name": "feed_temperature", "value": 340.0, "t": 10.0}


def stepped(setup, dt, t_end, action=None, **owned):
    s = pb.Stepper(setup, dt, **owned)
    xs = [s.x.copy()]
    for _ in range(int(round(t_end / dt))):
        xs.append(s.step(action).x)
    return np.array(xs).T


def close(a, b):
    """Largest difference relative to each state's magnitude."""
    scale = np.maximum(np.abs(b).max(axis=1, keepdims=True), 1e-6)
    return np.max(np.abs(a - b) / scale)


CASES = [
    (TEMPLATE, {"structure": "level PI",
                "disturbance": {"name": "step", "field": "F_in", "fraction": 0.2, "t": 5.0}},
     1.0, 60.0),
    (CSTR, {"structure": "temperature PI", "disturbance": FEED}, 0.5, 100.0),
    # Breakpoints inside the intervals: the disturbance at 10.3 and a set-point step at 7.5.
    (CSTR, {"structure": "composition cascade", "setpoint_steps": {"reactor C_A": [(7.5, -0.1)]},
            "disturbance": {"name": "step", "field": "CA0", "fraction": 0.05, "t": 10.3}},
     1.0, 100.0),
    (load_case("reactor_separator_recycle"),
     {"disturbance": {"name": "throughput", "fraction": 0.10}}, 1.0, 100.0),
]


@pytest.mark.parametrize("case, cfg, dt, t_end", CASES,
                         ids=["template", "cstr", "cstr breakpoints", "rsr"])
def test_nothing_owned_matches_run(case, cfg, dt, t_end):
    config = case.config(**cfg)
    ref = pb.run(case, config, t_end=t_end, dt=dt)
    assert close(stepped(pb.build(case, config), dt, t_end), ref.x) < RTOL


def test_an_owned_input_at_its_design_value_is_the_open_loop():
    """Fj owned under `temperature PI` is the open-loop reactor; the 340 K feed then
    carries it to the high branch, as the case card states."""
    ref = pb.run(CSTR, CSTR.config("open loop", disturbance=FEED), t_end=200.0, dt=0.5)
    setup = pb.build(CSTR, CSTR.config("temperature PI", disturbance=FEED))
    got = stepped(setup, 0.5, 200.0, inputs=("Fj",))
    assert close(got, ref.x) < RTOL and got[1, -1] > 370.0


def test_an_owned_set_point_is_the_configured_one():
    ref = pb.run(CSTR, CSTR.config("temperature PI", setpoints={"reactor T": 365.0}),
                 t_end=100.0, dt=0.5)
    got = stepped(pb.build(CSTR, CSTR.config("temperature PI")), 0.5, 100.0,
                  action={"reactor T": 365.0}, setpoints=("reactor T",))
    assert close(got, ref.x) < RTOL


def test_the_displaced_loops_are_loops_with_their_tuning():
    s = pb.Stepper(pb.build(CSTR, CSTR.config("composition cascade")), 0.5, inputs=("Fj",))
    assert [e.name for e in s.displaced] == ["reactor T", "reactor C_A"]
    assert all(isinstance(e, Loop) for e in s.displaced)
    assert s.displaced[0].Kc == TEMPERATURE_TUNING["Kc"] and s.structure == []


def test_a_ratio_station_follows_an_owned_input():
    s = pb.Stepper(pb.build(TANKS_CASE, TANKS_CASE.config()), 1.0, inputs=("q1",))
    assert [e.name for e in s.structure] == ["q2 ratio"]
    step = s.step({"q1": 1.3})
    assert step.u["q1"] == 1.3 and step.u["q2"] == pytest.approx(s.structure[0].ratio * 1.3)


def test_time_state_and_observations_are_kept_between_steps():
    s = pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI", disturbance=FEED)), 0.5,
                   inputs=("Fj",))
    first = s.observe()
    assert first.t == 0.0 and first.u["Fj"] == s.design.u.Fj and first.nfev == 0
    for _ in range(3):
        last = s.step({"Fj": 0.4})
    assert s.t == 1.5 and last.t == 1.5 and np.array_equal(last.x, s.x)
    assert last.u["Fj"] == 0.4 and last.y["T"] == s.x[1] and last.nfev > 0
    assert s.observe().y == last.y and s.nfev >= last.nfev
    s.reset()
    assert s.t == 0.0 and np.array_equal(s.x, s.setup.x0) and s.action == {"Fj": s.design.u.Fj}


def test_the_stepper_starts_where_the_configuration_does():
    setup = pb.build(CSTR, CSTR.config("temperature PI", x0={"T": 370.0},
                                       instruments={"reactor T": {"tau": 1.0}}))
    s = pb.Stepper(setup, 0.5)
    assert s.x[1] == 370.0 and s.observe().c["reactor T.PV"] == 370.0
    ref = pb.run(CSTR, setup.config, t_end=50.0, dt=0.5)
    assert close(stepped(setup, 0.5, 50.0), ref.x) < RTOL


@pytest.mark.parametrize("config, owned, error, match", [
    ({}, {"inputs": ("nope",)}, KeyError, "cannot own"),
    ({"hold": {"Fj": 0.5}}, {"inputs": ("Fj",)}, ValueError, "held"),
    ({"disturbance": {"name": "step", "field": "T0", "value": 340.0}}, {"inputs": ("T0",)},
     ValueError, "disturbance changes"),
    ({}, {"inputs": ("Fj",), "setpoints": ("reactor T",)}, ValueError, "removed"),
    ({}, {"setpoints": ("nope",)}, KeyError, "no loop"),
    ({"structure": "composition cascade"}, {"setpoints": ("reactor T",)}, ValueError,
     "cascade primary"),
    ({"setpoint_steps": {"reactor T": [(5.0, 1.0)]}}, {"setpoints": ("reactor T",)},
     ValueError, "schedule"),
])
def test_conflicting_ownership_is_refused(config, owned, error, match):
    setup = pb.build(CSTR, CSTR.config(**{"structure": "temperature PI", **config}))
    with pytest.raises(error, match=match):
        pb.Stepper(setup, 0.5, **owned)


def test_a_ratio_station_has_no_set_point_and_actions_name_what_is_owned():
    setup = pb.build(TANKS_CASE, TANKS_CASE.config())
    with pytest.raises(TypeError, match="ratio station"):
        pb.Stepper(setup, 1.0, setpoints=("q2 ratio",))
    with pytest.raises(ValueError, match="positive"):
        pb.Stepper(setup, 0.0)
    s = pb.Stepper(setup, 1.0, inputs=("q1",))
    with pytest.raises(KeyError, match="exactly"):
        s.step({"q1": 1.0, "q2": 0.5})


def test_a_failed_interval_leaves_the_stepper_where_it_was(monkeypatch):
    s = pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI")), 0.5)
    s.step()
    t, x = s.t, s.x.copy()

    class Failed:
        success, message = False, "forced"

    monkeypatch.setattr(stp, "solve_ivp", lambda *a, **k: Failed())
    with pytest.raises(stp.StepFailed, match="forced"):
        s.step()
    assert s.t == t and np.array_equal(s.x, x)
