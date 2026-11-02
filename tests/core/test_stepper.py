"""The stepper against `run`: with nothing owned, or with owned inputs and set-points at
the values a configuration would give them, it integrates the same closed loop and gives
the same trajectory to within the integrator's tolerance."""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest
from _tanks import TANKS_CASE

import plantbench as pb
from plantbench.cases import load_case
from plantbench.cases.jacketed_cstr.definition import TEMPERATURE_TUNING
from plantbench.cases.template import CASE as TEMPLATE
from plantbench.core import stepper as stp
from plantbench.core.control import Loop, WallTimeExceeded

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


# -- time limits ------------------------------------------------------------------


def _stall_stepper(**budgets):
    from _stall import CASE

    return pb.Stepper(pb.build(CASE, CASE.config()), 1.0, inputs=("stall",), **budgets)


def test_a_stepper_with_a_limit_steps_as_one_without():
    config = CSTR.config("temperature PI", disturbance=FEED, x0={"T": 365.0})
    alone = pb.Stepper(pb.build(CSTR, config), 0.5, inputs=("Fj",))
    with pb.Stepper(pb.build(CSTR, config), 0.5, inputs=("Fj",), step_budget=30.0) as limited:
        for k in range(20):
            a, b = alone.step({"Fj": 0.3 + 0.01 * k}), limited.step({"Fj": 0.3 + 0.01 * k})
            assert np.array_equal(a.x, b.x) and a.nfev == b.nfev and a.y == b.y
        assert limited.spent > 0.0
        assert limited._worker.process.is_alive()
    assert not limited._worker.process.is_alive()


def test_a_step_stalled_in_native_code_is_stopped_and_the_stepper_invalidated():
    s = _stall_stepper(step_budget=1.0)
    s.step({"stall": 0.0})
    start = time.perf_counter()
    with pytest.raises(WallTimeExceeded, match="exceeded 1 s"):
        s.step({"stall": 30.0})
    assert time.perf_counter() - start < 10.0
    assert not s._worker.process.is_alive()
    with pytest.raises(RuntimeError, match="cannot step again"):
        s.step({"stall": 0.0})
    assert s.t == 1.0  # where the stalled step began
    s.close()  # a second close does nothing


def test_the_episode_budget_counts_the_steps_only():
    s = _stall_stepper(episode_budget=1.5)
    time.sleep(2.0)  # the caller's own time is not the episode's
    s.step({"stall": 0.0})
    with pytest.raises(WallTimeExceeded):
        for _ in range(100):
            s.step({"stall": 0.05})
    assert 1.5 <= s.spent < 10.0 and not s._worker.process.is_alive()


def test_reset_starts_a_new_episode():
    s = _stall_stepper(episode_budget=60.0)
    s.step({"stall": 0.0})
    s.reset()
    assert s.spent == 0.0 and s.t == 0.0
    assert s.step({"stall": 0.0}).x == pytest.approx([1.0])
    s.close()


def test_a_limit_works_from_a_thread_other_than_the_main_one():
    caught = []

    def run():
        s = _stall_stepper(step_budget=1.0)
        try:
            s.step({"stall": 30.0})
        except WallTimeExceeded as exc:
            caught.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(30.0)
    assert not thread.is_alive() and len(caught) == 1


def test_a_worker_that_fails_or_stalls_at_its_start_is_reported(monkeypatch):
    from _stall import CASE, FAIL, SLOW

    setup = pb.build(CASE, CASE.config())
    monkeypatch.setenv(FAIL, "1")
    with pytest.raises(ValueError, match="told to fail"):
        pb.Stepper(setup, 1.0, step_budget=1.0)
    monkeypatch.delenv(FAIL)
    monkeypatch.setenv(SLOW, "30")
    with pytest.raises(WallTimeExceeded, match="start"):
        pb.Stepper(setup, 1.0, step_budget=1.0, startup_budget=1.0)


def test_a_case_that_no_module_holds_cannot_have_a_limit():
    case = dataclasses.replace(CSTR, title="changed after import")
    setup = pb.build(case, case.config())
    with pytest.raises(ValueError, match="cannot be rebuilt in a worker"):
        pb.Stepper(setup, 0.5, step_budget=1.0)
    pb.Stepper(setup, 0.5).step()  # without a limit it steps


def test_a_setup_not_built_from_its_configuration_cannot_have_a_limit():
    setup = pb.build(CSTR, CSTR.config())
    setup = dataclasses.replace(setup, design=dataclasses.replace(setup.design,
                                                                  x=setup.design.x + 1.0))
    with pytest.raises(ValueError, match="differs from the setup"):
        pb.Stepper(setup, 0.5, step_budget=5.0)


def test_a_failed_interval_in_the_worker_is_raised_and_the_worker_kept(monkeypatch):
    s = pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI")), 0.5, step_budget=30.0)
    s.step()
    with pytest.raises(KeyError, match="exactly"):
        s.step({"Fj": 1.0})
    assert s._worker.process.is_alive()
    s.close()


def test_a_worker_that_dies_is_reported_and_the_stepper_invalidated():
    s = _stall_stepper(step_budget=30.0)
    s.step({"stall": 0.0})
    s._worker.process.kill()
    s._worker.process.join()
    with pytest.raises(RuntimeError, match="worker ended (before|during)"):
        s.step({"stall": 0.0})
    with pytest.raises(RuntimeError, match="cannot step again"):
        s.step({"stall": 0.0})


# -- measurement noise ------------------------------------------------------------

NOISE = {"seed": 3, "std": {"T": 0.5}}


def _measured(config, steps=20, **owned):
    s = pb.Stepper(pb.build(CSTR, config), 0.5, **owned)
    out = [s.observe()]
    out += [s.step() for _ in range(steps)]
    return s, out


def test_noise_is_drawn_on_the_measurement_and_kept_out_of_the_plant():
    base = CSTR.config("temperature PI", disturbance=FEED)
    _, clean = _measured(base)
    _, noisy = _measured(CSTR.config("temperature PI", disturbance=FEED, noise=NOISE))
    for a, b in zip(clean, noisy):
        assert np.array_equal(a.x, b.x) and a.y == b.y_true and a.c == b.c
        assert b.y["C_A"] == b.y_true["C_A"] and b.y["T"] != b.y_true["T"]


def test_the_noise_has_its_standard_deviation():
    case = TEMPLATE
    s = pb.Stepper(pb.build(case, case.config(noise={"seed": 0, "std": {"level": 0.2}})), 1.0)
    e = np.array([(lambda st: st.y["level"] - st.y_true["level"])(s.step()) for _ in range(2000)])
    assert abs(e.mean()) < 4 * 0.2 / np.sqrt(e.size)
    assert e.std() == pytest.approx(0.2, rel=0.1)


def test_noise_is_reproducible_from_its_seed_and_reset():
    def errors(stepper, n=10):
        return [(lambda st: st.y["T"] - st.y_true["T"])(stepper.step()) for _ in range(n)]

    config = CSTR.config("temperature PI", noise=NOISE)
    s = pb.Stepper(pb.build(CSTR, config), 0.5)
    first = [s.observe().y["T"] - s.observe().y_true["T"], *errors(s)]
    s.reset()
    assert [s.observe().y["T"] - s.observe().y_true["T"], *errors(s)] == first
    other = pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI", noise=NOISE)), 0.5)
    assert errors(other) == first[1:]
    s.reset(noise_seed=4)
    assert errors(s) != first[1:]


def test_the_noise_seed_and_the_scenario_seed_are_independent():
    rand = {"name": "random_sequence", "of": "feed_temperature", "level": "change",
            "low": -5.0, "high": 5.0, "dwell": [2.0, 4.0], "start": 1.0, "end": 10.0}

    def run(scenario_seed, noise_seed):
        _, out = _measured(CSTR.config("temperature PI", disturbance={**rand, "seed": scenario_seed},
                                       noise={"seed": noise_seed, "std": {"T": 0.5}}))
        return (np.array([o.x for o in out]),
                [o.y["T"] - o.y_true["T"] for o in out])

    x, e = run(1, 1)
    x_noise, e_noise = run(1, 2)
    x_scenario, e_scenario = run(2, 1)
    assert np.array_equal(x, x_noise) and e != e_noise
    assert not np.array_equal(x, x_scenario) and e == e_scenario


def test_observing_again_draws_nothing_new():
    s = pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI", noise=NOISE)), 0.5)
    s.step()
    assert s.observe().y == s.observe().y


def test_a_continuous_run_refuses_noise(tmp_path):
    config = CSTR.config("temperature PI", noise=NOISE)
    with pytest.raises(ValueError, match="stepped run"):
        pb.run(CSTR, config, t_end=10.0)
    results = pb.run_ensemble(CSTR, [config], t_end=10.0)
    assert isinstance(results[0].error, ValueError)


@pytest.mark.parametrize("noise, error", [
    ({"seed": 1, "std": {"nope": 0.1}}, KeyError),
    ({"seed": 1, "std": {"T": -0.1}}, ValueError),
    ({"seed": 1.5, "std": {"T": 0.1}}, TypeError),
    ({"seed": 1}, KeyError),
    ({"std": {"T": 0.1}, "sigma": 1}, KeyError),
])
def test_malformed_noise_is_refused(noise, error):
    with pytest.raises(error):
        pb.build(CSTR, CSTR.config("temperature PI", noise=noise))
