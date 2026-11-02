"""Configured plants stepped together: agreement with scalar stepping, independence of
the members, failures, noise, extremes and time limits."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
from _tanks import TANKS_CASE

import plantbench as pb
from plantbench.cases.template import CASE as TEMPLATE
from plantbench.core.control import WallTimeExceeded
from plantbench.core.ensemble_stepper import EnsembleStepper
from plantbench.core.limits import Monitor, Rule
from plantbench.core.stepper import StepFailed

CSTR = pb.load_case("jacketed_cstr")
# Feed temperatures stepped at 5.25 min, inside the interval [5, 5.5].
CONFIGS = [CSTR.config("temperature PI", disturbance={"name": "feed_temperature",
                                                      "value": v, "t": 5.25})
           for v in (330.0, 340.0, 345.0)]
DT = 0.5


def _relative(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def test_members_agree_with_scalar_steppers():
    """Rodas5P against LSODA, member by member, over 40 steps across the feed steps: the
    states, the measurements and the extremes to a relative 1e-6."""
    ens = EnsembleStepper.from_configs(CSTR, CONFIGS, DT, extrema=("T",))
    scalar = [pb.Stepper(pb.build(CSTR, c), DT, extrema=("T",)) for c in CONFIGS]
    for _ in range(40):
        e, s = ens.step(), [m.step() for m in scalar]
        assert e.t == s[0].t
        for i in range(len(CONFIGS)):
            assert _relative(e.steps[i].x, s[i].x) < 1e-6
            assert _relative(e.y["T"][i], s[i].y["T"]) < 1e-6
            assert _relative(e.y_max["T"][i], s[i].y_max["T"]) < 1e-6
            assert _relative(e.y_min["T"][i], s[i].y_min["T"]) < 1e-6


def test_an_owned_input_takes_one_value_per_member():
    """With the coolant flow owned the reactor is open-loop unstable and amplifies the
    solvers' errors: 2.0e-6 apart over 20 steps at the default tolerances, 2.7e-9 at 1e-10
    and 1e-12, so the members agree to 1e-5 here."""
    ens = EnsembleStepper.from_configs(CSTR, CONFIGS, DT, inputs=("Fj",))
    scalar = [pb.Stepper(pb.build(CSTR, c), DT, inputs=("Fj",)) for c in CONFIGS]
    for k in range(20):
        fj = np.array([0.9, 1.0, 1.1]) + 0.01 * k
        e = ens.step({"Fj": fj})
        for i, m in enumerate(scalar):
            assert _relative(e.steps[i].x, m.step({"Fj": float(fj[i])}).x) < 1e-5
        assert np.array_equal(e.u["Fj"], fj)
    e = ens.step({"Fj": 1.0})  # one value for every member
    assert np.array_equal(e.u["Fj"], np.ones(3))
    with pytest.raises(ValueError, match="one value per member"):
        ens.step({"Fj": [1.0, 1.0]})
    with pytest.raises(KeyError, match="exactly"):
        ens.step({"Tj": 1.0})


def test_a_member_does_not_depend_on_the_others():
    """A member alone, and the members in reverse order, step bit for bit as in the
    ensemble."""
    together = EnsembleStepper.from_configs(CSTR, CONFIGS, DT)
    alone = EnsembleStepper.from_configs(CSTR, CONFIGS[1:2], DT)
    backwards = EnsembleStepper.from_configs(CSTR, CONFIGS[::-1], DT)
    for _ in range(20):
        a, b, c = together.step(), alone.step(), backwards.step()
        assert np.array_equal(a.steps[1].x, b.steps[0].x)
        for i in range(3):
            assert np.array_equal(a.steps[i].x, c.steps[2 - i].x)
            assert a.steps[i].nfev == c.steps[2 - i].nfev


def test_a_failed_member_is_reported_and_the_others_go_on():
    """A coolant flow of NaN fails one member; it stays where its interval began and takes
    no further steps until `reset`, while the others step as they would alone."""
    ens = EnsembleStepper.from_configs(CSTR, CONFIGS, DT, inputs=("Fj",))
    rest = EnsembleStepper.from_configs(CSTR, [CONFIGS[0], CONFIGS[2]], DT, inputs=("Fj",))
    ens.step({"Fj": 1.0}), rest.step({"Fj": 1.0})
    x_before = ens.members[1].x.copy()
    e = ens.step({"Fj": [1.0, np.nan, 1.0]})
    r = rest.step({"Fj": 1.0})
    assert isinstance(e.errors[1], StepFailed) and e.steps[1] is None
    assert e.running.tolist() == [True, False, True]
    assert np.isnan(e.y["T"][1]) and np.array_equal(ens.members[1].x, x_before)
    assert np.array_equal(e.steps[0].x, r.steps[0].x) and np.array_equal(e.steps[2].x, r.steps[1].x)
    e = ens.step({"Fj": 1.0})
    assert e.steps[1] is None and e.steps[0] is not None
    ens.reset()
    assert ens.step({"Fj": 1.0}).running.all()


def test_each_member_draws_the_noise_of_its_configuration():
    configs = [CSTR.config("temperature PI", noise={"seed": s, "std": {"T": 0.5}})
               for s in (1, 2)]
    ens = EnsembleStepper.from_configs(CSTR, configs, DT)
    scalar = [pb.Stepper(pb.build(CSTR, c), DT) for c in configs]
    for _ in range(10):
        e, s = ens.step(), [m.step() for m in scalar]
        for i in range(2):
            noise = e.steps[i].y["T"] - e.steps[i].y_true["T"]
            assert noise == pytest.approx(s[i].y["T"] - s[i].y_true["T"], abs=1e-9)
    # Seeded again the other way round, member 0 draws what a stepper seeded 2 draws.
    ens.reset(noise_seeds=[2, 1])
    e = ens.observe()
    other = pb.Stepper(pb.build(CSTR, configs[1]), DT).observe()
    assert e.y["T"][0] - e.y_true["T"][0] == pytest.approx(other.y["T"] - other.y_true["T"],
                                                           abs=1e-12)


def test_a_member_can_be_monitored():
    ens = EnsembleStepper.from_configs(CSTR, CONFIGS, DT, extrema=("T",))
    monitors = [Monitor([Rule("high", "T", "above", 363.0)]) for _ in CONFIGS]
    for _ in range(60):
        e = ens.step()
        for m, st in zip(monitors, e.steps):
            m.step(st)
    hot = [m.first is not None for m in monitors]
    assert hot == [False, True, True]  # the 330 K feed cools the reactor


def test_incompatible_members_are_refused():
    with pytest.raises(ValueError, match="different designs"):
        EnsembleStepper.from_configs(CSTR, [CSTR.config(), CSTR.config(params={"U": 300.0})],
                                     DT)
    with pytest.raises(ValueError, match="one design object"):
        EnsembleStepper([pb.build(TANKS_CASE, TANKS_CASE.config()),
                         pb.build(TANKS_CASE, TANKS_CASE.config())], 1.0)
    with pytest.raises(ValueError, match="batched"):
        EnsembleStepper.from_configs(TEMPLATE, [TEMPLATE.config()], 1.0)
    with pytest.raises(ValueError, match="different designs"):
        EnsembleStepper.from_configs(CSTR, [CSTR.config("temperature PI"),
                                            CSTR.config("composition cascade")], DT)


def test_a_ratio_station_and_non_ideal_elements_step_as_alone():
    """The tanks' cascade, with a ratio station, a Pade deadtime and a rate-limited
    actuator: the ensemble agrees with scalar stepping."""
    configs = [TANKS_CASE.config(setpoints={"level 2": v}) for v in (1.0, 1.2)]
    ens = EnsembleStepper.from_configs(TANKS_CASE, configs, 1.0)
    scalar = [pb.Stepper(pb.build(TANKS_CASE, c), 1.0) for c in configs]
    for _ in range(20):
        e = ens.step()
        for i, m in enumerate(scalar):
            assert _relative(e.steps[i].x, m.step().x) < 1e-6


# -- time limits ------------------------------------------------------------------------


def _stall_ensemble(**budgets):
    from _stall import CASE

    return EnsembleStepper.from_configs(CASE, [CASE.config()] * 3, 1.0, inputs=("stall",),
                                        **budgets)


def test_an_ensemble_with_a_limit_steps_as_one_without():
    with EnsembleStepper.from_configs(CSTR, CONFIGS, DT, step_budget=30.0) as limited:
        free = EnsembleStepper.from_configs(CSTR, CONFIGS, DT)
        for _ in range(10):
            a, b = free.step(), limited.step()
            for i in range(3):
                assert np.array_equal(a.steps[i].x, b.steps[i].x)
        assert limited._worker.process.is_alive()
    assert not limited._worker.process.is_alive()


def test_a_stall_in_one_member_loses_the_whole_worker():
    ens = _stall_ensemble(step_budget=1.0)
    ens.step({"stall": 0.0})
    start = time.perf_counter()
    with pytest.raises(WallTimeExceeded, match="exceeded 1 s"):
        ens.step({"stall": [0.0, 30.0, 0.0]})
    assert time.perf_counter() - start < 10.0
    assert not ens._worker.process.is_alive()
    with pytest.raises(RuntimeError, match="every member of its worker is lost"):
        ens.step({"stall": 0.0})
    assert [m.t for m in ens.members] == [1.0, 1.0, 1.0]
    ens.close()


def test_an_ensemble_of_cases_no_module_holds_cannot_have_a_limit():
    case = dataclasses.replace(TANKS_CASE, id="tanks2")
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        EnsembleStepper.from_configs(case, [case.config()], 1.0, step_budget=5.0)
