"""Disturbances act on any inputs dataclass and never change the nominal inputs."""

from __future__ import annotations

import pytest
from _toy import ToyInputs

from plantbench.core import disturbances as dist


def test_step_by_fraction_and_to_a_value():
    u0 = ToyInputs(m=2.0)
    assert dist.step("m", fraction=0.1, t=5.0)(4.9, u0) is u0
    assert dist.step("m", fraction=0.1, t=5.0)(5.0, u0).m == pytest.approx(2.2)
    assert dist.step("m", value=7.0, t=5.0)(6.0, u0).m == 7.0
    assert u0.m == 2.0


def test_step_needs_exactly_one_target():
    with pytest.raises(ValueError):
        dist.step("m", t=0.0)(1.0, ToyInputs())
    with pytest.raises(ValueError):
        dist.step("m", fraction=0.1, value=1.0, t=0.0)(1.0, ToyInputs())


def test_step_on_a_field_the_inputs_lack():
    with pytest.raises(AttributeError, match="nope"):
        dist.step("nope", value=1.0, t=0.0)(1.0, ToyInputs())


def test_ramp_is_linear_and_then_holds():
    d = dist.ramp("m", value=3.0, t=10.0, duration=20.0)
    u0 = ToyInputs(m=1.0)
    assert d(20.0, u0).m == pytest.approx(2.0)
    assert d(100.0, u0).m == pytest.approx(3.0)


def test_combine_applies_in_order():
    d = dist.combine(dist.step("m", value=4.0, t=0.0), dist.step("m", fraction=0.5, t=0.0))
    assert d(1.0, ToyInputs(m=1.0)).m == pytest.approx(6.0)


def test_a_sequence_takes_each_level_from_the_nominal_inputs():
    """Two fractional steps combined compound; the same levels as a sequence do not."""
    u0 = ToyInputs(m=2.0)
    d = dist.step_sequence("m", [10.0, 20.0, 30.0], [0.1, 0.2, -0.1])
    assert d(5.0, u0) is u0
    assert [d(t, u0).m for t in (10.0, 19.9, 20.0, 35.0)] == pytest.approx([2.2, 2.2, 2.4, 1.8])
    assert d.times == (10.0, 20.0, 30.0) and d.levels == (0.1, 0.2, -0.1)
    chained = dist.combine(dist.step("m", fraction=0.1, t=10.0), dist.step("m", fraction=0.2, t=20.0))
    assert chained(25.0, u0).m == pytest.approx(2.64)
    assert u0.m == 2.0


def test_a_sequence_of_values_and_of_ramps():
    u0 = ToyInputs(m=1.0)
    assert dist.step_sequence("m", [0.0, 5.0], [3.0, 4.0], kind="value")(6.0, u0).m == 4.0
    d = dist.sequence([dist.ramp("m", value=3.0, t=10.0, duration=10.0),
                       dist.step("m", value=0.0, t=30.0)], [10.0, 30.0])
    assert d(15.0, u0).m == pytest.approx(2.0) and d(31.0, u0).m == 0.0
    assert d.times == (10.0, 20.0, 30.0)  # the ramp's end is declared as well


@pytest.mark.parametrize("call", [
    lambda: dist.step_sequence("m", [1.0, 1.0], [0.1, 0.2]),
    lambda: dist.step_sequence("m", [1.0, 2.0], [0.1]),
    lambda: dist.step_sequence("m", [1.0], [0.1], kind="change"),
    lambda: dist.sequence([], []),
])
def test_a_malformed_sequence_is_refused(call):
    with pytest.raises(ValueError):
        call()


def test_random_levels_are_reproducible_and_inside_their_bounds():
    args = dict(low=-0.1, high=0.2, dwell=[5.0, 15.0], start=10.0, end=500.0)
    times, levels = dist.random_levels(seed=7, **args)
    assert (times, levels) == dist.random_levels(seed=7, **args)
    assert (times, levels) != dist.random_levels(seed=8, **args)
    assert times[0] == 10.0 and times[-1] < 500.0
    assert all(5.0 <= b - a <= 15.0 for a, b in zip(times, times[1:]))
    assert all(-0.1 <= v <= 0.2 for v in levels) and len(set(levels)) == len(levels)
    _, binary = dist.random_levels(seed=7, kind="binary", **args)
    assert set(binary) == {-0.1, 0.2}


@pytest.mark.parametrize("change", [{"kind": "gaussian"}, {"dwell": [0.0, 1.0]},
                                    {"dwell": [2.0, 1.0]}, {"low": 1.0}, {"end": 10.0}])
def test_malformed_random_levels_are_refused(change):
    args = {"seed": 0, "low": 0.0, "high": 0.5, "dwell": [1.0, 2.0], "start": 10.0,
            "end": 50.0, **change}
    with pytest.raises(ValueError):
        dist.random_levels(**args)
