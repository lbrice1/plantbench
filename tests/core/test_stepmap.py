"""The stepped map of a configured plant: fixed points, Jacobians and spectra."""

from __future__ import annotations

import numpy as np
import pytest
from _tanks import TANKS_CASE

import plantbench as pb
from plantbench.core import stepmap
from plantbench.units import cstr

CSTR = pb.load_case("jacketed_cstr")
DT = 0.5


def _matched(mu, expected):
    """The largest distance between two sets of eigenvalues, each matched to its nearest."""
    return max(np.min(np.abs(np.asarray(expected) - m)) for m in mu)


def test_with_nothing_owned_the_map_is_the_exponential_of_the_closed_loop():
    """CSTR under `temperature PI` at dt = 0.5 min: the eigenvalues of the map are
    exp(lambda dt) of the continuous closed loop, the slowest 0.97452."""
    setup = pb.build(CSTR, CSTR.config("temperature PI"))
    lin = stepmap.linearize(pb.Stepper(setup, DT))
    expected = np.exp(setup.spectrum() * DT)
    assert len(lin.eigenvalues) == len(expected)
    assert _matched(lin.eigenvalues, expected) < 1e-5
    assert lin.eigenvalues[0].real == pytest.approx(0.97452, abs=1e-5)
    assert lin.stable and lin.slowest_stable() == 0
    slowest = setup.spectrum()[np.argmax(setup.spectrum().real)]
    assert lin.decay_time == pytest.approx(-1.0 / slowest.real, rel=1e-4)
    assert np.allclose(lin.continuous, setup.spectrum()[np.argsort(
        -np.abs(np.exp(setup.spectrum() * DT)), kind="stable")], atol=1e-4)
    assert lin.names == ["C_A", "T", "Tj", "reactor T.integral"]
    # The shares depend on the scales: over 45 K, the temperature's span, rather than its
    # magnitude of 362.40 K, its share relative to the concentration's grows by the
    # square of their ratio.
    by_value = lin.shares()
    by_span = lin.shares(scale={"T": 45.0})
    assert sum(by_span.values()) == pytest.approx(1.0)
    assert by_span["T"] / by_span["C_A"] == pytest.approx(
        by_value["T"] / by_value["C_A"] * (lin.y[1] / 45.0) ** 2, rel=1e-9)


def test_an_owned_input_held_gives_the_open_loop_map():
    """With the coolant flow owned and held at its design value, the map is that of the
    open-loop reactor: exp(lambda dt) of its eigenvalues, 1.03048, 0.93475 and 0.67067,
    the first unstable."""
    setup = pb.build(CSTR, CSTR.config("temperature PI"))
    lin = stepmap.linearize(pb.Stepper(setup, DT, inputs=("Fj",)))
    open_loop = np.exp(cstr.eigenvalues(setup.design.x, setup.design.pp) * DT)
    assert _matched(lin.eigenvalues, open_loop) < 1e-5
    assert np.round(lin.eigenvalues.real, 5).tolist() == [1.03048, 0.93475, 0.67067]
    assert not lin.stable and lin.slowest_stable() == 1
    assert lin.decay_time == pytest.approx(-DT / np.log(0.93475), rel=1e-4)
    assert np.allclose(lin.y, setup.design.x, rtol=1e-9)
    shares = lin.shares()
    assert set(shares) == {"C_A", "T", "Tj"} and sum(shares.values()) == pytest.approx(1.0)


def test_the_fixed_point_moves_with_the_plant_and_the_action():
    """Under a mismatch of the heat-transfer coefficient the integral action brings the
    temperature back to its set-point at another coolant flow; an owned set-point moves
    the fixed point to the action."""
    setup = pb.build(CSTR, CSTR.config("temperature PI", mismatch={"U": 300.0}))
    s = pb.Stepper(setup, DT)
    lin = stepmap.linearize(s)
    F = stepmap.StepMap(s)
    assert lin.residual < 1e-8
    assert np.max(np.abs(F(lin.y) - lin.y)) < 1e-6
    assert lin.y[1] == pytest.approx(setup.design.x[1], abs=1e-6)
    assert abs(lin.y[3]) > 1e-3  # the integral holds the new coolant flow
    owned = pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI")), DT,
                       setpoints=("reactor T",))
    lin = stepmap.linearize(owned, {"reactor T": 360.0})
    assert lin.y[1] == pytest.approx(360.0, abs=1e-6)


def test_the_integral_of_a_ratio_station_is_not_a_coordinate():
    """The tanks' cascade has a ratio station, whose integral never changes; the map's
    eigenvalues are exp(lambda dt) of the spectrum with its structural zero removed."""
    setup = pb.build(TANKS_CASE, TANKS_CASE.config())
    lin = stepmap.linearize(pb.Stepper(setup, 1.0))
    assert "q2 ratio.integral" not in lin.names
    assert len(lin.names) == len(setup.spectrum())
    assert _matched(lin.eigenvalues, np.exp(setup.spectrum() * 1.0)) < 1e-5


def test_the_analysis_leaves_the_stepper_and_its_noise_as_they_were():
    config = CSTR.config("temperature PI", noise={"seed": 3, "std": {"T": 0.5}})
    s, twin = (pb.Stepper(pb.build(CSTR, config), DT) for _ in range(2))
    s.step(), twin.step()
    stepmap.linearize(s)
    a, b = s.step(), twin.step()
    assert a.t == b.t and np.array_equal(a.x, b.x) and a.y == b.y


def test_a_map_that_depends_on_time_is_refused():
    feed = {"name": "feed_temperature", "value": 340.0, "t": 10.0}
    with pytest.raises(ValueError, match="depends on time"):
        stepmap.StepMap(pb.Stepper(pb.build(CSTR, CSTR.config(disturbance=feed)), DT))
    steps = CSTR.config(setpoint_steps={"reactor T": [[5.0, 1.0]]})
    with pytest.raises(ValueError, match="depends on time"):
        stepmap.StepMap(pb.Stepper(pb.build(CSTR, steps), DT))


def test_a_fixed_point_not_found_is_reported():
    """A map that moves every point by one unit has no fixed point."""
    F = stepmap.StepMap(pb.Stepper(pb.build(CSTR, CSTR.config("temperature PI")), DT))

    class Shifted:
        start = F.start

        def __call__(self, y):
            return F(y) + 1.0 + (y - F(y))

    with pytest.raises(RuntimeError, match="no fixed point"):
        stepmap.fixed_point(Shifted())
