"""Operating limits of the cases, and the monitor of the outcomes they define."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import plantbench as pb
from plantbench import contract
from plantbench.cases.template import CASE as TEMPLATE
from plantbench.core import control as ctl
from plantbench.core.casekit import limits_from
from plantbench.core.limits import (
    CHOICE,
    SOURCE,
    Limit,
    Limits,
    Monitor,
    Rule,
    Span,
    check,
    limit_rules,
)

CSTR = pb.load_case("jacketed_cstr")
RSR = pb.load_case("reactor_separator_recycle")


# -- declaring limits ------------------------------------------------------------------


def test_limits_are_ordered_and_tagged():
    Limits(span=Span(0.0, 1.0, SOURCE), trip_lo=Limit(0.0, CHOICE), alarm_lo=Limit(0.1, CHOICE),
           alarm_hi=Limit(0.9, CHOICE), trip_hi=Limit(1.0, CHOICE))
    with pytest.raises(ValueError, match="ordered"):
        Limits(alarm_lo=Limit(0.2, CHOICE), trip_lo=Limit(0.3, CHOICE))
    with pytest.raises(ValueError, match="ordered"):
        Limits(alarm_lo=Limit(0.5, CHOICE), alarm_hi=Limit(0.5, CHOICE))
    with pytest.raises(ValueError, match="basis"):
        Limit(1.0, "guess")
    with pytest.raises(ValueError, match="larger hi"):
        Span(1.0, 1.0, SOURCE)


def test_a_table_of_plain_data_gives_the_limits():
    lim = limits_from({"T": {"span": (340.0, 385.0, "why"), "trip_hi": (372.0,)}}, CHOICE,
                      steady_states={"T": [338.0, 362.0], "C_A": [4.0, 2.0]})
    assert lim["T"].span == Span(340.0, 385.0, CHOICE, "why")
    assert lim["T"].trip_hi == Limit(372.0, CHOICE)
    assert lim["C_A"] == Limits(steady_states=(4.0, 2.0))
    with pytest.raises(KeyError, match="trip_high"):
        limits_from({"T": {"trip_high": (372.0,)}}, CHOICE)


def test_the_reactor_has_one_set_of_spans_and_limits():
    """`jacketed_cstr` and the reactor of `reactor_separator_recycle` read one table."""
    own = CSTR.limits(pb.build(CSTR).design)
    rsr = RSR.limits(pb.build(RSR).design)
    for name in ("T", "C_A"):
        assert dataclasses.replace(own[name], steady_states=()) == rsr[f"reactor {name}"]
    assert own["T"].trip_hi.value == 372.0 and own["T"].alarm_lo.value == 350.0


def test_the_cstr_declares_its_three_steady_states_low_branch_first():
    design = pb.build(CSTR).design
    lim = CSTR.limits(design)
    assert np.round(lim["T"].steady_states, 2).tolist() == [338.22, 362.40, 377.08]
    # Hotter means more conversion: the concentration falls along the branches.
    assert list(lim["C_A"].steady_states) == sorted(lim["C_A"].steady_states, reverse=True)
    assert lim["T"].steady_states[1] == pytest.approx(design.x[1], rel=1e-12)


def test_the_steady_states_are_those_of_the_plant_at_its_inputs():
    """A held coolant flow moves the branches; each one reported is a zero of the plant's
    right-hand side at the held input."""
    design = pb.build(CSTR, CSTR.config("open loop", hold={"Fj": 1.3})).design
    lim = CSTR.limits(design)
    nominal = CSTR.limits(pb.build(CSTR).design)
    assert lim["T"].steady_states != nominal["T"].steady_states
    for x in zip(*(lim[k].steady_states for k in ("C_A", "T", "Tj"))):
        r = design.rhs(0.0, np.array(x), design.u, design.pp)
        assert np.max(np.abs(r)) < 1e-8


def _span_of_each_loop(case, setup, by_measurement):
    lim = case.limits(setup.design)
    return {e.name: (e.pv_range, lim[by_measurement[e.name]].span)
            for e in setup.structure if getattr(e, "pv_range", None) is not None}


RSR_LOOPS = {"reactor T": "reactor T", "reactor C_A": "reactor C_A",
             "reactor level": "reactor V", "drum level": "drum level",
             "base level": "base level", "tray T": "tray 17 T", "feed T": "feed T"}


@pytest.mark.parametrize("network", ["none", "D3", "D4"])
def test_the_loops_spans_are_the_measurements_spans(network):
    setup = pb.build(RSR, RSR.config(options={"network": network}))
    pairs = _span_of_each_loop(RSR, setup, RSR_LOOPS)
    assert "tray T" in pairs
    for name, (pv_range, span) in pairs.items():
        assert pv_range == (span.lo, span.hi), name


def test_the_tray_span_follows_the_column_pressure():
    """At 0.4 and 0.15 bar the tray runs at 361.20 and 330.75 K, below the 360 to 410 K
    of 0.8 bar; its signal in percent of span stays between 0 and 100 at the design."""
    for network, span in (("D3", (340.0, 390.0)), ("D4", (310.0, 360.0))):
        setup = pb.build(RSR, RSR.config(options={"network": network}))
        loop = next(e for e in setup.structure if e.name == "tray T")
        assert loop.pv_range == span
        pv = loop.measure(setup.design.x, setup.design.u, setup.design.pp)
        assert 0.0 < ctl._percent(pv, *span) < 100.0


def test_the_template_spans_twice_its_design_level():
    lim = TEMPLATE.limits(pb.build(TEMPLATE, TEMPLATE.config(params={"h_design": 1.0})).design)
    assert (lim["level"].span.lo, lim["level"].span.hi) == (0.0, 2.0)
    assert lim["level"].alarm_lo.value == pytest.approx(0.1)
    assert lim["level"].alarm_hi.value == pytest.approx(1.9)


def test_limits_that_exclude_the_design_point_break_the_contract():
    bad = dataclasses.replace(TEMPLATE, limits=lambda d: limits_from(
        {"level": {"trip_hi": (1.0,)}}, CHOICE))
    with pytest.raises(contract.ContractError, match="beyond its trip limits"):
        contract.limits_hold_at_the_design_point(bad)
    unknown = dataclasses.replace(TEMPLATE, limits=lambda d: {"depth": Limits()})
    with pytest.raises(contract.ContractError, match="depth"):
        contract.limits_hold_at_the_design_point(unknown)


def test_rules_from_limits():
    rules = limit_rules(RSR.limits(pb.build(RSR).design), "alarm", outcome="inventory")
    assert Rule("inventory", "drum level", "above", 0.95 * 24.0) in rules
    assert Rule("inventory", "reactor T", "below", 350.0) in rules
    assert limit_rules(CSTR.limits(pb.build(CSTR).design), "trip") == [
        Rule("trip", "T", "above", 372.0)]


# -- the monitor -----------------------------------------------------------------------


def _run(monitor, values, extremes=None, dt=1.0):
    """Feed the monitor one sample per value, t = 0, dt, 2 dt, ..., with the extremes of
    the interval before each where given."""
    for k, v in enumerate(values):
        lo, hi = (None, None) if not extremes or extremes[k] is None else extremes[k]
        monitor.observe(k * dt, {"T": v}, None if lo is None else {"T": lo},
                        None if hi is None else {"T": hi})
    return monitor


def test_a_dwell_is_counted_from_the_last_sample_within():
    """Below 350 from the sample at 2: the crossing lies after the sample at 1, so the
    20 min dwell is reached at 21, not at 22."""
    rule = Rule("extinction", "T", "below", 350.0, dwell=20.0)
    m = _run(Monitor([rule]), [360.0, 355.0] + [349.0] * 25)
    v = m.violations[rule]
    assert (v.t, v.since, v.value, v.sampled) == (21.0, 1.0, 349.0, True)
    assert m.first is v and m.outcome == "extinction"


def test_a_return_within_restarts_the_dwell():
    rule = Rule("extinction", "T", "below", 350.0, dwell=5.0)
    m = _run(Monitor([rule]), [349.0] * 4 + [351.0] + [349.0] * 4)
    assert rule not in m.violations  # 4 from 0 to 3, 4 from 4 to 8
    m = _run(Monitor([rule]), [349.0] * 9)
    assert m.violations[rule].t == 5.0


def test_an_interval_back_within_restarts_the_dwell():
    """Every sample is beyond, but the interval from 3 to 4 reaches 352 K: the plant came
    back within, so the time beyond starts again at 3."""
    rule = Rule("extinction", "T", "below", 350.0, dwell=5.0)
    extremes = [None] * 4 + [(349.0, 352.0)] + [(349.0, 349.0)] * 6
    m = _run(Monitor([rule]), [349.0] * 11, extremes)
    assert (m.violations[rule].t, m.violations[rule].since) == (8.0, 3.0)


def test_a_crossing_between_samples_is_an_estimate():
    rule = Rule("trip", "T", "above", 372.0)
    m = _run(Monitor([rule]), [360.0, 365.0, 368.0], [None, (360.0, 366.0), (365.0, 373.5)])
    v = m.violations[rule]
    assert (v.t, v.since, v.value, v.sampled) == (2.0, 1.0, 373.5, False)
    # A sample beyond is reported as sampled, even with the extremes known.
    m = _run(Monitor([rule]), [360.0, 373.0], [None, (360.0, 374.0)])
    assert m.violations[rule].sampled and m.violations[rule].value == 373.0


def test_a_dwell_is_not_met_between_samples():
    rule = Rule("extinction", "T", "below", 350.0, dwell=0.5)
    m = _run(Monitor([rule]), [360.0, 360.0], [None, (340.0, 360.0)])
    assert not m.violations


def test_a_value_at_the_level_is_within():
    rule = Rule("trip", "T", "above", 372.0)
    assert not _run(Monitor([rule]), [372.0, 372.0]).violations


def test_the_first_violation_of_several_rules():
    trip = Rule("trip", "T", "above", 372.0)
    cold = Rule("extinction", "T", "below", 350.0, dwell=2.0)
    m = _run(Monitor([trip, cold]), [349.0, 349.0, 349.0, 380.0])
    assert m.first.rule == cold and m.violations[trip].t == 3.0
    m.reset()
    assert m.first is None and m.outcome is None


def test_samples_come_in_order_and_hold_what_the_rules_read():
    m = Monitor([Rule("trip", "T", "above", 372.0)])
    m.observe(1.0, {"T": 360.0})
    with pytest.raises(ValueError, match="order of time"):
        m.observe(1.0, {"T": 360.0})
    with pytest.raises(KeyError, match="'T'"):
        m.observe(2.0, {"C_A": 2.0})
    with pytest.raises(ValueError, match="twice"):
        Monitor([Rule("trip", "T", "above", 372.0)] * 2)
    with pytest.raises(ValueError, match="side"):
        Rule("trip", "T", "over", 372.0)


def test_a_trajectory_is_checked_at_its_grid():
    """Open loop from a 340 K feed, the reactor leaves for the high branch and trips;
    from a 330 K feed, it leaves for the low branch and is extinguished."""
    trip, cold = Rule("trip", "T", "above", 372.0), Rule("extinction", "T", "below",
                                                          350.0, dwell=20.0)
    for value, expected in ((340.0, trip), (330.0, cold)):
        traj = pb.run(CSTR, CSTR.config("open loop", disturbance={
            "name": "feed_temperature", "value": value, "t": 1.0}), t_end=200.0, dt=0.5)
        m = check(traj, [trip, cold])
        assert m.first.rule == expected
        T = traj.y["T"]
        beyond = T > 372.0 if expected is trip else T < 350.0
        k = int(np.argmax(beyond))
        if expected is trip:
            assert m.first.t == traj.t[k]
        else:
            assert m.first.since == traj.t[k - 1]
            assert m.first.t == traj.t[k - 1] + 20.0
    with pytest.raises(KeyError, match="reactor T"):
        check(traj, [Rule("trip", "reactor T", "above", 372.0)])


def test_a_stepped_run_is_monitored_between_its_samples():
    """The level peaks between the samples at 2 and 3 (`test_stepper`): a rule between
    the sampled values and the peak is met as an estimate in that interval."""
    pulse = {"name": "sequence", "field": "F_in", "times": [2.1, 2.5], "levels": [1.0, 0.0]}
    config = TEMPLATE.config("level PI", disturbance=pulse)
    s = pb.Stepper(pb.build(TEMPLATE, config), 1.0, extrema=("level",))
    steps = [s.observe()] + [s.step() for _ in range(3)]
    level = max(st.y["level"] for st in steps) + 0.01
    assert steps[-1].y_max["level"] > level
    m = Monitor([Rule("high", "level", "above", level)])
    for st in steps:
        m.step(st)
    v = m.violations[m.rules[0]]
    assert (v.t, v.since, v.sampled) == (3.0, 2.0, False)


def test_the_monitor_reads_the_true_values():
    """Noise of 0.5 m on the level would cross 1.6 m at once; the plant stays at 1.5 m."""
    config = TEMPLATE.config("level PI", noise={"seed": 0, "std": {"level": 0.5}})
    s = pb.Stepper(pb.build(TEMPLATE, config), 1.0)
    m = Monitor([Rule("high", "level", "above", 1.6)])
    steps = [s.observe()] + [s.step() for _ in range(20)]
    assert any(st.y["level"] > 1.6 for st in steps)
    for st in steps:
        m.step(st)
    assert not m.violations
