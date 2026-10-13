"""Controller signals recorded in a run, and their percentages of span, on the toy plant."""

from __future__ import annotations

import json

import numpy as np
import pytest
from _toy import measure_x, toy_loop

from plantbench.core import control as ctl
from plantbench.core.case import Trajectory
from plantbench.core.control import Loop, Ratio
from plantbench.core.instruments import Actuator, Instrument


def _run(design, S, t_end=40.0, n_points=81):
    return ctl.integrate(design, S, t_end, n_points=n_points,
                         measurements={"x": measure_x},
                         breakpoints=ctl.breakpoints_of(None, S))


def test_the_setpoint_follows_its_schedule(design):
    S = ctl.bias_from_design([toy_loop()], design)
    S = ctl.with_setpoint_steps(S, {"level": [(10.0, 0.5)]})
    r = _run(design, S)
    assert set(r.C) == {"level.SP", "level.PV", "level.OP"}
    np.testing.assert_array_equal(r.C["level.SP"], np.where(r.t >= 10.0, 2.5, 2.0))
    np.testing.assert_array_equal(r.C["level.PV"], r.Y["x"])  # an ideal transmitter
    np.testing.assert_array_equal(r.C["level.OP"], r.U["m"])  # an ideal valve


def test_a_secondary_records_the_setpoint_its_primary_writes(design):
    primary = Loop("outer", measure_x, "sp:level", Kc=0.2, tau_I=50.0, lo=0.0, hi=10.0)
    S = ctl.bias_from_design([toy_loop(), primary], design)
    S = ctl.with_setpoints(S, {"outer": 2.4})
    r = _run(design, S)
    np.testing.assert_array_equal(r.C["level.SP"], r.C["outer.OP"])
    assert r.C["level.SP"][-1] != r.C["level.SP"][0]


def test_an_instrument_lags_the_pv_and_an_actuator_the_input(design):
    S = ctl.bias_from_design([toy_loop()], design)
    S = ctl.with_setpoint_steps(S, {"level": [(10.0, 0.5)]})
    S = ctl.with_instruments(S, {"level": Instrument(tau=2.0)}, {"level": Actuator(tau=1.0)})
    r = _run(design, S)
    after = r.t > 12.0
    assert np.all(np.abs(r.C["level.PV"] - r.Y["x"])[after] > 0)
    assert np.all(np.abs(r.C["level.OP"] - r.U["m"])[after] > 0)
    assert r.C["level.PV"][0] == pytest.approx(r.Y["x"][0], abs=1e-12)


def test_a_ratio_station_records_its_output_only(design):
    S = ctl.bias_from_design([toy_loop(), Ratio("r", "d", "m", hi=10.0)], design)
    r = _run(design, S)
    assert "r.OP" in r.C and "r.SP" not in r.C and "r.PV" not in r.C
    np.testing.assert_array_equal(r.C["r.OP"], r.U["d"])


def test_percentages_need_a_span(design):
    S = ctl.bias_from_design([toy_loop(pv_range=(0.0, 4.0), hi=2.0)], design)
    r = _run(design, S)
    np.testing.assert_allclose(r.C["level.PV%"], 25.0 * r.C["level.PV"], rtol=1e-15)
    np.testing.assert_allclose(r.C["level.OP%"], 50.0 * r.C["level.OP"], rtol=1e-15)
    plain = _run(design, ctl.bias_from_design([toy_loop()], design))
    assert "level.PV%" not in plain.C and "level.OP%" not in plain.C  # hi is infinite


def test_percent_gain():
    assert ctl.percent_gain(toy_loop()) is None
    assert ctl.percent_gain(toy_loop(pv_range=(0.0, 4.0))) is None
    loop = toy_loop(pv_range=(-1.0, 3.0), lo=1.0, hi=3.0)
    assert ctl.percent_gain(loop) == pytest.approx(0.5 * 4.0 / 2.0)


def test_the_signals_survive_a_file_and_an_old_file_loads(case, tmp_path):
    import plantbench as pb
    tr = pb.run(case, case.config(setpoint_steps={"level": [(5.0, 0.5)]}), t_end=20.0,
                n_points=21)
    assert set(tr.c) == {"level.SP", "level.PV", "level.OP"}
    tr.to_npz(tmp_path / "a.npz")
    back = Trajectory.from_npz(tmp_path / "a.npz")
    assert list(back.c) == list(tr.c)
    for k in tr.c:
        np.testing.assert_array_equal(back.c[k], tr.c[k])
    w = tr.window(5.0, 10.0)
    np.testing.assert_array_equal(w.c["level.SP"], tr.c["level.SP"][(tr.t >= 5) & (tr.t <= 10)])
    # A file written before the signals were recorded has no array c and no c_names.
    with np.load(tmp_path / "a.npz") as f:
        arrays = {k: f[k] for k in f.files if k != "c"}
    meta = json.loads(str(arrays["meta"]))
    del meta["c_names"]
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp_path / "old.npz", **arrays)
    old = Trajectory.from_npz(tmp_path / "old.npz")
    assert old.c == {}
    np.testing.assert_array_equal(old.x, tr.x)
