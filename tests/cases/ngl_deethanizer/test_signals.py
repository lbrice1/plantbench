"""The deethanizer's controller spans and signals against the HYSYS case."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.ngl_deethanizer import hysys_reference as H
from plantbench.cases.ngl_deethanizer.definition import COMPOSITION, TUNING
from plantbench.core import control as ctl

TAGS = {"feed temperature": "TIC-102", "overhead pressure": "PIC-100",
        "condenser level": "LIC-100", "reboiler level": "LIC-101",
        "stage 4 temperature": "TIC-101", "stage 28 temperature": "TIC-100",
        "ethane in LPG": "XIC-100"}
# The two loops detuned from the HYSYS gains (README, Tuning), by the factor their rounded
# gains give: -0.25 against -1.083 kmol/min per K, and -250 against -2900 K.
DETUNED = {"stage 4 temperature": 0.25 / (13.0 / 200.0 * 1000.0 / 60.0),
           "ethane in LPG": 250.0 / (20.0 * 145.0)}
# What each loop writes, in HYSYS and in this case, and the output span (README, the
# comparison with HYSYS): the OP percentages differ by these differences, to the rounding of
# the README's figures, at most 0.06 points on E-100's span.
WRITES = {"TIC-102": (420.0, 396.0, 833.3), "PIC-100": (1598.0, 1767.0, 2500.0),
          "TIC-101": (367.4, 371.4, 1000.0), "TIC-100": (2399.0, 2441.0, 5555.6),
          "XIC-100": (71.16, 71.58, 145.0)}


@pytest.fixture(scope="module")
def setup():
    return pb.build("ngl_deethanizer", {"structure": "composition cascade"})


def test_the_spans_are_the_hysys_controllers():
    ref = H.load()["controllers"]
    for name, tag in TAGS.items():
        tuning = COMPOSITION if name == "ethane in LPG" else TUNING[name]
        assert tuning["pv_range"] == (ref[tag]["PV_min"], ref[tag]["PV_max"]), tag


def test_percent_gains_are_the_hysys_tuning(setup):
    ref = H.load()["controllers"]
    for loop in setup.structure:
        tag = TAGS[loop.name]
        hysys = ref[tag]["Kc"] * (-1.0 if ref[tag]["action"] == "Direct" else 1.0)
        factor = DETUNED.get(loop.name, 1.0)
        assert ctl.percent_gain(loop) == pytest.approx(factor * hysys, rel=2e-3), tag


def test_the_design_state_reads_as_hysys_reports_it(setup):
    d, S = setup.design, setup.structure
    signals: dict = {}
    ctl.apply(S, d.x, ctl.initial_augmented(S, d), d.u, d.pp, 0.0, signals)
    C = ctl.percent_signals(S, {k: np.atleast_1d(v) for k, v in signals.items()})
    ref = H.load()["controllers"]
    for name, tag in TAGS.items():
        hysys, here, span = WRITES.get(tag, (0.0, 0.0, 1.0))
        expected = ref[tag]["OP_percent"] + 100.0 * (here - hysys) / span
        assert C[f"{name}.OP%"][0] == pytest.approx(expected, abs=0.1), tag
    # The cascade: XIC-100 writes the TIC-100 set-point.
    assert C["stage 28 temperature.SP"][0] == C["ethane in LPG.OP"][0]


def test_a_run_records_the_cascade(setup):
    tr = pb.run("ngl_deethanizer",
                {"structure": "composition cascade",
                 "disturbance": {"name": "feed ethane", "fraction": 0.1, "t": 0.5}},
                t_end=2.0, dt=0.5)
    np.testing.assert_array_equal(tr.c["stage 28 temperature.SP"], tr.c["ethane in LPG.OP"])
    np.testing.assert_array_equal(tr.c["ethane in LPG.PV"], tr.y["ethane in LPG"])
    assert np.ptp(tr.c["stage 28 temperature.SP"]) > 0
