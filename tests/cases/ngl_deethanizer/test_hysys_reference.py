"""The transcription of the HYSYS case is self-consistent: its streams balance, it gives
the recovery HYSYS's own spreadsheet computes, and its controllers read the profile."""

from __future__ import annotations

import numpy as np
import pytest

from plantbench.cases.ngl_deethanizer import hysys_reference as H
from plantbench.cases.ngl_deethanizer.components import FEED


def _flows(name):
    s = H.stream(name)
    return s["F_kmol_h"] * np.array(s["z"])


def test_the_column_balances_by_component():
    """The feed's component flows equal the distillate's and the bottoms', to the report's
    four significant figures."""
    assert np.allclose(_flows("3") + _flows("4"), _flows("Feed"), rtol=0, atol=0.2)


def test_the_feed_is_the_inlet_spreadsheet():
    assert np.allclose(H.stream("Feed")["z"], FEED, atol=5e-5)


def test_the_recovery_is_hysys_own():
    s3, feed = H.stream("3"), H.stream("Feed")
    recovery = s3["F_kmol_h"] * s3["z"][2] / (feed["F_kmol_h"] * feed["z"][2])
    assert recovery == pytest.approx(H.load()["ethane_recovery"], abs=5e-4)


def test_the_temperature_controllers_read_the_profile():
    ref = H.load()
    T, c = ref["column"]["T_C"], ref["controllers"]
    assert c["TIC-101"]["PV"] == pytest.approx(T[3], abs=1e-3)
    assert c["TIC-100"]["PV"] == pytest.approx(T[27], abs=1e-2)
    assert c["XIC-100"]["PV"] == pytest.approx(H.stream("4")["z"][2], abs=1e-4)


def test_every_controller_has_its_range_and_action():
    c = H.load()["controllers"]
    assert sorted(c) == ["FIC-100", "FIC-101", "LIC-100", "LIC-101", "PIC-100", "TIC-100",
                         "TIC-101", "TIC-102", "XIC-100"]
    assert all({"PV_min", "PV_max", "action", "Kc", "Ti_s"} <= set(v) for v in c.values())


def test_the_column_flows_close_at_both_ends():
    """Stage 1's vapour less the reflux is the distillate; stage 30's liquid less the
    boil-up is the bottoms (kmol/h)."""
    col = H.load()["column"]
    assert col["V_kmol_h"][0] - col["reflux_kmol_h"] == pytest.approx(
        H.stream("3")["F_kmol_h"], abs=0.5)
    assert col["L_kmol_h"][-1] - col["boilup_kmol_h"] == pytest.approx(
        H.stream("4")["F_kmol_h"], abs=0.5)
