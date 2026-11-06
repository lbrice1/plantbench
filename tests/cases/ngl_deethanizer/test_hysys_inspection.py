"""The case under `hysys_reference.config()` against the inspection of the tuned HYSYS
case (`hysys_reference.json`, inspection), at the criteria of `tests/_hysys.py`.

The HYSYS values are a saved dynamic snapshot, so its condenser duty and its streams
disagree (README, Comparison with the tuned HYSYS case).  The stage temperatures are
compared with the column report, whose stage 28 the inspection repeats.
"""

from __future__ import annotations

import numpy as np
import pytest
from _hysys import ENERGY, INTERNAL, PRODUCT, TEMPERATURE, relative

import plantbench as pb
from plantbench.cases.ngl_deethanizer import hysys_reference as H
from plantbench.cases.ngl_deethanizer.definition import CASE

C = 273.15
REF = H.inspection()


@pytest.fixture(scope="module")
def design():
    return pb.build(CASE, H.config()).design


@pytest.fixture(scope="module")
def y(design):
    return {k: f(design.x, design.u, design.pp) for k, f in CASE.measurements(design).items()}


@pytest.fixture(scope="module")
def streams(design):
    return design.plant.streams(design.x, design.u)


def test_the_configuration_is_the_one_hysys_runs():
    cfg = H.config()
    assert cfg.structure == "composition cascade"
    assert cfg.tuning == {}


@pytest.mark.parametrize("name, ref", [
    ("distillate flow", "distillate_kmol_h"), ("LPG flow", "LPG_kmol_h"),
    ("ethane in distillate", "ethane_in_distillate"), ("ethane in LPG", "ethane_in_LPG"),
])
def test_the_products(y, name, ref):
    assert relative(y[name], REF[ref]) < PRODUCT


def test_the_internal_flows(y, streams):
    """Reflux 1.1 %, vapor to the condenser 0.6 % and boil-up 2.6 % above HYSYS."""
    assert relative(y["reflux flow"], REF["reflux_kmol_h"]) < INTERNAL
    assert relative(60 * streams.F_condensed, REF["vapor_to_condenser_kmol_h"]) < INTERNAL
    assert relative(60 * streams.column.V[-1], REF["boilup_kmol_h"]) < INTERNAL


def test_the_duties(y):
    """The reboiler duty 1.7 % above HYSYS; the condenser duty 0.5 % above the duty
    HYSYS's own streams imply, the vapor to the condenser less the reflux and the
    distillate."""
    assert relative(y["reboiler duty"], REF["reboiler_duty_kW"]) < ENERGY
    assert relative(y["condenser duty"], REF["condenser_duty_from_streams_kW"]) < ENERGY


def test_the_temperatures(design, y):
    """The drum, the reboiler and stage 28 within 0.5 K; every stage within 1 K of the
    column report, the largest difference 0.96 K where the profile rises 6 K a stage."""
    assert abs(y["condenser temperature"] - REF["condenser_liquid_T_C"]) < TEMPERATURE
    assert abs(y["reboiler temperature"] - REF["reboiler_liquid_T_C"]) < TEMPERATURE
    assert abs(y["stage 28 temperature"] - REF["stage_28_T_C"]) < TEMPERATURE
    T = np.array([y[f"stage {n} temperature"] for n in range(1, 31)])
    assert np.max(np.abs(T - np.array(H.load()["column"]["T_C"]))) < TEMPERATURE


def test_the_saved_condenser_duty_is_an_exception(y):
    """10.5 % above the saved duty, which is 9 % below the duty of HYSYS's own streams."""
    d = relative(y["condenser duty"], REF["condenser_duty_saved_kW"])
    assert ENERGY < d < 0.11


def test_the_e100_duty_is_an_exception(y):
    """5.8 % below HYSYS: the feed leaves E-100 near its bubble point, where the two
    Peng-Robinson implementations differ most."""
    d = relative(y["E-100 duty"], REF["E-100_duty_kW"])
    assert ENERGY < d < 0.06


def test_the_e100_outlet_vapor_fraction_is_an_exception(streams):
    """0.0082 against 0.0230, for the same reason."""
    d = relative(streams.vapor_heated, REF["E-100_outlet_vapor_fraction"])
    assert PRODUCT < d < 0.65


def test_the_hysys_tuning_is_converted_from_the_hysys_controllers():
    """TIC-101 at 13 %/% over 0 to 200 C and 0 to 1000 kmol/h, 210 s; XIC-100 at 20 %/%
    over 0 to 1 and -5 to 140 C, 66 s; both direct-acting."""
    assert H.HYSYS_TUNING["stage 4 temperature"] == pytest.approx(
        {"Kc": -13 * (1000 / 60) / 200, "tau_I": 3.5})
    assert H.HYSYS_TUNING["ethane in LPG"] == pytest.approx({"Kc": -2900.0, "tau_I": 1.1})


def test_the_hysys_tuning_reaches_the_loops():
    setup = pb.build(CASE, H.config(tuning="hysys"))
    loops = {e.name: e for e in setup.structure}
    for name, values in H.HYSYS_TUNING.items():
        assert loops[name].Kc == pytest.approx(values["Kc"])
        assert loops[name].tau_I == pytest.approx(values["tau_I"])
    assert loops["feed temperature"].Kc == pb.build(CASE, H.config()).structure[0].Kc


def test_an_unknown_tuning_is_refused():
    with pytest.raises(ValueError, match="tuning must be one of"):
        H.config(tuning="detuned")
