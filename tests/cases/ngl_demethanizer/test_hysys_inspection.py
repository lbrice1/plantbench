"""The case under `hysys_reference.config()` against the inspection of the tuned HYSYS
case (`hysys_reference.json`, inspection), at the criteria of `tests/_hysys.py`.

`test_hysys_reference.py` keeps the comparison on the `hysys` basis as it designs, at the
TK-100 temperature of the reports; here the design is at ERIC-100's recovery set-point, as
the HYSYS case runs, and the TK-100 temperature is a result.  The stage and stream
temperatures and the flows are compared with the reports, which recover 0.8204.
"""

from __future__ import annotations

import numpy as np
import pytest
from _hysys import ENERGY, INTERNAL, PRODUCT, TEMPERATURE, relative

import plantbench as pb
from plantbench.cases.ngl_demethanizer import hysys_reference as H
from plantbench.cases.ngl_demethanizer.definition import CASE

REF = H.inspection()
S = H.load()["streams"]


@pytest.fixture(scope="module")
def y():
    d = pb.build(CASE, H.config()).design
    out = {k: f(d.x, d.u, d.pp) for k, f in CASE.measurements(d).items()}
    out.update(chiller=d.u.Q_chiller, reboiler=d.u.Q_reboiler, K101=d.u.W_recompressor,
               K102=d.u.W_K102)
    return out


def test_the_configuration_is_the_one_hysys_runs():
    cfg = H.config()
    assert cfg.structure == "recovery cascade"
    assert cfg.options == {"scheme": "crr", "basis": "hysys"}
    assert cfg.params["recovery_design"] == REF["ERIC-100"]["setpoint"]


def test_the_recovery_and_the_ngl(y):
    assert y["ethane recovery"] == pytest.approx(REF["ERIC-100"]["setpoint"], abs=1e-6)
    assert relative(y["NGL flow"], REF["NGL_kmol_h"]) < PRODUCT


def test_the_ric100_ratio(y):
    """F18 / F17 at 0.6869 against RIC-100's set-point of 0.69."""
    assert relative(H.hysys_ratio(y), REF["RIC-100"]["setpoint"]) < INTERNAL


@pytest.mark.parametrize("name, ref", [
    ("chiller", "chiller_duty_kW"), ("reboiler", "reboiler_duty_kW"),
    ("expander power", "expander_power_kW"), ("K101", "K-101_power_kW"),
])
def test_duties_and_powers(y, name, ref):
    """The chiller 2.7 % below HYSYS, the reboiler 1.0 % below, the expander 0.5 % and
    K-101 2.5 % above."""
    assert relative(y[name], REF[ref]) < ENERGY


@pytest.mark.parametrize("name, stream", [
    ("residue gas flow", "3"), ("recycle flow", "26"), ("branch flow", "18"),
    ("expander flow", "17"), ("NGL flow", "NGL"),
])
def test_flows_against_the_reports(y, name, stream):
    assert relative(y[name], S[stream]["F_kmol_h"]) < INTERNAL


def test_temperatures_against_the_reports(y):
    """Every stage within 0.69 K; the exchanger outlets, the NGL and the K-101 discharge
    within 0.9 K."""
    T = np.array([y[f"stage {n} temperature"] for n in range(1, 31)])
    assert np.max(np.abs(T - np.array(H.load()["column"]["T_C"]))) < TEMPERATURE
    for name, stream in [("E-100 hot outlet temperature", "4"),
                         ("E-102 hot outlet temperature", "20"),
                         ("E-104 hot outlet temperature", "24"),
                         ("NGL temperature", "NGL"),
                         ("K-101 discharge temperature", "Sales Gas")]:
        assert abs(y[name] - S[stream]["T_C"]) < TEMPERATURE, name


def test_the_k102_power_is_an_exception(y):
    """15.9 % above HYSYS on a 25 kW machine: the ideal-gas compressors."""
    assert ENERGY < relative(y["K102"], REF["K-102_power_kW"]) < 0.17


def test_the_residue_gas_leaving_e100_is_an_exception(y):
    """2.2 K warm: the ideal-gas machines and the cell exchangers."""
    assert TEMPERATURE < abs(y["overhead temperature leaving E-100"] - S["5"]["T_C"]) < 2.5


def test_the_hysys_tuning_of_eric100():
    """0.005 %/% over a span of 0 to 1 [choice] and -50 to 0 C, 3 s, direct-acting
    [choice]: -0.25 K per unit recovery and 0.05 min."""
    assert H.HYSYS_TUNING["ethane recovery"] == pytest.approx(
        {"Kc": -0.25, "tau_I": 0.05, "lo": -50.0, "hi": 0.0})


def test_the_hysys_tuning_reaches_eric100_only():
    hysys = {e.name: e for e in pb.build(CASE, H.config(tuning="hysys")).structure}
    own = {e.name: e for e in pb.build(CASE, H.config()).structure}
    for field, value in H.HYSYS_TUNING["ethane recovery"].items():
        assert getattr(hysys["ethane recovery"], field) == pytest.approx(value)
    for name in own.keys() - {"ethane recovery"}:
        assert (hysys[name].Kc, hysys[name].tau_I) == (own[name].Kc, own[name].tau_I)


@pytest.mark.slow
@pytest.mark.parametrize("tuning", H.TUNINGS)
def test_the_recovery_cascade_is_stable_under_either_tuning(tuning):
    ev = pb.build(CASE, H.config(tuning=tuning)).spectrum()
    assert ev.real.max() < 0


def test_an_unknown_tuning_is_refused():
    with pytest.raises(ValueError, match="tuning must be one of"):
        H.config(tuning="detuned")
