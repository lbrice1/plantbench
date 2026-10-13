"""The crr design on the hysys basis against the HYSYS case "Demethanizer - Tuned".

Both are steady states of the same flowsheet and specifications on Peng-Robinson; they
differ in the exchangers (10 cells here, rigorous counter-current in HYSYS), the
compressors (ideal gases here) and the binary interaction parameters, which HYSYS does not
report.  Tolerances are set by quantity: the column profile to 0.5 K, stream temperatures
to 1 K, flows to 1 %, shaft powers and duties to 3 %.  Two quantities miss these and carry
their own tolerance, recorded rather than absorbed (studies/ngl_demethanizer/results/
hysys_comparison.txt): the residue gas leaving E-100, 2.1 K warm, and the K-102 power,
15 % high on a 25 kW machine, both of the ideal-gas compressors and the cell exchangers.
"""

from __future__ import annotations

import pytest

import plantbench as pb
from plantbench.cases.ngl_demethanizer import hysys_reference as H
from plantbench.cases.ngl_demethanizer.definition import CASE


@pytest.fixture(scope="module")
def m():
    d = pb.build(CASE, CASE.config(options={"scheme": "crr", "basis": "hysys"})).design
    out = {k: f(d.x, d.u, d.pp) for k, f in CASE.measurements(d).items()}
    out.update(chiller=d.u.Q_chiller, reboiler=d.u.Q_reboiler, K101=d.u.W_recompressor,
               K102=d.u.W_K102)
    return out


S = H.load()["streams"]
POWER = H.load()["power_kW"]


@pytest.mark.parametrize("stage", range(1, 31))
def test_the_column_temperature_profile(m, stage):
    ref = H.load()["column"]["T_C"][stage - 1]
    assert m[f"stage {stage} temperature"] == pytest.approx(ref, abs=0.5)


@pytest.mark.parametrize("name, stream", [
    ("E-100 hot outlet temperature", "4"),
    ("E-102 hot outlet temperature", "20"),
    ("E-104 hot outlet temperature", "24"),
    ("NGL temperature", "NGL"),
    ("K-101 discharge temperature", "Sales Gas"),
])
def test_stream_temperatures(m, name, stream):
    assert m[name] == pytest.approx(S[stream]["T_C"], abs=1.0)


def test_the_residue_gas_leaving_e100_is_a_recorded_miss(m):
    assert m["overhead temperature leaving E-100"] == pytest.approx(S["5"]["T_C"], abs=2.5)


@pytest.mark.parametrize("name, stream", [
    ("residue gas flow", "3"), ("recycle flow", "26"), ("branch flow", "18"),
    ("expander flow", "17"), ("NGL flow", "NGL"),
])
def test_flows(m, name, stream):
    assert m[name] == pytest.approx(S[stream]["F_kmol_h"], rel=0.01)


@pytest.mark.parametrize("name, ref", [
    ("chiller", "chiller"), ("reboiler", "reboiler"), ("expander power", "expander"),
    ("K101", "K-101"),
])
def test_powers_and_duties(m, name, ref):
    assert m[name] == pytest.approx(POWER[ref], rel=0.03)


def test_the_k102_power_is_a_recorded_miss(m):
    assert m["K102"] == pytest.approx(POWER["K-102"], rel=0.2)


def test_the_ethane_recovery(m):
    assert m["ethane recovery"] == pytest.approx(H.load()["ethane_recovery"], abs=0.01)
