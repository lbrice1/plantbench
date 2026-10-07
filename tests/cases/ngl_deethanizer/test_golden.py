"""The reference configuration against the HYSYS case it is constructed from, and frozen.

The design is solved at the HYSYS set-points; everything else is a result.  Each
tolerance is argued from the comparison of `studies/ngl_deethanizer/hysys_comparison.py`
and from the report's precision (stream values to four significant figures, the profile
to hundredths of a kelvin).  The quantities the model does not reproduce are asserted as
misses, so that a change that closes or widens them is seen.
"""

from __future__ import annotations

import numpy as np
import pytest

from plantbench.cases.ngl_deethanizer import hysys_reference as H
from plantbench.cases.ngl_deethanizer.definition import measurements

C = 273.15


@pytest.fixture(scope="module")
def y(design):
    m = measurements(design)
    return {k: f(design.x, design.u, design.pp) for k, f in m.items()}


@pytest.fixture(scope="module")
def ref():
    return H.load()


def test_the_products(y, ref):
    """The split is fixed by the two composition specifications, so the flows follow from
    the balances; the report gives them to 0.1 kmol/h."""
    assert y["distillate flow"] == pytest.approx(H.stream("3")["F_kmol_h"], abs=0.2)
    assert y["LPG flow"] == pytest.approx(H.stream("4")["F_kmol_h"], abs=0.2)
    assert y["ethane in distillate"] == pytest.approx(H.stream("3")["z"][2], abs=1e-3)
    # Propane slips overhead at 0.0185 against 0.0188: the stage 4 specification holds
    # the rectifying section at HYSYS's temperatures, and the propane follows within 3 %.
    assert y["propane in distillate"] == pytest.approx(H.stream("3")["z"][3], rel=0.03)
    assert y["ethane recovery"] == pytest.approx(ref["ethane_recovery"], abs=5e-4)
    assert y["propane recovery"] == pytest.approx(141.1325 / 147.0064, abs=2e-3)


def test_the_column_profile(design, ref):
    """Every stage within 1.5 K: the largest difference, 0.96 K, is on stages 24 and 25,
    where the profile rises 6 K a stage, a sixth of a stage's shift.  Ethane and propane
    within 0.015 mole fraction on every stage (largest 0.007 and 0.010)."""
    st = design.plant.streams(design.x, design.u)
    s = design.plant.layout.unpack(design.x)
    T = st.column.T[:-1] - C
    assert np.max(np.abs(T - np.array(ref["column"]["T_C"]))) < 1.5
    x_ref = np.array(ref["column"]["x_N2_C1_C2_C3"])
    assert np.max(np.abs(s["x"][:, 2:4] - x_ref[:, 2:4])) < 0.015


def test_the_ends_of_the_column(y):
    """The drum and the reboiler are at the bubble points of the products, which the
    specifications fix; within 0.1 K."""
    assert y["condenser temperature"] == pytest.approx(H.stream("3")["T_C"], abs=0.1)
    assert y["reboiler temperature"] == pytest.approx(H.stream("4")["T_C"], abs=0.1)


def test_the_reflux_and_the_reboiler_duty(y, ref):
    """Within 2 % and 3 %: the reflux 1.1 % and the duty 1.7 % above HYSYS, the
    difference of the two implementations' heat-capacity correlations."""
    assert y["reflux flow"] == pytest.approx(ref["column"]["reflux_kmol_h"], rel=0.02)
    assert y["reboiler duty"] == pytest.approx(ref["duty_kJ_h"]["reboiler"] / 3600, rel=0.03)


def test_the_condenser_duty_against_hysys_streams(y):
    """The duty the HYSYS streams imply, the vapour to the condenser less the reflux and
    distillate, is matched within 2 % (0.7 %)."""
    S = H.load()["streams"]
    q = (S["To Condenser @COL2"]["F_kmol_h"] * S["To Condenser @COL2"]["h_kJ_kmol"]
         - (S["Reflux @COL2"]["F_kmol_h"] + S["3"]["F_kmol_h"]) * S["3"]["h_kJ_kmol"]) / 3600
    assert y["condenser duty"] == pytest.approx(q, rel=0.02)


def test_recorded_misses(design, y, ref):
    """What the model does not reproduce.

    The condenser duty HYSYS reports, 1598 kW, is 10.5 % below the model's and 9 % below
    the duty its own streams imply; the dynamic case's reported duty is not at the
    steady state of its streams.

    The E-100 outlet is 0.8 % vapour against HYSYS's 2.3 %, and its enthalpy rise 5 %
    smaller: the feed sits near its bubble point at 33 C, where the vapour fraction is
    sensitive to the small differences of the two Peng-Robinson implementations."""
    assert y["condenser duty"] > 1.08 * ref["duty_kJ_h"]["condenser"] / 3600
    st = design.plant.streams(design.x, design.u)
    assert st.vapour_heated < 0.015 < H.stream("2")["VF"]
    assert y["E-100 duty"] < 0.97 * ref["duty_kJ_h"]["E-100"] / 3600


# The reference configuration's results, frozen at their values on 7 October 2026.  A
# change that moves them is a change to the case, and is made with this table.
FROZEN = {
    "reflux flow": 371.4429763218536,
    "reboiler duty": 2440.740151425113,
    "condenser duty": 1766.8216869958933,
    "E-100 duty": 395.9934457622659,
    "stage 28 temperature": 71.58374805998307,
    "propane in distillate": 0.018518990097969108,
    "ethane recovery": 0.9796028559576712,
    "propane recovery": 0.9606101418427944,
}


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_frozen(y, name):
    assert y[name] == pytest.approx(FROZEN[name], rel=1e-6)
