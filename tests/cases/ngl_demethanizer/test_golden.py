"""The reference configuration, conventional on the chebeir2019 basis, against the paper and
against itself.

Against the paper (Chebeir et al. 2019, Section 2): the conditions it states for the
conventional scheme, each with the tolerance argued in the case card.  The cold end is
held to 1 K at TK-101 and 2 K at the TE-100 outlet, wider than the warm end because the
mixture is near its critical region there and the equation of state matters most.  The
TK-100 temperature and the TE-100 vapour fraction miss their tolerances and are asserted
as misses, so that closing either gap shows here.

Against itself: the design point to 1e-6, so that a change moving the reference
configuration fails here.  The case is frozen from these values; a change meant to move
them updates them here, saying which moved and why.
"""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.ngl_demethanizer.definition import CASE

C = 273.15


def _summary(scheme: str) -> dict:
    d = pb.build(CASE, CASE.config(options={"scheme": scheme})).design
    m = {k: f(d.x, d.u, d.pp) for k, f in CASE.measurements(d).items()}
    st = d.plant.streams(d.x, d.u)
    # TE-100 outlet: its enthalpy flashed at the pressure of the stage it feeds.
    f = d.plant.eos.flash_PH(np.array([st.column.P[0]]), np.array([st.h_expander_out]),
                             np.asarray(st.y_expander)[None], T0=np.array([st.column.T[0]]))
    m.update(TE100_T=float(f.T[0] - C), TE100_V=float(f.V[0]), Q_chiller=d.u.Q_chiller,
             Q_reboiler=d.u.Q_reboiler, W_recompressor=d.u.W_recompressor,
             rise_top=m["stage 8 temperature"] - m["stage 1 temperature"])
    return m


@pytest.fixture(scope="module")
def ref():
    return _summary("conventional")


@pytest.fixture(scope="module")
def branched():
    return {s: _summary(s) for s in ("gsp", "crr")}


# -- against the paper ----------------------------------------------------------------

PAPER = [
    # measurement, paper, tolerance
    ("TK-101 temperature", -59.6, 1.0),
    ("TE100_T", -116.3, 2.0),
    ("ethane recovery", 0.82, 1e-8),  # the design specification, ERIC-100
    ("overhead pressure", 1010.0, 1e-8),  # the column top, a specification
]


@pytest.mark.parametrize("name, paper, tol", PAPER, ids=[p[0] for p in PAPER])
def test_the_conditions_the_paper_states(ref, name, paper, tol):
    assert ref[name] == pytest.approx(paper, abs=tol)


def test_the_tk100_temperature_is_a_recorded_miss(ref):
    """-40.03 C for a recovery of 0.82, where the paper states -29.3 C (case card)."""
    assert ref["TK-100 temperature"] == pytest.approx(-40.03, abs=0.01)
    assert abs(ref["TK-100 temperature"] - (-29.3)) > 10.0


def test_the_expander_vapour_fraction_is_a_recorded_miss(ref):
    """0.891 against 0.88 published, outside the proposed 0.01."""
    assert ref["TE100_V"] == pytest.approx(0.8915, abs=1e-4)
    assert abs(ref["TE100_V"] - 0.88) > 0.01


def test_only_the_branched_schemes_separate_at_the_top(ref, branched):
    """Section 4.1: no separation at the top of the conventional column; gsp and crr
    separate at both ends.  Read as the temperature rise over stages 1 to 8."""
    assert ref["rise_top"] < 2.0
    for scheme, m in branched.items():
        assert m["rise_top"] > 10.0, scheme


# -- against itself -------------------------------------------------------------------

FROZEN = {
    "TK-100 temperature": -40.030386,
    "TK-101 temperature": -59.623875,
    "TE100_T": -116.282346,
    "TE100_V": 0.891451,
    "NGL flow": 273.882629,
    "NGL temperature": -8.228645,
    "E-100 hot outlet temperature": -7.48795,
    "overhead temperature leaving E-100": -12.621392,
    "booster discharge pressure": 1227.1686,
    "K-101 discharge temperature": 179.316929,
    "Q_chiller": 1989.874124,
    "Q_reboiler": 1278.467966,
    "W_recompressor": 8122.865399,
}


@pytest.mark.parametrize("name", FROZEN)
def test_the_reference_design_is_unchanged(ref, name):
    assert ref[name] == pytest.approx(FROZEN[name], rel=1e-6, abs=1e-6)
