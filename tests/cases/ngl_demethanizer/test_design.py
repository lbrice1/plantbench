"""The design point of each scheme and basis: a steady state, conserving mass and energy
over the plant, reached from the stored start and, slowly, from scratch."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.ngl_demethanizer.definition import CASE
from plantbench.cases.ngl_demethanizer.model import DESIGN_INPUTS, SCHEMES, solve_design
from plantbench.cases.ngl_demethanizer.parameters import parameters

DESIGNS = [(scheme, "chebeir2019") for scheme in SCHEMES] + [("crr", "hysys")]


@pytest.fixture(scope="module", params=DESIGNS, ids=lambda p: "-".join(p))
def design(request):
    scheme, basis = request.param
    return pb.build(CASE, CASE.config(options={"scheme": scheme, "basis": basis})).design


def test_the_design_is_a_steady_state(design):
    assert np.max(np.abs(design.rhs(0.0, design.x, design.u, design.pp))) < 1e-9


def test_every_component_leaves_as_it_came(design):
    """Feed = NGL + residue gas, component by component."""
    st = design.plant.streams(design.x, design.u)
    s = design.plant.layout.unpack(design.x)
    feed = design.u.F_feed * design.u.z_feed
    out = st.B * s["x_B"] + st.F_comp * st.column.y[0]
    assert np.allclose(out, feed, rtol=1e-9, atol=1e-11)


def test_energy_is_conserved_over_the_plant(design):
    """Up to the booster inlet: what the feed, the reboiler and K-102 bring equals what
    the NGL, the residue gas, the chiller and the expander's shaft take away."""
    st = design.plant.streams(design.x, design.u)
    energy_in = (design.u.F_feed * st.h_feed + design.u.Q_reboiler * 60.0 + st.H_K102)
    energy_out = (st.B * st.column.h_L[-1] + st.F_comp * st.h_residue + st.Q_chiller
                  + st.W_expander)
    assert energy_in == pytest.approx(energy_out, rel=1e-8)


def test_holding_the_specifications_changes_nothing_at_the_design(design):
    d, st = design.plant.evaluate(design.x, design.u, hold=True)
    assert np.max(np.abs(d)) < 1e-9
    for name, value in st.implied.items():
        assert value == pytest.approx(getattr(design.u, name), rel=1e-7), name


def test_the_design_inputs_are_inside_their_ranges(design):
    for name in DESIGN_INPUTS[design.plant.scheme]:
        value = getattr(design.u, name)
        assert value > 0, name
        if name.startswith("v_"):
            assert value < 1, name


def test_the_design_holds_its_specifications(design):
    """Recovery 0.82 on the paper's basis (ERIC-100), TK-100 at -38.71 C on the HYSYS
    basis, and 1 % methane in the NGL on both (XIC-100)."""
    m = {k: f(design.x, design.u, design.pp) for k, f in CASE.measurements(design).items()}
    if design.plant.basis == "chebeir2019":
        assert m["ethane recovery"] == pytest.approx(0.82, abs=1e-8)
    else:
        assert m["TK-100 temperature"] == pytest.approx(-38.71, abs=1e-8)
    assert m["methane in NGL"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_the_conventional_design_is_reached_from_scratch():
    """Settle in hold mode from a rough state, then Newton; the stored start agrees."""
    design = solve_design(parameters(), "conventional")
    stored = pb.build(CASE).design
    assert np.max(np.abs(design.x - stored.x)) < 1e-6
