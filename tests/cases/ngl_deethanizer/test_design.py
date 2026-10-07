"""The design point: a steady state that conserves every component and the energy, at
its specifications, and reached from a rough state as well as from the HYSYS one."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plantbench.cases.ngl_deethanizer.model import (
    Plant,
    default_inputs,
    design_residuals,
    initial_state,
    settle,
    solve_design,
)

C = 273.15


def test_the_design_is_a_steady_state(design):
    d = design.plant.evaluate(design.x, design.u)[0]
    assert np.max(np.abs(d)) < 1e-9


def test_the_specifications_hold(design):
    r = design_residuals(design.plant, design.x, design.u)[-6:]
    assert np.max(np.abs(r)) < 1e-9


def test_components_are_conserved(design):
    u, st = design.u, design.plant.streams(design.x, design.u)
    s = design.plant.layout.unpack(design.x)
    feed = u.F_feed * np.asarray(u.z_feed)
    out = st.D * s["x_D"] + st.B * s["x_B"]
    assert np.allclose(out, feed, rtol=0, atol=1e-10)


def test_energy_is_conserved(design):
    """The feed's enthalpy and the two duties added equal the products' enthalpy and the
    condenser duty, kJ/min."""
    u, st = design.u, design.plant.streams(design.x, design.u)
    into = u.F_feed * st.h_feed + 60 * (u.Q_E100 + u.Q_reboiler)
    out = st.D * st.h_drum + st.B * st.column.h_L[-1] + 60 * u.Q_condenser
    assert out == pytest.approx(into, rel=0, abs=1e-6 * abs(into))


def test_the_levels_and_the_pressure_are_at_their_set_points(design):
    pp, s = design.pp, design.plant.layout.unpack(design.x)
    assert s["M_D"] / pp.M_drum_full == pytest.approx(pp.level_drum_design, abs=1e-12)
    assert s["M_B"] / pp.M_reboiler_full == pytest.approx(pp.level_reboiler_design, abs=1e-12)
    assert s["P_top"] == pytest.approx(pp.P_top, abs=1e-9)


@pytest.mark.slow
def test_settle_reaches_the_design_from_a_disturbed_start(design):
    """The fallback of `solve_design`: hold mode from the HYSYS state, the reflux 15 % low
    and the reboiler duty 10 % high, brought near the design and solved by Newton."""
    plant = Plant(design.pp)
    u0 = default_inputs(plant)
    start = settle(plant, initial_state(plant),
                   replace(u0, L_reflux=0.85 * u0.L_reflux, Q_reboiler=1.1 * u0.Q_reboiler))
    again = solve_design(design.pp, start=start)
    assert np.max(np.abs(again.x - design.x)) < 1e-6
