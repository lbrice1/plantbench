"""The control structures: each starts at rest at the design, and each is damped as the
case card records; the named disturbance keeps the feed normalized."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.ngl_deethanizer.definition import CASE, STRUCTURES, feed_ethane
from plantbench.core import control as ctl


@pytest.mark.parametrize("structure", list(STRUCTURES))
def test_every_structure_starts_at_rest(structure):
    s = pb.build(CASE, CASE.config(structure=structure))
    y0 = np.concatenate([s.design.x, ctl.initial_augmented(s.structure, s.design)])
    dy = ctl.closed_loop_rhs(0.0, y0, s.structure, s.design.u, s.design.pp, None,
                             s.design.rhs)
    assert np.max(np.abs(dy)) < 1e-9


def test_feed_ethane_keeps_the_feed_normalized(design):
    d = feed_ethane(design, None, fraction=0.1, t=5.0)
    assert d(0.0, design.u) is design.u
    z = np.asarray(d(10.0, design.u).z_feed)
    assert z.sum() == pytest.approx(1.0, abs=1e-14)
    C2 = design.plant.eos.c.names.index("ethane")
    assert z[C2] == pytest.approx(1.1 * design.u.z_feed[C2])
    assert d.times == (5.0,)


def test_feed_ethane_refuses_a_fraction_outside_one(design):
    with pytest.raises(ValueError, match="outside"):
        feed_ethane(design, None, fraction=2.0)(1e3, design.u)


# The damping ratios of `studies/ngl_deethanizer/results/tuning.txt`.
DAMPING = {"open loop": 0.512, "basic": 0.445, "composition cascade": 0.427}


@pytest.mark.slow
@pytest.mark.parametrize("structure", sorted(DAMPING))
def test_damping_ratio(structure):
    ev = pb.build(CASE, CASE.config(structure=structure)).spectrum()
    assert ctl.damping_ratio(ev) == pytest.approx(DAMPING[structure], abs=1e-3)
    if structure != "open loop":
        assert ev.real.max() < 0
