"""The control structures and the named disturbance.

Every structure starts at rest at the design point of every scheme and basis, the ratio
cascade is refused where there is no bypass, and `feed ethane` keeps the feed composition
normalized.  The damping ratios recorded in the case card, from
studies/ngl_demethanizer/results/tuning.txt, are asserted in the slow tests: each needs a
closed-loop Jacobian, 15 to 50 s.
"""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.ngl_demethanizer.definition import CASE, feed_ethane
from plantbench.core import control as ctl

PLANTS = [("conventional", "chebeir2019"), ("gsp", "chebeir2019"), ("crr", "chebeir2019"),
          ("crr", "hysys")]
STRUCTURES = ["basic", "recovery cascade", "recovery cascade on ratio"]
CASES = [(s, b, k) for s, b in PLANTS for k in STRUCTURES
         if not (k == "recovery cascade on ratio" and s == "conventional")]

# The smallest damping ratio among the oscillatory modes, results/tuning.txt.
DAMPING = {
    ("conventional", "chebeir2019", "basic"): 0.550,
    ("conventional", "chebeir2019", "recovery cascade"): 0.470,
    ("gsp", "chebeir2019", "basic"): 0.557,
    ("gsp", "chebeir2019", "recovery cascade"): 0.496,
    ("gsp", "chebeir2019", "recovery cascade on ratio"): 0.440,
    ("crr", "chebeir2019", "basic"): 0.501,
    ("crr", "chebeir2019", "recovery cascade"): 0.482,
    ("crr", "chebeir2019", "recovery cascade on ratio"): 0.412,
    ("crr", "hysys", "basic"): 0.515,
    ("crr", "hysys", "recovery cascade"): 0.475,
    ("crr", "hysys", "recovery cascade on ratio"): 0.422,
}


def _setup(scheme, basis, structure, **config):
    return pb.build(CASE, CASE.config(structure=structure,
                                      options={"scheme": scheme, "basis": basis}, **config))


@pytest.mark.parametrize("scheme, basis, structure", CASES, ids=lambda v: str(v))
def test_every_structure_starts_at_rest(scheme, basis, structure):
    s = _setup(scheme, basis, structure)
    y0 = np.concatenate([s.design.x, ctl.initial_augmented(s.structure, s.design)])
    dy = ctl.closed_loop_rhs(0.0, y0, s.structure, s.design.u, s.design.pp, None,
                             s.design.rhs)
    assert np.max(np.abs(dy)) < 1e-9


def test_the_ratio_cascade_needs_a_bypass():
    with pytest.raises(ValueError, match="bypass"):
        _setup("conventional", "chebeir2019", "recovery cascade on ratio")


def test_the_transmitters_read_their_flows_at_the_design():
    d = _setup("crr", "chebeir2019", "basic").design
    st = d.plant.streams(d.x, d.u)
    s = d.plant.layout.unpack(d.x)
    assert s["FT_NGL"] == pytest.approx(st.B, rel=1e-12)
    assert s["FT_recycle"] == pytest.approx(st.F_recycle, rel=1e-12)
    assert s["FT_residue"] == pytest.approx(st.V0 - st.F_recycle, rel=1e-12)


def test_feed_ethane_steps_ethane_and_keeps_the_sum():
    d = _setup("conventional", "chebeir2019", "basic").design
    C2 = d.plant.eos.c.names.index("ethane")
    dist = feed_ethane(d, None, fraction=0.1, t=10.0)
    assert dist(5.0, d.u) is d.u
    z0, z = np.asarray(d.u.z_feed), np.asarray(dist(10.0, d.u).z_feed)
    assert z[C2] == pytest.approx(1.1 * z0[C2], rel=1e-14)
    assert z.sum() == pytest.approx(1.0, abs=1e-14)
    others = np.arange(len(z)) != C2
    assert np.allclose(z[others] / z0[others], (1 - z[C2]) / (1 - z0[C2]), rtol=1e-14)
    assert dist.times == (10.0,)


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(DAMPING), ids=lambda k: "-".join(k))
def test_the_recorded_damping_ratio(key):
    ev = _setup(*key).spectrum()
    assert ev.real.max() < 0
    assert ctl.damping_ratio(ev) == pytest.approx(DAMPING[key], abs=1e-3)
