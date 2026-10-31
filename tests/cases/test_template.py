"""The template case through its public module: it runs, and it stays out of the
registry, so `load_case` and `list_cases` know only the cases that are offered."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases import template


def test_the_template_runs_from_its_public_module():
    case = template.CASE
    config = case.config(disturbance={"name": "step", "field": "F_in", "fraction": 0.2,
                                      "t": 5.0})
    traj = pb.run(case, config, t_end=60.0, dt=1.0)
    level = traj.x[0]
    assert np.isfinite(level).all()
    # A larger inflow raises the level, and the level loop holds it near its set-point.
    assert level.max() > level[0]
    assert level[-1] == pytest.approx(level[0], rel=0.01)


def test_the_template_is_not_registered():
    assert "template" not in pb.list_cases()
    with pytest.raises(KeyError):
        pb.load_case("template")


def test_the_public_module_is_the_template_itself():
    from plantbench.cases._template import definition, model
    assert template.CASE is definition.CASE
    assert template.TankParameters is model.TankParameters
