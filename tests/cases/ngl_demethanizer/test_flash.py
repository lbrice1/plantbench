"""A flash from a run of the case, on which Newton once diverged inside its batch."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from plantbench.cases.ngl_demethanizer.components import equation_of_state

FIXTURE = Path(__file__).parent / "fixtures" / "dew_point_flash.json"


def test_a_flash_where_newton_once_diverged_settles_every_row_as_vapour():
    """Twenty rows flashed together, warm-started as the run had them.  When Rachford-Rice
    bisected its converged rows, Newton, kept iterating by the other rows, diverged on the
    row at its dew point (170.64 K), its ln K running out to about 430.  The flash must
    settle every row as vapour, converged and without a numerical warning."""
    f = json.loads(FIXTURE.read_text())
    T, P = np.array(f["T_K"]), np.array(f["P_Pa"])
    z = np.tile(f["z"], (len(T), 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        flash = equation_of_state("chebeir7").flash_PT(T, P, z, K0=np.array(f["K0"]))
    assert np.array_equal(flash.V, np.ones(len(T)))
    assert np.allclose(flash.y, z, rtol=0, atol=0)
