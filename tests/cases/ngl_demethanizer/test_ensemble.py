"""The demethanizer run as an ensemble, against the same configurations run one by one."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.core.case import run_ensemble

pytestmark = pytest.mark.slow


def test_members_agree_with_their_serial_runs():
    case = pb.load_case("ngl_demethanizer")
    configs = [case.config(),
               case.config(disturbance={"name": "feed ethane", "fraction": 0.1, "t": 0.25})]
    for config, res in zip(configs, run_ensemble(case, configs, 1.0, dt=0.25)):
        assert res.error is None
        ref = pb.run(case, config, t_end=1.0, dt=0.25)
        tr = res.trajectory
        assert np.max(np.abs(tr.x - ref.x) / (1e-6 + np.abs(ref.x))) < 1e-6
        for k in ref.y:
            np.testing.assert_allclose(tr.y[k], ref.y[k], rtol=1e-6, atol=1e-9, err_msg=k)
