"""A batch of closed-loop states evaluated as its rows, for every case that declares a
batched right-hand side: a member's derivatives in a batch are those of the member in a
batch of its own and in a batch in another order, to the last bit.  The nested solves of
the NGL cases hold each row where it converges, so that no member waits on another."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.core import control as ctl

BATCHED = [c for c in pb.list_cases()
           if ctl.supports_batch(pb.build(pb.load_case(c)).design)]


@pytest.mark.parametrize("case_id", BATCHED)
def test_a_member_of_a_batch_is_evaluated_as_alone(case_id):
    case = pb.load_case(case_id)
    setup = pb.build(case)
    d, S = setup.design, setup.structure
    y0 = np.concatenate([d.x, ctl.initial_augmented(S, d)])
    rng = np.random.default_rng(0)
    Y = y0 + 1e-4 * np.abs(y0) * rng.standard_normal((6, len(y0)))
    args = (S, d.u, d.pp, None, d.rhs)
    F = ctl.closed_loop_rhs_batch(0.0, Y, *args)
    alone = np.concatenate([ctl.closed_loop_rhs_batch(0.0, y[None], *args) for y in Y])
    np.testing.assert_array_equal(alone, F)
    np.testing.assert_array_equal(ctl.closed_loop_rhs_batch(0.0, Y[::-1], *args)[::-1], F)
