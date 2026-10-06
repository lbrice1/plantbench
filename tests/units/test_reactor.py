"""The reactor must reproduce the reference (Romagnoli and Palazoglu, 2020) before anything is built on top of it.

These are the gating tests.  If they fail, the three parameter modifications recorded in
`parameters.py` are wrong and every number downstream is unsupported.
"""

import numpy as np
import pytest

from plantbench.units import cstr
from plantbench.units.parameters import (
    REFERENCE_EIGENVALUES,
    REFERENCE_STEADY_STATE,
    reference_parameters,
)


@pytest.fixture(scope="module")
def p():
    return reference_parameters()


@pytest.fixture(scope="module")
def ss(p):
    return cstr.steady_state(p)


def test_steady_state_solves(p, ss):
    assert np.max(np.abs(cstr.rhs_constant_volume(0.0, ss, p))) < 1e-10


def test_steady_state_matches_table_17_2(p, ss):
    """Table 17.2 is printed to four significant figures, so match it to that."""
    target = np.array(list(REFERENCE_STEADY_STATE.values()))
    assert ss[1] == pytest.approx(target[1], abs=0.01)  # T, the pinned variable
    assert ss[0] == pytest.approx(target[0], rel=0.005)  # C_A
    assert ss[2] == pytest.approx(target[2], abs=0.15)  # T_j


def test_eigenvalues_match_section_17_1_3(p, ss):
    """The published eigenvalues are 0.060, -0.1349, -0.7989."""
    lam = cstr.eigenvalues(ss, p)
    assert np.allclose(lam.imag, 0.0)
    got = np.sort(lam.real)
    want = np.sort(np.array(REFERENCE_EIGENVALUES))
    assert got == pytest.approx(want, abs=1e-3)


def test_operating_point_is_open_loop_unstable(p, ss):
    """One eigenvalue is positive, which is why the reactor needs feedback at all."""
    lam = cstr.eigenvalues(ss, p)
    assert (lam.real > 0).sum() == 1


def test_steady_state_multiplicity(p):
    """Figure 17.3: heat generated and heat removed cross three times."""
    branches = cstr.steady_state_branches(p)
    assert branches.size == 3
    lo, mid, hi = branches
    assert lo < mid < hi
    # The middle crossing is the operating point of Table 17.2.
    assert mid == pytest.approx(REFERENCE_STEADY_STATE["T"], abs=0.5)


def test_open_loop_settles_on_the_low_branch():
    """Figures 17.4a and 17.5a: a colder feed reaches the low-temperature state."""
    from scipy.integrate import solve_ivp

    p = reference_parameters()
    ss = cstr.steady_state(p)
    sol = solve_ivp(
        lambda t, x: cstr.rhs_constant_volume(t, x, p, T0=330.0),
        (0, 600), ss, method="LSODA", rtol=1e-9, atol=1e-11, t_eval=[600.0],
    )
    assert sol.y[1, -1] < REFERENCE_STEADY_STATE["T"] - 20


def test_open_loop_settles_on_the_high_branch():
    """Figures 17.4b and 17.5b: a hotter feed reaches the high-temperature state."""
    from scipy.integrate import solve_ivp

    p = reference_parameters()
    ss = cstr.steady_state(p)
    sol = solve_ivp(
        lambda t, x: cstr.rhs_constant_volume(t, x, p, T0=340.0),
        (0, 600), ss, method="LSODA", rtol=1e-9, atol=1e-11, t_eval=[600.0],
    )
    assert sol.y[1, -1] > REFERENCE_STEADY_STATE["T"] + 15


def test_multicomponent_reduces_to_constant_volume(p, ss):
    """With no inert, no product recycled and equal in and out flows, the plantwide
    reactor block must be the constant-volume model."""
    x = np.array([p.V, 0.0, ss[0], 0.0, ss[1], ss[2]])
    C_in = np.array([0.0, p.CA0, 0.0])
    got = cstr.rhs_multicomponent(x, p, p.F0, C_in, p.T0, p.F0, p.Fj)
    want = cstr.rhs_constant_volume(0.0, ss, p)
    assert got[0] == pytest.approx(0.0, abs=1e-12)  # volume holds
    assert got[2] == pytest.approx(want[0], abs=1e-12)  # dC_A/dt
    assert got[4] == pytest.approx(want[1], abs=1e-12)  # dT/dt
    assert got[5] == pytest.approx(want[2], abs=1e-12)  # dT_j/dt
