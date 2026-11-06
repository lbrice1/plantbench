"""`mma_polymerization_cstr` against Doyle, Ogunnaike and Pearson (1995), Section 5.1:
the coefficients of their Eq. (77), the operating point, the balances, and the loop."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases.mma_polymerization_cstr import model as M
from plantbench.cases.mma_polymerization_cstr.definition import (
    CASE,
    ENGINEERING,
    GRADES,
    grade_flow,
)

P = M.MMAParameters()


def test_the_rate_constants_give_the_coefficients_of_eq_77():
    """Eq. (77) prints the balances with Tables 1 and 2 substituted, to the digits below."""
    k = M.rate_constants(P)
    P0 = np.sqrt(2 * P.f_star * k["I"] / (k["Td"] + k["Tc"]))  # per sqrt(C_I)
    printed = {
        "monomer": ((k["P"] + k["fm"]) * P0, 2.4568),
        "initiator": (k["I"] + P.F / P.V, 10.1022),
        "initiator feed": (P.C_Iin / P.V, 80.0),
        "transfer": (k["fm"] * P0, 0.0024121),
        "termination": ((0.5 * k["Tc"] + k["Td"]) * P0 ** 2, 0.112191),
        "first moment": (P.M_m * (k["P"] + k["fm"]) * P0, 245.978),
    }
    for name, (value, ref) in printed.items():
        assert value == pytest.approx(ref, rel=1e-4), name


def test_the_design_is_the_published_operating_point():
    d = pb.build(CASE, CASE.config("open loop")).design
    # Each within one unit in the last digit the paper prints: its C_I and D_1 are one
    # unit above the case's 0.1329050 and 49.38173, where C_I follows from the initiator
    # balance alone, F_I C_Iin / (k_I V + F).
    printed = np.array([5.50677, 0.132906, 0.0019752, 49.3818])
    assert np.all(np.abs(d.x - printed) <= np.array([1e-5, 1e-6, 1e-7, 1e-4]))
    assert M.namw(d.x) == pytest.approx(25_000.5, abs=0.05)
    assert d.u.F_I == 0.016783


def test_the_design_is_a_steady_state():
    d = pb.build(CASE, CASE.config("open loop")).design
    assert np.max(np.abs(d.rhs(0.0, d.x, d.u, d.pp) / d.x)) < 1e-12


def test_the_balances_close_at_the_design():
    """Monomer enters as monomer and leaves as monomer or as polymer, D_1 / M_m; the
    initiator enters with F_I and leaves by outflow and decomposition."""
    x, k = M.steady_state(P, P.F_I), M.rate_constants(P)
    C_m, C_I, _, D_1 = x
    assert P.F * P.C_min == pytest.approx(P.F * (C_m + D_1 / P.M_m), rel=1e-12)
    assert P.F_I * P.C_Iin == pytest.approx(P.F * C_I + k["I"] * C_I * P.V, rel=1e-12)


def test_the_namw_falls_with_the_initiator_flow():
    flows = np.linspace(0.003, 0.07, 20)
    y = [float(M.namw(M.steady_state(P, f))) for f in flows]
    assert np.all(np.diff(y) < 0)


@pytest.mark.parametrize("grade, flow", [("A", 0.0168), ("B", 0.0046), ("C", 0.0358)])
def test_the_grade_flows(grade, flow):
    """The initiator flows of the grades, to the four decimals a grade table carries."""
    assert grade_flow(GRADES[grade]) == pytest.approx(flow, abs=5e-5)


def test_the_engineering_model():
    """Without chain transfer to monomer the efficiency that matches the design is 0.969;
    the model then gives longer chains at low initiator flow and shorter at high."""
    assert ENGINEERING["Z_fm"] == 0.0
    assert ENGINEERING["f_star"] == pytest.approx(0.969, abs=5e-4)
    eng = replace(P, **ENGINEERING)
    assert M.namw(M.steady_state(eng, P.F_I)) == pytest.approx(M.namw(M.steady_state(P, P.F_I)))
    assert M.namw(M.steady_state(eng, grade_flow(GRADES["B"]))) > 1.2 * GRADES["B"]
    assert M.namw(M.steady_state(eng, grade_flow(GRADES["C"]))) < 0.95 * GRADES["C"]


def test_the_engineering_model_runs_as_a_mismatch():
    """From the reactor's design state, the engineering model at the design flow stays
    at the design NAMW, since it matches the reactor there."""
    traj = pb.run(CASE, CASE.config("open loop", mismatch=ENGINEERING), t_end=60.0)
    assert traj.y["NAMW"][-1] == pytest.approx(25_000.5, rel=1e-3)


def test_a_grade_change_reaches_the_band():
    """Under `NAMW PI`, a set-point step from the design to grade B at 10 min brings the
    NAMW within 3 % of it."""
    cfg = CASE.config("NAMW PI", setpoint_steps={"NAMW": [[10.0, GRADES["B"] - 25_000.5]]})
    traj = pb.run(CASE, cfg, t_end=150.0)
    assert traj.y["NAMW"][-1] == pytest.approx(GRADES["B"], rel=0.03)


def test_the_loop_stabilizes_every_grade():
    for target in GRADES.values():
        ev = pb.build(CASE, CASE.config("NAMW PI", params={"F_I": grade_flow(target)})).spectrum()
        assert ev.real.max() < 0
