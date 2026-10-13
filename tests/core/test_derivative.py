"""Derivative action: a filter on the measurement, its layout, and its place in batches,
ensembles, the spectrum and configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from _tanks import tanks_design, tanks_structure
from _toy import toy_loop
from scipy.integrate import solve_ivp

from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
from plantbench.core.control import Loop
from plantbench.core.ensemble import integrate_ensemble
from plantbench.core.instruments import Instrument


@dataclass
class _In:
    m: float = 0.0


@dataclass
class _Ramp:
    """A measurement rising at `slope` whatever the controller does."""

    x: np.ndarray
    u: _In
    slope: float
    pp: None = None
    batched: ClassVar[bool] = True

    def rhs(self, t, x, u, pp):
        return np.full_like(x, self.slope)


def _pv(x, u, pp):
    return x[..., 0]


def test_a_ramp_gives_the_filtered_derivative():
    """For pv = slope t, the derivative term is -Kc tau_D slope (1 - exp(-t / (alpha tau_D)))."""
    Kc, tau_D, alpha, slope = 2.0, 0.5, 0.1, 0.3
    design = _Ramp(x=np.array([0.0]), u=_In(), slope=slope)
    S = ctl.bias_from_design([Loop("pv", _pv, "m", Kc=Kc, tau_I=1e9, lo=-np.inf, hi=np.inf,
                                   tau_D=tau_D, alpha=alpha)], design)
    y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
    assert y0.tolist() == [0.0, 0.0, 0.0]  # the filter starts at the measurement
    t = np.linspace(0.0, 0.5, 11)
    sol = solve_ivp(ctl.closed_loop_rhs, (0.0, 0.5), y0, t_eval=t, rtol=1e-12, atol=1e-14,
                    args=(S, design.u, None, None, design.rhs))
    x, z = sol.y[0], sol.y[2]
    term = -Kc * (x - z) / alpha
    np.testing.assert_allclose(term, -Kc * tau_D * slope * (1.0 - np.exp(-t / (alpha * tau_D))),
                               rtol=1e-8, atol=1e-12)


def test_a_setpoint_step_gives_no_derivative_kick(design):
    """At the same state, the output moves by Kc times the step, as without derivative
    action: the derivative acts on the measurement, not on the error."""
    S = ctl.bias_from_design([toy_loop(tau_D=2.0, alpha=0.1)], design)
    S = ctl.with_setpoint_steps(S, {"level": [(5.0, 0.4)]})
    aug = ctl.initial_augmented(S, design)
    assert aug.tolist() == [0.0, design.x[0]]
    before, after = {}, {}
    ctl.apply(S, design.x, aug, design.u, design.pp, 4.0, before)
    ctl.apply(S, design.x, aug, design.u, design.pp, 6.0, after)
    assert after["level.OP"] - before["level.OP"] == pytest.approx(S[0].Kc * 0.4, rel=1e-14)


def test_a_pid_structure_starts_at_rest(design):
    S = ctl.bias_from_design([toy_loop(tau_D=1.0)], design)
    y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
    r = ctl.closed_loop_rhs(0.0, y0, S, design.u, design.pp, None, design.rhs)
    assert np.max(np.abs(r)) < 1e-12


def test_the_filter_state_follows_the_instrument_and_actuator_states():
    S = [toy_loop(tau_D=1.0, instrument=Instrument(tau=0.5)), toy_loop(), toy_loop(tau_D=0.0)]
    slots, n_aug = ctl.state_layout(S)
    assert n_aug == 3 + 1 + 1
    assert [(s[0].start, s[0].stop, s[2].start, s[2].stop) for s in slots[:1]] == [(3, 4, 4, 5)]
    assert all(s[2].stop == s[2].start for s in slots[1:])


def test_tuning_adds_and_removes_the_filter(design):
    S = ctl.bias_from_design([toy_loop()], design)
    assert ctl.n_augmented(S) == 1
    pid = ctl.with_tuning(S, {"level": {"tau_D": 0.5, "alpha": 0.2}})
    assert ctl.n_augmented(pid) == 2 and pid[0].tau_D == 0.5 and pid[0].alpha == 0.2
    assert ctl.n_augmented(ctl.with_tuning(pid, {"level": {"tau_D": 0.0}})) == 1
    with pytest.raises(KeyError, match="cannot tune"):
        ctl.with_tuning(S, {"level": {"tau_d": 0.5}})
    with pytest.raises(ValueError, match="tau_D"):
        ctl.with_tuning(S, {"level": {"tau_D": -1.0}})
    with pytest.raises(ValueError, match="alpha"):
        ctl.with_tuning(S, {"level": {"tau_D": 1.0, "alpha": 0.0}})


def _pid(design, **level1):
    return ctl.with_tuning(tanks_structure(design),
                           {"level 1": {"tau_D": 0.4, **level1}, "level 2": {"tau_D": 2.0}})


def test_a_batch_evaluates_as_its_rows_do():
    """With derivative action on both loops, an instrument on one and an actuator on the
    other, a batch of states against one call each."""
    design = tanks_design()
    S = _pid(design)
    y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
    Y = y0 + 0.05 * (1.0 + np.abs(y0)) * np.random.default_rng(1).standard_normal((4, len(y0)))
    F = ctl.closed_loop_rhs_batch(1.0, Y, S, design.u, design.pp, None, design.rhs)
    for i in range(len(Y)):
        np.testing.assert_array_equal(
            F[i], ctl.closed_loop_rhs(1.0, Y[i], S, design.u, design.pp, None, design.rhs))


def test_members_with_their_own_derivative_tuning_evaluate_as_their_own_loops():
    design = tanks_design()
    structures = [_pid(design), _pid(design, alpha=0.25, Kc=-0.6)]
    rng = np.random.default_rng(2)
    Y = []
    for S in structures:
        y0 = np.concatenate([design.x, ctl.initial_augmented(S, design)])
        Y.append(y0 + 0.05 * (1.0 + np.abs(y0)) * rng.standard_normal(len(y0)))
    Y = np.array(Y)
    t = np.array([0.5, 2.0])
    rows = np.arange(2)
    F = ctl.ensemble_rhs(t, Y, ctl.take(ctl.stack(structures), rows),
                         ctl.ensemble_inputs(t, [None, None], design.u, rows), design.pp,
                         design.rhs)
    for i, S in enumerate(structures):
        np.testing.assert_array_equal(
            F[i], ctl.closed_loop_rhs(t[i], Y[i], S, design.u, design.pp, None, design.rhs))
    with pytest.raises(ValueError, match="state layout"):
        ctl.stack([tanks_structure(design), structures[0]])


def test_an_ensemble_with_derivative_action_agrees_with_lsoda():
    design = tanks_design()
    structures = [_pid(design), _pid(design, alpha=0.25, Kc=-0.6)]
    ds = [dist.step("d", value=0.2, t=1.0), dist.step("q2", fraction=0.1, t=3.0)]
    r = integrate_ensemble(design, structures, ds, 10.0, n_points=41)
    assert r.status == ["ok", "ok"]
    n = len(design.x)
    for i, (S, d) in enumerate(zip(structures, ds)):
        ref = ctl.integrate(design, S, 10.0, d, n_points=41, breakpoints=ctl.breakpoints_of(d, S))
        X = r.Y[i, :, :n].T
        assert np.max(np.abs(X - ref.X) / (1.0 + np.abs(ref.X))) < 1e-6, i


def test_the_filter_adds_one_eigenvalue_and_moves_the_others_continuously():
    design = tanks_design()
    pi = np.sort_complex(ctl.closed_loop_spectrum(design, tanks_structure(design)))
    for tau_D, tol in ((1e-3, 1e-2), (1e-4, 1e-3)):
        S = ctl.with_tuning(tanks_structure(design), {"level 1": {"tau_D": tau_D}})
        ev = ctl.closed_loop_spectrum(design, S)
        assert len(ev) == len(pi) + 1
        fast = np.argmin(ev.real)
        assert ev[fast].real == pytest.approx(-1.0 / (0.1 * tau_D), rel=0.05)
        rest = np.sort_complex(np.delete(ev, fast))
        assert np.max(np.abs(rest - pi)) < tol, (tau_D, rest, pi)


def test_a_specification_can_sample_the_derivative_time(tmp_path):
    from plantbench import datagen
    from plantbench.core.spec import Spec
    spec = {"study": {"name": "pid", "case": "jacketed_cstr", "seed": 1},
            "base": {"structure": "temperature PI",
                     "disturbance": {"name": "feed_temperature", "change": 2.0, "t": 5.0}},
            "sample": {"n": 2, "tuning.reactor T.tau_D": [0.5, 2.0]},
            "run": {"t_end": 30.0, "dt": 5.0}}
    out = datagen.generate(Spec.from_dict(spec), tmp_path / "d", log=lambda msg: None)
    records = datagen.load_records(out)
    assert [r["status"] for r in records] == ["ok", "ok"]
    for r in records:
        assert 0.5 <= r["config"]["tuning"]["reactor T"]["tau_D"] <= 2.0
