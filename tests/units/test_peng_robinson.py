"""The NumPy Peng-Robinson of `units.peng_robinson` against `thermo` on the same constants,
over the temperatures and pressures of a natural-gas plant: 35 to -116 C, 1000 to 6015 kPa,
on a seven-component natural gas."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("thermo")

from plantbench.units import peng_robinson as prm  # noqa: E402
from plantbench.units.peng_robinson import Components, PengRobinson  # noqa: E402

C = 273.15
COMPOSITIONS = {
    "feed": None,  # filled from the component set
    "top liquid": [0.002, 0.60, 0.25, 0.10, 0.03, 0.01, 0.008],
    "bottom liquid": [0.0, 0.02, 0.35, 0.30, 0.18, 0.08, 0.07],
}
NAMES = ("nitrogen", "methane", "ethane", "propane", "n-butane", "n-pentane", "n-hexane")
FEED = (0.010, 0.930, 0.030, 0.015, 0.009, 0.003, 0.003)
GRID = [(t, p) for t in (35.0, -4.8, -29.3, -59.6, -90.0, -116.3)
        for p in (1000e3, 1265e3, 4672e3, 6015e3)]


@pytest.fixture(scope="module")
def comps():
    return Components.from_names(NAMES)


@pytest.fixture(scope="module")
def pr(comps):
    return PengRobinson(comps)


@pytest.fixture(scope="module")
def phases(comps):
    from thermo import PRMIX, CEOSGas, CEOSLiquid

    c = comps
    consts, props = c._thermo
    kw = dict(Tcs=list(c.Tc), Pcs=list(c.Pc), omegas=list(c.omega), kijs=c.kij.tolist())
    return {"liquid": CEOSLiquid(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases),
            "vapour": CEOSGas(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases)}


def _x(name):
    x = np.array(FEED) if COMPOSITIONS[name] is None else np.array(COMPOSITIONS[name])
    return x / x.sum()


@pytest.mark.parametrize("phase", ["liquid", "vapour"])
@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_fugacity_and_enthalpy_agree_with_thermo_at_every_state(pr, phases, name, phase):
    x = _x(name)
    T = np.array([C + t for t, _ in GRID])
    P = np.array([p for _, p in GRID])
    X = np.tile(x, (len(GRID), 1))
    ln_phi = pr.ln_phi(T, P, X, phase)
    h = pr.h(T, P, X, phase)
    for n in range(len(GRID)):
        ref = phases[phase].to(T=T[n], P=P[n], zs=list(x))
        assert np.allclose(ln_phi[n], ref.lnphis(), rtol=0, atol=1e-10), GRID[n]
        assert h[n] == pytest.approx(ref.H(), rel=1e-9, abs=1e-6), GRID[n]


def test_bubble_points_agree_with_thermo(pr, comps):
    flasher = comps.flasher()
    x = np.array([_x("top liquid"), _x("bottom liquid"), _x("top liquid")])
    P = np.array([1010e3, 1000e3, 4672e3])
    T, y = pr.bubble_T(P, x)
    for n in range(len(P)):
        ref = flasher.flash(P=P[n], VF=0, zs=list(x[n]))
        assert T[n] == pytest.approx(ref.T, abs=1e-6)
        assert np.allclose(y[n], ref.gas.zs, atol=1e-5)
    K = pr.K(T, P, x, y)
    assert np.allclose((K * x).sum(axis=1), 1.0, atol=1e-12)


def test_the_bubble_point_does_not_depend_on_its_start(pr):
    """The result is smooth enough to difference: two starts agree to round-off."""
    x = np.tile(_x("top liquid"), (2, 1))
    P = np.full(2, 1010e3)
    T1, _ = pr.bubble_T(P, x)
    T2, _ = pr.bubble_T(P, x, T0=T1 + np.array([3.0, -3.0]))
    assert np.allclose(T1, T2, rtol=0, atol=1e-9)


def test_the_enthalpy_table_is_exact_to_a_millijoule(comps):
    assert comps.h_table_error < 1e-3


def test_a_temperature_outside_the_table_is_refused(pr):
    with pytest.raises(ValueError, match="outside the enthalpy table"):
        pr.h(np.array([30.0]), np.array([1e5]), np.array([_x("feed")]), "vapour")


def _two_phase_feed(pr):
    """The feed flashed where it is two-phase, and the flash's own result there."""
    T = C + np.repeat([-45.0, -60.0, -75.0, -90.0], 4)
    P = np.tile([1000e3, 2000e3, 3000e3, 4672e3], 4)
    z = np.tile(_x("feed"), (len(T), 1))
    f = pr.flash_PT(T, P, z)
    two = (f.V > 0.0) & (f.V < 1.0)
    return T[two], P[two], z[two], f


def test_a_row_newton_does_not_settle_is_finished_by_substitution(pr, monkeypatch):
    """With one Newton step, too few to converge, the rows it leaves unsettled go back to
    substitution, and the flash reaches the same answer."""
    T, P, z, _ = _two_phase_feed(pr)
    ref = pr.flash_PT(T, P, z)
    unsettled = []
    newton = PengRobinson._newton

    def one_step(self, *args):
        out = newton(self, *args, max_iter=1)
        unsettled.append(int((~out[2]).sum()))
        return out

    monkeypatch.setattr(PengRobinson, "_newton", one_step)
    f = pr.flash_PT(T, P, z)
    assert sum(unsettled) > 0
    assert np.allclose(f.V, ref.V, rtol=0, atol=1e-10)
    assert np.allclose(f.x, ref.x, rtol=0, atol=1e-10)
    assert np.allclose(f.y, ref.y, rtol=0, atol=1e-10)


def test_newton_holds_a_diverged_row_and_leaves_the_others(pr, monkeypatch):
    """A row driven out of the bound on ln K is held and reported unsettled, and the rows
    beside it get what they would alone, to round-off: the held row keeps the batch
    iterating, and they take further steps of the order of round-off.  The divergence is made by shifting the
    substitution of one row, marked by its temperature, by 1000 in ln K."""
    T, P, z, f = _two_phase_feed(pr)
    lnK = np.log(f.K[(f.V > 0.0) & (f.V < 1.0)]) * (1.0 + 1e-4)
    beta = np.full(len(T), 0.5)
    alone = pr._newton(T, P, z, lnK, beta, 1e-12)
    marker = T[0] + 1e-9
    substitute = PengRobinson._substitute

    def shifted(self, T, P, z, lnK, beta):
        return substitute(self, T, P, z, lnK, beta) + 1000.0 * (T == marker)[:, None]

    monkeypatch.setattr(PengRobinson, "_substitute", shifted)
    out = pr._newton(np.append(T, marker), np.append(P, P[0]), np.vstack([z, z[0]]),
                     np.vstack([lnK, lnK[0]]), np.append(beta, 0.5), 1e-12)
    assert alone[2].all()
    assert not out[2][-1]
    assert np.array_equal(out[0][-1], lnK[0])
    assert np.allclose(out[0][:-1], alone[0], rtol=0, atol=1e-12)
    assert np.array_equal(out[2][:-1], alone[2])


@pytest.mark.parametrize("solve", [prm._eigen_real_roots, prm._cubic_real_roots])
def test_a_double_or_triple_root_is_the_liquid_root(solve, double_roots):
    """At a double root the eigenvalues split into a complex pair and the discriminant
    comes out of either sign; the root is kept all the same, for either solver, and a
    triple root is found too."""
    c2, c1, c0, a, b, tol = double_roots
    B = np.zeros_like(a)
    with np.errstate(all="ignore"):
        liquid = prm._select_root(solve(c2, c1, c0), c2, c1, c0, B, "liquid")
        vapour = prm._select_root(solve(c2, c1, c0), c2, c1, c0, B, "vapour")
    assert np.all(np.abs(liquid - a) <= tol)
    assert np.all(np.abs(vapour - b) <= tol)


def test_a_close_complex_pair_is_not_taken_for_a_root():
    """(Z - b)((Z - a)^2 + w^2) has one real root, b, unless w is so small that the
    cubic at a is zero to round-off."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.005, 0.6, 2000)
    b = a + rng.uniform(0.01, 0.9, 2000)
    w = 10.0 ** rng.uniform(-5, -1, 2000)
    c2, c1, c0 = -(b + 2 * a), a * a + w * w + 2 * a * b, -b * (a * a + w * w)
    for solve in (prm._eigen_real_roots, prm._cubic_real_roots):
        Z = prm._select_root(solve(c2, c1, c0), c2, c1, c0, np.zeros_like(a), "liquid")
        np.testing.assert_allclose(Z, b, rtol=0, atol=1e-10)


@pytest.mark.parametrize("absent", [False, True])
def test_flashes_agree_with_thermo_over_random_feeds(pr, comps, absent):
    """Random feeds of the seven components over 140 to 330 K and 0.3 to 7.5 MPa, and
    the same without nitrogen and, in every other feed, without n-hexane: the same
    phases as `thermo`, which tests stability, and the same vapour fraction, to
    `thermo`'s own tolerance (agreement is to about 1e-7).  A row is not to be taken
    for one phase from a vapour fraction of Wilson's K, which is outside [0, 1] on many
    of these two-phase feeds, and a one-phase row is labelled as `thermo` labels it.
    Feeds `thermo` splits into two liquids are left out: our flash labels the lighter
    of those phases vapour."""
    rng = np.random.default_rng(1)
    n = 300
    z = rng.dirichlet(np.full(len(NAMES), 0.7), n)
    if absent:
        z[:, 0] = 0.0
        z[::2, -1] = 0.0
        z /= z.sum(axis=1, keepdims=True)
    T = rng.uniform(140.0, 330.0, n)
    P = rng.uniform(3e5, 7.5e6, n)
    with np.errstate(all="ignore"):
        f = pr.flash_PT(T, P, z)
    flasher = comps.flasher()
    compared = 0
    for i in range(n):
        ref = flasher.flash(T=T[i], P=P[i], zs=list(z[i]))
        if ref.phase == "LL":
            continue
        compared += 1
        assert abs(f.V[i] - ref.VF) < 1e-6, (i, f.V[i], ref.VF)
    assert compared > 0.9 * n


def test_a_two_phase_feed_converging_to_the_trivial_solution_is_split(pr):
    """Started from K within 1e-6 of 1, substitution converges to the trivial solution
    on most two-phase feeds.  The tangent-plane test finds them unstable and the flash
    returns the split it finds from Wilson's K; without the test they come back as one
    phase."""
    rng = np.random.default_rng(1)
    n = 300
    z = rng.dirichlet(np.full(len(NAMES), 0.7), n)
    T = rng.uniform(140.0, 330.0, n)
    P = rng.uniform(3e5, 7.5e6, n)
    ref = pr.flash_PT(T, P, z)
    two = (ref.V > 0.0) & (ref.V < 1.0)
    K0 = np.tile(np.exp(1e-6 * np.where(np.arange(len(NAMES)) < 2, 1.0, -1.0)), (n, 1))
    without = pr.flash_PT(T, P, z, K0=K0, stability=False)
    assert np.sum(two & ((without.V == 0.0) | (without.V == 1.0))) > 100
    f = pr.flash_PT(T, P, z, K0=K0)
    np.testing.assert_allclose(f.V[two], ref.V[two], rtol=0, atol=1e-9)
    np.testing.assert_allclose(f.x[two], ref.x[two], rtol=0, atol=1e-9)


def test_the_stability_test_finds_the_two_phase_feeds(pr):
    """On random feeds, unstable exactly where the flash from Wilson's K splits."""
    rng = np.random.default_rng(2)
    n = 300
    z = rng.dirichlet(np.full(len(NAMES), 0.7), n)
    T = rng.uniform(140.0, 330.0, n)
    P = rng.uniform(3e5, 7.5e6, n)
    f = pr.flash_PT(T, P, z)
    unstable, _ = pr._stability(T, P, z)
    np.testing.assert_array_equal(unstable, (f.V > 0.0) & (f.V < 1.0))


def test_rachford_rice_ignores_absent_components_and_the_batch(pr):
    """Two feeds without nitrogen, one with its root beside the end of the interval that
    nitrogen's K would set: each gets the same vapour fraction in a batch as alone, and
    the batch flashes.  Nitrogen's K counted, the first row ended on that false pole and,
    iterated on while the second converged, gave NaN for both."""
    K = np.array([[10.395195, 4.359806, 1.163768, 0.483217, 0.19472, 0.087131, 0.02935],
                  [1.029519, 1.016816, 1.001447, 0.996938, 0.98894, 0.993222, 0.980652]])
    z = np.array([[0.0, 3.204426e-02, 3.965281e-01, 3.670951e-01, 2.601369e-02,
                   1.020476e-01, 7.627122e-02],
                  [0.0, 1.055532e-01, 2.355596e-01, 1.970999e-01, 3.924383e-01,
                   1.419289e-04, 6.920709e-02]])
    beta0 = np.array([-0.106151, -5.673748])
    both = prm._rachford_rice(z, K, beta0)
    alone = [prm._rachford_rice(z[i:i + 1], K[i:i + 1], beta0[i:i + 1])[0] for i in range(2)]
    assert np.all(np.isfinite(both))
    np.testing.assert_array_equal(both, alone)
    r = np.sum(z * (K - 1.0) / (1.0 + both[:, None] * (K - 1.0)), axis=1)
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    f = pr.flash_PT(np.array([297.52, 197.34]), np.array([3243388.70, 4622565.31]), z)
    assert np.all(np.isfinite(f.V))
