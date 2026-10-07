"""Peng–Robinson equation of state, vectorised over stages, for use inside a right-hand side.

The column of a light-hydrocarbon plant needs a bubble-point temperature and the phase
enthalpies on every stage at every evaluation of its right-hand side.  A general flash
library does one stage at a time in Python, which at about 2 ms a stage is too slow for
a stiff integration whose Jacobian is built by finite differences.  This module does the
same calculation for all stages at once in NumPy.

    comps = Components.from_names(["nitrogen", "methane", "ethane", ...])
    pr = PengRobinson(comps)
    T, y = pr.bubble_T(P, x)                 # P (N,), x (N, n_c) -> T (N,), y (N, n_c)
    h_L, h_V = pr.h(T, P, x, "liquid"), pr.h(T, P, y, "vapour")

The model is the Peng–Robinson (1976) equation with the classical alpha function and van
der Waals mixing with binary interaction parameters k_ij.  Pure-component constants come
from `chemicals` and the k_ij from the ChemSep PR set distributed with `thermo`.  The ideal-
gas enthalpy is `thermo`'s integral of its heat-capacity correlation, tabulated when the
components are built and interpolated by cubic Hermite polynomials on the enthalpy and the
heat capacity at each node, so that it is exact at the nodes and continuously
differentiable between them; the largest interpolation error is kept on the `Components`.  The rigorous flashes outside the
right-hand side (at given enthalpy or entropy, for the expander and the valves) are made by
`thermo` with the same constants, through `Components.flasher`.

Units: K, Pa, J/mol (equal to kJ/kmol), mole fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

R = 8.314462618  # J/(mol K)
T_REF = 298.15  # K, ideal-gas enthalpy and entropy reference
P_REF = 101325.0  # Pa, ideal-gas entropy reference
SQRT2 = np.sqrt(2.0)
# The Peng–Robinson constants, as `thermo` uses them, so that the two agree to round-off.
OMEGA_A = 0.45723552892138218938
OMEGA_B = 0.077796073903888455972

# The ideal-gas enthalpy table: nodes every H_STEP kelvin over H_RANGE.
H_RANGE = (60.0, 800.0)
H_STEP = 2.0


@dataclass(frozen=True, eq=False)
class Components:
    """The constants of a component set: names, critical constants, acentric factors,
    binary interaction parameters and ideal-gas heat-capacity polynomials."""

    names: tuple[str, ...]
    CAS: tuple[str, ...]
    Tc: np.ndarray  # K
    Pc: np.ndarray  # Pa
    omega: np.ndarray
    kij: np.ndarray  # (n_c, n_c)
    T_nodes: np.ndarray  # (n_T,), K
    H_nodes: np.ndarray  # (n_T, n_c), ideal-gas enthalpy relative to T_REF, J/mol
    Cp_nodes: np.ndarray  # (n_T, n_c), J/(mol K)
    S_nodes: np.ndarray  # (n_T, n_c), integral of Cp/T from T_REF, J/(mol K)
    h_table_error: float = 0.0  # largest error of the interpolated enthalpy, J/mol
    _thermo: tuple = field(default=(), repr=False)

    @property
    def n(self) -> int:
        return len(self.names)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> Components:
        from thermo import ChemicalConstantsPackage
        from thermo.interaction_parameters import IPDB

        consts, props = ChemicalConstantsPackage.from_IDs(list(names))
        kij = np.array(IPDB.get_ip_asymmetric_matrix("ChemSep PR", consts.CASs, "kij"))
        T = np.arange(H_RANGE[0], H_RANGE[1] + H_STEP / 2, H_STEP)
        cps = props.HeatCapacityGases
        H = np.array([[cp.T_dependent_property_integral(T_REF, t) for cp in cps] for t in T])
        Cp = np.array([[cp(t) for cp in cps] for t in T])
        S = np.array([[cp.T_dependent_property_integral_over_T(T_REF, t) for cp in cps]
                      for t in T])
        comps = cls(names=tuple(names), CAS=tuple(consts.CASs), Tc=np.array(consts.Tcs),
                    Pc=np.array(consts.Pcs), omega=np.array(consts.omegas), kij=kij,
                    T_nodes=T, H_nodes=H, Cp_nodes=Cp, S_nodes=S, _thermo=(consts, props))
        mid = T[:-1] + H_STEP / 2
        exact = np.array([[cp.T_dependent_property_integral(T_REF, t) for cp in cps]
                          for t in mid])
        error = float(np.max(np.abs(comps.h_ideal_pure(mid) - exact)))
        object.__setattr__(comps, "h_table_error", error)
        return comps

    def _hermite(self, T, values, slopes):
        T = np.asarray(T, dtype=float)
        if np.any(T < self.T_nodes[0]) or np.any(T > self.T_nodes[-1]):
            raise ValueError(f"temperature outside the enthalpy table, "
                             f"{self.T_nodes[0]:g} to {self.T_nodes[-1]:g} K")
        i = np.clip(np.searchsorted(self.T_nodes, T) - 1, 0, len(self.T_nodes) - 2)
        h = self.T_nodes[i + 1] - self.T_nodes[i]
        s = ((T - self.T_nodes[i]) / h)[:, None]
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        return (h00 * values[i] + h10 * h[:, None] * slopes[i]
                + h01 * values[i + 1] + h11 * h[:, None] * slopes[i + 1])

    def h_ideal_pure(self, T) -> np.ndarray:
        """Ideal-gas enthalpy of each pure component relative to T_REF, (N, n_c), J/mol."""
        return self._hermite(T, self.H_nodes, self.Cp_nodes)

    def s_ideal_pure(self, T) -> np.ndarray:
        """Integral of Cp/T of each pure component from T_REF, (N, n_c), J/(mol K)."""
        return self._hermite(T, self.S_nodes, self.Cp_nodes / self.T_nodes[:, None])

    def flasher(self):
        """A `thermo` flasher on the same constants, for the flashes outside `rhs`."""
        from thermo import PRMIX, CEOSGas, CEOSLiquid, FlashVL

        consts, props = self._thermo
        kw = dict(Tcs=list(self.Tc), Pcs=list(self.Pc), omegas=list(self.omega),
                  kijs=self.kij.tolist())
        return FlashVL(consts, props,
                       liquid=CEOSLiquid(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases),
                       gas=CEOSGas(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases))


class PengRobinson:
    """The Peng–Robinson equation on a component set, every method vectorised over a
    leading axis of states (stages): T and P of shape (N,), compositions (N, n_c)."""

    def __init__(self, comps: Components):
        self.c = comps
        self.b_i = OMEGA_B * R * comps.Tc / comps.Pc
        self.ac_i = OMEGA_A * (R * comps.Tc) ** 2 / comps.Pc
        w = comps.omega
        self.m_i = 0.37464 + 1.54226 * w - 0.26992 * w**2
        self.one_minus_k = 1.0 - comps.kij

    # -- pure and mixture parameters -------------------------------------------------

    def _a_i(self, T):
        """a_i(T) and da_i/dT, each (N, n_c)."""
        sTr = np.sqrt(T[:, None] / self.c.Tc)
        root = 1.0 + self.m_i * (1.0 - sTr)
        a = self.ac_i * root**2
        da = -self.ac_i * self.m_i * root / np.sqrt(T[:, None] * self.c.Tc)
        return a, da

    def _mixture(self, T, x):
        a_i, da_i = self._a_i(T)
        sq = np.sqrt(a_i)
        a_ij = sq[:, :, None] * sq[:, None, :] * self.one_minus_k  # (N, n_c, n_c)
        sum_j = np.einsum("nij,nj->ni", a_ij, x)  # sum_j x_j a_ij
        a = np.einsum("ni,ni->n", x, sum_j)
        # d a_ij/dT = a_ij / 2 (a_i'/a_i + a_j'/a_j)
        r = da_i / a_i
        da = 0.5 * np.einsum("ni,nj,nij->n", x, x, a_ij * (r[:, :, None] + r[:, None, :]))
        b = x @ self.b_i
        return a, da, b, sum_j

    # -- the cubic -------------------------------------------------------------------

    @staticmethod
    def _Z(A, B, phase: str):
        """The compressibility root of the phase: the smallest real root above B for a
        liquid, the largest for a vapour.  Where one real root exists it serves both."""
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B**2 - 2.0 * B
        c0 = -(A * B - B**2 - B**3)
        n = len(A)
        comp = np.zeros((n, 3, 3))
        comp[:, 0, :] = np.stack([-c2, -c1, -c0], axis=1)
        comp[:, 1, 0] = comp[:, 2, 1] = 1.0
        roots = np.linalg.eigvals(comp)
        real = np.where((np.abs(roots.imag) < 1e-9) & (roots.real > B[:, None]),
                        roots.real, np.nan)
        Z = np.nanmin(real, axis=1) if phase == "liquid" else np.nanmax(real, axis=1)
        # Polish on the cubic itself: the eigenvalues carry round-off that finite-
        # difference Jacobians of the right-hand side would see as noise.
        for _ in range(2):
            f = ((Z + c2) * Z + c1) * Z + c0
            Z = Z - f / ((3.0 * Z + 2.0 * c2) * Z + c1)
        return Z

    def _state(self, T, P, x, phase):
        a, da, b, sum_j = self._mixture(T, x)
        A = a * P / (R * T) ** 2
        B = b * P / (R * T)
        Z = self._Z(A, B, phase)
        L = np.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
        return a, da, b, sum_j, A, B, Z, L

    # -- properties ------------------------------------------------------------------

    def ln_phi(self, T, P, x, phase: str):
        """Log fugacity coefficients, (N, n_c)."""
        a, _, b, sum_j, A, B, Z, L = self._state(T, P, x, phase)
        bi_b = self.b_i / b[:, None]
        return (bi_b * (Z - 1.0)[:, None] - np.log(Z - B)[:, None]
                - (A / (2.0 * SQRT2 * B))[:, None] * (2.0 * sum_j / a[:, None] - bi_b)
                * L[:, None])

    def h_ideal(self, T, x):
        """Ideal-gas enthalpy relative to the ideal gas at T_REF, J/mol, (N,)."""
        return np.einsum("ni,ni->n", x, self.c.h_ideal_pure(T))

    def h_departure(self, T, P, x, phase: str):
        a, da, b, _, _, B, Z, L = self._state(T, P, x, phase)
        return R * T * (Z - 1.0) + (T * da - a) / (2.0 * SQRT2 * b) * L

    def h(self, T, P, x, phase: str):
        """Molar enthalpy of the phase, J/mol, (N,)."""
        return self.h_ideal(T, x) + self.h_departure(T, P, x, phase)

    def s_ideal(self, T, P, x):
        """Ideal-gas entropy of the mixture relative to the pure ideal gases at T_REF and
        P_REF, mixing included, J/(mol K), (N,)."""
        xs = np.where(x > 0, x, 1.0)
        return (np.einsum("ni,ni->n", x, self.c.s_ideal_pure(T)) - R * np.log(P / P_REF)
                - R * np.einsum("ni,ni->n", x, np.log(xs)))

    def s_departure(self, T, P, x, phase: str):
        a, da, b, _, _, B, Z, L = self._state(T, P, x, phase)
        return R * np.log(Z - B) + da / (2.0 * SQRT2 * b) * L

    def s(self, T, P, x, phase: str):
        """Molar entropy of the phase, J/(mol K), (N,)."""
        return self.s_ideal(T, P, x) + self.s_departure(T, P, x, phase)

    def K(self, T, P, x, y):
        """Equilibrium ratios for liquid x and vapour y, (N, n_c)."""
        return np.exp(self.ln_phi(T, P, x, "liquid") - self.ln_phi(T, P, y, "vapour"))

    def wilson_K(self, T, P):
        return (self.c.Pc / P[:, None]) * np.exp(
            5.373 * (1.0 + self.c.omega) * (1.0 - self.c.Tc / T[:, None]))

    # -- bubble point ----------------------------------------------------------------

    def bubble_T(self, P, x, T0=None, tol: float = 1e-12, max_iter: int = 60):
        """Bubble-point temperature and incipient vapour of liquid x at pressure P.

        Newton on ln(sum K x) = 0 in T, with the vapour composition updated by successive
        substitution at each step, from a Wilson estimate or from `T0`.  Converges to
        `tol` in ln(sum K x) and in the relative step of T, so that the result is smooth
        enough to difference.
        """
        P = np.asarray(P, dtype=float)
        x = np.asarray(x, dtype=float)
        T = self._wilson_bubble(P, x) if T0 is None else np.array(T0, dtype=float)
        K = self.wilson_K(T, P)
        y = K * x
        y /= y.sum(axis=1, keepdims=True)
        dT = 1e-4
        for _ in range(max_iter):
            K = self.K(T, P, x, y)
            s = (K * x).sum(axis=1)
            f = np.log(s)
            y = K * x / s[:, None]
            Kp = self.K(T + dT, P, x, y)
            fp = np.log((Kp * x).sum(axis=1))
            step = -f / ((fp - f) / dT)
            step = np.clip(step, -20.0, 20.0)
            T = T + step
            if np.all(np.abs(f) < tol) and np.all(np.abs(step) < tol * T):
                break
        else:
            raise RuntimeError(f"bubble point did not converge: |f| {np.max(np.abs(f)):.3g}")
        K = self.K(T, P, x, y)
        y = K * x / (K * x).sum(axis=1, keepdims=True)
        if np.any(np.max(np.abs(y - x), axis=1) < 1e-6):
            raise RuntimeError("bubble point converged to the trivial solution y = x")
        return T, y

    def _wilson_bubble(self, P, x):
        """The Wilson-K bubble temperature, by bisection on sum K x = 1."""
        lo = np.full(len(P), 60.0)
        hi = np.full(len(P), 700.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            over = (self.wilson_K(mid, P) * x).sum(axis=1) > 1.0
            hi = np.where(over, mid, hi)
            lo = np.where(over, lo, mid)
        return 0.5 * (lo + hi)

    # -- flashes ---------------------------------------------------------------------

    def flash_PT(self, T, P, z, K0=None, tol: float = 1e-12, max_iter: int = 400):
        """Two-phase flash at temperature and pressure, every row at once.

        Successive substitution on the K-values, with the Rachford–Rice equation solved
        without bounds on the vapour fraction (the negative flash of Whitson and
        Michelsen, 1989), so that a row converging to a fraction outside (0, 1) is single
        phase: liquid below 0, vapour above 1.  A single-phase row stops once its phase is
        settled, since its properties do not depend on K; a two-phase row is finished by
        Newton on ln K once substitution has brought it close.  Starts from Wilson's K or
        from `K0`.

        Returns the `Flash` of each row: vapour fraction clipped to [0, 1], phase
        compositions, K-values and the molar enthalpy and entropy of the whole.
        """
        T = np.asarray(T, dtype=float)
        P = np.asarray(P, dtype=float)
        z = np.asarray(z, dtype=float)
        lnK = np.log(self.wilson_K(T, P) if K0 is None else np.array(K0, dtype=float))
        beta = np.full(len(T), 0.5)
        active = np.ones(len(T), dtype=bool)
        for _ in range(max_iter):
            idx = np.flatnonzero(active)
            beta[idx] = _rachford_rice(z[idx], np.exp(lnK[idx]), beta[idx])
            new = self._substitute(T[idx], P[idx], z[idx], lnK[idx], beta[idx])
            change = np.max(np.abs(new - lnK[idx]), axis=1)
            lnK[idx] = new
            b = beta[idx]
            # Far outside [0, 1] the phase is settled.  Near [0, 1], substitution slows as
            # a phase boundary approaches, so Newton finishes the row, the negative flash
            # telling afterwards whether it is one phase or two.
            one_phase = (b < -0.01) | (b > 1.01)
            trivial = np.max(np.abs(new), axis=1) < 1e-4
            close = ~one_phase & ~trivial & (change < 1e-3)
            if np.any(close):
                c = idx[close]
                lnK[c], beta[c] = self._newton(T[c], P[c], z[c], lnK[c], beta[c], tol)
            active[idx[one_phase | trivial | close | (change < tol)]] = False
            if not active.any():
                break
        else:
            raise RuntimeError(f"PT flash did not converge: change in ln K {np.max(change):.3g}")
        K = np.exp(lnK)
        beta = _rachford_rice(z, K, beta)
        return self._result(T, P, z, K, beta)

    def _substitute(self, T, P, z, lnK, beta):
        """One step of successive substitution: ln K from the phases ln K gives."""
        x, y = _phases(z, np.exp(lnK), beta)
        return self.ln_phi(T, P, x, "liquid") - self.ln_phi(T, P, y, "vapour")

    def _newton(self, T, P, z, lnK, beta, tol, max_iter: int = 8):
        """Newton on g(ln K) = ln K - substitute(ln K) = 0, its Jacobian by forward
        differences evaluated as extra rows of one batch."""
        n, nc = lnK.shape
        eps = 1e-7
        for _ in range(max_iter):
            rows = np.concatenate([lnK[None], lnK[None] + eps * np.eye(nc)[:, None, :]])
            rows = rows.reshape(-1, nc)  # (nc + 1) blocks of n rows
            rep = lambda v: np.tile(v, (nc + 1,) + (1,) * (v.ndim - 1))  # noqa: E731
            zb = rep(z)
            Kb = np.exp(rows)
            bb = _rachford_rice(zb, Kb, rep(beta))
            g = (rows - self._substitute(rep(T), rep(P), zb, rows, bb)).reshape(nc + 1, n, nc)
            J = np.transpose((g[1:] - g[0]) / eps, (1, 2, 0))  # (n, nc, nc): dg_i/dlnK_j
            step = np.linalg.solve(J, -g[0][:, :, None])[:, :, 0]
            lnK = lnK + step
            beta = bb[:n]
            if np.max(np.abs(step)) < tol:
                break
        return lnK, _rachford_rice(z, np.exp(lnK), beta)

    def _result(self, T, P, z, K, beta) -> Flash:
        x, y = _phases(z, K, beta)
        # A row that is one phase, or that converged to the trivial solution K = 1, is
        # all liquid or all vapour: labelled by the fraction the negative flash gives, and
        # for the trivial solution by which root is the more stable.
        trivial = np.max(np.abs(np.log(K)), axis=1) < 1e-4
        liquid = (beta <= 0.0) | (trivial & self._liquid_like(T, P, z))
        vapour = (beta >= 1.0) | (trivial & ~liquid)
        V = np.where(liquid, 0.0, np.where(vapour, 1.0, beta))
        x = np.where(liquid[:, None], z, x)
        y = np.where(vapour[:, None], z, y)
        h = (1 - V) * self.h(T, P, x, "liquid") + V * self.h(T, P, y, "vapour")
        s = (1 - V) * self.s(T, P, x, "liquid") + V * self.s(T, P, y, "vapour")
        return Flash(T=T, P=P, V=V, x=x, y=y, K=K, h=h, s=s)

    def _liquid_like(self, T, P, z):
        """Where one phase: whether the liquid root has the lower Gibbs energy."""
        gL = np.einsum("ni,ni->n", z, self.ln_phi(T, P, z, "liquid"))
        gV = np.einsum("ni,ni->n", z, self.ln_phi(T, P, z, "vapour"))
        return gL < gV - 1e-12

    def _flash_P(self, P, z, target, value, T0, K0, tol, max_iter):
        """Flash at pressure and a second property (`h` or `s`), both increasing in T:
        Newton in T on PT flashes, each warm-started from the last, with T and T + dT
        flashed in one batch, kept inside a bracket once the residual has changed sign and
        bisecting when a step leaves it.  The property has a kink at each phase boundary,
        where Newton alone oscillates."""
        P = np.asarray(P, dtype=float)
        z = np.asarray(z, dtype=float)
        T = np.array(T0, dtype=float)
        n = len(T)
        lo = np.full_like(T, -np.inf)
        hi = np.full_like(T, np.inf)
        K = None if K0 is None else np.asarray(K0, dtype=float)
        dT = 1e-4
        for _ in range(max_iter):
            both = self.flash_PT(np.concatenate([T, T + dT]), np.concatenate([P, P]),
                                 np.concatenate([z, z]),
                                 K0=None if K is None else np.concatenate([K, K]))
            f0, f1 = getattr(both, target)[:n], getattr(both, target)[n:]
            r = f0 - value
            lo = np.where(r < 0, T, lo)
            hi = np.where(r > 0, T, hi)
            new = T - np.clip(r / ((f1 - f0) / dT), -30.0, 30.0)
            bracketed = np.isfinite(lo) & np.isfinite(hi)
            outside = bracketed & ((new <= lo) | (new >= hi))
            new = np.where(outside, 0.5 * (lo + hi), new)
            new = np.maximum(new, self.c.T_nodes[0] + 1.0)
            step = new - T
            T = new
            K = both.K[:n]
            if np.all(np.abs(step) < tol * T):
                return self.flash_PT(T, P, z, K0=K)
        raise RuntimeError(f"P{target.upper()} flash did not converge: "
                           f"step {np.max(np.abs(step)):.3g} K")

    def flash_PH(self, P, h, z, T0, K0=None, tol: float = 1e-11, max_iter: int = 60):
        """Flash at pressure and molar enthalpy: a valve, or a stream given a duty."""
        return self._flash_P(P, z, "h", np.asarray(h, dtype=float), T0, K0, tol, max_iter)

    def flash_PS(self, P, s, z, T0, K0=None, tol: float = 1e-11, max_iter: int = 60):
        """Flash at pressure and molar entropy: the isentropic outlet of a machine."""
        return self._flash_P(P, z, "s", np.asarray(s, dtype=float), T0, K0, tol, max_iter)


@dataclass(frozen=True, eq=False)
class Flash:
    """The result of a flash, one row per stream: T K, P Pa, vapour fraction V, liquid x
    and vapour y (the feed composition in a phase that is absent), K, and the molar
    enthalpy h J/mol and entropy s J/(mol K) of the whole."""

    T: np.ndarray
    P: np.ndarray
    V: np.ndarray
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    h: np.ndarray
    s: np.ndarray


def _phases(z, K, beta):
    d = 1.0 + beta[:, None] * (K - 1.0)
    x = z / d
    y = K * x
    return x / x.sum(axis=1, keepdims=True), y / y.sum(axis=1, keepdims=True)


def _rachford_rice(z, K, beta0, tol: float = 1e-14, max_iter: int = 100):
    """The vapour fraction solving sum z (K - 1) / (1 + beta (K - 1)) = 0, by Newton kept
    inside the interval where every denominator is positive, which extends beyond [0, 1]
    for the negative flash.  A row whose K are all above 1 has no root and is all vapour
    (returned as 2); all below 1, all liquid (returned as -1)."""
    Km1 = K - 1.0
    Kmax, Kmin = K.max(axis=1), K.min(axis=1)
    beta = np.where(Kmax <= 1.0, -1.0, np.where(Kmin >= 1.0, 2.0, np.asarray(beta0, float)))
    rows = np.flatnonzero((Kmax > 1.0) & (Kmin < 1.0))
    if len(rows) == 0:
        return beta
    zr, Kr = z[rows], Km1[rows]
    lo = 1.0 / (1.0 - Kmax[rows])
    hi = 1.0 / (1.0 - Kmin[rows])
    b = np.clip(beta[rows], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    for _ in range(max_iter):
        d = 1.0 + b[:, None] * Kr
        f = np.sum(zr * Kr / d, axis=1)
        df = -np.sum(zr * Kr**2 / d**2, axis=1)
        lo = np.where(f > 0, b, lo)
        hi = np.where(f < 0, b, hi)
        new = b - f / df
        new = np.where((new <= lo) | (new >= hi), 0.5 * (lo + hi), new)
        done = np.abs(new - b) <= tol * (1.0 + np.abs(b))
        b = new
        if np.all(done):
            break
    beta[rows] = b
    return beta
