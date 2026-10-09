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

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from plantbench.backend import namespace, to_device
from plantbench.units import _kernels

R = 8.314462618  # J/(mol K)
T_REF = 298.15  # K, ideal-gas enthalpy and entropy reference
P_REF = 101325.0  # Pa, ideal-gas entropy reference
SQRT2 = np.sqrt(2.0)
EPS = float(np.finfo(float).eps)
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

    @property
    def xp(self):
        """The array module the constants live in: NumPy, or CuPy after `on("cuda")`."""
        return namespace(self.Tc)

    def on(self, device: str) -> Components:
        """A copy with every constant array on `device`."""
        arrays = ("Tc", "Pc", "omega", "kij", "T_nodes", "H_nodes", "Cp_nodes", "S_nodes")
        return replace(self, **{k: to_device(getattr(self, k), device) for k in arrays})

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
        S = np.array([[cp.T_dependent_property_integral_over_T(T_REF, t) for cp in cps] for t in T])
        comps = cls(
            names=tuple(names),
            CAS=tuple(consts.CASs),
            Tc=np.array(consts.Tcs),
            Pc=np.array(consts.Pcs),
            omega=np.array(consts.omegas),
            kij=kij,
            T_nodes=T,
            H_nodes=H,
            Cp_nodes=Cp,
            S_nodes=S,
            _thermo=(consts, props),
        )
        mid = T[:-1] + H_STEP / 2
        exact = np.array([[cp.T_dependent_property_integral(T_REF, t) for cp in cps] for t in mid])
        error = float(np.max(np.abs(comps.h_ideal_pure(mid) - exact)))
        object.__setattr__(comps, "h_table_error", error)
        return comps

    def _hermite(self, T, values, slopes):
        xp = self.xp
        T = xp.asarray(T, dtype=float)
        if xp.any(T < self.T_nodes[0]) or xp.any(T > self.T_nodes[-1]):
            raise ValueError(
                f"temperature outside the enthalpy table, "
                f"{float(self.T_nodes[0]):g} to {float(self.T_nodes[-1]):g} K"
            )
        i = xp.clip(xp.searchsorted(self.T_nodes, T) - 1, 0, len(self.T_nodes) - 2)
        h = self.T_nodes[i + 1] - self.T_nodes[i]
        s = ((T - self.T_nodes[i]) / h)[:, None]
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        return (
            h00 * values[i]
            + h10 * h[:, None] * slopes[i]
            + h01 * values[i + 1]
            + h11 * h[:, None] * slopes[i + 1]
        )

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
        kw = dict(
            Tcs=list(self.Tc), Pcs=list(self.Pc), omegas=list(self.omega), kijs=self.kij.tolist()
        )
        return FlashVL(
            consts,
            props,
            liquid=CEOSLiquid(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases),
            gas=CEOSGas(PRMIX, kw, HeatCapacityGases=props.HeatCapacityGases),
        )


class PengRobinson:
    """The Peng–Robinson equation on a component set, every method vectorised over a
    leading axis of states (stages): T and P of shape (N,), compositions (N, n_c).

    `roots` chooses how the cubic is solved: "eigen", as the eigenvalues of its companion
    matrix, or "closed", by the trigonometric and Cardano formulas.  To either are added
    the double roots that round-off hides from it (`_near_double_roots`), and both are
    polished by the same two Newton steps.  They agree to round-off, except at a double
    root, which the coefficients determine only to about 1e-8 and which each finds to
    that accuracy.  "eigen" is the default and what
    the published results were produced with.  "closed" costs the same on the few rows of
    one column and about 2.4 times less per row on the thousands of rows of a batch, and
    it is the one a GPU can run; `on` chooses it.  On a GPU, with "closed", the state of
    a phase, its fugacities and the Rachford–Rice solve each run as one CUDA kernel
    (`_kernels`), which agree with the NumPy code to round-off.
    """

    def __init__(self, comps: Components, roots: str = "eigen"):
        if roots not in ("eigen", "closed"):
            raise ValueError(f"roots must be 'eigen' or 'closed', not {roots!r}")
        self.c = comps
        self.roots = roots
        self.xp = comps.xp
        self.b_i = OMEGA_B * R * comps.Tc / comps.Pc
        self.ac_i = OMEGA_A * (R * comps.Tc) ** 2 / comps.Pc
        w = comps.omega
        self.m_i = 0.37464 + 1.54226 * w - 0.26992 * w**2
        self.one_minus_k = 1.0 - comps.kij
        # On a GPU the state of a phase and its fugacities are each one kernel launch.
        self._use_kernels = roots == "closed" and _kernels.applies(self.xp)

    def on(self, device: str, roots: str = "closed") -> PengRobinson:
        """The same equation with its constants on `device`, for the batched path."""
        return PengRobinson(self.c.on(device), roots=roots)

    # -- pure and mixture parameters -------------------------------------------------

    def _a_i(self, T):
        """a_i(T) and da_i/dT, each (N, n_c)."""
        xp = self.xp
        sTr = xp.sqrt(T[:, None] / self.c.Tc)
        root = 1.0 + self.m_i * (1.0 - sTr)
        a = self.ac_i * root**2
        da = -self.ac_i * self.m_i * root / xp.sqrt(T[:, None] * self.c.Tc)
        return a, da

    def _mixture(self, T, x):
        xp = self.xp
        a_i, da_i = self._a_i(T)
        sq = xp.sqrt(a_i)
        a_ij = sq[:, :, None] * sq[:, None, :] * self.one_minus_k  # (N, n_c, n_c)
        sum_j = xp.einsum("nij,nj->ni", a_ij, x)  # sum_j x_j a_ij
        a = xp.einsum("ni,ni->n", x, sum_j)
        # d a_ij/dT = a_ij / 2 (a_i'/a_i + a_j'/a_j)
        r = da_i / a_i
        da = 0.5 * xp.einsum("ni,nj,nij->n", x, x, a_ij * (r[:, :, None] + r[:, None, :]))
        b = x @ self.b_i
        return a, da, b, sum_j

    # -- the cubic -------------------------------------------------------------------

    def _Z(self, A, B, phase):
        """The compressibility root of the phase: the smallest real root above B for a
        liquid, the largest for a vapour.  Where one real root exists it serves both.
        `phase` is "liquid", "vapour", or a boolean array true on the liquid rows."""
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B**2 - 2.0 * B
        c0 = -(A * B - B**2 - B**3)
        solve = _eigen_real_roots if self.roots == "eigen" else _cubic_real_roots
        return _select_root(solve(c2, c1, c0), c2, c1, c0, B, phase)

    def _state(self, T, P, x, phase):
        if self._use_kernels:
            return _kernels.state(self, T, P, x, phase)
        xp = self.xp
        a, da, b, sum_j = self._mixture(T, x)
        A = a * P / (R * T) ** 2
        B = b * P / (R * T)
        Z = self._Z(A, B, phase)
        L = xp.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
        return a, da, b, sum_j, A, B, Z, L

    def _stack(self, T, P, x, y):
        """Liquid x and vapour y at the same T and P as one stack of 2N rows, liquid
        first, with the phase of each row.  Every row's arithmetic is the one it has on
        its own; the stack halves the number of array operations, which is what one
        evaluation on a GPU costs."""
        xp = self.xp
        n = len(T)
        return (xp.concatenate([T, T]), xp.concatenate([P, P]), xp.concatenate([x, y]),
                xp.arange(2 * n) < n)

    def _stacks(self, T) -> bool:
        """Whether liquid and vapour are evaluated as one stack.  A single row is not, on
        the host: BLAS forms x @ b_i for one row by a dot product and for two by a
        matrix-vector product, which differ in the last bit, and a serial evaluation is
        to stay as it was."""
        return len(T) > 1 or self._use_kernels

    # -- properties ------------------------------------------------------------------

    def ln_phi(self, T, P, x, phase):
        """Log fugacity coefficients, (N, n_c).  `phase` is "liquid", "vapour", or a
        boolean array true on the liquid rows."""
        if self._use_kernels:
            return _kernels.ln_phi(self, T, P, x, phase)
        return self._ln_phi(self._state(T, P, x, phase))

    def _ln_phi_pair(self, T, P, x, y):
        """Log fugacity coefficients of liquid x and of vapour y at the same T and P."""
        if not self._stacks(T):
            return self.ln_phi(T, P, x, "liquid"), self.ln_phi(T, P, y, "vapour")
        n = len(T)
        ln_phi = self.ln_phi(*self._stack(T, P, x, y))
        return ln_phi[:n], ln_phi[n:]

    def _ln_phi(self, state):
        xp = self.xp
        a, _, b, sum_j, A, B, Z, L = state
        bi_b = self.b_i / b[:, None]
        return (
            bi_b * (Z - 1.0)[:, None]
            - xp.log(Z - B)[:, None]
            - (A / (2.0 * SQRT2 * B))[:, None] * (2.0 * sum_j / a[:, None] - bi_b) * L[:, None]
        )

    def h_ideal(self, T, x):
        """Ideal-gas enthalpy relative to the ideal gas at T_REF, J/mol, (N,)."""
        xp = self.xp
        return xp.einsum("ni,ni->n", x, self.c.h_ideal_pure(T))

    def h_departure(self, T, P, x, phase: str):
        return self._h_departure(T, self._state(T, P, x, phase))

    def _h_departure(self, T, state):
        a, da, b, _, _, B, Z, L = state
        return R * T * (Z - 1.0) + (T * da - a) / (2.0 * SQRT2 * b) * L

    def h(self, T, P, x, phase: str):
        """Molar enthalpy of the phase, J/mol, (N,)."""
        return self.h_ideal(T, x) + self.h_departure(T, P, x, phase)

    def s_ideal(self, T, P, x):
        """Ideal-gas entropy of the mixture relative to the pure ideal gases at T_REF and
        P_REF, mixing included, J/(mol K), (N,)."""
        xp = self.xp
        xs = xp.where(x > 0, x, 1.0)
        return (
            xp.einsum("ni,ni->n", x, self.c.s_ideal_pure(T))
            - R * xp.log(P / P_REF)
            - R * xp.einsum("ni,ni->n", x, xp.log(xs))
        )

    def s_departure(self, T, P, x, phase: str):
        return self._s_departure(self._state(T, P, x, phase))

    def _s_departure(self, state):
        xp = self.xp
        a, da, b, _, _, B, Z, L = state
        return R * xp.log(Z - B) + da / (2.0 * SQRT2 * b) * L

    def s(self, T, P, x, phase: str):
        """Molar entropy of the phase, J/(mol K), (N,)."""
        return self.s_ideal(T, P, x) + self.s_departure(T, P, x, phase)

    def K(self, T, P, x, y):
        """Equilibrium ratios for liquid x and vapour y, (N, n_c)."""
        xp = self.xp
        ln_phi_L, ln_phi_V = self._ln_phi_pair(T, P, x, y)
        return xp.exp(ln_phi_L - ln_phi_V)

    def wilson_K(self, T, P):
        xp = self.xp
        return (self.c.Pc / P[:, None]) * xp.exp(
            5.373 * (1.0 + self.c.omega) * (1.0 - self.c.Tc / T[:, None])
        )

    # -- bubble point ----------------------------------------------------------------

    def bubble_T(self, P, x, T0=None, tol: float = 1e-12, max_iter: int = 60):
        """Bubble-point temperature and incipient vapour of liquid x at pressure P.

        Newton on ln(sum K x) = 0 in T, with the vapour composition updated by successive
        substitution at each step, from a Wilson estimate or from `T0`.  Converges to
        `tol` in ln(sum K x) and in the relative step of T, so that the result is smooth
        enough to difference.
        """
        xp = self.xp
        P = xp.asarray(P, dtype=float)
        x = xp.asarray(x, dtype=float)
        T = self._wilson_bubble(P, x) if T0 is None else xp.array(T0, dtype=float)
        K = self.wilson_K(T, P)
        y = K * x
        y /= y.sum(axis=1, keepdims=True)
        dT = 1e-4
        for _ in range(max_iter):
            K = self.K(T, P, x, y)
            s = (K * x).sum(axis=1)
            f = xp.log(s)
            y = K * x / s[:, None]
            Kp = self.K(T + dT, P, x, y)
            fp = xp.log((Kp * x).sum(axis=1))
            step = -f / ((fp - f) / dT)
            step = xp.clip(step, -20.0, 20.0)
            T = T + step
            if xp.all(xp.abs(f) < tol) and xp.all(xp.abs(step) < tol * T):
                break
        else:
            raise RuntimeError(f"bubble point did not converge: |f| {float(xp.max(xp.abs(f))):.3g}")
        K = self.K(T, P, x, y)
        y = K * x / (K * x).sum(axis=1, keepdims=True)
        if xp.any(xp.max(xp.abs(y - x), axis=1) < 1e-6):
            raise RuntimeError("bubble point converged to the trivial solution y = x")
        return T, y

    def _wilson_bubble(self, P, x):
        """The Wilson-K bubble temperature, by bisection on sum K x = 1."""
        xp = self.xp
        lo = xp.full(len(P), 60.0)
        hi = xp.full(len(P), 700.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            over = (self.wilson_K(mid, P) * x).sum(axis=1) > 1.0
            hi = xp.where(over, mid, hi)
            lo = xp.where(over, lo, mid)
        return 0.5 * (lo + hi)

    # -- flashes ---------------------------------------------------------------------

    def flash_PT(self, T, P, z, K0=None, tol: float = 1e-12, max_iter: int = 400,
                 stability: bool = True):
        """Two-phase flash at temperature and pressure, every row at once.

        Successive substitution on the K-values, with the Rachford–Rice equation solved
        without bounds on the vapour fraction (the negative flash of Whitson and
        Michelsen, 1989), so that a row converging to a fraction outside (0, 1) is single
        phase: liquid below 0, vapour above 1.  A single-phase row stops once its phase is
        settled, since its properties do not depend on K; a two-phase row is finished by
        Newton on ln K once substitution has brought it close, and continues by
        substitution if Newton does not settle it.  Starts from Wilson's K or from `K0`.

        A row converging to a vapour fraction far outside [0, 1] is one phase.  Neither
        convergence to the trivial solution K = 1 nor a fraction far outside [0, 1] that
        does not converge shows the same (Michelsen, 1982).  Those rows, and only those,
        are given the tangent-plane test (`_stability`): a stable one is one phase, and an
        unstable one is flashed again from the K of its trial phase; one that comes back
        to the trivial solution raises an error.  `stability=False` leaves the test out.

        Returns the `Flash` of each row: vapour fraction clipped to [0, 1], phase
        compositions, K-values and the molar enthalpy and entropy of the whole.
        """
        xp = self.xp
        T = xp.asarray(T, dtype=float)
        P = xp.asarray(P, dtype=float)
        z = xp.asarray(z, dtype=float)
        lnK = xp.log(self.wilson_K(T, P) if K0 is None else xp.array(K0, dtype=float))
        beta = xp.full(len(T), 0.5)
        active = xp.ones(len(T), dtype=bool)
        no_newton = xp.zeros(len(T), dtype=bool)
        tested = xp.zeros(len(T), dtype=bool)
        stable = xp.zeros(len(T), dtype=bool)
        for it in range(max_iter):
            idx = xp.flatnonzero(active)
            beta[idx] = _rachford_rice(z[idx], xp.exp(lnK[idx]), beta[idx])
            new = self._substitute(T[idx], P[idx], z[idx], lnK[idx], beta[idx])
            change = xp.max(xp.abs(new - lnK[idx]), axis=1)
            lnK[idx] = new
            b = beta[idx]
            # A fraction more than 0.01 outside [0, 1] shows a row to be one phase once
            # its ln K is within 1e-3 of converging, the closeness at which Newton is
            # called: the negative flash is reliable where it converges (Whitson and
            # Michelsen, 1989).  Before that the fraction says nothing: from Wilson's K it
            # can be -1 on a feed that is two-phase.  Far outside [0, 1] substitution can
            # also cycle without converging, so a row still far outside and unconverged
            # after _FAR_ITERATIONS is given the tangent-plane test, once: if stable it is
            # one phase and stops; if not, substitution starts again from the K of its
            # trial phase.  Nearer [0, 1], substitution slows as a phase boundary
            # approaches, so Newton finishes the row, the negative flash telling
            # afterwards whether it is one phase or two.  On a row Newton could not
            # settle, which happens beside a phase boundary, the negative flash decides at
            # once: substitution there can stall at round-off without converging.
            far = (b < -0.01) | (b > 1.01)
            one_phase = far & (change < 1e-3)
            restarted = xp.zeros(len(idx), dtype=bool)
            if stability and it >= _FAR_ITERATIONS:
                ask = far & ~one_phase & ~tested[idx]
                if xp.any(ask):
                    a = idx[ask]
                    unstable, K_trial = self._stability(T[a], P[a], z[a])
                    tested[a] = True
                    lnK[a] = xp.where(unstable[:, None], xp.log(K_trial), lnK[a])
                    stable[a] = ~unstable
                    one_phase[ask] = ~unstable
                    restarted[ask] = unstable
            one_phase |= no_newton[idx] & ((b <= 0.0) | (b >= 1.0))
            trivial = (xp.max(xp.abs(lnK[idx]), axis=1) < 1e-4) & ~restarted
            close = ~one_phase & ~trivial & ~restarted & (change < 1e-3) & ~no_newton[idx]
            settled = xp.zeros(len(idx), dtype=bool)
            if xp.any(close):
                c = idx[close]
                lnK_c, beta_c, ok = self._newton(T[c], P[c], z[c], lnK[c], beta[c], tol)
                # A row Newton did not settle goes on by substitution from where it was.
                lnK[c] = xp.where(ok[:, None], lnK_c, lnK[c])
                beta[c] = xp.where(ok, beta_c, beta[c])
                no_newton[c] = ~ok
                settled[close] = ok
            active[idx[one_phase | trivial | settled | ((change < tol) & ~restarted)]] = False
            if not active.any():
                break
        else:
            raise RuntimeError(f"PT flash did not converge: change in ln K {float(xp.max(change)):.3g}")
        K = xp.exp(lnK)
        beta = _rachford_rice(z, K, beta)
        # A row the test found stable stopped where its K need not have converged, and
        # the fraction from that K says nothing: it is the phase of the lower Gibbs
        # energy, as at the trivial solution.  The fraction is set to 0 or 1, not beyond:
        # outside [0, 1] the absent phase's composition from K can be negative.
        if xp.any(stable):
            st = xp.flatnonzero(stable)
            beta[st] = xp.where(self._liquid_like(T[st], P[st], z[st]), 0.0, 1.0)
        trivial = xp.flatnonzero((xp.max(xp.abs(lnK), axis=1) < 1e-4) & ~tested)
        if stability and len(trivial):
            unstable, K_trial = self._stability(T[trivial], P[trivial], z[trivial])
            u = trivial[unstable]
            if len(u):
                again = self.flash_PT(T[u], P[u], z[u], K0=K_trial[unstable], tol=tol,
                                      max_iter=max_iter, stability=False)
                if xp.any(xp.max(xp.abs(xp.log(again.K)), axis=1) < 1e-4):
                    raise RuntimeError("PT flash of a feed the tangent-plane test finds "
                                       "unstable converged to the trivial solution")
                K[u] = again.K
                beta[u] = again.V
        return self._result(T, P, z, K, beta)

    def _stability(self, T, P, z, tol: float = 1e-10, max_iter: int = 200):
        """The tangent-plane test of each feed z as one phase (Michelsen, 1982).

        The feed takes the root of lower Gibbs energy, as `_liquid_like` chooses, and
        d_i = ln z_i + ln phi_i(z).  From two trial phases, vapour-like W = z K and
        liquid-like W = z / K with Wilson's K, successive substitution on
        ln W_i = d_i - ln phi_i(w), w = W / sum W, finds the stationary points of
        tm = 1 + sum W_i (ln W_i + ln phi_i(w) - d_i - 1), the tangent-plane distance
        of w.  The feed is unstable where tm < 0 at a trial that has not collapsed onto
        the feed.  Both trials of every row are one stack, each row stopping once its
        ln W changes by less than `tol`; a trial still moving after `max_iter` is judged
        where it is, since tm < 0 anywhere shows instability.

        Returns whether each row is unstable, and K from the trial of lower tm: w / z for
        the vapour-like trial, z / w for the liquid-like one, 1 for a component absent
        from the feed.
        """
        xp = self.xp
        n = len(T)
        present = z > 0.0
        zs = xp.where(present, z, 1e-300)
        d = xp.log(zs) + self.ln_phi(T, P, z, self._liquid_like(T, P, z))
        lnKw = xp.log(self.wilson_K(T, P))
        T2, P2, z2, d2, present2 = (xp.concatenate([v, v]) for v in (T, P, zs, d, present))
        liquid = xp.arange(2 * n) >= n  # the vapour-like trials first
        lnW = xp.log(z2) + xp.concatenate([lnKw, -lnKw])
        active = xp.ones(2 * n, dtype=bool)
        for _ in range(max_iter):
            i = xp.flatnonzero(active)
            w = xp.exp(lnW[i])
            w /= w.sum(axis=1, keepdims=True)
            new = d2[i] - self.ln_phi(T2[i], P2[i], w, liquid[i])
            change = xp.max(xp.abs(new - lnW[i]), axis=1)
            lnW[i] = new
            collapsed = _collapsed(w, z2[i], present2[i])
            active[i[(change < tol) | collapsed]] = False
            if not active.any():
                break
        W = xp.exp(lnW)
        w = W / W.sum(axis=1, keepdims=True)
        ln_phi = self.ln_phi(T2, P2, w, liquid)
        terms = xp.where(present2, W * (lnW + ln_phi - d2 - 1.0), 0.0)
        tm = xp.where(_collapsed(w, z2, present2), xp.inf, 1.0 + xp.sum(terms, axis=1))
        vapour_trial = tm[:n] <= tm[n:]
        tm_min = xp.where(vapour_trial, tm[:n], tm[n:])
        lnw = xp.log(w)
        lnK = xp.where(vapour_trial[:, None], lnw[:n] - xp.log(zs), xp.log(zs) - lnw[n:])
        K = xp.where(present, xp.exp(lnK), 1.0)
        return tm_min < -1e-8, K

    def _substitute(self, T, P, z, lnK, beta):
        """One step of successive substitution: ln K from the phases ln K gives."""
        xp = self.xp
        x, y = _phases(z, xp.exp(lnK), beta)
        ln_phi_L, ln_phi_V = self._ln_phi_pair(T, P, x, y)
        return ln_phi_L - ln_phi_V

    def _newton(self, T, P, z, lnK, beta, tol, max_iter: int = 8):
        """Newton on g(ln K) = ln K - substitute(ln K) = 0, its Jacobian by forward
        differences evaluated as extra rows of one batch.

        Returns ln K, the vapour fraction, and the rows it settled: those whose last step
        was below 1e-6, after which Newton, its Jacobian accurate to about 1e-7, leaves an
        error of order 1e-12, and those ending one phase (a vapour fraction outside (0, 1)), whose
        properties do not depend on K.  A row whose ln K leaves `_LNK_BOUND` has diverged;
        it is held where it was and not settled."""
        xp = self.xp
        n, nc = lnK.shape
        eps = 1e-7
        held = xp.zeros(n, dtype=bool)
        for _ in range(max_iter):
            rows = xp.concatenate([lnK[None], lnK[None] + eps * xp.eye(nc)[:, None, :]])
            rows = rows.reshape(-1, nc)  # (nc + 1) blocks of n rows
            rep = lambda v: xp.tile(v, (nc + 1,) + (1,) * (v.ndim - 1))  # noqa: E731
            zb = rep(z)
            Kb = xp.exp(rows)
            bb = _rachford_rice(zb, Kb, rep(beta))
            g = (rows - self._substitute(rep(T), rep(P), zb, rows, bb)).reshape(nc + 1, n, nc)
            J = xp.transpose((g[1:] - g[0]) / eps, (1, 2, 0))  # (n, nc, nc): dg_i/dlnK_j
            step = xp.linalg.solve(J, -g[0][:, :, None])[:, :, 0]
            new = lnK + step
            held = held | ~xp.all(xp.abs(new) < _LNK_BOUND, axis=1)  # NaN included
            lnK = xp.where(held[:, None], lnK, new)
            beta = bb[:n]
            if xp.max(xp.abs(step)) < tol:
                break
        beta = _rachford_rice(z, xp.exp(lnK), beta)
        last = xp.max(xp.abs(step), axis=1)
        return lnK, beta, ~held & ((last < 1e-6) | (beta <= 0.0) | (beta >= 1.0))

    def _result(self, T, P, z, K, beta) -> Flash:
        xp = self.xp
        x, y = _phases(z, K, beta)
        # A row that is one phase, or that converged to the trivial solution K = 1, is
        # all liquid or all vapour, as `_liquid_like` decides, as `thermo` does.  The sign
        # of the negative flash's fraction is not used: beside the trivial solution, where
        # a dense feed with one real root converges, it is meaningless.
        trivial = xp.max(xp.abs(xp.log(K)), axis=1) < 1e-4
        one = (beta <= 0.0) | (beta >= 1.0) | trivial
        liquid = one & self._liquid_like(T, P, z)
        vapour = one & ~liquid
        V = xp.where(liquid, 0.0, xp.where(vapour, 1.0, beta))
        x = xp.where(liquid[:, None], z, x)
        y = xp.where(vapour[:, None], z, y)
        if not self._stacks(T):
            h = (1 - V) * self.h(T, P, x, "liquid") + V * self.h(T, P, y, "vapour")
            s = (1 - V) * self.s(T, P, x, "liquid") + V * self.s(T, P, y, "vapour")
            return Flash(T=T, P=P, V=V, x=x, y=y, K=K, h=h, s=s)
        # The h and s of each phase, as `h` and `s` give them, from one stacked state.
        n = len(T)
        T2, P2, X, phase = self._stack(T, P, x, y)
        state = self._state(T2, P2, X, phase)
        h2 = self.h_ideal(T2, X) + self._h_departure(T2, state)
        s2 = self.s_ideal(T2, P2, X) + self._s_departure(state)
        h = (1 - V) * h2[:n] + V * h2[n:]
        s = (1 - V) * s2[:n] + V * s2[n:]
        return Flash(T=T, P=P, V=V, x=x, y=y, K=K, h=h, s=s)

    def _liquid_like(self, T, P, z):
        """Where one phase: whether it is liquid.  The liquid if its root has the lower
        Gibbs energy; where the cubic has one real root and the two are the same, by the
        phase identification parameter (Venkatarathnam and Oellrich, 2011), liquid above
        1, as `thermo` decides."""
        xp = self.xp
        ln_phi_L, ln_phi_V = self._ln_phi_pair(T, P, z, z)
        gL = xp.einsum("ni,ni->n", z, ln_phi_L)
        gV = xp.einsum("ni,ni->n", z, ln_phi_V)
        tie = xp.abs(gL - gV) <= 1e-12
        if not xp.any(tie):
            return gL < gV - 1e-12
        t = xp.flatnonzero(tie)
        pip = xp.full(len(T), xp.nan)
        pip[t] = self._pip(T[t], P[t], self._state(T[t], P[t], z[t], "liquid"))
        return (gL < gV - 1e-12) | (tie & (pip > 1.0))

    def _pip(self, T, P, state):
        """The phase identification parameter of a phase,
        v (d2P/dTdv / dP/dT - d2P/dv2 / dP/dv), from the derivatives of the equation."""
        a, da, b, _, _, _, Z, _ = state
        v = Z * R * T / P
        D = v * v + 2.0 * b * v - b * b
        dD = 2.0 * v + 2.0 * b
        dP_dT = R / (v - b) - da / D
        d2P_dTdv = -R / (v - b) ** 2 + da * dD / D**2
        dP_dv = -R * T / (v - b) ** 2 + a * dD / D**2
        d2P_dv2 = 2.0 * R * T / (v - b) ** 3 + 2.0 * a / D**2 - 2.0 * a * dD**2 / D**3
        return v * (d2P_dTdv / dP_dT - d2P_dv2 / dP_dv)

    def _flash_P(self, P, z, target, value, T0, K0, tol, max_iter):
        """Flash at pressure and a second property (`h` or `s`), both increasing in T:
        Newton in T on PT flashes, each warm-started from the last, with T and T + dT
        flashed in one batch, kept inside a bracket once the residual has changed sign and
        bisecting when a step leaves it.  The property has a kink at each phase boundary,
        where Newton alone oscillates."""
        xp = self.xp
        P = xp.asarray(P, dtype=float)
        z = xp.asarray(z, dtype=float)
        T = xp.array(T0, dtype=float)
        n = len(T)
        lo = xp.full_like(T, -xp.inf)
        hi = xp.full_like(T, xp.inf)
        K = None if K0 is None else xp.asarray(K0, dtype=float)
        dT = 1e-4
        for _ in range(max_iter):
            both = self.flash_PT(
                xp.concatenate([T, T + dT]),
                xp.concatenate([P, P]),
                xp.concatenate([z, z]),
                K0=None if K is None else xp.concatenate([K, K]),
            )
            f0, f1 = getattr(both, target)[:n], getattr(both, target)[n:]
            r = f0 - value
            lo = xp.where(r < 0, T, lo)
            hi = xp.where(r > 0, T, hi)
            new = T - xp.clip(r / ((f1 - f0) / dT), -30.0, 30.0)
            bracketed = xp.isfinite(lo) & xp.isfinite(hi)
            outside = bracketed & ((new <= lo) | (new >= hi))
            mid = 0.5 * (xp.where(bracketed, lo, 0.0) + xp.where(bracketed, hi, 0.0))
            new = xp.where(outside, mid, new)
            new = xp.maximum(new, self.c.T_nodes[0] + 1.0)
            step = new - T
            T = new
            K = both.K[:n]
            if xp.all(xp.abs(step) < tol * T):
                return self.flash_PT(T, P, z, K0=K)
        raise RuntimeError(
            f"P{target.upper()} flash did not converge: step {float(xp.max(xp.abs(step))):.3g} K"
        )

    def flash_PH(self, P, h, z, T0, K0=None, tol: float = 1e-11, max_iter: int = 60):
        """Flash at pressure and molar enthalpy: a valve, or a stream given a duty."""
        xp = self.xp
        return self._flash_P(P, z, "h", xp.asarray(h, dtype=float), T0, K0, tol, max_iter)

    def flash_PS(self, P, s, z, T0, K0=None, tol: float = 1e-11, max_iter: int = 60):
        """Flash at pressure and molar entropy: the isentropic outlet of a machine."""
        xp = self.xp
        return self._flash_P(P, z, "s", xp.asarray(s, dtype=float), T0, K0, tol, max_iter)


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


# A PT flash row still far outside [0, 1] and unconverged after this many substitutions
# is given the stability test.  Rows that converge there mostly do so within 10 (95 % of
# random natural-gas feeds) and nearly all within 40; one that is tested early is only
# decided sooner.
_FAR_ITERATIONS = 30

# Where Newton's ln K leaves (-_LNK_BOUND, _LNK_BOUND) the row has diverged: no physical
# K-value comes near the bound, and inside it exp(ln K) and its square stay finite.
_LNK_BOUND = 100.0


def _collapsed(w, z, present):
    """Whether each trial phase w of the stability test is the feed z, to 1e-4 in ln w
    over the components present."""
    xp = namespace(z)
    with np.errstate(divide="ignore"):
        gap = xp.where(present, xp.abs(xp.log(w) - xp.log(z)), 0.0)
    return xp.max(gap, axis=1) < 1e-4


def _phases(z, K, beta):
    d = 1.0 + beta[:, None] * (K - 1.0)
    x = z / d
    y = K * x
    return x / x.sum(axis=1, keepdims=True), y / y.sum(axis=1, keepdims=True)


def _rachford_rice(z, K, beta0, tol: float = 1e-14, max_iter: int = 100):
    """The vapour fraction solving sum z (K - 1) / (1 + beta (K - 1)) = 0, by Newton kept
    inside the interval where every denominator is positive, which extends beyond [0, 1]
    for the negative flash.  A row whose K are all above 1 has no root and is all vapour
    (returned as 2); all below 1, all liquid (returned as -1).  Only the components
    present in z count: an absent one has no term in the sum, and its K, which may be
    anything, would otherwise set a false end of the interval.  Each row stops once it
    has converged, as on a GPU, so that its result does not depend on the batch."""
    xp = namespace(z)
    if _kernels.applies(xp):
        return _kernels.rachford_rice(z, K, beta0, tol, max_iter)
    present = z > 0.0
    Km1 = xp.where(present, K - 1.0, 0.0)
    Kmax = xp.where(present, K, -xp.inf).max(axis=1)
    Kmin = xp.where(present, K, xp.inf).min(axis=1)
    beta = xp.where(Kmax <= 1.0, -1.0, xp.where(Kmin >= 1.0, 2.0, xp.asarray(beta0, float)))
    rows = xp.flatnonzero((Kmax > 1.0) & (Kmin < 1.0))
    if len(rows) == 0:
        return beta
    zr, Kr = z[rows], Km1[rows]
    lo = 1.0 / (1.0 - Kmax[rows])
    hi = 1.0 / (1.0 - Kmin[rows])
    b = xp.clip(beta[rows], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    finished = xp.zeros(len(rows), dtype=bool)
    for _ in range(max_iter):
        d = 1.0 + b[:, None] * Kr
        f = xp.sum(zr * Kr / d, axis=1)
        df = -xp.sum(zr * Kr**2 / d**2, axis=1)
        lo = xp.where(f > 0, b, lo)
        hi = xp.where(f < 0, b, hi)
        new = b - f / df
        # Bisect only where Newton leaves the bracket.  A converged row's step is zero and
        # lands on the bracket's end, which a test with <= would take for leaving it,
        # sending the row to bisect the bracket down to the tolerance.
        new = xp.where((new < lo) | (new > hi), 0.5 * (lo + hi), new)
        done = xp.abs(new - b) <= tol * (1.0 + xp.abs(b))
        b = xp.where(finished, b, new)
        finished = finished | done
        if xp.all(finished):
            break
    beta[rows] = b
    return beta


def _select_root(real, c2, c1, c0, B, phase):
    """The compressibility root of each row from the real roots of its cubic, (N, 3):
    with the near-double roots added, the smallest above B for a liquid and the largest
    for a vapour, polished by two Newton steps on the cubic.  The eigenvalues carry
    round-off that finite-difference Jacobians of the right-hand side would see as noise.
    A step is taken only where it does not increase the residual, since at a double root
    the derivative vanishes with it and the step is 0/0 or runs off."""
    xp = namespace(c2)
    real = xp.concatenate([real, _near_double_roots(c2, c1, c0)], axis=1)
    real = xp.where(real > B[:, None], real, xp.nan)
    if isinstance(phase, str):
        Z = xp.nanmin(real, axis=1) if phase == "liquid" else xp.nanmax(real, axis=1)
    else:
        Z = xp.where(phase, xp.nanmin(real, axis=1), xp.nanmax(real, axis=1))
    f = ((Z + c2) * Z + c1) * Z + c0
    for _ in range(2):
        new = Z - f / ((3.0 * Z + 2.0 * c2) * Z + c1)
        f_new = ((new + c2) * new + c1) * new + c0
        keep = xp.abs(f_new) <= xp.abs(f)  # false where NaN
        Z = xp.where(keep, new, Z)
        f = xp.where(keep, f_new, f)
    return Z


def _near_double_roots(c2, c1, c0):
    """The double roots of Z^3 + c2 Z^2 + c1 Z + c0 = 0 that a solver can miss, (N, 2).

    At a double root the two roots are determined by the coefficients only to about the
    square root of the machine epsilon: the companion-matrix eigenvalues split into a
    complex pair, and the discriminant of the depressed cubic comes out of either sign.
    A double root is a stationary point of the cubic where the cubic vanishes, so each
    stationary point is returned where the cubic there is zero to round-off, and NaN
    elsewhere.  A complex pair with an imaginary part below about 1e-6 is taken for a
    double root by this test; in double precision the two cannot be told apart.
    """
    xp = namespace(c2)
    d = c2 * c2 - 3.0 * c1
    s = xp.sqrt(xp.maximum(d, 0.0))
    Z = xp.stack([(-c2 - s) / 3.0, (-c2 + s) / 3.0], axis=1)
    c2, c1, c0 = c2[:, None], c1[:, None], c0[:, None]
    f = ((Z + c2) * Z + c1) * Z + c0
    size = xp.abs(Z) * Z * Z + xp.abs(c2) * Z * Z + xp.abs(c1 * Z) + xp.abs(c0)
    double = (d >= 0.0)[:, None] & (xp.abs(f) <= 16.0 * EPS * size)
    return xp.where(double, Z, xp.nan)


def _eigen_real_roots(c2, c1, c0):
    """The real roots of Z^3 + c2 Z^2 + c1 Z + c0 = 0 as the eigenvalues of its companion
    matrix with an imaginary part below 1e-9, NaN in place of the others, (N, 3)."""
    xp = namespace(c2)
    comp = xp.zeros((len(c2), 3, 3))
    comp[:, 0, :] = xp.stack([-c2, -c1, -c0], axis=1)
    comp[:, 1, 0] = comp[:, 2, 1] = 1.0
    roots = xp.linalg.eigvals(comp)
    return xp.where(xp.abs(roots.imag) < 1e-9, roots.real, xp.nan)


def _cubic_real_roots(c2, c1, c0):
    """The real roots of Z^3 + c2 Z^2 + c1 Z + c0 = 0, one row per cubic, (N, 3).

    The cubic is depressed to t^3 + p t + q = 0 by Z = t - c2 / 3.  Where it has three
    distinct real roots (discriminant at most zero, p < 0) they are given by the
    trigonometric formula; elsewhere, by Cardano's, repeated in all three columns, which
    at a triple root (p = q = 0) gives it exactly.  A double root for which round-off
    makes the discriminant positive is lost here and supplied by `_near_double_roots`.
    The roots carry round-off from the cancellation in Cardano's formula, which the
    Newton polish of `_select_root` removes.
    """
    xp = namespace(c2)
    shift = c2 / 3.0
    p = c1 - c2 * shift
    q = 2.0 * shift**3 - shift * c1 + c0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    sq = xp.sqrt(xp.maximum(disc, 0.0))
    one = xp.cbrt(-q / 2.0 + sq) + xp.cbrt(-q / 2.0 - sq) - shift
    # Three distinct real roots need p < 0.  Where p >= 0 the value is discarded, and
    # p = -1 stands in so that the arithmetic stays finite; a bound near zero would
    # underflow pn * r.
    pn = xp.where(p < 0.0, p, -1.0)
    r = 2.0 * xp.sqrt(-pn / 3.0)
    theta = xp.arccos(xp.clip(3.0 * q / (pn * r), -1.0, 1.0)) / 3.0
    k = 2.0 * np.pi / 3.0 * xp.arange(3)
    three = r[:, None] * xp.cos(theta[:, None] - k) - shift[:, None]
    return xp.where(((disc > 0.0) | (p >= 0.0))[:, None], one[:, None], three)
