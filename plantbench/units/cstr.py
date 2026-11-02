"""Non-isothermal CSTR with a cooling jacket, as developed by Romagnoli and
Palazoglu (2020).

Two forms of the model are provided.

`rhs_constant_volume` is the three-state constant-volume model of the reference, written
exactly as its Equation 17.10 gives it.  It is what reproduces the reference steady-state
multiplicity, eigenvalues and closed-loop responses.

`rhs_multicomponent` relaxes the constant-volume assumption, as the reference suggests,
and carries the inert and the product as additional species.  It is the reactor block of
the plantwide flowsheet in `plant.py`.  With a single species, a constant volume and
equal inlet and outlet flows it reduces to `rhs_constant_volume`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from .parameters import R_GAS, ReactorParameters

# Species order used throughout the package: inert, reactant, product.
I, A, B = 0, 1, 2
N_SPECIES = 3


def rate_constant(T: float | np.ndarray, p: ReactorParameters) -> float | np.ndarray:
    """Arrhenius rate constant, in 1/min."""
    return p.k0 * np.exp(-p.E / (R_GAS * T))


# --------------------------------------------------------------------------------
# Reference form: three states, constant volume, single species
# --------------------------------------------------------------------------------


def rhs_constant_volume(
    t: float,
    x: np.ndarray,
    p: ReactorParameters,
    T0: float | None = None,
    CA0: float | None = None,
    Fj: float | None = None,
    F0: float | None = None,
) -> np.ndarray:
    """Right-hand side of Equation 17.10.  State is [C_A, T, T_j].

    `T0`, `CA0`, `Fj` and `F0` override the nominal values so that disturbances and the
    manipulated variable can be driven from outside.
    """
    CA, T, Tj = x
    T0 = p.T0 if T0 is None else T0
    CA0 = p.CA0 if CA0 is None else CA0
    Fj = p.Fj if Fj is None else Fj
    F0 = p.F0 if F0 is None else F0

    k = rate_constant(T, p)
    q_ht = p.U * p.A_ht * (T - Tj)  # kJ/min removed from the reactor

    dCA = F0 * CA0 / p.V - CA * (F0 / p.V + k)
    dT = (
        F0 * (T0 - T) / p.V
        + p.dH * k * CA / (p.rho * p.cp)
        - q_ht / (p.V * p.rho * p.cp)
    )
    dTj = Fj * (p.Tj0 - Tj) / p.Vj + q_ht / (p.Vj * p.rhoj * p.cpj)
    return np.array([dCA, dT, dTj])


def steady_state(
    p: ReactorParameters,
    guess: np.ndarray | None = None,
    **overrides: float,
) -> np.ndarray:
    """Solve `rhs_constant_volume` for a steady state near `guess`."""
    if guess is None:
        guess = np.array([2.353, 362.4, 345.69])
    sol, _, ier, msg = fsolve(
        lambda x: rhs_constant_volume(0.0, x, p, **overrides),
        guess,
        full_output=True,
        xtol=1e-13,
    )
    if ier != 1:
        raise RuntimeError(f"CSTR steady state did not converge: {msg}")
    return sol


def jacobian(
    x: np.ndarray, p: ReactorParameters, eps: float = 1e-7, **overrides: float
) -> np.ndarray:
    """Central-difference Jacobian of `rhs_constant_volume` at `x`."""
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.empty((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (
            rhs_constant_volume(0.0, xp, p, **overrides)
            - rhs_constant_volume(0.0, xm, p, **overrides)
        ) / (2 * h)
    return J


def eigenvalues(x: np.ndarray, p: ReactorParameters, **overrides: float) -> np.ndarray:
    """Eigenvalues of the linearized reactor, sorted by real part descending."""
    lam = np.linalg.eigvals(jacobian(x, p, **overrides))
    return lam[np.argsort(-lam.real)]


# --------------------------------------------------------------------------------
# Heat generation and removal, Section 17.1.2
# --------------------------------------------------------------------------------


def heat_generated(T: np.ndarray, p: ReactorParameters, CA0: float | None = None):
    """Heat released by reaction at steady state, Equation 17.11, in kJ/min.

    At steady state the component balance fixes C_A for a given T, so the heat
    released follows the S-shaped curve in T that produces the multiplicity.
    """
    CA0 = p.CA0 if CA0 is None else CA0
    k = rate_constant(T, p)
    tau = p.V / p.F0
    CA = CA0 / (1.0 + k * tau)
    return p.dH * k * CA * p.V


def heat_removed(T: np.ndarray, p: ReactorParameters, T0: float | None = None):
    """Heat carried away by the feed and the jacket, Equation 17.12, in kJ/min.

    The jacket temperature is eliminated using its own steady-state balance, which
    makes this a straight line in the reactor temperature.
    """
    T0 = p.T0 if T0 is None else T0
    # Steady-state jacket balance: Fj (Tj0 - Tj) rhoj cpj + U A (T - Tj) = 0
    c = p.U * p.A_ht / (p.Fj * p.rhoj * p.cpj)
    Tj = (p.Tj0 + c * T) / (1.0 + c)
    return p.F0 * p.rho * p.cp * (T - T0) + p.U * p.A_ht * (T - Tj)


def steady_state_branches(p: ReactorParameters, T_range=(290.0, 420.0), n=20_000):
    """Temperatures at which generation and removal balance.

    Returns the reactor temperatures of all steady states, low branch first.
    """
    T = np.linspace(*T_range, n)
    f = heat_generated(T, p) - heat_removed(T, p)
    sign_change = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = []
    for i in sign_change:
        lo, hi = T[i], T[i + 1]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (heat_generated(mid, p) - heat_removed(mid, p)) * (
                heat_generated(lo, p) - heat_removed(lo, p)
            ) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append(0.5 * (lo + hi))
    return np.array(roots)


def branch_start(T: float, p: ReactorParameters) -> np.ndarray:
    """A start for `steady_state` on the branch at temperature T: C_A from the component
    balance and T_j from the jacket's at that temperature."""
    CA = p.CA0 / (1.0 + rate_constant(T, p) * p.V / p.F0)
    c = p.U * p.A_ht / (p.Fj * p.rhoj * p.cpj)
    Tj = (p.Tj0 + c * T) / (1.0 + c)
    return np.array([CA, T, Tj])


def steady_states(p: ReactorParameters) -> list[np.ndarray]:
    """Every steady state of `rhs_constant_volume`, [C_A, T, T_j], low branch first: each
    temperature of `steady_state_branches` refined on the full model."""
    return [steady_state(p, guess=branch_start(T, p)) for T in steady_state_branches(p)]


# --------------------------------------------------------------------------------
# Measurement spans and operating limits
# --------------------------------------------------------------------------------
#
# Shared by the cases that hold this reactor, `jacketed_cstr` and the reactor of
# `reactor_separator_recycle`, so that the two cannot drift apart.  The reference gives
# no transmitter ranges and no limits, so each number is a choice, as plain data that the
# cases turn into `core.limits.Limits` (`casekit.limits_from`): a span is (lo, hi, why), a
# limit (value, why), by the measurement names of `jacketed_cstr`.
LIMITS = {
    "T": {
        "span": (340.0, 385.0, "the range the composition primary moves the temperature "
                               "set-point in"),
        "trip_hi": (372.0, "10 K above the design point (362.40 K), 5 K below the "
                           "high-temperature steady state (377.08 K)"),
        "alarm_lo": (350.0, "on the way to the low-temperature steady state (338.22 K), "
                            "where the reactor loses its production"),
    },
    "C_A": {"span": (0.0, 5.0, "up to the feed concentration, C_A0 = 5 kmol/m3")},
}  # K, kmol/m3


# --------------------------------------------------------------------------------
# Plantwide form: variable volume, three species
# --------------------------------------------------------------------------------


def rhs_multicomponent(
    x: np.ndarray,
    p: ReactorParameters,
    F_in: float,
    C_in: np.ndarray,
    T_in: float,
    F_out: float,
    Fj: float,
) -> np.ndarray:
    """Reactor block of the plantwide model.  State is [V, C_I, C_A, C_B, T, T_j].

    Derived from the total, component and energy balances without assuming constant
    volume:

        dV/dt     = F_in - F_out
        V dC_j/dt = F_in (C_in,j - C_j) + V r_j
        V dT/dt   = F_in (T_in - T) + (-dH) k C_A V / (rho cp)
                    - U A (T - T_j) / (rho cp)

    The accumulation term C_j dV/dt cancels against the outflow, so the component
    equations take the same form as the constant-volume case with F_in in place of
    F_0.  With one species, V constant and F_in = F_out this is Equation 17.10.
    """
    V = x[0]
    C = x[1 : 1 + N_SPECIES]
    T, Tj = x[1 + N_SPECIES], x[2 + N_SPECIES]

    k = rate_constant(T, p)
    r = np.array([0.0, -k * C[A], k * C[A]])  # kmol/(m3 min)
    q_ht = p.U * p.A_ht * (T - Tj)

    dV = F_in - F_out
    dC = (F_in * (C_in - C) + V * r) / V
    dT = (
        F_in * (T_in - T) + p.dH * k * C[A] * V / (p.rho * p.cp) - q_ht / (p.rho * p.cp)
    ) / V
    dTj = Fj * (p.Tj0 - Tj) / p.Vj + q_ht / (p.Vj * p.rhoj * p.cpj)

    return np.concatenate([[dV], dC, [dT, dTj]])
