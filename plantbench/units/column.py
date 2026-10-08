"""Tray-by-tray dynamic model of the distillation column that separates the reactor
effluent.

Three components in order of decreasing volatility: inert I, reactant A, product B.
The inert and the unreacted reactant leave overhead, the product leaves as bottoms.

Assumptions, which are those of the standard tray-by-tray formulation:

  * ideal stages, with vapor and liquid leaving a tray in equilibrium
  * constant relative volatility, so the equilibrium is algebraic
  * constant molar overflow for the vapor, set by the reboiler duty
  * Francis weir hydraulics for the liquid, so that tray holdups are states and the
    reflux drum and column base have real level dynamics
  * a total condenser, with the distillate split downstream into recycle and purge
  * negligible vapor holdup

State layout, all holdups in kmol and all compositions mole fractions:

    M_D, x_D[3]                     reflux drum
    M[n_trays], x[n_trays, 3]       trays, index 0 at the top
    M_B, x_B[3]                     column base and reboiler

Tray 1 (index 0) is the top tray, fed by the reflux; the vapor from tray 1 goes to
the condenser.  The feed enters on `feed_tray` counted from the top, 1-based.
"""

from __future__ import annotations

import numpy as np

from plantbench.backend import namespace

from .parameters import ColumnParameters

N_SPECIES = 3


def n_states(cp: ColumnParameters) -> int:
    """Number of column states: (drum + trays + base) x (1 holdup + 3 compositions)."""
    return (cp.n_trays + 2) * (1 + N_SPECIES)


def unpack(z: np.ndarray, cp: ColumnParameters):
    """Split the flat state vector into drum, tray and base holdups and compositions."""
    nt = cp.n_trays
    i = 0
    MD = z[i]
    i += 1
    xD = z[i : i + N_SPECIES]
    i += N_SPECIES
    M = z[i : i + nt]
    i += nt
    x = z[i : i + nt * N_SPECIES].reshape(nt, N_SPECIES)
    i += nt * N_SPECIES
    MB = z[i]
    i += 1
    xB = z[i : i + N_SPECIES]
    return MD, xD, M, x, MB, xB


def pack(MD, xD, M, x, MB, xB) -> np.ndarray:
    return np.concatenate(
        [[MD], xD, M, np.asarray(x).reshape(-1), [MB], xB]
    )


def equilibrium(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vapor composition in equilibrium with liquid `x` at constant relative volatility.

    Works on a single composition vector or on an array of them, one row per tray.
    """
    # Clamp before normalizing.  A stiff transient, or an off-design iterate inside
    # the flowsheet solver, can push a mole fraction slightly negative; without the
    # clamp the normalization divides by zero and the whole integration derails.
    num = alpha * np.maximum(x, 0.0)
    total = num.sum(axis=-1, keepdims=True)
    return num / np.where(total > 0.0, total, 1.0)


def weir_flow(M: np.ndarray, cp: ColumnParameters) -> np.ndarray:
    """Liquid leaving a tray, from the Francis weir relation.

    Only the holdup above the weir crest flows over it, so

        L = weir_coeff * max(M - holdup_weir, 0) ** 1.5

    and the liquid rate is free to take whatever value the holdup implies.  Nothing
    in the model prescribes the internal liquid flow, which is what allows it to step
    up below the feed tray on its own, and the clamp at zero keeps a tray that runs
    dry during a transient from producing a negative flow.
    """
    xp = namespace(M)
    return cp.weir_coeff * xp.power(xp.maximum(M - cp.holdup_weir, 0.0), 1.5)


def rhs(
    z: np.ndarray,
    cp: ColumnParameters,
    F: float,
    zF: np.ndarray,
    q: float,
    V: float,
    L_reflux: float,
    D: float,
    Bm: float,
) -> np.ndarray:
    """Right-hand side of the column model.

    - `F`, `zF`, `q`: feed molar flow (kmol/min), composition and liquid fraction.
      `q` = 1 is a saturated liquid feed.
    - `V`: vapor boil-up from the reboiler (kmol/min), the manipulated variable that
      stands in for the reboiler duty.
    - `L_reflux`: reflux returned to the top tray (kmol/min).
    - `D`, `Bm`: distillate and bottoms molar flows (kmol/min), set by the level
      controllers.
    """
    nt = cp.n_trays
    alpha = np.asarray(cp.alpha, dtype=float)
    MD, xD, M, x, MB, xB = unpack(z, cp)

    ft = cp.feed_tray - 1  # 0-based index of the feed tray
    FL = q * F  # liquid part of the feed
    FV = (1.0 - q) * F  # vapor part of the feed

    # Vapor composition leaving each tray and the reboiler.
    y = equilibrium(x, alpha)
    yB = equilibrium(xB, alpha)

    # Liquid leaving each tray, set entirely by the holdup through the weir.
    L = weir_flow(M, cp)

    # Liquid entering each tray: reflux at the top, the tray above elsewhere, plus the
    # liquid part of the feed on the feed tray.
    L_in = np.empty(nt)
    L_in[0] = L_reflux
    L_in[1:] = L[:-1]
    x_in = np.empty((nt, N_SPECIES))
    x_in[0] = xD
    x_in[1:] = x[:-1]

    # Vapor rate leaving each tray.  Constant molar overflow makes it the reboiler
    # boil-up everywhere below the feed and, when the feed is partly vaporized, that
    # plus the vapor part of the feed at and above the feed tray.
    V_out = np.where(np.arange(nt) <= ft, V + FV, V)
    V_in = np.empty(nt)
    V_in[:-1] = V_out[1:]
    V_in[-1] = V  # the bottom tray is fed by the reboiler
    y_in = np.empty((nt, N_SPECIES))
    y_in[-1] = yB
    y_in[:-1] = y[1:]

    # Feed addition.
    feed_L = np.zeros(nt)
    feed_L[ft] = FL
    feed_V = np.zeros(nt)
    feed_V[ft] = FV
    feed_zL = np.zeros((nt, N_SPECIES))
    feed_zL[ft] = zF
    feed_zV = np.zeros((nt, N_SPECIES))
    feed_zV[ft] = zF

    # -- tray balances --------------------------------------------------------
    dM = L_in + feed_L + V_in - L - V_out
    dMx = (
        L_in[:, None] * x_in
        + feed_L[:, None] * feed_zL
        + V_in[:, None] * y_in
        + feed_V[:, None] * feed_zV
        - L[:, None] * x
        - V_out[:, None] * y
    )
    dx = (dMx - x * dM[:, None]) / M[:, None]

    # -- condenser and reflux drum -------------------------------------------
    # Total condenser: the vapor from the top tray condenses completely, so the drum
    # composition is the composition of that vapor.
    V_top = V_out[0]
    dMD = V_top - L_reflux - D
    dMDx = V_top * y[0] - (L_reflux + D) * xD
    dxD = (dMDx - xD * dMD) / MD

    # -- column base and reboiler --------------------------------------------
    dMB = L[-1] - V - Bm
    dMBx = L[-1] * x[-1] - V * yB - Bm * xB
    dxB = (dMBx - xB * dMB) / MB

    return pack(dMD, dxD, dM, dx, dMB, dxB)


def initial_state(
    cp: ColumnParameters,
    xD: np.ndarray,
    xB: np.ndarray,
) -> np.ndarray:
    """A starting state with nominal holdups and a linear composition profile.

    Only used to seed the steady-state solver; it is not itself a steady state.
    """
    nt = cp.n_trays
    frac = np.linspace(0.0, 1.0, nt)
    x = (1.0 - frac)[:, None] * np.asarray(xD) + frac[:, None] * np.asarray(xB)
    x = x / x.sum(axis=1, keepdims=True)
    return pack(
        cp.drum_holdup,
        np.asarray(xD, dtype=float),
        np.full(nt, cp.tray_holdup),
        x,
        cp.base_holdup,
        np.asarray(xB, dtype=float),
    )


def product_flows(cp: ColumnParameters, F: float, q: float, V: float, L_reflux: float):
    """Distillate and bottoms flows implied by a steady state at these conditions.

    The reflux drum balance fixes the distillate as the vapor reaching the condenser
    less the reflux; the overall balance then fixes the bottoms.  Using the overall
    balance rather than the base balance for B guarantees that the solved steady state
    closes on total moles.
    """
    D = (V + (1.0 - q) * F) - L_reflux
    return D, F - D


def steady_state(
    cp: ColumnParameters,
    F: float,
    zF: np.ndarray,
    q: float,
    V: float,
    L_reflux: float,
    guess: np.ndarray | None = None,
    t_settle: float = 4_000.0,
    warm: bool = False,
) -> np.ndarray:
    """Solve the column for the steady state at the given feed and internal flows.

    A hundred-odd coupled tray equations defeat a plain Newton solve from a crude
    initial profile, so the column is first integrated to rest and the result then
    polished with a Newton step.  Integration is the reliable half; the polish is
    what drives the residual to solver tolerance.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve

    D, Bm = product_flows(cp, F, q, V, L_reflux)
    if D <= 0 or Bm <= 0:
        raise ValueError(
            f"infeasible internal flows: D={D:.4f}, B={Bm:.4f} kmol/min. "
            "Reflux must be below the vapor rate and the bottoms flow positive."
        )
    def cold():
        """The initial profile: light ends above the feed, heavy below it."""
        z = np.asarray(zF, dtype=float)
        light = z.copy()
        light[2] *= 0.05
        heavy = z.copy()
        heavy[0] *= 0.01
        heavy[1] *= 0.05
        return initial_state(cp, light / light.sum(), heavy / heavy.sum())

    if guess is None:
        guess = cold()

    f = lambda z: rhs(z, cp, F, zF, q, V, L_reflux, D, Bm)

    # A warm start from a nearby steady state is close enough for Newton on its own,
    # which matters because this solve sits inside the plantwide design loop and the
    # integration below costs seconds rather than milliseconds.
    if warm and guess is not None:
        sol, _, ier, _ = fsolve(f, guess, full_output=True, xtol=1e-13)
        if ier == 1 and np.max(np.abs(f(sol))) < 1e-8:
            return sol

    def settle(start):
        return solve_ivp(lambda t, z: f(z), (0.0, t_settle), start, method="LSODA",
                         rtol=1e-10, atol=1e-12)

    traj = settle(guess)
    if warm and traj.success and not np.all(np.isfinite(traj.y[:, -1])):
        # LSODA can report success on a trajectory that has overflowed, which a warm start
        # has been seen to produce on one platform; settle again from the cold profile.
        traj = settle(cold())
    if not traj.success:
        raise RuntimeError(f"column integration to steady state failed: {traj.message}")
    if not np.all(np.isfinite(traj.y[:, -1])):
        raise RuntimeError("column integration to steady state ended in a non-finite state")

    sol, _, ier, _ = fsolve(f, traj.y[:, -1], full_output=True, xtol=1e-13)
    polished = np.max(np.abs(f(sol)))
    if ier != 1 or not np.isfinite(polished) or polished > np.max(np.abs(f(traj.y[:, -1]))):
        sol = traj.y[:, -1]  # keep the integrated result if the polish did not help
    return sol
