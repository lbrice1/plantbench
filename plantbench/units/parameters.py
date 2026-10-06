"""Parameters of the unit operations: the non-isothermal CSTR of Romagnoli and
Palazoglu (2020), the column and the vapor-pressure law that gives the column its temperatures.  The defaults
are the reference values used by `reactor_separator_recycle` and `jacketed_cstr`.

The parameter table of the reference (its Table 17.1) does not reproduce the steady state
and eigenvalues it reports.  Three defects were identified by solving backwards from the steady state in
Table 17.2 (C_A = 2.353 kmol/m3, T = 362.4 K, T_j = 345.69 K); each modification is marked
MODIFICATION below with the evidence for it.  With all three applied, the eigenvalues of
the linearized reactor are -0.799, -0.1349, +0.0595, against the -0.7989, -0.1349, +0.060
reported in the reference.

The Simulink model `CSTR_Model_Controlmodified.mdl` is a *different* parameter set
(k0 = 1.69e13, E = 82500, -dH = 63000, Tj0 = 294, steady state 1.76 / 358 / 342.1) and
does not reproduce the reference results.  It is recorded in SIMULINK_OVERRIDES for
reference but is not used.

Units are minutes, cubic meters, kmol and kelvin throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

R_GAS = 8.314  # kJ / kmol K


@dataclass(frozen=True)
class ReactorParameters:
    """The reference non-isothermal CSTR, first-order irreversible A -> B."""

    # -- kinetics -------------------------------------------------------------
    # Confirmed correct as printed: these give k = 0.1573 1/min at T = 362.4 K,
    # which is exactly what the component balance requires at the Table 17.2
    # steady state.
    k0: float = 2.81e11  # 1/min
    E: float = 85_000.0  # kJ/kmol
    dH: float = 84_000.0  # -(delta H), kJ/kmol, positive for exothermic

    # -- feed -----------------------------------------------------------------
    CA0: float = 5.0  # kmol/m3
    F0: float = 3.2  # m3/min
    # MODIFICATION: T0 is absent from Table 17.1 altogether.  Solving the three
    # steady-state balances with T pinned to the published 362.4 K gives
    # T0 = 336.97 K, and with it C_A = 2.3565 and T_j = 345.78 against the
    # published 2.353 and 345.69, and eigenvalues +0.060, -0.135, -0.799 against
    # the published +0.060, -0.1349, -0.7989 -- exact to the printed precision.
    # The 330 K and 340 K in the reference text (Section 17.1.2) are disturbance cases
    # that drive the reactor onto the low- and high-temperature stable branches,
    # not the nominal feed temperature.
    T0: float = 336.97  # K

    # -- reactor --------------------------------------------------------------
    V: float = 22.82  # m3
    rho: float = 1500.0  # kg/m3
    cp: float = 3.9  # kJ/kg K
    A_ht: float = 42.08  # m2
    # MODIFICATION: Table 17.1 gives Ut = 20124 kJ/(h m2 C) while every flow in the
    # table is per minute.  The model runs in minutes, so U = Ut/60 = 335.4.
    # The jacket energy balance at the Table 17.2 steady state independently
    # requires 332.95 kJ/(min m2 K), a 0.7% match on rounded state values.
    U: float = 20_124.0 / 60.0  # kJ/(min m2 K)

    # -- jacket ---------------------------------------------------------------
    Fj: float = 1.092  # m3/min
    Tj0: float = 294.4  # K
    # MODIFICATION: Table 17.1 prints 5.4 m3.  With 5.4 the eigenvalues come out
    # -0.884, -0.133, +0.060; with the 6.0 m3 used in the Simulink model they are
    # -0.799, -0.1349, +0.0595, matching the reference eigenvalues to three decimals.
    Vj: float = 6.0  # m3
    rhoj: float = 1000.0  # kg/m3
    cpj: float = 4.18  # kJ/kg K


# Steady state reported in the reference (its Table 17.2).  Used as the verification
# target and as the operating point the plantwide design is built around.
REFERENCE_STEADY_STATE = {"CA": 2.353, "T": 362.4, "Tj": 345.69}

# Eigenvalues reported in the reference.
REFERENCE_EIGENVALUES = (0.060, -0.1349, -0.7989)

# Parameter set of CSTR_Model_Controlmodified.mdl, for reference only.
SIMULINK_OVERRIDES = {
    "k0": 1.69e13,
    "E": 82_500.0,
    "dH": 63_000.0,
    "Tj0": 294.0,
    "T0": 335.0,
    "Vj": 6.0,
}

# Controller settings carried in the Simulink model.  The temperature subsystem holds
# three PID blocks behind a manual switch; the middle one is the PI used here as a
# starting point for tuning.  Sign convention: negative gain, since raising the coolant
# flow lowers the reactor temperature.
SIMULINK_TUNING = {
    "T_loop": {"Kc": -0.961, "tau_I": 1.0 / 0.31},
    "C_loop": {"Kc": 3.15, "tau_I": 3.15 / 2.0},
}


@dataclass(frozen=True)
class ColumnParameters:
    """Tray-by-tray distillation column separating the reactor effluent.

    Three components, ordered light to heavy: inert I, reactant A, product B.
    Relative volatilities are referenced to the heavy product B.  The inert is the
    lightest component, so it leaves overhead with the unreacted reactant and
    accumulates in the recycle loop unless purged.
    """

    n_trays: int = 30  # theoretical stages between condenser and reboiler
    feed_tray: int = 15  # 1 = top tray
    alpha: tuple[float, ...] = (6.0, 3.0, 1.0)  # I, A, B relative to B

    # Francis weir hydraulics: the liquid leaving a tray is
    #     L = weir_coeff * max(M - holdup_weir, 0) ** 1.5
    # so that holdup is a state and the internal liquid rate is free to step up
    # below the feed tray without being told to.
    holdup_weir: float = 0.5  # kmol held below the weir crest, no flow
    weir_coeff: float = 34.0  # kmol/min per kmol**1.5
    tray_holdup: float = 1.0  # kmol, nominal total holdup per tray (seed only)
    drum_holdup: float = 12.0  # kmol, nominal reflux drum holdup
    base_holdup: float = 18.0  # kmol, nominal column base holdup

    # Vapor is treated as constant molar overflow, set by the reboiler duty through
    # a constant latent heat.
    lambda_vap: float = 32_000.0  # kJ/kmol

    # Column pressure of the base plant, D1.  Compositions do not depend on it (see
    # ThermoParameters); tray temperatures, and so the tray temperature loop, do.
    P_base: float = 0.8  # bar

    # Column pressure a heat-integrated plant runs at, overriding the pressure its
    # heat-exchanger network was defined for; None keeps that pressure.  It is declared
    # here, as a parameter rather than a case option, because a case that gains an option
    # changes the hash of every configuration ever made from it, and with it the identity
    # of every run in every dataset already generated.  Parameters carry no such default
    # into a configuration, so this one is free to add.
    P_network: float | None = None  # bar


@dataclass(frozen=True)
class ThermoParameters:
    """Vapor pressures, and with them the temperatures the column never needed before.

    Every species follows ln Psat_j = a_j - lambda/(R T) with the same latent heat that
    constant molar overflow already assumes.  Relative volatility is then exactly
    independent of temperature and pressure, alpha_j = exp(a_j - a_B), so adding
    temperatures changes no composition, flow or result computed without them.  The
    cost of the simplification is that column pressure moves temperature levels but not
    the ease of separation.

    The product's normal boiling point is a design choice, made so that the question
    of whether reactor heat can drive the reboiler has different answers at different
    column pressures: yes below about 0.25 bar, no above.  The reactant's and inert's
    boiling points then follow from the volatilities: 359.0 K and 337.2 K.
    """

    Tb_product: float = 400.0  # K, normal boiling point of B
    P_ref: float = 1.01325  # bar
    T_storage: float = 300.0  # K, fresh feed as delivered
    # Stored hot enough that cooling water alone can bring them there: its limit is 318 K
    # (a 308 K mean plus the minimum approach), so 320 K keeps chilled water out of product
    # cooling, which would otherwise appear at every column pressure.
    T_product_storage: float = 320.0  # K, products and purge sent to storage
    dT_min: float = 10.0  # K, minimum approach for heat recovery


def reference_parameters() -> ReactorParameters:
    """Reactor parameters as the reference intends them, with the three modifications."""
    return ReactorParameters()


def simulink_parameters() -> ReactorParameters:
    """The parameter set actually carried in CSTR_Model_Controlmodified.mdl.

    Provided so the difference can be demonstrated; it does not reproduce the
    reference results.
    """
    return replace(ReactorParameters(), **SIMULINK_OVERRIDES)
