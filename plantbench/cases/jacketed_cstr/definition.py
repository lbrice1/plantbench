"""`jacketed_cstr` as a `Case`: the reference CSTR on its own.

The smallest case in the library, and the one to start from: three states, one
manipulated variable, and an operating point that is open-loop unstable.  It carries the
same verified ground truth as `reactor_separator_recycle`'s reactor, the published steady state and
eigenvalues, without the recycle, column or heat network around it.

Signs follow this library's convention, error = set-point minus measurement: raising
the coolant flow lowers the temperature, and raising the temperature lowers the
concentration, so both gains are negative.

The tunings are chosen for damping, as `reactor_separator_recycle`'s are.  The Simulink model that
accompanies the reference carries PI settings (`SIMULINK_TUNING`), but for a different
parameter set (steady state 1.76 kmol/m3, 358 K), and on this reactor they leave the
temperature loop at a damping ratio of 0.090 and the cascade at 0.014, where a 5% feed
concentration step drives the coolant valve onto its limit and into a sustained
oscillation.  They remain available through the `tuning` section of a configuration.
"""

from __future__ import annotations

from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
from plantbench.core.case import Case, Config, override
from plantbench.core.casekit import cached_design, state_measurement
from plantbench.core.control import Loop
from plantbench.units.parameters import ReactorParameters

from .model import LAYOUT, Design, solve_design

OPTIONS = {"branch": "middle"}


_design = cached_design(solve_design)


def make_design(config: Config) -> Design:
    return _design(override(ReactorParameters(), config.params), config.options["branch"])


m_CA = state_measurement("C_A", LAYOUT)
m_T = state_measurement("T", LAYOUT)
m_Tj = state_measurement("Tj", LAYOUT)


# The temperature loop is tuned as `reactor_separator_recycle`'s reactor temperature loop is, which gives a
# damping ratio of 0.27 on the reactor alone.  Over it, a primary of K_c = -3 K per
# kmol/m3 and tau_I = 20 min keeps the least damped mode at 0.21 with the slowest mode
# at -0.0115 1/min (87 min); `reactor_separator_recycle`'s primary, -6 and 120 min, gives 0.16 and 320 min
# here, and a gain of -10 falls below 0.1.
TEMPERATURE_TUNING = {"Kc": -1.0, "tau_I": 20.0}  # m3/min per K, min
COMPOSITION_TUNING = {"Kc": -3.0, "tau_I": 20.0}  # K per kmol/m3, min


def temperature_loop() -> Loop:
    """Reactor temperature on the coolant flow: the loop that stabilizes the reactor."""
    return Loop("reactor T", m_T, "Fj", **TEMPERATURE_TUNING, lo=0.05, hi=6.0)


def composition_loop() -> Loop:
    """Reactor concentration on the temperature set-point, the reference cascade."""
    return Loop("reactor C_A", m_CA, "sp:reactor T", **COMPOSITION_TUNING, lo=340.0, hi=385.0)


STRUCTURES = {
    "open loop": lambda: [],
    "temperature PI": lambda: [temperature_loop()],
    "composition cascade": lambda: [temperature_loop(), composition_loop()],
}


def make_structure(design: Design, config: Config) -> ctl.Structure:
    return ctl.bias_from_design(STRUCTURES[config.structure](), design)


def feed_temperature(design, config: Config, value: float | None = None,
                     change: float | None = None, t: float = 10.0):
    """The feed temperature moved to `value`, or by `change`, at time t.

    330 K and 340 K are the reference feed-temperature cases that carry the open-loop reactor onto
    its low- and high-temperature branches.
    """
    if (value is None) == (change is None):
        raise ValueError("give exactly one of value and change")
    return dist.step("T0", value=design.u.T0 + change if value is None else value, t=t)


CASE = Case(
    id="jacketed_cstr",
    title="Non-isothermal CSTR with a cooling jacket",
    summary=(
        "The reference reactor on its own: first-order exothermic A -> B, three states, "
        "three steady states, and a design point on the open-loop unstable middle "
        "branch. Structures from open loop to the composition cascade. Its steady state "
        "and eigenvalues reproduce the published values once three modifications to the "
        "published parameter table are applied."
    ),
    options=OPTIONS,
    structures=tuple(STRUCTURES),
    default_structure="temperature PI",
    make_design=make_design,
    make_structure=make_structure,
    disturbances={"feed_temperature": feed_temperature},
    measurements=lambda design: {"C_A": m_CA, "T": m_T, "Tj": m_Tj},
    state_names=lambda design: LAYOUT.names(),
)
