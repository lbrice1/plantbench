"""`reactor_separator_recycle` as a `Case`: a reactor-separator-recycle plant with heat integration.

The reference
configuration, `CASE.config()`, is the base plant under the reference structure
(`effluent fixed + cascade`).  The `network` option selects one of the heat-integrated
plants of the design comparison instead, and `pump_around_share` moves D4 along the trade between heat
recovery and jacket authority.  Nothing here changes the model: every option selects
something the study scripts already build, so the reference results are untouched.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from plantbench.core import control as ctl
from plantbench.core import disturbances as dist
from plantbench.core.case import Case, Config, override
from plantbench.core.casekit import parameter_mismatch

from . import economics as ec
from . import hen, integrated, plant, streams
from . import measurements as meas
from . import scenarios as sc
from .parameters import PlantParameters

SPECIES = ("I", "A", "B")
NETWORKS = ("none", "D1", "D3", "D4", "D5")
# Structures a heat-integrated plant supports: the reference structure with its feed
# temperature loop, with or without the composition primary.
INTEGRATED_STRUCTURES = {"effluent fixed + cascade": True, "effluent fixed": False}

OPTIONS = {
    "network": "none",  # "none" is the base plant, whose reactor inlet temperature is held
    "pump_around_share": None,  # D4 only: share of the jacket duty sent to the reboiler
    "V_boil": 28.0,  # kmol/min, the design boil-up, which sets the separation
}


@lru_cache(maxsize=16)
def _design(pp: PlantParameters, V_boil: float) -> plant.Design:
    # About 1.2 s.  Parameters are frozen dataclasses and so hashable, and a sweep that
    # varies only tuning or disturbances then solves the design once per process.
    return plant.solve_design(pp, V_boil=V_boil)


@lru_cache(maxsize=32)
def _integrated(
    pp: PlantParameters, V_boil: float, network: str, share: float | None
) -> integrated.IntegratedDesign:
    design = _design(pp, V_boil)
    if share is not None:
        if network != "D4":
            raise ValueError("pump_around_share applies to the vacuum network D4 only")
        net = hen.vacuum_network(share)
    else:
        net = hen.DESIGNS[network]
    # The column pressure moves every temperature in the column and leaves the separation
    # unchanged, so it decides which matches a network can make rather than how well the
    # column separates.  Overriding it re-sizes the network and moves the tray temperature
    # loop with it, because `integrated.stage2` reads the pressure from the network.
    P = pp.column.P_network
    if P is not None:
        if P <= 0.0:
            raise ValueError("column.P_network is a pressure in bar and must be positive")
        net = replace(net, pressure=float(P))
    return integrated.build(design, net)


def make_design(config: Config):
    pp = override(PlantParameters(), config.params)
    opts = config.options
    network = opts["network"]
    if network not in NETWORKS:
        raise ValueError(f"network must be one of {NETWORKS}, not {network!r}")
    if network == "none":
        if opts["pump_around_share"] is not None:
            raise ValueError("pump_around_share needs network D4")
        return _design(pp, float(opts["V_boil"]))
    share = opts["pump_around_share"]
    return _integrated(pp, float(opts["V_boil"]), network, None if share is None else float(share))


# Parameters a structure or a network is built on: the tray temperature loop reads the
# column pressure when it is made, and the network is sized at it.
_DESIGN_CHOICES = ("column.P_base", "column.P_network")


def apply_mismatch(design, overrides):
    """The plant under changed parameters on the design (`Config.mismatch`).  The column
    pressures are refused: they are operating choices the loops and the network are built
    on, not properties in which the plant could differ from its design."""
    refused = sorted(set(overrides) & set(_DESIGN_CHOICES))
    if refused:
        raise ValueError(f"{refused} cannot be a mismatch: the tray temperature loop and the "
                         "heat-exchanger network are built on the column pressure; use params")
    return parameter_mismatch(design, overrides)


def make_structure(design, config: Config) -> ctl.Structure:
    if isinstance(design, integrated.IntegratedDesign):
        if config.structure not in INTEGRATED_STRUCTURES:
            raise ValueError(
                f"a heat-integrated plant supports the structures "
                f"{list(INTEGRATED_STRUCTURES)}, not {config.structure!r}"
            )
        return integrated.stage2(design, cascade=INTEGRATED_STRUCTURES[config.structure])
    return sc.STRUCTURES[config.structure](design)


def _pressure(design) -> float:
    if isinstance(design, integrated.IntegratedDesign):
        return design.pressure
    return design.pp.column.P_base


def measurements(design) -> dict:
    tray = sc.TEMPERATURE_TRAY
    out = {
        "reactor V": meas.m_reactor_volume,
        "reactor T": meas.m_reactor_T,
        "reactor C_A": meas.m_reactor_CA,
        "drum level": meas.m_drum_level,
        "base level": meas.m_base_level,
        "bottoms x_A": meas.m_bottoms_A,
        f"tray {tray} T": meas.m_tray_T(tray, _pressure(design)),
        "recycle": meas.m_recycle,
    }
    if isinstance(design, integrated.IntegratedDesign):
        nb = design.n_base
        out["feed T"] = lambda x, u, pp: float(x[nb])
    return out


def economics(setup) -> dict:
    """Operating profit and global warming potential at the design operating point.

    The design point is the basis on which the design comparison compares the networks, and it is
    the only basis on which a design that cycles can be evaluated at all: such a run has
    no settled operating point, so an objective read from the end of its trajectory would
    depend on where in the cycle the run happened to stop.

    Profit and utility cost are $/h, both global warming potentials t CO2e/h, and the heat
    recovered GJ/h: the network's duties are kJ/min internally and are converted here, so
    that every number the feature records is on the same per-hour basis as the reference
    results.  Under a mismatch the design point is the nominal design's.
    """
    design = setup.design if setup.nominal is None else setup.nominal
    if isinstance(design, integrated.IntegratedDesign):
        pp = design.pp
        op = streams.design_point(design.base)
        loads = design.sized.loads  # sized at this same operating point in `integrated.build`
        recovered = design.sized.recovered * ec.KJ_PER_MIN_TO_GJ_PER_H
    else:
        pp = design.pp
        op = streams.design_point(design)
        loads = ec.direct_loads(streams.stream_table(op, pp.column.P_base, pp), pp)
        recovered = 0.0
    e = ec.evaluate(op, loads, pp)
    return {
        "profit": e.profit,
        "utility_cost": e.utility_cost,
        "gwp_energy": e.gwp_energy,
        "gwp_total": e.gwp_total,
        "recovered": recovered,
    }


def state_names(design) -> list[str]:
    base = design.base if isinstance(design, integrated.IntegratedDesign) else design
    n_trays = base.pp.column.n_trays
    names = ["V"] + [f"C_{s}" for s in SPECIES] + ["T", "Tj"]
    names += ["M_D"] + [f"xD_{s}" for s in SPECIES]
    names += [f"M_{k}" for k in range(1, n_trays + 1)]
    names += [f"x_{k}_{s}" for k in range(1, n_trays + 1) for s in SPECIES]
    names += ["M_B"] + [f"xB_{s}" for s in SPECIES]
    if isinstance(design, integrated.IntegratedDesign):
        names += ["T_in"] + [f"Q[{e.match.hot} -> {e.match.cold}]" for e in design.sized.exchangers]
    return names


# -- disturbances ------------------------------------------------------------------------


def _handle(design, config: Config) -> str:
    if isinstance(design, integrated.IntegratedDesign):
        return "F_out"
    return sc.THROUGHPUT_HANDLE[config.structure]


def throughput(design, config: Config, fraction: float, t: float = 30.0):
    """A throughput change, made through whichever input sets production in the structure."""
    return sc.throughput_step(fraction, _handle(design, config), t)


def inert(design, config: Config, fraction: float, t: float = 30.0):
    """A change in the inert content of the fresh feed, by a fraction of its value."""
    return sc.inert_step(fraction, t)


def purge_closed(design, config: Config, t: float = 30.0):
    """The purge valve shut: the inert has no way out, and the plant no steady state."""
    return sc.purge_closed(t)


def fresh_feed_temperature(design, config: Config, change: float, t: float = 30.0):
    """The fresh feed arriving `change` kelvin away from its storage temperature.

    Only the heat-integrated plants see it; the base plant holds its reactor inlet
    temperature, so step `T_fresh` with the generic "step" disturbance there instead.
    """
    if not isinstance(design, integrated.IntegratedDesign):
        raise ValueError(
            "fresh_feed_temperature acts on the heat-integrated plants; "
            "for the base plant step T_fresh"
        )
    T0 = design.pp.thermo.T_storage
    return dist.step("T_storage", value=T0 + change, t=t)


CASE = Case(
    id="reactor_separator_recycle",
    title="Reactor-separator-recycle plant with heat integration",
    summary=(
        "A non-isothermal CSTR on the open-loop unstable branch of its multiplicity, in a "
        "recycle loop closed by a thirty-stage column with a purge; four heat-exchanger "
        "networks at three column pressures; regulatory structures declared as data. "
        "134 states for the base plant. Frozen: its reference configuration reproduces "
        "the reference results byte for byte."
    ),
    options=OPTIONS,
    structures=tuple(sc.STRUCTURES),
    default_structure="effluent fixed + cascade",
    make_design=make_design,
    make_structure=make_structure,
    disturbances={
        "throughput": throughput,
        "inert": inert,
        "purge_closed": purge_closed,
        "fresh_feed_temperature": fresh_feed_temperature,
    },
    measurements=measurements,
    state_names=state_names,
    features={"economics": economics},
    apply_mismatch=apply_mismatch,
)
