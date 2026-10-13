"""The case interface: what a test problem provides, how a configuration selects one plant
from it, and the trajectory a run returns.

A case is a plant model with a design point, a set of named control structures, a set of
named disturbances and a set of named measurements.  A `Config` selects one member of
the case's design space: the structure, the case's own options (a heat-exchanger
network, say), overrides of its parameters, set-points, tuning, non-idealities and a
disturbance.  Everything in a `Config` is plain data, so it can be written to TOML or
JSON and hashed, which is what lets a dataset record exactly how each run was made.

`build` turns a case and a configuration into a `Setup` ready to simulate; `run`
simulates it and returns a `Trajectory`.  Neither knows anything about a particular
case: each step that is case-specific is a function the case supplies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np

from . import control as ctl
from . import disturbances as dist
from .instruments import Actuator, Instrument

Measurement = Callable[[np.ndarray, Any, Any], float]
# A feature is computed from the configured plant at its design point, not from the run, so
# it is recorded even when the integration fails and it costs nothing beyond the design.
Feature = Callable[["Setup"], dict]


# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One member of a case's design space, as plain data.

    structure        name of a control structure the case provides
    options          case-specific choices, checked against the case's declared options
    params           overrides of the case's parameters by dotted path, "reactor.U"
    setpoints        loop name -> set-point, replacing the design value
    setpoint_steps   loop name -> [(time, change), ...] during the run
    tuning           loop name -> {"Kc", "tau_I", "lo", "hi"}
    instruments      loop name -> keyword arguments of `Instrument`
    actuators        loop name -> keyword arguments of `Actuator`
    disturbance      {"name": ..., **arguments}, or a list of them; empty for none

    Every section is empty by default, and a configuration with only a structure named
    is the case's reference plant.
    """

    structure: str
    options: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, float] = field(default_factory=dict)
    setpoints: Mapping[str, float] = field(default_factory=dict)
    setpoint_steps: Mapping[str, list] = field(default_factory=dict)
    tuning: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    instruments: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    actuators: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    disturbance: Mapping[str, Any] | list = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Config:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown configuration sections {sorted(unknown)}; "
                           f"known: {sorted(known)}")
        if "structure" not in d:
            raise KeyError("a configuration must name its structure")
        return cls(**{k: _plain(v) for k, v in d.items()})

    def key(self) -> str:
        """A stable hash of the configuration, the same across processes and sessions."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode()).hexdigest()[:16]


def _plain(v):
    """Convert to JSON-compatible plain data: dicts, lists, str, float, int, bool, None."""
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def override(obj, overrides: Mapping[str, Any]):
    """A copy of a (possibly nested) frozen dataclass with fields replaced by dotted path.

    `override(pp, {"reactor.U": 300.0})` replaces `pp.reactor.U`.  Raises on a path that
    does not name a field, so a misspelt parameter cannot be silently ignored.
    """
    for path, value in overrides.items():
        obj = _override_one(obj, path.split("."), value, path)
    return obj


def _override_one(obj, parts: list[str], value, path: str):
    if not dataclasses.is_dataclass(obj):
        raise KeyError(f"{path!r}: {type(obj).__name__} is not a parameter set")
    names = {f.name for f in dataclasses.fields(obj)}
    head = parts[0]
    if head not in names:
        raise KeyError(f"{path!r}: {type(obj).__name__} has no field {head!r}; "
                       f"fields: {sorted(names)}")
    if len(parts) == 1:
        current = getattr(obj, head)
        if isinstance(current, tuple):
            value = tuple(value)
        return replace(obj, **{head: value})
    return replace(obj, **{head: _override_one(getattr(obj, head), parts[1:], value, path)})


# ------------------------------------------------------------------------------------
# Cases
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """What a test problem supplies to the library.

    id, title, summary     identification, and one paragraph on what the case is for
    options                case-specific option names and their defaults
    structures             names of the control structures the case provides
    default_structure      the structure of the case's reference plant
    make_design            config -> a design at its steady state, with `x`, `u`, `pp`,
                           `rhs(t, x, u, pp)` and optionally `derived`
    make_structure         (design, config) -> the named structure, biased from the design
    disturbances           name -> factory(design, config, **arguments) -> disturbance;
                           the generic "step" and "ramp" are always available as well
    measurements           design -> {name: measure(x, u, pp)}, recorded in trajectories
    state_names            design -> one name per plant state
    features               name -> f(setup) -> {name: value}, case-specific quantities a
                           dataset may record alongside each run; the generic features of
                           `GENERIC_FEATURES` are available to every case
    """

    id: str
    title: str
    summary: str
    options: Mapping[str, Any]
    structures: tuple[str, ...]
    default_structure: str
    make_design: Callable[[Config], Any]
    make_structure: Callable[[Any, Config], ctl.Structure]
    disturbances: Mapping[str, Callable[..., dist.Disturbance]]
    measurements: Callable[[Any], Mapping[str, Measurement]]
    state_names: Callable[[Any], list[str]]
    features: Mapping[str, Feature] = field(default_factory=dict)

    def config(self, structure: str | None = None, **sections) -> Config:
        """A configuration of this case, with its options defaulted and checked."""
        cfg = Config(structure=structure or self.default_structure, **sections)
        return self.validate(cfg)

    def validate(self, config: Config | Mapping) -> Config:
        """Fill in default options and reject anything this case does not provide."""
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        if config.structure not in self.structures:
            raise KeyError(f"case {self.id} has no structure {config.structure!r}; "
                           f"structures: {list(self.structures)}")
        unknown = set(config.options) - set(self.options)
        if unknown:
            raise KeyError(f"case {self.id} has no options {sorted(unknown)}; "
                           f"options: {sorted(self.options)}")
        options = {**self.options, **config.options}
        return replace(config, options=options)


GENERIC_DISTURBANCES: dict[str, Callable[..., dist.Disturbance]] = {
    "step": lambda design, config, **kw: dist.step(**kw),
    "ramp": lambda design, config, **kw: dist.ramp(**kw),
}


@dataclass
class Setup:
    """A configured plant at its design point, ready to simulate."""

    case: Case
    config: Config
    design: Any
    structure: ctl.Structure
    disturbance: dist.Disturbance | None

    def spectrum(self) -> np.ndarray:
        """Closed-loop eigenvalues at the design point, structural zeros removed."""
        return ctl.closed_loop_spectrum(self.design, self.structure)


def _spectrum_feature(setup: Setup) -> dict:
    """Damping and stability of the closed loop linearized at the design point."""
    ev = setup.spectrum()
    return {"damping_ratio": ctl.damping_ratio(ev),
            "rightmost": float(ev.real.max()),
            "n_unstable": int((ev.real > 0).sum())}


# Features every case can record, because they need only the closed loop.
GENERIC_FEATURES: dict[str, Feature] = {"spectrum": _spectrum_feature}


def features_of(case: Case) -> dict[str, Feature]:
    """Every feature this case can record: the generic ones, and its own."""
    return {**GENERIC_FEATURES, **case.features}


def build(case: Case, config: Config | Mapping) -> Setup:
    """Resolve a configuration of `case` into a design, a structure and a disturbance."""
    config = case.validate(config)
    return build_on(case, case.make_design(config), config)


def build_on(case: Case, design, config: Config) -> Setup:
    """`build` on a design already made, for a validated configuration: the structure and
    the disturbance, which are all that the members of an ensemble do not share."""
    structure = case.make_structure(design, config)
    structure = ctl.with_tuning(structure, config.tuning)
    structure = ctl.with_setpoints(structure, config.setpoints)
    structure = ctl.with_setpoint_steps(structure, config.setpoint_steps)
    structure = ctl.with_instruments(
        structure,
        instruments={k: Instrument(**v) for k, v in config.instruments.items()},
        actuators={k: Actuator(**v) for k, v in config.actuators.items()},
    )
    return Setup(case, config, design, structure, _disturbance(case, design, config))


def design_key(config: Config) -> str:
    """What decides a configuration's design: configurations with the same key can be
    built on one design and run as one ensemble.  A case's `make_design` reads the
    parameters and the options; the structure is included as well, since the structures
    of an ensemble must have the same elements."""
    return json.dumps({"structure": config.structure, "options": dict(config.options),
                       "params": dict(config.params)}, sort_keys=True, default=str)


def _disturbance(case: Case, design, config: Config) -> dist.Disturbance | None:
    specs = config.disturbance
    if not specs:
        return None
    if isinstance(specs, Mapping):
        specs = [specs]
    available = {**GENERIC_DISTURBANCES, **case.disturbances}
    parts = []
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("name", None)
        if name not in available:
            raise KeyError(f"case {case.id} has no disturbance {name!r}; "
                           f"available: {sorted(available)}")
        parts.append(available[name](design, config, **spec))
    return parts[0] if len(parts) == 1 else dist.combine(*parts)


# ------------------------------------------------------------------------------------
# Trajectories
# ------------------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A closed-loop run: states, inputs and measurements on a common time grid.

    `x` is (n_states, n_times) with one name per row in `state_names`; `u` and `y` map
    names to arrays of n_times.  `wall_time` and `nfev` record what the run cost, which
    varies by more than an order of magnitude across a design space and is informative
    in its own right.  `c` maps "<loop>.SP", "<loop>.PV" and "<loop>.OP" to the
    signals of each loop, with "<loop>.PV%" and "<loop>.OP%" where the loop declares its
    spans (`control.apply`, `control.percent_signals`).
    """

    case: str
    config: dict
    t: np.ndarray
    x: np.ndarray
    state_names: list[str]
    u: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    wall_time: float = np.nan
    nfev: int = -1
    c: dict[str, np.ndarray] = field(default_factory=dict)

    def state(self, name: str) -> np.ndarray:
        return self.x[self.state_names.index(name)]

    def window(self, t0: float, t1: float) -> Trajectory:
        """The part of the run with t0 <= t <= t1."""
        m = (self.t >= t0) & (self.t <= t1)
        return replace(self, t=self.t[m], x=self.x[:, m],
                       u={k: v[m] for k, v in self.u.items()},
                       y={k: v[m] for k, v in self.y.items()},
                       c={k: v[m] for k, v in self.c.items()})

    def to_npz(self, path) -> None:
        """Write the run to `path`.  A run without controller signals, an open-loop one,
        is written without the array `c`, as every run was before they were recorded."""
        meta = dict(case=self.case, config=self.config, wall_time=self.wall_time,
                    nfev=self.nfev, state_names=self.state_names,
                    u_names=list(self.u), y_names=list(self.y))
        arrays = {}
        if self.c:
            meta["c_names"] = list(self.c)
            arrays["c"] = np.array([self.c[k] for k in self.c]).reshape(len(self.c), -1)
        np.savez_compressed(
            path, t=self.t, x=self.x,
            u=np.array([self.u[k] for k in self.u]).reshape(len(self.u), -1),
            y=np.array([self.y[k] for k in self.y]).reshape(len(self.y), -1),
            meta=np.array(json.dumps(meta)), **arrays,
        )

    @classmethod
    def from_npz(cls, path) -> Trajectory:
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            return cls(case=meta["case"], config=meta["config"], t=f["t"], x=f["x"],
                       state_names=meta["state_names"],
                       u=dict(zip(meta["u_names"], f["u"])),
                       y=dict(zip(meta["y_names"], f["y"])),
                       wall_time=meta["wall_time"], nfev=meta["nfev"],
                       c=dict(zip(meta.get("c_names", []), f["c"])) if "c" in f else {})


def run(case: Case, config: Config | Mapping, t_end: float, n_points: int | None = None,
        dt: float | None = None, rtol: float = 1e-8, atol: float = 1e-10,
        wall_budget: float | None = None, jacobian: str = "internal",
        device: str = "cpu") -> Trajectory:
    """Simulate one configuration of a case and return its trajectory.

    Give the output grid as `n_points` over [0, t_end] or as a spacing `dt`.  A
    `wall_budget` in seconds stops a run that is taking too long by raising
    `control.WallTimeExceeded`.  The integration restarts at every discontinuity the
    disturbance and the set-point schedule declare (`control.breakpoints_of`).
    `jacobian` and `device` choose how the solver's Jacobian is formed; see
    `control.integrate`.
    """
    setup = build(case, config)
    if n_points is None:
        n_points = 1200 if dt is None else int(round(t_end / dt)) + 1
    measurements = case.measurements(setup.design)
    start = time.perf_counter()
    r = ctl.integrate(setup.design, setup.structure, t_end, setup.disturbance, n_points,
                      rtol, atol, measurements=measurements, wall_budget=wall_budget,
                      breakpoints=ctl.breakpoints_of(setup.disturbance, setup.structure),
                      jacobian=jacobian, device=device)
    return Trajectory(case=case.id, config=setup.config.to_dict(), t=r.t, x=r.X,
                      state_names=list(case.state_names(setup.design)), u=r.U, y=r.Y,
                      wall_time=time.perf_counter() - start, nfev=r.nfev, c=r.C)


@dataclass
class MemberResult:
    """The outcome of one member of an ensemble: its trajectory, or the exception that
    stopped it (`control.WallTimeExceeded` for a timeout, `RuntimeError` for a failed
    integration).  `wall_time` of the trajectory is its share of the ensemble's."""

    trajectory: Trajectory | None
    error: Exception | None
    ensemble_size: int
    ensemble_wall_time: float


def run_setups(setups: list[Setup], t_end: float, n_points: int | None = None,
               dt: float | None = None, rtol: float = 1e-8, atol: float = 1e-10,
               wall_budget: float | None = None, device: str = "cpu",
               size: int | None = None) -> list[MemberResult]:
    """Simulate configured plants as ensembles (`core.ensemble`), at most `size` members
    each, and return one result per setup, in order.

    Setups are run together when they share a design object and the layout of their
    structures (`control.layout_signature`); `build_on` gives setups that share a design.
    The integration is Rodas5P rather than LSODA, so a trajectory agrees with `run`'s to
    the tolerance, not to the last bit.  `wall_budget` applies to each ensemble."""
    from .ensemble import integrate_ensemble

    if n_points is None:
        n_points = 1200 if dt is None else int(round(t_end / dt)) + 1
    groups: dict[tuple, list[int]] = {}
    for k, s in enumerate(setups):
        groups.setdefault((id(s.design), ctl.layout_signature(s.structure)), []).append(k)
    results: list[MemberResult | None] = [None] * len(setups)
    for members in groups.values():
        step = size or len(members)
        for c in range(0, len(members), step):
            chunk = members[c:c + step]
            design = setups[chunk[0]].design
            r = integrate_ensemble(design, [setups[k].structure for k in chunk],
                                   [setups[k].disturbance for k in chunk], t_end, n_points,
                                   rtol, atol, wall_budget=wall_budget, device=device)
            n = len(design.x)
            share = r.wall_time / len(chunk)
            for j, k in enumerate(chunk):
                s = setups[k]
                if r.status[j] == "timeout":
                    res = MemberResult(None, ctl.WallTimeExceeded(r.message[j]), len(chunk),
                                       r.wall_time)
                elif r.status[j] == "failed":
                    res = MemberResult(None, RuntimeError(f"closed-loop integration failed: "
                                                          f"{r.message[j]}"), len(chunk),
                                       r.wall_time)
                else:
                    U, Y, C = ctl.batch_outputs(design, s.structure, s.disturbance, r.t,
                                                r.Y[j], s.case.measurements(design))
                    traj = Trajectory(case=s.case.id, config=s.config.to_dict(), t=r.t,
                                      x=r.Y[j, :, :n].T.copy(),
                                      state_names=list(s.case.state_names(design)), u=U, y=Y,
                                      wall_time=share, nfev=int(r.nfev[j]), c=C)
                    res = MemberResult(traj, None, len(chunk), r.wall_time)
                results[k] = res
    return results


def run_ensemble(case: Case, configs: list, t_end: float, size: int | None = None,
                 **kwargs) -> list[MemberResult]:
    """Simulate configurations of a case as ensembles and return one result per
    configuration, in order.  Configurations with the same `design_key` share one design;
    a configuration that does not build has its exception as the result.  Keyword
    arguments go to `run_setups`."""
    designs: dict[str, Any] = {}
    setups, slots = [], []
    results: list[MemberResult | None] = [None] * len(configs)
    for k, config in enumerate(configs):
        try:
            config = case.validate(config)
            key = design_key(config)
            if key not in designs:
                designs[key] = case.make_design(config)
            setups.append(build_on(case, designs[key], config))
            slots.append(k)
        except Exception as exc:  # noqa: BLE001 -- the caller decides what a bad member is
            results[k] = MemberResult(None, exc, 0, 0.0)
    for k, res in zip(slots, run_setups(setups, t_end, size=size, **kwargs)):
        results[k] = res
    return results
