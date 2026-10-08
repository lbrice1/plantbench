"""The deethanizer: a feed heater, a 30-stage column with a total condenser and a
reboiler, and the two product let-down valves.

Units and their models:

  * E-100, the feed heater: one well-mixed volume whose outlet temperature is a state,
    heated by its duty.  The feed's composition passes through unchanged.  The outlet,
    two-phase at the design, enters stage 15 at its own enthalpy.
  * The column, `plantbench.units.staged_column` with the reboiler: ideal stages, Francis
    weirs, the stage energy balance, the HYSYS stage pressures above the top.
  * The condenser: total, its drum a well-mixed liquid at its bubble point at the top
    pressure.  The top pressure is that of one lumped vapour inventory, the column's and
    the drum's vapour space, which fills with the vapour leaving stage 1 and empties as
    the condenser duty condenses it to the drum liquid.
  * The reflux is an input, a flow: FIC-101, which sets it in the HYSYS case, has an
    integral time of half a second and is taken as ideal.  The products leave through
    linear valves whose flow is their opening times their flow at full opening.

States, in kmol, mole fractions, K and kPa:

    T_E100                         heater outlet
    M_D, x_D[n_c]                  condenser drum
    M[30], x[30, n_c]              stages, index 0 at the top
    M_B, x_B[n_c]                  reboiler
    P_top                          stage 1, the vapour to the condenser

`Plant.evaluate(..., hold=True)` is the model with four of the design specifications held
exactly (the heater outlet, the top pressure and the two levels), the inputs that hold
them computed from the balances and reported.  It is used to bring a rough state near the
design point before Newton; it is not a different model of the plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from plantbench.backend import namespace
from plantbench.core.casekit import StateLayout, inputs_key, measured, same_values
from plantbench.core.control import batch_inputs
from plantbench.units.staged_column import ColumnSpec, Feed, equilibrium_batch, evaluate_batch

from .components import LABELS, equation_of_state, feed
from .parameters import DeethanizerParameters

KPA = 1e3  # Pa per kPa
KW = 60.0  # kJ/min per kW
R_GAS = 8.314462618  # kJ/(kmol K)

# The design inputs, one per specification of `design_residuals`.
DESIGN_INPUTS = ("Q_E100", "Q_condenser", "Q_reboiler", "L_reflux", "v_VLV101", "v_VLV102")


@dataclass
class Inputs:
    """Every quantity a loop may manipulate or a disturbance may change."""

    F_feed: float  # kmol/min
    z_feed: np.ndarray  # mole fractions
    T_feed: float  # K
    P_feed: float  # kPa, the E-100 inlet
    Q_E100: float  # kW supplied by E-100
    L_reflux: float  # kmol/min
    Q_condenser: float  # kW removed by the condenser
    Q_reboiler: float  # kW supplied by the reboiler
    v_VLV101: float  # 0-1, distillate
    v_VLV102: float  # 0-1, LPG


def layout(pp: DeethanizerParameters, nc: int) -> StateLayout:
    n = pp.n_stages
    return StateLayout([("T_E100", ()), ("M_D", ()), ("x_D", nc), ("M", n), ("x", (n, nc)),
                        ("M_B", ()), ("x_B", nc), ("P_top", ())])


@dataclass
class Streams:
    """What one evaluation computes besides the derivatives, read by the measurements,
    and in hold mode the inputs that hold the specifications."""

    column: object  # staged_column.Profile
    T_drum: float  # K
    h_drum: float  # kJ/kmol, the drum liquid: distillate and reflux
    F_condensed: float  # kmol/min
    D: float  # distillate, kmol/min
    B: float  # LPG, kmol/min
    h_feed: float  # kJ/kmol, the feed entering E-100
    h_heated: float  # kJ/kmol, leaving E-100
    vapour_heated: float  # the vapour fraction leaving E-100
    implied: dict = field(default_factory=dict)

    def row(self, i: int) -> Streams:
        """The streams of member i of a batch, as one evaluation returns them."""
        return Streams(column=type(self.column)(**{k: v[i] for k, v in vars(self.column).items()}),
                       **{k: float(getattr(self, k)[i]) for k in _SCALARS},
                       implied={k: float(v[i]) for k, v in self.implied.items()})


_SCALARS = ("T_drum", "h_drum", "F_condensed", "D", "B", "h_feed", "h_heated",
            "vapour_heated")


@dataclass
class Plant:
    """The model on one parameter set.

    Its iterative solves (flashes, bubble points) start from values it holds.  While the
    design is being solved these follow the last evaluation; `freeze` fixes them at the
    design point, after which an evaluation is a function of the state and inputs alone,
    to the last bit, as a finite-difference Jacobian needs.

    `evaluate` takes one state or a batch.  A batch starts every member's solves from the
    values held and never updates them, and solves the cubic of the equation of state in
    closed form (`PengRobinson.on`), on the device its states are on."""

    pp: DeethanizerParameters
    _warm: dict = field(default_factory=dict, repr=False)
    _frozen: bool = field(default=False, repr=False)
    _snapshot: tuple = field(default=(None, None, None), repr=False)
    _state_only: tuple = field(default=(None, None, None), repr=False)

    def __post_init__(self):
        pp = self.pp
        if not 1 < pp.feed_stage < pp.n_stages:
            raise ValueError(f"feed_stage must lie between 2 and {pp.n_stages - 1}, "
                             f"not {pp.feed_stage}")
        if len(pp.P_offsets) != pp.n_stages + 1:
            raise ValueError(f"P_offsets needs {pp.n_stages + 1} values, the stages and the "
                             f"reboiler; it has {len(pp.P_offsets)}")
        self.eos = equation_of_state()
        self.nc = self.eos.c.n
        self.layout = layout(pp, self.nc)
        self.column = ColumnSpec(n_stages=pp.n_stages, n_components=self.nc,
                                 weir_coeff=pp.weir_coeff, holdup_weir=pp.holdup_weir,
                                 dP_stage=0.0,
                                 P_offsets=tuple(o * KPA for o in pp.P_offsets))
        self.C_P = pp.V_overhead / (R_GAS * pp.T_overhead)  # kmol/kPa
        self._batch_eos = {}

    def _eos(self, xp, batch: bool):
        if not batch:
            return self.eos
        device = "cpu" if xp is np else "cuda"
        if device not in self._batch_eos:
            self._batch_eos[device] = self.eos.on(device)
        return self._batch_eos[device]

    def _start(self, key, m: int, xp):
        """The held start of a solve, repeated for the m members of a batch."""
        v = self._warm.get(key)
        if v is None:
            return None
        v = xp.asarray(v)
        return xp.tile(v, (m,) + (1,) * (v.ndim - 1))

    def state_names(self) -> list[str]:
        return self.layout.names({k: LABELS for k in ("x_D", "x", "x_B")})

    def freeze(self, x: np.ndarray, u: Inputs) -> None:
        """Start every later evaluation's iterations from their values at (x, u)."""
        self._frozen = False
        # Both caches hold results from the previous warm starts.
        self._state_only = (None, None, None)
        self._snapshot = (None, None, None)
        self.evaluate(x, u)
        self._frozen = True

    def _keep(self, key, value) -> None:
        if not self._frozen:
            self._warm[key] = value

    def streams(self, x: np.ndarray, u: Inputs) -> Streams:
        """The streams at a state, evaluated once for all the measurements that read them."""
        # Keyed by the values of the inputs, not the object: a control structure writes its
        # loops' outputs into one inputs object in turn, measuring between the writes.
        key = inputs_key(u)
        key_x, key_u, st = self._snapshot
        if key_x is None or not same_values(key_x, x) or key_u != key:
            st = self.evaluate(x, u)[1]
            self._snapshot = (x.copy(), key, st)
        return st

    def temperatures(self, x):
        """The drum's bubble point (K) and the stage and reboiler temperatures (K,
        n_stages + 1), of one state or a batch: once the plant is frozen these depend on
        the state alone, so a measure that reads them needs no evaluation of the streams,
        whose key also holds the inputs a control structure is still writing."""
        if x.ndim == 1:
            T_D, eq = self._equilibrium(x[None], batch=False)
            return float(T_D[0]), eq[1][0]
        T_D, eq = self._equilibrium(x, batch=True)
        return T_D, eq[1]

    def _equilibrium(self, x, batch: bool):
        """The drum's bubble point and the column's `equilibrium_batch`, x (m, n_x); held
        for the next call at the same state once the plant is frozen."""
        key_batch, key_x, held = self._state_only
        if self._frozen and key_batch is batch and key_x is not None and same_values(key_x, x):
            return held
        xp, m = namespace(x), x.shape[0]
        eos, keep = self._eos(xp, batch), not batch
        s = self.layout.unpack(x)
        P_drum = s["P_top"] * KPA
        T_D, _ = eos.bubble_T(P_drum, s["x_D"], T0=self._start("drum_T", m, xp))
        if keep:
            self._keep("drum_T", T_D)
        eq = equilibrium_batch(self.column, eos, s["x"], s["x_B"], s["P_top"] * KPA,
                               T0=self._start("column_T", m, xp))
        if keep:
            self._keep("column_T", eq[1][0])
        if self._frozen:
            self._state_only = (batch, x.copy(), (T_D, eq))
        return T_D, eq

    def _flash(self, eos, key, T, P, z, keep: bool):
        xp = namespace(z)
        f = eos.flash_PT(T, P * KPA, z, K0=self._start(key, len(z), xp))
        if keep:
            self._keep(key, f.K)
        return f

    def evaluate(self, x: np.ndarray, u: Inputs, hold: bool = False):
        """The derivatives of every state, and the streams.

        For a batch, `x` is (m, n_x) and the fields of `u` carry a leading axis of m
        (`control.batch_inputs`); the derivatives are then (m, n_x), and each stream
        quantity (m,)."""
        if x.ndim == 1:
            d, st = self._evaluate(x[None], batch_inputs(u, 1), hold, batch=False)
            return d[0], st.row(0)
        return self._evaluate(x, u, hold, batch=True)

    def _evaluate(self, x, u: Inputs, hold: bool, batch: bool):
        xp = namespace(x)
        m = x.shape[0]
        pp, eos, keep = self.pp, self._eos(xp, batch), not batch
        s = self.layout.unpack(x)
        z = u.z_feed
        F = u.F_feed
        implied = {}

        # E-100.
        h_in = self._flash(eos, "feed", u.T_feed, u.P_feed, z, keep).h
        heated = self._flash(eos, "heated", s["T_E100"], u.P_feed - pp.dP_E100, z, keep)
        h_out = heated.h
        if hold:
            implied["Q_E100"] = F * (h_out - h_in) / KW
        Q_E100 = implied["Q_E100"] if hold else u.Q_E100
        dT_E100 = 0.0 if hold else (F * (h_in - h_out) + Q_E100 * KW) / pp.C_E100

        # The drum liquid, at its bubble point at the top pressure, and the column.
        T_D, eq = self._equilibrium(x, batch)
        h_D = eos.h(T_D, s["P_top"] * KPA, s["x_D"], "liquid")
        feeds = [Feed(0, u.L_reflux, s["x_D"], h_D), Feed(pp.feed_stage - 1, F, z, h_out)]
        B = u.v_VLV102 * pp.F_VLV102_max
        dM, dxs, dM_B, dx_B, col = evaluate_batch(
            self.column, eos, s["M"], s["x"], s["M_B"], s["x_B"], s["P_top"] * KPA, feeds,
            Q_R=u.Q_reboiler * KW, B=B, eq=eq)
        if hold:
            # The reboiler composition does not depend on the draw, so it needs no
            # second evaluation.
            B = col.L[:, -1] - col.V[:, -1]
            implied["v_VLV102"] = B / pp.F_VLV102_max
            dM_B = 0.0

        # The condenser, from the vapour leaving stage 1 to the drum liquid.
        V_top, h_top, y_top = col.V[:, 0], col.h_V[:, 0], col.y[:, 0]
        latent = h_top - h_D
        if hold:
            implied["Q_condenser"] = V_top * latent / KW
        Q_cond = implied["Q_condenser"] if hold else u.Q_condenser
        F_cond = Q_cond * KW / latent
        dP_top = 0.0 if hold else (V_top - F_cond) / self.C_P

        D = u.v_VLV101 * pp.F_VLV101_max
        if hold:
            D = F_cond - u.L_reflux
            implied["v_VLV101"] = D / pp.F_VLV101_max
        dM_D = F_cond - u.L_reflux - D
        dx_D = F_cond[:, None] * (y_top - s["x_D"]) / s["M_D"][:, None]

        dx_state = self.layout.pack(T_E100=dT_E100, M_D=dM_D, x_D=dx_D, M=dM, x=dxs,
                                    M_B=dM_B, x_B=dx_B, P_top=dP_top)
        full = (lambda v: xp.broadcast_to(xp.asarray(v, dtype=float), (m,)))
        streams = Streams(column=col, T_drum=T_D, h_drum=h_D, F_condensed=F_cond,
                          D=full(D), B=full(B), h_feed=h_in, h_heated=h_out,
                          vapour_heated=heated.V,
                          implied={k: full(v) for k, v in implied.items()})
        return dx_state, streams


# ------------------------------------------------------------------------------------
# The design point
# ------------------------------------------------------------------------------------


@dataclass
class Design:
    """The plant at its design steady state, for the closed-loop simulator."""

    pp: DeethanizerParameters
    x: np.ndarray
    u: Inputs
    plant: Plant
    # The right-hand side and the measures accept a batch of states.
    batched: ClassVar[bool] = True

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: DeethanizerParameters) -> np.ndarray:
        return self.plant.evaluate(x, u)[0]


def default_inputs(plant: Plant) -> Inputs:
    """The feed at its stated conditions, and the HYSYS case's duties, reflux and
    openings."""
    from . import hysys_reference

    pp, ref = plant.pp, hysys_reference.load()
    q, c = ref["duty_kJ_h"], ref["controllers"]
    return Inputs(F_feed=pp.F_feed, z_feed=feed(), T_feed=pp.T_feed, P_feed=pp.P_feed,
                  Q_E100=q["E-100"] / 3600, L_reflux=ref["column"]["reflux_kmol_h"] / 60,
                  Q_condenser=q["condenser"] / 3600, Q_reboiler=q["reboiler"] / 3600,
                  v_VLV101=c["LIC-100"]["OP_percent"] / 100,
                  v_VLV102=c["LIC-101"]["OP_percent"] / 100)


def initial_state(plant: Plant) -> np.ndarray:
    """A state near the HYSYS steady state, for `settle`: the stage holdups from the
    reported liquid flows, the stage liquids from the reported light components, and the
    remainder in the proportions of the LPG's butanes and heavier.  Not a steady state."""
    from . import hysys_reference

    return hysys_reference.initial_state(plant)


def ethane_recovery(plant: Plant, u: Inputs, st: Streams, x_D: np.ndarray) -> float:
    """The fraction of the feed's ethane leaving in the distillate."""
    C2 = plant.eos.c.names.index("ethane")
    return measured(st.D * x_D[..., C2] / (u.F_feed * u.z_feed[..., C2]))


def design_residuals(plant: Plant, x: np.ndarray, u: Inputs) -> np.ndarray:
    """The right-hand side and the specifications of the design, one per design input:

        the E-100 outlet at T_E100_design           fixes the E-100 duty
        the top at P_top                            fixes the condenser duty
        stage 4 at T_stage4_design (TIC-101)        fixes the reflux
        ethane in the LPG at x_C2_LPG_design        fixes the reboiler duty
                                       (XIC-100, through TIC-100)
        each vessel at its design level             fixes its product valve

    Temperatures in K; the ethane error is scaled to kelvin, 0.002 mole fraction to the
    kelvin, about the slope of the bottom stages; the levels to percent."""
    pp = plant.pp
    d, st = plant.evaluate(x, u)
    s = plant.layout.unpack(x)
    C2 = plant.eos.c.names.index("ethane")
    spec = [s["T_E100"] - pp.T_E100_design, s["P_top"] - pp.P_top,
            st.column.T[3] - pp.T_stage4_design,
            (s["x_B"][C2] - pp.x_C2_LPG_design) / 0.002,
            100.0 * (s["M_D"] / pp.M_drum_full - pp.level_drum_design),
            100.0 * (s["M_B"] / pp.M_reboiler_full - pp.level_reboiler_design)]
    return np.concatenate([d, spec])


# kmol/min of reflux per K per min: the rate at which `settle` moves the reflux against
# the stage 4 temperature's error; and kW per unit of the ethane error (scaled as in
# `design_residuals`) per min, the rate at which it moves the reboiler duty.  Slower
# than the column, some tens of minutes, so that what each reads is near settled.
SETTLE_REFLUX_GAIN = 0.02
SETTLE_REBOILER_GAIN = 5.0
SETTLE_T_RANGE = (200.0, 450.0)  # K, the temperatures settle evaluates at


def settle(plant: Plant, x0: np.ndarray, u0: Inputs, t_end: float = 600.0,
           tol: float = 1e-4) -> tuple[np.ndarray, Inputs]:
    """Integrate the plant in hold mode from a rough state towards its design point, and
    return the state and the inputs that hold it.  The reflux and the reboiler duty, the
    design inputs hold mode leaves free, are moved by integral actions on the stage 4
    temperature and the ethane in the LPG.  Stops at `t_end`, or once the largest
    derivative is below `tol`."""
    from scipy.integrate import solve_ivp

    pp = plant.pp
    nx = len(x0)
    C2 = plant.eos.c.names.index("ethane")
    k_T = plant.layout.slices["T_E100"].start
    k_xB = plant.layout.slices["x_B"].start + C2
    x0 = np.array(x0, dtype=float)
    x0[k_T] = pp.T_E100_design
    x0[plant.layout.slices["P_top"].start] = pp.P_top

    def f(t, X):
        x = X[:nx].copy()
        x[k_T] = np.clip(x[k_T], *SETTLE_T_RANGE)
        u = replace(u0, L_reflux=X[nx], Q_reboiler=X[nx + 1])
        d, st = plant.evaluate(x, u, hold=True)
        dL = SETTLE_REFLUX_GAIN * (st.column.T[3] - pp.T_stage4_design)
        dQ = SETTLE_REBOILER_GAIN * (x[k_xB] - pp.x_C2_LPG_design) / 0.002
        return np.concatenate([d, [dL, dQ]])

    def near(t, X):
        return np.max(np.abs(f(t, X))) - tol
    near.terminal = True

    sol = solve_ivp(f, (0.0, t_end), np.concatenate([x0, [u0.L_reflux, u0.Q_reboiler]]),
                    method="BDF", rtol=1e-6, atol=1e-8, events=near)
    if sol.status == -1:
        raise ValueError(f"settling towards the design failed at t = {sol.t[-1]:.3g} min: "
                         f"{sol.message}")
    x = sol.y[:nx, -1]
    u = replace(u0, L_reflux=sol.y[nx, -1], Q_reboiler=sol.y[nx + 1, -1])
    implied = plant.evaluate(x, u, hold=True)[1].implied
    return x, replace(u, **implied)


def solve_design(pp: DeethanizerParameters, tol: float = 1e-10,
                 start: tuple[np.ndarray, Inputs] | None = None) -> Design:
    """The design steady state: Newton (MINPACK's hybrid method) on the states and the
    design inputs together, from `start`.  Without one, from the HYSYS state
    (`initial_state`), which converges at the default parameters in a few seconds, and
    failing that from the HYSYS state brought near the design point by `settle`."""
    plant = Plant(pp)
    if start is not None:
        return _newton(plant, start, tol)
    x0, u0 = initial_state(plant), default_inputs(plant)
    try:
        return _newton(plant, (x0, u0), tol)
    except ValueError:
        return _newton(Plant(pp), settle(Plant(pp), x0, u0), tol)


def _newton(plant: Plant, start: tuple[np.ndarray, Inputs], tol: float) -> Design:
    from scipy.optimize import root

    pp = plant.pp
    x0, u0 = start
    nx = len(x0)

    def unpack(X):
        return X[:nx], replace(u0, **dict(zip(DESIGN_INPUTS, X[nx:])))

    def residual(X):
        try:
            return design_residuals(plant, *unpack(X))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            # A flash or bubble point that fails far from the solution: steer away.
            return np.full(len(X), 1e6)

    X0 = np.concatenate([x0, [getattr(u0, k) for k in DESIGN_INPUTS]])
    # Warm starts fixed at the start, so that the residual is a function of X alone and
    # the finite-difference Jacobian carries no history.
    plant.freeze(x0, u0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sol = root(residual, X0, method="hybr", options=dict(maxfev=50 * len(X0), xtol=1e-13))
        # Again from the solution with the warm starts fixed there, as the design will
        # hold them: the bubble points' round-off depends on their start, and the top
        # pressure, whose inventory is small, amplifies it to 1e-8 kPa/min.
        plant.freeze(*unpack(sol.x))
        sol = root(residual, sol.x, method="hybr",
                   options=dict(maxfev=10 * len(X0), xtol=1e-13))
    worst = float(np.max(np.abs(residual(sol.x))))
    if worst > tol:
        raise ValueError(f"the design solve did not converge: largest residual {worst:.3g} "
                         f"({sol.message})")
    x, u = unpack(sol.x)
    valves = [v for v in DESIGN_INPUTS if v.startswith("v_")]
    if not all(0.0 < getattr(u, v) < 1.0 for v in valves):
        raise ValueError("the design needs a valve outside (0, 1): "
                         + ", ".join(f"{v} {getattr(u, v):.3f}" for v in valves))
    plant.freeze(x, u)
    return Design(pp=pp, x=x, u=u, plant=plant)
