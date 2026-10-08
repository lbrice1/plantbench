"""The demethanizer section as one system of ordinary differential equations.

Three recovery schemes (Chebeir, Salas and Romagnoli 2019, Figs. 1-3), sharing the front
end, the column and the recompression:

    all       feed -> E-100 (hot side) -> E-101 chiller -> TK-100
              TK-100 liquid -> JTV-100 -> stage 26
              T-100 overhead -> [less the crr recycle] -> E-102 (cold side)
                  -> E-100 (cold side) -> booster (TE-100 shaft) -> K-101 -> sales gas
              reboiler E-103 -> NGL bottoms

    conventional   TK-100 vapour -> E-102 (hot side) -> TK-101 -> vapour -> TE-100 -> stage 1
                   TK-101 liquid -> JTV-101 -> stage 8
    gsp            TK-100 vapour split: R / (1 + R) -> TE-100 -> stage 8; the rest, with a
                   part of the TK-100 liquid draw, is the branch:
                   branch -> E-102 (hot side) -> JTV-101 -> stage 1
    crr            as gsp, the branch through E-104 (cold side) to stage 2; a part of the
                   overhead -> K-102 -> E-104 (hot side) -> stage 1

Stage numbers are those marked on the figures; the assignment of streams to them, the
branch's liquid and the crr recycle follow the HYSYS case "Demethanizer - Tuned"
(README.md, Provenance).  In the code, stages are 0-based.

Under crr, stage 1 receives vapour only, the recycle and the vapour of stage 2, and in
HYSYS passes no liquid.  An equilibrium tray there holds a liquid that nothing renews,
whose heavy components accumulate on a time scale of a day, so the steady state is not
determined by the tray.  Stage 1 is therefore an adiabatic mixing point without holdup:
the column's trays are stages 2 to 30, and the overhead is the vapour of stage 2 mixed
with the recycle, any condensate travelling with it.

Units and their states:

  * E-100, E-102 and E-104, counter-current exchangers, each as `n_cells` cells in
    series; each side of each cell has a temperature state with the heat capacity of its
    share of the metal, and exchanges UA / n_cells times the difference with the other
    side.  The hot stream enters cell 0 and the cold stream cell n - 1.  Cell enthalpies
    are those of the stream at the cell temperature, from a PT flash, so condensation on
    a hot side is in the balance.
  * E-101, a duty removed from the stream; no state.
  * TK-100 and TK-101, flash drums: the inlet is flashed at the drum temperature and
    pressure, its vapour leaves at once and its liquid joins the holdup, which leaves by
    the level valve.  States: temperature, which relaxes to the adiabatic flash
    temperature of the inlet with the drum's heat capacity, liquid holdup and liquid
    composition.
  * TE-100, adiabatic at `eta_machines`, discharging at the pressure of the stage it
    feeds; its shaft work drives the booster.  K-102 likewise, an ideal gas, raising the
    recycle by `dP_K102`.
  * T-100, `plantbench.units.staged_column` with the reboiler.
  * The column top and overhead circuit, one lumped vapour inventory whose pressure is the
    top pressure.  It fills with the overhead vapour and empties through K-102 and through
    the booster and K-101, whose flows at given powers follow from their pressure ratios,
    the residue gas taken as an ideal gas with each machine's heat capacity.
  * Flow transmitters on the NGL and, under crr, on the recycle and the residue gas,
    first-order with time constant `tau_FT`: the measurements ERIC-100 and RFIC-100
    read, which are flows algebraic in valves and machine powers.

The separator pressures are parameters, as published, and the feed flow is an input: the
front end has no pressure-flow relation.

`Plant.evaluate(..., hold=True)` is the model with five of the design specifications held
exactly (the levels, the top pressure, the TK-100 temperature, and in crr the recycle
ratio), the inputs that hold them computed from the balances and reported.  It
is used to bring a rough state near the design point before Newton; it is not a
different model of the plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import ClassVar

import numpy as np

from plantbench.backend import namespace
from plantbench.core.casekit import StateLayout, inputs_key, same_values
from plantbench.core.control import batch_inputs
from plantbench.units.peng_robinson import Flash
from plantbench.units.staged_column import ColumnSpec, Feed, equilibrium_batch, evaluate_batch

from .components import LABELS, equation_of_state, feed
from .parameters import BASES, NGLParameters

KPA = 1e3  # Pa per kPa
KW = 60.0  # kJ/min per kW
R_GAS = 8.314462618  # kJ/(kmol K)
SCHEMES = ("conventional", "gsp", "crr")

# 0-based trays of the feeds of each scheme.  Under crr the trays are stages 2-30 and
# the recycle enters the mixing point above them (stage 1).
STAGES = {
    "conventional": {"expander": 0, "TK101": 7, "TK100": 25},
    "gsp": {"branch": 0, "expander": 7, "TK100": 25},
    "crr": {"branch": 0, "expander": 6, "TK100": 24},
}


def n_trays(pp: NGLParameters, scheme: str) -> int:
    """The equilibrium trays: every stage, but stage 1 under crr."""
    return pp.n_stages - 1 if scheme == "crr" else pp.n_stages


# The design inputs of each scheme, one per specification of `design_residuals`.
DESIGN_INPUTS = {
    "conventional": ("Q_chiller", "Q_reboiler", "W_recompressor", "v_LCV100", "v_LCV101",
                     "v_LCV102"),
    "gsp": ("Q_chiller", "Q_reboiler", "W_recompressor", "v_LCV100", "v_LCV102"),
    "crr": ("Q_chiller", "Q_reboiler", "W_recompressor", "v_LCV100", "v_LCV102", "W_K102"),
}


@dataclass
class Inputs:
    """Every quantity a loop may manipulate or a disturbance may change.  One dataclass
    for the three schemes, so that a dataset spanning them has one set of columns; a
    field a scheme does not use is recorded as a constant."""

    F_feed: float  # kmol/min
    z_feed: np.ndarray  # mole fractions
    T_feed: float  # K
    P_feed: float  # kPa
    Q_chiller: float  # kW removed by E-101
    Q_reboiler: float  # kW supplied by E-103
    W_recompressor: float  # kW, K-101
    v_LCV100: float  # 0-1, TK-100 liquid
    v_LCV101: float  # 0-1, TK-101 liquid, conventional
    v_LCV102: float  # 0-1, NGL bottoms
    split: float  # R, TE-100 flow over the branch's vapour, gsp and crr
    W_K102: float  # kW, K-102, crr
    liquid_split: float = 0.0  # 0-1, the part of the TK-100 liquid draw to the branch


def layout(pp: NGLParameters, nc: int, scheme: str) -> StateLayout:
    n = pp.n_cells
    blocks = [("T_E100h", n), ("T_E100c", n), ("T_E102h", n), ("T_E102c", n)]
    if scheme == "crr":
        blocks += [("T_E104h", n), ("T_E104c", n)]
    blocks += [("T_TK100", ()), ("M_TK100", ()), ("x_TK100", nc)]
    if scheme == "conventional":
        blocks += [("T_TK101", ()), ("M_TK101", ()), ("x_TK101", nc)]
    nt = n_trays(pp, scheme)
    blocks += [("M", nt), ("x", (nt, nc)), ("M_B", ()), ("x_B", nc),
               ("P_top", ()), ("FT_NGL", ())]
    if scheme == "crr":
        blocks += [("FT_recycle", ()), ("FT_residue", ())]
    return StateLayout(blocks)


@dataclass
class Streams:
    """What one evaluation computes besides the derivatives: the streams between units,
    read by the measurements, and in hold mode the inputs that hold the specifications."""

    tk100: object  # the TK-100 inlet flashed at the drum
    V1: float  # TK-100 vapour, kmol/min
    L100: float  # TK-100 liquid draw
    F_expander: float
    y_expander: np.ndarray
    T_expander_in: float  # K
    P_expander_in: float  # kPa
    h_expander_out: float  # kJ/kmol
    W_expander: float  # kJ/min
    column: object  # staged_column.Profile
    B: float  # NGL, kmol/min
    V0: float  # overhead vapour, under crr the recycle included
    F_comp: float  # residue gas through the booster and K-101
    P_booster_out: float  # kPa
    T_booster_out: float  # K, ideal gas
    T_K101_out: float  # K, ideal gas
    tk101: object = None  # conventional
    L101: float = 0.0
    F_branch: float = 0.0  # gsp, crr: vapour and liquid
    F_recycle: float = 0.0  # crr
    # Enthalpies at the plant's boundary, for its energy balance, kJ/kmol and kJ/min.
    h_feed: float = 0.0
    h_residue: float = 0.0  # the residue gas leaving E-100, entering the booster
    Q_chiller: float = 0.0  # kJ/min
    H_K102: float = 0.0  # enthalpy K-102 adds to the recycle, crr
    h_overhead: float = 0.0  # kJ/kmol, the overhead, after the crr mixing point
    implied: dict = field(default_factory=dict)

    def row(self, i: int) -> Streams:
        """The streams of member i of a batch, as one evaluation returns them."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "implied":
                out[f.name] = {k: float(w[i]) for k, w in v.items()}
            elif f.name in ("tk100", "tk101"):
                out[f.name] = None if v is None else v.member(i)
            elif f.name == "column":
                out[f.name] = type(v)(**{k: w[i] for k, w in vars(v).items()})
            elif f.name == "y_expander":
                out[f.name] = v[i]
            else:
                out[f.name] = float(v[i])
        return Streams(**out)


@dataclass
class Plant:
    """The model of one scheme on one parameter set and component set.

    Its iterative solves (flashes, bubble points, the recompression) start from values it
    holds.  While the design is being solved these follow the last evaluation; `freeze`
    fixes them at the design point, after which an evaluation is a function of the state
    and inputs alone, to the last bit, as a finite-difference Jacobian needs.

    `evaluate` takes one state or a batch.  A batch starts every member's solves from the
    values held and never updates them, and solves the cubic of the equation of state in
    closed form (`PengRobinson.on`), on the device its states are on."""

    pp: NGLParameters
    scheme: str = "conventional"
    basis: str = "chebeir2019"
    _warm: dict = field(default_factory=dict, repr=False)
    _frozen: bool = field(default=False, repr=False)
    _snapshot: tuple = field(default=(None, None, None), repr=False)

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, not {self.scheme!r}")
        if self.basis not in BASES:
            raise ValueError(f"basis must be one of {tuple(BASES)}, not {self.basis!r}")
        if self.basis == "hysys" and self.scheme != "crr":
            raise ValueError("the hysys basis is a crr plant; it has no "
                             f"{self.scheme} design")
        self.components = BASES[self.basis]
        self.eos = equation_of_state(self.components)
        self.nc = self.eos.c.n
        self.layout = layout(self.pp, self.nc, self.scheme)
        pp = self.pp
        offsets = (np.asarray(pp.P_offsets, dtype=float) if pp.P_offsets is not None
                   else pp.dP_stage * np.arange(pp.n_stages + 1))
        skip = pp.n_stages - n_trays(pp, self.scheme)  # stage 1 under crr
        self.column = ColumnSpec(n_stages=pp.n_stages - skip, n_components=self.nc,
                                 weir_coeff=pp.weir_coeff, holdup_weir=pp.holdup_weir,
                                 dP_stage=pp.dP_stage * KPA,
                                 P_offsets=tuple(o * KPA for o in offsets[skip:]))
        self.C_P = self.pp.V_overhead / (R_GAS * self.pp.T_overhead)  # kmol/kPa
        self._batch_eos = {}

    def state_names(self) -> list[str]:
        labels = LABELS[self.components]
        return self.layout.names({k: labels for k in ("x_TK100", "x_TK101", "x", "x_B")})

    def freeze(self, x: np.ndarray, u: Inputs) -> None:
        """Start every later evaluation's iterations from their values at (x, u)."""
        self._frozen = False
        self._snapshot = (None, None, None)  # held from the previous warm starts
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

    # -- one evaluation ----------------------------------------------------------------

    def _eos(self, xp, batch: bool):
        if not batch:
            return self.eos
        device = "cpu" if xp is np else "cuda"
        if device not in self._batch_eos:
            self._batch_eos[device] = self.eos.on(device)
        return self._batch_eos[device]

    def _start(self, key, m: int, xp, default=None):
        """The held start of a solve, repeated for the m members of a batch."""
        v = self._warm.get(key)
        if v is None:
            return default
        v = xp.asarray(v)
        return xp.tile(v, (m,) + (1,) * (v.ndim - 1))

    def _flash(self, eos, key, T, P, z, keep: bool):
        """PT flashes of r streams in each of m members: T and P (m, r), z (m, r, n_c).
        The result's arrays carry both axes."""
        xp = namespace(z)
        m, r, nc = z.shape
        f = eos.flash_PT(T.reshape(-1), P.reshape(-1), z.reshape(-1, nc),
                         K0=self._start(key, m, xp))
        if keep:
            self._keep(key, f.K)
        return Flash(**{k: v.reshape((m, r) + v.shape[1:]) for k, v in vars(f).items()})

    def _vapour(self, eos, T, P, y):
        """Enthalpy and entropy of a vapour stream at T K and P kPa, each (m,)."""
        P = namespace(T).full_like(T, P * KPA) if np.ndim(P) == 0 else P * KPA
        return eos.h(T, P, y, "vapour"), eos.s(T, P, y, "vapour")

    def _liquid_h(self, eos, T, P, x):
        P = namespace(T).full_like(T, P * KPA) if np.ndim(P) == 0 else P * KPA
        return eos.h(T, P, x, "liquid")

    def evaluate(self, x: np.ndarray, u: Inputs, hold: bool = False):
        """The derivatives of every state, and the streams.

        For a batch, `x` is (m, n_x) and the fields of `u` carry a leading axis of m
        (`control.batch_inputs`); the derivatives are then (m, n_x), and each stream
        quantity carries the leading axis."""
        if x.ndim == 1:
            d, st = self._evaluate(x[None], batch_inputs(u, 1), hold, batch=False)
            return d[0], st.row(0)
        return self._evaluate(x, u, hold, batch=True)

    def _evaluate(self, x, u: Inputs, hold: bool, batch: bool):
        xp = namespace(x)
        m = x.shape[0]
        pp, n, scheme = self.pp, self.pp.n_cells, self.scheme
        eos, keep = self._eos(xp, batch), not batch
        s = self.layout.unpack(x)
        z = u.z_feed
        F = u.F_feed
        stages = STAGES[scheme]
        implied = {}
        col_ = (lambda v: v[:, None])  # (m,) -> (m, 1), to concatenate along the streams
        rows = (lambda v, r: xp.broadcast_to(v[:, None], (m, r) + v.shape[1:]))
        full = (lambda v: xp.broadcast_to(xp.asarray(v, dtype=float), (m,)))

        # Feed, E-100 hot side and TK-100: one batch of flashes.
        T_a = xp.concatenate([col_(u.T_feed), s["T_E100h"], col_(s["T_TK100"])], axis=1)
        P_a = xp.concatenate([rows(u.P_feed, n + 1), xp.full((m, 1), pp.P_TK100)], axis=1) * KPA
        fa = self._flash(eos, "a", T_a, P_a, rows(z, n + 2), keep)
        h_feed, h_E100h, tk100 = fa.h[:, 0], fa.h[:, 1:n + 1], _row(fa, n + 1)
        V1, y1 = tk100.V * F, tk100.y
        L_in100 = (1 - tk100.V) * F
        Q_chiller = u.Q_chiller
        if hold:
            implied["Q_chiller"] = Q_chiller = F * (h_E100h[:, -1] - tk100.h) / KW
            implied["v_LCV100"] = L_in100 / pp.F_LCV100_max
        L100 = implied["v_LCV100"] * pp.F_LCV100_max if hold \
            else u.v_LCV100 * pp.F_LCV100_max
        h_chiller_out = h_E100h[:, -1] - Q_chiller * KW / F
        h_V1, _ = self._vapour(eos, s["T_TK100"], pp.P_TK100, y1)

        h_L100 = self._liquid_h(eos, s["T_TK100"], pp.P_TK100, s["x_TK100"])
        P_col = self.column.pressures(col_(s["P_top"] * KPA)) / KPA

        # E-102 hot side, and TK-101 in the conventional scheme.
        if scheme == "conventional":
            F_E102h, z_br, h_br = V1, y1, h_V1
            T_b = xp.concatenate([s["T_E102h"], col_(s["T_TK101"])], axis=1)
            P_b = xp.concatenate([xp.full((m, n), pp.P_TK100),
                                  xp.full((m, 1), pp.P_TK101)], axis=1) * KPA
            fb = self._flash(eos, "b", T_b, P_b, rows(y1, n + 1), keep)
            h_E102h, tk101 = fb.h[:, :n], _row(fb, n)
            F_exp, y_exp = tk101.V * V1, tk101.y
            T_exp, P_exp = s["T_TK101"], pp.P_TK101
            L_in101 = (1 - tk101.V) * V1
            if hold:
                implied["v_LCV101"] = L_in101 / pp.F_LCV101_max
            L101 = (implied["v_LCV101"] if hold else u.v_LCV101) * pp.F_LCV101_max
            L_to26 = L100
        else:
            # The branch: the vapour not expanded and a part of the liquid draw, mixed
            # adiabatically after their valves.
            F_bv, F_bl = V1 / (1 + u.split), u.liquid_split * L100
            F_E102h = F_bv + F_bl
            z_br = (F_bv[:, None] * y1 + F_bl[:, None] * s["x_TK100"]) / F_E102h[:, None]
            h_br = (F_bv * h_V1 + F_bl * h_L100) / F_E102h
            fb = self._flash(eos, "b", s["T_E102h"], xp.full((m, n), pp.P_branch * KPA),
                             rows(z_br, n), keep)
            h_E102h, tk101, L101 = fb.h, None, 0.0
            F_exp, y_exp = V1 - F_bv, y1
            T_exp, P_exp = s["T_TK100"], pp.P_TK100
            L_to26 = L100 - F_bl

        # TE-100: expanded to the pressure of the stage it feeds.
        P_exp_out = P_col[:, stages["expander"]]
        h_in, s_in = self._vapour(eos, T_exp, P_exp, y_exp)
        iso = eos.flash_PS(P_exp_out * KPA, s_in, y_exp,
                           T0=self._start("iso_T", m, xp, default=T_exp - 55.0),
                           K0=self._start("iso_K", m, xp))
        if keep:
            self._keep("iso_T", iso.T)
            self._keep("iso_K", iso.K)
        h_exp_out = h_in - pp.eta_machines * (h_in - iso.h)
        W_expander = F_exp * (h_in - h_exp_out)

        # T-100: the equilibrium first, since the crr recycle is overhead vapour.
        eq = equilibrium_batch(self.column, eos, s["x"], s["x_B"], s["P_top"] * KPA,
                               T0=self._start("column_T", m, xp))
        if keep:
            self._keep("column_T", eq[1][0])
        y0 = eq[2][:, 0]
        feeds = [Feed(stages["expander"], F_exp, y_exp, h_exp_out),
                 Feed(stages["TK100"], L_to26, s["x_TK100"], h_L100)]
        if scheme == "conventional":
            feeds.append(Feed(stages["TK101"], L101, s["x_TK101"],
                              self._liquid_h(eos, s["T_TK101"], pp.P_TK101, s["x_TK101"])))
        elif scheme == "gsp":
            feeds.append(Feed(stages["branch"], F_E102h, z_br, h_E102h[:, -1]))  # JTV-101

        # crr: the branch through E-104's cold side to stage 2.
        if scheme == "crr":
            P_E104c = P_col[:, stages["branch"]] + pp.dP_E104c
            P_rec = s["P_top"] + pp.dP_K102
            fr = self._flash(eos, "r", xp.concatenate([s["T_E104h"], s["T_E104c"]], axis=1),
                             xp.concatenate([rows(P_rec, n), rows(P_E104c, n)], axis=1) * KPA,
                             xp.concatenate([rows(y0, n), rows(z_br, n)], axis=1), keep)
            h_E104h, h_E104c = fr.h[:, :n], fr.h[:, n:]
            feeds.append(Feed(stages["branch"], F_E102h, z_br, h_E104c[:, 0]))

        B = u.v_LCV102 * pp.F_LCV102_max
        dM, dxs, dM_B, dx_B, col = evaluate_batch(
            self.column, eos, s["M"], s["x"], s["M_B"], s["x_B"], s["P_top"] * KPA, feeds,
            Q_R=u.Q_reboiler * KW, B=B, eq=eq)
        V_top, h_top = col.V[:, 0], col.h_V[:, 0]
        if hold:
            B = col.L[:, -1] - col.V[:, -1]
            implied["v_LCV102"] = B / pp.F_LCV102_max
            dM_B = 0.0  # the reboiler composition does not depend on the draw

        # crr: stage 1 mixes the vapour of stage 2 with the recycle leaving E-104; K-102
        # draws the recycle from the mixture.  The recycle has the overhead's composition,
        # which is therefore that of the stage-2 vapour.  K-102 is an ideal gas from the
        # stage-2 temperature, the mixing point's being within a kelvin of it.
        F_rec, H_K102, h_ov = 0.0, 0.0, h_top
        derivs = {}
        if scheme == "crr":
            r = (P_rec / s["P_top"]) ** ((pp.kappa_K102 - 1) / pp.kappa_K102) - 1
            w_K102 = pp.cp_K102 * col.T[:, 0] * r / pp.eta_machines  # kJ/kmol
            if hold:
                # The recycle at ratio f to the residue gas, F_rec = f (V0 - F_rec) with
                # V0 = V_top + F_rec, so F_rec = f V_top.
                implied["W_K102"] = pp.recycle_ratio_design * V_top * w_K102 / KW
            W_K102 = implied["W_K102"] if hold else u.W_K102
            F_rec = W_K102 * KW / w_K102
            H_K102 = F_rec * w_K102
            h_ov = (V_top * h_top + F_rec * h_E104h[:, -1]) / (V_top + F_rec)
            derivs["T_E104h"], derivs["T_E104c"] = _exchanger(
                s["T_E104h"], s["T_E104c"], h_E104h, h_E104c, F_rec, F_E102h,
                h_ov + w_K102, h_E102h[:, -1], pp.UA_E104, pp.C_cell)
        V0 = V_top + F_rec

        # Cold sides: E-102 then E-100, the residue gas's pressure falling through them.
        P_E102c = s["P_top"] - pp.dP_overhead / 2
        P_E100c = s["P_top"] - pp.dP_overhead
        T_c = xp.concatenate([s["T_E102c"], s["T_E100c"]], axis=1)
        P_c = xp.concatenate([rows(P_E102c, n), rows(P_E100c, n)], axis=1) * KPA
        fc = self._flash(eos, "c", T_c, P_c, rows(y0, T_c.shape[1]), keep)
        h_E102c, h_E100c = fc.h[:, :n], fc.h[:, n:2 * n]
        F_cold = V_top
        derivs["T_E100h"], derivs["T_E100c"] = _exchanger(
            s["T_E100h"], s["T_E100c"], h_E100h, h_E100c, F, F_cold, h_feed, h_E102c[:, 0],
            pp.UA_E100, pp.C_cell)
        derivs["T_E102h"], derivs["T_E102c"] = _exchanger(
            s["T_E102h"], s["T_E102c"], h_E102h, h_E102c, F_E102h, F_cold, h_br, h_ov,
            pp.UA_E102, pp.C_cell)

        # Separators.
        derivs["T_TK100"] = F * (h_chiller_out - tk100.h) / pp.C_separator
        derivs["M_TK100"] = L_in100 - L100
        derivs["x_TK100"] = L_in100[:, None] * (tk100.x - s["x_TK100"]) / s["M_TK100"][:, None]
        if scheme == "conventional":
            derivs["T_TK101"] = V1 * (h_E102h[:, -1] - tk101.h) / pp.C_separator
            derivs["M_TK101"] = L_in101 - L101
            derivs["x_TK101"] = (L_in101[:, None] * (tk101.x - s["x_TK101"])
                                 / s["M_TK101"][:, None])

        # Recompression and the overhead pressure.
        T1 = s["T_E100c"][:, 0]
        if hold:
            F_comp = F_cold
            implied["W_recompressor"] = self._recompression_power(W_expander, F_comp, T1,
                                                                  P_E100c) / KW
        W_K101 = implied["W_recompressor"] if hold else u.W_recompressor
        F_comp, P2, T2, T3 = self._recompression(W_expander, W_K101 * KW, T1, P_E100c, keep)
        derivs["P_top"] = (V_top - F_comp) / self.C_P
        if hold:
            derivs["P_top"] = 0.0

        # Flow transmitters, kmol/min; the residue gas is the overhead less the recycle.
        derivs["FT_NGL"] = (B - s["FT_NGL"]) / pp.tau_FT
        if scheme == "crr":
            derivs["FT_recycle"] = (F_rec - s["FT_recycle"]) / pp.tau_FT
            derivs["FT_residue"] = (V_top - s["FT_residue"]) / pp.tau_FT

        dx_state = self.layout.pack(M=dM, x=dxs, M_B=dM_B, x_B=dx_B, **derivs)
        streams = Streams(
            tk100=tk100, V1=V1, L100=full(L100), F_expander=F_exp, y_expander=y_exp,
            T_expander_in=full(T_exp), P_expander_in=full(P_exp), h_expander_out=h_exp_out,
            W_expander=W_expander, column=col, B=full(B), V0=V0, F_comp=F_comp,
            P_booster_out=P2, T_booster_out=T2, T_K101_out=T3, tk101=tk101, L101=full(L101),
            F_branch=full(F_E102h if scheme != "conventional" else 0.0),
            F_recycle=full(F_rec), h_feed=h_feed, h_residue=h_E100c[:, 0],
            Q_chiller=full(Q_chiller * KW), H_K102=full(H_K102), h_overhead=h_ov,
            implied={k: full(v) for k, v in implied.items()})
        return dx_state, streams

    # -- recompression -----------------------------------------------------------------
    #
    # The booster (b) and K-101 (k) as ideal gases, each with its own heat capacity and
    # ratio of heat capacities, at efficiency eta, with r = (P_out / P_in)**a - 1 and
    # a = (kappa - 1) / kappa:
    #
    #     W_booster = F cp_b T1 r_b / eta,     T2 = T1 (1 + r_b / eta)
    #     W_K101    = F cp_k T2 r_k / eta_k,   T3 = T2 (1 + r_k / eta_k)

    def _recompression(self, W_booster, W_K101, T1, P1, keep: bool = True):
        """The residue-gas flow at the given shaft powers (kJ/min), and the pressure and
        temperatures between the machines.  Eliminating F leaves one equation in P2,
        decreasing from +inf at P1 to -W_K101 at P_sales, solved by Newton in ln P2
        inside that bracket.  Each argument (m,), for the members of a batch, which
        iterate together, each holding its value once its step has fallen below the
        tolerance."""
        pp = self.pp
        xp = namespace(T1)
        ab = (pp.kappa_booster - 1) / pp.kappa_booster
        ak = (pp.kappa_K101 - 1) / pp.kappa_K101
        eta, eta_k, Ps = pp.eta_machines, pp.eta_K101, pp.P_sales
        ratio = pp.cp_K101 / pp.cp_booster * eta / eta_k
        log_Ps = float(np.log(Ps))  # a float, which combines with a device array

        def g(lnP2):
            r1 = xp.exp(ab * (lnP2 - xp.log(P1))) - 1
            r2 = xp.exp(ak * (log_Ps - lnP2)) - 1
            return W_booster * ratio * (1 + r1 / eta) * r2 / r1 - W_K101

        lo, hi = xp.log(P1) + 1e-9, log_Ps - 1e-9 + 0.0 * P1
        v = self._start("lnP2", len(T1), xp, default=0.5 * (lo + hi))
        v = xp.minimum(xp.maximum(v, lo), hi)
        done = xp.zeros(v.shape, dtype=bool)
        for _ in range(100):
            gv = g(v)
            lo = xp.where(gv > 0, v, lo)
            hi = xp.where(gv > 0, hi, v)
            new = v - gv / ((g(v + 1e-7) - gv) / 1e-7)
            new = xp.where((lo < new) & (new < hi), new, 0.5 * (lo + hi))
            step_done = xp.abs(new - v) < 1e-14
            v = xp.where(done, v, new)
            done = done | step_done
            if bool(done.all()):
                break
        if keep:
            self._keep("lnP2", v)
        P2 = xp.exp(v)
        r1 = (P2 / P1) ** ab - 1
        F = W_booster * eta / (pp.cp_booster * T1 * r1)
        T2 = T1 * (1 + r1 / eta)
        T3 = T2 * (1 + ((Ps / P2) ** ak - 1) / eta_k)
        return F, P2, T2, T3

    def _recompression_power(self, W_booster, F, T1, P1):
        """The K-101 power (kJ/min) that passes the flow F with the booster at W_booster."""
        pp = self.pp
        ab = (pp.kappa_booster - 1) / pp.kappa_booster
        ak = (pp.kappa_K101 - 1) / pp.kappa_K101
        r1 = W_booster * pp.eta_machines / (F * pp.cp_booster * T1)
        P2 = P1 * (1 + r1) ** (1 / ab)
        T2 = T1 * (1 + r1 / pp.eta_machines)
        return F * pp.cp_K101 * T2 * ((pp.P_sales / P2) ** ak - 1) / pp.eta_K101


@dataclass(frozen=True)
class Row:
    """One stream of a set of flashes: scalars and vectors for one evaluation, with a
    leading axis of m for a batch."""

    T: object
    V: object
    x: object
    y: object
    h: object
    K: object

    def member(self, i: int) -> Row:
        return Row(**{k: v[i] for k, v in vars(self).items()})


def _row(f, i):
    """Stream i of a set of flashes whose arrays are (m, r, ...): (m,) and (m, n_c)."""
    return Row(T=f.T[:, i], V=f.V[:, i], x=f.x[:, i], y=f.y[:, i], h=f.h[:, i], K=f.K[:, i])


def _exchanger(T_h, T_c, h_h, h_c, F_h, F_c, h_h_in, h_c_in, UA, C_cell):
    """Counter-current cells: the hot stream from cell 0 to n - 1, the cold from n - 1 to 0.
    Temperatures and enthalpies (m, n); flows and inlet enthalpies (m,)."""
    xp = namespace(T_h)
    n = T_h.shape[-1]
    Q = UA / n * (T_h - T_c)
    h_h_up = xp.concatenate([h_h_in[:, None], h_h[:, :-1]], axis=1)
    h_c_up = xp.concatenate([h_c[:, 1:], h_c_in[:, None]], axis=1)
    dT_h = (F_h[:, None] * (h_h_up - h_h) - Q) / C_cell
    dT_c = (F_c[:, None] * (h_c_up - h_c) + Q) / C_cell
    return dT_h, dT_c


# ------------------------------------------------------------------------------------
# The design point
# ------------------------------------------------------------------------------------


@dataclass
class Design:
    """The plant at its design steady state, for the closed-loop simulator."""

    pp: NGLParameters
    x: np.ndarray
    u: Inputs
    plant: Plant
    # The right-hand side and the measures accept a batch of states.
    batched: ClassVar[bool] = True

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: NGLParameters) -> np.ndarray:
        return self.plant.evaluate(x, u)[0]


def default_inputs(plant: Plant) -> Inputs:
    """The feed at its stated conditions, and rough duties, powers and openings."""
    pp = plant.pp
    branched = plant.scheme != "conventional"
    return Inputs(F_feed=pp.F_feed, z_feed=feed(plant.components), T_feed=pp.T_feed,
                  P_feed=pp.P_feed, Q_chiller=1400.0,
                  Q_reboiler=1500.0 if plant.basis == "hysys" else 900.0,
                  W_recompressor=8000.0, v_LCV100=0.5,
                  v_LCV101=0.0 if branched else 0.5, v_LCV102=0.5,
                  split=pp.R_split_design if branched else 0.0,
                  W_K102=25.0 if plant.scheme == "crr" else 0.0,
                  liquid_split=pp.liquid_split_design if branched else 0.0)


def initial_state(plant: Plant) -> np.ndarray:
    """A starting point for `settle`: the front end at the published temperatures,
    linear exchanger and column profiles, and the vessels at their design holdups.  Not
    a steady state.  On the hysys basis, the HYSYS case's own profiles."""
    if plant.basis == "hysys":
        from . import hysys_reference
        return hysys_reference.initial_state(plant)
    pp, eos, n = plant.pp, plant.eos, plant.pp.n_cells
    C = 273.15
    z = feed(plant.components)
    tk100 = eos.flash_PT(np.array([pp.T_TK100_design]), np.array([pp.P_TK100 * KPA]),
                         z[None])
    tk101 = eos.flash_PT(np.array([C - 59.6]), np.array([pp.P_TK101 * KPA]), tk100.y)
    top = tk101.x[0]
    # The bottoms: the feed's ethane and heavier, with 2 % methane and no nitrogen.
    names = eos.c.names
    heavy = np.array([0.0 if nm in ("nitrogen", "methane") else zi
                      for nm, zi in zip(names, z)])
    bottom = 0.98 * heavy / heavy.sum()
    bottom[names.index("methane")] = 0.02
    nt = n_trays(pp, plant.scheme)
    f = np.linspace(0, 1, nt)[:, None]
    x_col = (1 - f) * top + f * bottom
    x_col /= x_col.sum(axis=1, keepdims=True)
    blocks = dict(
        T_E100h=np.linspace(pp.T_feed, C - 4.8, n), T_E100c=np.linspace(C - 11.7, C - 45, n),
        T_E102h=np.linspace(pp.T_TK100_design, C - 59.6, n),
        T_E102c=np.linspace(C - 45, C - 100, n),
        T_TK100=pp.T_TK100_design, M_TK100=pp.M_TK100_design, x_TK100=tk100.x[0],
        M=np.full(nt, pp.holdup_weir + (8.0 / pp.weir_coeff) ** (2 / 3)),
        x=x_col, M_B=pp.M_reboiler_design, x_B=bottom, P_top=pp.P_top,
        **transmitters_at_zero(plant))
    if plant.scheme == "conventional":
        blocks.update(T_TK101=C - 59.6, M_TK101=pp.M_TK101_design, x_TK101=tk101.x[0])
    if plant.scheme == "crr":
        # The recycle cooled from the K-102 discharge by a few kelvin, against the branch
        # warming from below the top temperature (the HYSYS case: -105.6 to -107.5 C and
        # -107.5 to -110.4 C).
        blocks.update(T_E104h=np.linspace(C - 105.6, C - 107.5, n),
                      T_E104c=np.linspace(C - 107.5, C - 110.4, n))
    return plant.layout.pack(**blocks)


def transmitters_at_zero(plant: Plant) -> dict:
    """The flow transmitters of a starting state, which `settle` brings to their flows."""
    names = ["FT_NGL"] + (["FT_recycle", "FT_residue"] if plant.scheme == "crr" else [])
    return dict.fromkeys(names, 0.0)


def ethane_recovery(plant: Plant, x: np.ndarray, u: Inputs, st: Streams) -> float:
    """The fraction of the feed's ethane leaving in the NGL."""
    C2 = plant.eos.c.names.index("ethane")
    x_B = plant.layout.unpack(x)["x_B"]
    return float(st.B * x_B[C2] / (u.F_feed * u.z_feed[C2]))


def separator_error(plant: Plant, x: np.ndarray, u: Inputs, st: Streams) -> float:
    """The error of the specification that fixes the chiller duty, signed so that a
    colder separator lowers it: the recovery below its design value, scaled to kelvin
    (about 0.02 of recovery per kelvin of TK-100 at 0.82, the recovery sweep), or TK-100
    above its design temperature."""
    pp = plant.pp
    if pp.separator_spec == "recovery":
        return (pp.recovery_design - ethane_recovery(plant, x, u, st)) / 0.02
    return plant.layout.unpack(x)["T_TK100"] - pp.T_TK100_design


def reboiler_error(plant: Plant, x: np.ndarray, column) -> float:
    """The error of the specification that fixes the reboiler duty, signed so that more
    duty lowers it: the reboiler temperature below its design value, or the methane in the
    NGL above its design value, scaled to kelvin (0.002 mole fraction to the kelvin, about
    the slope of the bottom stages)."""
    pp = plant.pp
    if pp.reboiler_spec == "temperature":
        return pp.T_reboiler_design - column.T[-1]
    x_B = plant.layout.unpack(x)["x_B"]
    return (x_B[plant.eos.c.names.index("methane")] - pp.x_C1_NGL_design) / 0.002


def design_residuals(plant: Plant, x: np.ndarray, u: Inputs) -> np.ndarray:
    """The right-hand side and the specifications of the design, one per design input:

        the ethane recovery at recovery_design, or TK-100 at T_TK100_design, fixes the
                                       chiller duty
        the column top at P_top        fixes the K-101 power
        the NGL at T_reboiler_design, or its methane at x_C1_NGL_design, fixes the
                                       reboiler duty
        each vessel at its design level fixes its liquid valve
        crr: the recycle at recycle_ratio_design times the residue gas fixes the K-102
                                       power
    """
    pp = plant.pp
    d, st = plant.evaluate(x, u)
    s = plant.layout.unpack(x)
    spec = [separator_error(plant, x, u, st), s["P_top"] - pp.P_top,
            reboiler_error(plant, x, st.column), s["M_TK100"] - pp.M_TK100_design]
    if plant.scheme == "conventional":
        spec.append(s["M_TK101"] - pp.M_TK101_design)
    spec.append(s["M_B"] - pp.M_reboiler_design)
    if plant.scheme == "crr":
        spec.append(st.F_recycle - pp.recycle_ratio_design * (st.V0 - st.F_recycle))
    return np.concatenate([d, spec])


# kW per K per min: the rate at which `settle` moves the reboiler duty against the
# reboiler temperature's error.  With the levels held the reboiler cannot run dry, so the
# only concern is to follow the column's own settling, some tens of minutes.
SETTLE_REBOILER_GAIN = 2.0
# 1/min: the rate at which `settle` moves the TK-100 temperature against the recovery's
# error, under the recovery specification; slower than the column, so that the recovery
# it reads is near its settled value.
SETTLE_SEPARATOR_GAIN = 0.02
SETTLE_T_RANGE = (120.0, 450.0)  # K, the temperatures settle evaluates at


def settle(plant: Plant, x0: np.ndarray, u0: Inputs, t_end: float = 600.0,
           tol: float = 1e-4) -> tuple[np.ndarray, Inputs]:
    """Integrate the plant in hold mode from a rough state towards its design point, and
    return the state and the inputs that hold it.  The reboiler duty, the one design
    input hold mode leaves free, is moved by an integral action on the reboiler
    temperature, or the methane in the NGL (`reboiler_error`).  Hold mode keeps the TK-100
    temperature where it starts; under the recovery specification it is moved too, by an
    integral action on the recovery (`separator_error`).  Stops at `t_end`, or once the
    largest derivative is below `tol`."""
    from scipy.integrate import solve_ivp

    nx = len(x0)

    # BDF's Newton iterations try points far outside the physical range (an exchanger
    # cell below absolute zero).  The temperatures are clamped before the evaluation, so
    # such a point returns finite derivatives that fail the iteration's convergence test.
    is_T = np.array([k.startswith("T_") for k in plant.state_names()])
    k_TK100 = plant.layout.slices["T_TK100"].start
    by_recovery = plant.pp.separator_spec == "recovery"

    def f(t, X):
        x = np.where(is_T, np.clip(X[:nx], *SETTLE_T_RANGE), X[:nx])
        u = replace(u0, Q_reboiler=X[nx])
        d, st = plant.evaluate(x, u, hold=True)
        if by_recovery:
            d[k_TK100] = -SETTLE_SEPARATOR_GAIN * separator_error(plant, x, u, st)
        return np.append(d, SETTLE_REBOILER_GAIN * reboiler_error(plant, x, st.column))

    def near(t, X):
        return np.max(np.abs(f(t, X))) - tol
    near.terminal = True

    sol = solve_ivp(f, (0.0, t_end), np.append(x0, u0.Q_reboiler), method="BDF",
                    rtol=1e-6, atol=1e-8, events=near)
    if sol.status == -1:
        raise ValueError(f"settling towards the design failed at t = {sol.t[-1]:.3g} min: "
                         f"{sol.message}")
    x, u = sol.y[:nx, -1], replace(u0, Q_reboiler=sol.y[nx, -1])
    implied = plant.evaluate(x, u, hold=True)[1].implied
    return x, replace(u, **implied)


def solve_design(pp: NGLParameters, scheme: str = "conventional",
                 basis: str = "chebeir2019", tol: float = 1e-10,
                 start: tuple[np.ndarray, Inputs] | None = None) -> Design:
    """The design steady state: Newton (MINPACK's hybrid method) on the states and the
    design inputs together, from `start`, or from `initial_state` brought near the design
    point by `settle`."""
    from scipy.optimize import root

    plant = Plant(pp, scheme, basis)
    if start is None:
        start = settle(plant, initial_state(plant), default_inputs(plant))
    x0, u0 = start
    names = DESIGN_INPUTS[scheme]
    nx = len(x0)

    def unpack(X):
        return X[:nx], replace(u0, **dict(zip(names, X[nx:])))

    def residual(X):
        try:
            return design_residuals(plant, *unpack(X))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            # A flash or bubble point that fails far from the solution: steer away.
            return np.full(len(X), 1e6)

    X0 = np.concatenate([x0, [getattr(u0, k) for k in names]])
    # Warm starts fixed at the start, so that the residual is a function of X alone and
    # the finite-difference Jacobian carries no history; near a dew point it otherwise
    # stalls the iteration (the crr overhead).
    plant.freeze(x0, u0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sol = root(residual, X0, method="hybr", options=dict(maxfev=50 * len(X0), xtol=1e-13))
    worst = float(np.max(np.abs(residual(sol.x))))
    if worst > tol:
        raise ValueError(f"the design solve did not converge: largest residual {worst:.3g} "
                         f"({sol.message})")
    x, u = unpack(sol.x)
    valves = [v for v in names if v.startswith("v_")]
    if not all(0.0 < getattr(u, v) < 1.0 for v in valves):
        raise ValueError("the design needs a valve outside (0, 1): "
                         + ", ".join(f"{v} {getattr(u, v):.3f}" for v in valves))
    plant.freeze(x, u)
    return Design(pp=pp, x=x, u=u, plant=plant)
