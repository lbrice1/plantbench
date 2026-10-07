"""The constants of the demethanizer section, with their units and where each comes from.

`[C]` marks a value stated by Chebeir, Salas and Romagnoli (2019); `[H]` one read from the
HYSYS case "Demethanizer - Tuned" (`hysys_reference.json`), a CRR plant on its own feed;
`[D]` one derived from stated values, with the derivation beside it; `[choice]` one this
case chooses because no source gives it, with the reason.  Temperatures in K, pressures in
kPa, flows in kmol/min, enthalpies in kJ/kmol, duties in kJ/min, UA in kJ/(K min), time in
minutes.

Two bases: `chebeir2019`, the conditions the paper states (the defaults of
`NGLParameters`), and `hysys`, the conditions of the HYSYS case (`parameters("hysys")`).
Where the paper is silent, the `chebeir2019` basis takes the HYSYS value if HYSYS has one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from plantbench.units.staged_column import weir_constants

C = 273.15

# Column geometry, used below to derive the hydraulics.
_D_COLUMN = 1.72  # m [C]
_ACTIVE = 0.80  # [choice] active fraction of the tray cross-section, a common design value
_A_TRAY = _ACTIVE * math.pi / 4 * _D_COLUMN**2  # m2
_L_WEIR = 0.73 * _D_COLUMN  # m [choice] weir length of a segmental downcomer, 0.73 D
_H_WEIR = 0.05  # m [choice] weir height, a common design value; [H] 0.05 m
# [D] liquid molar density of the stage liquid, Peng-Robinson at the column conditions:
# 21 kmol/m3 at the methane-rich top, 13 at the bottom; one value for the whole column.
_RHO_STAGE = 15.0  # kmol/m3


def _weir(L_weir: float, A_tray: float) -> tuple[float, float]:
    """(weir_coeff, holdup_weir) of a tray, kmol/min per kmol**1.5 and kmol."""
    return weir_constants(L_weir, A_tray, _H_WEIR, _RHO_STAGE)


_WEIR_COEFF, _HOLDUP_WEIR = _weir(_L_WEIR, _A_TRAY)

# Vessels
_V_SEPARATOR = math.pi / 4 * 1.0**2 * 2.5  # m3 [C] TK-100 and TK-101, 1.0 m by 2.5 m
_V_REBOILER = math.pi / 4 * 1.193**2 * 1.789  # m3 [C] E-103, 1.193 m by 1.789 m


@dataclass(frozen=True)
class NGLParameters:
    # -- feed [C], section 2 ------------------------------------------------------------
    F_feed: float = 4980.0 / 60  # kmol/min
    T_feed: float = 35.0 + C  # K
    P_feed: float = 6015.0  # kPa

    # -- separators ------------------------------------------------------------------------
    # [C] section 2: the liquid of TK-100 is let down from 4672 kPa and that of TK-101
    # from 5199 kPa.  TK-101 is fed by the vapour of TK-100, so the pressure rises along
    # the flow; taken as published, the expander outlet is reproduced to 0.1 K, and
    # exchanged it is not (README.md, Provenance).  Without vapour holdup a
    # separator's pressure enters only its flash.
    P_TK100: float = 4672.0  # kPa
    P_TK101: float = 5199.0  # kPa
    # [choice] 50 % of the vessel volume, at the liquid density of each separator's
    # design liquid (Peng-Robinson: 12.9 and 17.5 kmol/m3).  The level a holdup reads as
    # is M / (M_design / level_design).
    M_TK100_design: float = 0.5 * _V_SEPARATOR * 12.9  # kmol
    M_TK101_design: float = 0.5 * _V_SEPARATOR * 17.5  # kmol
    level_TK100_design: float = 0.5
    # [choice] heat capacity of a separator's metal and contents; sets how fast its
    # temperature follows its inlet (about 2 t of steel).
    C_separator: float = 1000.0  # kJ/K

    # -- exchangers [C] section 3.1 ------------------------------------------------------
    UA_E100: float = 2.038e5 / 60  # kJ/(K min)
    UA_E102: float = 1.961e5 / 60  # kJ/(K min)
    UA_E104: float = 1.576e5 / 60  # kJ/(K min), crr only
    # [choice] each exchanger as this many counter-current cells in series.  The heat
    # transferred grows with the number of cells towards that of true counter-current
    # flow; at the published UA, 10 cells put the E-100 hot outlet at -4.95 C (published
    # -4.8) and the booster discharge at 1265 kPa (published 1265), where 5 cells give
    # -3.9 C and 1280 kPa and 20 give -5.5 C and 1258 kPa.  HYSYS's own discretisation is
    # not stated.
    n_cells: int = 10
    # [choice] heat capacity of the metal and holdup of one side of one cell: about 5 t of
    # aluminium per exchanger, shared between its cells and sides.
    C_cell: float = 225.0  # kJ/K

    # -- column --------------------------------------------------------------------------
    n_stages: int = 30  # [C] section 3.1
    P_top: float = 1010.0  # kPa [C] the TE-100 discharge, which feeds stage 1
    # [D] (1026 - 1010) / 25: JTV-100 discharges at 1026 kPa to stage 26 and JTV-101 at
    # 1015 kPa to stage 8 (1014.5 on this drop).
    dP_stage: float = 0.64  # kPa
    # The stage and reboiler pressures above the top, kPa, when not uniform; overrides
    # dP_stage.  [H] for the hysys basis, where the drop follows the vapour load.
    P_offsets: tuple | None = None
    weir_coeff: float = _WEIR_COEFF  # kmol/min per kmol**1.5
    holdup_weir: float = _HOLDUP_WEIR  # kmol
    # [choice] 50 % of the reboiler volume at 13 kmol/m3, the bottoms' density.
    M_reboiler_design: float = 0.5 * _V_REBOILER * 13.0  # kmol

    # -- overhead circuit and recompression --------------------------------------------
    # [C] section 2: the residue gas leaves E-100 at 821.3 kPa, 188.7 kPa below the top
    # of the column; [choice] the drop is split evenly between E-102 and E-100.
    dP_overhead: float = 1010.0 - 821.3  # kPa
    P_sales: float = 6081.0  # kPa [C]
    eta_machines: float = 0.75  # [C] section 3.1, TE-100, the booster, K-102
    eta_K101: float = 0.75  # [C] section 3.1; [H] 0.74
    # [choice] the vapour volume of the column top and the overhead circuit, which sets how
    # fast the column pressure moves when the vapour made and the vapour compressed differ.
    V_overhead: float = 60.0  # m3
    T_overhead: float = 200.0  # K, the temperature at which that volume's inventory is taken
    # [D] the residue gas as an ideal gas in the machines' flow relations: its ideal-gas
    # heat capacity and ratio of heat capacities, Peng-Robinson on the design overhead
    # (95 % methane), at the mean temperature of each machine: the booster about 5 C, K-101
    # about 105 C, K-102 about -68 C.  A single value at -11.7 C put the K-101 discharge
    # 24 K above the rigorous one; these leave 8 K, the gas's departure from ideality.
    cp_booster: float = 35.1  # kJ/(kmol K)
    kappa_booster: float = 1.310
    cp_K101: float = 39.5
    kappa_K101: float = 1.266
    # K-102 compresses the overhead at about -106 C, [H].
    cp_K102: float = 33.4
    kappa_K102: float = 1.331

    # -- liquid valves [choice] ----------------------------------------------------------
    # Each liquid draw is its valve's opening times the flow at full opening, set to
    # about twice the design flow so that the design opening is near one half.
    F_LCV100_max: float = 6.0  # kmol/min, TK-100 liquid (design 2.3 to 2.8)
    F_LCV101_max: float = 3.5  # kmol/min, TK-101 liquid (design about 1.7)
    F_LCV102_max: float = 10.0  # kmol/min, NGL bottoms

    # -- flow transmitters [choice] ------------------------------------------------------
    # The NGL flow, and under crr the recycle and the residue gas, are read through
    # first-order transmitters.  A flow is algebraic in the valve or the machine power
    # that sets it, and a loop measures before its manipulated variable is written
    # (plantbench.core.control.apply), so ERIC-100 and RFIC-100 need a measurement that
    # is a state.  A tenth of a minute is fast against every loop that reads them.
    tau_FT: float = 0.1  # min

    # -- gsp and crr [H]; Chebeir et al. (2019) give no stream values for these ---------------
    # The TK-100 vapour expanded in TE-100 over that sent to the branch, R (2849 / 1755
    # kmol/h), and the part of the TK-100 liquid draw mixed into the branch (211 / 376).
    # The branch is the vapour and that liquid, mixed at P_branch, through E-102's hot
    # side and JTV-101 to stage 1 (gsp) or through E-104's cold side to stage 2 (crr).
    R_split_design: float = 2849.05 / 1754.66
    liquid_split_design: float = 211.19 / 376.40
    # [choice] the separator pressure: the paper gives no let-down before the mixer;
    # [H] 5553 kPa, 485 below the separator.
    P_branch: float = 4672.0  # kPa
    # CRR: the overhead is split before E-102; K-102 raises the recycle by dP_K102 ("slightly
    # higher", section 2; [H] 1010 to 1036 kPa), E-104 cools it against the branch, and it
    # returns to stage 1.  RFIC-100 holds the recycle at recycle_ratio_design times the
    # residue gas.  The branch crosses E-104's cold side dP_E104c above stage 2.
    dP_K102: float = 26.0  # kPa
    recycle_ratio_design: float = 0.5
    dP_E104c: float = 1202.0 - 1031.0  # kPa

    # -- design specifications -----------------------------------------------------------
    # What fixes the chiller duty: "recovery", the ethane recovery at recovery_design
    # (ERIC-100 cascaded on TIC-100, [C] set-point 0.82), or "temperature", TK-100 at
    # T_TK100_design (TIC-100 alone).  The paper's schemes are compared at a common
    # recovery; at its -29.3 C the branched schemes recover 0.37 and 0.34 (README.md, Provenance).
    separator_spec: str = "recovery"
    recovery_design: float = 0.82
    # K [C] the E-101 outlet the paper states for the conventional scheme; under the
    # recovery specification, only the temperature `settle` holds before Newton.
    T_TK100_design: float = -29.3 + C
    # [C] the NGL leaves at -9.6 C and 1000 kPa; [choice] taken as the reboiler
    # temperature, the let-down to 1000 kPa moving it by less than a kelvin.
    T_reboiler_design: float = -9.6 + C  # K
    # What fixes the reboiler duty: "temperature", the reboiler at T_reboiler_design, or
    # "methane", the methane mole fraction of the NGL at x_C1_NGL_design (XIC-100, [H]).
    # [choice] methane on both bases, so that XIC-100 holds its design point; the NGL
    # temperature is then a result.
    reboiler_spec: str = "methane"
    x_C1_NGL_design: float = 0.01


# [H] the HYSYS column: stage pressures 1010 kPa at the top to 1104 in the reboiler, the
# drop concentrated above stage 8 where the vapour load is eight times that below.
_P_HYSYS = (1010, 1031, 1041, 1051, 1061, 1071, 1080, 1090, 1090, 1091, 1092, 1092, 1093,
            1093, 1094, 1095, 1095, 1096, 1097, 1097, 1098, 1099, 1099, 1100, 1100, 1101,
            1102, 1102, 1103, 1104, 1104)
_WEIR_HYSYS = _weir(1.376, 1.662)  # [H] weir length 1.376 m, active area 1.662 m2
_V_REBOILER_HYSYS = 2.0  # m3 [H]

BASES = {"chebeir2019": "chebeir7", "hysys": "hysys10"}  # basis -> component set


def parameters(basis: str = "chebeir2019") -> NGLParameters:
    """The default parameters of a basis."""
    if basis == "chebeir2019":
        return NGLParameters()
    if basis != "hysys":
        raise ValueError(f"basis must be one of {tuple(BASES)}, not {basis!r}")
    return NGLParameters(
        F_feed=4980.0776 / 60, T_feed=35.0 + C,
        P_feed=6184.0,  # [H] stream 2, after the feed valve VLV-100 (6320 kPa upstream)
        P_TK100=6038.0,  # [H]
        # [H] 40 % level; liquid 15.8 kmol/m3 (Peng-Robinson, stream 8).
        M_TK100_design=0.4 * _V_SEPARATOR * 15.8, level_TK100_design=0.4,
        P_top=1010.0, P_offsets=tuple(p - 1010.0 for p in _P_HYSYS),
        weir_coeff=_WEIR_HYSYS[0], holdup_weir=_WEIR_HYSYS[1],
        # [H] 50 % level; NGL 14.4 kmol/m3 (Peng-Robinson).
        M_reboiler_design=0.5 * _V_REBOILER_HYSYS * 14.4,
        dP_overhead=1010.0 - 805.4,  # [H] stream 28 to stream 5
        eta_K101=0.74,
        P_branch=5553.0,
        F_LCV100_max=12.5, F_LCV102_max=20.0,  # [choice] twice the HYSYS design flows
        # [H] TIC-100, cascaded from ERIC-100 at 0.82; held as a temperature, so that the
        # design is the HYSYS steady state it is tested against (recovery 0.830 here).
        separator_spec="temperature", T_TK100_design=-38.71 + C,
        T_reboiler_design=-19.84 + C,  # [H] the NGL; not a specification on this basis
        x_C1_NGL_design=0.01)
