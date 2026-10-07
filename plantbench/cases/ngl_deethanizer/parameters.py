"""The parameters of the deethanizer.

Sources are marked [H], the HYSYS case "Deethanizer - Tuned" (hysys_reference.json and
its reports), [D], derived here from [H] values, and [choice], a value the case chooses
where HYSYS gives none or the model needs one it does not have.  Units are in each
comment; flows kmol/min, pressures kPa, temperatures K, duties kW.
"""

from __future__ import annotations

from dataclasses import dataclass

from plantbench.units.staged_column import weir_constants

C = 273.15

# Trays [H]: diameter 2.798 m, active area 4.397 m2, weir length 2.238 m, weir height
# 0.05 m; 0.307 m3 of liquid each at the design.
_A_TRAY = 4.39729633  # m2
_L_WEIR = 2.238  # m
_H_WEIR = 0.05  # m
# [D] liquid molar density of the stage liquid, Peng-Robinson on the HYSYS profile:
# 13.3 kmol/m3 on stage 1, 11.3 on the feed stage, 8.4 on stage 30; one value for the
# column.  With it the design holdups are within 5 % of the 0.307 m3 HYSYS reports.
_RHO_STAGE = 11.0  # kmol/m3
_WEIR_COEFF, _HOLDUP_WEIR = weir_constants(_L_WEIR, _A_TRAY, _H_WEIR, _RHO_STAGE)

# Vessels [H]: the condenser drum and the reboiler, 1.193 m by 1.789 m, 2 m3 each, at 45 %
# level.  [choice] a level reads as the liquid's fraction of the volume; HYSYS's vessels
# are horizontal cylinders, and its 45 % holds 0.873 m3, 44 % of the volume.  [D] liquid densities, Peng-Robinson: the distillate at its bubble point,
# 13.6 kmol/m3, and the LPG at 100 C and 2475 kPa, 7.8 kmol/m3.
_V_VESSEL = 2.0  # m3
_RHO_DRUM = 13.6  # kmol/m3
_RHO_REBOILER = 7.8  # kmol/m3

# [H] stage pressures, kPa, stages 1 to 30 and the reboiler, from the tray-by-tray table.
# The reboiler is taken at the stage 30 pressure: the report gives the bottoms at 2472
# and the boil-up at 2470 kPa, both below stage 30, which the integer rounding of the
# table cannot account for.
_P_STAGES = (2453, 2454, 2455, 2455, 2456, 2456, 2457, 2457, 2458, 2458, 2459, 2459, 2460,
             2460, 2461, 2462, 2462, 2463, 2463, 2464, 2465, 2465, 2466, 2466, 2467, 2468,
             2468, 2469, 2470, 2475, 2475)


@dataclass(frozen=True)
class DeethanizerParameters:
    # -- feed [H] -----------------------------------------------------------------------
    F_feed: float = 625.0 / 60  # kmol/min, FIC-100 set-point
    T_feed: float = 15.0 + C  # K
    # kPa, downstream of VLV-100 (2548 kPa upstream), which FIC-100 moves to hold the
    # flow; the flow is an input, so the valve is not modelled.
    P_feed: float = 2500.0

    # -- feed heater E-100 ---------------------------------------------------------------
    dP_E100: float = 5.0  # kPa [H]
    # [choice] heat capacity of the heater's contents and metal: [H] 0.1 m3 of feed at
    # 12 kmol/m3 and 140 kJ/(kmol K), with as much again for the tubes.  Sets how fast
    # the outlet follows the duty: about a quarter of a minute at the design flow, whose
    # heat capacity rate is 1320 kJ/(K min).
    C_E100: float = 340.0  # kJ/K
    T_E100_design: float = 33.0 + C  # K [H] TIC-102 set-point

    # -- column --------------------------------------------------------------------------
    n_stages: int = 30  # [H]
    feed_stage: int = 15  # [H] 1-based, counted from the top
    P_top: float = 2453.0  # kPa [H] PIC-100 set-point, the vapour to the condenser
    P_offsets: tuple = tuple(p - 2453.0 for p in _P_STAGES)  # kPa above stage 1
    weir_coeff: float = _WEIR_COEFF  # kmol/min per kmol**1.5
    holdup_weir: float = _HOLDUP_WEIR  # kmol

    # -- condenser -----------------------------------------------------------------------
    # [choice] the vapour volume whose inventory carries the top pressure: the vapour
    # space of the column, 30 trays of 3.38 m3 less their liquid [H], and of the drum
    # above its liquid.  Taken as an ideal gas at T_overhead, the column's mean.
    V_overhead: float = 30 * (3.381 - 0.307) + 0.55 * _V_VESSEL  # m3
    T_overhead: float = 30.0 + C  # K
    M_drum_full: float = _V_VESSEL * _RHO_DRUM  # kmol, the drum at 100 % level
    level_drum_design: float = 0.45  # [H] LIC-100 set-point

    # -- reboiler ------------------------------------------------------------------------
    M_reboiler_full: float = _V_VESSEL * _RHO_REBOILER  # kmol
    level_reboiler_design: float = 0.45  # [H] LIC-101 set-point

    # -- product valves [H] --------------------------------------------------------------
    # Each draw is its valve's opening times the flow at full opening, linear as the
    # HYSYS valves are, with the full-opening flow chosen so that the design opening is
    # HYSYS's: VLV-101 at 44.56 % passing 312.7 kmol/h, VLV-102 at 40.63 % passing
    # 312.3 kmol/h.  The let-down pressures (2010 and 1790 kPa) are far enough below the
    # column that its pressure barely moves the flow, which is not modelled.
    F_VLV101_max: float = 312.7 / 60 / 0.4456  # kmol/min
    F_VLV102_max: float = 312.3 / 60 / 0.4063  # kmol/min
    P_distillate: float = 2010.0  # kPa, downstream of VLV-101
    P_LPG: float = 1790.0  # kPa, downstream of VLV-102

    # -- design specifications [H], the HYSYS set-points ---------------------------------
    T_stage4_design: float = 9.536 + C  # K, TIC-101
    x_C2_LPG_design: float = 0.02  # XIC-100


def parameters() -> DeethanizerParameters:
    return DeethanizerParameters()
