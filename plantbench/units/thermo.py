"""Temperatures for the column, from a vapor-pressure law consistent with the model.

The column model was built on constant relative volatility and constant molar overflow,
and so never needed a temperature.  Heat integration needs little else.  This module
supplies temperatures without disturbing any result the column already produces.

Every species obeys

    ln Psat_j(T) = a_j - lambda / (R T)

with one latent heat lambda for all three, the same value constant molar overflow
already assumes.  Two consequences follow, and both are exact:

  * alpha_j = Psat_j / Psat_B = exp(a_j - a_B) is independent of T and P, so the
    volatilities 6 : 3 : 1 the column uses are reproduced exactly;
  * bubble and dew points have closed forms, because Psat_B carries all the
    temperature dependence:

        bubble:  Psat_B(T) = P / sum_j x_j alpha_j
        dew:     Psat_B(T) = P * sum_j y_j / alpha_j

Column pressure therefore moves every temperature in the column and leaves the
separation unchanged.
"""

from __future__ import annotations

import numpy as np

from plantbench.backend import namespace

from .parameters import R_GAS, ColumnParameters, ThermoParameters


def _lam_over_R(cp: ColumnParameters) -> float:
    return cp.lambda_vap / R_GAS


def a_product(tp: ThermoParameters, cp: ColumnParameters) -> float:
    """Vapor-pressure constant of the product B, fixed by its normal boiling point."""
    return np.log(tp.P_ref) + _lam_over_R(cp) / tp.Tb_product


def a_constants(tp: ThermoParameters, cp: ColumnParameters) -> np.ndarray:
    """Vapor-pressure constants a_j for [I, A, B], implied by the volatilities."""
    return a_product(tp, cp) + np.log(np.asarray(cp.alpha, dtype=float))


def psat(T, tp: ThermoParameters, cp: ColumnParameters) -> np.ndarray:
    """Vapor pressures of [I, A, B] at temperature T, in bar."""
    T = np.asarray(T, dtype=float)
    return np.exp(a_constants(tp, cp) - _lam_over_R(cp) / T[..., None])


def normal_boiling_points(tp: ThermoParameters, cp: ColumnParameters) -> np.ndarray:
    """Temperatures at which each species' vapor pressure reaches P_ref, in K."""
    return _lam_over_R(cp) / (a_constants(tp, cp) - np.log(tp.P_ref))


def bubble_point(x, P: float, tp: ThermoParameters, cp: ColumnParameters):
    """Bubble temperature of liquid x at pressure P (bar).  x may be a stack of rows, on
    NumPy or CuPy."""
    xp = namespace(x)
    x = xp.asarray(x, dtype=float)
    s = x @ xp.asarray(cp.alpha, dtype=float)
    return _lam_over_R(cp) / (a_product(tp, cp) - xp.log(P / s))


def dew_point(y, P: float, tp: ThermoParameters, cp: ColumnParameters):
    """Dew temperature of vapor y at pressure P (bar).  y may be a stack of rows."""
    y = np.asarray(y, dtype=float)
    s = y @ (1.0 / np.asarray(cp.alpha, dtype=float))
    return _lam_over_R(cp) / (a_product(tp, cp) - np.log(P * s))


def bubble_pressure(x, T: float, tp: ThermoParameters, cp: ColumnParameters):
    """Pressure (bar) at which liquid x begins to boil at temperature T."""
    x = np.asarray(x, dtype=float)
    return psat(T, tp, cp) @ x


def column_temperatures(z_column: np.ndarray, P: float, tp: ThermoParameters,
                        cp: ColumnParameters) -> dict:
    """Temperatures through the column at pressure P.

    Trays, the reflux drum and the base are at their liquid bubble points; the vapor
    reaching the condenser is at the dew point of the distillate, since a total
    condenser returns vapor of that composition as liquid.
    """
    from . import column as col

    MD, xD, M, x, MB, xB = col.unpack(z_column, cp)
    return {
        "trays": bubble_point(x, P, tp, cp),
        "drum": float(bubble_point(xD, P, tp, cp)),
        "condenser_dew": float(dew_point(xD, P, tp, cp)),
        "bottoms": float(bubble_point(xB, P, tp, cp)),
    }
