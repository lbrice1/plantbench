"""The criteria of the comparisons with the tuned HYSYS cases, one per kind of quantity
(`test_hysys_inspection.py` of ngl_deethanizer and ngl_demethanizer).

A quantity outside its criterion is a named exception: it is asserted outside the
criterion and within a tolerance of its own, so that a change that closes or widens it is
seen.  The case READMEs give the reason for each.
"""

PRODUCT = 0.01  # relative: product flows and compositions
INTERNAL = 0.03  # relative: reflux, boil-up, vapor to the condenser, internal flows, ratios
ENERGY = 0.05  # relative: duties and machine powers
TEMPERATURE = 1.0  # K, absolute


def relative(value: float, reference: float) -> float:
    return abs(value / reference - 1.0)
