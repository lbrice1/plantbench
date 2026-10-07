"""The component sets of the case and the published feed.

`chebeir7` is the feed of Chebeir, Salas and Romagnoli (2019), Table 1: seven entries
summing to 1.000, with the butanes, pentanes and hexanes each given as a lump.  Each lump
is represented here by its normal isomer (n-butane, n-pentane, n-hexane); the source gives
no isomer split, and the choice is this case's, not the paper's.

`hysys10` is the component list of the HYSYS case "Demethanizer - Tuned" (Peng-Robinson
fluid package; triethylene glycol and water, at zero, left out), with its feed, which is
not the paper's: 86 % methane and 9.5 % ethane, against 93 % and 3 %.  Carbon dioxide is
listed at zero, as in HYSYS, so that a feed carrying it needs no other component set.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from plantbench.units.peng_robinson import Components, PengRobinson

COMPONENT_SETS = {
    "chebeir7": ("nitrogen", "methane", "ethane", "propane", "n-butane", "n-pentane",
                 "n-hexane"),
    "hysys10": ("nitrogen", "carbon dioxide", "methane", "ethane", "propane", "isobutane",
                "n-butane", "isopentane", "n-pentane", "n-hexane"),
}
LABELS = {
    "chebeir7": ("N2", "C1", "C2", "C3", "nC4", "nC5", "nC6"),
    "hysys10": ("N2", "CO2", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "nC6"),
}
# Chebeir et al. (2019), Table 1, mole fractions in the order of COMPONENT_SETS.
FEED = {
    "chebeir7": (0.010, 0.930, 0.030, 0.015, 0.009, 0.003, 0.003),
    # The HYSYS spreadsheet "Inlet Conditions", methane by difference.
    "hysys10": (0.0113, 0.0, 0.8598, 0.0949, 0.0170, 0.0051, 0.0051, 0.0017, 0.0017, 0.0034),
}


@lru_cache(maxsize=4)
def components(name: str) -> Components:
    """The constants of a component set; built once per process, since `thermo` is slow
    to load them."""
    if name not in COMPONENT_SETS:
        raise ValueError(f"component set {name!r} is not available; sets: "
                         f"{sorted(COMPONENT_SETS)}")
    return Components.from_names(COMPONENT_SETS[name])


@lru_cache(maxsize=4)
def equation_of_state(name: str) -> PengRobinson:
    return PengRobinson(components(name))


def feed(name: str) -> np.ndarray:
    return np.array(FEED[name])
