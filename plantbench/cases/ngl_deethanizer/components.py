"""The component set of the case and its feed.

The component list is that of the HYSYS case "Deethanizer - Tuned" (Peng-Robinson fluid
package), nitrogen to n-heptane; the feed is the case's "Inlet Conditions" spreadsheet,
propane by difference.  Nitrogen is listed at zero, as in HYSYS, so that a feed carrying
it needs no other component set.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from plantbench.units.peng_robinson import Components, PengRobinson

NAMES = ("nitrogen", "methane", "ethane", "propane", "isobutane", "n-butane", "isopentane",
         "n-pentane", "n-hexane", "n-heptane")
LABELS = ("N2", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "nC6", "nC7")
_GIVEN = {"nitrogen": 0.0, "methane": 0.011, "ethane": 0.49, "isobutane": 0.0659,
          "n-butane": 0.0751, "isopentane": 0.0376, "n-pentane": 0.0284, "n-hexane": 0.0284,
          "n-heptane": 0.0284}
FEED = tuple(_GIVEN[n] if n in _GIVEN else 1.0 - sum(_GIVEN.values()) for n in NAMES)


@lru_cache(maxsize=1)
def components() -> Components:
    """The constants of the component set; built once per process, since `thermo` is
    slow to load them."""
    return Components.from_names(NAMES)


@lru_cache(maxsize=1)
def equation_of_state() -> PengRobinson:
    return PengRobinson(components())


def feed() -> np.ndarray:
    return np.array(FEED)
