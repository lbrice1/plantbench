"""Fixtures shared by the tests of the unit models."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def double_roots():
    """Cubics with roots a <= a + sep < b, from an exact double root (sep = 0) to roots
    1e-5 apart, and triple roots: the coefficients, a, the largest root, and the error
    allowed in a root.  Every root is above B = 0.  A double root is determined by the
    coefficients to about 1e-8 and is allowed 1e-6; a triple root, to about the cube root
    of the machine epsilon, 6e-6, and is allowed 1e-5."""
    a = np.linspace(0.005, 0.6, 120)
    roots = [np.stack([a, a + sep, a + gap], axis=1)
             for sep in (0.0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-5) for gap in (0.01, 0.1, 0.5)]
    r = np.concatenate(roots + [np.stack([a, a, a], axis=1)])
    c2 = -r.sum(axis=1)
    c1 = r[:, 0] * r[:, 1] + r[:, 0] * r[:, 2] + r[:, 1] * r[:, 2]
    c0 = -r.prod(axis=1)
    tol = np.where(np.arange(len(r)) < len(r) - len(a), 1e-6, 1e-5)
    return c2, c1, c0, r[:, 0], r[:, 2], tol
