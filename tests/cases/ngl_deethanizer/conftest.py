from __future__ import annotations

import pytest

from plantbench.cases.ngl_deethanizer.definition import _design
from plantbench.cases.ngl_deethanizer.parameters import parameters


@pytest.fixture(scope="session")
def design():
    return _design(parameters())
