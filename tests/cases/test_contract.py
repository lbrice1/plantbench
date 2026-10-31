"""The contract every case keeps, checked for every registered case.

The checks are `plantbench.contract.CHECKS`, shipped with the library so that a case
developed in its own package runs the same ones.  A case that fails here is a defect in
the case or in the interface.  If the interface is at fault, fix it in core rather than
special-casing the plant.
"""

from __future__ import annotations

import pytest

import plantbench as pb
from plantbench import contract

# The template is not registered, but it is checked like a case so that copying it
# always starts from something that works.
CASES = [*pb.list_cases(), "template"]


@pytest.fixture(scope="module", params=CASES)
def case(request):
    if request.param == "template":
        from plantbench.cases.template import CASE
        return CASE
    return pb.load_case(request.param)


@pytest.mark.parametrize("check", contract.CHECKS, ids=lambda c: c.__name__)
def test_contract(case, check):
    check(case)


@pytest.mark.parametrize("case_id", sorted(pb.cases._MODULES))
def test_built_in_cases_keep_the_library_rules(case_id):
    results = contract.library_rules(pb.load_case(case_id))
    assert all(r.passed for r in results), [r for r in results if not r.passed]
