"""The contract every case keeps, and the rules a case added to the library keeps as well.

`CHECKS` are the checks of the interface, one function per check, each taking a `Case`
and raising `ContractError` when the case breaks it.  The library's test suite runs them
on every registered case and on the template; a case developed in its own package runs
them from its own tests,

    @pytest.mark.parametrize("check", contract.CHECKS, ids=lambda c: c.__name__)
    def test_contract(check):
        check(CASE)

or from the command line, `plantbench check <case>`.  A case that fails one is
a defect in the case or in the interface; if the interface is at fault, it is fixed in
core rather than special-cased for the plant.

`library_rules` are the rules of docs/adding-a-case.md that can be checked mechanically,
for a case about to be added to the library: which modules it imports, its case card,
its own tests and its dependencies.  `JUDGMENT` lists those that cannot.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import packages_distributions
from pathlib import Path

import numpy as np

from plantbench import build, cases, run
from plantbench.core import case as cs
from plantbench.core import control as ctl
from plantbench.core.case import Case, Config
from plantbench.core.instruments import Actuator, Instrument

FIXED_POINT_TOL = 1e-9


class ContractError(AssertionError):
    """A case breaks a check of the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


# ------------------------------------------------------------------------------------
# The contract
# ------------------------------------------------------------------------------------


def name_follows_the_convention(case: Case) -> None:
    """The id follows docs/adding-a-case.md, and a built-in case lives in the directory
    of that name."""
    try:
        cases.check_id(case.id)
    except ValueError as exc:
        raise ContractError(str(exc)) from None
    if case.id in cases._MODULES:
        _require(cases._MODULES[case.id] == f"plantbench.cases.{case.id}.definition",
                 f"{case.id} is registered from {cases._MODULES[case.id]}")


def design_is_a_fixed_point_of_every_structure(case: Case) -> None:
    """The right-hand side of the closed loop vanishes at the design point under every
    structure the reference options support; a structure that raises `ValueError` when
    built is one they do not.  At least one structure must be checked."""
    checked, skipped = 0, []
    for name in case.structures:
        try:
            setup = build(case, case.config(name))
        except ValueError as exc:
            skipped.append(f"{name!r} ({exc})")
            continue
        checked += 1
        d = setup.design
        y0 = np.concatenate([d.x, ctl.initial_augmented(setup.structure, d)])
        r = ctl.closed_loop_rhs(0.0, y0, setup.structure, d.u, d.pp, None, d.rhs)
        worst = float(np.max(np.abs(r)))
        _require(worst < FIXED_POINT_TOL,
                 f"structure {name!r}: |rhs| reaches {worst:.3g} at the design point")
    _require(checked > 0, "no structure builds with the reference options: "
             + "; ".join(skipped))


def reference_structure_is_available(case: Case) -> None:
    """The default structure is one the case provides, and the reference plant builds."""
    _require(case.default_structure in case.structures,
             f"default structure {case.default_structure!r} is not among {case.structures}")
    build(case)


def state_names_name_every_state_once(case: Case) -> None:
    d = build(case).design
    names = case.state_names(d)
    _require(len(names) == len(d.x),
             f"{len(names)} state names for {len(d.x)} states")
    repeated = sorted({n for n in names if names.count(n) > 1})
    _require(not repeated, f"state names repeated: {repeated}")


def measurements_are_finite_at_the_design_point(case: Case) -> None:
    d = build(case).design
    bad = [name for name, f in case.measurements(d).items()
           if not np.isfinite(f(d.x, d.u, d.pp))]
    _require(not bad, f"measurements not finite at the design point: {bad}")


def configuration_round_trips(case: Case) -> None:
    """The reference configuration survives conversion to plain data and back."""
    c = case.config()
    _require(case.validate(Config.from_dict(c.to_dict())) == c,
             "the reference configuration does not round-trip through to_dict")


def ideal_instruments_add_nothing(case: Case) -> None:
    """Ideal instruments and valves on every loop leave the closed-loop spectrum as it is."""
    setup = build(case)
    names = [e.name for e in setup.structure if isinstance(e, ctl.Loop)]
    seamed = ctl.with_instruments(setup.structure,
                                  instruments={n: Instrument() for n in names},
                                  actuators={n: Actuator() for n in names})
    _require(ctl.n_augmented(seamed) == ctl.n_augmented(setup.structure),
             "ideal instruments add states")
    a = np.sort_complex(ctl.closed_loop_spectrum(setup.design, setup.structure))
    b = np.sort_complex(ctl.closed_loop_spectrum(setup.design, seamed))
    _require(np.allclose(a, b, rtol=0, atol=1e-12), "ideal instruments move the spectrum")


def a_short_run_stays_at_rest(case: Case) -> None:
    """The reference plant, undisturbed, stays at its design point."""
    tr = run(case, t_end=20.0, n_points=5)
    _require(tr.x.shape == (len(tr.state_names), 5),
             f"trajectory of shape {tr.x.shape} for {len(tr.state_names)} states")
    moved = float(np.max(np.abs(tr.x - tr.x[:, :1])))
    _require(np.allclose(tr.x, tr.x[:, :1], rtol=1e-7, atol=1e-7),
             f"the states move from the design point, by up to {moved:.3g}")
    bad = [k for k, v in tr.y.items() if not np.all(np.isfinite(v))]
    _require(not bad, f"measurements not finite: {bad}")


def generic_disturbances_act_on_the_inputs(case: Case) -> None:
    """The generic `step` changes a scalar input by the fraction asked for."""
    setup = build(case)
    field = next((k for k, v in vars(setup.design.u).items() if np.ndim(v) == 0
                  and v is not None and v != 0), None)
    _require(field is not None, "the inputs have no non-zero scalar field")
    s = build(case, case.config(disturbance={"name": "step", "field": field,
                                             "fraction": 0.01, "t": 1.0}))
    got, want = getattr(s.disturbance(2.0, s.design.u), field), 1.01 * getattr(setup.design.u, field)
    _require(np.isclose(got, want, rtol=1e-12, atol=0),
             f"a 1% step on {field} gives {got!r}, not {want!r}")
    _require(isinstance(cs.GENERIC_DISTURBANCES, dict), "no generic disturbances")


CHECKS = (
    name_follows_the_convention,
    design_is_a_fixed_point_of_every_structure,
    reference_structure_is_available,
    state_names_name_every_state_once,
    measurements_are_finite_at_the_design_point,
    configuration_round_trips,
    ideal_instruments_add_nothing,
    a_short_run_stays_at_rest,
    generic_disturbances_act_on_the_inputs,
)


@dataclass(frozen=True)
class Result:
    check: str
    passed: bool
    message: str = ""


def _outcome(name: str, fn, *args) -> Result:
    try:
        fn(*args)
    except ContractError as exc:
        return Result(name, False, str(exc))
    except Exception as exc:  # noqa: BLE001 -- a check that raises is a check failed
        return Result(name, False, f"{type(exc).__name__}: {exc}")
    return Result(name, True)


def check(case: Case) -> list[Result]:
    """Every check of the contract on one case, each passed or failed with its reason."""
    return [_outcome(fn.__name__, fn, case) for fn in CHECKS]


# ------------------------------------------------------------------------------------
# The library's rules, for a case to be added to it
# ------------------------------------------------------------------------------------

# docs/adding-a-case.md, item 10: the sections of a case card.  A heading may hold more
# than one ("States and inputs").
CARD_SECTIONS = ("provenance", "process", "states", "inputs", "options", "structures",
                 "disturbances", "measurements", "reference results", "verification",
                 "limitations")

# What a case may import from inside the library.
LIBRARY_LAYERS = ("plantbench.core", "plantbench.units", "plantbench.heat")
# What it may import from outside, besides the standard library, without an extra.
LIBRARY_DEPENDENCIES = ("numpy", "scipy")

JUDGMENT = (
    "A case built on published results carries a golden test of its reference "
    "configuration, and its tolerances are argued from the source.",
    "Tunings are chosen for damping, not merely stability, and the damping ratio each "
    "gives is recorded beside it.",
    "Every constant carries its unit, and a value corrected from its source carries the "
    "correction and its evidence.",
    "The case card states the provenance of every published number the case reproduces.",
    "The name describes the process, and no number, version, author or source is in it.",
    "A case presented in a publication of its own gives that publication in a Citation "
    "section of its case card.",
)


def project_dir(path: Path) -> Path | None:
    """The nearest directory at or above `path` holding a pyproject.toml."""
    for p in (path, *path.parents):
        if (p / "pyproject.toml").exists():
            return p
    return None


def _imports(path: Path) -> list[tuple[str, int]]:
    """Absolute module names a file imports, with their line; relative imports are the
    package's own and are left out."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.append((node.module, node.lineno))
    return out


def _extras(project: Path) -> set[str]:
    """Distribution names declared in the optional dependencies of a project."""
    table = tomllib.loads((project / "pyproject.toml").read_text())
    names = set()
    for requirements in table.get("project", {}).get("optional-dependencies", {}).values():
        for r in requirements:
            names.add(_distribution_name(r))
    return names


def _distribution_name(requirement: str) -> str:
    name = requirement
    for stop in "[<>=!~; @(":
        name = name.split(stop)[0]
    return name.strip().lower().replace("_", "-")


def imports_only_what_a_case_may(case: Case) -> None:
    """A case imports core, units and heat from the library, no other case, and nothing
    outside the standard library, NumPy and SciPy that its project does not declare in
    an extra."""
    here = cases.package_dir(case)
    project = project_dir(here)
    declared = _extras(project) if project is not None else set()
    own = here.name
    distributions = packages_distributions()
    problems = []
    for path in sorted(here.rglob("*.py")):
        for name, line in _imports(path):
            top = name.split(".")[0]
            where = f"{path.relative_to(here.parent)}:{line}"
            if top == "plantbench":
                if not any(name == m or name.startswith(m + ".") for m in LIBRARY_LAYERS):
                    problems.append(f"{where} imports {name}")
            elif top in (own, "__future__") or top in sys.stdlib_module_names \
                    or top in LIBRARY_DEPENDENCIES:
                continue
            else:
                dists = {_distribution_name(d) for d in distributions.get(top, [top])}
                if not dists & declared:
                    problems.append(f"{where} imports {name}, which no extra of "
                                    f"{project / 'pyproject.toml' if project else 'the project'} "
                                    f"declares")
    _require(not problems, "; ".join(problems))


def has_a_case_card(case: Case) -> None:
    """A README.md beside the definition, with a heading for each section of a card, and
    a reference under its Citation heading if it has one."""
    card = cases.package_dir(case) / "README.md"
    _require(card.exists(), f"no case card at {card}")
    headings = [line.lstrip("#").strip().lower()
                for line in card.read_text().splitlines() if line.startswith("## ")]
    missing = [s for s in CARD_SECTIONS if not any(s in h for h in headings)]
    _require(not missing, f"{card} has no section on {', '.join(missing)}")
    _require(cases.card_section(card.read_text(), "citation") != "",
             f"{card} has a Citation section with nothing in it")


def has_tests_of_its_own(case: Case) -> None:
    """Tests beyond the contract: conservation, limits, the published values reproduced.

    Looked for in `tests/cases/<id>/` for a built-in case, and in the project's `tests/`
    for one in its own package, in a file other than one holding only the contract."""
    here = cases.package_dir(case)
    project = project_dir(here)
    if case.id in cases._MODULES:
        where = project / "tests" / "cases" / case.id if project else None
    else:
        where = project / "tests" if project else None
    files = [] if where is None or not where.is_dir() else [
        p for p in where.rglob("test_*.py")
        if any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
               and n.name != "test_contract" for n in ast.walk(ast.parse(p.read_text())))]
    _require(bool(files), f"no tests of the case's own under {where}")


RULES = (imports_only_what_a_case_may, has_a_case_card, has_tests_of_its_own)


def library_rules(case: Case) -> list[Result]:
    """The mechanical rules for a case to be added to the library, each passed or failed."""
    return [_outcome(fn.__name__, fn, case) for fn in RULES]
