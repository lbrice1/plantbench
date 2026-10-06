"""The cases: each a plant model offered as a test problem.

A case is found by its id in one of three places, looked up in this order:

    built in        the cases of this package, listed in `_MODULES`
    entry point     a case in an installed package that declares it in its pyproject.toml,

                        [project.entry-points."plantbench.cases"]
                        my_tank = "my_tank.definition:CASE"

    session         a case made available by `@plantbench.case` or `plantbench.register`
                    in the running interpreter

Built-in cases and entry points are found by every process, including the workers that
generate a dataset and the command line.  A session case exists in the process that
defined it; `datagen` re-imports the module that defined it in each worker.  Cases are
imported only when asked for, so importing the library does not import every model.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Callable

from plantbench.core.case import Case

_MODULES = {
    "reactor_separator_recycle": "plantbench.cases.reactor_separator_recycle.definition",
    "jacketed_cstr": "plantbench.cases.jacketed_cstr.definition",
}
ENTRY_POINT_GROUP = "plantbench.cases"

# docs/adding-a-case.md, "Naming a case": lowercase words joined by single underscores,
# beginning with a letter, not the word `case`, at most 40 characters.
_ID = re.compile(r"(?!case(_|$))[a-z]+(_[a-z0-9]+)*")
ID_MAX = 40


@dataclass
class _Session:
    """A case defined in the running interpreter: built on first use when `factory` is set."""

    module: str
    case: Case | None = None
    factory: Callable[[], Case] | None = None


_SESSION: dict[str, _Session] = {}


def check_id(case_id: str) -> None:
    """Refuse an id that does not follow the naming convention of docs/adding-a-case.md."""
    if not isinstance(case_id, str) or not _ID.fullmatch(case_id) or len(case_id) > ID_MAX:
        raise ValueError(f"{case_id!r} is not a case id: lowercase words joined by single "
                         f"underscores, beginning with a letter, not the word 'case', at "
                         f"most {ID_MAX} characters")


def _entry_points() -> dict[str, EntryPoint]:
    """The cases installed packages declare, by id.  An id declared twice is refused."""
    found: dict[str, EntryPoint] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _MODULES:
            raise ValueError(f"{_describe(ep)} declares {ep.name!r}, a built-in case")
        if ep.name in found and found[ep.name].value != ep.value:
            raise ValueError(f"case {ep.name!r} is declared by two packages: "
                             f"{_describe(found[ep.name])} and {_describe(ep)}")
        found[ep.name] = ep
    return found


def _describe(ep: EntryPoint) -> str:
    dist = ep.dist.name if ep.dist is not None else "an unknown distribution"
    return f"{ep.value} ({dist})"


def list_cases() -> list[str]:
    return sorted({*_MODULES, *_entry_points(), *_SESSION})


def load_case(case_id: str) -> Case:
    if case_id in _MODULES:
        return importlib.import_module(_MODULES[case_id]).CASE
    eps = _entry_points()
    if case_id in eps:
        return _checked(eps[case_id].load(), case_id, _describe(eps[case_id]))
    if case_id in _SESSION:
        entry = _SESSION[case_id]
        if entry.case is None:
            entry.case = _checked(entry.factory(), case_id,
                                  f"{entry.module}.{entry.factory.__name__}")
        return entry.case
    raise KeyError(f"no case {case_id!r}; cases: {list_cases()}")


def _checked(case, case_id: str, source: str) -> Case:
    if not isinstance(case, Case):
        raise TypeError(f"{source} gives a {type(case).__name__}, not a Case")
    if case.id != case_id:
        raise ValueError(f"{source} is registered as {case_id!r} but its id is {case.id!r}")
    return case


def source(case_id: str) -> dict:
    """Where a case comes from: `kind` (built in, entry point or session) and its module,
    with the distribution and its version when the case is installed."""
    if case_id in _MODULES:
        return {"kind": "built in", "module": _MODULES[case_id]}
    eps = _entry_points()
    if case_id in eps:
        ep = eps[case_id]
        out = {"kind": "entry point", "module": ep.module}
        if ep.dist is not None:
            out.update(distribution=ep.dist.name, version=ep.dist.version)
        return out
    if case_id in _SESSION:
        return {"kind": "session", "module": _SESSION[case_id].module}
    raise KeyError(f"no case {case_id!r}; cases: {list_cases()}")


def package_dir(case: Case) -> Path:
    """The directory of the package that defines a case: that of its `make_design`."""
    module = sys.modules.get(case.make_design.__module__) \
        or importlib.import_module(case.make_design.__module__)
    return Path(module.__file__).resolve().parent


def card_section(card: str, name: str) -> str | None:
    """The text under the `## <name>` heading of a case card, or None without one.

    The heading is matched without regard to case; the section runs to the next heading
    of the same or a higher level outside a code block."""
    lines, out, fence = card.splitlines(), None, False
    for line in lines:
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith(("# ", "## ")):
            if out is not None:
                break
            if line[3:].strip().lower() == name.lower() and line.startswith("## "):
                out = []
            continue
        if out is not None:
            out.append(line)
    return None if out is None else "\n".join(out).strip()


def citation(case: Case) -> dict | None:
    """The publication a case was presented in, from the Citation section of its card.

    `reference` is the first paragraph of the section, the reference as it is written to
    be read, and `text` the whole section, BibTeX included.  None when the card has no
    Citation section, or the case has no card; and for a session case, whose module may
    sit beside a README that is not a case card.  A case not registered at all, such as
    one imported to be checked, is read like an installed one."""
    try:
        if source(case.id)["kind"] == "session":
            return None
    except KeyError:
        pass
    card = package_dir(case) / "README.md"
    text = card_section(card.read_text(), "citation") if card.exists() else None
    if not text:
        return None
    return {"reference": " ".join(text.split("\n\n")[0].split()), "text": text}


def session_modules() -> list[str]:
    """The modules that defined the session's cases, for a worker process to import again."""
    return sorted({e.module for e in _SESSION.values()})


def _claim(case_id: str) -> None:
    check_id(case_id)
    if case_id in _MODULES:
        raise ValueError(f"{case_id!r} is a built-in case")
    eps = _entry_points()
    if case_id in eps:
        raise ValueError(f"{case_id!r} is already declared by {_describe(eps[case_id])}")


def register(case: Case) -> None:
    """Make a case defined elsewhere available by id for this session.

    Registering an id again replaces the earlier case, so that a notebook cell or a
    script can be run twice.  The module that calls `register` is the one a worker
    process imports to register the case again.
    """
    _claim(case.id)
    _SESSION[case.id] = _Session(module=sys._getframe(1).f_globals["__name__"], case=case)


def case(case_id: str) -> Callable[[Callable[[], Case]], Callable[[], Case]]:
    """Decorator: make the case a function returns available as `case_id` for this session.

        @plantbench.case("my_tank")
        def my_tank() -> plantbench.Case:
            return plantbench.Case(id="my_tank", ...)

    The function is called on the first `load_case("my_tank")`, not at import.
    """
    _claim(case_id)

    def keep(factory: Callable[[], Case]) -> Callable[[], Case]:
        _SESSION[case_id] = _Session(module=factory.__module__, factory=factory)
        return factory

    return keep
