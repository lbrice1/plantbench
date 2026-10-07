"""A case in a package of its own, added to a clone of the library.

    plantbench contribute my_tank [--description TEXT] [--studies DIR] [--dry-run]
    plantbench contribute my_tank --revert

does the mechanical steps of docs/adding-a-case.md, "Contributing a case from its own
package", in the clone that the running plantbench is imported from: the package moves to
`plantbench/cases/<id>/` with its own imports made relative, its tests to
`tests/cases/<id>/` with their imports rewritten, and the case is registered, its data
files and extras declared, and its page and table rows written.  Nothing is written until
every check has passed and every place an edit goes has been found exactly once.

Each file created and each line inserted is recorded in `.contribute/<id>.json`, and
`--revert` removes them again, which leaves other uncommitted work in the clone alone.
What needs the contributor's judgment or depends on the installer is printed, not done.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import plantbench as pb
from plantbench import cases, contract

MANIFEST_DIR = ".contribute"
CASE_TABLE_HEADER = "| Case | States | Description |"
_SKIP = shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache", "*.pyc")


class ContributeError(Exception):
    """A reason the case cannot be added, found before anything is written."""


@dataclass
class Plan:
    """What `apply` will do, worked out and checked in full beforehand."""

    case_id: str
    root: Path
    copies: list[tuple[Path, Path, str]] = field(default_factory=list)  # source, target, imports to
    tests: Path | None = None  # the package's tests, whose contract-only files are left out
    created: dict[str, str] = field(default_factory=dict)  # path: content
    inserted: dict[str, list[tuple[int, str]]] = field(default_factory=dict)  # path: (before, line)
    notes: list[str] = field(default_factory=list)
    left: list[str] = field(default_factory=list)


def library_root() -> Path:
    """The clone the running plantbench is imported from; refused for an installed copy."""
    root = Path(pb.__file__).resolve().parent.parent
    project = root / "pyproject.toml"
    if not (root / ".git").exists() or not project.exists() \
            or tomllib.loads(project.read_text()).get("project", {}).get("name") != "plantbench":
        raise ContributeError(
            f"plantbench is imported from {root}, which is not a clone of the library; "
            f"contribute from an environment with a clone installed editable")
    return root


# ------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------

def _rewrite_imports(text: str, case_id: str, to: str, where: str, depth: int = 0) -> str:
    """`from <id>[.x] import ...` rewritten to `from <to>[.x] import ...`, where `to` is
    "." for a relative import within the package, from a module `depth` subpackages below
    its top.  `import <id>...` is refused, since the names it binds would change with it."""
    lines = text.splitlines(keepends=True)
    for node in ast.walk(ast.parse(text, filename=where)):
        if isinstance(node, ast.Import) and any(
                a.name == case_id or a.name.startswith(case_id + ".") for a in node.names):
            raise ContributeError(f"{where}:{node.lineno} imports {case_id} with `import`; "
                                  f"write it as `from {case_id}... import ...` first")
        if isinstance(node, ast.ImportFrom) and not node.level and node.module and (
                node.module == case_id or node.module.startswith(case_id + ".")):
            rest = node.module[len(case_id):]
            new = ("." * (depth + 1) + rest.lstrip(".")) if to == "." else to + rest
            i = node.lineno - 1
            line = lines[i]
            old = re.compile(rf"\bfrom\s+{re.escape(node.module)}(?=\s+import\b)")
            line, n = old.subn(f"from {new}", line, count=1)
            if n != 1:
                raise ContributeError(f"{where}:{node.lineno}: cannot rewrite the import")
            lines[i] = line
    return "".join(lines)


def _only_the_contract(path: Path) -> bool:
    tests = [n.name for n in ast.walk(ast.parse(path.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    return tests == ["test_contract"]


# ------------------------------------------------------------------------------------
# Insertion points
# ------------------------------------------------------------------------------------

def _once(lines: list[str], test, what: str, path: str) -> int:
    found = [i for i, line in enumerate(lines) if test(line)]
    if len(found) != 1:
        raise ContributeError(f"{path}: expected {what} once, found it {len(found)} times")
    return found[0]


def _table_end(lines: list[str], header: int) -> int:
    """The index after the last row of the Markdown table whose header is at `header`."""
    i = header + 1
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    return i


def _toml_table_end(lines: list[str], header: int) -> int:
    """The index after the last line of the TOML table whose header is at `header`."""
    i, end = header + 1, header + 1
    while i < len(lines) and not lines[i].startswith("["):
        if lines[i].strip() and not lines[i].lstrip().startswith("#"):
            end = i + 1
        i += 1
    return end


def _read(root: Path, path: str) -> list[str]:
    return (root / path).read_text().splitlines()


def _package_data(package: Path, case_id: str) -> str | None:
    """The package-data line for the package's files other than Python and its card."""
    globs = set()
    for p in sorted(package.rglob("*")):
        rel = p.relative_to(package)
        if not p.is_file() or p.suffix in (".py", ".pyc") or rel == Path("README.md") \
                or "__pycache__" in rel.parts or any(s.endswith(".egg-info") for s in rel.parts):
            continue
        globs.add(str(rel) if p.parent == package else f"{rel.parent.as_posix()}/*{p.suffix}")
    if not globs:
        return None
    return f'"plantbench.cases.{case_id}" = {json.dumps(sorted(globs))}'


# ------------------------------------------------------------------------------------
# The plan
# ------------------------------------------------------------------------------------

def plan(case_id: str, *, description: str | None = None, studies: Path | None = None,
         root: Path | None = None) -> Plan:
    """Check that the case can be added and work out every edit, writing nothing."""
    cases.check_id(case_id)
    try:
        src = cases.source(case_id)
    except KeyError as exc:
        raise ContributeError(str(exc.args[0])) from None
    if src["kind"] != "entry point":
        what = "already built in" if src["kind"] == "built in" else "a session case"
        raise ContributeError(f"{case_id!r} is {what}; contribute moves a case installed "
                              f"from a package of its own")
    root = library_root() if root is None else root
    case = cases.load_case(case_id)
    failed = [r for r in contract.check(case) + contract.library_rules(case) if not r.passed]
    if failed:
        raise ContributeError(f"`plantbench check {case_id} --contribute` fails: "
                              + "; ".join(f"{r.check.replace('_', ' ')}: {r.message}"
                                          for r in failed))
    package = cases.package_dir(case)
    project = contract.project_dir(package)
    if package.name != case_id or project is None:
        raise ContributeError(f"{package} is not laid out as `plantbench new-case` writes a "
                              f"package: <project>/{case_id}/definition.py")

    p = Plan(case_id=case_id, root=root)
    targets = {"package": f"plantbench/cases/{case_id}", "tests": f"tests/cases/{case_id}",
               "page": f"docs/cases/{case_id}.md"}
    if studies is not None:
        targets["studies"] = f"studies/{case_id}"
    for t in targets.values():
        if (root / t).exists():
            raise ContributeError(f"{root / t} exists already")

    # The package, its tests, the studies.
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" not in path.parts:
            _rewrite_imports(path.read_text(), case_id, ".", str(path))  # refusals only
    p.copies.append((package, root / targets["package"], "."))
    tests = project / "tests"
    if tests.is_dir():
        for path in sorted(tests.rglob("*.py")):
            if "__pycache__" not in path.parts:
                _rewrite_imports(path.read_text(), case_id, f"plantbench.cases.{case_id}",
                                 str(path))
        p.copies.append((tests, root / targets["tests"], f"plantbench.cases.{case_id}"))
        p.tests = tests
    if studies is not None:
        if not studies.is_dir():
            raise ContributeError(f"{studies} is not a directory")
        for path in sorted(studies.rglob("*.py")):
            if "__pycache__" not in path.parts:
                _rewrite_imports(path.read_text(), case_id, f"plantbench.cases.{case_id}",
                                 str(path))
        p.copies.append((studies, root / targets["studies"], f"plantbench.cases.{case_id}"))

    # The registry.
    path = "plantbench/cases/__init__.py"
    lines = _read(root, path)
    start = _once(lines, lambda s: s.startswith("_MODULES = {"), "`_MODULES = {`", path)
    close = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    if close is None:
        raise ContributeError(f"{path}: `_MODULES` has no closing brace on a line of its own")
    p.inserted[path] = [(close, f'    "{case_id}": "plantbench.cases.{case_id}.definition",')]

    # pyproject.toml: the data files, and the extras the library does not have.
    path = "pyproject.toml"
    lines = _read(root, path)
    entry = _package_data(package, case_id)
    if entry is not None:
        header = _once(lines, lambda s: s.strip() == "[tool.setuptools.package-data]",
                       "[tool.setuptools.package-data]", path)
        p.inserted.setdefault(path, []).append((_toml_table_end(lines, header), entry))
    theirs = tomllib.loads((project / "pyproject.toml").read_text()) \
        .get("project", {}).get("optional-dependencies", {})
    ours = tomllib.loads((root / path).read_text())["project"].get("optional-dependencies", {})
    new_extras = []
    for name, requirements in theirs.items():
        if name == "test":
            continue
        if name not in ours:
            new_extras.append(f"{name} = {json.dumps(requirements)}")
        elif sorted(ours[name]) != sorted(requirements):
            p.notes.append(f"the extra {name!r} differs from the library's: theirs "
                           f"{requirements}, the library's {ours[name]}; reconcile by hand")
    if new_extras:
        header = _once(lines, lambda s: s.strip() == "[project.optional-dependencies]",
                       "[project.optional-dependencies]", path)
        end = _toml_table_end(lines, header)
        p.inserted.setdefault(path, []).extend((end, line) for line in new_extras)

    # The docs page and the case tables.
    p.created[targets["page"]] = f"```{{include}} ../../plantbench/cases/{case_id}/README.md\n```\n"
    states = len(pb.build(case).design.x)
    text = (description or case.summary).replace("|", "\\|")
    if description is None:
        p.notes.append(f"the table rows describe the case by its summary, {case.summary!r}; "
                       f"pass --description to write another")
    path = "docs/cases/index.md"
    lines = _read(root, path)
    header = _once(lines, lambda s: s.strip() == CASE_TABLE_HEADER, "the case table", path)
    toctree = _once(lines, lambda s: s.strip() == "```{toctree}", "the toctree", path)
    toc_end = next((i for i in range(toctree + 1, len(lines)) if lines[i].startswith("```")),
                   None)
    if toc_end is None:
        raise ContributeError(f"{path}: the toctree is not closed")
    p.inserted[path] = [(_table_end(lines, header),
                         f"| [`{case_id}`]({case_id}.md) | {states} | {text} |"),
                        (toc_end, case_id)]
    path = "README.md"
    lines = _read(root, path)
    header = _once(lines, lambda s: s.strip() == CASE_TABLE_HEADER, "the case table", path)
    p.inserted[path] = [(_table_end(lines, header), f"| `{case_id}` | {states} | {text} |")]

    # What is left to the contributor.
    dist = src.get("distribution", case_id.replace("_", "-"))
    p.left.append(f"uninstall the package now: `pip uninstall -y {dist}`, `uv pip uninstall "
                  f"{dist}`, or `pixi remove --pypi {dist}`; plantbench refuses an installed "
                  f"package that declares a built-in case")
    paths = " ".join(t for k, t in targets.items() if k != "page")
    p.left.append(f"`ruff check --fix {paths}`, for the order of the rewritten imports")
    if new_extras:
        p.left.append("`uv lock` and `pixi install` for the new extras, and whether the "
                      "test environments (the `dev` extra) need them")
    card = package / "README.md"
    for n, line in enumerate(card.read_text().splitlines(), 1):
        if re.search(r"`(tests|studies)/", line):
            p.left.append(f"README.md:{n} of the card names a path that has moved: "
                          f"{line.strip()[:80]}")
    p.left.append(f"`plantbench check {case_id} --contribute` and "
                  f"`pytest tests/cases/{case_id}`")
    p.left += [f"for the pull request: {item}" for item in contract.JUDGMENT]
    return p


def describe(p: Plan) -> list[str]:
    """The edits of a plan, one line each."""
    out = []
    for source, target, to in p.copies:
        how = "imports made relative" if to == "." else f"imports to {to}"
        try:
            shown = os.path.relpath(source)
        except ValueError:  # another drive
            shown = str(source)
        out.append(f"copy    {shown} -> {target.relative_to(p.root)}  ({how})")
    out += [f"create  {path}" for path in p.created]
    for path, edits in p.inserted.items():
        out += [f"insert  {path}:{before + 1}  {line}" for before, line in edits]
    return out


# ------------------------------------------------------------------------------------
# Apply and revert
# ------------------------------------------------------------------------------------

def _manifest(root: Path, case_id: str) -> Path:
    return root / MANIFEST_DIR / f"{case_id}.json"


def apply(p: Plan) -> Path:
    """Make the edits of a plan; returns the record of them, which is written first, so
    that `revert` also undoes a run that stopped part of the way."""
    created = [str(target.relative_to(p.root)) for _, target, _ in p.copies] + list(p.created)
    record = {"case": p.case_id, "created": created,
              "inserted": {path: [line for _, line in edits]
                           for path, edits in p.inserted.items()}}
    out = _manifest(p.root, p.case_id)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    for source, target, to in p.copies:
        shutil.copytree(source, target, ignore=_SKIP)
        for path in sorted(target.rglob("*.py")):
            if source == p.tests and _only_the_contract(path):
                path.unlink()
                continue
            depth = len(path.relative_to(target).parts) - 1
            path.write_text(_rewrite_imports(path.read_text(), p.case_id, to, str(path), depth))
    for path, content in p.created.items():
        (p.root / path).write_text(content)
    for path, edits in p.inserted.items():
        text = (p.root / path).read_text()
        lines = text.splitlines()
        # From the bottom up, and the later of two lines at one place first, so that each
        # index still refers to the file as read and the lines keep their order.
        for _, (before, line) in sorted(enumerate(edits), key=lambda e: (e[1][0], e[0]),
                                        reverse=True):
            lines.insert(before, line)
        (p.root / path).write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    return out


def revert(case_id: str, root: Path | None = None) -> list[str]:
    """Undo `apply` from its record: delete what it created, with any later changes to
    it, and remove each line it inserted.

    An inserted line found twice leaves its file as it is, and the record keeps it.  One
    not found, changed since or never written by a run that stopped, is reported, so that
    a changed line can be removed by hand."""
    root = library_root() if root is None else root
    record_path = _manifest(root, case_id)
    if not record_path.exists():
        raise ContributeError(f"no record of a contribution of {case_id!r} at {record_path}")
    record = json.loads(record_path.read_text())
    problems, kept = [], {}
    for path, inserted in record["inserted"].items():
        text = (root / path).read_text()
        lines = text.splitlines()
        counts = {line: lines.count(line) for line in inserted}
        twice = [line for line, n in counts.items() if n > 1]
        if twice:
            problems += [f"{path}: {line!r} is there {counts[line]} times; the file is left "
                         f"as it is" for line in twice]
            kept[path] = inserted
            continue
        problems += [f"{path}: {line!r} is not there; if it was changed, remove it by hand"
                     for line, n in counts.items() if n == 0]
        lines = [line for line in lines if line not in counts]
        (root / path).write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    for path in record["created"]:
        target = root / path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    if kept:
        record_path.write_text(json.dumps({**record, "created": [], "inserted": kept},
                                          indent=1) + "\n")
    else:
        record_path.unlink()
        if not any(record_path.parent.iterdir()):
            record_path.parent.rmdir()
    return problems
