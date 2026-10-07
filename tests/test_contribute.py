"""`plantbench contribute`: a scaffolded case added to a copy of the files the pipeline
edits, and removed again by `--revert`."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from plantbench import cases, contribute, datasets
from plantbench.scaffold import new_case

REPO = Path(__file__).resolve().parents[1]
EDITED = ("plantbench/cases/__init__.py", "pyproject.toml", "README.md", "docs/cases/index.md")

OWN_TEST = """\
from pipe_tank.definition import CASE
from pipe_tank.model import TankParameters

import plantbench as pb


def test_the_design_level():
    assert pb.build(CASE).design.x[0] == TankParameters().h_design
"""


@pytest.fixture
def library(tmp_path):
    """A copy of the files the pipeline edits, standing in for a clone."""
    root = tmp_path / "library"
    for path in EDITED:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / path, root / path)
    return root


@pytest.fixture
def package(tmp_path, monkeypatch):
    """`pipe_tank`, scaffolded with a test of its own and a data file, found as an
    installed package declaring it as an entry point."""
    root = new_case("pipe_tank", tmp_path / "pipe_tank")
    (root / "tests" / "test_pipe_tank.py").write_text(OWN_TEST)
    (root / "pipe_tank" / "data").mkdir()
    (root / "pipe_tank" / "data" / "levels.json").write_text("[]\n")
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m.split(".")[0] == "pipe_tank"]:
        monkeypatch.delitem(sys.modules, name)
    ep = EntryPoint(name="pipe_tank", value="pipe_tank.definition:CASE",
                    group=cases.ENTRY_POINT_GROUP)
    monkeypatch.setattr(cases, "_entry_points", lambda: {"pipe_tank": ep})
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_apply_then_revert_leaves_the_library_as_it_was(library, package):
    before = _snapshot(library)
    plan = contribute.plan("pipe_tank", root=library, description="A tank, piped in.")
    record = contribute.apply(plan)
    assert json.loads(record.read_text())["case"] == "pipe_tank"

    case_dir = library / "plantbench" / "cases" / "pipe_tank"
    definition = (case_dir / "definition.py").read_text()
    assert "from .model import" in definition and "from pipe_tank" not in definition
    assert (case_dir / "data" / "levels.json").exists()
    assert not (library / "tests" / "cases" / "pipe_tank" / "test_contract.py").exists()
    own = (library / "tests" / "cases" / "pipe_tank" / "test_pipe_tank.py").read_text()
    assert "from plantbench.cases.pipe_tank.definition import CASE" in own
    assert '"pipe_tank": "plantbench.cases.pipe_tank.definition",' in \
        (library / "plantbench/cases/__init__.py").read_text()
    assert '"plantbench.cases.pipe_tank" = ["data/*.json"]' in \
        (library / "pyproject.toml").read_text()
    index = (library / "docs/cases/index.md").read_text()
    assert "| [`pipe_tank`](pipe_tank.md) | 1 | A tank, piped in. |" in index
    assert index.split("```{toctree}")[1].rstrip().endswith("pipe_tank\n```")
    assert "| `pipe_tank` | 1 | A tank, piped in. |" in (library / "README.md").read_text()
    assert (library / "docs/cases/pipe_tank.md").read_text().startswith("```{include}")

    assert contribute.revert("pipe_tank", root=library) == []
    assert _snapshot(library) == before


def test_the_registry_entry_is_valid_python(library, package):
    contribute.apply(contribute.plan("pipe_tank", root=library))
    text = (library / "plantbench/cases/__init__.py").read_text()
    modules = {}
    exec(text[text.index("_MODULES = {"):text.index("}", text.index("_MODULES = {")) + 1],
         modules)
    assert list(modules["_MODULES"])[-1] == "pipe_tank"


def test_a_dry_run_writes_nothing(library, package):
    before = _snapshot(library)
    plan = contribute.plan("pipe_tank", root=library)
    lines = contribute.describe(plan)
    assert any(line.startswith("copy") for line in lines)
    assert any("insert  README.md" in line for line in lines)
    assert any("summary" in note for note in plan.notes)
    assert _snapshot(library) == before


def test_a_built_in_case_is_refused(library):
    with pytest.raises(contribute.ContributeError, match="already built in"):
        contribute.plan("jacketed_cstr", root=library)


def test_an_existing_target_is_refused(library, package):
    (library / "tests" / "cases" / "pipe_tank").mkdir(parents=True)
    with pytest.raises(contribute.ContributeError, match="exists already"):
        contribute.plan("pipe_tank", root=library)


def test_an_insertion_point_found_twice_is_refused(library, package):
    readme = library / "README.md"
    readme.write_text(readme.read_text() + "\n" + contribute.CASE_TABLE_HEADER + "\n")
    with pytest.raises(contribute.ContributeError, match="found it 2 times"):
        contribute.plan("pipe_tank", root=library)


def test_a_case_failing_its_rules_is_refused(library, package):
    (package / "tests" / "test_pipe_tank.py").unlink()
    with pytest.raises(contribute.ContributeError, match="has tests of its own"):
        contribute.plan("pipe_tank", root=library)


def test_revert_reports_a_changed_line(library, package):
    contribute.apply(contribute.plan("pipe_tank", root=library))
    readme = library / "README.md"
    readme.write_text(readme.read_text().replace("| `pipe_tank` | 1 |", "| `pipe_tank` | one |"))
    problems = contribute.revert("pipe_tank", root=library)
    assert len(problems) == 1 and "is not there" in problems[0]
    assert "| `pipe_tank` | one |" in readme.read_text()
    assert not (library / "plantbench" / "cases" / "pipe_tank").exists()
    assert not (library / ".contribute").exists()


def test_revert_leaves_a_file_where_a_line_is_there_twice(library, package):
    contribute.apply(contribute.plan("pipe_tank", root=library))
    readme = library / "README.md"
    row = "| `pipe_tank` | 1 | " + cases.load_case("pipe_tank").summary + " |"
    readme.write_text(readme.read_text() + row + "\n")
    problems = contribute.revert("pipe_tank", root=library)
    assert len(problems) == 1 and "2 times" in problems[0]
    assert readme.read_text().count(row) == 2
    record = json.loads((library / ".contribute" / "pipe_tank.json").read_text())
    assert list(record["inserted"]) == ["README.md"]


def test_a_module_in_a_subpackage_imports_relative_to_its_depth(library, package):
    sub = package / "pipe_tank" / "tools"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "levels.py").write_text("from pipe_tank.model import TankParameters\n")
    contribute.apply(contribute.plan("pipe_tank", root=library))
    moved = library / "plantbench" / "cases" / "pipe_tank" / "tools" / "levels.py"
    assert moved.read_text() == "from ..model import TankParameters\n"


def test_a_pipe_in_the_description_is_escaped(library, package):
    contribute.apply(contribute.plan("pipe_tank", root=library, description="in | out"))
    assert "| `pipe_tank` | 1 | in \\| out |" in (library / "README.md").read_text()


def test_the_contributed_case_loads_from_the_library(library, package, monkeypatch):
    contribute.apply(contribute.plan("pipe_tank", root=library))
    for name in [m for m in sys.modules if m.split(".")[0] == "pipe_tank"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(library / "plantbench" / "cases"))
    # Imported as a top-level package, its relative imports resolve within it.
    case = importlib.import_module("pipe_tank.definition").CASE
    assert case.id == "pipe_tank"


def test_a_built_in_case_source_reads_as_built_in():
    assert datasets.case_source_text(None) == "case built into plantbench"
