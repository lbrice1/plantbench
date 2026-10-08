"""Imports run one way: cases may use units, heat and core; nothing uses a case.

`core` knows nothing about any plant, so a new case cannot need a change there that
encodes its own assumptions; `units` and `heat` know nothing about control or cases.
The modules that generate, read and score datasets sit above the cases, and the library
as a whole knows nothing of the studies that use it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "plantbench"

# The modules above the cases: they use the layers, and no layer uses them.
TOP = {"datagen", "datasets", "report", "task", "sensitivity", "contract", "scaffold"}

# What each layer may import from inside plantbench, besides itself.  `backend`, the
# array module a batched evaluation runs on (NumPy or CuPy), is beneath all of them and
# knows nothing of plants or control.
ALLOWED = {
    "backend": set(),
    "core": {"backend"},
    "units": {"backend"},
    "heat": {"backend"},
}


def _plantbench_imports(path: pathlib.Path) -> set[str]:
    """Top-level plantbench subpackages a module imports, resolving relative imports."""
    rel = path.relative_to(PACKAGE.parent).with_suffix("")
    package = list(rel.parts[:-1])
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                names = [".".join(base + ([node.module] if node.module else []))]
            else:
                names = [node.module or ""]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "plantbench" and len(parts) > 1:
                out.add(".".join(parts[1:3]) if parts[1] == "cases" else parts[1])
    return out


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layers_import_only_what_they_may(layer):
    paths = [PACKAGE / f"{layer}.py"] if (PACKAGE / f"{layer}.py").exists() \
        else (PACKAGE / layer).rglob("*.py")
    for path in paths:
        imported = _plantbench_imports(path) - {layer}
        assert imported <= ALLOWED[layer], f"{path.name} imports {sorted(imported)}"


def test_cases_do_not_import_each_other():
    for case_dir in (PACKAGE / "cases").iterdir():
        if not case_dir.is_dir() or case_dir.name.startswith("__"):
            continue
        for path in case_dir.rglob("*.py"):
            others = {m for m in _plantbench_imports(path)
                      if m.startswith("cases.") and m != f"cases.{case_dir.name}"}
            assert not others, f"{case_dir.name}/{path.name} imports {sorted(others)}"


def test_no_case_imports_the_modules_above_it():
    for path in (PACKAGE / "cases").rglob("*.py"):
        above = _plantbench_imports(path) & TOP
        assert not above, f"{path.relative_to(PACKAGE)} imports {sorted(above)}"


def test_the_library_does_not_import_the_studies():
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            assert not any(n == "studies" or n.startswith("studies.") for n in names), \
                f"{path.relative_to(PACKAGE)} imports {names}"
