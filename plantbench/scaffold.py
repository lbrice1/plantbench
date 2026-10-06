"""A new case in a package of its own, started from the template.

    plantbench new-case my_tank [DIR]

writes an installable package under DIR (default `./my_tank`):

    pyproject.toml            declares the case as an entry point, so that once the
                              package is installed plantbench finds it by its id
    my_tank/definition.py     the template's definition, with the id filled in
    my_tank/model.py          the template's model, a gravity-drained tank to replace
    my_tank/README.md         the case card, with the sections a card must have
    tests/test_contract.py    plantbench's contract, run on the case

`pip install -e DIR` then makes `my_tank` available to `plantbench.load_case`, to the
command line and to the workers of a dataset generation.  The layout inside `my_tank/`
is that of a case in the library, so that adding the case to the library is a move of
that directory (docs/adding-a-case.md).
"""

from __future__ import annotations

from pathlib import Path

from plantbench import __version__, cases

_TEMPLATE = Path(cases.__file__).resolve().parent / "_template"

_PYPROJECT = """\
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "{dist}"
version = "0.1.0"
description = "The plantbench case {id}"
readme = "{id}/README.md"
requires-python = ">=3.11"
dependencies = ["plantbench>={version}"]

[project.optional-dependencies]
# Dependencies of the model beyond NumPy and SciPy go in an extra named for what they
# provide; `plantbench check {id} --contribute` refuses an import no extra declares.
test = ["pytest>=8.0"]

# This line is what makes the case visible to plantbench once the package is installed.
[project.entry-points."plantbench.cases"]
{id} = "{id}.definition:CASE"

[tool.setuptools]
packages = ["{id}"]

[tool.setuptools.package-data]
{id} = ["README.md"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

_TEST = '''\
"""plantbench's contract, run on this case: `plantbench.contract.CHECKS`.

The case's own tests, of conservation, analytic limits and the published values it
reproduces, go in other files beside this one.
"""

import pytest

from plantbench import contract
from {id}.definition import CASE


@pytest.mark.parametrize("check", contract.CHECKS, ids=lambda c: c.__name__)
def test_contract(check):
    check(CASE)
'''

_DEFINITION_DOC = '''"""`{id}`, the definition: what plantbench needs to know about the model.

Started from the plantbench template by `plantbench new-case {id}`.  The
package's pyproject.toml declares `CASE` as an entry point, so that plantbench finds it
by its id once the package is installed.
"""'''


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"the template's {name} has changed; the scaffold expects {old!r}")
    return text.replace(old, new)


def _module_doc(text: str) -> str:
    """The module docstring of a template file, quotes included."""
    start = text.index('"""')
    return text[start: text.index('"""', start + 3) + 3]


def new_case(case_id: str, directory: str | Path | None = None) -> Path:
    """Write a package for a new case under `directory` and return it.

    The id must follow the naming convention and must not be a case plantbench already
    finds.  The directory must not exist, or be empty.
    """
    cases.check_id(case_id)
    if case_id in cases.list_cases():
        raise ValueError(f"{case_id!r} is already a case ({cases.source(case_id)['kind']})")
    root = Path(directory) if directory is not None else Path(case_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} exists and is not empty")
    package = root / case_id
    package.mkdir(parents=True)
    (root / "tests").mkdir()

    definition = (_TEMPLATE / "definition.py").read_text()
    definition = definition.replace(_module_doc(definition), _DEFINITION_DOC.format(id=case_id))
    definition = _replace_once(definition, 'id="template"', f'id="{case_id}"', "definition")
    definition = _replace_once(definition, 'title="Gravity-drained tank (template)"',
                               f'title="{case_id} (from the plantbench template)"',
                               "definition")
    model = (_TEMPLATE / "model.py").read_text()
    model = _replace_once(model, "Template case, the model", f"`{case_id}`, the model", "model")
    card = (_TEMPLATE / "README.md").read_text()
    card = _replace_once(card, "# `template`:", f"# `{case_id}`:", "card")

    (package / "__init__.py").write_text("")
    (package / "definition.py").write_text(definition)
    (package / "model.py").write_text(model)
    (package / "README.md").write_text(card)
    (root / "pyproject.toml").write_text(_PYPROJECT.format(
        id=case_id, dist=case_id.replace("_", "-"), version=__version__))
    (root / "tests" / "test_contract.py").write_text(_TEST.format(id=case_id))
    return root
