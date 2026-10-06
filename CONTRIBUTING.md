# Contributing

## Setting up

With uv, pixi or pip (see Installation in `README.md`):

```
uv sync --extra dev && uv run pre-commit install
pixi install && pixi run pre-commit install
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pre-commit install
```

The commands below are written for the pip environment; with uv, replace `.venv/bin/` with `uv run `, and with pixi, with `pixi run `. A change to the dependencies in `pyproject.toml` is committed with `uv.lock` and `pixi.lock` regenerated (`uv lock`, `pixi lock`); continuous integration installs from both lockfiles and fails when either is out of date.

Read `docs/philosophy.md` before changing the library. It holds the conventions the code keeps, and each exists because the alternative caused a problem.

## Before you open a pull request

```
.venv/bin/python -m pytest -q             # about 90 s
.venv/bin/python -m pytest -q -m slow     # about 150 s; run it in the background
.venv/bin/pre-commit run --all-files
```

Continuous integration (`.github/workflows/ci.yml`) runs the default selection of the suite on Python 3.11 and 3.14 with pip, and once each in the environments locked by uv and pixi, ruff, and the documentation build on every pull request. The slow golden test is not part of it; run it yourself when a change touches `reactor_separator_recycle`.

A change to `docs/` or to a docstring is checked by building the documentation, which fails on any warning, as it does on Read the Docs:

```
pixi run docs                                                  # or, with pip:
.venv/bin/pip install -e ".[docs]"
.venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

The default run includes the golden test of `studies/reactor_separator_recycle/results/reference.txt`, and the slow run the golden test of `designs.txt`. Both regenerate the file into a temporary directory and compare bytes. Published numbers are transcribed from these files, so a number that moves silently makes them wrong.

A change that is meant to move a reference number is almost always a new option or a new case instead, because `reactor_separator_recycle` is frozen. Where it is not, regenerate the file with its script, commit it with the change, and say in the pull request which numbers moved, by how much, and why the new value is right:

```
.venv/bin/python -m studies.reactor_separator_recycle.reference
.venv/bin/python -m studies.reactor_separator_recycle.designs    # several minutes; designs.pkl to cache/
```

`designs --cached` redraws the figures from `cache/designs.pkl`. The cache has no staleness check, so delete it after changing the model.

## Pre-commit hooks

The hooks check whitespace, YAML and TOML, merge markers and file size, lint with ruff (no formatter: the code keeps its hand-aligned tables), strip notebook outputs, and refuse generated data (`data/`, `*.npz`, `*.pkl`). Datasets are regenerated from their specifications, never committed.

## Contributing from a fork

- Fork `lbrice1/plantbench` on GitHub and branch from `main` (`feat/<name>`, or `fix/<name>` for a bug fix).
- Open the pull request against `main`.
- A case added from a fork passes `plantbench check <case> --contribute` and follows the steps in `docs/adding-a-case.md`, and a change to a frozen number owes the explanation above.

## Commit messages

`type[scope]: description`, lowercase after the colon, no trailing period. `type` is one of `feat`, `fix`, `docs`, `exp` (a study or analysis run, as opposed to the code that produced it), `chore`. `scope` names the area touched, for example `study`, `library`, `case`, `task`, `sensitivity`, `datagen`.

## Adding a case

Follow `docs/adding-a-case.md`. A new case passes the contract (`plantbench.contract`) without any change to `plantbench/core` that serves only that case. A case developed in its own package with `plantbench new-case` is added by the steps under "Contributing a case from its own package" there.

A case that was presented in a publication of its own gives that publication in a `## Citation` section of its case card: the reference as the first paragraph, then its BibTeX. `plantbench describe` prints it and dataset cards list it, and the citation section of `README.md` asks users to cite it beside the library's paper.

A case is named for the process it models, in lowercase words joined by underscores: a flowsheet by its principal units in flow order or its established name in the literature (`reactor_separator_recycle`, `tennessee_eastman`), a single unit by the unit and its defining feature (`jacketed_cstr`). No number, version or source goes in the name, and a published name is not changed. The full convention is in `docs/adding-a-case.md` under "Naming a case", and the contract checks it.

## Layout of the tests

| Path | Contents |
|---|---|
| `tests/core/` | the control layer, instruments, disturbances, specifications, on a toy plant (`tests/_toy.py`) |
| `tests/units/`, `tests/heat/` | the unit models and pinch analysis |
| `tests/cases/test_contract.py` | the contract of `plantbench.contract`, run on every case and the template, and the library rules on the built-in cases |
| `tests/test_extending.py` | cases outside the library: entry points, the decorator, workers, provenance, `new-case`, `check` and the helpers |
| `tests/cases/<case>/` | each case's own verification, and `reactor_separator_recycle`'s golden tests |
| `tests/test_architecture.py` | the direction of imports between layers |
| `tests/test_datagen.py`, `tests/test_examples.py` | dataset generation, and the examples as written |
| `tests/test_datasets.py`, `tests/test_task.py` | reading and comparing datasets; tasks and their scoring |
| `tests/test_sensitivity.py` | Sobol' indices against the closed-form indices of the Ishigami function |
| `tests/studies/` | the study code: the dataset reader, the task of the regimes study, the data mining steps |
