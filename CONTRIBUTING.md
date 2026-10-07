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

## Branches

The repository keeps two long-lived branches.

| Branch | Holds | Receives |
|---|---|---|
| `main` | the released code; each commit on it is a release, tagged `vX.Y.Z` | release and hotfix branches, by merge commit |
| `develop` | the code of the next release | topic branches, by squash merge |

Work happens on a short-lived **topic branch**, cut from `develop` and named `<type>/<short-name>`, where `<type>` is a commit type (see Commit messages) and `<short-name>` is a few lowercase words joined by hyphens:

```
git switch develop && git pull
git switch -c feat/noisy-analyzer
```

| Prefix | For | Cut from | Merged into |
|---|---|---|---|
| `feat/`, `fix/`, `docs/`, `exp/`, `chore/` | a change of that type | `develop` | `develop` |
| `release/X.Y.Z` | preparing a release (see Releases and versioning) | `develop` | `main`, then `main` into `develop` |
| `hotfix/<short-name>` | a fix to the released code that cannot wait for the next release | `main` | `main`, then `main` into `develop` |

A topic branch carries one concern and is deleted once merged. Keep it current by rebasing on `develop` (`git fetch && git rebase origin/develop`) rather than merging `develop` into it.

Contributors without write access fork `lbrice1/plantbench`, cut the topic branch from the fork's `develop`, and open the pull request against `develop` of `lbrice1/plantbench`.

## Commit messages

`type[scope]: description`, lowercase after the colon, no trailing period. `type` is one of `feat`, `fix`, `docs`, `exp` (a study or analysis run, as opposed to the code that produced it), `chore`. `scope` names the area touched, for example `study`, `library`, `case`, `task`, `sensitivity`, `datagen`.

A topic branch is squash-merged, so the commit that lands on `develop` takes its message from the pull request: the title is the subject line, in the form above, and the description supplies the body. Commits on the branch itself may be less formal. Release and hotfix branches are merged into `main` with a merge commit, so that `main` and `develop` keep a shared history and the merge back into `develop` brings nothing but the release.

## Pull requests

A pull request into `develop` carries one concern. Its title is the commit subject it will be squashed into, and its description states:

- what changes and why, with the issue it closes if there is one;
- how it was verified: the tests run, including the slow selection when `reactor_separator_recycle` is touched, and the documentation build when `docs/` or a docstring changed;
- any number in a results file that moved, by how much, and why the new value is right (see below);
- any change to the public interface listed under Releases and versioning, which decides the next version.

It is merged once continuous integration passes and a maintainer has approved it. A case added to the library also answers, in the description, each rule `plantbench check <case> --contribute` lists as needing a reviewer's judgment.

### Checks before a pull request

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

## Code and prose style

The conventions of the code and the reasons for each are in `docs/philosophy.md` under Conventions, Layers and Generalizing on the second use; a change keeps them. Beyond those:

- ruff checks the code with a line length of 100, and no formatter runs: tables of parameters and states are aligned by hand, and an edit keeps them aligned.
- Docstrings are written in Markdown, because the API reference parses them with MyST: backticks for code, indented blocks for examples. The first line states what the object is or does.
- Every physical constant carries its unit, in its name, its docstring or a comment beside it, and a value corrected from its source carries the correction and its evidence.
- Comments state why the code is as it is, not what it does.
- American spelling throughout, in identifiers, file names, docstrings, documentation and the text written to results files (analyzer, vapor, modeling, normalize). Proper names and cited titles keep their own spelling. Changing a string in a results file means regenerating the file with its script, because the golden tests compare bytes.
- Documentation is written in the third person and the present tense, and states each point once. Headings name their subject.

## Pre-commit hooks

The hooks check whitespace, YAML and TOML, merge markers and file size, lint with ruff (no formatter: the code keeps its hand-aligned tables), strip notebook outputs, and refuse generated data (`data/`, `*.npz`, `*.pkl`). Datasets are regenerated from their specifications, never committed.

## Adding a case

Follow `docs/adding-a-case.md`. A new case passes the contract (`plantbench.contract`) without any change to `plantbench/core` that serves only that case. A case developed in its own package with `plantbench new-case` is added with `plantbench contribute`, as described under "Contributing a case from its own package" there and followed on an example in `docs/tutorial/contributing.md`.

A case that was presented in a publication of its own gives that publication in a `## Citation` section of its case card: the reference as the first paragraph, then its BibTeX. `plantbench describe` prints it and dataset cards list it, and the citation section of `README.md` asks users to cite it beside the library's paper.

A case is named for the process it models, in lowercase words joined by underscores: a flowsheet by its principal units in flow order or its established name in the literature (`reactor_separator_recycle`, `tennessee_eastman`), a single unit by the unit and its defining feature (`jacketed_cstr`). No number, version or source goes in the name, and a published name is not changed. The full convention is in `docs/adding-a-case.md` under "Naming a case", and the contract checks it.

## Releases and versioning

Versions follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`. While the major version is 0, a minor release may break the public interface and a patch release does not.

The public interface is what a user's code, data or paper depends on:

- the Python API exported by `plantbench` and documented in the API reference, and the command line;
- the ids of the cases, and the options, structures, disturbances and measurements each case declares;
- the identity of configurations and specifications: anything that changes a configuration's hash, a `pb-spec` or a `pb-protocol` digest, such as a new option on an existing case, breaks every dataset and protocol reported against it;
- the numbers a frozen case reproduces.

A change to any of these is a breaking change and is said to be one in its pull request.

The version is recorded in `pyproject.toml` and `CITATION.cff`; `plantbench.__version__` and the documentation read it from the installed package. A release is prepared on `release/X.Y.Z`, cut from `develop`:

1. Set the version in `pyproject.toml` and `CITATION.cff`.
2. Regenerate the lockfiles (`uv lock`, `pixi lock`). `uv.lock` records plantbench's own version, and continuous integration installs from it with `--locked`.
3. Run the default and slow selections of the suite, the documentation build, and `plantbench verify` on the identifiers the README reports.
4. Open a pull request into `main`, merged with a merge commit.
5. Tag the merge commit on `main` with an annotated tag and push it:

   ```
   git switch main && git pull
   git tag -a vX.Y.Z -m "plantbench X.Y.Z"
   git push origin vX.Y.Z
   ```

6. Merge `main` back into `develop`.

A hotfix follows steps 1 to 6 on `hotfix/<short-name>`, cut from `main`, with a patch version.

## Issues

Report a bug as an issue on GitHub, with:

- the plantbench version (`plantbench --version`), the commit if running from a clone, Python and the platform;
- how plantbench was installed: uv, pixi or pip;
- the smallest configuration or specification that reproduces the problem, with the command or code that runs it, and its output in full;
- what was expected instead.

A proposed case is opened as an issue before the work starts, naming the process, its source, and the studies it would serve, so that its name and scope can be agreed under the convention of `docs/adding-a-case.md`.

## Layout of the tests

| Path | Contents |
|---|---|
| `tests/core/` | the control layer, instruments, disturbances, specifications, on a toy plant (`tests/_toy.py`) |
| `tests/units/`, `tests/heat/` | the unit models and pinch analysis |
| `tests/cases/test_contract.py` | the contract of `plantbench.contract`, run on every case and the template, and the library rules on the built-in cases |
| `tests/test_extending.py` | cases outside the library: entry points, the decorator, workers, provenance, `new-case`, `check` and the helpers |
| `tests/test_contribute.py` | `plantbench contribute`: a case added to a copy of the files it edits, and removed again |
| `tests/test_tutorial.py` | both tutorial pages as written, the contributing one in a temporary copy of the repository |
| `tests/cases/<case>/` | each case's own verification, and `reactor_separator_recycle`'s golden tests |
| `tests/test_architecture.py` | the direction of imports between layers |
| `tests/test_datagen.py`, `tests/test_examples.py` | dataset generation, and the examples as written |
| `tests/test_datasets.py`, `tests/test_task.py` | reading and comparing datasets; tasks and their scoring |
| `tests/test_sensitivity.py` | Sobol' indices against the closed-form indices of the Ishigami function |
| `tests/studies/` | the study code: the dataset reader, the task of the regimes study, the data mining steps |
