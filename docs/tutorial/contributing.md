# Contributing a case

This tutorial takes a case from the template to the library: a package of its own, a change to the plant, the checks a case must pass, the move into a clone of the library with `plantbench contribute`, and the checks again from there. Its last section undoes every step, so the whole page can be followed in a clone to learn the procedure and leave the clone as it was. The test suite runs this page as written.

The example is `my_tank`, the template's gravity-drained tank with a second tank upstream of it. [Developing a case](../developing-a-case.md) describes working on a case in its own package at length, and [Adding a case](../adding-a-case.md) the rules a case in the library keeps.

## Setup

Contributing writes into the library, so it needs a clone with the development environment ([CONTRIBUTING.md](https://github.com/lbrice1/plantbench/blob/main/CONTRIBUTING.md)), activated with `source .venv/bin/activate`. Every command below runs from the root of the clone, and the case's package is written beside the clone, at `../my_tank`. `git status` should show nothing before starting, so that the end of the page can show the clone unchanged.

The commands use pip; with uv, `uv pip install` and `uv pip uninstall` take their place. A pixi environment is managed from its manifest, which installing the case would change, so this tutorial is best followed in a uv or pip environment of the clone.

## A package from the template

`new-case` writes an installable package holding the template under the new id:

```console
$ plantbench new-case my_tank ../my_tank
case my_tank written to ../my_tank
...
```

Installing it makes the case available to plantbench by its id, as an entry point of the package:

```console
$ pip install -e ../my_tank
$ plantbench list
```

```
jacketed_cstr              Non-isothermal CSTR with a cooling jacket  [built in]
my_tank                    my_tank (from the plantbench template)  [entry point (my-tank)]
ngl_deethanizer            Deethanizer of an NGL fractionation train  [built in]
ngl_demethanizer           Demethanizer section of a cryogenic NGL recovery plant  [built in]
reactor_separator_recycle  Reactor-separator-recycle plant with heat integration  [built in]
```

## Changing the plant

The model is in `../my_tank/my_tank/model.py`. Here the feed enters an upper tank, which drains freely into the template's tank:

```{code-block} python
:caption: ../my_tank/my_tank/model.py

"""`my_tank`, the model: two gravity-drained tanks in series.

The feed enters the upper tank, which drains freely into the lower one; the lower tank
drains through a valve.

    A_up dh_up/dt = F_in - Cv_up * sqrt(h_up)
    A    dh/dt    = Cv_up * sqrt(h_up) - Cv * valve * sqrt(h)

State: [h, h_up].  Inputs: `F_in` m3/min, `valve` 0-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plantbench.core.casekit import StateLayout

# The lower tank keeps the first place, so that the definition's measurements, which read
# `h` by name and the outflow from x[0], hold as the template wrote them.
LAYOUT = StateLayout([("h", ()), ("h_up", ())])


@dataclass(frozen=True)
class TankParameters:
    """Every constant of the model, with its unit; nothing is read from inside a function."""

    A: float = 2.0  # m2, cross-section of the lower tank
    Cv: float = 1.0  # m3/min per m**0.5, outlet valve fully open
    A_up: float = 1.0  # m2, cross-section of the upper tank
    Cv_up: float = 1.0  # m3/min per m**0.5, the upper tank's fixed outlet
    h_design: float = 1.5  # m, the level the lower tank is designed to run at
    F_design: float = 1.0  # m3/min, the design throughput


@dataclass
class Inputs:
    F_in: float  # m3/min, inflow
    valve: float  # outlet valve opening, 0-1


@dataclass
class Design:
    """The plant at its design steady state: what the closed-loop simulator needs."""

    pp: TankParameters
    x: np.ndarray
    u: Inputs

    def rhs(self, t: float, x: np.ndarray, u: Inputs, pp: TankParameters) -> np.ndarray:
        h, h_up = max(x[0], 0.0), max(x[1], 0.0)
        between = pp.Cv_up * np.sqrt(h_up)
        return np.array([(between - pp.Cv * u.valve * np.sqrt(h)) / pp.A,
                         (u.F_in - between) / pp.A_up])


def solve_design(pp: TankParameters) -> Design:
    """The upper level that passes the design throughput, and the valve opening that
    holds the lower tank at its design level."""
    h_up = (pp.F_design / pp.Cv_up) ** 2
    valve = pp.F_design / (pp.Cv * np.sqrt(pp.h_design))
    if not 0.0 < valve <= 1.0:
        raise ValueError(f"the design needs a valve opening of {valve:.3f}, outside (0, 1]")
    return Design(pp=pp, x=np.array([pp.h_design, h_up]),
                  u=Inputs(F_in=pp.F_design, valve=valve))
```

The structures and measurements in `definition.py` hold as they are: the level loop and both measurements read the lower tank, which keeps the first place in the state vector, and the state names come from `LAYOUT`. What the definition says about the plant does change. Its title and summary are what `plantbench list` and `describe` print, and the summary becomes the case's row in the case tables of the library:

```
    title="Two gravity-drained tanks in series",
    summary="Two gravity-drained tanks in series, the lower one under level control.",
```

The case card, `../my_tank/my_tank/README.md`, describes the plant in the same way, so its title, its Process section and its States table follow the model:

```
| `h` | m | level of the lower tank |
| `h_up` | m | level of the upper tank |
```

These three edits are to text, and the checks below pass with or without them.

## Checking the case

`check` runs the contract every case keeps, and reports how far the design point is from a steady state:

```console
$ plantbench check my_tank
```

```
pass  name follows the convention
pass  design is a fixed point of every structure
...
open-loop |rhs| at the design point: 0; largest at h (+0), h_up (+0)

9 passed, 0 failed
```

`--contribute` adds the rules for a case in the library. One of them fails, since the template has no test of the case's own:

```
$ plantbench check my_tank --contribute
...
FAIL  has tests of its own
      no tests of the case's own under ../my_tank/tests
...
11 passed, 1 failed
```

A test of the case's own checks something the contract cannot know, here that the upper level of the design passes the design throughput through its outlet:

```{code-block} python
:caption: ../my_tank/tests/test_my_tank.py

"""The tanks' own tests: the design against the outlet law of the upper tank."""

import pytest

from my_tank.model import TankParameters, solve_design


def test_the_upper_level_passes_the_design_throughput():
    pp = TankParameters()
    h_up = solve_design(pp).x[1]
    assert pp.Cv_up * h_up**0.5 == pytest.approx(pp.F_design)
```

```console
$ pytest ../my_tank -q
$ plantbench check my_tank --contribute
```

```
...
pass  has tests of its own

also required, and not checked here (docs/adding-a-case.md):
  - ...

12 passed, 0 failed
```

## Adding the case to the library

`contribute` moves the package into the clone. `--dry-run` shows each edit and writes nothing:

```console
$ plantbench contribute my_tank --dry-run
```

```
would add my_tank to /path/to/plantbench:

  copy    ../my_tank/my_tank -> plantbench/cases/my_tank  (imports made relative)
  copy    ../my_tank/tests -> tests/cases/my_tank  (imports to plantbench.cases.my_tank)
  create  docs/cases/my_tank.md
  insert  plantbench/cases/__init__.py:37      "my_tank": "plantbench.cases.my_tank.definition",
  insert  docs/cases/index.md:11  | [`my_tank`](my_tank.md) | 2 | Two gravity-drained tanks in series, the lower one under level control. |
  insert  docs/cases/index.md:21  my_tank
  insert  README.md:34  | `my_tank` | 2 | Two gravity-drained tanks in series, the lower one under level control. |

note: the table rows describe the case by its summary, 'Two gravity-drained tanks in series, the lower one under level control.'; pass --description to write another
```

The package's modules import each other relatively already, as the template writes them, and its tests import `my_tank`, which becomes `plantbench.cases.my_tank`. `test_contract.py` is left behind, since the library runs the contract on every case it holds. The state count in the rows, 2, is read from the design. The same command without `--dry-run` makes the edits, records them in `.contribute/my_tank.json`, and lists what is left to do:

```console
$ plantbench contribute my_tank
```

```
adding my_tank to /path/to/plantbench:
...
recorded in .contribute/my_tank.json; `plantbench contribute my_tank --revert` undoes it.

left to you:
  - uninstall the package now: `pip uninstall -y my-tank`, `uv pip uninstall my-tank`, or `pixi remove --pypi my-tank`; plantbench refuses an installed package that declares a built-in case
  - `ruff check --fix plantbench/cases/my_tank tests/cases/my_tank`, for the order of the rewritten imports
  - `plantbench check my_tank --contribute` and `pytest tests/cases/my_tank`
  - for the pull request: A case built on published results carries a golden test of its reference configuration, and its tolerances are argued from the source.
  - for the pull request: ...
```

A case with data files beside its modules, extras of its own, or studies (`--studies DIR`) has more edits: a `package-data` entry and the extras in `pyproject.toml`, and the studies under `studies/my_tank/`. Lines of the case card that name the package's `tests/` or `studies/` are listed for the contributor to update.

The package is uninstalled first, since plantbench refuses an installed package that declares a built-in case:

```console
$ pip uninstall -y my-tank
$ ruff check --fix plantbench/cases/my_tank tests/cases/my_tank
$ plantbench list
```

```
jacketed_cstr              Non-isothermal CSTR with a cooling jacket  [built in]
my_tank                    Two gravity-drained tanks in series  [built in]
ngl_deethanizer            Deethanizer of an NGL fractionation train  [built in]
ngl_demethanizer           Demethanizer section of a cryogenic NGL recovery plant  [built in]
reactor_separator_recycle  Reactor-separator-recycle plant with heat integration  [built in]
```

The case is now part of the library: its tests run from `tests/cases/my_tank/`, the library's contract suite includes it, and `check --contribute` applies the rules to it where it now is.

```console
$ pytest tests/cases/my_tank -q
$ plantbench check my_tank --contribute
$ plantbench describe my_tank
$ git status --short
```

```
 M README.md
 M docs/cases/index.md
 M plantbench/cases/__init__.py
?? docs/cases/my_tank.md
?? plantbench/cases/my_tank/
?? tests/cases/my_tank/
```

These are the files a pull request adding the case would hold, beside the verification of [Adding a case](../adding-a-case.md#verification) and the answers to the rules `check --contribute` lists for a reviewer.

## Cleaning up

`--revert` deletes what `contribute` created and removes the lines it inserted, from its record, and leaves anything else in the clone as it is. The package was uninstalled above, so what remains is its directory:

```console
$ plantbench contribute my_tank --revert
$ rm -rf ../my_tank
$ git status --short
```

`git status` prints nothing: the clone is as it was before the tutorial.

## A case of your own

For a case of your own, the same steps run on a branch of a fork of the library, without the last section: [Developing a case](../developing-a-case.md) for the work in the case's own package, and [Adding a case](../adding-a-case.md#contributing-a-case-from-its-own-package) for the pull request.
