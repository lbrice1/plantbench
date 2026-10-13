# Developing a case

A case does not have to be part of the library to use it. A plant developed in a package of its own is found by its id like a built-in case, and runs, generates datasets, carries tasks and is analyzed exactly as one. Adding it to the library later is a move of one directory ([Adding a case](adding-a-case.md)).

The workflow, from the environment plantbench is installed in ([Installation](index.md#installation)):

::::{tab-set}
:sync-group: installer

:::{tab-item} uv
:sync: uv

```
uv run plantbench new-case my_tank ~/work/my_tank      # a package, from the template
uv pip install -e ~/work/my_tank'[test]'               # plantbench now finds my_tank
uv run pytest ~/work/my_tank                           # the contract, on the case
uv run plantbench check my_tank                        # the same, with a residual report
uv run plantbench generate spec.toml --workers 8       # a dataset; the spec names my_tank
uv run plantbench check my_tank --contribute           # before proposing it to the library
```

`uv run` keeps the case installed, and so does `uv sync --inexact`; a plain `uv sync` restores the locked environment exactly and removes it, after which `uv pip install -e` installs it again.
:::

:::{tab-item} pixi
:sync: pixi

A pixi workspace of your own holds plantbench and the case, so the library's manifest is left unchanged:

```
pixi init ~/work/my_work && cd ~/work/my_work
pixi add "python>=3.11"
pixi add --pypi --editable "plantbench @ file:///path/to/plantbench"   # the clone
pixi run plantbench new-case my_tank                   # a package, from the template
pixi add --pypi --editable "my_tank[test] @ file://$PWD/my_tank"       # plantbench now finds my_tank
pixi run pytest my_tank                                # the contract, on the case
pixi run plantbench check my_tank                      # the same, with a residual report
pixi run plantbench generate spec.toml --workers 8     # a dataset; the spec names my_tank
pixi run plantbench check my_tank --contribute         # before proposing it to the library
```
:::

:::{tab-item} pip
:sync: pip

```
plantbench new-case my_tank ~/work/my_tank             # a package, from the template
pip install -e ~/work/my_tank'[test]'                  # plantbench now finds my_tank
pytest ~/work/my_tank                                  # the contract, on the case
plantbench check my_tank                               # the same, with a residual report
plantbench generate spec.toml --workers 8              # a dataset; the spec names my_tank
plantbench check my_tank --contribute                  # before proposing it to the library
```
:::

::::

## Starting from the template

`new-case` checks the id against the [naming convention](adding-a-case.md#naming-a-case), refuses an id plantbench already finds, and writes:

| Path | Contents |
|---|---|
| `pyproject.toml` | the package, with the case declared as an entry point |
| `my_tank/definition.py` | the `Case`: options, structures, disturbances, measurements, state names |
| `my_tank/model.py` | parameters, inputs, the design and its `rhs`; a gravity-drained tank to replace |
| `my_tank/README.md` | the case card, with the sections a card must have |
| `tests/test_contract.py` | the contract of `plantbench.contract`, run on the case |

The generated case is the template's tank under the new id, and it passes the contract as written. Replace the tank with the plant one piece at a time and keep the contract passing; the [checklist](adding-a-case.md#checklist) lists what each piece is.

## Making a case available

Plantbench finds a case in three places, in this order.

**Built in.** The cases of the library, listed in `plantbench/cases/__init__.py`.

**Entry point.** An installed package that declares the case in its `pyproject.toml`:

```toml
[project.entry-points."plantbench.cases"]
my_tank = "my_tank.definition:CASE"
```

This is what `new-case` writes. Once the package is installed (`pip install -e`, `uv pip install -e` or `pixi add --pypi --editable` while developing), every process finds the case: a script, the command line, and each worker of a dataset generation.

**Session.** A case defined in a script or notebook, without a package:

```python
import plantbench as pb

@pb.case("my_tank")
def my_tank() -> pb.Case:
    return pb.Case(id="my_tank", ...)
```

The function is called the first time the case is loaded. `pb.register(CASE)` does the same for a `Case` already built. A session case exists in the process that defined it, and dataset generation imports the defining module again in each worker. A case defined in a script therefore needs its generation under `if __name__ == "__main__":`, as any script using worker processes does, and a case typed into an interactive session generates with one worker only.

An id is refused if it breaks the naming convention, names a built-in case, or is declared by an installed package; two packages declaring one id are refused when either is looked up.

## Helpers

`plantbench.core.casekit` holds the pieces every case repeats. None is required.

| Helper | Use |
|---|---|
| `StateLayout` | the state vector as named blocks, a holdup or a composition array: `pack`, `unpack`, `names()` for `state_names`, `index` |
| `state_measurement` | a measurement reading one state, by index or by its name in a layout |
| `cached_design` | caches a slow design solve per worker process, as `jacketed_cstr` does |
| `fixed_point_residual` | the largest `rhs` at the design point and the states where it is largest, for a design that is not yet a steady state |

```python
from plantbench.core.casekit import StateLayout, state_measurement

LAYOUT = StateLayout([("M_B", ()), ("x_B", 3), ("M", 20), ("x", (20, 3))])
state_names = lambda design: LAYOUT.names({"x": ["I", "A", "B"], "x_B": ["I", "A", "B"]})
m_base_level = state_measurement("M_B", LAYOUT)
```

## Batched evaluation

A design may declare `batched = True` (a class attribute) when its `rhs` and the case's measures accept a batch of m states, `x` of shape (m, n), with inputs whose fields carry the same leading axis (`control.batch_inputs`), and return (m, n) derivatives and (m,) measurements. `jacobian="batched"` then evaluates the closed loop's Jacobian as one batch, on the host or on a GPU (User guide, The batched Jacobian and GPUs). It is optional, and a case that does not declare it runs as before.

The usual way to write one function for one state and for a batch is to index the last axis, `x[..., k]` and `u.z_feed[..., k]`, and to end a measure in `casekit.measured(...)` rather than `float(...)`. `StateLayout.unpack` and `pack` take a batch. A case on the GPU takes its array module from its states (`backend.namespace(x)`) rather than calling NumPy, and `PengRobinson.on(device)` gives the equation of state on the device. The contract check `a_declared_batch_evaluates_as_its_rows` compares a batch with single evaluations.

## Testing

The contract is shipped with the library as `plantbench.contract.CHECKS`, the same checks the library runs on its own cases: the id, a fixed point under every structure, the reference structure, state names, finite measurements, the configuration round trip, ideal instruments, a short run at rest and the generic disturbances. The generated `tests/test_contract.py` runs them under pytest:

```python
@pytest.mark.parametrize("check", contract.CHECKS, ids=lambda c: c.__name__)
def test_contract(check):
    check(CASE)
```

`plantbench check my_tank` runs them without pytest and reports how far the open-loop design point is from a steady state, which is the first thing to look at when the fixed-point check fails. The case's own tests, of conservation, analytic limits and the published values it reproduces, go beside the contract in `tests/`.

## Using the case

Every part of the library takes the case by its id:

```python
traj = pb.run("my_tank", pb.load_case("my_tank").config(), t_end=100.0)
```

A specification names it in its `[study]` table, and `generate`, `datasets`, `compare`, tasks and Sobol' indices work on the dataset as on any other ([user guide](user-guide.md#generating-a-dataset)).

## Provenance

A dataset over a case that is not built in records where the case came from, beside the library version and commit it records for every dataset. The manifest's `case_source` holds the module, the distribution and its version for an installed package, and the commit of the checkout the case's module is in, with whether that checkout had uncommitted changes. `plantbench datasets` prints it, and a results file stamped from the dataset carries it on a line of its own. A copy installed under site-packages records no commit, since the repository that encloses an environment is not the one that produced the code.

Run ids and specification digests depend on the configuration and the case id only, so a dataset regenerated from the same case in the library after it is added reproduces the same ids.

## Contributing the case

When the case is ready to be offered to the library, `plantbench check my_tank --contribute` adds the library's rules to the contract: imports from the library's `backend`, `core`, `units` and `heat` only, no other case, and no third-party import outside NumPy and SciPy that the project does not declare in an extra; a case card with every section; and tests of the case's own. It then lists the rules that need a reviewer's judgment. `plantbench contribute my_tank` then moves the case into a clone of the library; [Contributing a case](tutorial/contributing.md) follows it on an example, and [Adding a case](adding-a-case.md#contributing-a-case-from-its-own-package) gives the steps of the pull request.
