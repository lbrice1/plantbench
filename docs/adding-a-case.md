# Adding a case

This page is the reference for a case in the library: its name, what it must provide, the rules it keeps and how it joins. A case is usually developed first in a package of its own and used from there; [Developing a case](developing-a-case.md) describes that workflow.

## Starting point

`plantbench/cases/_template/` is a working case, a gravity-drained tank with a level loop, kept working by the contract tests. Start from it in a package of your own,

```
plantbench new-case <name>
```

or, working in a clone of the library, copy it into a directory that does not exist yet (`cp -r` into an existing directory copies the template inside it instead):

```
cp -r plantbench/cases/_template plantbench/cases/<name>
```

Then replace the tank with the plant. `plantbench/cases/jacketed_cstr/` is the smallest real case and shows the same structure on a reactor with three steady states.

## Naming a case

A case is named for the process it models. The name is its id in `load_case`, the name of its directory under `plantbench/cases/`, its directory under `tests/cases/`, and its directory under `studies/` when it has studies of its own.

- Lowercase words joined by single underscores, beginning with a letter, at most 40 characters: a valid Python identifier, since the directory is a package.
- A flowsheet is named by its principal units in flow order (`reactor_separator_recycle`), or by the established name of a benchmark from the literature (`tennessee_eastman`, `williams_otto`).
- A single unit is named by the unit and the feature that defines it (`jacketed_cstr`).
- No number, version, author, book or source in the name, and not the word `case`. The provenance goes in the case card.
- A variant that would move a frozen case's reference results is a new case, named after the original with a qualifier that says what differs (`reactor_separator_recycle_noisy_analyzer`).
- A published name is permanent. The case id enters the digest of every specification that names it, so renaming a case changes the digests of every dataset and protocol reported against it.

`plantbench.contract` checks the form of the name, and for a case in the library that it matches the directory. An id that breaks the convention is also refused when a case is registered or scaffolded.

## Checklist

1. **Parameters.** A frozen dataclass holding every constant with its unit. Where a value is corrected from a published source, record the correction and its evidence in a comment beside it. Unit-operation models that another case could use belong in `plantbench/units/`, with their parameter sets.
2. **Inputs.** A dataclass of every quantity a loop may manipulate or a disturbance may change. Scalar fields are recorded in every trajectory, and array fields element by element.
3. **Design.** An object with `x`, `u`, `pp` and a method `rhs(t, x, u, pp)`, at a steady state: the right-hand side at the design point must vanish to 1e-9. Declare signals derived from the inputs, such as a recycle flow computed from two others, as a `derived` class attribute mapping names to functions of the inputs.
4. **Structures.** Functions returning lists of `Loop` and `Ratio` elements, passed through `control.bias_from_design`. Choose tunings for damping, not merely stability, and record the damping ratio they give.
5. **Disturbances.** The generic `step` and `ramp` cover a change in one input. Add a named disturbance, `factory(design, config, **arguments)`, where the change is anything else.
6. **Measurements and state names.** Name every signal a study would want in `y`, and every state.
7. **The `Case`.** In `definition.py`, with the options the case accepts and their defaults. `make_design` applies `config.params` with `core.case.override` and caches the design solve if it is slow. `features` is optional: a mapping from name to a function of the `Setup` returning a dictionary, for quantities a dataset should record beside each run, such as the objectives a design is to be judged on. They are computed from the design point, so they cost nothing beyond the design and survive a run that fails.

   Adding an option to a case that already has datasets changes the hash of every configuration ever made from it, because a configuration is validated with the case's defaults filled in before it is hashed, and with it the identity of every run already generated. A quantity that has to be added to a settled case is therefore better carried as a parameter, which a configuration never acquires by default.
8. **Registration.** Add the case's `definition` module to `_MODULES` in `plantbench/cases/__init__.py`. A case kept outside the library is declared as an entry point of its package, or made available for a session with `@plantbench.case` or `plantbench.register(CASE)` ([Developing a case](developing-a-case.md#making-a-case-available)).
9. **Tests.** The contract (`plantbench.contract.CHECKS`, run by `tests/cases/test_contract.py`) runs on every registered case: fixed point under every structure, state names, configuration round trip, finite measurements, ideal instruments adding nothing, a short run at rest. Add the case's own tests in `tests/cases/<case>/`: conservation, analytic limits, and the published values it reproduces.
10. **Case card.** `README.md` in the case's directory, with the sections of the existing cards: provenance, process, states, inputs, options, structures, disturbances, measurements, reference results, verification and limitations. A case presented in a publication of its own adds a Citation section: the reference as its first paragraph, then its BibTeX. `plantbench describe` prints the section, dataset cards list the reference, and the library's citation text asks users to cite the publication as well.

## Rules

- A case imports from `core`, `units` and `heat`, and from nothing else in the library. It does not import another case.
- `core` is not changed to accommodate one case. If the interface cannot express a case, the interface is changed so that it serves every case, with the contract tests and both golden tests passing.
- Code is moved into a shared layer when a second case needs it ([design principles](philosophy.md#generalizing-on-the-second-use)).
- A case built on published results carries a golden test of its reference configuration and is frozen from then on.

## Contributing a case from its own package

A case developed with `new-case` has the layout of a case in the library inside its package directory, so adding it is a move of that directory. First,

```
plantbench check <case> --contribute
```

must pass. It runs the contract and the rules above that can be checked mechanically: imports from `core`, `units` and `heat` only, no other case, and no third-party import that an extra does not declare; a case card with every section; tests of the case's own. It then lists the rules that need a reviewer's judgment, which the pull request should answer. Then, in a fork of the library ([CONTRIBUTING.md](https://github.com/lbrice1/plantbench/blob/main/CONTRIBUTING.md)):

1. Move the package directory to `plantbench/cases/<case>/`, card included.
2. Make imports between the case's own modules relative (`from .model import ...`), and imports of the library's layers absolute, as they are already.
3. Add the case to `_MODULES` in `plantbench/cases/__init__.py`.
4. Move the case's tests to `tests/cases/<case>/`, leaving out `test_contract.py`, which the library runs on every registered case.
5. Move any extra of the package into the library's `pyproject.toml`, named for what it provides.
6. Uninstall the package, so that its entry point no longer declares the id the library now holds.
7. Add the case to the case tables of `README.md` and `docs/cases/index.md`, with a page under `docs/cases/`.
8. If the case was presented in a publication of its own, give it in a Citation section of the case card (checklist item 10).
9. Run `plantbench check <case> --contribute` again, and the verification below.

## Verification

::::{tab-set}
:sync-group: installer

:::{tab-item} uv
:sync: uv

```
uv run pytest -q                                 # includes the contract suite for the new case
uv run pytest -q -m slow                         # the slow golden test of `reactor_separator_recycle`
uv run plantbench check <new case> --contribute
uv run plantbench describe <new case>
```
:::

:::{tab-item} pixi
:sync: pixi

```
pixi run test                                    # includes the contract suite for the new case
pixi run test-slow                               # the slow golden test of `reactor_separator_recycle`
pixi run plantbench check <new case> --contribute
pixi run plantbench describe <new case>
```
:::

:::{tab-item} pip
:sync: pip

```
.venv/bin/python -m pytest -q                    # includes the contract suite for the new case
.venv/bin/python -m pytest -q -m slow            # the slow golden test of `reactor_separator_recycle`
.venv/bin/plantbench check <new case> --contribute
.venv/bin/plantbench describe <new case>
```
:::

::::
