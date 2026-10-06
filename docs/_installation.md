Python 3.11 or later, from a clone of [the repository](https://github.com/lbrice1/plantbench). The environment is managed by [uv](https://docs.astral.sh/uv/), by [pixi](https://pixi.sh), or by pip in a virtual environment; uv and pixi install the versions recorded in `uv.lock` and `pixi.lock`.

```
git clone https://github.com/lbrice1/plantbench
cd plantbench
```

::::{tab-set}
:sync-group: installer

:::{tab-item} uv
:sync: uv

```
uv sync --extra dev                  # the library and its tests
uv sync --extra dev --extra study    # and the studies
uv run plantbench --help
uv run pytest -q
```

Commands run in the environment through `uv run`, or directly after `source .venv/bin/activate`.
:::

:::{tab-item} pixi
:sync: pixi

```
pixi install                         # the library and its tests
pixi install -e study                # and the studies
pixi run plantbench --help
pixi run test
```

Commands run in the environment through `pixi run` (`pixi run -e study` for the studies), or directly inside `pixi shell`. NumPy, SciPy and Python come from conda-forge.
:::

:::{tab-item} pip
:sync: pip

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # the library and its tests
.venv/bin/pip install -e ".[dev,study]"    # and the studies
.venv/bin/plantbench --help
.venv/bin/python -m pytest -q
```
:::

::::

The examples on these pages write `plantbench` for the command line; `python -m plantbench` is equivalent.
