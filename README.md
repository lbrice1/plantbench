<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.svg">
  <img alt="plantbench" src="docs/_static/logo.svg" width="338">
</picture>

# plantbench

A library of plantwide process simulations for use as test problems in process systems engineering: control structure selection, integration of design and control, surrogate and data-driven modeling, and knowledge discovery from closed-loop data.

Each case is a plant model with a verified ground truth and a design space. The design space covers structures, options, parameters, set-points, tuning, measurement and valve non-idealities, and disturbances. Every one of these is set by a configuration that is plain data, and a specification of many configurations generates a dataset that records how it was made.

```python
import plantbench as pb

case = pb.load_case("reactor_separator_recycle")
config = case.config(options={"network": "D4"},
                     instruments={"reactor C_A": {"deadtime": 5.0}},
                     disturbance={"name": "throughput", "fraction": 0.10})
traj = pb.run(case, config, t_end=1500.0, dt=1.0)
```

```
plantbench generate examples/04_dataset.toml --workers 8
```

## Cases

| Case | States | Description |
|---|---|---|
| `reactor_separator_recycle` | 134 (base plant) | A non-isothermal CSTR on the open-loop unstable branch of its multiplicity, in a recycle loop closed by a thirty-stage column with a purge; four heat-exchanger networks at three column pressures; four regulatory structures. Extends the reactor control example of Romagnoli and Palazoglu (2020); frozen. |
| `jacketed_cstr` | 3 | The same reactor on its own, with three steady states and structures from open loop to a composition cascade. |

Each case has a card in `plantbench/cases/<case>/README.md`. Cases are named for the process they model, following the convention in `docs/adding-a-case.md`.

A plant of your own is a case in a package of its own, which plantbench finds by its id once installed, with every part of the library available to it:

```
plantbench new-case my_tank     # a package from the template, the contract test included
pip install -e my_tank          # or `uv pip install -e my_tank`
plantbench check my_tank        # the contract every case keeps
```

See `docs/developing-a-case.md`, which also covers pixi.

## Installation

Python 3.11 or later, from a clone of the repository. The environment is managed by [uv](https://docs.astral.sh/uv/), by [pixi](https://pixi.sh), or by pip in a virtual environment; uv and pixi install the versions recorded in `uv.lock` and `pixi.lock`.

```
git clone https://github.com/lbrice1/plantbench
cd plantbench
```

### uv

```
uv sync --extra dev                  # the library and its tests
uv sync --extra dev --extra study    # and the studies
uv run plantbench --help
```

### pixi

```
pixi install                         # the library and its tests
pixi install -e study                # and the studies
pixi run plantbench --help
```

NumPy, SciPy and Python come from conda-forge.

### pip

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # the library and its tests
.venv/bin/pip install -e ".[dev,study]"    # and the studies
.venv/bin/plantbench --help
```

The commands below are written for the pip environment. With uv, replace `.venv/bin/` with `uv run `; with pixi, with `pixi run ` (`pixi run -e study ` for the studies). `python -m plantbench` is equivalent to `plantbench`.

The reference results were produced with Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, PaCMAP 0.9.1 and HDBSCAN 0.8.44 on macOS (arm64).

## Tests

```
.venv/bin/python -m pytest -q             # about 90 s, including the golden test of reference.txt
.venv/bin/python -m pytest -q -m slow     # the golden test of designs.txt, about 150 s
```

## Reproducing the accompanying study

The study that accompanies the library is described in a manuscript submitted to the 2027 American Control Conference (ACC) and not yet published (see Citation). Every number in the manuscript is transcribed from a results file under `studies/*/results/`, and every figure is drawn by `studies/acc_figures.py`. The datasets are not distributed; each is regenerated from the specification committed beside its study. From the repository root, with the `study` extra installed, and in this order:

```
# `reactor_separator_recycle`: the reference plant and the design comparison (about 20 s and several minutes)
.venv/bin/python -m studies.reactor_separator_recycle.reference
.venv/bin/python -m studies.reactor_separator_recycle.designs    # designs.txt to results/, designs.pkl to cache/

# The enumerated dataset of 2016 closed-loop runs (about 1.5 h on 10 cores) and its analysis
.venv/bin/plantbench generate studies/regimes/spec.toml --workers 10
.venv/bin/python -m studies.regimes.analysis data/regimes

# The deadtime sweep and the Sobol' indices: the damping ratio, then the outcome sample
# (about 9 h of integration, divided by the workers), then the outcome's indices
.venv/bin/python -m studies.design.analysis --workers 10
.venv/bin/python -m studies.design.verify --workers 10
.venv/bin/python -m studies.design.analysis --workers 10

# The three figures, written to figures/acc
.venv/bin/python -m studies.acc_figures --designs-cache cache
```

| Results file | Contents | In the manuscript |
|---|---|---|
| `studies/reactor_separator_recycle/results/reference.txt` | the reactor against its reference values; the recycle under each structure | Section II (reactor eigenvalues); Section IV (recycle amplification) |
| `studies/reactor_separator_recycle/results/designs.txt` | the four heat-exchanger networks, their damping ratios and closed-loop responses | Table II; Section IV (tuning, retuning of D4, the throughput response); Fig. 2 |
| `studies/regimes/results/results_regimes.txt` | the enumerated dataset, the damping ratio as a detector, the early-window classifiers, the mechanism | Table I; Table III (sections 5 and 6 of the file); Section V; Fig. 3 |
| `studies/design/results/results_design.txt` | the damping ratio against the analyzer deadtime; the Sobol' indices | Table IV (section 2 of the file); Section VI (the deadtime sweep, section 1) |

The Sobol' indices of Table IV are computed over the box in `SOBOL_BOUNDS` of `studies/design/analysis.py`: pump-around share 0 to 0.7, column pressure 0.15 to 0.275 bar, K_c −4 to −0.25 m³/(min K), τ_I 5 to 160 min and analyzer deadtime 0 to 10 min.

Wall times depend on the machine and its load, so the integration times in a regenerated results file (and the vertical axis of the damping-ratio figure) differ from those in the manuscript. Every other number reproduces exactly on the reference platform (macOS arm64, with the versions listed under Installation). The mechanism of Section V-C passes through t-SNE, PaCMAP and HDBSCAN, whose output can differ in the last digits on another platform or BLAS build.

The two datasets were generated before the repository existed, so their manifests record no commit (`generated at no commit` in the results files). The analyses were run before the history was condensed into the release commit, at a commit the results files name as `6fc4e69` and this repository does not contain. Rerunning both analyses from the release commit reproduces every number in the two results files.

The identifiers reported in the manuscript, `pb-spec:6a6fc673261b` for the enumerated dataset and `pb-protocol:dab0d47475e9` for the dataset with its early-window task, are printed in `results_regimes.txt`, and `plantbench verify studies/regimes/spec.toml pb-spec:6a6fc673261b pb-protocol:dab0d47475e9` recomputes both.

## Layout

| Path | Contents |
|---|---|
| `plantbench/core/` | control layer, instruments, disturbances, the case interface, specifications, helpers for writing a case |
| `plantbench/units/` | unit-operation models: the CSTR, the tray column, vapor pressures |
| `plantbench/heat/` | process streams and pinch analysis |
| `plantbench/cases/` | the cases, how they are found, and a template for new ones |
| `plantbench/contract.py` | the contract every case keeps, and the rules for a case added to the library |
| `plantbench/scaffold.py` | `new-case`: a package for a new case, from the template |
| `plantbench/datagen.py` | dataset generation, from a specification or from any list of configurations |
| `plantbench/datasets.py` | reading datasets back: what is on disk, provenance stamps, comparing two generations |
| `plantbench/task.py` | tasks: what is predicted from a dataset, and how an answer is scored |
| `plantbench/sensitivity.py` | Sobol' first-order and total indices |
| `docs/` | design principles, user guide, adding a case |
| `examples/` | short scripts, each run by the test suite |
| `studies/reactor_separator_recycle/` | the reference results and the design comparison of `reactor_separator_recycle`, and the flowsheet drawings |
| `studies/regimes/` | operating-regime discovery on `reactor_separator_recycle`'s design space |
| `studies/design/` | the damping ratio against the analyzer deadtime, and its Sobol' indices |
| `studies/acc_figures.py` | the figures of the manuscript, at the ACC 2027 column widths |
| `tests/` | the verification suite |

## Documents

The documentation is at <https://plantbench.readthedocs.io>: the guides below, the case cards, the examples and the API reference.

- `docs/tutorial.md`: installation, running a case, generating a dataset, and the tools around it, in one session.
- `docs/philosophy.md`: what a case is, the conventions the code keeps, and why.
- `docs/user-guide.md`: running cases, configurations, non-idealities, disturbances, datasets.
- `docs/developing-a-case.md`: a case in a package of its own, found by entry point or decorator, and used without adding it to the library.
- `docs/adding-a-case.md`: the checklist, the contract a new case must pass, and contributing a case to the library.

## Citation

If you use `plantbench`, please cite the paper that introduces it:

Territo, K., Briceno-Mena, L. A. and Romagnoli, J. A. (2027). A Plantwide Simulation Library for Control-Oriented Design: Analytical and Data-driven Use Cases. Submitted to the 2027 American Control Conference (ACC).

```bibtex
@inproceedings{territo2027plantbench,
  author    = {Territo, Kyle and Briceno-Mena, Luis A. and Romagnoli, Jose A.},
  title     = {A Plantwide Simulation Library for Control-Oriented Design: Analytical and Data-driven Use Cases},
  booktitle = {2027 American Control Conference (ACC)},
  year      = {2027},
  note      = {Submitted}
}
```

Some cases were introduced in a publication of their own. The card of such a case gives that publication in a Citation section, and `plantbench describe <case>` prints it. When you use such a case, please cite its publication.

## References

- Briceno-Mena, L. A., Nnadili, M., Benton, M. G. and Romagnoli, J. A. (2022). Data mining and knowledge discovery in chemical processes: effect of alternative processing techniques. *Data-Centric Engineering*, 3, e18.
- Luyben, W. L., Tyréus, B. D. and Luyben, M. L. (1998). *Plantwide Process Control*. McGraw-Hill.
- Romagnoli, J. A. and Palazoglu, A. (2020). *Introduction to Process Control*, 3rd edition. CRC Press.
- Seghers, E. E., Briceno-Mena, L. A. and Romagnoli, J. A. (2023). Unsupervised learning: local and global structure preservation in industrial data. *Computers & Chemical Engineering*, 178, 108378.

## License

BSD 3-Clause; see [LICENSE](LICENSE). Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
