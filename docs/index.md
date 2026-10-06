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

## Installation

```{include} _installation.md
```

## Contents

- [Tutorial](tutorial.md): installation, running a case, generating a dataset, and the tools around it, in one session.
- [User guide](user-guide.md): running cases, configurations, non-idealities, disturbances, datasets and tasks.
- [Design principles](philosophy.md): what a case is, the conventions the code keeps, and why.
- [Cases](cases/index.md): the card of each case.
- [Examples](examples.md): short scripts, each run by the test suite.
- [Developing a case](developing-a-case.md): a case in a package of its own, used with every part of the library without adding it.
- [Adding a case](adding-a-case.md): naming, the checklist, the contract a new case must pass, and contributing a case to the library.
- [API reference](api.md).

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

:::{note}
Some cases were introduced in a publication of their own. The [card](cases/index.md) of such a case gives that publication in a Citation section, and `plantbench describe <case>` prints it. When you use such a case, please cite its publication.
:::

`plantbench` is distributed under the BSD 3-Clause license.

```{toctree}
:hidden:
:maxdepth: 2

tutorial
user-guide
philosophy
cases/index
examples
developing-a-case
adding-a-case
api
```
