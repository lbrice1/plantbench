# API reference

`plantbench` exposes `load_case`, `list_cases`, `case`, `register`, `build`, `run`, `run_ensemble` and `provenance` at the top level, with the classes `Case`, `Config`, `Setup`, `Trajectory`, `Stepper` and `Step` for a plant advanced one interval at a time (`plantbench.core.stepper`), and `EnsembleStepper` and `EnsembleStep` for many advanced together (`plantbench.core.ensemble_stepper`). The subpackages hold the layers beneath: `plantbench.core` the control layer, instruments, disturbances, the case interface, operating limits and their monitor (`plantbench.core.limits`), the stepped map's fixed point and linearization (`plantbench.core.stepmap`) and specifications; `plantbench.units` the unit-operation models; `plantbench.heat` process streams and pinch analysis; `plantbench.backend` the array module a batched evaluation runs on, NumPy or CuPy; `plantbench.cases` the cases, and `plantbench.cases.template` the template case, which is not registered. Dataset generation, reading, reporting, tasks and Sobol' indices are the modules `plantbench.datagen`, `plantbench.datasets`, `plantbench.report`, `plantbench.task` and `plantbench.sensitivity`. The contract a case keeps is `plantbench.contract`, the helpers for writing one `plantbench.core.casekit`, and the package `new-case` writes `plantbench.scaffold`.

```{toctree}
:maxdepth: 3

apidocs/plantbench/plantbench
```
