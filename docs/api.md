# API reference

`plantbench` exposes `load_case`, `list_cases`, `case`, `register`, `build` and `run` at the top level, with the classes `Case`, `Config`, `Setup` and `Trajectory`. The subpackages hold the layers beneath: `plantbench.core` the control layer, instruments, disturbances, the case interface and specifications; `plantbench.units` the unit-operation models; `plantbench.heat` process streams and pinch analysis; `plantbench.backend` the array module a batched evaluation runs on, NumPy or CuPy; `plantbench.cases` the cases. Dataset generation, reading, reporting, tasks and Sobol' indices are the modules `plantbench.datagen`, `plantbench.datasets`, `plantbench.report`, `plantbench.task` and `plantbench.sensitivity`. The contract a case keeps is `plantbench.contract`, the helpers for writing one `plantbench.core.casekit`, and the package `new-case` writes `plantbench.scaffold`.

```{toctree}
:maxdepth: 3

apidocs/plantbench/plantbench
```
