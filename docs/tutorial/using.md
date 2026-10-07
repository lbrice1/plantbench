# Using plantbench

This tutorial follows one session with plantbench from installation to a scored task. It works on `jacketed_cstr`, the reactor of `reactor_separator_recycle` on its own: three states, a design point on the open-loop unstable middle branch, and runs that take a fraction of a second. Every step applies unchanged to any other case. The [user guide](../user-guide.md) covers each topic in full, and the test suite runs this page as written.

## Installation

```{include} ../_installation.md
```

The rest of the tutorial runs inside that environment (`source .venv/bin/activate` for uv and pip, `pixi shell` for pixi), from an empty directory of its own so that the files it writes stay together:

```
mkdir plantbench-tutorial
cd plantbench-tutorial
```

```console
$ plantbench --version
plantbench 0.2.1
$ plantbench list
jacketed_cstr              Non-isothermal CSTR with a cooling jacket  [built in]
ngl_deethanizer            Deethanizer of an NGL fractionation train  [built in]
ngl_demethanizer           Demethanizer section of a cryogenic NGL recovery plant  [built in]
reactor_separator_recycle  Reactor-separator-recycle plant with heat integration  [built in]
```

## Running a case

`describe` prints what a case provides: its states, inputs, options, control structures, disturbances and measurements.

```console
$ plantbench describe jacketed_cstr
jacketed_cstr: Non-isothermal CSTR with a cooling jacket

The reference reactor on its own: first-order exothermic A -> B, three states, three steady states, and a design point on the open-loop unstable middle branch. Structures from open loop to the composition cascade. Its steady state and eigenvalues reproduce the published values once three modifications to the published parameter table are applied.

states          3
inputs          F0, CA0, T0, Fj
options         branch = 'middle'
structures      open loop, temperature PI (reference), composition cascade
disturbances    feed_temperature, step, ramp
measurements    C_A, T, Tj

reference plant: temperature PI; loops reactor T
closed-loop rightmost eigenvalue -0.0516 1/min, least damping ratio 0.274
```

In Python, a case is loaded by its id, and a configuration names a structure and whatever departs from the reference plant. Here the feed arrives at 340 K, 3 K hotter than design, from 10 min on.

```python
import plantbench as pb

case = pb.load_case("jacketed_cstr")
hotter = {"name": "feed_temperature", "value": 340.0, "t": 10.0}
config = case.config("temperature PI", disturbance=hotter)
traj = pb.run(case, config, t_end=600.0, dt=1.0)

print(traj.t.shape, traj.x.shape)
print(list(traj.y), list(traj.u))
print(f"T {traj.y['T'][0]:.2f} -> {traj.y['T'][-1]:.2f} K")
```

```
(601,) (3, 601)
['C_A', 'T', 'Tj'] ['F0', 'CA0', 'T0', 'Fj']
T 362.40 -> 362.40 K
```

A trajectory holds the time grid `t` in minutes, the states `x`, the inputs `u` and the case's named measurements `y`. The temperature loop returns the reactor to its set-point. With the loop open, the same disturbance carries the reactor to the high-temperature steady state:

```python
open_loop = pb.run(case, case.config("open loop", disturbance=hotter), t_end=600.0, dt=1.0)
print(f"T {open_loop.y['T'][0]:.2f} -> {open_loop.y['T'][-1]:.2f} K")
```

```
T 362.40 -> 384.39 K
```

The two runs side by side:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(open_loop.t, open_loop.y["T"], label="open loop")
ax.plot(traj.t, traj.y["T"], label="temperature PI")
ax.set(xlabel="time (min)", ylabel="T (K)")
ax.legend()
fig.savefig("feed_temperature.png", dpi=150, bbox_inches="tight")
```

`traj.window(t0, t1)` returns the part of a trajectory between two times, and `traj.to_npz(path)` writes it to a file that `pb.Trajectory.from_npz` reads back with its configuration:

```python
traj.to_npz("closed_loop.npz")
again = pb.Trajectory.from_npz("closed_loop.npz")
print(again.config["structure"], again.window(500.0, 600.0).t[[0, -1]])
```

```
temperature PI [500. 600.]
```

## Changing the configuration

A configuration is plain data. Its sections set the tuning of a loop, schedule set-point changes, and add sensor and valve non-idealities; the [user guide](../user-guide.md#configurations) lists them all. A 2 K step in the temperature set-point at 10 min, under a gentler tuning:

```python
step = case.config("temperature PI",
                   tuning={"reactor T": {"Kc": -0.5, "tau_I": 20.0}},
                   setpoint_steps={"reactor T": [(10.0, 2.0)]})
traj = pb.run(case, step, t_end=600.0, dt=1.0)
T = traj.y["T"]
print(f"T {T[0]:.2f} -> {T[-1]:.2f} K, peak {T.max():.2f} K")
print(step.to_dict())
print(step.key())
```

```
T 362.40 -> 364.40 K, peak 365.65 K
{'structure': 'temperature PI', 'options': {'branch': 'middle'}, 'params': {}, 'setpoints': {}, 'setpoint_steps': {'reactor T': [[10.0, 2.0]]}, 'tuning': {'reactor T': {'Kc': -0.5, 'tau_I': 20.0}}, 'instruments': {}, 'actuators': {}, 'disturbance': {}}
22922c6933749232
```

`case.config` fills in the case's default options, and every name is checked when the plant is built, so a misspelled loop or parameter raises rather than being ignored. `to_dict` gives the configuration as plain data and `key` a stable hash of it. The same configuration, written as a TOML file, runs from the command line:

```{code-block} toml
:caption: step.toml

structure = "temperature PI"
tuning = {"reactor T" = {Kc = -0.5, tau_I = 20.0}}
setpoint_steps = {"reactor T" = [[10.0, 2.0]]}
```

```console
$ plantbench run jacketed_cstr --config step.toml --t-end 600 --out step.npz
jacketed_cstr temperature PI: 601 points over 600 min, 0.01 s, 925 right-hand-side evaluations
  C_A                    2.3564 ->      2.16473
  T                     362.401 ->      364.401
  Tj                    345.783 ->      346.785
written to step.npz
```

## Linear analysis

`pb.build` returns the configured plant at its design point without simulating it, and `spectrum` the eigenvalues of the closed loop linearized there. A deadtime on the temperature measurement shows how quickly this loop loses its margin:

```python
from plantbench.core import control as ctl

for deadtime in (0.0, 0.5, 1.0):
    lagged = case.config("temperature PI", instruments={"reactor T": {"deadtime": deadtime}})
    ev = pb.build(case, lagged).spectrum()
    print(f"deadtime {deadtime:.1f} min: damping ratio {ctl.damping_ratio(ev):6.3f}, "
          f"rightmost eigenvalue {ev.real.max():7.4f} 1/min")
```

```
deadtime 0.0 min: damping ratio  0.274, rightmost eigenvalue -0.0516 1/min
deadtime 0.5 min: damping ratio  0.024, rightmost eigenvalue -0.0204 1/min
deadtime 1.0 min: damping ratio -0.158, rightmost eigenvalue  0.1198 1/min
```

A negative damping ratio is a growing oscillation: one minute of deadtime makes the reference loop unstable. The deadtime is represented by a Padé approximation, whose states enter the Jacobian with those of the plant and the controller. The nonlinear run agrees, settling into a sustained oscillation:

```python
lagged = case.config("temperature PI", instruments={"reactor T": {"deadtime": 1.0}},
                     disturbance=hotter)
late = pb.run(case, lagged, t_end=600.0, dt=1.0).window(500.0, 600.0)
print(f"swing in T over the last 100 min: {late.y['T'].max() - late.y['T'].min():.2f} K")
```

```
swing in T over the last 100 min: 7.20 K
```

## Generating a dataset

A specification describes many runs at once: a base configuration, variants of it, a Cartesian sweep, and a Latin hypercube sample, each over dotted paths into the configuration. This one crosses two structures with two feed-temperature upsets, and samples four combinations of the temperature-loop gain and the measurement deadtime for each, 16 runs in all:

```{code-block} toml
:caption: tutorial.toml

[study]
name = "tutorial"
case = "jacketed_cstr"
seed = 1

[base]
structure = "temperature PI"

[[variants]]
structure = "temperature PI"

[[variants]]
structure = "composition cascade"

[sweep]
"disturbance" = [{name = "feed_temperature", change = -4.0, t = 10.0},
                 {name = "feed_temperature", change = 4.0, t = 10.0}]

[sample]
n = 4
"tuning.reactor T.Kc" = [-2.0, -0.2]
"instruments.reactor T.deadtime" = [0.0, 1.0]

[run]
t_end = 600.0
dt = 1.0
wall_budget = 60.0
features = ["spectrum"]
```

`features` records quantities computed from the configured plant beside each run; `spectrum` is the damping ratio, rightmost eigenvalue and number of unstable modes at the design point.

```console
$ plantbench generate tutorial.toml --workers 2
tutorial: 16 runs, 0 already recorded, 16 to run on 2 worker(s)
[1/16] 776bfaf38427a557 ok         0.02 s
[2/16] 7372c5e82efaed03 ok         0.02 s
...
[16/16] 46b48cfd3860c57e ok         0.29 s
dataset in data/tutorial
```

The dataset is a directory: the specification, a manifest of the code and environment that generated it, one record per run in `runs.jsonl` and `index.csv`, and one trajectory per completed run.

```
data/tutorial/
  spec.toml
  manifest.json
  runs.jsonl
  index.csv
  runs/<run_id>.npz
```

Generation resumes where it stopped, so running the command again does nothing:

```console
$ plantbench generate tutorial.toml
tutorial: 16 runs, 16 already recorded, 0 to run on 1 worker(s)
dataset in data/tutorial
```

The records hold each run's status, cost, features and configuration, and `load_run` returns its trajectory. Here the linear feature is set against the nonlinear outcome, the swing in T over the last 100 min:

```python
import numpy as np
from plantbench import datagen

records = datagen.load_records("data/tutorial")
for r in sorted(records, key=lambda r: r["features"]["damping_ratio"])[::3]:
    late = datagen.load_run("data/tutorial", r["run_id"]).window(500.0, 600.0)
    print(f"{r['config']['structure']:20s} damping ratio {r['features']['damping_ratio']:6.3f}"
          f"   late swing in T {np.ptp(late.y['T']):8.2e} K")
```

```
temperature PI       damping ratio -0.252   late swing in T 7.38e+00 K
temperature PI       damping ratio -0.135   late swing in T 4.88e+00 K
temperature PI       damping ratio  0.010   late swing in T 1.15e-10 K
composition cascade  damping ratio  0.066   late swing in T 4.81e-04 K
temperature PI       damping ratio  0.167   late swing in T 4.48e-10 K
temperature PI       damping ratio  0.280   late swing in T 3.89e-09 K
```

`index.csv` holds the same records with the configurations flattened to dotted columns, for a spreadsheet or pandas:

```python
import csv

with open("data/tutorial/index.csv") as f:
    rows = list(csv.DictReader(f))
print([c for c in rows[0] if c.startswith(("feature.", "config.tuning", "config.instruments"))])
```

```
['feature.damping_ratio', 'feature.rightmost', 'feature.n_unstable', 'config.tuning.reactor T.Kc', 'config.instruments.reactor T.deadtime']
```

## Provenance

`datasets` lists the datasets on disk with the specification each is of, the commit that generated it, and whether the specification left in the directory still describes the runs:

```console
$ plantbench datasets
directory                      case          specification         generated at   runs                     integration  specification on disk
data/tutorial                  jacketed_cstr pb-spec:7d155fc4b98b  <commit> clean 16 ok                          1.9 s  matches
```

`pb-spec:7d155fc4b98b` names the runs the specification describes: it is the start of the SHA-256 of the specification's canonical form, so any copy of `tutorial.toml` carries the same identifier and a changed one carries another. `card` prints what a paper reports about the dataset, as text, Markdown, LaTeX table rows or JSON:

```console
$ plantbench card data/tutorial --format md
| Entry | Value |
|---|---|
| Study | `tutorial` |
| Case | `jacketed_cstr` |
| Code | `plantbench` 0.2.1 (checkout <commit>, clean) |
| Specification | `data/tutorial/spec.toml`; `pb-spec:7d155fc4b98b` |
| Runs | 16 completed; 0.0 min of integration |
| Environment | Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1 |
...
```

A reader checks a reported identifier against a specification file or a dataset:

```console
$ plantbench verify tutorial.toml pb-spec:7d155fc4b98b
match     pb-spec:7d155fc4b98b: tutorial.toml is pb-spec:7d155fc4b98b
```

A second generation of the same specification, compared with the first, differs only in its wall time:

```console
$ plantbench generate tutorial.toml --out data/tutorial-again --workers 2
$ plantbench compare data/tutorial data/tutorial-again
...
specification: the same
manifest: the same but for the start time
runs: 16 in both, 0 in A only, 0 in B only, 0 whose status differs
features: 16 runs compared over 3 recorded features, 0 moved by more than 1e-09
solver evaluations: 0 of 16 runs took a different path through the integrator
trajectories: 16 compared, 0 differ
cost: wall time differs on 16 of 16 runs, B a median 1.009 times A's; a property of the machine, not of the data
```

## Sensitivity

`plantbench.sensitivity` estimates Sobol' indices: the share of the variance of an output that each variable explains alone (first order) and with its interactions (total). The output here is the rightmost closed-loop eigenvalue, which needs a design solve and no simulation, over the gain, the integral time and the measurement deadtime of the temperature loop. The function receives the whole sample at once, as an array with one row per point.

```python
from plantbench import sensitivity


def rightmost(X):
    out = []
    for Kc, tau_I, deadtime in X:
        config = case.config("temperature PI",
                             tuning={"reactor T": {"Kc": Kc, "tau_I": tau_I}},
                             instruments={"reactor T": {"deadtime": deadtime}})
        out.append(pb.build(case, config).spectrum().real.max())
    return np.array(out)

indices = sensitivity.sobol_indices(rightmost, [(-2.0, -0.2), (3.0, 40.0), (0.0, 1.0)],
                                    n=256, names=["Kc", "tau_I", "deadtime"])
print(f"{indices.n_evaluations} evaluations")
for name, first, total in indices.ranked():
    print(f"{name:8s} first {first:5.2f}  total {total:5.2f}")
```

```
1280 evaluations
deadtime first  0.46  total  0.68
Kc       first  0.28  total  0.44
tau_I    first  0.05  total  0.07
```

The deadtime and the gain account for most of the variance, and their total indices exceed their first-order ones, so part of what each explains is through its interaction with the other. The integral time matters little over this range. The cost is `n * (k + 2)` evaluations for k variables.

## Tasks

A task states what is to be predicted from a dataset and how a prediction is scored, so that two methods reported against one specification are comparable. Tasks need scikit-learn, which is in the `study` extra: `uv sync --extra dev --extra study`, `pixi install -e study`, or `pip install -e '.[dev,study]'`.

The task is a `[task]` table added to the specification. It does not change which runs exist, so the dataset already generated serves it. This one asks whether a run ends in a sustained oscillation, from the first 200 min of its reactor temperature and coolant flow:

```{code-block} toml
:caption: task.toml

[task]
target   = "tutorial.oscillating"
signals  = ["T", "Fj"]
window   = [0.0, 200.0]
split    = {folds = 4, stratified = true, seed = 0}
metrics  = ["balanced_accuracy", "found", "false_alarms"]
positive = "oscillating"
```

```python
from pathlib import Path

Path("tutorial_task.toml").write_text(Path("tutorial.toml").read_text() + "\n"
                                      + Path("task.toml").read_text())
```

The label of a run is read by a function registered under the name the task gives as its `target`:

```python
from collections import Counter

from plantbench import task


@task.labeler("tutorial.oscillating")
def _oscillating(tr):
    late = tr.window(500.0, 600.0)
    return "oscillating" if np.ptp(late.y["T"]) > 0.1 else "settled"

t = task.Task.of("tutorial_task.toml")
obs = t.data("data/tutorial")
print(obs.X.shape, obs.signals, Counter(obs.y.tolist()))
```

```
(16, 201, 2) ('T', 'Fj') Counter({'oscillating': 8, 'settled': 8})
```

`obs.X` holds the signals of each run over the window, `obs.y` the labels and `obs.folds` the train and test indices of each fold. The method is whatever maps `obs.X` to a prediction; the task fixes only the data and the scoring. A rule that predicts an oscillation from the swing in T over the last 50 min of the window:

```python
swing = np.ptp(obs.X[:, obs.t >= 150.0, 0], axis=1)
prediction = np.where(swing > 0.5, "oscillating", "settled")
print(t.score(obs.y, prediction))
```

```
{'balanced_accuracy': 1.0, 'found': 8, 'false_alarms': 0}
```

A method that learns from the data is fitted on the training indices of each fold in `obs.folds` and scored on the test indices. `plantbench datasets` and `card` report the protocol of a dataset generated from a specification that carries a task, as `pb-protocol:<digest>` beside its `pb-spec`.

## Further reading

- [User guide](../user-guide.md): every section of a configuration, disturbances, datasets, tasks and reports in full.
- [Cases](../cases/index.md): the card of each case, with its structures, disturbances and reference results.
- [Examples](../examples.md): short scripts run by the test suite.
- [Contributing a case](contributing.md): a case of your own, from the template to the library.
- [API reference](../api.md).
