# User guide

## Installation

Python 3.11 or later, with uv, pixi or pip ([Installation](index.md#installation)). The library depends on NumPy and SciPy. The `dev` extra adds pytest, Matplotlib, ruff and pre-commit, and the `study` extra adds Matplotlib, scikit-learn, PaCMAP and HDBSCAN for the studies.

## Cases

```python
import plantbench as pb

pb.list_cases()                 # ['reactor_separator_recycle', 'jacketed_cstr']
case = pb.load_case("jacketed_cstr")
print(case.summary)
```

From the command line, `plantbench list` and `plantbench describe reactor_separator_recycle`. The case cards in `plantbench/cases/<case>/README.md` document each case's states, inputs, options, structures, disturbances, measurements and reference results.

| Case | States | Description |
|---|---|---|
| `reactor_separator_recycle` | 134 (base plant) | reactor–separator–recycle plant with heat integration; frozen |
| `jacketed_cstr` | 3 | the reactor of `reactor_separator_recycle` on its own, open-loop unstable |

A case outside the library is used the same way once plantbench finds it: from an installed package that declares it as an entry point, or from a function decorated with `@pb.case` in a script. `plantbench list` shows where each case comes from. [Developing a case](developing-a-case.md) describes both, and `plantbench new-case` starts one.

## Running a configuration

```python
case = pb.load_case("jacketed_cstr")
config = case.config("temperature PI",
                     disturbance={"name": "feed_temperature", "value": 340.0, "t": 10.0})
traj = pb.run(case, config, t_end=600.0, dt=1.0)

traj.t                  # time, min
traj.x                  # states, (n_states, n_times)
traj.state("T")         # one state by name
traj.u["Fj"]            # every input, and derived signals such as `reactor_separator_recycle`'s recycle
traj.y["T"]             # the case's named measurements
traj.wall_time, traj.nfev
traj.window(300.0, 600.0)
traj.to_npz("run.npz")
```

`pb.run(case)` with no configuration runs the reference plant. `pb.build(case, config)` returns the configured plant without simulating it: `setup.design`, `setup.structure`, `setup.disturbance`, and `setup.spectrum()` for the closed-loop eigenvalues.

## Configurations

A configuration has these sections; every one except the structure may be omitted.

| Section | Contents | Example |
|---|---|---|
| `structure` | a structure the case provides | `"effluent fixed + cascade"` |
| `options` | the case's own choices | `{"network": "D3"}` |
| `params` | parameter overrides by dotted path | `{"reactor.U": 300.0}` |
| `setpoints` | loop name to set-point | `{"reactor T": 365.0}` |
| `setpoint_steps` | loop name to (time, change) pairs | `{"reactor T": [(30.0, 1.0)]}` |
| `tuning` | loop name to Kc, tau_I, lo, hi | `{"reactor T": {"Kc": -0.5}}` |
| `instruments` | loop name to `Instrument` arguments | `{"reactor C_A": {"deadtime": 5.0}}` |
| `actuators` | loop name to `Actuator` arguments | `{"reactor T": {"tau": 0.5}}` |
| `disturbance` | a named disturbance and its arguments, or a list of them | `{"name": "throughput", "fraction": 0.1}` |

`case.config(...)` fills in the case's default options and rejects a structure or option the case does not have. Every name is checked when the plant is built: an unknown loop, parameter path or disturbance raises instead of being ignored. `config.to_dict()` and `Config.from_dict()` convert to and from plain data, and `config.key()` is a stable hash of it.

## Set-points and tuning

Loops start at rest at the design point: `bias_from_design` sets each bias to the design value of its manipulated variable and each set-point to the design value of its measurement. A set-point in `setpoints` replaces the design value from time zero. A change during the run goes in `setpoint_steps`. A cascade primary writes its secondary's set-point, so setting the secondary's has no effect while the primary is present.

```python
config = case.config("temperature PI",
                     tuning={"reactor T": {"Kc": -0.5, "tau_I": 20.0}},
                     setpoint_steps={"reactor T": [(10.0, 2.0)]})
```

The same operations are available on a structure directly: `control.with_setpoints`, `with_setpoint_steps`, `with_tuning` and `with_instruments`.

## Non-idealities

An `Instrument` adds a first-order sensor lag (`tau`) and a deadtime (`deadtime`), represented by a Padé approximation of order 1 or 2 or by a chain of `order` lags (`method="lags"`). An `Actuator` adds a first-order valve travel (`tau`) and a slew limit (`rate_limit`). Noise, bias, quantization, stiction and deadband are declared and raise, for the reasons given in `plantbench/core/instruments.py`.

```python
config = pb.load_case("reactor_separator_recycle").config(
    instruments={"reactor C_A": {"deadtime": 5.0, "order": 2}},
    actuators={"reactor T": {"tau": 0.5, "rate_limit": 2.0}})
```

An unconfigured element adds no states, so a configuration without non-idealities reproduces the reference results exactly.

## Disturbances

A disturbance is a function `d(t, u0) -> u` returning the inputs in force at time t. Every case provides the generic `step` and `ramp`, which act on any input:

```python
{"name": "step", "field": "CA0", "fraction": 0.05, "t": 10.0}
{"name": "ramp", "field": "F0", "value": 3.5, "t": 10.0, "duration": 60.0}
```

Cases add named disturbances where the change is not a change of one input, such as `reactor_separator_recycle`'s `throughput`, which acts through whichever input sets production in the chosen structure. A list of disturbances is applied in order.

A disturbance that changes the inputs discontinuously declares the times it does so as an attribute `times`; `step`, `ramp` and `combine` set it. `plantbench.run` restarts the integration at those times and at every scheduled set-point change, so the solver never steps across one. A disturbance written by hand should set `times` as well. `control.simulate`, which the `reactor_separator_recycle` study scripts use, integrates in one call, as the reference results were produced.

## Linear analysis

```python
from plantbench.core import control as ctl

setup = pb.build("reactor_separator_recycle", pb.load_case("reactor_separator_recycle").config(options={"network": "D4"}))
ev = setup.spectrum()                  # closed-loop eigenvalues at the design point
ctl.damping_ratio(ev)                  # 0.013 for D4
```

The Jacobian is taken by central differences on the closed-loop right-hand side, including the integral, instrument and actuator states. One structural zero per ratio station is removed.

## The batched Jacobian and GPUs

LSODA forms the Jacobian of the closed loop by forward differences, one evaluation of the right-hand side per state. On a plant with several hundred states and an equation of state inside its right-hand side, these evaluations are almost the whole cost of a run: on `ngl_deethanizer`, 97 % of them. A case whose right-hand side and measures accept a batch of states can instead have the whole Jacobian evaluated as one batch and handed to LSODA:

```python
traj = pb.run(case, config, t_end=60.0, jacobian="batched")                 # on the host
traj = pb.run(case, config, t_end=60.0, jacobian="batched", device="cuda")  # on a GPU
```

and from the command line, `plantbench run ngl_deethanizer --jacobian batched` and `plantbench generate spec.toml --jacobian batched --device cuda`. The default, `jacobian="internal"`, is LSODA's own Jacobian, with which every published result was produced. The batched Jacobian agrees with it to the accuracy of the differences, so a trajectory agrees to the solver's tolerance but not to the last bit; a dataset records the option on each run made with it, under `solver`, and the specification's fingerprint does not include it. `setup.spectrum()` and `ctl.closed_loop_spectrum(..., jacobian="batched")` use the same batch for the linearization.

On the host, a 20-minute run of `ngl_deethanizer` with a step in the feed's ethane takes 10 s with the batched Jacobian against 59 s with LSODA's own, and its states differ by at most 5e-12 relative; `benchmarks/jacobian.py` measures this for each case and device, and `benchmarks/scaling.py` the time of one batched evaluation against the size of the batch. `ngl_deethanizer` and `ngl_demethanizer` declare a batched right-hand side; `reactor_separator_recycle` and `jacketed_cstr` do not.

On a node with one NVIDIA A100, the same run takes 157 s with LSODA's own Jacobian, 27 s with the batched Jacobian on the host and 20 s with it on the GPU. The GPU gains little on one run: between Jacobians, LSODA advances the solution one state at a time on the host, and one Jacobian of `ngl_deethanizer`, 361 states, is too small a batch to occupy the device, which a job of single runs kept busy 1 % of the time. A single run with the batched Jacobian therefore belongs on a CPU node. The device pays when a batch holds many Jacobians: 64 of them, 23,104 states, take 0.55 s against 0.24 s for one, 24 µs a state against 1.5 ms on the host.

`device="cuda"` needs an NVIDIA GPU with fast double precision (A100, H100) and CuPy: `pip install 'plantbench[gpu]'` with a CUDA 12 toolkit on the machine, or the pixi environment `cuda` (linux-64), which takes CuPy and the CUDA runtime from conda-forge and needs only the driver. Everything is evaluated in double precision, since the integrations run at a relative tolerance of 1e-8.

## Generating a dataset

A specification is a TOML file:

```toml
[study]
name = "example"
case = "jacketed_cstr"
seed = 1

[base]
structure = "temperature PI"

[[variants]]                       # partial configurations, each merged into the base
structure = "composition cascade"

[sweep]                            # Cartesian product over dotted paths
"disturbance" = [{name = "feed_temperature", change = 2.0, t = 10.0},
                 {name = "feed_temperature", change = -4.0, t = 10.0}]

[sample]                           # a Latin hypercube of n points per combination
n = 8
"tuning.reactor T.Kc" = [-2.0, -0.2]

[run]
t_end = 600.0
dt = 1.0
wall_budget = 60.0                 # seconds per run; longer runs are recorded as timeouts
features = ["spectrum"]            # damping ratio and rightmost eigenvalue per run
```

A feature is a quantity recorded beside each run, computed from the configured plant at its
design point rather than from the trajectory. It therefore costs nothing beyond the design
solve, and it is recorded even when the run fails or is too short to say anything. Every
case provides `spectrum`, the damping ratio, rightmost eigenvalue and number of unstable
modes of the closed loop linearized at the design point. A case may add its own: `reactor_separator_recycle`
provides `economics`, the operating profit, utility cost, energy and total global warming
potential, and the heat recovered, all at the design operating point. A name the case does
not provide is refused before any run is made, and a feature that raises is recorded as
`<name>_error` rather than losing the run.

A dotted path names a configuration section and then its keys: `"tuning.reactor T.Kc"` is the gain of the loop named `reactor T`, and `"params.reactor.U"` is the parameter override `reactor.U`. A path may name a whole section, with whole sections as its values.

```
plantbench generate spec.toml --workers 8
```

or `plantbench.datagen.generate("spec.toml", workers=8)`. A script that generates with more than one worker needs an `if __name__ == "__main__":` guard, because each worker process imports the script it was started from. The dataset is written to `data/<name>/` unless `--out` is given:

| File | Contents |
|---|---|
| `spec.toml` | the specification |
| `manifest.json` | library version, git commit and dirty flag, package versions, start time |
| `runs.jsonl` | one record per run: id, status, wall time, solver evaluations, features, configuration |
| `index.csv` | the same records, configurations flattened to dotted columns |
| `runs/<run_id>.npz` | the trajectory of each run that finished |

Generation resumes where it stopped. Runs already recorded are skipped, including those that failed, since a failure recurs. An interrupt, or the SIGTERM a scheduler sends to end a job, cancels the runs not yet started; those already running finish, and a resume runs whatever was not recorded. A directory holds one specification, and a changed specification needs a new directory. One generation writes to a directory at a time: a second one started on it is refused. The manifest keeps the specification's fingerprint, the canonical text of everything that decides which runs exist, so a dataset always says which protocol its runs are of.

The commit is recorded only when the library runs from a checkout of its repository; an installed copy records none. The dirty flag looks at the code alone, `plantbench/` and `studies/` without their results and figures, so an edited manuscript or a results file an analysis has just written does not mark a dataset or a result dirty. A dataset built by `run_configurations` from a list of configurations writes a manifest of the same kind, with no specification; a directory of runs with no manifest at all is listed by `plantbench datasets` with its provenance unknown.

## Reading a dataset

```python
from plantbench import datagen

records = datagen.load_records("data/example")
ok = [r for r in records if r["status"] == "ok"]
traj = datagen.load_run("data/example", ok[0]["run_id"])
```

`index.csv` loads directly into pandas or a spreadsheet. `examples/04_generate_dataset.py` generates and reads a 48-run dataset in a few seconds.

## Tasks

A specification says which runs to generate. A task says what is to be predicted from them and how an answer is scored, so that two methods reported against one specification are comparable. It is a `[task]` table in the same file:

```toml
[task]
target   = "regimes.outcome"      # a registered labeler
signals  = ["reactor T", "Fj"]    # names, or the groups "measured" and "manipulated"
window   = [0.0, 130.0]           # min, inclusive at both ends
split    = {folds = 5, stratified = true, seed = 0}
metrics  = ["balanced_accuracy", "found", "false_alarms"]
positive = "limit cycle"          # the class the last two metrics count
```

```python
import studies.regimes.dataset   # registers the labeler the task names, "regimes.outcome"
from plantbench import task

t = task.Task.of("studies/regimes/spec.toml")
obs = t.data("data/regimes")   # obs.X (runs, times, signals), obs.y, obs.folds, obs.t
scores = t.score(obs.y, predictions)
```

A task names its labeler rather than containing it, so the module that registers the labeler is imported first; `Task.of` refuses a target that is not registered and lists those that are.

The task fixes the observations, the labels, the folds and the scoring. What happens between the signals and the prediction — averaging, normalization, reduction, the classifier — is the method, and is deliberately not part of the protocol. Runs are taken in order of run id, so nothing downstream depends on the order a parallel generation finished in, and a group in `signals` is expanded against the case, so an explicit list is the stronger protocol.

A label is read by a function registered under a name, so a study keeps its criterion where it was written and the library holds no domain knowledge:

```python
@task.labeler("regimes.outcome")
def _outcome(tr):
    ...
```

`task.labelers()` and `task.metrics()` list what is registered. The metrics provided are `balanced_accuracy`, `accuracy`, `found` and `false_alarms`; a new one is registered the same way with `@task.metric`, and takes `(y, prediction, positive)`.

A task does not decide which runs exist, so it leaves `Spec.fingerprint` and a dataset's specification digest alone. `Spec.protocol_digest` names the specification and the task together, and that is what a method reports against. A dataset generated from a specification that carries a task records the protocol in its manifest, and `plantbench datasets` then shows it in a column of its own.

Tasks need scikit-learn, which the rest of the library does not. It is in the `study` extra: `uv sync --extra dev --extra study`, `pixi install -e study`, or `pip install -e '.[dev,study]'`.

## What is on disk

```
plantbench datasets
```

lists every dataset under `data/`, one row each: the directory, the case, the identifier of the specification the runs are of, the commit that generated them and whether that tree was clean, the runs by status, the integration time they cost, and whether the specification left in the directory still describes them. A digest is the first twelve hex characters of the SHA-256 of the fingerprint, so two datasets of one protocol carry the same digest and a changed protocol carries a different one. It is printed with a prefix naming what was hashed, `pb-spec:<digest>` for the specification without its task and `pb-protocol:<digest>` for the specification with it, so that it is not read as a commit. `plantbench.datasets.summary("data/example")` returns the same record as a dictionary.

## Reporting a dataset

```
plantbench card data/example --format latex
```

prints what a paper reports about a dataset: the study and case, the version and commit of the code that generated it, the specification file with its `pb-spec` identifier, the protocol with its `pb-protocol` identifier when the specification carries a task, how the runs ended and what they cost, and the Python, NumPy and SciPy versions. `--format` takes `text` (the default), `md` for a README or supplement, `latex` for the rows of a two-column table, and `json`, which adds the full SHA-256 of each fingerprint. Every format carries a sentence defining the identifiers, which in a paper belongs in the table caption. A card also warns when the report would be weaker than it looks: a dataset generated from uncommitted code, one without a manifest, or one whose kept specification no longer describes its runs.

A reader checks the identifiers against the specification file, or against a dataset, with

```
plantbench verify data/example/spec.toml pb-spec:1a2b3c4d5e6f pb-protocol:6f5e4d3c2b1a
```

which prints `match` or `MISMATCH` for each identifier and exits non-zero on any mismatch. A bare digest is checked against both kinds, and a digest longer than twelve characters is checked as a prefix of the full SHA-256. `plantbench.report.card` and `plantbench.report.verify` return the same as data.

## Comparing two datasets

```
plantbench compare data/example-old data/example
```

reports what two generations disagree on: manifest fields, runs recorded in one only, runs whose status differs, recorded features that moved, runs that took a different path through the integrator, and trajectories that differ, with the largest relative deviation of each. Wall time is reported on a line of its own rather than counted as a difference, being a property of the machine and the moment. `--tol` sets the relative tolerance, `1e-9` by default; `--no-trajectories` compares the records alone; `--limit` caps the lines printed per section.

The same command takes two results files:

```
plantbench compare old/results.txt new/results.txt
```

Lines are paired by their text with the numbers masked out, so that two reports of the same quantities align, and paired lines are then compared number by number against the tolerance. Lines that differ in their text are listed apart from those whose numbers moved.

## Recording what a result was computed from

```python
from plantbench import datasets

mark = datasets.stamp("data/example")          # the dataset, and the tree reading it
print("\n".join(datasets.stamp_lines(mark)))   # as the head of a results file
datasets.check_stamp(mark, "data/other")       # why it is not of that dataset, or None
```

A stamp holds the dataset, the digest of its specification, the commit and dirty flag of the tree that generated it, the number of completed runs, and the commit and package versions of the tree that read it, including scikit-learn, PaCMAP and HDBSCAN when they are installed, since embeddings and clusterings depend on them. None of it is a clock reading, so a result computed twice from one dataset carries the same stamp and the file stays comparable byte for byte. `studies/regimes/analysis.py` writes one into `results_regimes.txt` and `analysis.npz`, and `studies/regimes/figures.py` checks it before drawing, so one dataset's figures cannot be drawn from another dataset's analysis.

## Examples

| Script | Shows |
|---|---|
| `examples/01_run_case.py` | a run of the reference plant and the same disturbance in open loop |
| `examples/02_setpoints_and_tuning.py` | set-point steps under three tunings, linear against nonlinear |
| `examples/03_analyzer_deadtime.py` | analyzer deadtime on `reactor_separator_recycle`'s composition loop |
| `examples/04_generate_dataset.py` | a dataset from `04_dataset.toml`, generated and read back |
