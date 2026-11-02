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
traj.c["reactor T.SP"]  # each loop's set-point, measurement and output (Controller signals)
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
| `tuning` | loop name to Kc, tau_I, lo, hi, tau_D, alpha | `{"reactor T": {"Kc": -0.5}}` |
| `instruments` | loop name to `Instrument` arguments | `{"reactor C_A": {"deadtime": 5.0}}` |
| `actuators` | loop name to `Actuator` arguments | `{"reactor T": {"tau": 0.5}}` |
| `disturbance` | a named disturbance and its arguments, or a list of them | `{"name": "throughput", "fraction": 0.1}` |
| `mismatch` | parameter overrides applied to the plant only, by dotted path | `{"reactor.U": 300.0}` |
| `x0` | state name to its value at the start | `{"T": 367.4}` |
| `hold` | input name to a constant value; the loops that write it are removed | `{"Fj": 0.5}` |
| `noise` | a seed, and measurement name to the standard deviation of its noise in a stepped run | `{"seed": 0, "std": {"T": 0.5}}` |

`params` and `mismatch` change the same parameters in two ways. Under `params` the design is solved for the changed parameters, and the loops are biased from that design, so the regulatory layer moves with the plant. Under `mismatch` the design is solved without them; the run starts at that design, with its biases and set-points, and only the plant's right-hand side and measures take the changed parameters. This is model-plant mismatch, and it needs no design for the changed parameters, which may have none. `setup.nominal` is then the design the loops were biased from, and `setup.spectrum()` raises, since the start is not a steady state of the plant. A mismatch that changes neither the right-hand side nor a measurement at the design point is refused: a parameter that the design turns into an input, such as a feed condition, stays at the design and is changed by a disturbance instead. The cases whose models read the parameters they are passed support `mismatch` (`jacketed_cstr`, `reactor_separator_recycle` and the template); the NGL cases refuse it, and `reactor_separator_recycle` refuses the column pressures, on which its tray temperature loop and its heat-exchanger networks are built.

A run starts at the design point unless `x0` names other values for some of its states, by the names `case.state_names(design)` gives them. Instrument and derivative-filter states then start at rest at that state, reporting its true measurement. Actuators start at the design position of their valve and integrals at zero, so each loop begins with an error and only its proportional action. `setup.x0` is the start.

`hold` fixes a scalar input for the whole run. Every loop or ratio station that writes the input is removed, and so is a cascade primary whose secondary is removed; `setup.held` lists them. A ratio station that follows a held input stays. A held input changes the ownership of the structure, so the configuration is refused when another section names a removed loop, when a disturbance changes the held input at the start or at a time it declares, and for an input that is not a scalar. An input held at a value other than its design value moves the plant from its design point, and `setup.spectrum()` then raises, as under `mismatch`. `x0` and `hold` together describe the region-of-attraction grids and recoverability checks of a study as a `[sweep]`, for example over `"x0.T"` and `"hold.Fj"`.

`case.config(...)` fills in the case's default options and rejects a structure or option the case does not have. Every name is checked when the plant is built: an unknown loop, parameter path or disturbance raises instead of being ignored. `config.to_dict()` and `Config.from_dict()` convert to and from plain data, and `config.key()` is a stable hash of it.

## Set-points and tuning

Loops start at rest at the design point: `bias_from_design` sets each bias to the design value of its manipulated variable and each set-point to the design value of its measurement. A set-point in `setpoints` replaces the design value from time zero. A change during the run goes in `setpoint_steps`. A cascade primary writes its secondary's set-point, so setting the secondary's has no effect while the primary is present.

```python
config = case.config("temperature PI",
                     tuning={"reactor T": {"Kc": -0.5, "tau_I": 20.0}},
                     setpoint_steps={"reactor T": [(10.0, 2.0)]})
```

The same operations are available on a structure directly: `control.with_setpoints`, `with_setpoint_steps`, `with_tuning` and `with_instruments`.

Every loop is PI unless it is given a derivative time. With `tau_D` above zero, in minutes, a loop is PID in the ideal (ISA standard) form, with `Kc` multiplying every term and the derivative acting on the measurement through a first-order filter of time constant `alpha * tau_D` (`alpha` 0.1 by default):

$$
\mathrm{OP} = \mathrm{bias} + K_c\left(e + \frac{1}{\tau_I}\int e\,dt - \tau_D \frac{dz}{dt}\right), \qquad \alpha\tau_D \frac{dz}{dt} = \mathrm{PV} - z .
$$

The derivative acts on the measurement rather than the error, so a set-point step moves the output by the proportional term alone. The filter state `z` is one more augmented state, which starts at the measurement, so a run starts at rest. No case tunes a loop with derivative action, and a loop without it adds no state, so every reference result is unchanged.

```python
config = case.config("temperature PI", tuning={"reactor T": {"tau_D": 2.0}})
```

## Non-idealities

An `Instrument` adds a first-order sensor lag (`tau`) and a deadtime (`deadtime`), represented by a Padé approximation of order 1 or 2 or by a chain of `order` lags (`method="lags"`). An `Actuator` adds a first-order valve travel (`tau`) and a slew limit (`rate_limit`). Noise, bias, quantization, stiction and deadband are declared and raise, for the reasons given in `plantbench/core/instruments.py`.

```python
config = pb.load_case("reactor_separator_recycle").config(
    instruments={"reactor C_A": {"deadtime": 5.0, "order": 2}},
    actuators={"reactor T": {"tau": 0.5, "rate_limit": 2.0}})
```

An unconfigured element adds no states, so a configuration without non-idealities reproduces the reference results exactly.

## Controller signals

A run records the signals of each loop in `traj.c`, under the loop's name:

| Key | Signal |
|---|---|
| `<loop>.SP` | the set-point in force, after any set-point step and any primary's write |
| `<loop>.PV` | the measurement the controller sees, after its instrument |
| `<loop>.OP` | the output it commands, before its actuator; a primary's is its secondary's set-point |
| `<loop>.PV%`, `<loop>.OP%` | the same in percent of span, where the loop declares one |

A ratio station records `<station>.OP` only. The true measurement is in `traj.y`, and the input the plant receives, after the actuator, in `traj.u`. Where the HYSYS tags and these signals differ in name, the loop is named after its measurement: TIC-100 of the deethanizer is `stage 28 temperature`.

The span of a measurement is the loop's `pv_range`, (PV_min, PV_max) in its units, and the span of an output is (`lo`, `hi`). The NGL cases take theirs from the HYSYS controllers, and the reactor cases declare spans of their own choosing, marked as such in their definitions. A span only scales the recorded signals: the measurement is not clamped to it. `control.percent_gain(loop)` gives the gain in percent of output span per percent of measurement span, the form HYSYS states a tuning in, negative for a direct-acting loop.

## Disturbances

A disturbance is a function `d(t, u0) -> u` returning the inputs in force at time t. Every case provides the generic `step` and `ramp`, which act on any input:

```python
{"name": "step", "field": "CA0", "fraction": 0.05, "t": 10.0}
{"name": "ramp", "field": "F0", "value": 3.5, "t": 10.0, "duration": 60.0}
```

Cases add named disturbances where the change is not a change of one input, such as `reactor_separator_recycle`'s `throughput`, which acts through whichever input sets production in the chosen structure. A list of disturbances is applied in order, each to the inputs the previous ones produced, so two fractional steps of one input compound: +5% and then +10% of `CA0` give +15.5%.

A `sequence` moves an input through levels that do not compound, each relative to the nominal inputs. `of` names the disturbance the sequence is made of, `step` by default or any the case provides; `level` names the argument that takes each level, `fraction` by default; the other arguments are passed to every part, with its time:

```python
{"name": "sequence", "field": "CA0", "times": [10.0, 40.0], "levels": [0.05, 0.10]}
{"name": "sequence", "of": "feed_temperature", "level": "change",
 "times": [50.0, 150.0, 300.0], "levels": [-4.0, 6.0, 0.0]}
```

A `random_sequence` draws the times and the levels from a seed, as identification data are excited: the first switch at `start`, each next one after a dwell drawn uniformly in `dwell`, the last before `end`, and each level uniform in [`low`, `high`] (`kind = "uniform"`, an amplitude-modulated pseudo-random binary signal) or one of the two bounds (`kind = "binary"`):

```python
{"name": "random_sequence", "of": "throughput", "seed": 3, "low": -0.12, "high": 0.15,
 "dwell": [100.0, 400.0], "start": 30.0, "end": 1500.0}
```

The seed is part of the configuration, so a sweep over `"disturbance.seed"` gives a reproducible set of excitations, each run identified by its seed. Every switching time is declared, and the integration restarts there. `core.disturbances.sequence`, `step_sequence` and `random_levels` build the same disturbances in code. The setup's disturbance carries the levels it was built with as `levels`.

A disturbance that changes the inputs discontinuously declares the times it does so as an attribute `times`; `step`, `ramp`, `combine` and the sequences set it. `plantbench.run` restarts the integration at those times and at every scheduled set-point change, so the solver never steps across one. A disturbance written by hand should set `times` as well. `control.simulate`, which the `reactor_separator_recycle` study scripts use, integrates in one call, as the reference results were produced.

## Stepping a plant

A controller outside the library, a policy that acts once per control interval, needs the plant advanced one interval at a time rather than a whole trajectory. `pb.Stepper` holds a configured plant, its time and its closed-loop state between steps:

```python
case = pb.load_case("jacketed_cstr")
setup = pb.build(case, case.config("composition cascade", disturbance={"name": "feed_temperature", "value": 340.0}))
stepper = pb.Stepper(setup, dt=0.5, inputs=["Fj"])   # the caller owns the coolant flow
obs = stepper.observe()                              # t, x, u, y, c at the start
for k in range(400):
    obs = stepper.step({"Fj": policy(obs.y)})         # held over [t, t + 0.5]
```

The caller owns inputs, the set-points of loops, or both, and each action names exactly what is owned. An owned input is removed from the structure with every element that writes it, as a configuration's `hold` removes it: here both loops of the cascade, which `stepper.displaced` returns as `Loop` objects with their tunings. The action sets the input before the remaining loops are evaluated, so a ratio station that follows it follows the action. An owned set-point leaves its loop in place. Ownership is refused for an input the configuration holds or a disturbance changes, for the set-point of a loop that is removed or whose set-point a cascade primary writes, and for a set-point with a schedule.

The action is held over the interval (zero-order hold), and the integration restarts at each breakpoint of the disturbance and the set-point schedules inside it. Each `step` returns a `Step`: the time, the plant state, the inputs, the case's measurements, the signals of the remaining loops and the right-hand-side evaluations the interval took. `step()` without an action holds the nominal one, `stepper.nominal_action()`, which leaves the plant as configured; with nothing owned, a stepped run is the run of `pb.run` to the solver's tolerance. The stepper starts where the configuration starts the run, at `setup.x0`, and `reset()` returns it there. An interval whose integration fails raises `StepFailed` and leaves the stepper where the interval began.

A configuration's `noise` section gives the measurements a sampled instrument reports with white noise: at each sample, the start and the end of every step, each named measurement receives a draw of zero mean and the given standard deviation. The draws come from a generator of their own, seeded by the section's `seed`, so the noise and a scenario's random disturbance (`random_sequence`) vary independently. `Step.y` holds the measured values and `Step.y_true` the true ones, which the loops that remain see as well, as in a continuous run; the noise never enters the right-hand side. `observe()` returns the current sample without drawing again, and `reset()` seeds the generator again, from the section or from `reset(noise_seed=...)`. A continuous run has no samples, so `pb.run`, ensembles and dataset generation refuse a configuration with `noise`; the noise of an `Instrument` (`noise_std`) remains unimplemented for the same reason.

A stepper sees the plant at its samples only, and a measurement can leave a limit between two samples and return before the next. `extrema` names measurements whose smallest and largest true values over each interval a step reports, in `Step.y_min` and `Step.y_max`:

```python
stepper = pb.Stepper(setup, dt=1.0, extrema=["T"], refine=10)
st = stepper.step()
st.y_min["T"], st.y_max["T"]
```

The values are taken on the solver's dense output at `refine` equal parts of each segment of the interval, the segments being the parts between breakpoints, and on both sides of each breakpoint, where an input that steps makes a measurement that reads it jump. The interval's ends are included. The refinement changes nothing in the integration. The values are numerical estimates that converge as `refine` grows, not guaranteed bounds of the continuous-time signal: an excursion shorter than one part can still be missed.

`step_budget` limits the wall-clock seconds of one step, and `episode_budget` those of all the steps since the start or the last `reset()`; the caller's own time between steps does not count. A step that stalls inside LSODA cannot be interrupted from Python, since the solver is compiled and a signal reaches only the main thread of a POSIX process. A stepper with either budget therefore integrates in a worker process of its own, spawned once and kept until `close()` or the end of a `with` block. When a step runs over, the worker is terminated and the step raises `control.WallTimeExceeded`. The stepper is then invalid, and a new one is built to go on. This works from any thread and on any platform:

```python
with pb.Stepper(setup, dt=0.5, inputs=["Fj"], step_budget=10.0, episode_budget=600.0) as stepper:
    for k in range(400):
        obs = stepper.step({"Fj": policy(obs.y)})
```

The worker builds the plant again from the case's module and the configuration, which takes as long as `pb.build` does, within `startup_budget` seconds if one is given. A case must therefore be importable in a fresh process: a built-in case, an installed one or one defined in an importable module. A case defined in a script, or changed after import, is refused a budget and steps without one. A setup changed after `build` is refused as well, since the worker would step another plant.

## Limits and outcomes

A case declares the span, the alarm and trip limits, and the open-loop steady states of the measurements it bounds, `case.limits(design)`, a `core.limits.Limits` per measurement name. Each number is tagged as the case cards tag theirs: `"source"` when the case's reference gives it, `"choice"` when the case decides it, with the reason in `why`:

```python
case = pb.load_case("jacketed_cstr")
lim = case.limits(pb.build(case).design)
lim["T"].trip_hi            # Limit(value=372.0, basis='choice', why='10 K above the design point ...')
lim["T"].steady_states      # (338.22..., 362.40..., 377.08...), low-temperature branch first
```

The spans are those the loops express their signals in percent of (`pv_range`), from one definition. The steady states are those of the plant at the design's inputs, so a `hold` or a `mismatch` moves them. The case cards list every limit.

Whether a limit ends a run is a decision of the study, so it is data as well. A `Rule` names an outcome, a measurement, a side of a level and a dwell, the time the measurement must stay beyond the level without a break. `limit_rules` makes rules from a case's limits, and a `Monitor` applies them to the samples of a run:

```python
from plantbench.core.limits import Monitor, Rule, check, limit_rules

rules = [*limit_rules(lim, "trip"),                                   # T above 372 K
         Rule("extinction", "T", "below", lim["T"].alarm_lo.value, dwell=20.0)]
monitor = Monitor(rules)
stepper = pb.Stepper(setup, dt=0.5, inputs=["Fj"], extrema=["T"])
obs = stepper.observe()
monitor.step(obs)
while monitor.first is None and obs.t < 200.0:
    obs = stepper.step({"Fj": policy(obs.y)})
    monitor.step(obs)
monitor.first               # Violation(rule, t, value, since, sampled), or None
check(trajectory, rules)    # the same over the grid of a `Trajectory`
```

The monitor reads true values, `Step.y_true`, not the measured ones that a configuration's `noise` perturbs. It records the first violation of each rule in `violations` and the earliest in `first`. The time beyond a level is counted from the last sample within it, since the crossing lies after that sample, so a sampled check reports a dwell no later than the plant reaches it. A step that reports the extremes of its interval (`extrema`) adds two things: an interval whose extremes show the measurement back within the level restarts the count at the interval's start, and a rule without a dwell is met between samples when an extreme lies beyond its level while the samples do not. Such a violation has `sampled = False`. It is an estimate made on the refined dense output, and the crossing lies somewhere between `since` and `t`.

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

On the host, a 20-minute run of `ngl_deethanizer` with a step in the feed's ethane takes 8 s with the batched Jacobian against 48 s with LSODA's own, and its states differ by at most 1e-10 relative; `benchmarks/jacobian.py` measures this for each case and device, and `benchmarks/scaling.py` the time of one batched evaluation against the size of the batch. `ngl_deethanizer` and `ngl_demethanizer` declare a batched right-hand side; `reactor_separator_recycle` and `jacketed_cstr` do not.

On a node with one NVIDIA A100 (80 GB, PCIe), the same run takes 125 s with LSODA's own Jacobian, 26 s with the batched Jacobian on the host and 12 s with it on the GPU, whose states differ from the host's by at most 5e-11 relative. On a GPU, the cost of one evaluation is set by the number of array operations it issues, each a kernel launched from the host at a nearly fixed cost, rather than by the number of states. The state of a phase, its fugacities and the Rachford–Rice solve therefore run as one CUDA kernel each (`plantbench.units._kernels`), and one evaluation of either NGL case costs about the same from 16 to more than 1,000 states: 0.034 s for `ngl_deethanizer` and 0.27 to 0.30 s for `ngl_demethanizer`. A single run gains little from this, since between Jacobians LSODA advances the solution one state at a time; a single run belongs on a CPU node. The device pays when a batch holds many Jacobians: 64 of `ngl_deethanizer`, 23,104 states, take 0.071 s, 3.1 µs a state against 2.0 ms on the host, and 64 of `ngl_demethanizer` take 0.44 s.

`device="cuda"` needs an NVIDIA GPU with fast double precision (A100, H100) and CuPy: `pip install 'plantbench[gpu]'` with a CUDA 12 toolkit on the machine, or the pixi environment `cuda` (linux-64), which takes CuPy and the CUDA runtime from conda-forge and needs only the driver. Everything is evaluated in double precision, since the integrations run at a relative tolerance of 1e-8.

## Ensembles in lockstep

Many configurations of one design can be integrated together, as one ensemble whose members advance in lockstep:

```python
results = pb.run_ensemble("ngl_deethanizer", configs, t_end=60.0, dt=1.0, device="cuda")
traj = results[0].trajectory        # or results[0].error, for a member that failed
```

and from the command line, `plantbench generate spec.toml --ensemble 64 --device cuda`. Members share a design when they have the same structure, options and parameters, and they are run together when their control structures also have the same state layout; across members, the tunings, set-points, set-point steps, instrument and actuator parameters and disturbances may all differ. Each step evaluates the right-hand sides of every member still running as one batch, and the Jacobians of those that need one as another, so that on a GPU, where an evaluation costs about the same from one state to a few thousand, an ensemble of many members costs little more per step than one.

The integrator is the Rosenbrock method Rodas5P (Steinebach, 2023), of order 5 with an embedded estimate of order 4, rather than LSODA: a Rosenbrock method does the same, fixed work every step, an exact Jacobian and eight solves with it, which is what lets members with different step sizes share every evaluation. Each member has its own time, step size and error control, and its steps end on its own breakpoints and on the output grid. A member that fails stops, and the others go on; `wall_budget` applies to the ensemble, and `generate` gives an ensemble the specification's budget, which is per run, times its number of members. The trajectories agree with LSODA's to the tolerance, not to the last bit: over a 20-minute run of `ngl_deethanizer`, to within half a tolerance (|Δx| ≤ 0.46 (atol + rtol |x|)), and against a run at a relative tolerance of 1e-11 the ensemble's states are the closer of the two. A dataset records `solver = {"ensemble": M, "device": ...}` on each run made this way, with `ensemble_size` and `ensemble_wall_time` beside the run's share of the wall time.

On the host an ensemble gains nothing, since its evaluations cost in proportion to their size: a 20-minute run of `ngl_deethanizer` takes 25 s alone or as one of four, against 8 s with the batched Jacobian and LSODA, which remains the choice there. On the A100 node above, the same run takes 38 s as an ensemble of one, and 12 s, 3.1 s and 0.86 s a member in ensembles of 4, 16 and 64: 4,200 runs an hour, about 30 times the 140 an hour of LSODA with the batched Jacobian on one core of that node. The members' states agree with LSODA's to within 0.28 tolerances on either device. `benchmarks/ensemble.py` measures the throughput against the number of members on each device.

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

or `plantbench.datagen.generate("spec.toml", workers=8)`. A script that generates with more than one worker needs an `if __name__ == "__main__":` guard, because each worker process imports the script it was started from. For a case with a batched right-hand side, `--ensemble M` runs the configurations in one process as ensembles of up to M members instead (Ensembles in lockstep); it cannot be combined with more than one worker. The dataset is written to `data/<name>/` unless `--out` is given:

| File | Contents |
|---|---|
| `spec.toml` | the specification |
| `manifest.json` | library version, git commit and dirty flag, package versions, start time |
| `runs.jsonl` | one record per run: id, status, wall time, solver evaluations, features, configuration |
| `index.csv` | the same records, configurations flattened to dotted columns |
| `runs/<run_id>.npz` | the trajectory of each run that finished |

Generation resumes where it stopped. Runs already recorded are skipped, including those that failed, since a failure recurs. An interrupt, or the SIGTERM a scheduler sends to end a job, cancels the runs not yet started; those already running finish, and a resume runs whatever was not recorded. A directory holds one specification, and a changed specification needs a new directory. One generation writes to a directory at a time: a second one started on it is refused. The manifest keeps the specification's fingerprint, the canonical text of everything that decides which runs exist, so a dataset always says which protocol its runs are of.

The commit is recorded only when the library runs from a checkout of its repository; an installed copy records none. The dirty flag looks at the code alone, `plantbench/` and `studies/` without their results and figures, so an edited manuscript or a results file an analysis has just written does not mark a dataset or a result dirty. A dataset built by `run_configurations` from a list of configurations writes a manifest of the same kind, with no specification; a directory of runs with no manifest at all is listed by `plantbench datasets` with its provenance unknown.

`plantbench.provenance()` returns the manifest's record of the library and its environment, the version, commit, dirty flag, package versions, platform and time, for a program that produces results outside `datagen`, such as a training run. Given the id of a case that is not built in, it adds where that case comes from, as `case_source`.

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
