# Design principles

`plantbench` is a library of plantwide process simulations intended as test problems for methods in process systems engineering: control structure selection, integration of design and control, surrogate and data-driven modeling, monitoring and knowledge discovery. This document states what the library commits to and why. The [user guide](user-guide.md) covers how to use it, and [Adding a case](adding-a-case.md) covers how to extend it.

## Test problems between single units and fixed flowsheets

Available test problems sit at two extremes. Single-unit environments, a reactor or a column on its own, have no recycle, no inventory to accumulate and no energy integration, and therefore none of the couplings that make plantwide problems difficult. The Tennessee Eastman process is plantwide, but it is a single flowsheet with a fixed list of faults, so it offers no design alternatives to choose between, no heat-recovery targets and no competing objectives across designs.

A `plantbench` case is a parameterized design space rather than a single flowsheet. `reactor_separator_recycle` exposes column pressure, four heat-exchanger networks, a continuous pump-around share, four regulatory structures, controller tuning, throughput and measurement non-idealities, all as configuration. Its economic, environmental and operability measures are computed from the same converged state, so rankings on each are commensurable.

## The protocol a method reports against

A test problem is used through a protocol: which scenarios a method is exposed to, what it is asked to predict, and how its answer is scored. Where the scenarios are a fixed list, the protocol is settled once and thereafter left implicit, and a method that does well on the list carries an unstated advantage over one reported on a subset of it.

`plantbench` makes the protocol part of what is reported. The scenarios are a specification that expands deterministically into the runs, and a `[task]` table names the signals and window a method may see, how a run is labeled, how the runs are split and what is scored. The specification names the runs by a digest, the specification and the task together name the protocol by a second digest, and a dataset records both, so that a reported protocol is a hash a reader can regenerate from. What lies between the signals and the prediction is the method, not the protocol.

An established flowsheet is brought in the same way: as a case, reported through the same protocol. That is the migration path intended for the Tennessee Eastman process, rather than a replacement for it.

## What a case is

A **case** is a plant model offered as a test problem. It supplies:

- a model: a right-hand side, a steady-state design point, and the parameters both depend on;
- a verified ground truth: at least one result checked against an independent source, and conservation checks on the rest;
- a design space: options and parameters a configuration can change;
- control structures, declared as data;
- disturbances and measurements, named.

A **configuration** selects one member of a case's design space: a structure, the case's options, parameter overrides, set-points, tuning, non-idealities and a disturbance. It is plain data. It can be written to TOML or JSON, it round-trips exactly, and its hash identifies the run it describes.

A case's **reference configuration** is the one with only the default structure named. For a case built on published results, the reference configuration reproduces them, and a golden test holds it to that.

## Frozen cases

`reactor_separator_recycle` is an extension of the reactor control example of Romagnoli and Palazoglu (2020), and the numbers reported for it are transcribed from its results files. It is frozen: its reference configuration must reproduce `studies/reactor_separator_recycle/results/reference.txt` and `designs.txt` byte for byte, and `tests/cases/reactor_separator_recycle/test_golden.py` checks that it does. Everything added to `reactor_separator_recycle` since has been added as an option that is off by default. A change that would move a reference number is a new option or a new case, not an edit.

## Conventions

Each of these exists because the alternative caused a problem in this code.

**Parameters are frozen dataclasses.** Every physical constant lives in a parameter set, with any correction to a published value recorded in a comment beside the value it corrects. No function reads a constant from its own body. Parameter overrides in a configuration address these sets by dotted path and fail on a path that names no field.

**Control structures are data.** A structure is a list of `Loop` and `Ratio` elements, each naming a measurement, a manipulated variable, a tuning and a range. Comparing two structures means comparing two lists, not rewiring the model. Set-points, tuning and non-idealities are applied to a structure by loop name, and an unknown name raises.

**Non-idealities are ideal by default.** An `Instrument` on the measurement path and an `Actuator` on the valve are ideal when default-constructed, and an ideal element carries no state. A structure of ideal elements therefore has exactly one augmented state per element and reproduces the reference results exactly. A new non-ideality must contribute zero states when it is switched off.

**Declared but unimplemented raises.** `Instrument.noise_std` and `Actuator.stiction` raise rather than do nothing, because a study that sets them and sees no effect would draw a false conclusion. A field that cannot yet be implemented is declared and made to raise.

**The right-hand side stays continuous.** The closed loop is integrated by a stiff solver and linearized by finite differences for damping ratios and spectra. Anti-windup is by back-calculation rather than conditional integration, and deadtime is a Padé approximation rather than a history buffer. Anything discontinuous, such as stiction, a deadband or a relay, needs a decision about smoothing or event handling before it is added. Discontinuities in time are declared instead: a disturbance or a set-point schedule names the times it acts at, and `plantbench.run` restarts the integration there. A solver left to itself from a plant at rest grows its step until one step spans the upset, and the state it lands on can be unphysical; one run of the first regimes dataset went that way.

**A failed integration is an error, not a trajectory.** A state that is not finite raises, because the solver does not always reject the step that produced it.

**Tests assert conservation and limits, not stored outputs.** Mass and energy close, mole fractions sum to one, an inventory with no exit accumulates what is fed, and the pinch target bounds every network. Tests that compare against stored outputs are reserved for published results, where the stored output is the point. A test that would catch a physical error is preferred to one that pins the current output.

## Layers

```
plantbench/core         control layer, instruments, disturbances, the case interface, specifications
plantbench/units        unit-operation models and their parameter sets
plantbench/heat         streams and pinch analysis
plantbench/cases        the cases, each depending on the layers above and on nothing else
plantbench/datagen      dataset generation, which composes cases and core
plantbench/datasets     reading datasets back: summaries, provenance stamps, comparison
plantbench/task         what is predicted from a dataset and how an answer is scored
plantbench/sensitivity  Sobol' indices over a design space
```

Imports run one way. `core`, `units` and `heat` import nothing from each other or from any case, only `backend`, the array module beneath them (NumPy or CuPy), no case imports another, and no case imports the modules that generate, read or score datasets. Nothing in the library imports the studies. `tests/test_architecture.py` enforces this. `core` knows nothing about any plant, so a new case cannot require a change there that encodes its own assumptions. If a case does not fit the interface, the interface is changed so that it fits every case.

## Generalizing on the second use

Code moves out of a case into a shared layer when a second case needs it, not before. Pinch analysis and the stream type were already independent of `reactor_separator_recycle` and live in `heat`. The Peng–Robinson equation of state and the staged column with a stage energy balance were written for `ngl_demethanizer` and moved to `units` when `ngl_deethanizer` needed them. The heat-exchanger network model, the economics and the stream table remain in `reactor_separator_recycle`. Generalizing them from a single example would encode that example's assumptions: a network's `pressure`, for instance, is a column pressure, which a plant without a column does not have.

## Data

Datasets are generated from a specification ([user guide](user-guide.md#generating-a-dataset)), and a dataset records how it was made: the specification, the library version and git commit, the package versions, and every run's configuration. A run's identifier is the hash of its configuration, so runs are found by what they are rather than by their position in a sweep.

A run that fails is a result, not an error. A configuration the case refuses is recorded as `invalid`, an integration that raises as `failed`, and one that exceeds its wall-clock budget as `timeout`. The cost of each run is recorded, because it varies by more than an order of magnitude across a design space and correlates with the behavior of the plant: a limit-cycling design costs far more to integrate than one that settles.

## Scope

Every result the library produces is model-to-model. There is no plant data, and nothing here validates a method against industrial measurements. The chemistry of the current cases is a single first-order reaction. Conclusions about the performance of a method can transfer; conclusions about chemistry cannot.
