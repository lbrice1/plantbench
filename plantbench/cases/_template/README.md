# `template`: gravity-drained tank

The case card: what the case is, where its numbers come from, and how far it can be trusted. Each section below is required of a case added to the library (`plantbench check <case> --contribute` checks that the headings are there). Replace the tank with your plant. A case presented in a publication of its own adds a `## Citation` section after Limitations: the reference as its first paragraph, then a `bibtex` block. `plantbench describe` prints it and dataset cards list it.

## Provenance

The sources the model and its numbers come from, and which source is authoritative for which. A number corrected from its source is listed here with the evidence for the correction. The tank has no source; it is the smallest model that keeps the contract.

## Process

What the plant does, in the order the material flows through it: a tank of cross-section `A` fed at `F_in` and drained through a valve, `A dh/dt = F_in - Cv * valve * sqrt(h)`.

## States

| State | Unit | Description |
|---|---|---|
| `h` | m | level |

## Inputs

| Field | Unit | Role |
|---|---|---|
| `F_in` | m3/min | inflow; a disturbance |
| `valve` | 0–1 | outlet valve opening; manipulated by the level loop |

## Options

None. Each option, its default and its values, and what it changes in the plant.

## Control structures

| Structure | Loops |
|---|---|
| `open loop` | none |
| `level PI` | level on the valve, `Kc = -0.5`, `tau_I = 10 min` |

Record the damping ratio each tuning gives.

## Disturbances

The generic `step` and `ramp` on any input. List named disturbances with their arguments.

## Measurements

`level` (m) and `outflow` (m3/min).

## Reference results

The values the case reproduces, each with its source and tolerance. None for the tank.

## Verification

What the tests check: the contract, conservation, analytic limits, and the reference results.

## Limitations

What the model leaves out, and the studies it should not be used for.
