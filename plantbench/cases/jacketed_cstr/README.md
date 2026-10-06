# `jacketed_cstr`: non-isothermal CSTR with a cooling jacket

```python
import plantbench as pb
case = pb.load_case("jacketed_cstr")
```

## Provenance

The non-isothermal CSTR of Romagnoli and Palazoglu (2020) on its own, as their constant-volume model. It is the same reactor that sits inside `reactor_separator_recycle`, with the same three modifications to the published parameter table (`plantbench/units/parameters.py`), and it reproduces the published steady state and eigenvalues. It is the smallest case in the library and the simplest one to start from.

## Process

A first-order exothermic reaction A → B in a constant-volume CSTR cooled by a jacket. Heat generation and removal cross three times, at 338.22, 362.40 and 377.08 K. The middle steady state is open-loop unstable, with eigenvalues +0.0600, −0.1350 and −0.7990 1/min.

## States and inputs

| State | Unit | | Input | Unit |
|---|---|---|---|---|
| `C_A` | kmol/m³ | | `F0`, feed flow | m³/min |
| `T` | K | | `CA0`, feed concentration | kmol/m³ |
| `Tj` | K | | `T0`, feed temperature | K |
| | | | `Fj`, coolant flow | m³/min |

## Options

| Option | Default | Values |
|---|---|---|
| `branch` | `"middle"` | `"low"`, `"middle"`, `"high"`: the steady state the design point is taken on |

Parameter overrides address `ReactorParameters` directly, for example `params = {"U": 300.0}`. A set of parameters that does not give three steady states cannot have its branch chosen and is refused.

## Control structures

| Structure | Loops |
|---|---|
| `open loop` | none |
| `temperature PI` | `reactor T` on `Fj`, K_c = −1 m³/min per K, τ_I = 20 min; the reference |
| `composition cascade` | the above, with `reactor C_A` writing its set-point, K_c = −3 K per kmol/m³, τ_I = 20 min |

The tunings are chosen for damping. The least damping ratio is 0.274 under `temperature PI` and 0.213 under `composition cascade`. The PI settings of the Simulink model that accompanies the reference were tuned for a different parameter set. On this reactor they give 0.090 and 0.014, and a 5% feed-concentration step then drives the coolant flow onto its lower limit on every cycle of a sustained oscillation. They can be reproduced through the `tuning` section of a configuration.

## Disturbances

| Name | Arguments | Effect |
|---|---|---|
| `feed_temperature` | `value` or `change`, `t` | the feed temperature moved; 330 K and 340 K carry the open-loop reactor to the low and high branches |
| `step`, `ramp` | `field`, `fraction` or `value`, `t` | any input, for example `CA0` or `F0` |

## Measurements

`C_A`, `T`, `Tj`.

## Reference results

The steady state of the middle branch, C_A = 2.353 kmol/m³, T = 362.4 K and Tj = 345.69 K, and its open-loop eigenvalues, +0.060, −0.1349 and −0.7989 1/min, which the case reproduces within the tolerances of `tests/units/test_reactor.py`. No trajectory is kept as a reference result.

## Verification

`tests/units/test_reactor.py` checks the steady state against the published 2.353 kmol/m³, 362.4 K and 345.69 K, and the eigenvalues against the published +0.060, −0.1349 and −0.7989. `tests/cases/jacketed_cstr/` checks that the case's right-hand side is the reference constant-volume model exactly, that only the middle branch is open-loop unstable, that both structures stabilize it, and the open-loop departures to the other branches.

## Limitations

- A single unit: no recycle, no column and no heat integration, so none of the plantwide couplings of `reactor_separator_recycle`.
- Constant volume, one first-order reaction, and a jacket modeled as one well-mixed volume.
- The tunings are chosen for damping, not taken from the reference's model (see Control structures).
- Measurements and valves are ideal unless instruments or actuators are configured.
