# `ngl_deethanizer`: deethanizer of an NGL fractionation train

```python
import plantbench as pb
case = pb.load_case("ngl_deethanizer")
```

## Provenance

The case is constructed from an unpublished Aspen HYSYS 12 case, "Deethanizer - Tuned", hereafter referred to as the HYSYS case. The HYSYS case is the deethanizer of an NGL fractionation train, run in dynamic mode on its own ten-component feed, with its workbook, stream, column-profile and heat-profile reports. These give:

- the feed and its conditions;
- the heater, column, condenser and reboiler, with the tray geometry and vessel sizes;
- the steady state, with every stream and the column's temperature, flow and light-component profiles;
- the nine controllers, with their ranges, actions and tunings.

`hysys_reference.json` holds the transcription. Its `inspection` section holds a later reading of the same case file, "Deethanizer - Tuned (1).hsc" (SHA256 4bfe9b22423e7fd1a0738f41228ba64c44c5bf0aa694b50a34229cb9c6592cbd), through the HYSYS COM interface with the integrator stopped: the products, internal flows, duties, stage and stream pressures and vessel holdups of its saved state, to more figures than the reports. No publication describes this column, and the case reproduces no published number.

In `parameters.py`, each constant carries its source: `[H]` for the HYSYS case, `[D]` for a value derived from it, and `[choice]` for a value it does not give. The case departs from the HYSYS case as follows.

| Item | HYSYS case | This case | Reason |
|---|---|---|---|
| Feed valve VLV-100 and FIC-100 | A valve that FIC-100 moves to hold 625 kmol/h | The feed flow is an input | The flow is the throughput and a disturbance; the valve adds a loop on an input |
| Reflux and FIC-101 | A flow loop, its set-point from TIC-101, with an integral time of 0.0088 s | The reflux flow is an input, written by TIC-101 | The loop is three orders of magnitude faster than any other |
| Heater E-100 | A 0.1 m³ dynamic heater | One well-mixed volume whose outlet temperature is a state | TIC-102 then reads a state, not a quantity algebraic in its own manipulated variable |
| Condenser | A 2 m³ total condenser with vapor and liquid holdup | A drum liquid at its bubble point, with the top pressure carried by one lumped vapor inventory of the column and the drum | The smallest change that keeps the pressure loop |
| Stage pressures | 2453.000 kPa on stage 1 to 2469.756 kPa on stage 30 (inspection); the stream to the reboiler at 2474.906, the reboiler vapor at 2469.752 and the bottoms at 2472.486 kPa; the reflux at 2462.542 kPa | The column report's profile, in whole kPa, 2453 kPa on stage 1 to 2475 kPa on stage 30; the reboiler at the stage 30 pressure; the drum at the top pressure | The report places the bottoms and the boil-up below stage 30, which its integer rounding cannot account for; the inspection's pressure locations are not mapped onto the case's stages and vessels |
| Stages | Ideal, efficiency 1 | Ideal | As HYSYS |
| Product valves | Rigorous Fisher valves, Cv 18 and 29 | Linear in opening, sized to the HYSYS design openings, 44.56 % and 40.63 % | The let-down pressures, 2010 and 1790 kPa, are far enough below the column that its pressure barely moves the flows |
| Delivered products | Flashed through the valves: the ethane product at −10.054 °C and a vapor fraction of 0.0860, the LPG at 86.346 °C and 0.2046 (inspection) | The liquids before the valves; `P_distillate` and `P_LPG` are informational | An unsplit throttle keeps the overall composition, which XIC-100 reads; the delivered temperatures and phase splits are not computed |
| Vessel inventories | 11.782 and 6.860 kmol at 45 % in the 2 m³ drum and reboiler (inspection) | 12.24 and 7.02 kmol at 45 %: a level is a share of a molar capacity at a fixed density, 13.6 and 7.8 kmol/m³ | The level loops need a level; equal levels are not equal inventories, and the level responses differ |
| Energy storage | The trays', the vessels' and the heater's | Stage and reboiler balances with d(Mh)/dt taken as h dM/dt, the vapor flows algebraic, no tray vapor holdup; E-100 one temperature state with a heat capacity of 340 kJ/K [choice] and no composition holdup | The steady balances close; the thermal transients differ from HYSYS's |
| Equation of state | HYSYS Peng–Robinson | Peng–Robinson with the ChemSep binary interaction parameters, in NumPy (`plantbench.units.peng_robinson`) | The same equation; the interaction parameters and ideal-gas heat capacities differ, and with them the duties (Reference results) |
| Design | The dynamic case's state | The steady state at the HYSYS set-points | The design has to be a steady state |
| Tuning | The HYSYS tuning | Converted from it, with TIC-101 and XIC-100 detuned for damping (Control structures); the HYSYS tuning of both is `hysys_reference.config(tuning="hysys")` | The HYSYS tuning damps the closed loop at 0.234 |

## Process

The feed, 625 kmol/h of 49 % ethane, 23.5 % propane, 1.1 % methane and butanes to heptane, arrives at 15 °C and 2548 kPa. It is let down to 2500 kPa and heated in E-100 to 33 °C, where it is 1 % vapor. It enters stage 15 of the 30-stage column T-100.

The overhead vapor leaves stage 1 at 2453 kPa and 2.0 °C and is totally condensed. The drum liquid, at −2.6 °C, is split between the reflux, 371 kmol/h, and the distillate, 313 kmol/h of 96 % ethane, which is let down through VLV-101 to 2010 kPa. The reboiler, at 100 °C, boils up 624 kmol/h. The bottoms, 312 kmol/h of LPG with 2 % ethane, is let down through VLV-102 to 1790 kPa.

The column recovers 98.0 % of the feed's ethane in the distillate and 96.1 % of its propane in the LPG.

## States

Holdups are in kmol, compositions in mole fractions, temperatures in K and pressure in kPa. `n_c` is 10.

| State | Description |
|---|---|
| `T_E100` | heater outlet temperature |
| `M_D`, `x_D[n_c]` | condenser drum |
| `M[30]`, `x[30, n_c]` | the stages, top first |
| `M_B`, `x_B[n_c]` | reboiler |
| `P_top` | stage 1, the vapor to the condenser |

The plant has 354 states.

## Inputs

| Field | Unit | Role |
|---|---|---|
| `F_feed` | kmol/min | feed flow; the throughput, and a disturbance |
| `z_feed[n_c]` | – | feed composition, recorded element by element |
| `T_feed` | K | feed temperature, a disturbance |
| `P_feed` | kPa | the E-100 inlet pressure |
| `Q_E100` | kW | E-100 duty; TIC-102 |
| `L_reflux` | kmol/min | reflux; TIC-101 |
| `Q_condenser` | kW | condenser duty; PIC-100 |
| `Q_reboiler` | kW | reboiler duty; TIC-100 |
| `v_VLV101` | 0–1 | distillate valve; LIC-100 |
| `v_VLV102` | 0–1 | LPG valve; LIC-101 |

## Options

The case has no options. Parameter overrides address `DeethanizerParameters` by name, for example `params = {"x_C2_LPG_design": 0.015}`. An override starts the design solve from the HYSYS state, and falls back to continuation in the changed parameters.

## Control structures

| Structure | Loops |
|---|---|
| `open loop` | none |
| `basic` | the regulatory layer of the HYSYS case, with the stage 28 temperature at a fixed set-point; the reference |
| `composition cascade` | `basic`, with XIC-100 writing the stage 28 temperature set-point, as the HYSYS case runs |

The loops are named by what they control.

| Loop | Tag | Manipulated variable | K_c | τ_I, min |
|---|---|---|---|---|
| `feed temperature` | TIC-102 | `Q_E100` | 66 kW/K | 1.68 |
| `overhead pressure` | PIC-100 | `Q_condenser` | −12.8 kW/kPa | 5.06 |
| `condenser level` | LIC-100 | `v_VLV101` | −0.05 per % | 0.7 |
| `reboiler level` | LIC-101 | `v_VLV102` | −0.03 per % | 0.7 |
| `stage 4 temperature` | TIC-101 | `L_reflux` | −0.25 kmol/min per K | 3.5 |
| `stage 28 temperature` | TIC-100 | `Q_reboiler` | 402 kW/K | 1.47 |
| `ethane in LPG` | XIC-100 | stage 28 temperature set-point | −250 K per unit mole fraction | 5 |

A negative gain means that raising the manipulated variable lowers the measurement. The loops start from the HYSYS case's tuning, which is given in percent of its controller ranges. Converted to these units, a HYSYS "direct" controller has a negative gain and a "reverse" one a positive gain.

| Loop | Action | PV range | OP range | K_c, %/% | τ_I, s |
|---|---|---|---|---|---|
| TIC-102 | reverse | 0 to 80 °C | 0 to 833 kW | 6.33 | 101 |
| PIC-100 | direct | 0 to 3500 kPa | 0 to 2500 kW | 17.9 | 304 |
| LIC-100 | direct | 0 to 100 % | valve | 5 | 42 |
| LIC-101 | direct | 0 to 100 % | valve | 3 | 42 |
| TIC-101 | direct | 0 to 200 °C | 0 to 1000 kmol/h | 13 | 210 |
| TIC-100 | reverse | −5 to 140 °C | 0 to 5556 kW | 10.5 | 88 |
| XIC-100 | direct | 0 to 1 | −5 to 140 °C | 20 | 66 |

Converted unchanged, the tuning is stable but damps `basic` at 0.234 and `composition cascade` at 0.206. Two loops are detuned to remedy this:

- **TIC-101.** The poorly damped mode, at 2 rad/min, is the stage 4 temperature loop. A gain of −0.25 kmol/min per K, 0.23 of the converted gain, damps `basic` at 0.445.
- **XIC-100.** Its proportional action passes through that of TIC-100 to the reboiler duty. A gain of −250 K per unit mole fraction, 0.086 of the converted gain, with a 5 min integral time, damps the cascade at 0.427.

Detuning TIC-101 further gains little damping. At 0.15, 0.10 and 0.05 of the converted gain, the least damped mode is a slow one near 0.1 rad/min, damped at 0.448, 0.438 and 0.460. The slowest mode lengthens meanwhile from 12.3 min to 15.9, 19.6 and 25.8 min. The open loop's least damped mode is 0.512.

The damping ratio of each structure is the smallest among the oscillatory modes of the closed loop linearized at the design point, from `studies/ngl_deethanizer/tuning.py`. The slowest mode is −1/Re λ of the rightmost eigenvalue λ. The open loop has the integrating modes of the two levels and the pressure.

| Structure | Damping ratio | Slowest mode, min |
|---|---|---|
| `open loop` | 0.512 | – |
| `basic` | 0.445 | 12.3 |
| `composition cascade` | 0.427 | 15.4 |

Two loops of the HYSYS case are not separate loops here, because the case makes the flows they control inputs:

- **FIC-100.** `F_feed` is an input, so the feed-flow loop is the input.
- **FIC-101.** `L_reflux` is an input, so TIC-101 writes the reflux directly.

The HYSYS case also carries a reflux ratio spreadsheet, the reflux set to 0.62 times the feed. No controller reads it in the HYSYS case, and the case leaves it out.

## Disturbances

| Name | Arguments | Effect |
|---|---|---|
| `feed ethane` | `fraction`, `t` | the feed's ethane mole fraction stepped by `fraction` of itself, the other components scaled to keep the sum at one |
| `step`, `ramp` | `field`, `fraction` or `value`, `t` | any input; `F_feed` and `T_feed` are the two disturbances of the HYSYS case's operator panel |

## Measurements

The controller PVs, in the HYSYS case's units:

- `feed flow` (kmol/h);
- `feed temperature`, the E-100 outlet (°C);
- `overhead pressure` (kPa);
- `condenser level` and `reboiler level` (%);
- `reflux flow` (kmol/h);
- `ethane in LPG` (mole fraction);
- the 30 stage temperatures, `stage 1 temperature` to `stage 30 temperature` (°C), of which TIC-101 reads stage 4 and TIC-100 stage 28.

The limits (`case.limits(design)`) give the controller PVs their spans in the HYSYS case [H], which the loops' signals in percent are expressed in: `feed temperature` 0 to 80 °C, `overhead pressure` 0 to 3500 kPa, the levels 0 to 100 %, `stage 4 temperature` 0 to 200 °C, `stage 28 temperature` −5 to 140 °C and `ethane in LPG` 0 to 1. The HYSYS case declares no alarm or trip limits.

The products:

- `distillate flow` and `LPG flow` (kmol/h);
- `ethane in distillate` and `propane in distillate` (mole fractions);
- `ethane recovery`, to the distillate, and `propane recovery`, to the LPG;
- `condenser temperature` and `reboiler temperature` (°C);
- `E-100 duty`, `condenser duty` and `reboiler duty` (kW).

## Reference results

The reference configuration is the design at the HYSYS set-points under `basic`. `studies/ngl_deethanizer/hysys_comparison.py` compares it with the HYSYS steady state.

| Quantity | HYSYS | This case |
|---|---|---|
| Distillate, kmol/h | 312.7 | 312.67 |
| Ethane in distillate | 0.9592 | 0.9595 |
| Propane in distillate | 0.0188 | 0.0185 |
| Ethane recovery | 0.97959 | 0.97960 |
| Propane recovery | 0.9600 | 0.9606 |
| Reflux, kmol/h | 367.4 | 371.4 |
| Boil-up, kmol/h | 608.4 | 624.3 |
| Reboiler duty, kW | 2399 | 2441 |
| Condenser duty, kW | 1598 (reported); 1754 (from its streams) | 1767 |
| E-100 duty, kW | 420 | 396 |
| E-100 outlet vapor fraction | 0.023 | 0.008 |
| Drum temperature, °C | −2.63 | −2.60 |
| Reboiler temperature, °C | 100.30 | 100.34 |
| Stage 28 temperature, °C | 71.16 | 71.58 |

Every stage temperature is within 0.96 K of HYSYS. The largest difference is on stages 24 and 25, where the profile rises 6 K a stage. The stage ethane and propane fractions are within 0.007 and 0.010.

The reflux, boil-up and reboiler duty are 1.1 %, 2.6 % and 1.7 % above HYSYS. Two quantities are recorded as misses:

- **Condenser duty.** HYSYS reports 1598 kW, but the enthalpies of its own streams imply 1754 kW in the reports' rounded values and 1757.6 kW in the inspection's: the vapor to the condenser less the reflux and the distillate. The case's 1767 kW agrees with the second. A saved dynamic state does not distinguish accumulation in the condenser, another heat term or a reporting convention, so the case's duty is not fitted to either value.
- **E-100 outlet.** The feed leaves E-100 near its bubble point. There the vapor fraction is sensitive to the two implementations' differences, and the heater's enthalpy rise is 5 % smaller than HYSYS's.

`studies/ngl_deethanizer/closed_loop.py` records three indicative closed-loop runs under `composition cascade`, each with a step at 10 min and run to 120 min. These are not reference results.

| Run | Ethane in LPG, largest deviation | Reflux at 120 min, kmol/h | Reboiler duty at 120 min, kW | Wall time, s |
|---|---|---|---|---|
| Feed flow +10 % | 0.00044 | 408.6 | 2685 | 300 |
| Feed temperature +5 K | 0.00011 | 371.4 | 2441 | 485 |
| Feed ethane +10 % | 0.00047 | 388.4 | 2283 | 361 |

In every run each loop returns to its set-point within the 120 min. In the feed-ethane run, XIC-100 moves the stage 28 set-point from 71.58 to 71.14 °C, and the ethane recovery settles at 0.9833.

The inferred ethane recovery moves by up to 0.11 at a feed-flow step. The feed rises at once, and the distillate follows only as the drum level loop opens VLV-101.

## Comparison with the tuned HYSYS case

`hysys_reference.config()` is the configuration of the comparison: `composition cascade`, the structure the HYSYS case runs, at the design of the HYSYS set-points, which is the design of every structure. `tests/cases/ngl_deethanizer/test_hysys_inspection.py` compares it with the inspection at one criterion per kind of quantity: product flows and compositions within 1 %, internal flows within 3 %, duties within 5 %, temperatures within 1 K. A quantity outside its criterion is asserted as an exception. With `tuning="hysys"`, the HYSYS tuning of TIC-101 and XIC-100, the configuration is damped at 0.206, with its slowest mode at 5.1 min, against 0.427 and 15.4 min.

| Quantity | HYSYS, inspection | This case | Criterion |
|---|---|---|---|
| Distillate, kmol/h | 312.756 | 312.669 | met |
| LPG, kmol/h | 312.255 | 312.331 | met |
| Ethane in distillate | 0.959242 | 0.959493 | met |
| Ethane in LPG | 0.02 | 0.02 | met |
| Reflux, kmol/h | 367.38 | 371.443 | met |
| Vapor to the condenser, kmol/h | 679.913 | 684.112 | met |
| Boil-up, kmol/h | 608.615 | 624.258 | met |
| Reboiler duty, kW | 2399.06 | 2440.74 | met |
| Condenser duty, kW, from the streams | 1757.556 | 1766.82 | met |
| Condenser duty, kW, as saved | 1598.442 | 1766.82 | exception |
| E-100 duty, kW | 420.248 | 395.993 | exception |
| E-100 outlet vapor fraction | 0.0229675 | 0.00824216 | exception |
| Drum liquid, °C | −2.626 | −2.604 | met |
| Reboiler liquid, °C | 100.308 | 100.343 | met |
| Stage 28, °C | 71.160 | 71.584 | met |

Every stage temperature is within 0.963 K of the column report. The three exceptions are those of Reference results: the saved condenser duty, and the heater, whose feed leaves near its bubble point.

The case departs from the HYSYS case in the following, beyond the tuning. Each is either a reduction the case keeps, for the reason in Provenance, or open.

| Difference | Disposition |
|---|---|
| FIC-100 and FIC-101 are inputs, not loops | kept: the feed is the throughput, and FIC-101 is three orders of magnitude faster than any other loop |
| Linear product valves, blind to the column and downstream pressures | kept |
| One lumped pressure state, the drum at its bubble point, no tray vapor holdup, d(Mh)/dt taken as h dM/dt | kept |
| Levels at fixed molar capacities | kept |
| The delivered products, after the valves | open: not computed |
| The stage, reboiler and drum pressures of the inspection | open: not mapped |
| The saved condenser duty against the streams | open: needs a steady energy record of the HYSYS case |
| The property data of the two Peng–Robinson implementations (interaction parameters, ideal-gas heat capacities) | open: HYSYS's are not exported; the duties and the heater outlet differ with them |

## Verification

`tests/cases/ngl_deethanizer/` holds the case's own tests:

- **`test_design.py`.** The design is a steady state, every component and the energy are conserved, and the specifications hold. The `settle` fallback reaches the same design from a disturbed start, which is the slow test.
- **`test_golden.py`.** The reference configuration is checked against the HYSYS case, with tolerances argued from the comparison, and the misses are asserted as misses. Eight results are frozen to a relative 1e-6.
- **`test_hysys_reference.py`.** The transcription is self-consistent: the streams balance, the recovery is HYSYS's own, the controllers read the profile, and the column flows close at both ends.
- **`test_hysys_inspection.py`.** `hysys_reference.config()` against the inspection, at the criteria of Comparison with the tuned HYSYS case, with the exceptions asserted as exceptions; the HYSYS tuning converted from the HYSYS controllers.
- **`test_structures.py`.** Every structure is a fixed point, and `feed ethane` keeps the feed normalized. The damping ratios are asserted in the slow tests.
- **Shared modules.** The equation of state and the staged column are tested in the library (`tests/units/test_peng_robinson.py`, `tests/units/test_staged_column.py`).

The library's contract suite runs on every structure.

## Limitations

- **Upstream.** The feed is the HYSYS case's own, not the bottoms of `ngl_demethanizer`. The two cases do not form a train.
- **Simplified units.** The top pressure is one lumped state, the drum is at its bubble point, the heater has no composition holdup, and the stages have no vapor holdup.
- **Valves.** The product valves are linear and do not see the column pressure, and the feed and reflux valves are not modeled. The products are reported before their let-down.
- **Cost.** One right-hand side costs 6 ms. A 120 min closed-loop run after a disturbance takes 5 to 8 min.
- **Instruments.** Measurements and valves are ideal unless instruments or actuators are configured. The composition analyzer XIC-100 has no dead time.
