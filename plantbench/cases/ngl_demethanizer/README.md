# `ngl_demethanizer`: demethanizer section of a cryogenic NGL recovery plant

```python
import plantbench as pb
case = pb.load_case("ngl_demethanizer")
```

## Provenance

The case is a modified version of the demethanizer section studied by Chebeir, Salas and Romagnoli (2019), *Operability assessment on alternative natural gas liquids recovery schemes*, J. Nat. Gas Sci. Eng. 71, 102974, hereafter referred to as the paper. The paper gives the flowsheets of three recovery schemes (its Figs. 1–3), the feed (Table 1), the equipment sizes (Section 3.1), the regulatory structure and its tuning (Fig. 4, Section 3.2, Table 2), a dozen stream conditions of the conventional scheme (Section 2), and closed-loop results (Section 4). It gives no stream table, no conditions for the other two schemes and no equation of state.

A second source is the unpublished HYSYS 12 case "Demethanizer - Tuned", a converged cold residue recycle plant on its own ten-component feed, with its workbook, stream, heat-exchanger and column reports. It fixes the cold residue recycle topology that the paper's Fig. 3 leaves open, and it is authoritative for the `hysys` basis. Its streams, column profile, machine powers and controller ranges are transcribed in `hysys_reference.json`.

In `parameters.py`, each constant carries its source: `[C]` for the paper, `[H]` for the HYSYS case, `[D]` for a value derived from them, and `[choice]` for a value neither gives. The case departs from the paper as follows.

| Item | Paper | This case | Reason |
|---|---|---|---|
| Design specification | TK-100 at −29.3 °C and the NGL at −9.6 °C, stated for the conventional scheme | Ethane recovery 0.82 (ERIC-100) and 1 % methane in the NGL (XIC-100), for every scheme; the TK-100 temperature is a result: −40.03 °C (conventional), −44.21 °C (gsp), −44.82 °C (crr) | The paper compares the schemes at a recovery set-point of 0.82. At −29.3 °C the case recovers 0.723 (conventional), 0.372 (gsp) and 0.337 (crr). At −40.0 °C the conventional design reproduces the paper's TK-101 and TE-100 outlet temperatures to 0.1 K, so the stated −29.3 °C is recorded as a miss |
| Cold residue recycle | A residue recycle compressed to a "slightly higher" pressure and returned to the top | One third of the overhead, drawn before E-102, raised by K-102 from 1010 to 1036 kPa, cooled about 2 K in E-104 against the branch, entering stage 1 as vapour | Read from the HYSYS case |
| Branch of gsp and crr | The TK-100 vapour that bypasses TE-100 | The bypass vapour and 56 % of the TK-100 liquid | Read from the HYSYS case; gsp takes the liquid share too, so that crr is gsp with the recycle |
| Feed stages | Stages 1, 8, 26 and 30 marked; no stream assigned | Under crr: recycle on 1, branch on 2, expander on 8, TK-100 liquid on 26. Under gsp: the same with the branch on 1. Under conventional: expander on 1, TK-101 liquid on 8, TK-100 liquid on 26 | Fixed for crr by the HYSYS case; stage 30 is read as the bottom stage, not an inlet |
| Stage 1 under crr | An equilibrium stage | An adiabatic mixing point without holdup; 29 trays | HYSYS passes no liquid from stage 1. As an equilibrium tray, its liquid was renewed at 0.05 kmol/h and accumulated heavy components over about a day, leaving the steady state undetermined |
| Equation of state | Not stated | Peng–Robinson with the ChemSep binary interaction parameters, in NumPy (`plantbench.units.peng_robinson`) | An inference, standard for this service. `thermo` and `feos` were too slow inside the right-hand side |
| Butanes, pentanes, hexanes | Lumps | Their normal isomers | Replacing them by the iso-isomers moves the expander outlet by 0.01 K |
| Separator pressures | TK-100 at 4672 kPa, TK-101 at 5199 kPa | As published | The rise along the flow is unusual, but with the pressures as published the design reproduces TK-101 and the TE-100 outlet; with them exchanged it does not |
| Column pressure | Not modelled as such | One lumped vapour inventory for the top and the overhead circuit, so that PIC-100 has a state to act on | The smallest change that keeps the pressure loop |
| Machines | HYSYS rigorous, 75 % | Ideal-gas booster, K-101 and K-102, each with its own heat capacity, at 75 % | Leaves K-101 8 K below the rigorous discharge temperature |
| Exchangers | UA given | Ten counter-current cells per exchanger, each side one temperature state | A dynamic exchanger with the published UA |
| Tray hydraulics and holdups | Not given | Francis weirs on a 1.72 m column, with the active area and weir height as `[choice]`; the HYSYS tray geometry on the `hysys` basis | They set the dynamics, not the design point |
| Flow transmitters | Not modelled | First-order, 0.1 min, on the NGL flow and, under crr, on the recycle and residue-gas flows | A loop reads its measurement before its manipulated variable is written, so ERIC-100 and RFIC-100 need measurements that are states (Control structures) |
| Tuning | Table 2, in percent of ranges the paper does not give | Retuned for damping (Control structures) | Table 2 cannot be converted without the ranges |

## Process

Raw gas at 4980 kmol/h, 6015 kPa and 35 °C is cooled in the gas–gas exchanger E-100 against the residue gas, and then in the chiller E-101, a duty. It is flashed in the separator TK-100 at 4672 kPa. The TK-100 liquid is let down to the column. What happens to the TK-100 vapour depends on the scheme:

- **`conventional`.** The vapour is cooled in E-102 against the overhead and flashed in the cold separator TK-101. The TK-101 vapour is expanded in the turboexpander TE-100 to the top of the column, and the TK-101 liquid enters stage 8.
- **`gsp` (gas subcooled process).** The vapour is split at the ratio R between TE-100, which discharges to stage 8, and a branch. The branch carries the rest of the vapour and 56 % of the TK-100 liquid. It is subcooled in E-102, let down, and enters stage 1.
- **`crr` (cold residue recycle).** As `gsp`, with the branch crossing the cold side of E-104 and entering stage 2. One third of the overhead is recompressed by K-102, cooled in E-104, and returned to stage 1.

The demethanizer T-100 has 30 stages and the reboiler E-103, and no condenser. The bottoms are the NGL product. The overhead crosses E-102 and E-100, is raised by the booster on the TE-100 shaft, and then by the recompressor K-101 to the sales-gas pressure.

## States

Holdups are in kmol, compositions in mole fractions, temperatures in K, pressure in kPa and flows in kmol/min. `n_c` is 7 on the `chebeir2019` basis and 10 on `hysys`.

| State | Description |
|---|---|
| `T_E100h`, `T_E100c`, `T_E102h`, `T_E102c` | exchanger cell temperatures, hot and cold sides, ten cells each |
| `T_E104h`, `T_E104c` | E-104 cells, crr only |
| `T_TK100`, `M_TK100`, `x_TK100[n_c]` | separator TK-100 |
| `T_TK101`, `M_TK101`, `x_TK101[n_c]` | cold separator TK-101, conventional only |
| `M[n]`, `x[n, n_c]` | the trays, top first; n is 30, or 29 under crr |
| `M_B`, `x_B[n_c]` | reboiler |
| `P_top` | column top and overhead circuit |
| `FT_NGL` | NGL flow transmitter |
| `FT_recycle`, `FT_residue` | recycle and residue-gas flow transmitters, crr only |

The plant has 308 states under `conventional`, 299 under `gsp`, 313 under `crr` and 406 under `crr` on the `hysys` basis.

## Inputs

| Field | Unit | Role |
|---|---|---|
| `F_feed` | kmol/min | feed flow; the throughput, and a disturbance |
| `z_feed[n_c]` | – | feed composition, recorded element by element |
| `T_feed`, `P_feed` | K, kPa | feed conditions |
| `Q_chiller` | kW | E-101 duty; TIC-100 |
| `Q_reboiler` | kW | E-103 duty; XIC-100 |
| `W_recompressor` | kW | K-101 power; PIC-100 |
| `v_LCV100` | 0–1 | TK-100 liquid valve |
| `v_LCV101` | 0–1 | TK-101 liquid valve; conventional only |
| `v_LCV102` | 0–1 | NGL valve |
| `split` | – | R, the TE-100 flow over the branch vapour; gsp and crr only |
| `W_K102` | kW | K-102 power; RFIC-100; crr only |
| `liquid_split` | 0–1 | the part of the TK-100 liquid sent to the branch; gsp and crr only |

One `Inputs` dataclass serves the three schemes, so that a dataset spanning them has one set of columns. A field that a scheme does not use is recorded as a constant.

## Options

| Option | Default | Values |
|---|---|---|
| `scheme` | `"conventional"` | `"conventional"`, `"gsp"`, `"crr"`: the recovery scheme |
| `basis` | `"chebeir2019"` | `"chebeir2019"`: the paper's feed and equipment, on the seven components of its Table 1, at the design specification above. `"hysys"`: the HYSYS case's feed, tray geometry and pressure profile, on its ten components with carbon dioxide at zero, at TK-100 −38.71 °C and 1 % methane in the NGL; crr only |

Parameter overrides address `NGLParameters` by name, for example `params = {"recovery_design": 0.85}`. An override starts the design solve from the stored design point and falls back to continuation in the changed parameters.

## Control structures

| Structure | Loops |
|---|---|
| `open loop` | none |
| `basic` | the regulatory layer of the paper's Fig. 4, below; the reference |
| `recovery cascade` | `basic`, with ERIC-100 writing the TK-100 temperature set-point |
| `recovery cascade on ratio` | `basic`, with ERIC-100 writing R; gsp and crr only |

The loops of `basic` are named by what they control.

| Loop | Tag | Manipulated variable | K_c | τ_I, min |
|---|---|---|---|---|
| `TK-100 temperature` | TIC-100 | `Q_chiller` | −167 kW/K | 5 |
| `TK-100 level` | LIC-100 | `v_LCV100` | −0.017 per % | 8 |
| `TK-101 level`, conventional | LIC-101 | `v_LCV101` | −0.017 per % | 8 |
| `reboiler level` | LIC-102 (conventional), LIC-101 (gsp, crr) | `v_LCV102` | −0.017 per % | 8 |
| `overhead pressure` | PIC-100 | `W_recompressor` | −5.12 kW/kPa | 4 |
| `methane in NGL` | XIC-100 | `Q_reboiler` | −27.2 kW per mol % | 2 |
| `recycle ratio`, crr | RFIC-100 | `W_K102` | 25 kW per unit ratio | 0.5 |
| `ethane recovery`, on temperature | ERIC-100 | TK-100 temperature set-point | −5 K per unit recovery | 3 |
| `ethane recovery`, on ratio | ERIC-100 | `split` | 1 per unit recovery | 3 |

A negative gain means that raising the manipulated variable lowers the measurement. The regulatory loops start from the HYSYS case's own tuning, which is given in percent of its controller ranges, converted to these units:

| Loop | PV range | OP range | K_c, %/% | τ_I, s |
|---|---|---|---|---|
| TIC-100 | −50 to 0 °C | 0 to 5556 kW | 1.5 | 300 |
| LIC-100, LIC-101 | 0 to 100 % | valve | 1.7 | 480 |
| PIC-100 | 500 to 4000 kPa | 0 to 11944 kW | 1.5 | 240 |
| XIC-100 | 0 to 1 | 0 to 2722 kW | 1.0 | 120 |
| RFIC-100 | 0 to 1 | 0 to 50 kW | 0.5 | 30 |

ERIC-100 is tuned for this case. On the ratio it has a positive gain: at a fixed TK-100 temperature, raising R by 5 % raises the recovery by 0.0064 under both gsp and crr, because less of the TK-100 vapour goes to the branch.

The damping ratio of each structure is the smallest among the oscillatory modes of the closed loop linearized at the design point. The values below are from `studies/ngl_demethanizer/tuning.py`.

| Scheme | Basis | Structure | Damping ratio | Slowest mode, min |
|---|---|---|---|---|
| `conventional` | `chebeir2019` | `basic` | 0.550 | 11.7 |
| `conventional` | `chebeir2019` | `recovery cascade` | 0.470 | 91.9 |
| `gsp` | `chebeir2019` | `basic` | 0.557 | 8.5 |
| `gsp` | `chebeir2019` | `recovery cascade` | 0.496 | 15.2 |
| `gsp` | `chebeir2019` | `recovery cascade on ratio` | 0.440 | 32.6 |
| `crr` | `chebeir2019` | `basic` | 0.501 | 8.5 |
| `crr` | `chebeir2019` | `recovery cascade` | 0.482 | 15.6 |
| `crr` | `chebeir2019` | `recovery cascade on ratio` | 0.412 | 30.6 |
| `crr` | `hysys` | `basic` | 0.515 | 8.6 |
| `crr` | `hysys` | `recovery cascade` | 0.475 | 29.9 |
| `crr` | `hysys` | `recovery cascade on ratio` | 0.422 | 63.4 |

The slowest mode is the time constant of the rightmost eigenvalue λ, −1/Re λ.

Two loops of the paper are not separate loops here, because the case makes the controlled quantity itself an input:

- **FIC-100.** `F_feed` is an input, so the feed-flow loop is the input.
- **RIC-100.** R is the input `split`, so the ratio station is the input, and `recovery cascade on ratio` writes R directly.

The recovery set-point sequence of the paper's Section 4.7, 0.82 → 0.902 → 0.7667, is given as `setpoint_steps` on the `ethane recovery` loop:

```toml
structure = "recovery cascade"
[setpoint_steps]
"ethane recovery" = [[30.0, 0.082], [330.0, -0.1353]]
```

## Disturbances

| Name | Arguments | Effect |
|---|---|---|
| `feed ethane` | `fraction`, `t` | the ethane mole fraction of the feed stepped by `fraction` of itself, the other components scaled to keep the sum at one; the paper's +10 % (Section 4.3) |
| `step`, `ramp` | `field`, `fraction` or `value`, `t` | any input; `F_feed` ±10 % is the paper's throughput disturbance (Sections 4.2–4.4) |

## Measurements

`feed flow` (kmol/h), `TK-100 temperature` (°C), `TK-100 level` (%), `reboiler level` (%), `overhead pressure` (kPa), `methane in NGL` (mol %), `ethane recovery` (–), `NGL flow` (kmol/h), `NGL temperature` (°C), `expander flow` (kmol/h), `expander power` (kW), the E-100 and E-102 hot outlet temperatures, the temperature of the overhead leaving E-100, the booster discharge pressure, the K-101 discharge temperature, `residue gas flow` (kmol/h), and the 30 stage temperatures (°C). The scheme adds measurements as follows:

- **`conventional`:** `TK-101 temperature` and `TK-101 level`.
- **`gsp` and `crr`:** `branch flow`.
- **`crr`:** `recycle flow`, `recycle ratio` and the E-104 outlet temperatures.

Under crr, `stage 1 temperature` is the temperature of the mixing point.

`ethane recovery` is inferred, as in the plant, from the NGL flow transmitter, the ethane in the NGL and the feed: FT_NGL · x_B,C2 / (F_feed · z_C2). `NGL flow` and `recycle ratio` are read through their transmitters.

## Reference results

**The conditions the paper states for the conventional scheme.** These are compared with the conventional design at a recovery of 0.82 (`studies/ngl_demethanizer/designs.py`):

| Quantity | Paper | Case |
|---|---|---|
| TK-101 | −59.6 °C | −59.6 °C |
| TE-100 outlet | −116.3 °C | −116.3 °C |
| TE-100 outlet, vapour fraction | 0.880 | 0.891 |
| TK-100 | −29.3 °C | −40.0 °C |
| E-100 hot outlet | −4.8 °C | −7.5 °C |
| NGL | −9.6 °C | −8.2 °C |
| Residue gas leaving E-100 | −11.7 °C | −12.6 °C |
| Booster discharge | 1265 kPa | 1227 kPa |
| K-101 discharge | 184.4 °C | 179.3 °C |

The TK-100 temperature and the vapour fraction are recorded misses. The warm-end differences follow from the colder separator and the ideal-gas machines.

**The paper's steady-state ordinal result** (its Section 4.1). The conventional column shows no separation at the top, where gsp and crr separate at both ends. The rise over stages 1 to 8 is 1.41 K under conventional, against 11.15 K (gsp) and 10.68 K (crr).

**The HYSYS case, on the `hysys` basis** (`studies/ngl_demethanizer/hysys_comparison.py`):

- All 30 stage temperatures agree within 0.24 K.
- The flows agree within 1 %: overhead 6621 against 6628 kmol/h, recycle 2207 against 2209, NGL 565.8 against 561.1.
- The recovery is 0.830 against 0.820.
- The TE-100 outlet is −97.95 °C at a vapour fraction of 0.893, against −98.07 °C and 0.894.
- The chiller duty is 3462 against 3498 kW, and the reboiler duty 1534 against 1522 kW.

Three quantities are recorded misses:

- the residue gas leaving E-100, which is 2.1 K warm;
- the K-102 power, 28.8 against 25.0 kW;
- the K-101 power, which is 2.4 % high.

All three follow from the ideal-gas machines and the cell exchangers.

**The closed-loop results of the paper's Section 4.** These are not reproduced as reference values. Their settling times depend on a tuning whose ranges the paper does not give. Two runs of the conventional scheme under `basic` are indicative; in each, every loop returns to its set-point within 180 min:

- After a +10 % step in the feed flow, the recovery settles at 0.807, a fall of 1.6 % against the paper's 2 % (Section 4.2).
- After a +10 % step in the feed ethane, the recovery settles at 0.817.

## Verification

`tests/cases/ngl_demethanizer/` holds the case's own tests:

- **`test_design.py`.** Each stored design is a steady state. Every component and the energy are conserved over the plant, and the specifications hold. The conventional design is also reached from scratch, which is the slow test.
- **`test_golden.py`.** The reference configuration (conventional, `chebeir2019`) is checked against the values above, with the paper's tolerances, together with the ordinal result of Section 4.1.
- **`test_hysys_reference.py`.** The `hysys` basis is checked against the HYSYS case, with the misses asserted as misses.
- **`test_structures.py`.** Every structure is a fixed point of every scheme, and `feed ethane` keeps the feed composition normalized. The damping ratios are asserted in the slow tests.
- **Shared modules.** The equation of state and the staged column are tested in the library (`tests/units/test_peng_robinson.py`, `tests/units/test_staged_column.py`): the first against `thermo`, the second against the constant-volatility column.

The library's contract suite runs on every structure.

## Limitations

- **Refrigeration.** The propane cycle is a duty. gsp and crr need TK-100 near −44 °C, below the −42 °C at which propane boils at atmospheric pressure, so a propane chiller would run at sub-atmospheric suction; no refrigeration limit is enforced.
- **Carbon dioxide.** It is absent from the paper's feed and zero on the `hysys` basis, so the freeze-out margin that distinguishes the schemes in practice is not represented.
- **Separator temperature.** The conventional design runs 10.7 K colder than the paper states, and the difference is not explained.
- **Simplified units.** The column pressure is one lumped state, the machines are ideal gases, and the trays have no vapour holdup.
- **Cost.** One right-hand side costs 20 ms (conventional) to 40 ms (crr on `hysys`). A 180 min closed-loop run after a disturbance takes 11 to 13 min, so a dataset over this case is costly.
- **Instruments.** Measurements and valves are ideal unless instruments or actuators are configured, apart from the three flow transmitters, which belong to the plant.
