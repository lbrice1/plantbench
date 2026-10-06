# `reactor_separator_recycle`: reactor–separator–recycle plant with heat integration

```python
import plantbench as pb
case = pb.load_case("reactor_separator_recycle")
```

## Provenance

The plant is an extension of the reactor control example of Romagnoli and Palazoglu (2020). The reactor is their non-isothermal CSTR, with three modifications to its published parameter table recorded in `plantbench/units/parameters.py`. The model, the code and the analyses are original. The case is frozen. Its reference configuration reproduces `studies/reactor_separator_recycle/results/reference.txt` and `designs.txt` byte for byte, which `tests/cases/reactor_separator_recycle/test_golden.py` checks. Any extension is an option that is off by default.

## Process

Fresh feed, reactant A with 0.5 mol% of a light inert I, joins the recycle and enters a jacketed CSTR, where A → B runs first-order and exothermic. The reactor sits on the open-loop unstable middle branch of its steady-state multiplicity. Its effluent feeds a thirty-stage column (feed on stage 15, 1 = top) with constant relative volatilities 6 : 3 : 1 for I : A : B. The distillate carries the unreacted A and the inert and is split into a recycle and a purge. Product B leaves as bottoms. Column pressure is not a state; the pressure loop is taken to be perfect.

Four heat-exchanger networks can be placed on the plant:

| Network | Description | Column pressure |
|---|---|---|
| D1 | no heat integration; a trim heater carries the whole feed preheat | 0.8 bar |
| D3 | feed–effluent exchanger | 0.4 bar |
| D4 | vacuum column; reactor effluent and a reactor pump-around (share 0.7 of the jacket duty) to the reboiler, then a feed–effluent exchanger | 0.15 bar |
| D5 | condenser heat to the reactor feed | 0.8 bar |

D2 is the pinch target, a bound rather than a network, and is computed by `studies/reactor_separator_recycle/designs.py`.

The design point is set by specifications as well as balances. The plantwide balances determine the reflux and the recycle flow but not the inert level at the reactor inlet, because the product balance follows from the product recycle balance and the column's own; the steady-state column equations do not involve the reflux drum and column base holdups. All three are therefore specified in `plant.solve_design`, at the values the reference results were produced at: an inert level of 0.548524 kmol/m³ (`C_I0_REFERENCE`), a drum holdup of 12 kmol (the column's nominal `drum_holdup`) and a base holdup of 6.896 kmol (`BASE_HOLDUP_REFERENCE`). Before they were specified, the solver's path set them, and the design point then differed between platforms.

## States

| Plant | States | Layout |
|---|---|---|
| base (`network = "none"`) | 134 | `V, C_I, C_A, C_B, T, Tj`; `M_D, xD_*`; `M_1 … M_30`; `x_1_I … x_30_B`; `M_B, xB_*` |
| heat-integrated | 134 + 1 + exchangers | the base states, then the reactor inlet temperature `T_in` and one duty per exchanger |

Units are minutes, m³, kmol and kelvin. `case.state_names(design)` gives the names in order.

## Inputs

| Input | Unit | Meaning |
|---|---|---|
| `F_fresh` | kmol/min | fresh feed |
| `z_fresh` | – | fresh feed composition [I, A, B] |
| `Fj` | m³/min | jacket coolant flow |
| `F_out` | m³/min | reactor effluent, the column feed |
| `V_boil` | kmol/min | boil-up |
| `L_reflux` | kmol/min | reflux |
| `D`, `P`, `Bm` | kmol/min | distillate, purge, bottoms; the recycle is `D − P` |
| `T_fresh` | K | reactor inlet temperature, held in the base plant |
| `Q_feed_trim` | kJ/min | feed trim heater, heat-integrated plants only |
| `T_storage` | K | fresh feed temperature as delivered, heat-integrated plants only |

## Options

| Option | Default | Values |
|---|---|---|
| `network` | `"none"` | `"none"`, `"D1"`, `"D3"`, `"D4"`, `"D5"` |
| `pump_around_share` | `None` | with `network = "D4"`, a share of the jacket duty in [0, 1) sent to the reboiler |
| `V_boil` | `28.0` | design boil-up, kmol/min, which sets the separation; only the default is supported, see below |

Parameter overrides address `PlantParameters` by dotted path, for example `params = {"reactor.U": 300.0}` or `{"column.P_base": 0.6}`. The plant is re-designed at the overridden values.

Two of them bear on the design space. `column.P_base` is the pressure of the unintegrated plant. `column.P_network` is the pressure a heat-integrated plant runs at, overriding the pressure its network was defined for; it is `None` by default, which keeps that pressure. Pressure moves every temperature in the column and leaves the separation unchanged, so it decides which matches a network can make, and it carries the tray temperature loop with it. It is a parameter rather than an option because adding an option would change the run id of every configuration ever made from this case.

The design boil-up is not a design variable in practice. The plantwide design is solved by `fsolve` over two unknowns, the reflux and the recycle, with the inert level specified, and the inner column solve warm-starts from the previous residual evaluation, so the function `fsolve` differences depends on its own history. Of fifteen boil-ups tried between 20 and 32 kmol/min (every integer, and 27.5 and 27.8), the design solves at 20, 21, 24, 25, 27.8, 28 and 30, raises at 22, 23, 26, 27, 27.5, 29 and 31, and had not returned at 32 after 30 minutes. A design is accepted on its residual alone, at most 1e-8 over the two balances solved and the product balance, whatever MINPACK reports; at 26, 27.5 and 31 MINPACK reports convergence and the design is rejected, with residuals of 1.6e17, 1.9e-8 and 1.8e20. Every accepted design has a design residual below 4e-12 and a closed-loop residual below 5e-13 at the design point, within the 1e-9 the contract asks of a design. Across the accepted boil-ups the recycle (9.230 kmol/min) and the purge (0.2343 kmol/min) do not change, and the least damping ratio of the reference structure lies between 0.157 and 0.159.

## Features

Recorded beside each run of a dataset when named in `[run] features`, and computed from the design point rather than the trajectory.

| Feature | Values |
|---|---|
| `spectrum` | `damping_ratio`, `rightmost`, `n_unstable` of the closed loop linearized at the design point; provided by every case |
| `economics` | `profit` and `utility_cost` in \$/h, `gwp_energy` and `gwp_total` in t CO2e/h, `recovered` in GJ/h, at the design operating point, on the basis the design comparison uses to compare the networks |

## Control structures

| Structure | Production set by | Recycle loop | Notes |
|---|---|---|---|
| `recycle free` | fresh feed | no flow fixed | reactor level on the effluent |
| `effluent fixed` | reactor effluent | effluent flow fixed | fresh feed on reactor level |
| `effluent fixed + cascade` | reactor effluent | effluent flow fixed | adds a reactor composition loop on the temperature set-point; the reference structure |
| `distillate fixed` | fresh feed | distillate fixed | has no steady state under a throughput increase |

Every structure holds a stripping-section tray temperature (tray 17) on the boil-up and ratios the reflux to the column feed, except `distillate fixed`, where the reflux holds the drum level. A heat-integrated plant supports `effluent fixed + cascade` and `effluent fixed`, each with a feed-temperature loop on the trim heater added.

Loop names for set-points, tuning and instruments: `reactor T`, `reactor C_A`, `reactor level`, `drum level`, `base level`, `tray T`, `reflux/feed` (a ratio station), and `feed T` in the heat-integrated plants.

## Disturbances

| Name | Arguments | Effect |
|---|---|---|
| `throughput` | `fraction`, `t` | a production-rate change through whichever input sets production in the structure |
| `inert` | `fraction`, `t` | the inert content of the fresh feed changed by a fraction |
| `purge_closed` | `t` | the purge shut; the inert has no exit and there is no steady state |
| `fresh_feed_temperature` | `change`, `t` | the fresh feed arriving `change` K from its storage temperature; heat-integrated plants only |
| `step`, `ramp` | `field`, `fraction` or `value`, `t` | any input, generic to every case |

## Measurements

`reactor V`, `reactor T`, `reactor C_A`, `drum level`, `base level`, `bottoms x_A`, `tray 17 T` (the bubble point at the plant's column pressure), `recycle`, and `feed T` in the heat-integrated plants.

## Reference results

From `studies/reactor_separator_recycle/results/`:

- The design point is reactor C_A = 2.3564 kmol/m³, T = 362.40 K, T_j = 345.78 K; fresh feed 8.690, recycle 9.230, purge 0.2343 and product 8.455 kmol/min.
- A 10% throughput increase raises the recycle by 5.22 times the production increase with the recycle free and 4.51 times with the effluent fixed. With the composition cascade the ratio falls to 1.46.
- With the distillate fixed, a 10% fresh-feed increase fills the reflux drum without bound. With the purge closed, the inert accumulates in exact agreement with the inert fed.
- The least damping ratio of the reference structure is 0.158 in the base plant. In the heat-integrated plants it is 0.155 (D1), 0.157 (D3), 0.013 (D4) and 0.155 (D5). Under a 10% throughput increase D4 settles into a sustained oscillation of the reactor temperature between 356.05 and 371.87 K, with a period of 22.8 min and the jacket valve at a limit 75% of the time.

## Verification

`tests/units/test_reactor.py` checks the reactor against its published steady state and eigenvalues, and gates everything else. `tests/cases/reactor_separator_recycle/` asserts conservation (column material balance, mole fractions, whole-plant energy balance at three pressures, inert accumulated against inert fed), analytic limits, the pinch bound on every network, and the design point as a fixed point of every structure. `test_golden.py` reproduces the reference results.

## References

- Luyben, W. L., Tyréus, B. D. and Luyben, M. L. (1998). *Plantwide Process Control*. McGraw-Hill.
- Romagnoli, J. A. and Palazoglu, A. (2020). *Introduction to Process Control*, 3rd edition. CRC Press.

## Limitations

- Constant molar overflow and constant relative volatility; column pressure moves temperatures but not the separation.
- Perfect pressure control; one latent heat for all species; one reaction.
- The temperature tray is chosen at 0.8 bar and kept at every pressure.
- No safety layer, start-up or shutdown, fouling or capital cost.
- Measurements and valves are ideal unless instruments or actuators are configured.
- The design solve fails at about half the design boil-ups tried (see Options).
