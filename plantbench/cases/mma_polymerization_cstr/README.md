# `mma_polymerization_cstr`: isothermal MMA polymerization CSTR

```python
import plantbench as pb
case = pb.load_case("mma_polymerization_cstr")
```

## Provenance

The reactor of Doyle, Ogunnaike and Pearson (1995), *Nonlinear model-based control using second-order Volterra models*, Automatica 31(5), 697–714, doi:10.1016/0005-1098(94)00150-H, Section 5.1, hereafter referred to as the paper. The paper takes its rate laws from earlier work it cites (Congalidis et al., 1989; Daoutidis et al., 1990; Ray, 1972). It gives the model (its Eq. (76)), the kinetic and system parameters (Tables 1 and 2), the operating point, and the model with the parameters substituted (Eq. (77)).

In `model.py`, `MMAParameters` holds the constants of Tables 1 and 2 in the paper's units, each [source]. The paper's time is the hour and the library's the minute, so the case's right-hand side is the paper's divided by 60; the flows stay in m³/h. Table 2 prints the feed flow F in m³; the balances make it a flow, 1 m³/h. The grades, the initiator pump's range, the loop's tuning and the measurement spans are not in the paper, and each is a [choice].

## Process

Methyl methacrylate is polymerized by free radicals in a jacketed CSTR, with azobisisobutyronitrile as initiator and toluene as solvent. The paper's assumptions hold: the reactor is isothermal at 335 K, perfectly mixed and of constant volume; there is no gel effect and no polymer in the feed; the initiator flow is small beside the monomer flow; and the live radicals are at a quasi-steady state. The product grade is the number-average molecular weight, NAMW = D_1 / D_0, which the initiator flow sets: more initiator gives more chains, and shorter ones.

At the design point the conversion is 0.082 and the residence time 6 min. The open-loop eigenvalues lie between −0.182 and −0.167 1/min. The gain of the NAMW to the initiator flow changes twelvefold between the grades: −2.76 × 10⁶ kg/kmol per m³/h at grade B, −6.08 × 10⁵ at A and −2.31 × 10⁵ at C.

## States and inputs

| State | Unit | | Input | Unit |
|---|---|---|---|---|
| `C_m`, monomer | kmol/m³ | | `F_I`, initiator flow | m³/h |
| `C_I`, initiator | kmol/m³ | | `C_Iin`, initiator in its feed | kmol/m³ |
| `D_0`, zeroth moment of the dead polymer | kmol/m³ | | | |
| `D_1`, first moment | kg/m³ | | | |

## Options

None. Parameter overrides address `MMAParameters` directly: `params = {"F_I": 0.0046}` designs the reactor at another initiator flow, for example the grade-B flow of `grade_flow`.

## Grades

`GRADES` holds three product grades, as NAMW targets [choice]. `grade_flow(target)` gives the initiator flow of each.

| Grade | NAMW, kg/kmol | F_I, m³/h | Conversion |
|---|---|---|---|
| A | 25 000 | 0.016784 | 0.082 |
| B | 40 000 | 0.004603 | 0.045 |
| C | 18 000 | 0.035776 | 0.116 |

A grade change is a set-point step of the NAMW loop:

```toml
structure = "NAMW PI"
[setpoint_steps]
"NAMW" = [[10.0, 14999.5], [70.0, -22000.0]]
```

## Control structures

| Structure | Loops |
|---|---|
| `open loop` | none |
| `NAMW PI` | `NAMW` on `F_I`, K_c = −2 × 10⁻⁷ m³/h per kg/kmol, τ_I = 10 min, F_I within 0.002 to 0.08 m³/h; the reference |

A fixed PI tuning is a compromise across the grades, since the process gain changes twelvefold between them. The tuning keeps grade B, where the gain is largest, damped with a 5 min analyzer dead time. At grade C, where the gain is smallest, the loop is slow.

| Grade | Damping ratio | Slowest mode, min | With a 5 min analyzer: damping, slowest mode |
|---|---|---|---|
| A | 1.000 | 80.0 | 0.942, 74.7 |
| B | 0.833 | 17.9 | 0.618, 14.1 |
| C | 1.000 | 213.9 | 1.000, 208.8 |

A damping ratio of 1 means no oscillatory mode.

An online analyzer with a dead time is an `instruments` entry:

```toml
[instruments.NAMW]
deadtime = 5.0
```

## The engineering model

`ENGINEERING` is a `mismatch` section: the reactor's balances without chain transfer to monomer (`Z_fm = 0`), with the initiator efficiency `f_star` fitted so that the model gives the reactor's NAMW at the design flow, 0.969 against 0.58. It stands for a model tuned on a plant's main grade. At the grade-B flow it gives 50 214.8 kg/kmol against the reactor's 40 000, and at the grade-C flow 16 344.0 against 18 000.

```python
from plantbench.cases.mma_polymerization_cstr.definition import ENGINEERING
config = case.config("open loop", mismatch=ENGINEERING)
```

## Disturbances

| Name | Arguments | Effect |
|---|---|---|
| `step`, `ramp` | `field`, `fraction` or `value`, `t` | any input, for example `C_Iin` |

## Measurements

`NAMW` (kg/kmol), `conversion` (1 − C_m / C_m,in), and the four states. The limits give spans only [choice]: NAMW 0 to 60 000 kg/kmol, conversion 0 to 1.

## Reference results

The design is the paper's operating point: F_I = 0.016783 m³/h, x = (5.50677, 0.132906, 0.0019752, 49.3818) and NAMW = 25 000.5 kg/kmol. The case gives (5.506775, 0.1329050, 0.00197523, 49.38173) and NAMW = 25 000.504. Its C_I and D_1 lie one unit in the paper's last digit below the printed values; C_I follows from the initiator balance alone, F_I C_Iin / (k_I V + F). The rate constants at 335 K give the coefficients of Eq. (77) to a relative 6 × 10⁻⁵; the largest difference is in the termination coefficient, 0.112184 against the printed 0.112191.

## Verification

`tests/cases/mma_polymerization_cstr/` checks:

- the coefficients of Eq. (77);
- the operating point, and that the design is a steady state;
- the monomer and initiator balances;
- that the NAMW falls with the initiator flow;
- the grade flows;
- the engineering model, at the design point and as a mismatch;
- a grade change under `NAMW PI`, and that the loop stabilizes every grade.

The right-hand side and the measurements take a batch of states. `tests/cases/test_reactor_batch.py` checks a batch against its rows and an ensemble against `run`; on a GPU, `tests/core/test_gpu.py` checks the device against the host. The library's contract suite runs on every structure.

## Limitations

- Isothermal: the jacket and the reactor temperature are not modeled.
- No gel effect, and the radicals at a quasi-steady state, as in the paper.
- The molecular-weight distribution is described by two moments only, so the polydispersity is not available.
- Measurements and valves are ideal unless instruments or actuators are configured.
