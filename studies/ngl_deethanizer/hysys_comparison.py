"""The design point against the HYSYS case it is constructed from.

    .venv/bin/python -m studies.ngl_deethanizer.hysys_comparison

Solves the design at the HYSYS set-points and compares it with the HYSYS steady state
(`ngl_deethanizer/hysys_reference.json`): the products, the recoveries, the duties, the
reflux and boil-up, the condenser and reboiler temperatures, the heater outlet, and the
temperature and light-component profiles of the column.  The condenser duty HYSYS
reports is set beside the duty its own stream enthalpies imply, the vapour to the
condenser less the reflux and distillate.  Writes `results/hysys_comparison.txt`, from
which the tolerances of `tests/test_golden.py` are argued.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_deethanizer import hysys_reference as H
from plantbench.cases.ngl_deethanizer.definition import CASE, measurements

HERE = Path(__file__).resolve().parent
C = 273.15


def rows(d) -> list[tuple[str, float, float]]:
    """(quantity, HYSYS, this case) for the design `d`."""
    ref = H.load()
    S, col = ref["streams"], ref["column"]
    m = measurements(d)
    y = {k: f(d.x, d.u, d.pp) for k, f in m.items()}
    st = d.plant.streams(d.x, d.u)
    s = d.plant.layout.unpack(d.x)
    # HYSYS's condenser duty from its own streams: vapour in, reflux and distillate out.
    q_streams = (S["To Condenser @COL2"]["F_kmol_h"] * S["To Condenser @COL2"]["h_kJ_kmol"]
                 - (S["Reflux @COL2"]["F_kmol_h"] + S["3"]["F_kmol_h"])
                 * S["3"]["h_kJ_kmol"]) / 3600
    return [
        ("distillate, kmol/h", S["3"]["F_kmol_h"], y["distillate flow"]),
        ("LPG, kmol/h", S["4"]["F_kmol_h"], y["LPG flow"]),
        ("ethane in distillate", S["3"]["z"][2], y["ethane in distillate"]),
        ("methane in distillate", S["3"]["z"][1], float(s["x_D"][1])),
        ("propane in distillate", S["3"]["z"][3], y["propane in distillate"]),
        ("ethane in LPG", S["4"]["z"][2], y["ethane in LPG"]),
        ("ethane recovery", ref["ethane_recovery"], y["ethane recovery"]),
        ("propane recovery", 141.1325 / 147.0064, y["propane recovery"]),
        ("reflux, kmol/h", col["reflux_kmol_h"], y["reflux flow"]),
        ("vapour to condenser, kmol/h", col["V_kmol_h"][0], 60 * st.column.V[0]),
        ("boil-up, kmol/h", col["boilup_kmol_h"], 60 * st.column.V[-1]),
        ("E-100 duty, kW", ref["duty_kJ_h"]["E-100"] / 3600, y["E-100 duty"]),
        ("condenser duty, kW", ref["duty_kJ_h"]["condenser"] / 3600, y["condenser duty"]),
        ("  condenser duty from HYSYS's streams, kW", q_streams, y["condenser duty"]),
        ("reboiler duty, kW", ref["duty_kJ_h"]["reboiler"] / 3600, y["reboiler duty"]),
        ("E-100 outlet vapour fraction", S["2"]["VF"], st.vapour_heated),
        ("E-100 enthalpy rise, kJ/kmol", S["2"]["h_kJ_kmol"] - S["1"]["h_kJ_kmol"],
         st.h_heated - st.h_feed),
        ("drum temperature, C", S["3"]["T_C"], y["condenser temperature"]),
        ("reboiler temperature, C", S["4"]["T_C"], y["reboiler temperature"]),
        ("stage 1 temperature, C", col["T_C"][0], y["stage 1 temperature"]),
        ("stage 4 temperature, C (spec)", col["T_C"][3], y["stage 4 temperature"]),
        ("stage 15 temperature, C", col["T_C"][14], y["stage 15 temperature"]),
        ("stage 28 temperature, C", col["T_C"][27], y["stage 28 temperature"]),
        ("stage 30 temperature, C", col["T_C"][29], y["stage 30 temperature"]),
    ]


def profiles(d) -> dict:
    """The stage temperatures (C) and the ethane and propane liquid fractions, of HYSYS
    and of this case."""
    col = H.load()["column"]
    st = d.plant.streams(d.x, d.u)
    x = d.plant.layout.unpack(d.x)["x"]
    x_h = np.array(col["x_N2_C1_C2_C3"])
    return {"T_hysys": np.array(col["T_C"]), "T_model": st.column.T[:-1] - C,
            "C2_hysys": x_h[:, 2], "C2_model": x[:, 2],
            "C3_hysys": x_h[:, 3], "C3_model": x[:, 3]}


def main() -> None:
    d = pb.build(CASE, CASE.config()).design
    lines = []
    say = lines.append
    say("THE DESIGN POINT AGAINST THE HYSYS CASE")
    say(datasets.stamp_lines(datasets.stamp())[-1])
    say("")
    say(f"{'':34s}{'HYSYS':>12s}{'model':>12s}{'difference':>12s}{'relative':>10s}")
    for name, h, mod in rows(d):
        rel = f"{(mod - h) / h:10.2%}" if abs(h) > 1e-9 else ""
        say(f"{name:34s}{h:12.5g}{mod:12.5g}{mod - h:12.4g}{rel}")

    p = profiles(d)
    T_h, T_m = p["T_hysys"], p["T_model"]
    x_h = np.stack([p["C2_hysys"], p["C3_hysys"]], axis=1)
    x_m = np.stack([p["C2_model"], p["C3_model"]], axis=1)
    say("")
    say("column profile, stage by stage")
    say(f"{'stage':>6s}{'T HYSYS':>10s}{'T model':>10s}{'dT':>8s}"
        f"{'C2 HYSYS':>10s}{'C2 model':>10s}{'C3 HYSYS':>10s}{'C3 model':>10s}")
    for k in range(len(T_h)):
        say(f"{k + 1:6d}{T_h[k]:10.2f}{T_m[k]:10.2f}{T_m[k] - T_h[k]:8.2f}"
            f"{x_h[k, 0]:10.4f}{x_m[k, 0]:10.4f}{x_h[k, 1]:10.4f}{x_m[k, 1]:10.4f}")
    say(f"largest stage temperature difference {np.max(np.abs(T_m - T_h)):.2f} K, "
        f"largest ethane difference {np.max(np.abs(x_m[:, 0] - x_h[:, 0])):.4f}, "
        f"propane {np.max(np.abs(x_m[:, 1] - x_h[:, 1])):.4f}")

    out = HERE / "results" / "hysys_comparison.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
