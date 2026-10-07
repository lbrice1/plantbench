"""The crr design on the hysys basis against the HYSYS case it reproduces.

    .venv/bin/python -m studies.ngl_demethanizer.hysys_comparison

The HYSYS case "Demethanizer - Tuned" is a CRR plant on its own feed, and its reports give
the stream table and column profile the paper does not (`hysys_reference.json`).  The
design on the hysys basis is solved to the same specifications (TK-100 at -38.71 C, the
top at 1010 kPa, 1 % methane in the NGL, the recycle at half the residue gas, the vessels
at the HYSYS levels), and everything else is compared: stream temperatures, flows, shaft
powers and duties, the column temperature profile and the ethane recovery.  Writes
`results/hysys_comparison.txt`.

The E-104 cold side is at the JTV-101 discharge pressure in the model and falls by 171 kPa
across the exchanger in HYSYS, so the two report its outlet at different pressures; the
model's row is at the stage-2 pressure, after the let-down, for comparability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_demethanizer import hysys_reference as H
from plantbench.cases.ngl_demethanizer.definition import CASE

HERE = Path(__file__).resolve().parent
C = 273.15
KPA = 1e3


def rows(d) -> list[tuple[str, float, float, str]]:
    """(quantity, HYSYS, model, unit) for every compared quantity."""
    plant, x, u = d.plant, d.x, d.u
    s = plant.layout.unpack(x)
    st = plant.streams(x, u)
    m = {k: f(x, u, d.pp) for k, f in CASE.measurements(d).items()}
    eos = plant.eos
    P = st.column.P / KPA

    def flash_h(P_kPa, h, z, T0):
        f = eos.flash_PH(np.array([P_kPa * KPA]), np.array([h]), np.asarray(z)[None],
                         T0=np.array([T0]))
        return f.T[0] - C, f.V[0]

    # Trays are stages 2-30 under crr: the expander's stage 8 is tray 6, stage 2 tray 0.
    T17, V17 = flash_h(P[6], st.h_expander_out, st.y_expander, s["T_TK100"] - 60)
    # The branch leaving E-104 let down to stage 2, as HYSYS reports stream 22.
    F_bv = st.V1 / (1 + u.split)
    F_bl = u.liquid_split * st.L100
    z_br = (F_bv * st.y_expander + F_bl * s["x_TK100"]) / (F_bv + F_bl)
    h22 = eos.flash_PT(np.array([s["T_E104c"][0]]),
                       np.array([(P[0] + d.pp.dP_E104c) * KPA]), z_br[None]).h[0]
    T22, V22 = flash_h(P[0], h22, z_br, s["T_E104c"][0])
    ref = H.load()
    S = ref["streams"]
    pw = ref["power_kW"]
    out = [
        ("E-100 hot outlet (4)", S["4"]["T_C"], m["E-100 hot outlet temperature"], "C"),
        ("residue gas leaving E-102 (3)", S["3"]["T_C"], s["T_E102c"][0] - C, "C"),
        ("residue gas leaving E-100 (5)", S["5"]["T_C"],
         m["overhead temperature leaving E-100"], "C"),
        ("E-102 hot outlet (20)", S["20"]["T_C"], m["E-102 hot outlet temperature"], "C"),
        ("E-104 hot outlet (24)", S["24"]["T_C"], m["E-104 hot outlet temperature"], "C"),
        ("branch to stage 2 (22)", S["22"]["T_C"], T22, "C"),
        ("branch to stage 2, vapour fraction", S["22"]["VF"], V22, "-"),
        ("TE-100 outlet (17)", S["17"]["T_C"], T17, "C"),
        ("TE-100 outlet, vapour fraction", S["17"]["VF"], V17, "-"),
        ("booster discharge (6)", S["6"]["P_kPa"], m["booster discharge pressure"], "kPa"),
        ("booster discharge (6)", S["6"]["T_C"], st.T_booster_out - C, "C"),
        ("K-101 discharge (Sales Gas)", S["Sales Gas"]["T_C"],
         m["K-101 discharge temperature"], "C"),
        ("overhead (28)", S["28"]["F_kmol_h"], 60 * st.V0, "kmol/h"),
        ("residue gas (3)", S["3"]["F_kmol_h"], m["residue gas flow"], "kmol/h"),
        ("recycle (26)", S["26"]["F_kmol_h"], m["recycle flow"], "kmol/h"),
        ("branch (18)", S["18"]["F_kmol_h"], m["branch flow"], "kmol/h"),
        ("expander (17)", S["17"]["F_kmol_h"], m["expander flow"], "kmol/h"),
        ("TK-100 liquid (8)", S["8"]["F_kmol_h"], 60 * st.L100, "kmol/h"),
        ("NGL", S["NGL"]["F_kmol_h"], m["NGL flow"], "kmol/h"),
        ("NGL temperature", S["NGL"]["T_C"], m["NGL temperature"], "C"),
        ("ethane in NGL", S["NGL"]["z"][3], s["x_B"][3], "-"),
        ("ethane recovery", ref["ethane_recovery"], m["ethane recovery"], "-"),
        ("chiller duty", pw["chiller"], u.Q_chiller, "kW"),
        ("reboiler duty", pw["reboiler"], u.Q_reboiler, "kW"),
        ("TE-100 power", pw["expander"], m["expander power"], "kW"),
        ("K-101 power", pw["K-101"], u.W_recompressor, "kW"),
        ("K-102 power", pw["K-102"], u.W_K102, "kW"),
    ]
    for k, Tk in enumerate(ref["column"]["T_C"]):
        out.append((f"stage {k + 1}", Tk, m[f"stage {k + 1} temperature"], "C"))
    return out


def main() -> None:
    d = pb.build(CASE, CASE.config(options={"scheme": "crr", "basis": "hysys"})).design
    lines = []
    say = lines.append
    say("THE CRR DESIGN ON THE HYSYS BASIS AGAINST THE HYSYS CASE")
    say(datasets.stamp_lines(datasets.stamp())[-1])
    say("")
    say(f"{'quantity':40s}{'HYSYS':>12s}{'model':>12s}{'difference':>12s}  unit")
    for name, ref, model, unit in rows(d):
        say(f"{name:40s}{ref:12.5g}{model:12.5g}{model - ref:12.3g}  {unit}")
    out = HERE / "results" / "hysys_comparison.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
