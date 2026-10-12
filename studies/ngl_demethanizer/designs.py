"""The design points of the three recovery schemes, side by side.

    .venv/bin/python -m studies.ngl_demethanizer.designs

Each scheme is designed to the specifications Chebeir, Salas and Romagnoli (2019) compare
the schemes at: an ethane recovery of 0.82 (ERIC-100), 1 % methane in the NGL (XIC-100),
the column top at 1010 kPa and the vessels half full.  The TK-100 temperature each needs
is then a result of the case; the paper states it, -29.3 C, for the conventional scheme
only.  One of their steady-state findings is compared:

    section 4.1   the conventional scheme shows no separation at the top of the column,
                  where gsp and crr separate at both ends

The separation at each end is read as the temperature rise over the top eight stages and
over stage 26 to the reboiler.  Writes `results/designs.txt`.
"""

from __future__ import annotations

from pathlib import Path

import plantbench as pb
from plantbench import datasets
from plantbench.cases.ngl_demethanizer.definition import CASE
from plantbench.cases.ngl_demethanizer.model import SCHEMES

HERE = Path(__file__).resolve().parent
C = 273.15


def main() -> None:
    lines = []
    say = lines.append
    say("THE DESIGN POINTS OF THE THREE SCHEMES")
    say(datasets.stamp_lines(datasets.stamp())[-1])
    say("")
    rows = {}
    for scheme in SCHEMES:
        d = pb.build(CASE, CASE.config(options={"scheme": scheme})).design
        m = {k: f(d.x, d.u, d.pp) for k, f in CASE.measurements(d).items()}
        st = d.plant.streams(d.x, d.u)
        # Stage k as the case numbers it; under crr stage 1 is the mixing point above
        # the trays.
        T = [m[f"stage {k} temperature"] for k in range(1, d.pp.n_stages + 1)]
        T.append(m["NGL temperature"])
        rows[scheme] = {
            "TK-100 temperature, C": m["TK-100 temperature"],
            "ethane recovery": m["ethane recovery"],
            "methane in NGL, mol %": m["methane in NGL"],
            "NGL, kmol/h": m["NGL flow"],
            "residue gas, kmol/h": m["residue gas flow"],
            "expander flow, kmol/h": m["expander flow"],
            "branch, kmol/h": 60 * st.F_branch,
            "recycle, kmol/h": 60 * st.F_recycle,
            "chiller duty, kW": d.u.Q_chiller,
            "reboiler duty, kW": d.u.Q_reboiler,
            "expander power, kW": m["expander power"],
            "K-101 power, kW": d.u.W_recompressor,
            "K-102 power, kW": d.u.W_K102,
            "column top, C": T[0],
            "stage 8, C": T[7],
            "stage 26, C": T[25],
            "rise over stages 1-8, K": T[7] - T[0],
            "rise over stage 26 to reboiler, K": T[-1] - T[25],
            "E-100 hot outlet, C": m["E-100 hot outlet temperature"],
            "E-102 hot outlet, C": m["E-102 hot outlet temperature"],
            "overhead leaving E-100, C": m["overhead temperature leaving E-100"],
            "booster discharge, kPa": m["booster discharge pressure"],
        }
    say(f"{'':36s}" + "".join(f"{s:>14s}" for s in SCHEMES))
    for key in rows["conventional"]:
        say(f"{key:36s}" + "".join(f"{rows[s][key]:14.4g}" for s in SCHEMES))
    say("")
    top = {s: rows[s]["rise over stages 1-8, K"] for s in SCHEMES}
    say("section 4.1, separation at the top (rise over stages 1-8): "
        + ", ".join(f"{s} {top[s]:.2f} K" for s in SCHEMES))

    out = HERE / "results" / "designs.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
