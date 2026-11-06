"""The template case, a gravity-drained tank with a level loop, as a public import.

    import plantbench as pb
    from plantbench.cases.template import CASE
    traj = pb.run(CASE, CASE.config(disturbance={"name": "step", "field": "F_in",
                                                  "fraction": 0.2, "t": 5.0}), t_end=60.0)

The template is the case `plantbench new-case` copies, and the smallest working plant in
the library: one state, one loop.  It is not registered, so `load_case("template")`
raises and it is absent from `list_cases`; a dataset or a study that wants it passes the
`Case` itself.  Its source is in `plantbench/cases/_template/`.
"""

from __future__ import annotations

from plantbench.cases._template.definition import CASE, STRUCTURES
from plantbench.cases._template.model import Design, Inputs, TankParameters, solve_design

__all__ = ["CASE", "STRUCTURES", "Design", "Inputs", "TankParameters", "solve_design"]
