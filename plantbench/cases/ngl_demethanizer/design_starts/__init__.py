"""Stored design points, one per scheme and basis, from which Newton finds the design in seconds.

The design of this case is a Newton solve on some 300 states and the design inputs,
which converges only from nearby.  Reaching the design from a rough state takes an
integration in hold mode (`model.settle`) of several minutes, so the design point at the
default parameters is kept here, as JSON, and `start` returns it as the starting point
for `model.solve_design`.  A parameter override starts from it too, by continuation.

    python -m ngl_demethanizer.design_starts [scheme ...] [--basis B] [--write]

solves each scheme from scratch (settle, then Newton) and compares the result with the
stored point; `--write` stores it.  A stored point is only a starting point: the design
itself is always solved, so a stale file costs time, not correctness.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _path(scheme: str, basis: str) -> Path:
    return HERE / f"{scheme}-{basis}.json"


def start(plant):
    """The stored design point of the plant's scheme, as (x, inputs), or None.  Exchanger
    cells are interpolated when `n_cells` differs from the stored one; a different number
    of stages or components has no stored point."""
    from ..model import default_inputs

    path = _path(plant.scheme, plant.basis)
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    blocks = {k: np.asarray(v, dtype=float) if isinstance(v, list) else float(v)
              for k, v in stored["state"].items()}
    if set(blocks) != set(plant.layout.shapes):
        return None
    n = plant.pp.n_cells
    for k, v in blocks.items():
        if k.startswith("T_E1") and len(v) != n:
            blocks[k] = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)
    try:
        x = plant.layout.pack(**blocks)
    except ValueError:
        return None
    u = replace(default_inputs(plant), **{k: v for k, v in stored["inputs"].items()
                                          if k != "z_feed"})
    return x, u


def write(design) -> Path:
    """Store a design as the starting point of its scheme."""
    plant = design.plant
    state = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in plant.layout.unpack(design.x).items()}
    inputs = {f.name: getattr(design.u, f.name) for f in fields(design.u)
              if f.name != "z_feed"}
    path = _path(plant.scheme, plant.basis)
    path.write_text(json.dumps({"scheme": plant.scheme, "basis": plant.basis,
                                "n_cells": plant.pp.n_cells, "state": state,
                                "inputs": inputs}, indent=1) + "\n")
    return path


def main(argv=None) -> None:
    import argparse
    import time

    from ..model import SCHEMES, Plant, solve_design
    from ..parameters import parameters

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schemes", nargs="*", default=list(SCHEMES))
    parser.add_argument("--basis", default="chebeir2019")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    pp = parameters(args.basis)
    for scheme in args.schemes:
        begin = time.perf_counter()
        design = solve_design(pp, scheme, args.basis)
        took = time.perf_counter() - begin
        stored = start(Plant(pp, scheme, args.basis))
        if stored is None:
            report = "nothing stored"
        else:
            report = (f"largest difference from the stored point "
                      f"{np.max(np.abs(design.x - stored[0])):.2g}")
        print(f"{scheme}, {args.basis}: solved from scratch in {took:.0f} s; {report}")
        if args.write:
            print(f"  written to {write(design)}")


if __name__ == "__main__":
    main()
