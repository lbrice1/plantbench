"""Helpers for writing a case: the pieces every case repeats.

    StateLayout          named blocks of the state vector: pack, unpack, state names
    state_measurement    a measurement that reads one state
    measured             a measure's value, for one state or a batch
    cached_design        cache a slow design solve per worker process
    fixed_point_residual how far the design point is from a steady state, and where

None of these is required; a case that writes the same thing by hand keeps the contract
equally well.  They know nothing about any plant.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from plantbench.backend import namespace, to_host


class StateLayout:
    """The state vector as named blocks, in order, each a scalar or an array.

        layout = StateLayout([("M_D", ()), ("x_D", 3), ("M", 20), ("x", (20, 3))])
        blocks = layout.unpack(x)          # {"M_D": float, "x_D": (3,), "x": (20, 3), ...}
        x = layout.pack(**blocks)
        layout.names()                     # ["M_D", "x_D[0]", ..., "x[19,2]"]

    A shape is `()` for a scalar, an int for a vector or a tuple for an array; an array
    block is stored row by row.

    A batch of m states, an array (m, size), unpacks to blocks with the same leading
    axis, a scalar block to (m,), and packs back from them, on NumPy or CuPy alike.
    """

    def __init__(self, blocks: Sequence[tuple[str, int | tuple[int, ...]]]):
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.slices: dict[str, slice] = {}
        i = 0
        for name, shape in blocks:
            if name in self.shapes:
                raise ValueError(f"block {name!r} named twice")
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            n = int(np.prod(shape, dtype=int))
            self.shapes[name], self.slices[name] = shape, slice(i, i + n)
            i += n
        self.size = i

    def unpack(self, x: np.ndarray) -> dict[str, float | np.ndarray]:
        """The blocks of `x` by name: a float for a scalar block, a view for an array."""
        if x.shape[-1] != self.size:
            raise ValueError(f"{x.shape[-1]} states for a layout of {self.size}")
        if x.ndim == 1:
            return {name: float(x[s.start]) if self.shapes[name] == () else
                    x[s].reshape(self.shapes[name]) for name, s in self.slices.items()}
        lead = x.shape[:-1]
        return {name: x[..., s.start] if self.shapes[name] == () else
                x[..., s].reshape(lead + self.shapes[name]) for name, s in self.slices.items()}

    def pack(self, **blocks) -> np.ndarray:
        """The state vector from every block by name, with the leading axis of the blocks
        if they carry one."""
        missing = set(self.shapes) - set(blocks)
        extra = set(blocks) - set(self.shapes)
        if missing or extra:
            raise KeyError(f"blocks missing {sorted(missing)}, unknown {sorted(extra)}")
        xp = namespace(*blocks.values())
        values = {k: xp.asarray(v, dtype=float) for k, v in blocks.items()}
        lead = np.broadcast_shapes(*(v.shape[:v.ndim - len(self.shapes[k])]
                                     for k, v in values.items()))
        x = xp.empty(lead + (self.size,))
        for name, s in self.slices.items():
            v = values[name]
            x[..., s] = xp.broadcast_to(v, lead + self.shapes[name]).reshape(lead + (-1,))
        return x

    def names(self, labels: dict[str, Sequence[str]] | None = None) -> list[str]:
        """One name per state.  An element is named by its indices, `x[3,1]`, unless
        `labels` gives names for a block's last axis, `x[3,B]`."""
        labels = labels or {}
        out = []
        for name, shape in self.shapes.items():
            if shape == ():
                out.append(name)
                continue
            last = labels.get(name)
            for idx in np.ndindex(*shape):
                parts = [str(k) for k in idx]
                if last is not None:
                    parts[-1] = last[idx[-1]]
                out.append(f"{name}[{','.join(parts)}]")
        return out

    def index(self, name: str) -> int:
        """The position in the state vector of a scalar block, or of an element named as
        `names()` names it without labels."""
        if name in self.shapes and self.shapes[name] == ():
            return self.slices[name].start
        try:
            return self.names().index(name)
        except ValueError:
            raise KeyError(f"no state {name!r} in the layout") from None


def measured(v):
    """A measure's value: a float for one state, the array itself for a batch.

    A measure written with `x[..., k]` and `u.field[..., k]` serves one state and a batch
    alike; ending it in `measured(...)` rather than `float(...)` keeps it so."""
    return float(v) if np.ndim(v) == 0 else v


def inputs_key(u) -> tuple:
    """The values of an inputs dataclass as a hashable key, for a model that caches an
    evaluation on its inputs: a control structure writes its loops' outputs into one
    inputs object in turn, so the object itself is not a key."""
    return tuple(to_host(np.asarray(0.0) if v is None else v).astype(float).tobytes()
                 for v in (getattr(u, f.name) for f in dataclasses.fields(u)))


def same_values(a, b) -> bool:
    """Whether two arrays, each one state or a batch, on the host or the device, are equal."""
    xp = namespace(a)
    return xp is namespace(b) and a.shape == b.shape and bool(xp.array_equal(a, b))


def state_measurement(state: int | str, layout: StateLayout | None = None) -> Callable:
    """A measurement that reads one state, by its index or by its name in `layout`."""
    if isinstance(state, str) and layout is None:
        raise ValueError("a state named rather than indexed needs its layout")
    i = layout.index(state) if isinstance(state, str) else int(state)

    def measure(x, u, pp) -> float:
        return float(x[i]) if x.ndim == 1 else x[..., i]

    measure.__name__ = f"state_{state}"
    return measure


def cached_design(solve: Callable) -> Callable:
    """Cache a design solve on its arguments, as `jacketed_cstr` does.

    A sweep over tuning, instruments or disturbances then solves the design once per
    worker process instead of once per run.  The arguments must be hashable: a frozen
    parameter dataclass and the option values, not the `Config`, whose sections other
    than options and params do not change the design.
    """
    return lru_cache(maxsize=16)(solve)


def fixed_point_residual(design, names: Sequence[str] | None = None,
                         worst: int = 5) -> tuple[float, list[tuple[str, float]]]:
    """The largest |rhs| of the open-loop plant at the design point, and the states where
    the residual is largest, for finding what keeps a design from being a steady state."""
    r = np.asarray(design.rhs(0.0, design.x, design.u, design.pp), dtype=float)
    order = np.argsort(-np.abs(r))[:worst]
    label = (lambda i: names[i]) if names is not None else (lambda i: f"x[{i}]")
    return float(np.max(np.abs(r))), [(label(i), float(r[i])) for i in order]
