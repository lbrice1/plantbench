"""The array module a batched evaluation runs on: NumPy on the host, CuPy on a GPU.

The batched path of the closed loop (`control.closed_loop_rhs_batch` and what it calls)
is written against a NumPy-compatible module rather than NumPy itself, so that the same
code runs on either.  A function finds its module from its arguments with `namespace`,
the way CuPy's own `get_array_module` does, and never imports CuPy directly; CuPy is an
optional dependency, `pip install 'plantbench[gpu]'`, and only `device="cuda"` needs it.

Everything is float64.  The integrations run at a relative tolerance of 1e-8, which
single precision cannot meet, so a device without fast double precision gains little.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

DEVICES = ("cpu", "cuda")


def _cupy() -> ModuleType | None:
    try:
        import cupy
    except ImportError:
        return None
    return cupy


def available(device: str) -> bool:
    """Whether `device` can be used in this environment."""
    check(device)
    if device == "cpu":
        return True
    cp = _cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def check(device: str) -> str:
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    return device


def module(device: str) -> ModuleType:
    """The array module of `device`."""
    if check(device) == "cpu":
        return np
    cp = _cupy()
    if cp is None:
        raise RuntimeError("device 'cuda' needs CuPy: pip install 'plantbench[gpu]'")
    return cp


def namespace(*arrays) -> ModuleType:
    """The array module of the arrays given: CuPy if any of them is a CuPy array."""
    cp = _cupy()
    if cp is not None and any(isinstance(a, cp.ndarray) for a in arrays):
        return cp
    return np


def to_device(x, device: str):
    """`x` as a float64 array on `device`."""
    return module(device).asarray(x, dtype=float)


def to_host(x) -> np.ndarray:
    """`x` as a NumPy array, copied from the device if it is there."""
    cp = _cupy()
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def clamp(v, lo, hi):
    """`v` limited to [lo, hi]: by `min` and `max` for a scalar, as the serial path has
    always done, and elementwise for an array."""
    if np.ndim(v) == 0:
        return min(max(v, lo), hi)
    return namespace(v).clip(v, lo, hi)
