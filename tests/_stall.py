"""A one-state plant whose right-hand side can be made to stall in native code, for the
time limits of the stepper.

The input `stall` is seconds of `usleep` in libc per evaluation, a call no Python signal
interrupts; zero, the design value, costs nothing.  The design solve sleeps or fails when
the environment says so, which reaches a spawned worker as well, so that a worker's start
can be made to stall or fail after the parent has built the same plant.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from dataclasses import dataclass

import numpy as np

from plantbench.core import control as ctl
from plantbench.core.case import Case, Config
from plantbench.core.control import Loop

LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
FAIL, SLOW = "PLANTBENCH_TEST_STALL_FAIL", "PLANTBENCH_TEST_STALL_SLOW"


@dataclass
class Inputs:
    q: float = 1.0
    stall: float = 0.0


@dataclass
class Params:
    tau: float = 2.0


@dataclass
class Design:
    x: np.ndarray
    u: Inputs
    pp: Params

    def rhs(self, t, x, u, pp):
        if u.stall > 0.0:
            LIBC.usleep(int(u.stall * 1e6))
        return np.array([(u.q - x[0]) / pp.tau])


def make_design(config: Config) -> Design:
    if os.environ.get(FAIL):
        raise ValueError("the design was told to fail")
    if os.environ.get(SLOW):
        time.sleep(float(os.environ[SLOW]))
    return Design(x=np.array([1.0]), u=Inputs(), pp=Params())


def make_structure(design, config: Config):
    level = lambda x, u, pp: float(x[0])  # noqa: E731
    return ctl.bias_from_design([Loop("level", level, "q", Kc=0.5, tau_I=4.0)], design)


CASE = Case(
    id="stall",
    title="A plant that stalls on request",
    summary="One state; its right-hand side sleeps in libc for as long as `stall` says.",
    options={},
    structures=("PI",),
    default_structure="PI",
    make_design=make_design,
    make_structure=make_structure,
    disturbances={},
    measurements=lambda design: {"x": lambda x, u, pp: float(x[0])},
    state_names=lambda design: ["x"],
)
