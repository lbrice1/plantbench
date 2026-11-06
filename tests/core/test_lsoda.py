"""LSODA, which `control.integrate` calls once per trajectory and once per segment between
breakpoints, frees its work arrays when a solve returns.  SciPy 1.17 and 1.18 keep them
(scipy/scipy#26373); a process that integrates many runs then grows without bound.  The
requirement in pyproject.toml excludes those releases; this test fails if a release with
the leak comes back."""

from __future__ import annotations

import json
import subprocess
import sys

# A separate interpreter, so that nothing else the test session allocates is counted.
# The memory still held after a number of solves is measured twice, and the difference
# between the two counts is what each solve leaves behind.
SCRIPT = """
import gc, json, tracemalloc
import numpy as np
from scipy.integrate import solve_ivp

n = 50
A = -np.eye(n)
def solve(k):
    for _ in range(k):
        solve_ivp(lambda t, y: A @ y, (0.0, 1.0), np.ones(n), method="LSODA", t_eval=[1.0])
    gc.collect()

solve(10)
tracemalloc.start()
start = tracemalloc.get_traced_memory()[0]
solve(50)
first = tracemalloc.get_traced_memory()[0] - start
solve(200)
second = tracemalloc.get_traced_memory()[0] - start
print(json.dumps({"first": first, "second": second}))
"""


def test_lsoda_keeps_nothing_after_a_solve():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    held = json.loads(out.stdout)
    # The work arrays at 50 states are about 24 kB, so the leak holds about 5 MB more after
    # the further 200 solves; without it the two counts differ by allocator noise.
    per_solve = (held["second"] - held["first"]) / 200
    assert per_solve < 1_000, f"{per_solve:.0f} bytes held per LSODA solve"
