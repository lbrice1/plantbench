"""Integration of an ensemble of closed loops in lockstep, by the Rosenbrock method Rodas5P.

The members share a design and the layout of their control structures and differ in
whatever `control.stack` lets differ.  Each step evaluates the right-hand sides of every
member still running as one batch, and the Jacobians of those that need one as another,
so that on a GPU, where the cost of an evaluation hardly depends on its size, an ensemble
of many members costs little more per step than one.  Each member keeps its own time, step
size and error control; the batch is formed from whatever members are still running.

Rodas5P (Steinebach, 2023) is a Rosenbrock method of order 5 with an embedded estimate of
order 4, stiffly accurate and L-stable.  A Rosenbrock method needs the exact Jacobian at
the start of every step and solves linear systems instead of Newton iterations, which is
what makes the lockstep possible: every member does the same, fixed work per step.  It was
chosen over BDF and Rosenbrock-W methods on the step, Jacobian and evaluation counts of
the serial NGL cases.  The coefficients are those of OrdinaryDiffEq.jl (`RODAS5PA`,
`RODAS5PC`, `RODAS5Pc`, `RODAS5Pd`, `RODAS5PH`), in the transformed form of Hairer and
Wanner (1996, IV.7.26):

    (I / (h gamma) - J) k_i = f(t + c_i h, y + sum_j A_ij k_j) + h d_i df/dt
                              + sum_j (C_ij / h) k_j,
    y_1 = y + sum_j A_8j k_j + k_8,   error = k_8.

Between the ends of a step the solution is the method's dense output (`RODAS5PH`), with
kd_m = sum_i H_mi k_i and theta = (t - t_n) / h:

    y(theta) = (1 - theta) y + theta (y_1 + (1 - theta) (kd_1 + theta (kd_2 + theta kd_3))),

whose local error is of the order of the step's.

The integration agrees with LSODA's to the tolerance, not to the last bit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from plantbench.backend import check as check_device
from plantbench.backend import module, to_host

from . import control as ctl

_A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [3.0, 0, 0, 0, 0, 0, 0, 0],
    [2.849394379747939, 0.45842242204463923, 0, 0, 0, 0, 0, 0],
    [-6.954028509809101, 2.489845061869568, -10.358996098473584, 0, 0, 0, 0, 0],
    [2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641,
     0, 0, 0, 0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
     0.030198172008377946, 0, 0, 0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
     0.030198172008377946, 1, 0, 0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256,
     0.030198172008377946, 1, 1, 0]])
_C = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [-14.155112264123755, 0, 0, 0, 0, 0, 0],
    [-17.97296035885952, -2.859693295451294, 0, 0, 0, 0, 0],
    [147.12150275711716, -1.41221402718213, 71.68940251302358, 0, 0, 0, 0],
    [165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306,
     0, 0, 0],
    [24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125,
     -0.6610691825249471, 0, 0],
    [30.91273214028599, -3.1208243349937974, 77.79954646070892, 34.28646028294783,
     -19.097331116725623, -28.087943162872662, 0],
    [37.80277123390563, -3.2571969029072276, 112.26918849496327, 66.9347231244047,
     -40.06618937091002, -54.66780262877968, -9.48861652309627]])
_H = np.array([
    [25.948786856663858, -2.5579724845846235, 10.433815404888879, -2.3679251022685204,
     0.524948541321073, 1.1241088310450404, 0.4272876194431874, -0.17202221070155493],
    [-9.91568850695171, -0.9689944594115154, 3.0438037242978453, -24.495224566215796,
     20.176138334709044, 15.98066361424651, -6.789040303419874, -6.710236069923372],
    [11.419903575922262, 2.8879645146136994, 72.92137995996029, 80.12511834622643,
     -52.072871366152654, -59.78993625266729, -0.15582684282751913, 4.883087185713722]])
_c = np.array([0.0, 0.6358126895828704, 0.4095798393397535, 0.9769306725060716,
               0.4288403609558664, 1.0, 1.0, 1.0])
_d = np.array([0.21193756319429014, -0.42387512638858027, -0.3384627126235924,
               1.8046452872882734, 2.325825639765069, 0.0, 0.0, 0.0])
GAMMA = 0.21193756319429014
STAGES = 8
ORDER = 4  # of the embedded estimate, which sets the step-size controller's exponent

# Step-size control: the factor 0.9 e^(-1/(ORDER + 1)), held within [0.2, 5].
_SAFETY, _SHRINK, _GROW = 0.9, 0.2, 5.0
# The first step, and the largest after a breakpoint, in minutes.  It is the step a serial
# run of the NGL cases settles to within a few steps.
H0 = 1e-3
STATUSES = ("ok", "failed", "timeout")


@dataclass
class EnsembleRun:
    """The integration of an ensemble.

    `Y` is (members, n_points, n + n_aug), the plant and augmented states on the common
    grid `t`, NaN from where a member stopped.  `status` is "ok", "failed" or "timeout"
    per member, with a `message` for the last two.  The counts are per member: accepted
    and rejected steps, Jacobians, and rows evaluated (`nfev`, n + 1 per Jacobian).
    `at_breaks` holds, per member, its state at each of its breakpoints inside the run,
    where its steps end."""

    t: np.ndarray
    Y: np.ndarray
    status: list[str]
    message: list[str]
    steps: np.ndarray
    rejected: np.ndarray
    jacobians: np.ndarray
    nfev: np.ndarray
    wall_time: float
    at_breaks: list = field(default_factory=list)


def _dense(y0, y1, kd, theta):
    """The dense output of one step from y0 to y1 with kd = H k, at the fractions `theta`
    of the step: (len(theta), n)."""
    th = np.asarray(theta)[:, None]
    return (1 - th) * y0 + th * (y1 + (1 - th) * (kd[0] + th * (kd[1] + th * kd[2])))


def _factor(W, xp):
    """A solver for the stack of systems W (k, N, N), factorized once: by LU on the host,
    by the batched inverse on a GPU (`cupy.linalg.inv`), one call for the stack."""
    if xp is np:
        lu = lu_factor(W, check_finite=False)
        return lambda b: lu_solve(lu, b[..., None], check_finite=False)[..., 0]
    inv = xp.linalg.inv(W)
    return lambda b: xp.einsum("kij,kj->ki", inv, b)


def integrate_ensemble(design, structures: list, disturbances: list, t_end: float,
                       n_points: int = 1200, rtol: float = 1e-8, atol: float = 1e-10,
                       wall_budget: float | None = None, device: str = "cpu",
                       max_steps: int = 100_000, starts: list | None = None,
                       t0: float = 0.0, y0: list | None = None,
                       dense: bool = True) -> EnsembleRun:
    """Integrate the closed loops of `structures`, member i under `disturbances[i]`, from
    the design point, or from the plant state `starts[i]` where it is given, to `t_end`,
    all on the grid of `n_points` from 0 to `t_end`.  Given `y0`, member i starts instead
    from the full state `y0[i]`, plant and augmented, at time `t0`, and the grid runs from
    `t0`: what the ensemble stepper needs to continue from where its members are.

    The design must declare a batched right-hand side that does not depend on t.  Every
    step of a member ends on its breakpoints (`control.breakpoints_of`) and at `t_end`.
    The grid values between are the method's dense output, from the step they fall in,
    which never crosses a breakpoint.  With `dense=False` the steps end on the grid too, so
    that the grid values are the steps' own, at the cost of the steps the grid shortens.
    A member stops as "failed" when its step size falls below round-off of its time, after
    `max_steps` steps, or when its state is non-finite at the smallest step; the others
    go on.  When `wall_budget` seconds have passed, every member still running stops as
    "timeout"."""
    if not ctl.supports_batch(design):
        raise ValueError(f"{type(design).__name__} does not declare a batched right-hand "
                         "side (`batched = True`), which an ensemble needs")
    check_device(device)
    xp = module(device)
    start = time.perf_counter()
    M = len(structures)
    if len(disturbances) != M:
        raise ValueError("one disturbance (or None) is needed per structure")
    stacked = ctl.stack(structures, xp)
    model, pp, u0 = design.rhs, design.pp, design.u
    if starts is not None and len(starts) != M:
        raise ValueError("one start (or None) is needed per structure")
    starts = [design.x if x is None else np.asarray(x, dtype=float)
              for x in (starts or [None] * M)]
    if y0 is None:
        Y0 = np.array([np.concatenate([x, ctl.initial_augmented(S, design, x)])
                       for S, x in zip(structures, starts)])
    else:
        if len(y0) != M:
            raise ValueError("one full start is needed per structure")
        Y0 = np.array([np.asarray(y, dtype=float) for y in y0])
    N = Y0.shape[1]
    grid = np.linspace(t0, t_end, n_points)
    # Where each member's steps must end: its breakpoints and t_end, and the grid after its
    # start without dense output.
    ends = grid[1:] if not dense else [t_end]
    stops = [np.union1d(ends, [b for b in ctl.breakpoints_of(d, S) if t0 < b < t_end])
             for S, d in zip(structures, disturbances)]
    breaks = [set(ctl.breakpoints_of(d, S)) for S, d in zip(structures, disturbances)]

    y = xp.asarray(Y0)
    J = xp.zeros((M, N, N))
    out = np.full((M, n_points, N), np.nan)
    out[:, 0] = Y0
    t = np.full(M, float(t0))
    h = np.full(M, H0)
    at_breaks: list[dict[float, np.ndarray]] = [{} for _ in range(M)]
    nxt = np.zeros(M, dtype=int)
    filled = np.ones(M, dtype=int)  # the first grid point of each member not yet given
    stale = np.ones(M, dtype=bool)
    status = ["running"] * M
    message = [""] * M
    steps, rejected, jacobians, nfev = (np.zeros(M, dtype=int) for _ in range(4))
    running = np.ones(M, dtype=bool)
    # Indices into the device arrays are moved there; the bookkeeping stays on the host.
    ix = xp.asarray
    H = xp.asarray(_H)

    def rhs(t_m, Y_m, members):
        """The right-hand sides of `members` at their times t_m and states Y_m."""
        local = np.arange(len(members))
        nfev[members] += 1
        return ctl.ensemble_rhs(xp.asarray(t_m), Y_m, ctl.take(stacked, xp.asarray(members)),
                                ctl.ensemble_inputs(t_m, [disturbances[i] for i in members],
                                                    u0, local, xp), pp, model)

    while running.any():
        if wall_budget is not None and time.perf_counter() - start > wall_budget:
            for i in np.flatnonzero(running):
                status[i], message[i] = "timeout", (f"stopped at t = {t[i]:.1f} after "
                                                    f"{wall_budget:.0f} s")
            break
        a = np.flatnonzero(running)

        # Jacobians, for the members that moved since their last.
        s = a[stale[a]]
        if s.size:
            rows = np.repeat(s, N + 1)
            local = np.repeat(np.arange(s.size), N + 1)
            J[ix(s)] = ctl.ensemble_jacobian(
                xp.asarray(t[rows]), y[ix(s)], ctl.take(stacked, xp.asarray(rows)),
                ctl.ensemble_inputs(t[s], [disturbances[i] for i in s], u0, local, xp),
                pp, model)
            nfev[s] += N + 1
            jacobians[s] += 1
            stale[s] = False

        # The step of each member, ending on its next stop when it reaches it.
        target = np.array([stops[i][nxt[i]] for i in a])
        room = target - t[a]
        lands = h[a] >= room * (1.0 - 1e-12)
        hs = np.where(lands, room, h[a])
        t_a, y_a = t[a], y[ix(a)]

        solve = _factor(xp.eye(N) / xp.asarray(hs * GAMMA)[:, None, None] - J[ix(a)], xp)
        # df/dt, through the inputs only, for the members whose inputs move within the
        # step (a ramp); under steps and set-point changes, which end steps, it is zero.
        dT = xp.zeros_like(y_a)
        dt_probe = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(t_a), 1.0)
        moving = np.array([_inputs_move(disturbances[i], t_a[k], dt_probe[k], u0)
                           for k, i in enumerate(a)], dtype=bool)
        # The stages read the inputs at t + c_i h, and those with c_i = 1 at the left limit
        # of the step's end: a step ending on a breakpoint must not see what starts there.
        end = np.nextafter(t_a + hs, -np.inf)
        k_stages = []
        hs_x = xp.asarray(hs)[:, None]
        for i in range(STAGES):
            yi = y_a
            for j in range(i):
                if _A[i, j]:
                    yi = yi + _A[i, j] * k_stages[j]
            fi = rhs(np.minimum(t_a + _c[i] * hs, end), yi, a)
            if i == 0 and moving.any():
                m = np.flatnonzero(moving)
                f_ahead = rhs(t_a[m] + dt_probe[m], y_a[ix(m)], a[m])
                dT[ix(m)] = (f_ahead - fi[ix(m)]) / xp.asarray(dt_probe[m])[:, None]
            b = fi
            if _d[i]:
                b = b + (_d[i] * hs_x) * dT
            for j in range(i):
                b = b + (_C[i, j] / hs_x) * k_stages[j]
            k_stages.append(solve(b))
        y1 = y_a
        for j in range(STAGES - 1):
            y1 = y1 + _A[STAGES - 1, j] * k_stages[j]
        y1 = y1 + k_stages[-1]

        scale = atol + rtol * xp.maximum(xp.abs(y_a), xp.abs(y1))
        err = to_host(xp.sqrt(xp.mean((k_stages[-1] / scale) ** 2, axis=1)))
        finite = to_host(xp.isfinite(y1).all(axis=1))
        err = np.where(np.isfinite(err) & finite, err, np.inf)
        ok = err <= 1.0
        fac = _SAFETY * np.maximum(err, 1e-10) ** (-1.0 / (ORDER + 1))

        acc, rej = a[ok], a[~ok]
        if acc.size:
            y[ix(acc)] = y1[ix(ok)]
            landed = lands[ok]
            t[acc] = np.where(landed, target[ok], t_a[ok] + hs[ok])
            grown = hs[ok] * np.clip(fac[ok], _SHRINK, _GROW)
            # A step cut short to land on a stop says nothing against the longer one.
            h[acc] = np.where(landed, np.maximum(h[acc], grown), grown)
            steps[acc] += 1
            stale[acc] = True
            done_y = to_host(y1[ix(ok)])
            # The grid points each accepted step reaches: one it ends on takes the step's
            # own state, those before its end the dense output.
            last = np.searchsorted(grid, t[acc], side="right")
            within = [np.arange(filled[i], last[k]) for k, i in enumerate(acc)]
            within = [g[grid[g] < t[i]] for g, i in zip(within, acc)]
            need = np.flatnonzero([g.size > 0 for g in within])
            if need.size:
                rows = ix(np.flatnonzero(ok)[need])
                kd = to_host(xp.einsum("mi,ikn->kmn", H,
                                       xp.stack([k_i[rows] for k_i in k_stages])))
                y_in = to_host(y_a[rows])
                for q, k in enumerate(need):
                    theta = (grid[within[k]] - t_a[ok][k]) / hs[ok][k]
                    out[acc[k], within[k]] = _dense(y_in[q], done_y[k], kd[q], theta)
            for k, i in enumerate(acc):
                if last[k] > filled[i] and grid[last[k] - 1] == t[i]:
                    out[i, last[k] - 1] = done_y[k]
                filled[i] = max(filled[i], last[k])
                if not landed[k]:
                    continue
                if float(t[i]) in breaks[i]:
                    h[i] = min(h[i], H0)
                    at_breaks[i][float(t[i])] = done_y[k]
                nxt[i] += 1
                if nxt[i] == len(stops[i]):
                    status[i], running[i] = "ok", False
        if rej.size:
            h[rej] = hs[~ok] * np.clip(fac[~ok], _SHRINK, 1.0)
            rejected[rej] += 1
        for i in a:
            if not running[i]:
                continue
            if h[i] < 1e-12 * max(1.0, abs(t[i])):
                status[i], message[i] = "failed", (f"step size {h[i]:.1e} at t = {t[i]:.4g}, "
                                                   "below round-off")
                running[i] = False
            elif steps[i] + rejected[i] >= max_steps:
                status[i], message[i] = "failed", f"{max_steps} steps at t = {t[i]:.4g}"
                running[i] = False

    return EnsembleRun(grid, out, status, message, steps, rejected, jacobians, nfev,
                       time.perf_counter() - start, at_breaks)


def _inputs_move(disturbance, t: float, dt: float, u0) -> bool:
    """Whether the disturbance changes the inputs between t and t + dt."""
    if disturbance is None:
        return False
    a, b = disturbance(t, u0), disturbance(t + dt, u0)
    return any(not np.array_equal(getattr(a, f), getattr(b, f))
               for f in vars(a) if getattr(a, f) is not None)
