"""Generating datasets from a specification: parallel, resumable, and self-describing.

    generate("spec.toml", workers=8)

writes, under `data/<study name>/` unless told otherwise,

    spec.toml        the specification, copied
    manifest.json    library version, git commit, package versions, start time
    runs.jsonl       one line per finished run: id, status, cost, features, configuration
    index.csv        the same, one row per run, configuration flattened to dotted columns
    runs/<id>.npz    the trajectory of every run that finished (see `Trajectory.to_npz`)

A run's id is the hash of its configuration, so a run is found again by what it is
rather than by where it came in the order.  An interrupted generation resumes where it
stopped: runs already recorded in runs.jsonl are skipped, including those that failed,
since a failure is a property of the configuration and would recur.

A run ends in one of four states.  `ok` has a trajectory.  `invalid` is a configuration
the case refuses, which a Cartesian sweep can produce.  `failed` is an integration that
raised.  `timeout` exceeded the wall-clock budget, which in a design space with unstable
or limit-cycling members is the usual way a run ends early; the cost of the run is
recorded either way, because it varies by more than an order of magnitude across a
design space and says something about the plant it was spent on.

One generation writes to a dataset at a time: a second one started on the same directory
is refused rather than left to interleave its records with the first.  A run recorded
twice is read back once.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from plantbench import cases

# `repo_state` and `case_source` were defined here before `provenance` was public.
from plantbench._provenance import case_source, provenance, repo_state  # noqa: F401
from plantbench.cases import load_case
from plantbench.core import control as ctl
from plantbench.core.case import Config, Trajectory, build_on, design_key, features_of, run_setups
from plantbench.core.case import build as build_setup
from plantbench.core.case import run as run_case
from plantbench.core.spec import Spec, expand

try:
    import fcntl
except ImportError:  # Windows: generations are not guarded against one another there
    fcntl = None

STATUSES = ("ok", "invalid", "failed", "timeout")
# How each run is integrated, which is not part of the specification: the batched
# Jacobian changes a trajectory only within the solver's tolerance, as another machine
# would, and the device not at all beyond that.  See `control.integrate`.
SOLVER_DEFAULTS = {"jacobian": "internal", "device": "cpu"}


def run_one(case_id: str, config: dict, run: dict, runs_dir: str,
            solver: dict | None = None) -> dict:
    """Run one configuration and return its record.  Never raises for a bad run.

    `solver` holds `jacobian` and `device` for `control.integrate`; a record notes them
    when they are not the defaults."""
    solver = {**SOLVER_DEFAULTS, **(solver or {})}
    case = load_case(case_id)
    record, setup = _setup(case, config, run, build_setup)
    if setup is None:
        return record
    start = time.perf_counter()
    try:
        traj = run_case(case, setup.config, t_end=float(run["t_end"]), dt=float(run["dt"]),
                        wall_budget=run["wall_budget"], **solver)
    except ctl.WallTimeExceeded as exc:
        record.update(status="timeout", message=str(exc))
    except Exception as exc:  # noqa: BLE001 -- recorded, not hidden: a failure is a result
        record.update(status="failed", message=f"{type(exc).__name__}: {exc}")
    else:
        if not run.get("states", True):
            traj = replace(traj, x=traj.x[:0], state_names=[])
        traj.to_npz(Path(runs_dir) / f"{record['run_id']}.npz")
        record.update(status="ok", nfev=traj.nfev)
    record["wall_time"] = time.perf_counter() - start
    if solver != SOLVER_DEFAULTS:
        record["solver"] = solver
    return record


def _setup(case, config: dict, run: dict, build) -> tuple[dict, Any]:
    """A new record of `config`, and its setup from `build(case, validated config)` with
    the features recorded; or the record marked invalid or failed, and None."""
    record = {"run_id": run_id(case, config), "status": None, "message": "",
              "wall_time": np.nan, "nfev": -1, "features": {}, "config": config}
    try:
        cfg = case.validate(config)
        record["config"] = cfg.to_dict()
        setup = build(case, cfg)
    except (ValueError, KeyError, TypeError) as exc:
        record.update(status="invalid", message=f"{type(exc).__name__}: {exc}")
        return record, None
    except Exception as exc:  # noqa: BLE001 -- a design that does not solve is a result
        record.update(status="failed", message=f"design: {type(exc).__name__}: {exc}")
        return record, None
    available = features_of(case)
    for name in run["features"]:
        try:
            record["features"].update(available[name](setup))
        except Exception as exc:  # noqa: BLE001 -- a feature that fails is a result
            record["features"][f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return record, setup


def _run_ensembles(case_id: str, configs: list[dict], run: dict, runs_dir: str,
                  solver: dict) -> Iterator[dict]:
    """Run configurations as ensembles of at most `solver["ensemble"]` members and yield
    their records, an ensemble's together as it finishes.  The records are those of
    `run_one`, with `ensemble_size` and `ensemble_wall_time` beside `wall_time`, which is
    the member's share of the ensemble's.  Configurations share a design when they have the
    same `core.case.design_key`.  An ensemble's wall budget is the specification's, which
    is per run, times its number of members."""
    size = int(solver["ensemble"])
    device = solver.get("device", "cpu")
    case = load_case(case_id)
    designs: dict[str, Any] = {}

    def build(case, cfg):
        key = design_key(cfg)
        if key not in designs:
            designs[key] = case.make_design(cfg)
        return build_on(case, designs[key], cfg)

    groups: dict[tuple, list] = {}
    for config in configs:
        record, setup = _setup(case, config, run, build)
        if setup is None:
            yield record
            continue
        key = (design_key(setup.config), ctl.layout_signature(setup.structure))
        groups.setdefault(key, []).append((record, setup))
    for members in groups.values():
        for c in range(0, len(members), size):
            chunk = members[c:c + size]
            # A specification's budget is per run; an ensemble has its members' together.
            budget = None if run["wall_budget"] is None else run["wall_budget"] * len(chunk)
            results = run_setups([s for _, s in chunk], float(run["t_end"]),
                                 dt=float(run["dt"]), wall_budget=budget, device=device)
            for (record, _), res in zip(chunk, results):
                record.update(ensemble_size=res.ensemble_size,
                              ensemble_wall_time=res.ensemble_wall_time,
                              wall_time=res.ensemble_wall_time / res.ensemble_size,
                              solver={"ensemble": size, "device": device})
                if isinstance(res.error, ctl.WallTimeExceeded):
                    record.update(status="timeout", message=str(res.error))
                elif res.error is not None:
                    record.update(status="failed",
                                  message=f"{type(res.error).__name__}: {res.error}")
                else:
                    traj = res.trajectory
                    if not run.get("states", True):
                        traj = replace(traj, x=traj.x[:0], state_names=[])
                    traj.to_npz(Path(runs_dir) / f"{record['run_id']}.npz")
                    record.update(status="ok", nfev=traj.nfev)
                yield record


def _manifest(spec: Spec) -> dict:
    return {
        "study": spec.name, "case": spec.case, **provenance(spec.case),
        "spec_fingerprint": spec.fingerprint(),
        # Only when the specification carries a task; a dataset of runs alone has none.
        **({"protocol_fingerprint": spec.protocol()} if spec.task else {}),
    }


def _configurations_manifest(out: Path, case_id: str, recorded: int) -> dict:
    """The manifest of a dataset built from configurations rather than a specification."""
    manifest = {"study": out.name, "case": case_id, **provenance(case_id),
                "spec_fingerprint": None}
    if recorded:
        # Runs made before the dataset described itself: their provenance is not known.
        manifest["runs_before_manifest"] = recorded
    return manifest


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, dict)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def load_records(out_dir: str | Path) -> list[dict]:
    """Every run recorded so far, once each, in the order they first finished."""
    path = Path(out_dir) / "runs.jsonl"
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    records, seen = [], set()
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            continue  # a run stopped while its record was being written; it will run again
        # A run recorded twice, by two generations that shared the directory before it was
        # guarded, would otherwise count twice and could fall on both sides of a split.
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            records.append(r)
    return records


def load_run(out_dir: str | Path, run_id: str) -> Trajectory:
    return Trajectory.from_npz(Path(out_dir) / "runs" / f"{run_id}.npz")


def write_index(out_dir: str | Path) -> Path:
    """Rewrite index.csv from runs.jsonl, configurations flattened to dotted columns."""
    rows = []
    for r in load_records(out_dir):
        row = {k: r[k] for k in ("run_id", "status", "wall_time", "nfev", "message")}
        row.update({f"feature.{k}": v for k, v in r["features"].items()})
        row.update(_flatten(r["config"], "config."))
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(out_dir) / "index.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return path


def _print(msg: str) -> None:
    print(msg, flush=True)  # line by line, so a log file shows progress as it happens


def generate(spec: Spec | str | Path, out_dir: str | Path | None = None, workers: int = 1,
             log: Callable[[str], None] = _print, solver: dict | None = None) -> Path:
    """Run every configuration the specification describes that is not yet recorded.
    `solver` is passed to `run_one`."""
    spec_path = None
    if not isinstance(spec, Spec):
        spec_path = Path(spec)
        spec = Spec.load(spec_path)
    available = features_of(load_case(spec.case))
    unknown = set(spec.run["features"]) - set(available)
    if unknown:
        raise KeyError(f"case {spec.case!r} has no features {sorted(unknown)}; "
                       f"available: {sorted(available)}")

    out = Path(out_dir) if out_dir is not None else Path("data") / spec.name
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["spec_fingerprint"] != spec.fingerprint():
            raise ValueError(f"{out} holds runs of a different specification; "
                             "use a new directory, or remove it to start again")
    else:
        manifest_path.write_text(json.dumps(_manifest(spec), indent=2) + "\n")
        if spec_path is not None:
            shutil.copyfile(spec_path, out / "spec.toml")
        else:
            (out / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2) + "\n")

    run_configurations(spec.case, expand(spec), out, spec.run, workers=workers, log=log,
                       label=f"{spec.name}: ", solver=solver)
    return out


def run_configurations(case_id: str, configs, out_dir: str | Path, run: dict | None = None,
                       workers: int = 1, log: Callable[[str], None] = _print,
                       label: str = "", solver: dict | None = None) -> list[dict]:
    """Run the given configurations into the dataset at `out_dir`, skipping any already
    recorded there, and return the records of those run now.

    This is the unit that `generate` drives with the configurations a specification
    expands to.  It does not care how the configurations were chosen, so a sequential
    design, one that proposes the next configurations from the results so far as Bayesian
    optimization or active learning does, calls it once per batch and builds up the same
    kind of dataset.  `run` holds the options of a specification's [run] table.
    """
    from plantbench.core.spec import RUN_DEFAULTS

    run = {**RUN_DEFAULTS, **(run or {})}
    if solver and solver.get("ensemble") and workers > 1:
        raise ValueError("an ensemble runs in one process; leave workers at 1")
    out = Path(out_dir)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    case = load_case(case_id)
    keyed: dict[str, dict] = {}
    for c in configs:
        c = dict(c)
        c.setdefault("structure", case.default_structure)
        keyed.setdefault(run_id(case, c), c)  # a sweep may name the same run twice

    with _exclusive(out):
        records = load_records(out)
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("case") != case_id:
                raise ValueError(f"{out} holds runs of case {previous.get('case')!r}, "
                                 f"not {case_id!r}")
        else:
            manifest_path.write_text(json.dumps(
                _configurations_manifest(out, case_id, len(records)), indent=2) + "\n")
        journal_path = out / "runs.jsonl"
        if journal_path.exists() and not journal_path.read_text().endswith("\n"):
            # Drop a record cut off mid-write, so the next one starts on a line of its own.
            journal_path.write_text("".join(json.dumps(r, default=_json_default) + "\n"
                                            for r in records))
        done = {r["run_id"] for r in records}
        todo = [c for k, c in keyed.items() if k not in done]
        log(f"{label}{len(keyed)} runs, {len(keyed) - len(todo)} already recorded, "
            f"{len(todo)} to run on {workers} worker(s)")

        new: list[dict] = []
        with open(journal_path, "a") as journal, _terminate_as_interrupt():
            def record(r: dict, i: int) -> None:
                journal.write(json.dumps(r, default=_json_default) + "\n")
                journal.flush()
                new.append(r)
                log(f"[{i}/{len(todo)}] {r['run_id']} {r['status']:7s} {r['wall_time']:7.2f} s"
                    + (f"  {r['message'][:80]}" if r["message"] else ""))

            args = [(case_id, c, run, str(runs_dir), solver) for c in todo]
            if solver and solver.get("ensemble"):
                for i, r in enumerate(_run_ensembles(case_id, todo, run, str(runs_dir),
                                                    solver), 1):
                    record(r, i)
            elif workers <= 1:
                for i, a in enumerate(args, 1):
                    record(run_one(*a), i)
            else:
                _check_workers_can_load(case_id)
                with ProcessPoolExecutor(max_workers=workers, initializer=_import_modules,
                                         initargs=(cases.session_modules(),)) as pool:
                    futures = [pool.submit(run_one, *a) for a in args]
                    try:
                        for i, fut in enumerate(as_completed(futures), 1):
                            record(fut.result(), i)
                    except KeyboardInterrupt:
                        # Cancel the runs not yet started instead of waiting for all of
                        # them; those already running finish, and a resume runs whatever
                        # was not recorded.  The wait belongs to this call: leaving it to
                        # the pool's own exit resets the cancellation before it is read.
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
        write_index(out)
    return new


def _import_modules(modules: list[str]) -> None:
    """Worker initializer: import the modules that defined the session's cases, so that
    a case made available by `@plantbench.case` or `register` exists in the worker too.

    The script run as `__main__` needs nothing: a spawned worker imports it again as
    `__mp_main__`, which runs its decorators, provided the dataset is generated under
    `if __name__ == "__main__":`.
    """
    for m in modules:
        if m not in ("__main__", "__mp_main__"):
            importlib.import_module(m)


def _check_workers_can_load(case_id: str) -> None:
    """Refuse, before starting workers, a session case that no worker could define again:
    one defined in an interactive session, which has no file to import."""
    source = cases.source(case_id)
    if source["kind"] == "session" and source["module"] == "__main__" \
            and not hasattr(sys.modules["__main__"], "__file__"):
        raise RuntimeError(f"case {case_id!r} is defined in an interactive session, which "
                           "worker processes cannot import; define it in a module, or "
                           "generate with one worker")


@contextmanager
def _exclusive(out: Path):
    """Hold the dataset in `out` for one generation, refusing a second one."""
    with open(out / ".lock", "w") as lock:
        if fcntl is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"{out} is being written by another generation; wait for "
                                   "it to finish, or write to a directory of its own") from None
        yield


@contextmanager
def _terminate_as_interrupt():
    """Treat SIGTERM, which a scheduler sends to end a job, as an interrupt.

    A worker process forked while the handler is installed inherits it, and there it
    restores the default so that the signal ends the worker as it always did.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    parent = os.getpid()

    def interrupt(signum, frame):
        if os.getpid() != parent:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_id(case, config: dict) -> str:
    """The run id: the hash of the configuration as the case completes it, or as given
    when the case refuses it.

    A run is found again by what it is rather than by where it came in the order, so a
    study that proposes configurations and reads their results back from a dataset asks
    for them by this.
    """
    try:
        return case.validate(config).key()
    except (ValueError, KeyError, TypeError):
        pass
    try:
        return Config.from_dict(config).key()
    except (KeyError, TypeError):
        # Not a configuration at all, such as one with a misspelt section.  It is still
        # recorded, as invalid, under the hash of what was given, so that one malformed
        # entry does not stop a batch.
        text = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
        return hashlib.sha1(text.encode()).hexdigest()[:16]


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"cannot record {type(v).__name__}")
