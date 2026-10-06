"""Reading the datasets on disk: what is there, what a result was computed from, and how
two of them differ.

    plantbench datasets
    plantbench compare data/regimes-old data/regimes

A dataset describes itself as it is generated (`datagen.py`).  `summary` reads that
description back: which specification the runs are of, which commit generated them, how
they ended, and whether the specification left in the directory still describes them.  A
specification is named by the digest of its fingerprint, twelve hex characters that
identify a protocol without quoting it, and printed as `pb-spec:<digest>` so that it is not
read as a commit.

`stamp` is the same record attached to a result rather than to a dataset, so a results
file or an analysis archive says which dataset and which commit produced it; `check_stamp`
refuses a stamp that does not describe the dataset it is held against.

`compare` answers what the runs themselves cannot: two datasets of one specification, or
two results files computed from them, and what moved between them.  Wall time is reported
apart from the comparison, being a property of the machine and the moment rather than of
the data; the number of right-hand-side evaluations is not, and two runs that took a
different path through the integrator differ even when their trajectories agree.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np

from plantbench import datagen
from plantbench.core.spec import PROTOCOL_ID, SPEC_ID, Spec, digest, identifier

TOL = 1e-9  # relative, with an absolute floor of the same size
ANALYSIS_LIBRARIES = ("scikit-learn", "pacmap", "hdbscan")


def _is_dataset(p: Path) -> bool:
    return (p / "manifest.json").exists() or (p / "runs.jsonl").exists()


def find(root: str | Path = "data") -> list[Path]:
    """Every dataset under `root`, or `root` itself when it is one.

    A directory of runs without a manifest is a dataset whose provenance was not
    recorded; it is listed as such rather than left out.
    """
    root = Path(root)
    if _is_dataset(root):
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if _is_dataset(p))


def summary(out_dir: str | Path) -> dict:
    """What a dataset holds, from its manifest and its journal.

    A dataset without a manifest has its runs counted and its provenance reported as
    unknown: `manifest` is None and so are the fields read from it.
    """
    out = Path(out_dir)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else None
    records = datagen.load_records(out)
    counts = dict.fromkeys(datagen.STATUSES, 0)
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    runs_dir = out / "runs"
    m = manifest or {}
    fingerprint = m.get("spec_fingerprint")
    protocol = m.get("protocol_fingerprint")
    return {
        "dir": out,
        "study": m.get("study"),
        "case": m.get("case"),
        # Where a case that is not built in came from; None for a built-in case.
        "case_source": m.get("case_source"),
        "spec_digest": digest(fingerprint) if fingerprint is not None else None,
        "protocol_digest": digest(protocol) if protocol is not None else None,
        "spec_matches": _spec_matches(out, fingerprint),
        "git_commit": m.get("git_commit"),
        "git_dirty": m.get("git_dirty"),
        "started": m.get("started"),
        "counts": counts,
        "runs": len(records),
        "trajectories": len(list(runs_dir.glob("*.npz"))) if runs_dir.is_dir() else 0,
        "wall_time": float(np.nansum([r["wall_time"] for r in records])) if records else 0.0,
        "manifest": manifest,
    }


def _spec_matches(out: Path, fingerprint: str | None) -> bool | None:
    """Whether the specification left in the directory still describes the runs in it."""
    if fingerprint is None:
        return None
    toml_path, json_path = out / "spec.toml", out / "spec.json"
    try:
        if toml_path.exists():
            return Spec.load(toml_path).fingerprint() == fingerprint
        if json_path.exists():
            return Spec.from_dict(json.loads(json_path.read_text())).fingerprint() == fingerprint
    except (OSError, ValueError, KeyError, TypeError):
        return False  # a specification that no longer loads no longer describes anything
    return None


def stamp(data_dir: str | Path | None = None) -> dict:
    """What a result was computed from: its dataset, and the code that read it.

    Everything recorded is a property of the data and the tree, so a result computed
    twice carries the same stamp.  The time of the run is not part of it; the dataset's
    manifest holds when the runs were made, and the commit holds when the code was written.
    """
    import scipy

    from plantbench import __version__
    out: dict = {}
    if data_dir is not None:
        d = summary(data_dir)
        out.update(dataset=str(data_dir), spec_digest=d["spec_digest"],
                   dataset_commit=d["git_commit"], dataset_dirty=d["git_dirty"],
                   runs=d["counts"]["ok"])
        if d["manifest"] is None:
            out["dataset_manifest"] = False
        if d["protocol_digest"] is not None:
            out["protocol_digest"] = d["protocol_digest"]
        if d["case_source"] is not None:
            out["case_source"] = d["case_source"]
    state = datagen.repo_state()
    out.update(commit=state["git_commit"], dirty=state["git_dirty"],
               plantbench=__version__, python=sys.version.split()[0],
               numpy=np.__version__, scipy=scipy.__version__)
    # The libraries an analysis of the runs may read them with: embeddings, clusterings
    # and classifiers depend on their versions as trajectories depend on SciPy's.
    for name in ANALYSIS_LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def stamp_lines(s: dict) -> list[str]:
    """The stamp as the head of a results file."""
    lines = []
    if "dataset" in s:
        protocol = (f"{identifier(PROTOCOL_ID, s['protocol_digest'])}, "
                    if "protocol_digest" in s else "")
        spec = (identifier(SPEC_ID, s["spec_digest"]) if s["spec_digest"] is not None
                else "no specification")
        made = ("its provenance not recorded, having no manifest"
                if s.get("dataset_manifest") is False
                else f"generated at {commit_text(s['dataset_commit'], s['dataset_dirty'])}")
        lines.append(f"dataset {s['dataset']}, {spec}, {protocol}{s['runs']} runs, {made}")
        if "case_source" in s:
            lines.append(case_source_text(s["case_source"]))
    libraries = "".join(f", {name} {s[name]}" for name in ANALYSIS_LIBRARIES if name in s)
    lines.append(f"analyzed at {commit_text(s['commit'], s['dirty'])} with plantbench "
                 f"{s['plantbench']}, Python {s['python']}, NumPy {s['numpy']}, "
                 f"SciPy {s['scipy']}{libraries}")
    return lines


def check_stamp(s: dict, data_dir: str | Path) -> str | None:
    """Why the stamp does not describe the dataset in `data_dir`, or None when it does."""
    have = summary(data_dir)
    if s.get("spec_digest") != have["spec_digest"]:
        return (f"computed from {_named(SPEC_ID, s.get('spec_digest'))}, "
                f"and {data_dir} holds {_named(SPEC_ID, have['spec_digest'])}")
    if s.get("protocol_digest") != have["protocol_digest"] and "protocol_digest" in s:
        return (f"computed against {_named(PROTOCOL_ID, s['protocol_digest'])}, and "
                f"{data_dir} holds {_named(PROTOCOL_ID, have['protocol_digest'])}")
    if s.get("runs") is not None and s["runs"] != have["counts"]["ok"]:
        return f"computed from {s['runs']} runs, and {data_dir} holds {have['counts']['ok']}"
    return None


def _named(kind: str, d: str | None) -> str:
    """`specification pb-spec:<digest>`, as a message names what it refers to."""
    if d is None:
        return "no specification"
    return f"{'specification' if kind == SPEC_ID else 'protocol'} {identifier(kind, d)}"


def case_source_text(source: dict) -> str:
    """Where a case that is not built in came from, as a line of a results file."""
    package = (f" of {source['distribution']} {source['version']}"
               if "distribution" in source else "")
    return (f"case from {source['module']}{package} ({source['kind']}), at "
            f"{commit_text(source.get('git_commit'), source.get('git_dirty'))}")


def commit_text(commit: str | None, dirty: bool | None) -> str:
    """A commit and what its tree was, as `1a2b3c4 clean`."""
    if not commit:
        return "no commit"
    state = {True: " dirty", False: " clean", None: ""}[dirty]
    return f"{commit[:7]}{state}"


def _noop(msg: str) -> None:
    pass


def compare(a: str | Path, b: str | Path, tol: float = TOL, trajectories: bool = True,
            limit: int = 20, log: Callable[[str], None] = _noop) -> list[str]:
    """Two datasets or two results files, and what moved between them."""
    a, b = Path(a), Path(b)
    if a.is_dir() and b.is_dir():
        return compare_datasets(a, b, tol=tol, trajectories=trajectories, limit=limit, log=log)
    if a.is_file() and b.is_file():
        return compare_results(a, b, tol=tol, limit=limit)
    raise ValueError("compare takes two dataset directories or two results files, "
                     f"not {a} and {b}")


def compare_datasets(a: str | Path, b: str | Path, tol: float = TOL,
                     trajectories: bool = True, limit: int = 20,
                     log: Callable[[str], None] = _noop) -> list[str]:
    """Run for run, what two datasets disagree on."""
    A, B = summary(a), summary(b)
    out = [f"A {A['dir']}: {A['counts']['ok']} completed runs of "
           f"{_named(SPEC_ID, A['spec_digest'])}, "
           f"generated at {commit_text(A['git_commit'], A['git_dirty'])}",
           f"B {B['dir']}: {B['counts']['ok']} completed runs of "
           f"{_named(SPEC_ID, B['spec_digest'])}, "
           f"generated at {commit_text(B['git_commit'], B['git_dirty'])}",
           ""]
    if A["spec_digest"] != B["spec_digest"]:
        out.append("specification: different, so the two hold runs of different protocols")
    else:
        out.append("specification: the same")
    ma, mb = A["manifest"] or {}, B["manifest"] or {}
    moved = [(k, ma.get(k), mb.get(k))
             for k in ("study", "case", "plantbench", "git_commit", "git_dirty",
                       "python", "numpy", "scipy", "platform")
             if ma.get(k) != mb.get(k)]
    out.append("manifest: " + ("; ".join(f"{k} {x} -> {y}" for k, x, y in moved)
                               if moved else "the same but for the start time"))

    ra = {r["run_id"]: r for r in datagen.load_records(A["dir"])}
    rb = {r["run_id"]: r for r in datagen.load_records(B["dir"])}
    only_a, only_b = sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))
    shared = [k for k in ra if k in rb]
    status = [k for k in shared if ra[k]["status"] != rb[k]["status"]]
    out.append(f"runs: {len(shared)} in both, {len(only_a)} in A only, {len(only_b)} in B "
               f"only, {len(status)} whose status differs")
    out += _listing("only in A", [f"{k} {ra[k]['status']}" for k in only_a], limit)
    out += _listing("only in B", [f"{k} {rb[k]['status']}" for k in only_b], limit)
    out += _listing("status", [f"{k} {ra[k]['status']} -> {rb[k]['status']}" for k in status],
                    limit)

    names = sorted({n for k in shared for n in ra[k]["features"]})
    features = [f"{k} {n} {ra[k]['features'].get(n)} -> {rb[k]['features'].get(n)}"
                for k in shared for n in names
                if _moved(ra[k]["features"].get(n), rb[k]["features"].get(n), tol)]
    out.append(f"features: {len(shared)} runs compared over {len(names)} recorded "
               f"features, {len(features)} moved by more than {tol:g}")
    out += _listing("features", features, limit)
    nfev = [f"{k} {ra[k]['nfev']} -> {rb[k]['nfev']}" for k in shared
            if ra[k]["nfev"] != rb[k]["nfev"]]
    out.append(f"solver evaluations: {len(nfev)} of {len(shared)} runs took a different "
               "path through the integrator")
    out += _listing("evaluations", nfev, limit)

    if trajectories:
        ok = [k for k in shared if ra[k]["status"] == "ok" and rb[k]["status"] == "ok"]
        differ = []
        for i, k in enumerate(ok, 1):
            if i % 500 == 0:
                log(f"  {i} of {len(ok)} trajectories")
            why = _trajectory_difference(A["dir"], B["dir"], k, tol)
            if why is not None:
                differ.append(f"{k} {why}")
        out.append(f"trajectories: {len(ok)} compared, {len(differ)} differ")
        out += _listing("trajectories", differ, limit)

    cost = [(ra[k]["wall_time"], rb[k]["wall_time"]) for k in shared
            if ra[k]["status"] == "ok" and rb[k]["status"] == "ok"]
    changed = [(x, y) for x, y in cost if _moved(x, y, 1e-6)]
    ratios = [y / x for x, y in changed if x]
    ratio = float(np.median(ratios)) if ratios else float("nan")
    out.append(f"cost: wall time differs on {len(changed)} of {len(cost)} runs, B a median "
               f"{ratio:.3f} times A's; a property of the machine, not of the data")
    return out


def _listing(label: str, items: list[str], limit: int) -> list[str]:
    if not items:
        return []
    out = [f"  {label}:"] + [f"    {s}" for s in items[:limit]]
    if len(items) > limit:
        out.append(f"    ... and {len(items) - limit} more")
    return out


def _trajectory_difference(a_dir: Path, b_dir: Path, run_id: str, tol: float) -> str | None:
    """How the two recordings of one run differ, or None when they agree."""
    with np.load(a_dir / "runs" / f"{run_id}.npz") as fa, \
            np.load(b_dir / "runs" / f"{run_id}.npz") as fb:
        worst, where = 0.0, ""
        for key in ("t", "x", "u", "y"):
            xa, xb = fa[key], fb[key]
            if xa.shape != xb.shape:
                return f"{key} is {xa.shape} in A and {xb.shape} in B"
            same = np.array_equal(xa, xb, equal_nan=xa.dtype.kind == "f")
            if xa.size and not same:
                equal = xa == xb
                if xa.dtype.kind == "f":
                    equal |= np.isnan(xa) & np.isnan(xb)
                with np.errstate(invalid="ignore", over="ignore"):
                    d = np.where(equal, 0.0, np.abs(xa - xb) / np.maximum(1.0, np.abs(xa)))
                # A value recorded in one and missing (NaN) in the other differs by any
                # measure; so does an infinity against a number.
                largest = float(np.max(np.where(np.isnan(d), np.inf, d)))
                if largest > worst:
                    worst, where = largest, key
        meta_a, meta_b = json.loads(str(fa["meta"])), json.loads(str(fb["meta"]))
    for key in ("case", "config", "state_names", "u_names", "y_names"):
        if meta_a.get(key) != meta_b.get(key):
            return f"{key} differs"
    return f"{where} moved by up to {worst:.3g}" if worst > tol else None


def compare_results(a: str | Path, b: str | Path, tol: float = TOL,
                    limit: int = 20) -> list[str]:
    """Two results files, aligned on their text and compared on their numbers."""
    la, lb = Path(a).read_text().splitlines(), Path(b).read_text().splitlines()
    sa, sb = [_skeleton(s) for s in la], [_skeleton(s) for s in lb]
    moved, text = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                na, nb = _numbers(la[i]), _numbers(lb[j])
                if any(_moved(x, y, tol) for x, y in zip(na, nb)):
                    moved += [f"{i + 1:>6} A {la[i]}", f"{j + 1:>6} B {lb[j]}"]
        else:
            text += [f"{i + 1:>6} A {la[i]}" for i in range(i1, i2)]
            text += [f"{j + 1:>6} B {lb[j]}" for j in range(j1, j2)]
    out = [f"A {a}: {len(la)} lines", f"B {b}: {len(lb)} lines", ""]
    out.append(f"lines holding a number that moved by more than {tol:g}: {len(moved) // 2}")
    out += _listing("moved", moved, 2 * limit)
    out.append(f"lines that differ in their text: {len(text)}")
    out += _listing("text", text, 2 * limit)
    return out


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _skeleton(line: str) -> str:
    """The line with every number replaced, so two reports align on what they report."""
    return _NUMBER.sub("#", line)


def _numbers(line: str) -> list[float]:
    return [float(m.group()) for m in _NUMBER.finditer(line)]


def _moved(x, y, tol: float) -> bool:
    """Whether two recorded values differ by more than a relative tolerance."""
    if x is None or y is None or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return x != y
    if x == y or (np.isnan(x) and np.isnan(y)):
        return False
    return abs(x - y) > tol * max(1.0, abs(x), abs(y))
