"""Dataset specifications: which configurations of a case to run, and how.

A specification is a TOML file with these tables:

    [study]      name, case, seed
    [base]       a configuration, as in `core.case.Config`
    [[variants]] optional partial configurations, each merged into the base in turn; or
    [variants]   named groups of them, `plant = [...]`, `upset = [...]`, whose Cartesian
                 product is taken, one member of each group merged in the order given
    [sweep]      dotted path -> list of values; the Cartesian product is taken
    [sample]     n, and dotted path -> [low, high]; a Latin hypercube of n points is drawn
                 for every combination of variant and sweep point
    [run]        t_end, dt, wall_budget, features, and states (false to store only the
                 inputs and measurements of each run, not its full state)

Dotted paths address a configuration section and then its keys, so that
`"tuning.reactor T.Kc"` is the gain of the loop named "reactor T" and
`"params.reactor.U"` is the parameter override whose key is "reactor.U".  A path may also
name a whole section, `"disturbance"`, with whole sections as its values.

`expand` turns a specification into the list of configurations it describes.  It is
deterministic in the seed, so the same file always describes the same runs.

A specification is named by a digest of its canonical text, and reported with a prefix that
says which text was hashed: `pb-spec:6a6fc673261b` names the runs, the specification
without its task, and `pb-protocol:dab0d47475e9` the specification with its task.  The
prefix keeps a digest from being read as a commit.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

# How many path components below each section name a single entry.  None: the rest of
# the path is one key (parameter overrides are themselves dotted paths).
_DEPTH = {"structure": 0, "options": 1, "setpoints": 1, "setpoint_steps": 1, "params": None,
          "tuning": 2, "instruments": 2, "actuators": 2, "disturbance": 1, "mismatch": None}

RUN_DEFAULTS = {"t_end": 1500.0, "dt": 1.0, "wall_budget": None, "features": [],
                "states": True}
# Feature names are not checked here.  A case may supply its own beyond the generic ones of
# `core.case.GENERIC_FEATURES`, and `core.spec` cannot load a case to find out, so the names
# are checked against the case in `datagen.generate`.

TASK_KEYS = {"target", "signals", "window", "split", "metrics", "positive"}
SPLIT_DEFAULTS = {"folds": 5, "stratified": True, "seed": 0}


@dataclass
class Spec:
    name: str
    case: str
    seed: int = 0
    base: dict = field(default_factory=dict)
    variants: list[dict] | dict[str, list[dict]] = field(default_factory=list)
    sweep: dict[str, list] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Spec:
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> Spec:
        known = {"study", "base", "variants", "sweep", "sample", "run", "task"}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown tables {sorted(unknown)} in the specification; "
                           f"known: {sorted(known)}")
        study = dict(d.get("study", {}))
        for key in ("name", "case"):
            if key not in study:
                raise KeyError(f"[study] needs {key!r}")
        extra = set(study) - {"name", "case", "seed"}
        if extra:
            raise KeyError(f"unknown keys {sorted(extra)} in [study]")
        run = {**RUN_DEFAULTS, **d.get("run", {})}
        if set(run) - set(RUN_DEFAULTS):
            raise KeyError(f"unknown keys {sorted(set(run) - set(RUN_DEFAULTS))} in [run]")
        if not all(isinstance(f, str) for f in run["features"]):
            raise TypeError("[run] features must be a list of feature names")
        sample = dict(d.get("sample", {}))
        if sample and "n" not in sample:
            raise KeyError("[sample] needs n, the number of points per combination")
        for path, rng in sample.items():
            if path != "n" and (len(rng) != 2 or not rng[0] < rng[1]):
                raise ValueError(f"[sample] {path!r} needs [low, high] with low < high")
        variants = d.get("variants", [])
        if isinstance(variants, dict):
            variants = {str(k): list(v) for k, v in variants.items()}
            if any(not v for v in variants.values()):
                raise ValueError("every [variants] group needs at least one member")
        else:
            variants = list(variants)
        spec = cls(name=study["name"], case=study["case"], seed=int(study.get("seed", 0)),
                   base=dict(d.get("base", {})), variants=variants,
                   sweep=dict(d.get("sweep", {})), sample=sample, run=run,
                   task=task_table(d.get("task", {})))
        for path in [*spec.sweep, *(k for k in spec.sample if k != "n")]:
            _split(path)  # fail on a malformed path now, not in the middle of a run
        return spec

    def generation(self) -> dict:
        """The part of the specification that decides which runs exist."""
        return {"study": {"name": self.name, "case": self.case, "seed": self.seed},
                "base": self.base, "variants": self.variants, "sweep": self.sweep,
                "sample": self.sample, "run": self.run}

    def to_dict(self) -> dict:
        """The specification as plain data, with the task when it carries one."""
        d = self.generation()
        if self.task:
            d["task"] = self.task
        return d

    def fingerprint(self) -> str:
        """Canonical text of everything that decides which runs exist and what they are.

        The task is left out: it says how runs are scored, not which runs exist, and a
        dataset generated before a task was written to it is still the same dataset.
        """
        return json.dumps(self.generation(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Short name of the runs the specification describes."""
        return digest(self.fingerprint())

    def protocol(self) -> str:
        """Canonical text of the runs and the task together: what a method reports against."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def protocol_digest(self) -> str:
        """Short name of the protocol; the same as `digest` when there is no task."""
        return digest(self.protocol())


def digest(fingerprint: str, length: int | None = 12) -> str:
    """The first `length` hex characters of the SHA-256 of a fingerprint; None for all 64."""
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:length]


SPEC_ID, PROTOCOL_ID = "pb-spec", "pb-protocol"


def identifier(kind: str, digest: str) -> str:
    """A digest as it is reported, `pb-spec:6a6fc673261b`, so that it says what it names."""
    return f"{kind}:{digest}"


def parse_identifier(text: str) -> tuple[str | None, str]:
    """The kind and the hex digest of a reported identifier; kind None for a bare digest.

    A digest of at least 12 and at most 64 hex characters is accepted, so that a report
    may quote more of the SHA-256 than the 12 characters the library prints.
    """
    kind, _, hexdigest = text.strip().rpartition(":")
    if kind not in ("", SPEC_ID, PROTOCOL_ID):
        raise ValueError(f"{text!r}: unknown kind {kind!r}; known: {SPEC_ID}, {PROTOCOL_ID}")
    hexdigest = hexdigest.lower()
    if not 12 <= len(hexdigest) <= 64 or any(c not in "0123456789abcdef" for c in hexdigest):
        raise ValueError(f"{text!r}: a digest is 12 to 64 hex characters")
    return kind or None, hexdigest


def task_table(table: dict) -> dict:
    """The [task] table, validated as plain data.

    Only the shape is checked here.  Whether the target and the metrics name anything is
    decided by `plantbench.task`, which holds the registries; `core` knows no method.
    """
    if not table:
        return {}
    unknown = set(table) - TASK_KEYS
    if unknown:
        raise KeyError(f"unknown keys {sorted(unknown)} in [task]; known: {sorted(TASK_KEYS)}")
    for key in ("target", "signals", "window", "metrics"):
        if key not in table:
            raise KeyError(f"[task] needs {key!r}")
    signals = table["signals"]
    signals = [signals] if isinstance(signals, str) else [str(s) for s in signals]
    if not signals:
        raise ValueError("[task] signals names at least one signal or group of them")
    window = table["window"]
    if not isinstance(window, (list, tuple)) or len(window) != 2 or not window[0] < window[1]:
        raise ValueError("[task] window is [start, end] in minutes, with start < end")
    window = [float(window[0]), float(window[1])]
    split = {**SPLIT_DEFAULTS, **table.get("split", {})}
    extra = set(split) - set(SPLIT_DEFAULTS)
    if extra:
        raise KeyError(f"unknown keys {sorted(extra)} in the [task] split; "
                       f"known: {sorted(SPLIT_DEFAULTS)}")
    if int(split["folds"]) < 2:
        raise ValueError("[task] split needs at least two folds")
    metrics = [str(m) for m in table["metrics"]]
    if not metrics:
        raise ValueError("[task] metrics names at least one metric")
    out = {"target": str(table["target"]), "signals": signals, "window": window,
           "split": {"folds": int(split["folds"]), "stratified": bool(split["stratified"]),
                     "seed": int(split["seed"])},
           "metrics": metrics}
    if table.get("positive") is not None:
        out["positive"] = str(table["positive"])
    return out


def _split(path: str) -> list[str]:
    """[section, key, ...] for a dotted path, keeping multi-part keys whole."""
    section, *rest = path.split(".")
    if section not in _DEPTH:
        raise KeyError(f"{path!r}: {section!r} is not a configuration section; "
                       f"sections: {sorted(_DEPTH)}")
    depth = _DEPTH[section]
    if not rest:
        return [section]  # the whole section
    if depth == 0:
        raise KeyError(f"{path!r}: {section} has no keys below it")
    if depth == 2 and len(rest) > 1:
        return [section, ".".join(rest[:-1]), rest[-1]]  # loop name, then its field
    return [section, ".".join(rest)]


def set_path(config: dict, path: str, value) -> None:
    """Set a value in a configuration dict by dotted path, creating sections as needed."""
    keys = _split(path)
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise TypeError(f"{path!r}: cannot set a key inside a {type(node).__name__}")
    node[keys[-1]] = value


def _merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _merge_all(partials) -> dict:
    out: dict = {}
    for partial in partials:
        out = _merge(out, partial)
    return out


def expand(spec: Spec) -> list[dict]:
    """Every configuration the specification describes, as plain dicts, in a fixed order."""
    if isinstance(spec.variants, dict):
        variants = [_merge_all(combo) for combo in itertools.product(*spec.variants.values())]
    else:
        variants = spec.variants or [{}]
    grid_paths = list(spec.sweep)
    grid = list(itertools.product(*(spec.sweep[p] for p in grid_paths))) or [()]
    sample_paths = [p for p in spec.sample if p != "n"]
    n = int(spec.sample.get("n", 1)) if sample_paths else 1
    configs = []
    for combo, (variant, point) in enumerate(itertools.product(variants, grid)):
        base = _merge(spec.base, variant)
        for p, v in zip(grid_paths, point):
            set_path(base, p, copy.deepcopy(v))
        if not sample_paths:
            configs.append(base)
            continue
        lo = np.array([spec.sample[p][0] for p in sample_paths], dtype=float)
        hi = np.array([spec.sample[p][1] for p in sample_paths], dtype=float)
        unit = qmc.LatinHypercube(d=len(sample_paths),
                                  rng=np.random.default_rng([spec.seed, combo])).random(n)
        for row in qmc.scale(unit, lo, hi):
            c = copy.deepcopy(base)
            for p, v in zip(sample_paths, row):
                set_path(c, p, float(v))
            configs.append(c)
    return configs
