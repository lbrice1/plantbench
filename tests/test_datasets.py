"""Reading the datasets on disk: what one says it is, and what two of them disagree on."""

from __future__ import annotations

import json

import numpy as np
import pytest

from plantbench import datagen, datasets
from plantbench.__main__ import main
from plantbench.core.spec import Spec

SPEC = {
    "study": {"name": "t", "case": "jacketed_cstr", "seed": 1},
    "base": {"structure": "temperature PI",
             "disturbance": {"name": "feed_temperature", "change": 2.0, "t": 5.0}},
    "variants": [{}, {"options": {"branch": "nowhere"}}],
    "sample": {"n": 3, "tuning.reactor T.tau_I": [10.0, 30.0]},
    "run": {"t_end": 60.0, "dt": 5.0, "features": ["spectrum"]},
}


def _quiet(msg):
    pass


def _dataset(path, spec: dict | None = None):
    return datagen.generate(Spec.from_dict(spec or SPEC), path, log=_quiet)


def test_summary_reports_what_the_dataset_holds(tmp_path):
    s = datasets.summary(_dataset(tmp_path / "d"))
    assert s["study"] == "t" and s["case"] == "jacketed_cstr"
    assert s["counts"] == {"ok": 3, "invalid": 3, "failed": 0, "timeout": 0}
    assert s["runs"] == 6 and s["trajectories"] == 3 and s["wall_time"] > 0
    assert s["spec_digest"] == Spec.from_dict(SPEC).digest()
    assert s["spec_matches"] is True and s["git_dirty"] in (True, False, None)


def test_a_specification_that_no_longer_describes_the_runs_is_reported(tmp_path):
    out = _dataset(tmp_path / "d")
    spec = json.loads((out / "spec.json").read_text())
    spec["run"]["t_end"] = 30.0
    (out / "spec.json").write_text(json.dumps(spec))
    assert datasets.summary(out)["spec_matches"] is False
    (out / "spec.json").write_text("{ not a specification")
    assert datasets.summary(out)["spec_matches"] is False
    (out / "spec.json").unlink()
    assert datasets.summary(out)["spec_matches"] is None


def test_find_skips_what_is_not_a_dataset(tmp_path):
    _dataset(tmp_path / "d")
    (tmp_path / "logs").mkdir()
    assert datasets.find(tmp_path) == [tmp_path / "d"]
    assert datasets.find(tmp_path / "d") == [tmp_path / "d"]
    assert datasets.find(tmp_path / "nothing") == []


def test_runs_without_a_manifest_are_listed_with_their_provenance_unknown(tmp_path, capsys):
    out = _dataset(tmp_path / "d")
    (out / "manifest.json").unlink()
    assert datasets.find(tmp_path) == [out]
    s = datasets.summary(out)
    assert s["manifest"] is None and s["git_commit"] is None and s["counts"]["ok"] == 3
    assert main(["datasets", str(tmp_path)]) == 0
    assert "no manifest" in capsys.readouterr().out


def test_a_value_missing_from_one_recording_is_a_difference(tmp_path):
    a, b = _dataset(tmp_path / "a"), _dataset(tmp_path / "b")
    run_id = next(r["run_id"] for r in datagen.load_records(b) if r["status"] == "ok")
    path = b / "runs" / f"{run_id}.npz"
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["u"] = arrays["u"].copy()
    arrays["u"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    assert datasets._trajectory_difference(a, b, run_id, datasets.TOL) is not None


def test_a_recording_without_controller_signals_is_named_as_such(tmp_path):
    """A run written before the signals were recorded, against one written after."""
    a, b = _dataset(tmp_path / "a"), _dataset(tmp_path / "b")
    run_id = next(r["run_id"] for r in datagen.load_records(b) if r["status"] == "ok")
    path = a / "runs" / f"{run_id}.npz"
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if k != "c"}
    meta = json.loads(str(arrays["meta"]))
    del meta["c_names"]
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    assert (datasets._trajectory_difference(a, b, run_id, datasets.TOL)
            == "controller signals recorded only in B")


def test_two_generations_of_one_specification_agree(tmp_path):
    report = "\n".join(datasets.compare(_dataset(tmp_path / "a"), _dataset(tmp_path / "b")))
    assert "specification: the same" in report
    assert "runs: 6 in both, 0 in A only, 0 in B only, 0 whose status differs" in report
    assert "features: 6 runs compared over 3 recorded features, 0 moved" in report
    assert "solver evaluations: 0 of 6 runs took a different path" in report
    assert "trajectories: 3 compared, 0 differ" in report


def test_a_missing_run_and_a_changed_trajectory_are_named(tmp_path):
    a, b = _dataset(tmp_path / "a"), _dataset(tmp_path / "b")
    ok = [r["run_id"] for r in datagen.load_records(b) if r["status"] == "ok"]
    dropped, changed = ok[0], ok[1]
    kept = [ln for ln in (b / "runs.jsonl").read_text().splitlines() if dropped not in ln]
    (b / "runs.jsonl").write_text("\n".join(kept) + "\n")
    with np.load(b / "runs" / f"{changed}.npz") as f:
        arrays = dict(f)
    arrays["y"] = arrays["y"] + 1.0
    np.savez_compressed(b / "runs" / f"{changed}.npz", **arrays)
    report = "\n".join(datasets.compare(a, b))
    assert "runs: 5 in both, 1 in A only, 0 in B only" in report
    assert dropped in report
    assert "trajectories: 2 compared, 1 differ" in report and changed in report


def test_a_wall_time_that_moved_is_not_a_difference_in_the_data(tmp_path):
    a, b = _dataset(tmp_path / "a"), _dataset(tmp_path / "b")
    was = {r["run_id"]: r["wall_time"] for r in datagen.load_records(a)}
    records = [json.loads(ln) for ln in (b / "runs.jsonl").read_text().splitlines()]
    for r in records:
        r["wall_time"] = 3.0 * was[r["run_id"]]
    (b / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    report = "\n".join(datasets.compare(a, b))
    assert "trajectories: 3 compared, 0 differ" in report
    assert "wall time differs on 3 of 3 runs, B a median 3.000 times A's" in report


def test_results_files_are_compared_on_their_numbers(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("t-SNE clusters 34 noise 33.3%\nseed 0 ARI 0.552\nheader\n")
    b.write_text("t-SNE clusters 31 noise 42.5%\nseed 0 ARI 0.55200000001\nfooter\n")
    report = "\n".join(datasets.compare(a, b, tol=1e-6))
    assert "lines holding a number that moved by more than 1e-06: 1" in report
    assert "clusters 34" in report and "clusters 31" in report
    assert "lines that differ in their text: 2" in report
    with pytest.raises(ValueError, match="two dataset directories"):
        datasets.compare(tmp_path, a)


def test_a_stamp_is_refused_by_the_dataset_it_is_not_of(tmp_path):
    a = _dataset(tmp_path / "a")
    other = _dataset(tmp_path / "b", {**SPEC, "run": {**SPEC["run"], "t_end": 30.0}})
    mark = datasets.stamp(a)
    assert mark["spec_digest"] == Spec.from_dict(SPEC).digest() and mark["runs"] == 3
    assert datasets.check_stamp(mark, a) is None
    assert "specification" in datasets.check_stamp(mark, other)
    assert "runs" in datasets.check_stamp({**mark, "runs": 2}, a)
    assert datasets.stamp_lines(mark)[0].startswith(f"dataset {a}")
    assert datasets.stamp_lines(datasets.stamp())[0].startswith("analyzed at")


def test_the_command_line_lists_and_compares(tmp_path, capsys):
    spec = tmp_path / "spec.toml"
    spec.write_text('''
[study]
name = "cli"
case = "jacketed_cstr"

[base]
structure = "open loop"

[sweep]
"options.branch" = ["low", "high"]

[run]
t_end = 20.0
dt = 10.0
''')
    assert main(["generate", str(spec), "--out", str(tmp_path / "a")]) == 0
    capsys.readouterr()
    assert main(["datasets", str(tmp_path)]) == 0
    listed = capsys.readouterr().out
    assert "specification on disk" in listed and "2 ok" in listed and "matches" in listed
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "a")]) == 0
    assert "trajectories: 2 compared, 0 differ" in capsys.readouterr().out
    assert main(["datasets", str(tmp_path / "nothing")]) == 0
    assert "no dataset under" in capsys.readouterr().out


def test_a_manifest_that_names_no_specification_is_still_read(tmp_path, capsys):
    out = _dataset(tmp_path / "d")
    manifest = json.loads((out / "manifest.json").read_text())
    del manifest["spec_fingerprint"]
    (out / "manifest.json").write_text(json.dumps(manifest))
    s = datasets.summary(out)
    assert s["spec_digest"] is None and s["spec_matches"] is None and s["counts"]["ok"] == 3
    assert main(["datasets", str(tmp_path)]) == 0
    assert "not kept" in capsys.readouterr().out
