"""Dataset generation: every run recorded once, resumable, failures kept as results."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from plantbench import datagen
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


def test_generate_records_every_run(tmp_path):
    out = datagen.generate(Spec.from_dict(SPEC), tmp_path / "d", log=_quiet)
    records = datagen.load_records(out)
    assert len(records) == 6
    ok = [r for r in records if r["status"] == "ok"]
    invalid = [r for r in records if r["status"] == "invalid"]
    assert len(ok) == 3 and len(invalid) == 3
    assert all("branch" in r["message"] for r in invalid)
    for r in ok:
        tr = datagen.load_run(out, r["run_id"])
        assert tr.x.shape == (3, 13) and r["features"]["damping_ratio"] > 0
        assert tr.config == r["config"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["case"] == "jacketed_cstr" and manifest["numpy"] == np.__version__
    with open(out / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and "config.tuning.reactor T.tau_I" in rows[0]


def test_generation_resumes_where_it_stopped(tmp_path):
    spec = Spec.from_dict(SPEC)
    out = datagen.generate(spec, tmp_path / "d", log=_quiet)
    lines = (out / "runs.jsonl").read_text().splitlines()
    (out / "runs.jsonl").write_text("\n".join(lines[:4]) + "\n")  # as if stopped after 4
    messages = []
    datagen.generate(spec, out, log=messages.append)
    assert "2 to run" in messages[0]
    assert len(datagen.load_records(out)) == 6
    messages.clear()
    datagen.generate(spec, out, log=messages.append)
    assert "0 to run" in messages[0]


def test_a_record_cut_off_mid_write_is_run_again(tmp_path):
    spec = Spec.from_dict(SPEC)
    out = datagen.generate(spec, tmp_path / "d", log=_quiet)
    text = (out / "runs.jsonl").read_text()
    (out / "runs.jsonl").write_text(text[: len(text) - 40])
    assert len(datagen.load_records(out)) == 5
    datagen.generate(spec, out, log=_quiet)
    assert len({r["run_id"] for r in datagen.load_records(out)}) == 6


def test_states_can_be_left_out_of_a_dataset(tmp_path):
    spec = Spec.from_dict({**SPEC, "variants": [{}], "sample": {},
                           "run": {**SPEC["run"], "states": False}})
    out = datagen.generate(spec, tmp_path / "d", log=_quiet)
    (r,) = datagen.load_records(out)
    tr = datagen.load_run(out, r["run_id"])
    assert tr.x.shape == (0, 13) and tr.state_names == [] and tr.y["T"].shape == (13,)


def test_a_directory_holds_one_specification(tmp_path):
    datagen.generate(Spec.from_dict(SPEC), tmp_path / "d", log=_quiet)
    other = Spec.from_dict({**SPEC, "run": {**SPEC["run"], "t_end": 30.0}})
    with pytest.raises(ValueError, match="different specification"):
        datagen.generate(other, tmp_path / "d", log=_quiet)


def test_a_run_over_budget_is_recorded_as_a_timeout(tmp_path):
    spec = Spec.from_dict({**SPEC, "variants": [{}], "sample": {},
                           "run": {**SPEC["run"], "wall_budget": 0.0}})
    out = datagen.generate(spec, tmp_path / "d", log=_quiet)
    (r,) = datagen.load_records(out)
    assert r["status"] == "timeout" and not (out / "runs" / f"{r['run_id']}.npz").exists()


def test_parallel_generation_gives_the_same_runs(tmp_path):
    a = datagen.generate(Spec.from_dict(SPEC), tmp_path / "a", log=_quiet)
    b = datagen.generate(Spec.from_dict(SPEC), tmp_path / "b", workers=2, log=_quiet)
    ra = {r["run_id"]: r["status"] for r in datagen.load_records(a)}
    rb = {r["run_id"]: r["status"] for r in datagen.load_records(b)}
    assert ra == rb
    ok = next(k for k, v in ra.items() if v == "ok")
    assert np.array_equal(datagen.load_run(a, ok).x, datagen.load_run(b, ok).x)


def test_the_command_line_generates_from_a_file(tmp_path, capsys):
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
    assert main(["generate", str(spec), "--out", str(tmp_path / "d")]) == 0
    assert (tmp_path / "d" / "spec.toml").read_text() == spec.read_text()
    assert [r["status"] for r in datagen.load_records(tmp_path / "d")] == ["ok", "ok"]
    assert main(["run", "jacketed_cstr", "--t-end", "10", "--out", str(tmp_path / "r.npz")]) == 0
    assert (tmp_path / "r.npz").exists() and "reactor" not in capsys.readouterr().err


def test_a_design_that_does_not_solve_is_recorded_not_raised(tmp_path):
    import dataclasses

    import plantbench as pb
    from plantbench.cases.jacketed_cstr.definition import CASE

    def fails(config):
        raise RuntimeError("design did not converge")

    pb.register(dataclasses.replace(CASE, id="broken", make_design=fails))
    try:
        r = datagen.run_one("broken", {"structure": "temperature PI"},
                            {"t_end": 10.0, "dt": 5.0, "wall_budget": None, "features": []},
                            str(tmp_path))
    finally:
        pb.cases._SESSION.pop("broken")
    assert r["status"] == "failed" and "did not converge" in r["message"]


def test_a_sequential_design_can_add_batches_to_one_dataset(tmp_path):
    """The unit of work is a list of configurations, however it was chosen."""
    out = tmp_path / "d"
    batch = [{"structure": "temperature PI",
              "tuning": {"reactor T": {"Kc": kc, "tau_I": 20.0}}} for kc in (-1.0, -0.5)]
    first = datagen.run_configurations("jacketed_cstr", batch, out, {"t_end": 20.0, "dt": 5.0},
                                       log=_quiet)
    assert [r["status"] for r in first] == ["ok", "ok"]
    # The next batch repeats one configuration and proposes a new one.
    second = datagen.run_configurations(
        "jacketed_cstr", [batch[1], {"structure": "temperature PI",
                             "tuning": {"reactor T": {"Kc": -1.5, "tau_I": 20.0}}}],
        out, {"t_end": 20.0, "dt": 5.0}, log=_quiet)
    assert len(second) == 1 and len(datagen.load_records(out)) == 3


def test_a_dataset_built_from_configurations_describes_itself(tmp_path):
    out = tmp_path / "d"
    batch = [{"structure": "temperature PI", "tuning": {"reactor T": {"Kc": -1.0}}}]
    datagen.run_configurations("jacketed_cstr", batch, out, {"t_end": 20.0, "dt": 5.0}, log=_quiet)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["case"] == "jacketed_cstr" and manifest["spec_fingerprint"] is None
    assert manifest["numpy"] == np.__version__ and "runs_before_manifest" not in manifest
    with pytest.raises(ValueError, match="jacketed_cstr"):
        datagen.run_configurations("reactor_separator_recycle", [{}], out, log=_quiet)


def test_runs_made_before_a_manifest_are_counted_as_of_unknown_provenance(tmp_path):
    out = tmp_path / "d"
    batch = [{"structure": "temperature PI", "tuning": {"reactor T": {"Kc": kc}}}
             for kc in (-1.0, -0.5)]
    datagen.run_configurations("jacketed_cstr", batch, out, {"t_end": 20.0, "dt": 5.0}, log=_quiet)
    (out / "manifest.json").unlink()  # as a dataset made before manifests were written
    datagen.run_configurations("jacketed_cstr", batch, out, {"t_end": 20.0, "dt": 5.0}, log=_quiet)
    assert json.loads((out / "manifest.json").read_text())["runs_before_manifest"] == 2


def test_a_malformed_configuration_is_recorded_without_stopping_the_batch(tmp_path):
    batch = [{"structure": "temperature PI", "tunings": {"reactor T": {"Kc": -1.0}}},
             {"structure": "temperature PI"}]
    records = datagen.run_configurations("jacketed_cstr", batch, tmp_path / "d",
                                         {"t_end": 20.0, "dt": 5.0}, log=_quiet)
    assert sorted(r["status"] for r in records) == ["invalid", "ok"]
    assert "tunings" in next(r for r in records if r["status"] == "invalid")["message"]


def test_a_run_recorded_twice_is_read_back_once(tmp_path):
    out = datagen.generate(Spec.from_dict(SPEC), tmp_path / "d", log=_quiet)
    lines = (out / "runs.jsonl").read_text().splitlines()
    (out / "runs.jsonl").write_text("\n".join(lines + lines[:2]) + "\n")
    records = datagen.load_records(out)
    assert len(records) == 6 and len({r["run_id"] for r in records}) == 6


@pytest.mark.skipif(datagen.fcntl is None, reason="generations are not guarded on Windows")
def test_a_second_generation_into_the_same_directory_is_refused(tmp_path):
    out = tmp_path / "d"
    out.mkdir()
    with datagen._exclusive(out):
        with pytest.raises(RuntimeError, match="another generation"):
            datagen.run_configurations("jacketed_cstr", [{}], out, log=_quiet)


def test_provenance_is_recorded_only_from_a_checkout_of_this_repository(tmp_path, monkeypatch):
    """An environment inside some other repository must not record that repository's commit."""
    import subprocess
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    monkeypatch.setattr(datagen, "_SOURCE", other)
    assert datagen.repo_state() == {"git_commit": None, "git_dirty": None}


def test_only_the_code_can_make_a_checkout_dirty(tmp_path, monkeypatch):
    """A results file an analysis writes, or a manuscript being edited, is not code."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    for path in ("pyproject.toml", "plantbench/x.py", "studies/s/a.py",
                 "studies/s/results/r.txt", "studies/s/figures/f.png", "manuscript/m.tex"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("0\n")
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    monkeypatch.setattr(datagen, "_SOURCE", tmp_path.resolve())
    assert datagen.repo_state()["git_dirty"] is False
    for path in ("studies/s/results/r.txt", "studies/s/figures/f.png", "manuscript/m.tex"):
        (tmp_path / path).write_text("1\n")
    assert datagen.repo_state()["git_dirty"] is False
    (tmp_path / "studies/s/a.py").write_text("1\n")
    assert datagen.repo_state()["git_dirty"] is True


# -- features a case supplies itself -----------------------------------------------------

CASE0_SPEC = {
    "study": {"name": "f", "case": "reactor_separator_recycle", "seed": 1},
    "base": {"structure": "effluent fixed + cascade", "options": {"network": "D3"}},
    "run": {"t_end": 10.0, "dt": 5.0, "features": ["spectrum", "economics"]},
}


def test_an_unknown_feature_is_refused_before_any_run(tmp_path):
    """`core` cannot know a case's features, so `generate` checks them against the case."""
    spec = Spec.from_dict({**SPEC, "run": {**SPEC["run"], "features": ["economics"]}})
    with pytest.raises(KeyError, match="no features"):
        datagen.generate(spec, tmp_path / "d", log=_quiet)
    assert not (tmp_path / "d" / "runs.jsonl").exists()


def test_a_case_feature_is_recorded_beside_the_generic_ones(tmp_path):
    out = datagen.generate(Spec.from_dict(CASE0_SPEC), tmp_path / "d", log=_quiet)
    records = datagen.load_records(out)
    assert len(records) == 1
    features = records[0]["features"]
    assert records[0]["status"] == "ok"
    # the generic feature, and the one `reactor_separator_recycle` supplies
    assert features["damping_ratio"] == pytest.approx(0.157, abs=5e-4)
    assert features["utility_cost"] == pytest.approx(381.0, abs=0.5)
    assert features["gwp_energy"] == pytest.approx(4.049, abs=5e-4)


def test_a_feature_is_recorded_even_when_the_run_is_too_short_to_say_anything(tmp_path):
    """Features are read from the design point, so they do not depend on the trajectory."""
    out = datagen.generate(Spec.from_dict(CASE0_SPEC), tmp_path / "d", log=_quiet)
    short = datagen.load_records(out)[0]["features"]
    other = {**CASE0_SPEC, "run": {**CASE0_SPEC["run"], "t_end": 20.0}}
    out2 = datagen.generate(Spec.from_dict(other), tmp_path / "e", log=_quiet)
    assert datagen.load_records(out2)[0]["features"] == short


# ------------------------------------------------------------------------------------
# Ensembles
# ------------------------------------------------------------------------------------

TANKS_SPEC = {
    "study": {"name": "e", "case": "tanks", "seed": 2},
    "base": {"structure": "cascade", "disturbance": {"name": "step", "field": "d",
                                                     "value": 0.1, "t": 2.0}},
    "variants": [{}, {"options": {"x": 1.0}}],
    "sample": {"n": 4, "tuning.level 1.Kc": [-0.6, -0.2]},
    "run": {"t_end": 10.0, "dt": 1.0, "features": ["spectrum"]},
}


@pytest.fixture
def tanks():
    from _tanks import TANKS_CASE

    from plantbench.cases import register
    register(TANKS_CASE)


def test_an_ensemble_generation_records_what_a_serial_one_does(tmp_path, tanks):
    a = datagen.generate(Spec.from_dict(TANKS_SPEC), tmp_path / "a", log=_quiet)
    b = datagen.generate(Spec.from_dict(TANKS_SPEC), tmp_path / "b", log=_quiet,
                         solver={"ensemble": 3})
    ra = {r["run_id"]: r for r in datagen.load_records(a)}
    rb = {r["run_id"]: r for r in datagen.load_records(b)}
    assert {k: r["status"] for k, r in ra.items()} == {k: r["status"] for k, r in rb.items()}
    ok = [k for k, r in rb.items() if r["status"] == "ok"]
    assert len(ok) == 4 and len(rb) == 8
    assert sorted(rb[k]["ensemble_size"] for k in ok) == [1, 3, 3, 3]
    for k in ok:
        assert rb[k]["solver"] == {"ensemble": 3, "device": "cpu"}
        assert rb[k]["features"] == ra[k]["features"]
        xa, xb = datagen.load_run(a, k).x, datagen.load_run(b, k).x
        assert np.max(np.abs(xb - xa) / (1.0 + np.abs(xa))) < 1e-6
    messages = []
    datagen.generate(Spec.from_dict(TANKS_SPEC), b, log=messages.append,
                     solver={"ensemble": 3})
    assert "0 to run" in messages[0]


def test_an_ensemble_runs_in_one_process(tmp_path, tanks):
    with pytest.raises(ValueError, match="one process"):
        datagen.generate(Spec.from_dict(TANKS_SPEC), tmp_path / "a", log=_quiet, workers=2,
                         solver={"ensemble": 3})


def test_the_command_line_generates_ensembles(tmp_path, tanks, capsys):
    spec = tmp_path / "spec.toml"
    spec.write_text('''
[study]
name = "cli"
case = "tanks"

[base]
structure = "cascade"

[sweep]
"tuning.level 1.Kc" = [-0.6, -0.4, -0.2]

[run]
t_end = 10.0
dt = 1.0
''')
    assert main(["generate", str(spec), "--out", str(tmp_path / "d"), "--ensemble", "4"]) == 0
    records = datagen.load_records(tmp_path / "d")
    assert [r["status"] for r in records] == ["ok"] * 3
    assert all(r["solver"] == {"ensemble": 4, "device": "cpu"} and r["ensemble_size"] == 3
               for r in records)
