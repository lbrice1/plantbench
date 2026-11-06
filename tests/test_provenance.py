"""`plantbench.provenance`, the record a dataset's manifest holds, for results made outside
`datagen`: the library's commit only from a checkout of its own repository, the dirty flag
from the code alone, and nothing invented when Git is missing."""

from __future__ import annotations

import json
import subprocess

import pytest

import plantbench as pb
from plantbench import _provenance, datagen

KEYS = {"plantbench", "git_commit", "git_dirty", "python", "numpy", "scipy", "platform",
        "started"}


def _git(where, *args):
    subprocess.run(["git", "-C", str(where), *args], check=True, capture_output=True)


def _checkout(root):
    """A repository laid out as this one, committed clean."""
    for path in ("pyproject.toml", "plantbench/x.py", "studies/s/a.py",
                 "studies/s/results/r.txt", "studies/s/figures/f.png", "manuscript/m.tex"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("0\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    return root.resolve()


def test_provenance_holds_what_a_manifest_holds():
    p = pb.provenance()
    assert set(p) == KEYS
    assert p["plantbench"] == pb.__version__
    assert p["started"].endswith("+00:00")


def test_a_checkout_records_its_commit_and_whether_its_code_is_clean(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    monkeypatch.setattr(_provenance, "_SOURCE", root)
    p = pb.provenance()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert (p["git_commit"], p["git_dirty"]) == (head, False)
    (root / "plantbench/x.py").write_text("1\n")
    assert pb.provenance()["git_dirty"] is True


def test_only_the_code_can_make_a_checkout_dirty(tmp_path, monkeypatch):
    """A results file an analysis writes, or a manuscript being edited, is not code."""
    root = _checkout(tmp_path)
    monkeypatch.setattr(_provenance, "_SOURCE", root)
    for path in ("studies/s/results/r.txt", "studies/s/figures/f.png", "manuscript/m.tex"):
        (root / path).write_text("1\n")
    assert _provenance.repo_state()["git_dirty"] is False
    (root / "studies/s/a.py").write_text("1\n")
    assert _provenance.repo_state()["git_dirty"] is True


def test_an_installed_copy_records_no_commit(tmp_path, monkeypatch):
    """An installed copy has no checkout, and the repository that happens to enclose an
    environment is not the one that produced the code."""
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    monkeypatch.setattr(_provenance, "_SOURCE", other / "site-packages")
    assert _provenance.repo_state() == {"git_commit": None, "git_dirty": None}
    monkeypatch.setattr(_provenance, "_SOURCE", other)
    assert _provenance.repo_state() == {"git_commit": None, "git_dirty": None}


def test_without_git_nothing_is_recorded_rather_than_guessed(tmp_path, monkeypatch):
    monkeypatch.setattr(_provenance, "_SOURCE", _checkout(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "no-git-here"))
    p = pb.provenance()
    assert (p["git_commit"], p["git_dirty"]) == (None, None)


def test_a_case_of_ones_own_records_where_it_comes_from():
    from dataclasses import replace

    from plantbench.cases.template import CASE
    pb.register(replace(CASE, id="provenance_tank"))
    try:
        p = pb.provenance("provenance_tank")
    finally:
        pb.cases._SESSION.pop("provenance_tank", None)
    assert p["case_source"]["kind"] == "session"
    assert set(p) == KEYS | {"case_source"}
    assert "case_source" not in pb.provenance("jacketed_cstr")


@pytest.mark.parametrize("make", ["spec", "configurations"])
def test_a_manifest_is_the_provenance_record(tmp_path, make):
    if make == "spec":
        spec = tmp_path / "spec.toml"
        spec.write_text('[study]\nname = "p"\ncase = "jacketed_cstr"\n\n[base]\n'
                        'structure = "open loop"\n\n[run]\nt_end = 1.0\ndt = 0.5\n')
        datagen.generate(spec, tmp_path / "d", log=lambda *a: None)
    else:
        datagen.run_configurations("jacketed_cstr", [{"structure": "open loop"}],
                                   tmp_path / "d", {"t_end": 1.0, "dt": 0.5},
                                   log=lambda *a: None)
    manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
    assert KEYS <= set(manifest)
    p = pb.provenance()
    assert {k: manifest[k] for k in KEYS - {"started"}} == {k: p[k] for k in KEYS - {"started"}}
