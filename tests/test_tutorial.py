"""The tutorial runs as written, so the pages stay true to the library.

A page is run in order, in one directory, as a reader following it would: every
```python block goes into one script, a ```{code-block} with a caption is written to the
file it names, and each `$ ` line of a ```console block is run as the command it shows.
A plain fence holds output, or a step shown but not run.

`docs/tutorial/using.md` runs in an empty directory.  The Tasks section needs
scikit-learn; without it the page is run up to that section.

`docs/tutorial/contributing.md` writes into a clone of the library, so it runs in a
temporary copy of the repository, put first on PYTHONPATH and made a git repository of
its own, and the test fails unless plantbench is imported from that copy.  Its `pip
install -e` is emulated without touching the environment: the package's directory goes on
PYTHONPATH, and a dist-info directory declaring its entry points into a directory on
PYTHONPATH as well; `pip uninstall` removes both.  The page ends by restoring the copy,
so the test also requires `git status` of the copy to be empty at the end.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "tutorial"
FENCE = re.compile(r"^(?P<ticks>`{3,})(?P<lang>\S*)[ \t]*(?P<arg>\S*)\n(?P<body>.*?)^(?P=ticks)[ \t]*$",
                   re.M | re.S)
CAPTION = re.compile(r"^:caption:[ \t]*(\S+)\n")

# Run before the page: `sh` runs one `$ ` line, with `pip` emulated as the docstring says.
PRELUDE = '''\
import os, pathlib, shlex, shutil, subprocess, sys, tomllib

SITE = pathlib.Path(os.environ.get("TUTORIAL_SITE", "site")).resolve()
PATHS = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
INSTALLED = {}


def _env():
    return {**os.environ, "PYTHONPATH": os.pathsep.join(PATHS + list(INSTALLED.values()))}


def _pip(args):
    if args[:2] == ["install", "-e"]:
        project = pathlib.Path(args[2]).resolve()
        meta = tomllib.loads((project / "pyproject.toml").read_text())["project"]
        info = SITE / f"{meta['name'].replace('-', '_')}-{meta['version']}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\\nName: {meta['name']}\\n"
                                       f"Version: {meta['version']}\\n")
        (info / "entry_points.txt").write_text("".join(
            f"[{group}]\\n" + "".join(f"{k} = {v}\\n" for k, v in eps.items())
            for group, eps in meta.get("entry-points", {}).items()))
        INSTALLED[meta["name"]] = str(project)
    elif args[:2] == ["uninstall", "-y"]:
        INSTALLED.pop(args[2])
        for info in SITE.glob(f"{args[2].replace('-', '_')}-*.dist-info"):
            shutil.rmtree(info)
    else:
        raise ValueError(f"pip {args} is not emulated")


def sh(line):
    argv = shlex.split(line)
    sys.stdout.flush()
    if argv[0] == "pip":
        return _pip(argv[1:])
    if argv[0] in ("plantbench", "pytest", "ruff"):
        argv = [sys.executable, "-m", *argv]
    subprocess.run(argv, check=True, env=_env())
'''


def script(text: str) -> str:
    """The page as one Python script, its commands and files in the order they appear."""
    out = [PRELUDE]
    for m in FENCE.finditer(text):
        lang, body = m["lang"], m["body"]
        if lang == "python":
            out.append(body)
        elif lang == "{code-block}" and (caption := CAPTION.match(body)):
            content = body[caption.end():].lstrip("\n")
            out.append(f"open({caption[1]!r}, 'w').write({content!r})")
        elif lang == "console":
            out += [f"sh({line[2:]!r})" for line in body.splitlines() if line.startswith("$ ")]
    return "\n".join(out) + "\n"


def _run(page: str, cwd: pathlib.Path, env: dict) -> subprocess.CompletedProcess:
    (cwd.parent / "tutorial.py").write_text(script(page))
    return subprocess.run([sys.executable, str(cwd.parent / "tutorial.py")], cwd=cwd,
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "MPLBACKEND": "Agg", **env})


def test_using_runs(tmp_path):
    text = (PAGES / "using.md").read_text()
    if importlib.util.find_spec("sklearn") is None:
        text = text.split("\n## Tasks\n")[0]
    (tmp_path / "work").mkdir()
    out = _run(text, tmp_path / "work", {})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()


def _git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=tutorial", "-c", "user.email=tutorial@test",
                           *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_contributing_runs(tmp_path):
    clone = tmp_path / "plantbench"
    files = _git("ls-files", "--cached", "--others", "--exclude-standard", "-z", cwd=REPO)
    for name in filter(None, files.split("\0")):
        if (REPO / name).is_file():
            (clone / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, clone / name)
    _git("init", "-q", cwd=clone)
    _git("add", "-A", cwd=clone)
    _git("commit", "-q", "-m", "copy", cwd=clone)
    env = {"PYTHONPATH": os.pathsep.join([str(clone), str(tmp_path / "site")]),
           "TUTORIAL_SITE": str(tmp_path / "site")}
    where = subprocess.run([sys.executable, "-c", "import plantbench; print(plantbench.__file__)"],
                           cwd=clone, capture_output=True, text=True,
                           env={**os.environ, **env}).stdout
    if not pathlib.Path(where.strip()).resolve().is_relative_to(clone.resolve()):
        pytest.fail(f"plantbench is imported from {where.strip()}, not the copy")

    out = _run((PAGES / "contributing.md").read_text(), clone, env)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "adding my_tank" in out.stdout
    assert _git("status", "--porcelain", cwd=clone) == ""
    assert not (tmp_path / "my_tank").exists()
