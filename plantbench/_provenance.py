"""What a result was made with: the library's version and commit, the case's source, and
the versions of the interpreter and the numerical packages, as a dataset's manifest
records them (`datagen`) and `plantbench.provenance` returns them.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from plantbench import cases

# The checkout the package runs from, when it runs from one: the directory above it.
_SOURCE = Path(__file__).resolve().parent.parent
# What decides a result: the library and the studies, not their outputs.  A manuscript
# being edited, or a results file an analysis has just written, leaves the code clean.
_CODE = ("plantbench", "studies", ":(exclude,glob)studies/*/results/**",
         ":(exclude,glob)studies/*/figures/**")


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd or _SOURCE,
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def repo_state() -> dict:
    """The commit of the checkout the package runs from, and whether its code is uncommitted.

    Only a checkout of this repository has a commit to record.  An installed copy has
    none, and the repository that happens to enclose an environment is not the one that
    produced the code, so both record None rather than a commit that would mislead.
    """
    top = _git("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != _SOURCE or not (_SOURCE / "pyproject.toml").exists():
        return {"git_commit": None, "git_dirty": None}
    status = _git("status", "--porcelain", "--", *_CODE)
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def case_source(case_id: str) -> dict | None:
    """Where a case that is not built in comes from, for the manifest: its module, its
    distribution and version when installed, and the commit of its checkout.

    The commit is that of the repository holding the case's module, unless the module is
    an installed copy under site-packages, where the enclosing repository, if any, is not
    the one that produced it.  None for a built-in case, which `repo_state` covers.
    """
    src = cases.source(case_id)
    if src["kind"] == "built in":
        return None
    out = dict(src)
    module = sys.modules.get(src["module"]) or importlib.import_module(src["module"])
    path = getattr(module, "__file__", None)
    if src["module"] in ("__main__", "__mp_main__") and path is not None:
        out["file"] = str(Path(path).resolve())
    out.update(git_commit=None, git_dirty=None)
    if path is not None and "site-packages" not in Path(path).parts:
        where = Path(path).resolve().parent
        commit = _git("rev-parse", "HEAD", cwd=where)
        if commit is not None:
            status = _git("status", "--porcelain", "--", ".", cwd=where)
            out.update(git_commit=commit, git_dirty=None if status is None else bool(status))
    return out


def provenance(case: str | None = None) -> dict:
    """The record of what a result is made with; see `plantbench.provenance`."""
    import scipy

    from plantbench import __version__
    out = {}
    if case is not None and (source := case_source(case)) is not None:
        out["case_source"] = source
    out.update({
        "plantbench": __version__, **repo_state(),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return out
