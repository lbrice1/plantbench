"""plantbench: plantwide process simulations as test problems.

    import plantbench as pb
    case = pb.load_case("reactor_separator_recycle")
    traj = pb.run("reactor_separator_recycle", case.config(disturbance={"name": "throughput", "fraction": 0.1}),
                  t_end=1500.0, dt=1.0)

See docs/user-guide.md.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from plantbench import _provenance
from plantbench.cases import case, list_cases, load_case, register
from plantbench.core import case as _case
from plantbench.core.case import Case, Config, Setup, Trajectory

try:
    __version__ = version("plantbench")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"


def _resolve(case: str | Case) -> Case:
    return load_case(case) if isinstance(case, str) else case


def build(case: str | Case, config: Config | dict | None = None) -> Setup:
    """A configured plant at its design point.  No configuration means the reference plant."""
    case = _resolve(case)
    return _case.build(case, case.config() if config is None else config)


def run(case: str | Case, config: Config | dict | None = None, t_end: float = 1500.0,
        **kwargs) -> Trajectory:
    """Simulate a configuration of a case; keyword arguments go to `core.case.run`."""
    case = _resolve(case)
    return _case.run(case, case.config() if config is None else config, t_end, **kwargs)


def run_ensemble(case: str | Case, configs: list, t_end: float = 1500.0,
                 **kwargs) -> list[_case.MemberResult]:
    """Simulate configurations of a case as ensembles in lockstep; keyword arguments go to
    `core.case.run_ensemble`."""
    return _case.run_ensemble(_resolve(case), list(configs), t_end, **kwargs)


def provenance(case: str | None = None) -> dict:
    """The record of what a result is made with, as a dataset's manifest holds it.

    `plantbench` is the library's version; `git_commit` and `git_dirty` its commit and
    whether its code is uncommitted, both None unless it runs from a checkout of its
    repository (`repo_state`); `python`, `numpy`, `scipy` and `platform` the environment;
    and `started` the time of the call, in UTC, which a program takes when it starts.
    With the id of a case that is not built in, `case_source` says where that case comes
    from (`case_source`); a built-in case is covered by the library's own commit.

        import json, plantbench as pb
        json.dump(pb.provenance("my_tank"), open("run.json", "w"))
    """
    return _provenance.provenance(case)


__all__ = ["Case", "Config", "Setup", "Trajectory", "build", "case", "list_cases", "load_case",
           "provenance", "register", "run", "run_ensemble", "__version__"]
