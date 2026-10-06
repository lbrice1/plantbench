"""The command line: `plantbench`, its help and its version."""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

import plantbench as pb
from plantbench.__main__ import main

COMMANDS = ("list", "describe", "run", "generate", "datasets", "compare", "card", "verify",
            "new-case", "check")


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: plantbench ")
    assert all(f"\n    {c} " in out for c in COMMANDS)
    assert all(f"plantbench {c}" in out for c in COMMANDS)


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"plantbench {pb.__version__}"


def test_the_distribution_installs_the_command():
    (script,) = [ep for ep in entry_points(group="console_scripts") if ep.name == "plantbench"]
    assert script.value == "plantbench.__main__:main"
