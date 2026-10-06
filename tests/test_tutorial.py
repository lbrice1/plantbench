"""The tutorial runs as written, so the page stays true to the library.

The page is run in order, in one empty directory, as a reader following it would: every
```python block goes into one script, a ```{code-block} toml with a caption is written to
the file it names, and each `$ plantbench` line of a ```console block is run as the
command it shows.  A plain fence holds output, or a step shown but not run.  The Tasks
section needs scikit-learn; without it the page is run up to that section.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys

PAGE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "tutorial.md"
FENCE = re.compile(r"^(?P<ticks>`{3,})(?P<lang>\S*)[ \t]*(?P<arg>\S*)\n(?P<body>.*?)^(?P=ticks)[ \t]*$",
                   re.M | re.S)
CAPTION = re.compile(r"^:caption:[ \t]*(\S+)\n")


def script(text: str) -> str:
    """The page as one Python script, its commands and files in the order they appear."""
    out = ["import subprocess, sys"]
    for m in FENCE.finditer(text):
        lang, body = m["lang"], m["body"]
        if lang == "python":
            out.append(body)
        elif lang == "{code-block}" and (caption := CAPTION.match(body)):
            content = body[caption.end():].lstrip("\n")
            out.append(f"open({caption[1]!r}, 'w').write({content!r})")
        elif lang == "console":
            for line in body.splitlines():
                if line.startswith("$ plantbench "):
                    args = shlex.split(line[2:])[1:]
                    out.append(f"sys.stdout.flush(); subprocess.run([sys.executable, '-m', "
                               f"'plantbench', *{args!r}], check=True)")
    return "\n".join(out) + "\n"


def test_tutorial_runs(tmp_path):
    text = PAGE.read_text()
    if importlib.util.find_spec("sklearn") is None:
        text = text.split("\n## Tasks\n")[0]
    (tmp_path / "tutorial.py").write_text(script(text))
    out = subprocess.run([sys.executable, "tutorial.py"], cwd=tmp_path, capture_output=True,
                         text=True, timeout=300, env={**os.environ, "MPLBACKEND": "Agg"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()
