"""Reporting a dataset: the card a paper quotes, and the check a reader runs against it.

    plantbench card data/regimes [--format text|md|latex|json]
    plantbench verify data/regimes/spec.toml pb-protocol:dab0d47475e9

`card` gathers what a dataset records about itself -- its study and case, the version and
commit of the code that generated it, its specification and protocol, how its runs ended
and the environment they ran in -- into one block, in the format it is to be pasted into.
Each digest is printed as a typed identifier, `pb-spec:<digest>` or `pb-protocol:<digest>`,
with a sentence defining it, so that a reader knows what was hashed and does not take the
digest for a commit.  What would weaken the report, uncommitted code or a specification
that no longer describes the runs, is printed on the card rather than left for a reader
to find.

`verify` is the reader's side: it recomputes the digests of a specification file or a
dataset and says whether the identifiers a report quotes name it.
"""

from __future__ import annotations

import json
from pathlib import Path

from plantbench import cases, datagen, datasets
from plantbench.core.spec import PROTOCOL_ID, SPEC_ID, Spec, digest, identifier, parse_identifier

FORMATS = ("text", "md", "latex", "json")
_STATUS_TEXT = {"ok": "completed", "invalid": "invalid", "failed": "failed",
                "timeout": "timed out"}


def card(out_dir: str | Path) -> dict:
    """What a paper reports about a dataset, as plain data."""
    s = datasets.summary(out_dir)
    m = s["manifest"] or {}
    out = Path(out_dir)
    spec_file = next((out / name for name in ("spec.toml", "spec.json")
                      if (out / name).exists()), None)
    fingerprint, protocol = m.get("spec_fingerprint"), m.get("protocol_fingerprint")
    citation, citation_warning = _case_citation(s["case"])
    c = {
        "dataset": str(out),
        "study": s["study"],
        "case": s["case"],
        "case_source": (datasets.case_source_text(s["case_source"])
                        if s["case_source"] is not None else None),
        "case_citation": citation,
        "plantbench": m.get("plantbench"),
        "commit": (f"{s['git_commit'][:7]}, "
                   + {True: "uncommitted changes", False: "clean", None: "state unknown"}[
                       s["git_dirty"]] if s["git_commit"] else None),
        "spec_file": str(spec_file) if spec_file is not None else None,
        "spec_id": identifier(SPEC_ID, s["spec_digest"]) if s["spec_digest"] else None,
        "spec_sha256": digest(fingerprint, None) if fingerprint is not None else None,
        "protocol_id": (identifier(PROTOCOL_ID, s["protocol_digest"])
                        if s["protocol_digest"] else None),
        "protocol_sha256": digest(protocol, None) if protocol is not None else None,
        "runs": s["counts"],
        "wall_time": s["wall_time"],
        "environment": {k: m[k] for k in ("python", "numpy", "scipy", "platform") if k in m},
        "warnings": _warnings(s, spec_file) + ([citation_warning] if citation_warning else []),
    }
    c["reproduce"] = _reproduce(c)
    c["verify"] = (f"plantbench verify {c['spec_file']} "
                   f"{c['protocol_id'] or c['spec_id']}"
                   if c["spec_file"] and c["spec_id"] else None)
    return c


def _case_citation(case_id: str | None) -> tuple[dict | None, str | None]:
    """The publication the dataset's case was presented in, read from the card of the case
    as it is installed now, and a warning when the case cannot be loaded to read it.

    Any failure is caught: a case that is not installed, or whose package is broken,
    leaves the rest of the dataset's card as it would be without it."""
    if case_id is None:
        return None, None
    try:
        return cases.citation(cases.load_case(case_id)), None
    except Exception as exc:  # noqa: BLE001 -- reported in the card's warnings
        return None, (f"the case {case_id} could not be loaded ({type(exc).__name__}: "
                      f"{exc}), so whether it has a publication of its own to cite is "
                      "not known")


def _warnings(s: dict, spec_file: Path | None) -> list[str]:
    out = []
    if s["manifest"] is None:
        return ["no manifest: how the runs were generated is not recorded"]
    if s["spec_digest"] is None:
        out.append("no specification: the runs were built from configurations, and "
                   "`generate` cannot regenerate them")
    elif s["spec_matches"] is False:
        out.append(f"{spec_file} no longer describes the runs; the identifiers are those "
                   "of the specification the runs were generated from")
    elif spec_file is None:
        out.append(f"the specification was not kept in {s['dir']}, so a reader cannot "
                   "regenerate the runs from the dataset alone")
    if s["git_dirty"]:
        out.append("generated from a checkout with uncommitted changes, so neither the "
                   "version nor the commit identifies the code")
    if s["case_source"] is not None and s["case_source"].get("git_dirty"):
        out.append("the case came from a checkout with uncommitted changes")
    return out


def _reproduce(c: dict) -> list[str]:
    if c["spec_file"] is None or c["spec_id"] is None:
        return []
    needs = f"with plantbench {c['plantbench']}" if c["plantbench"] else "with plantbench"
    if c["case_source"] is not None:
        needs += f" and the case {c['case']} installed"
    return [needs + ":", f"plantbench generate {c['spec_file']}"]


def definition(c: dict) -> str:
    """The sentence that says what the identifiers on a card are digests of."""
    text = ("`pb-spec:<digest>` is the first 12 hexadecimal characters of the SHA-256 of "
            "the specification's canonical JSON without its task")
    if c["protocol_id"]:
        text += ", and `pb-protocol:<digest>` of the same with its task"
    return text + "."


def rows(c: dict) -> list[tuple[str, str]]:
    """The card as (entry, value) pairs; spans in backticks are literal names."""
    case = f"`{c['case']}`" + (f"; {c['case_source']}" if c["case_source"] else "")
    code = f"`plantbench` {c['plantbench'] or 'version not recorded'}"
    if c["commit"]:
        code += f" (checkout {c['commit']})"
    if c["spec_id"]:
        where = f"`{c['spec_file']}`; " if c["spec_file"] else ""
        spec = f"{where}`{c['spec_id']}`"
    else:
        spec = "none recorded"
    runs = [f"{c['runs'].get('ok', 0)} completed"]
    runs += [f"{n} {_STATUS_TEXT.get(k, k)}" for k, n in c["runs"].items()
             if k != "ok" and n]
    env = c["environment"]
    out = [("Study", f"`{c['study']}`" if c["study"] else "not recorded"),
           ("Case", case if c["case"] else "not recorded"), ("Code", code),
           ("Specification", spec)]
    if c["case_citation"]:
        out.insert(2, ("Case citation", c["case_citation"]["reference"]))
    if c["protocol_id"]:
        out.append(("Protocol", f"specification with its task; `{c['protocol_id']}`"))
    out.append(("Runs", ", ".join(runs) + f"; {_hours(c['wall_time'])} of integration"))
    if env:
        out.append(("Environment", ", ".join(
            f"{label} {env[k]}" for k, label in (("python", "Python"), ("numpy", "NumPy"),
                                                  ("scipy", "SciPy")) if k in env)))
    return out


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def render(c: dict, fmt: str = "text") -> str:
    """The card in one of `FORMATS`."""
    if fmt == "json":
        return json.dumps(c, indent=2)
    if fmt == "md":
        return _markdown(c)
    if fmt == "latex":
        return _latex(c)
    if fmt == "text":
        return _text(c)
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


def _text(c: dict) -> str:
    table = rows(c)
    if c["reproduce"]:
        table += [("Reproduce", c["reproduce"][0]), ("", c["reproduce"][1])]
    if c["verify"]:
        table.append(("Verify", c["verify"]))
    width = max(len(k) for k, _ in table)
    lines = [f"plantbench dataset card: {c['dataset']}"]
    lines += [f"  {k:{width}s}  {_plain(v)}" for k, v in table]
    if c["spec_id"]:
        lines += ["", _plain(definition(c))]
    lines += [f"warning: {_plain(w)}" for w in c["warnings"]]
    return "\n".join(lines)


def _plain(text: str) -> str:
    return text.replace("`", "")


def _markdown(c: dict) -> str:
    lines = ["| Entry | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows(c)]
    if c["spec_id"]:
        lines += ["", definition(c)]
    if c["reproduce"]:
        lines += ["", f"Reproduce {c['reproduce'][0]}", "",
                  "```", c["reproduce"][1], "```"]
    if c["verify"]:
        lines += ["", "Verify:", "", "```", c["verify"], "```"]
    lines += [f"\n> Warning: {w}" for w in c["warnings"]]
    return "\n".join(lines)


_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def _latex_value(text: str) -> str:
    """Backticked spans as \\texttt, the rest escaped."""
    parts = text.split("`")
    return "".join(_latex_escape(p) if i % 2 == 0 else rf"\texttt{{{_latex_escape(p)}}}"
                   for i, p in enumerate(parts))


def _latex(c: dict) -> str:
    lines = [f"% plantbench dataset card: {c['dataset']}",
             "% rows for a two-column tabular; the definition below belongs in the caption"]
    if c["spec_id"]:
        lines.append(f"% {_latex_value(definition(c))}")
    lines += [f"% warning: {_plain(w)}" for w in c["warnings"]]
    lines += [f"{k} & {_latex_value(v)} \\\\" for k, v in rows(c)]
    return "\n".join(lines)


def verify(target: str | Path, identifiers: list[str]) -> tuple[bool, list[str]]:
    """Whether each identifier names the specification file or dataset `target`.

    A typed identifier is checked against the digest of its kind; a bare digest against
    either, and the line says which it matched.  A digest longer than the 12 characters
    the library prints is checked as a prefix of the full SHA-256.
    """
    fingerprint, protocol, notes = _fingerprints(Path(target))
    full = {SPEC_ID: digest(fingerprint, None),
            # With no task the protocol is the specification alone, as in `Spec`.
            PROTOCOL_ID: digest(protocol if protocol is not None else fingerprint, None)}
    ok, lines = True, list(notes)
    for text in identifiers:
        kind, hexdigest = parse_identifier(text)
        matched = [k for k in ((kind,) if kind else (SPEC_ID, PROTOCOL_ID))
                   if full[k].startswith(hexdigest)]
        if matched:
            names = " and ".join(identifier(k, full[k][:12]) for k in matched)
            lines.append(f"match     {text}: {target} is {names}")
        else:
            ok = False
            lines.append(f"MISMATCH  {text}: {target} is {identifier(SPEC_ID, full[SPEC_ID][:12])}"
                         + (f", {identifier(PROTOCOL_ID, full[PROTOCOL_ID][:12])}"
                            if protocol is not None else ", with no task"))
    return ok, lines


def _fingerprints(target: Path) -> tuple[str, str | None, list[str]]:
    """The canonical texts of the runs and the protocol, and notes on where they came from."""
    if target.is_dir():
        if not datagen.load_records(target) and not (target / "manifest.json").exists():
            raise ValueError(f"{target} is not a dataset")
        s = datasets.summary(target)
        m = s["manifest"] or {}
        if m.get("spec_fingerprint") is None:
            raise ValueError(f"{target} records no specification, so it has no digest")
        notes = []
        if s["spec_matches"] is False:
            notes.append(f"note: the specification kept in {target} no longer describes its "
                         "runs; checking the one the runs were generated from")
        return m["spec_fingerprint"], m.get("protocol_fingerprint"), notes
    if target.suffix == ".json":
        spec = Spec.from_dict(json.loads(target.read_text()))
    else:
        spec = Spec.load(target)
    return spec.fingerprint(), spec.protocol() if spec.task else None, []
