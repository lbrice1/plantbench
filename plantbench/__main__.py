"""Command line: list and describe cases, run one configuration, generate a dataset.

    plantbench list
    plantbench describe reactor_separator_recycle
    plantbench run jacketed_cstr [--config config.toml] [--t-end 600] [--dt 1] [--out run.npz]
    plantbench generate spec.toml [--out DIR] [--workers N] [--jacobian batched] [--device cuda]
    plantbench datasets [DIR]
    plantbench compare A B [--tol T] [--no-trajectories]
    plantbench card data/regimes [--format text|md|latex|json]
    plantbench verify spec.toml pb-protocol:dab0d47475e9
    plantbench new-case my_tank [DIR]
    plantbench check my_tank [--contribute]
    plantbench contribute my_tank [--description TEXT] [--studies DIR] [--dry-run | --revert]

`python -m plantbench` is equivalent to `plantbench`.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import plantbench as pb
from plantbench.core import control as ctl
from plantbench.core.case import GENERIC_DISTURBANCES
from plantbench.core.spec import PROTOCOL_ID, SPEC_ID, identifier


def _list(args) -> None:
    ids = pb.list_cases()
    width = max(len(i) for i in ids)
    for case_id in ids:
        source = pb.cases.source(case_id)
        where = source["kind"] + (f" ({source['distribution']})" if "distribution" in source
                                  else "")
        print(f"{case_id:{width}s}  {pb.load_case(case_id).title}  [{where}]")


def _describe(args) -> None:
    case = pb.load_case(args.case)
    setup = pb.build(case)
    d = setup.design
    ev = setup.spectrum()
    loops = [e.name for e in setup.structure]
    print(f"{case.id}: {case.title}\n\n{case.summary}\n")
    print(f"states          {len(d.x)}")
    print(f"inputs          {', '.join(ctl.input_values(d.u))}")
    print("options         " + ", ".join(f"{k} = {v!r}" for k, v in case.options.items()))
    print("structures      " + ", ".join(
        f"{s}{' (reference)' if s == case.default_structure else ''}" for s in case.structures))
    print(f"disturbances    {', '.join([*case.disturbances, *GENERIC_DISTURBANCES])}")
    print(f"measurements    {', '.join(case.measurements(d))}")
    print(f"\nreference plant: {case.default_structure}; loops {', '.join(loops)}")
    print(f"closed-loop rightmost eigenvalue {ev.real.max():+.4f} 1/min, "
          f"least damping ratio {ctl.damping_ratio(ev):.3f}")
    cite = pb.cases.citation(case)
    if cite is not None:
        print(f"\nthis case was presented in a publication of its own; cite it beside "
              f"plantbench:\n\n{cite['text']}")


def _run(args) -> None:
    case = pb.load_case(args.case)
    config = case.config()
    if args.config:
        with open(args.config, "rb") as f:
            config = case.validate(tomllib.load(f))
    tr = pb.run(case, config, t_end=args.t_end, dt=args.dt, jacobian=args.jacobian,
                device=args.device)
    print(f"{case.id} {config.structure}: {tr.t.size} points over {tr.t[-1]:g} min, "
          f"{tr.wall_time:.2f} s, {tr.nfev} right-hand-side evaluations")
    for name, v in tr.y.items():
        print(f"  {name:16s} {v[0]:12.6g} -> {v[-1]:12.6g}")
    if args.out:
        tr.to_npz(args.out)
        print(f"written to {args.out}")


def _solver_arguments(p) -> None:
    p.add_argument("--jacobian", choices=("internal", "batched"), default="internal",
                   help="LSODA's own Jacobian (default), or one batched evaluation per "
                        "Jacobian, for a case that supports it")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="where a batched Jacobian is evaluated (default cpu); cuda needs "
                        "plantbench[gpu]")


def _generate(args) -> None:
    from plantbench.datagen import generate
    out = generate(args.spec, out_dir=args.out, workers=args.workers,
                   solver={"jacobian": args.jacobian, "device": args.device})
    print(f"dataset in {out}")


def _datasets(args) -> None:
    from plantbench import datasets
    found = datasets.find(args.dir)
    if not found:
        print(f"no dataset under {args.dir}")
        return
    rows = [datasets.summary(out) for out in found]
    protocols = any(s["protocol_digest"] for s in rows)
    cw = max(4, *(len(s["case"] or "?") for s in rows))
    head = f"{'directory':30s} {'case':{cw}s} {'specification':21s} "
    if protocols:
        head += f"{'protocol':25s}"
    print(head + f"{'generated at':14s} {'runs':24s} {'integration':>11s}  "
                 "specification on disk")
    for s in rows:
        runs = ", ".join(f"{n} {status}" for status, n in s["counts"].items() if n)
        spec = {True: "matches", False: "CHANGED", None: "not kept"}[s["spec_matches"]]
        spec_id = identifier(SPEC_ID, s["spec_digest"]) if s["spec_digest"] else "?"
        row = f"{str(s['dir']):30s} {s['case'] or '?':{cw}s} {spec_id:21s} "
        if protocols:
            protocol_id = (identifier(PROTOCOL_ID, s["protocol_digest"])
                           if s["protocol_digest"] else "-")
            row += f"{protocol_id:25s}"
        made = ("no manifest" if s["manifest"] is None
                else datasets.commit_text(s["git_commit"], s["git_dirty"]))
        print(row + f"{made:14s} "
                    f"{runs or 'none':24s} {_duration(s['wall_time']):>11s}  {spec}")
    for s in rows:
        if s["case_source"] is not None:
            print(f"{s['dir']}: {datasets.case_source_text(s['case_source'])}")


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.1f} s"


def _compare(args) -> None:
    from plantbench import datasets
    for line in datasets.compare(args.a, args.b, tol=args.tol,
                                 trajectories=not args.no_trajectories, limit=args.limit,
                                 log=lambda msg: print(msg, file=sys.stderr, flush=True)):
        print(line)


def _card(args) -> None:
    from plantbench import report
    c = report.card(args.dataset)
    print(report.render(c, args.format))
    if args.format == "json":  # the other formats carry their warnings in the card
        for w in c["warnings"]:
            print(f"warning: {w}", file=sys.stderr)


def _verify(args) -> int:
    from plantbench import report
    try:
        ok, lines = report.verify(args.target, args.identifiers)
    except (OSError, ValueError, KeyError) as exc:
        print(f"cannot verify: {exc}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if ok else 1


def _new_case(args) -> None:
    from plantbench.scaffold import new_case
    root = new_case(args.case, args.dir)
    print(f"case {args.case} written to {root}\n\n"
          f"  pip install -e {root}          # or uv pip install -e; plantbench then finds {args.case}\n"
          f"  pytest {root}                  # the contract, on the template's tank\n"
          f"  plantbench check {args.case}\n\n"
          f"then replace the tank in {root / args.case}/model.py with your plant, and once\n"
          f"`plantbench check {args.case} --contribute` passes, `plantbench contribute "
          f"{args.case}` adds it to a clone of the library.")


def _contribute(args) -> int:
    from plantbench import contribute
    try:
        if args.revert:
            problems = contribute.revert(args.case)
            print("\n".join(problems) if problems
                  else f"the contribution of {args.case} is undone")
            return 1 if problems else 0
        plan = contribute.plan(args.case, description=args.description, studies=args.studies)
    except contribute.ContributeError as exc:
        print(f"cannot contribute: {exc}", file=sys.stderr)
        return 2
    print(f"{'would add' if args.dry_run else 'adding'} {args.case} to {plan.root}:\n")
    print("\n".join(f"  {line}" for line in contribute.describe(plan)))
    for note in plan.notes:
        print(f"\nnote: {note}")
    if args.dry_run:
        return 0
    record = contribute.apply(plan)
    print(f"\nrecorded in {record.relative_to(plan.root)}; "
          f"`plantbench contribute {args.case} --revert` undoes it.\n\nleft to you:")
    print("\n".join(f"  - {item}" for item in plan.left))
    return 0


def _check(args) -> int:
    from plantbench import contract
    from plantbench.core.casekit import fixed_point_residual

    case = pb.load_case(args.case)
    results = contract.check(case)
    if args.contribute:
        results += contract.library_rules(case)
    for r in results:
        print(f"{'pass' if r.passed else 'FAIL'}  {r.check.replace('_', ' ')}"
              + (f"\n      {r.message}" if r.message else ""))
    try:
        design = pb.build(case).design
        worst, where = fixed_point_residual(design, case.state_names(design))
        print(f"\nopen-loop |rhs| at the design point: {worst:.3g}; largest at "
              + ", ".join(f"{name} ({value:+.3g})" for name, value in where[:3]))
    except Exception as exc:  # noqa: BLE001 -- reported; the contract has said why already
        print(f"\nthe design point could not be built: {type(exc).__name__}: {exc}")
    if args.contribute:
        print("\nalso required, and not checked here (docs/adding-a-case.md):")
        for item in contract.JUDGMENT:
            print(f"  - {item}")
    failed = sum(not r.passed for r in results)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    doc = __doc__.splitlines()
    parser = argparse.ArgumentParser(prog="plantbench", description=doc[0],
                                     epilog="examples:\n" + "\n".join(doc[1:]).strip("\n"),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"plantbench {pb.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the cases available").set_defaults(func=_list)
    p = sub.add_parser("describe", help="what a case provides")
    p.add_argument("case")
    p.set_defaults(func=_describe)
    p = sub.add_parser("run", help="simulate one configuration")
    p.add_argument("case")
    p.add_argument("--config", type=Path, help="TOML file holding a configuration")
    p.add_argument("--t-end", type=float, default=1500.0, help="minutes (default 1500)")
    p.add_argument("--dt", type=float, default=1.0, help="output spacing, minutes (default 1)")
    p.add_argument("--out", type=Path, help="write the trajectory to this .npz file")
    _solver_arguments(p)
    p.set_defaults(func=_run)
    p = sub.add_parser("generate", help="generate a dataset from a specification")
    p.add_argument("spec", type=Path)
    p.add_argument("--out", type=Path, help="directory (default data/<study name>)")
    p.add_argument("--workers", type=int, default=1)
    _solver_arguments(p)
    p.set_defaults(func=_generate)
    p = sub.add_parser("datasets", help="the datasets on disk and what they are of")
    p.add_argument("dir", nargs="?", default="data", help="a dataset, or a directory of them")
    p.set_defaults(func=_datasets)
    p = sub.add_parser("compare", help="two datasets, or two results files, and what moved")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tol", type=float, default=1e-9, help="relative (default 1e-9)")
    p.add_argument("--no-trajectories", action="store_true",
                   help="compare the records only, not the trajectories they point to")
    p.add_argument("--limit", type=int, default=20, help="lines per section (default 20)")
    p.set_defaults(func=_compare)
    p = sub.add_parser("card", help="a dataset as a paper reports it")
    p.add_argument("dataset", type=Path)
    p.add_argument("--format", choices=("text", "md", "latex", "json"), default="text")
    p.set_defaults(func=_card)
    p = sub.add_parser("verify", help="whether reported identifiers name a specification")
    p.add_argument("target", type=Path, help="a specification file, or a dataset")
    p.add_argument("identifiers", nargs="+", help="pb-spec:<digest>, pb-protocol:<digest>")
    p.set_defaults(func=_verify)
    p = sub.add_parser("new-case", help="a package for a new case, from the template")
    p.add_argument("case", help="the id of the new case")
    p.add_argument("dir", nargs="?", type=Path, help="where to write it (default ./<case>)")
    p.set_defaults(func=_new_case)
    p = sub.add_parser("check", help="the contract, on one case")
    p.add_argument("case")
    p.add_argument("--contribute", action="store_true",
                   help="also the rules for a case added to the library")
    p.set_defaults(func=_check)
    p = sub.add_parser("contribute", help="add a case in its own package to this clone")
    p.add_argument("case")
    p.add_argument("--description", help="the case's row in the case tables (default: its summary)")
    p.add_argument("--studies", type=Path, help="a directory of studies, to studies/<case>/")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show the edits, write nothing")
    mode.add_argument("--revert", action="store_true", help="undo a contribution from its record")
    p.set_defaults(func=_contribute)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
