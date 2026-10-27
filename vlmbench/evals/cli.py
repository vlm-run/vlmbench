"""`vlmbench eval | validate | board`: argparse wiring for the evals package (imports stay lazy, so `run` is fast)."""

import argparse
from pathlib import Path


def add_parsers(subparsers) -> None:
    ev = subparsers.add_parser("eval", help="Evaluate serves on a benchmark folder (accuracy + performance).")
    ev.add_argument("bench", type=Path, help="A benchmark folder, e.g. benchmarks/ocr-bench")
    ev.add_argument(
        "-s", "--serve", action="append", help="Only these serves (repeatable; the <bench>- prefix is optional)"
    )
    ev.add_argument("-d", "--dataset", action="append", help="Only these datasets (repeatable)")
    ev.add_argument("--split", choices=["test", "dev"], default="test", help="`dev` to tune prompts, `test` to report")
    ev.add_argument("--refresh", action="store_true", help="Ignore cached replies and call again")
    va = subparsers.add_parser("validate", help="Validate a benchmark folder and show each serve's resolved config.")
    va.add_argument("bench", type=Path)
    bo = subparsers.add_parser("board", help="Latest scores per serve on a benchmark, best first.")
    bo.add_argument("bench", type=Path)
    bo.add_argument("--split", choices=["test", "dev"], default="test")


def dispatch(args: argparse.Namespace, console) -> None:
    from rich.console import Console

    from vlmbench.evals import runner

    console = console if console.is_terminal else Console(width=200)
    log = lambda m: console.print(m, highlight=False)  # noqa: E731
    match args.command:
        case "eval":
            out = runner.run(args.bench, args.serve, args.dataset, args.split, args.refresh, log)
            console.print(f"[dim]{out}[/dim]")
            _board(runner.board(args.bench, args.split), console)
        case "validate":
            runner.validate(args.bench, log)
        case "board":
            _board(runner.board(args.bench, args.split), console)


def _board(rows: list[dict], console) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console if console.is_terminal else Console(width=200)

    if not rows:
        console.print("no results yet")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    t = Table(*cols, box=None, header_style="bold")
    fmt = (
        lambda v: "—"
        if v is None
        else f"{v:.2f}"
        if isinstance(v, float) and v < 10
        else f"{v:.1f}"
        if isinstance(v, float)
        else str(v)
    )  # noqa: E731
    for r in rows:
        t.add_row(*(fmt(r.get(c)) for c in cols))
    console.print(t)
