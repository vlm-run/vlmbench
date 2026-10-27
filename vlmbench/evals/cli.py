"""`vlmbench eval | lint | board | dashboard`: argparse wiring for the evals package (imports stay lazy, so `run` is fast)."""

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
    ev.add_argument("--samples", type=int, default=None, help="Items per dataset for this run (0: every item)")
    ev.add_argument("--no-judges", action="store_true", help="Skip the eval's LLM judges (they cost a call per reply)")
    ev.add_argument("--budget", type=float, default=None, help="Most this run may spend in USD (overrides budget_usd)")
    li = subparsers.add_parser("lint", help="Check configs and benchmark folders load and follow the layout.")
    li.add_argument("paths", type=Path, nargs="*", help="Files or folders (default: configs/ benchmarks/)")
    li.add_argument("-v", "--verbose", action="store_true", help="Show each serve's config id, model and prompt")
    da = subparsers.add_parser("dashboard", help="Write (and serve) an HTML page of every eval and run collected.")
    da.add_argument(
        "--out", type=Path, default=None, help="Where to write it (default: ~/.vlmbench/dashboard/index.html)"
    )
    da.add_argument("--host", default="127.0.0.1", help="Bind address (a Tailscale IP to share it)")
    da.add_argument("--port", type=int, default=8795)
    da.add_argument("--no-serve", action="store_true", help="Only write the page")
    bo = subparsers.add_parser("board", help="Latest scores per serve on a benchmark, best first.")
    bo.add_argument("bench", type=Path)
    bo.add_argument("--split", choices=["test", "dev"], default="test")


def dispatch(args: argparse.Namespace, console) -> None:
    from rich.console import Console

    try:
        from vlmbench.evals import runner
    except ImportError as e:
        raise SystemExit(f"{e.name} is missing: evals need the extra, `pip install 'vlmbench[evals]'`") from None

    console = console if console.is_terminal else Console(width=200)
    log = lambda m: console.print(m, highlight=False)  # noqa: E731
    match args.command:
        case "eval":
            out = runner.run(
                args.bench, args.serve, args.dataset, args.split, args.refresh, log, args.samples, args.budget
            )
            console.print(f"[dim]{out}[/dim]")
            _board(runner.board(args.bench, args.split), console)
        case "lint":
            from vlmbench.evals.lint import lint

            if lint(args.paths or [Path("configs"), Path("benchmarks")], args.verbose, log):
                raise SystemExit(1)
        case "dashboard":
            from vlmbench.evals import dashboard

            out = dashboard.build(args.out)
            console.print(f"[dim]{out}[/dim]")
            if not args.no_serve:
                console.print(f"http://{args.host}:{args.port}/  (Ctrl-C to stop)")
                dashboard.serve(out, args.host, args.port)
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
    for r in rows:
        t.add_row(*(_fmt(r.get(c)) for c in cols))
    console.print(t)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}" if v < 10 else f"{v:.1f}"
    return str(v)
