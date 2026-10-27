"""`vlmbench lint`: every config and benchmark folder loads, and follows the layout. Imports only the config
models, so the pre-commit hook needs nothing but pydantic and pyyaml.

Checks: each `configs/*.yaml` loads (a `_` stem is a fragment, checked through what extends it); each
`benchmarks/<b>/` has a `<b>.yaml` Eval whose serves exist as `<b>-<serve>.yaml`, load, and have distinct config
ids; every `<b>-*.yaml` in the folder is listed; prompt files exist; judges and their serves load.
"""

import sys
from pathlib import Path

from vlmbench.evals.config import Eval, Serve, bench_files, load


def _prompt_problem(s: Serve) -> str | None:
    p = s.request.prompt
    if p and "\n" not in p and p.endswith((".md", ".txt")) and not s.prompt_file:
        return f"prompt file {p!r} not found (it would be sent as the prompt text)"
    return None


def lint(paths: list[Path], verbose: bool = False, log=print) -> int:
    """The number of problems found; each is logged as `path: problem`."""
    problems: list[str] = []
    for path in paths:
        if path.is_file():
            files, benches = [path] if path.suffix == ".yaml" and not path.stem.startswith("_") else [], []
        elif (path / f"{path.name}.yaml").exists():
            files, benches = [], [path]
        else:
            files = [f for f in sorted(path.glob("*.yaml")) if not f.stem.startswith("_")]
            benches = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / f"{d.name}.yaml").exists()]
        for f in files:
            try:
                doc = load(f)
                if isinstance(doc, Serve) and (msg := _prompt_problem(doc)):
                    problems.append(f"{f}: {msg}")
            except Exception as e:
                problems.append(f"{f}: {e}".replace("\n", " "))
        for b in benches:
            problems += _bench(b, verbose, log)
    for p in problems:
        log(f"[red]✗[/red] {p}")
    if not problems:
        log(f"[green]✓[/green] {', '.join(str(p) for p in paths)}")
    return len(problems)


def _bench(b: Path, verbose: bool, log) -> list[str]:
    try:
        ev, serves, metrics = bench_files(b)
    except Exception as e:
        return [f"{b}: {e}".replace("\n", " ")]
    out = []
    assert isinstance(ev, Eval)
    listed = set(serves)
    for f in sorted(b.glob(f"{b.name}-*.yaml")):
        if f.stem not in listed:
            out.append(f"{f}: not listed in {b.name}.yaml `serves`")
            try:
                if msg := _prompt_problem(load(f)):
                    out.append(f"{f}: {msg}")
            except Exception as e:
                out.append(f"{f}: {e}".replace("\n", " "))
    for f in sorted(b.glob("*.yaml")):
        if f.stem != b.name and not f.stem.startswith(f"{b.name}-"):
            out.append(f"{f}: a serve in {b.name}/ is named {b.name}-<model>.yaml")
    seen: dict[str, str] = {}
    for n, s in serves.items():
        if msg := _prompt_problem(s):
            out.append(f"{b / n}.yaml: {msg}")
        if s.config_id in seen:
            out.append(f"{b / n}.yaml: same config as {seen[s.config_id]} ({s.config_id})")
        seen[s.config_id] = n
        if verbose:
            how = Path(s.prompt_file).name if s.prompt_file else f"method={s.request.method}"
            log(f"  {n.removeprefix(b.name + '-'):32s} {s.config_id}  {s.backend.type:6s} {s.model:40s} {how}")
    for m in metrics:
        try:
            judge = load(m.judge)
            if verbose:
                log(f"  judge {m.name}: {judge.model}, scale {m.scale}")
        except Exception as e:
            out.append(f"{m.judge}: {e}".replace("\n", " "))
    return out


def main() -> None:
    """Pre-commit entry point: lint the given paths (default: configs/ and benchmarks/)."""
    paths = [Path(a) for a in sys.argv[1:]] or [Path("configs"), Path("benchmarks")]
    sys.exit(
        1
        if lint(
            paths,
            log=lambda m: print(
                m.replace("[red]", "").replace("[/red]", "").replace("[green]", "").replace("[/green]", "")
            ),
        )
        else 0
    )


if __name__ == "__main__":
    main()
