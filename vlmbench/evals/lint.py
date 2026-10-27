"""`vlmbench lint`: every config under `configs/` loads and resolves. Imports only the config models, so the
pre-commit hook needs nothing but pydantic and pyyaml.

Checks: each `providers/*.yaml` is a provider; each `models/*.yaml` resolves to a serve; each
`benchmarks/<b>/<b>.yaml` resolves every serve and judge (names exist, prompt files exist) with distinct config ids.
Any path under a `configs/` folder lints that whole folder: a model edit can break a benchmark that uses it.
"""

import sys
from pathlib import Path

from pydantic import TypeAdapter

from vlmbench.evals.config import Remote, Serve, Vllm, bench_files, load, root_of

_PROVIDER = TypeAdapter(Vllm | Remote)


def _prompt_problem(s: Serve) -> str | None:
    p = s.request.prompt
    if p and "\n" not in p and p.endswith((".md", ".txt")) and not s.prompt_file:
        return f"prompt file {p!r} not found (it would be sent as the prompt text)"
    return None


def _try(problems: list[str], where: Path, fn):
    try:
        return fn()
    except (Exception, SystemExit) as e:
        problems.append(f"{where}: {e}".replace("\n", " "))
        return None


def lint(paths: list[Path], verbose: bool = False, log=print) -> int:
    """The number of problems found; each is logged as `path: problem`."""
    import yaml

    problems: list[str] = []
    for root in sorted({root_of(p) for p in paths}):
        for f in sorted((root / "providers").glob("*.yaml")):
            _try(problems, f, lambda f=f: _PROVIDER.validate_python(yaml.safe_load(f.read_text())))
        for f in sorted((root / "models").glob("*.yaml")):
            if (s := _try(problems, f, lambda f=f: load(f))) and (msg := _prompt_problem(s)):
                problems.append(f"{f}: {msg}")
        for b in sorted(d for d in (root / "benchmarks").glob("*") if (d / f"{d.name}.yaml").is_file()):
            problems += _bench(b, verbose, log)
    for p in problems:
        log(f"[red]✗[/red] {p}")
    if not problems:
        log(f"[green]✓[/green] {', '.join(str(p) for p in paths)}")
    return len(problems)


def _bench(b: Path, verbose: bool, log) -> list[str]:
    out: list[str] = []
    got = _try(out, b / f"{b.name}.yaml", lambda: bench_files(b))
    if not got:
        return out
    _, serves, metrics = got
    seen: dict[str, str] = {}
    for n, s in serves.items():
        key = n.removeprefix(b.name + "-")
        if msg := _prompt_problem(s):
            out.append(f"{b.name}.yaml serves.{key}: {msg}")
        if s.config_id in seen:
            out.append(f"{b.name}.yaml serves.{key}: same config as {seen[s.config_id]} ({s.config_id})")
        seen[s.config_id] = key
        if verbose:
            how = Path(s.prompt_file).name if s.prompt_file else f"method={s.request.method}"
            log(f"  {key:32s} {s.config_id}  {s.provider_name:8s} {s.model:40s} {how}")
    for m in metrics:
        if verbose:
            log(f"  judge {m.name}: {m.judge.model}, scale {m.scale}")
    return out


def main() -> None:
    """Pre-commit entry point: lint the given paths (default: configs/)."""
    paths = [Path(a) for a in sys.argv[1:]] or [Path("configs")]
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
