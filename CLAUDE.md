# vlmbench

Single-file, drop-in VLM benchmark CLI for your agents.

## Project Structure

```
vlmbench/cli.py             # The entire CLI — single-file, all logic here
pyproject.toml              # Package config, version, dependencies
.claude/skills/vlmbench/MODELS.md  # Tested models and their vLLM --serve-args
```

## Key Architecture

- **Single file**: All CLI logic lives in `vlmbench/cli.py`. This is intentional — the tool should remain a single file.
- **Entry point**: `main()` function at the bottom of cli.py. Uses `argparse` (stdlib).
- **Subcommands**: `run` (default), `compare`. The `run` subcommand is implicit if flags start with `--`.
- **Version**: Defined as `VERSION` in `vlmbench/cli.py`. `pyproject.toml` reads it dynamically via `setuptools.dynamic` attr.

## Backend System

- `auto`: Ollama on macOS, vLLM Docker on Linux
- `ollama`: Native Ollama
- `vllm`: Native vLLM (requires `pip install vllm`)
- `vllm-openai:<tag>`: Docker-based vLLM (`docker run --gpus all`)
- `sglang:<tag>`: Docker-based SGLang

Servers run in tmux sessions (`vlmbench-vllm`, `vlmbench-ollama`, `vlmbench-sglang`) with GPU monitoring in a split pane.

## Development

```bash
# Install dev dependencies
uv pip install -e '.[test]'

# Lint
make lint

# Test
make test
```

## Versioning

- **Every PR must bump the version** in `pyproject.toml`.
- **Patch bump by default**: Increment the patch version (e.g., `0.1.1` → `0.1.2`) for all PRs — bug fixes, new features, refactors, docs, etc.
- **Minor bump exception**: Only bump the minor version (e.g., `0.1.2` → `0.2.0`) when the PR explicitly states it is a minor release (e.g., breaking changes, large new capabilities). Reset patch to `0` on minor bumps.
- Do **not** bump the major version without explicit instruction.
- The version is defined **only** in `pyproject.toml` (`version = "X.Y.Z"`). Do not duplicate it elsewhere.

## Guidelines

- Keep cli.py as a single file. Do not split into modules.
- The PEP 723 script metadata at the top of cli.py enables `uv run cli.py` without install. Keep it in sync with pyproject.toml dependencies.
- Version is defined as `VERSION` in `vlmbench/cli.py`. `pyproject.toml` reads it dynamically. Update it in one place only.
- **Always run `make lint` before pushing commits.** Fix any errors before pushing.
- **Python 3.11+** — use modern features: `tomllib`, `StrEnum`, `ExceptionGroup`, `TaskGroup`, `type X = ...` aliases, `match/case`, `datetime.fromisoformat` improvements. Avoid legacy patterns and `from __future__ import annotations`.
- Code should be elegant and minimal — no unnecessary abstractions, no over-engineering.
- This is a developer tool. Prioritize clear output, fast iteration, and zero friction. Developers should be able to read the source and understand it immediately.
- Stay focused on VLMs and inference performance. Every feature should serve benchmarking, comparison, or reproduction of results. Do not add unrelated functionality.
- Terminal output matters. Use Rich tastefully — clean tables, readable panels, no visual clutter. Prefer data density over decoration.
