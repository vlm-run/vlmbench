# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "rich>=13",
# ]
# ///
"""vlmbench GPU sweep — launch benchmarks across GPU types on HF Jobs.

Dispatches bench_worker.py to multiple HF Jobs GPU instances,
waits for completion, downloads results, and prints a summary.

Usage:
    # Set your token
    export HF_TOKEN=hf_...

    # Launch on default GPUs (T4, L4, A10G) with Qwen3-VL-2B
    uv run jobs/launch_gpu_sweep.py

    # Custom model and GPUs
    uv run jobs/launch_gpu_sweep.py --model Qwen/Qwen3-VL-8B-Instruct --gpus a10g-small,a100-large

    # Launch and don't wait
    uv run jobs/launch_gpu_sweep.py --no-wait

    # After jobs finish, collate results
    uv run jobs/launch_gpu_sweep.py --collate-only --results-repo vlm-run/vlmbench-gpu-results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, fetch_job_logs, inspect_job, run_uv_job
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_GPUS = ["t4-small", "l4x1", "a10g-small"]
DEFAULT_RESULTS_REPO = "vlm-run/vlmbench-gpu-results"
DEFAULT_TIMEOUT = "45m"
WORKER_SCRIPT = Path(__file__).parent / "bench_worker.py"

# Extra vllm --with dependencies.  vllm brings PyTorch + CUDA runtime.
WORKER_DEPS = ["vllm", "vlmbench"]

# GPU flavor metadata for display
GPU_INFO = {
    "t4-small": {"gpu": "NVIDIA T4", "vram": "16 GB", "cost": "$0.40/hr"},
    "t4-medium": {"gpu": "NVIDIA T4", "vram": "16 GB", "cost": "$0.60/hr"},
    "l4x1": {"gpu": "NVIDIA L4", "vram": "24 GB", "cost": "$0.80/hr"},
    "l4x4": {"gpu": "4x NVIDIA L4", "vram": "96 GB", "cost": "$3.20/hr"},
    "a10g-small": {"gpu": "NVIDIA A10G", "vram": "24 GB", "cost": "$1.00/hr"},
    "a10g-large": {"gpu": "NVIDIA A10G", "vram": "24 GB", "cost": "$1.50/hr"},
    "a10g-largex2": {"gpu": "2x NVIDIA A10G", "vram": "48 GB", "cost": "$3.00/hr"},
    "a10g-largex4": {"gpu": "4x NVIDIA A10G", "vram": "96 GB", "cost": "$5.00/hr"},
    "a100-large": {"gpu": "NVIDIA A100", "vram": "80 GB", "cost": "$2.50/hr"},
}


# ── Job Dispatch ──────────────────────────────────────────────────────────────


def launch_jobs(
    model: str,
    gpu_flavors: list[str],
    results_repo: str,
    hf_token: str,
    serve_args: str,
    timeout: str,
) -> dict[str, str]:
    """Launch bench_worker.py on each GPU flavor. Returns {flavor: job_id}."""
    jobs: dict[str, str] = {}

    for flavor in gpu_flavors:
        info = GPU_INFO.get(flavor, {})
        gpu_label = info.get("gpu", flavor)
        cost = info.get("cost", "?")
        console.print(f"  Launching [bold cyan]{flavor}[/bold cyan] ({gpu_label}, {cost})...")

        try:
            job = run_uv_job(
                str(WORKER_SCRIPT),
                dependencies=WORKER_DEPS,
                flavor=flavor,
                timeout=timeout,
                secrets={"HF_TOKEN": hf_token},
                env={
                    "MODEL_ID": model,
                    "GPU_FLAVOR": flavor,
                    "HF_RESULTS_REPO": results_repo,
                    "SERVE_ARGS": serve_args,
                },
                token=hf_token,
            )
            jobs[flavor] = job.id
            console.print(f"    [green]Job {job.id}[/green]")
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")

    return jobs


# ── Job Monitoring ────────────────────────────────────────────────────────────


def poll_jobs(
    jobs: dict[str, str],
    hf_token: str,
    poll_interval: int = 30,
) -> dict[str, str]:
    """Block until all jobs reach a terminal state. Returns {flavor: status}."""
    terminal = {"completed", "error", "cancelled", "failed"}
    statuses = {flavor: "pending" for flavor in jobs}

    while True:
        for flavor, job_id in jobs.items():
            if statuses[flavor] in terminal:
                continue
            try:
                info = inspect_job(job_id=job_id, token=hf_token)
                stage = info.status.stage
                statuses[flavor] = stage.lower() if hasattr(stage, "lower") else str(stage).lower()
            except Exception:
                pass  # transient — retry next poll

        # Display
        parts = []
        for flavor, status in statuses.items():
            color = {
                "completed": "green",
                "error": "red",
                "failed": "red",
                "running": "cyan",
                "pending": "yellow",
                "building": "yellow",
            }.get(status, "white")
            parts.append(f"[{color}]{flavor}={status}[/{color}]")
        console.print(f"  {' | '.join(parts)}")

        if all(s in terminal for s in statuses.values()):
            break
        time.sleep(poll_interval)

    return statuses


def show_logs(job_id: str, hf_token: str, flavor: str) -> None:
    """Print the last logs from a finished job."""
    console.print(f"\n  [bold]--- Logs: {flavor} ({job_id}) ---[/bold]")
    try:
        for entry in fetch_job_logs(job_id=job_id, token=hf_token):
            print(f"    {entry}")
    except Exception as e:
        console.print(f"  [yellow]Could not fetch logs: {e}[/yellow]")


# ── Result Collation ──────────────────────────────────────────────────────────


def download_results(results_repo: str, hf_token: str, output_dir: Path) -> list[Path]:
    """Download all result JSONs from the HF dataset repo."""
    api = HfApi(token=hf_token)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_files = api.list_repo_files(results_repo, repo_type="dataset")
    except Exception as e:
        console.print(f"  [red]Cannot list {results_repo}: {e}[/red]")
        return []

    json_paths = [f for f in all_files if f.startswith("results/") and f.endswith(".json")]
    if not json_paths:
        console.print("  [yellow]No result files in repo.[/yellow]")
        return []

    downloaded: list[Path] = []
    for remote in json_paths:
        try:
            local = api.hf_hub_download(
                repo_id=results_repo,
                filename=remote,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            downloaded.append(Path(local))
        except Exception as e:
            console.print(f"  [yellow]Failed: {remote}: {e}[/yellow]")

    return downloaded


def print_summary_table(files: list[Path]) -> None:
    """Parse downloaded JSONs and print a Rich comparison table."""
    table = Table(
        title="GPU Sweep Results",
        show_header=True,
        header_style="bold white",
        box=box.ROUNDED,
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("GPU Flavor", style="bold cyan", min_width=12)
    table.add_column("GPU Name", min_width=16)
    table.add_column("Tok/s", justify="right", min_width=7)
    table.add_column("TTFT (ms)", justify="right", min_width=9)
    table.add_column("TPOT (ms)", justify="right", min_width=9)
    table.add_column("Latency (s)", justify="right", min_width=10)
    table.add_column("VRAM (MiB)", justify="right", min_width=9)
    table.add_column("Errors", justify="right", min_width=6)

    rows: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            r = data["results"]
            rows.append(
                {
                    "flavor": data.get("hf_job", {}).get("gpu_flavor", "?"),
                    "gpu": data.get("environment", {}).get("gpu_name", "-"),
                    "tok_s": r["tokens_per_sec"],
                    "ttft": r["ttft_ms"]["mean"],
                    "tpot": r["tpot_ms"]["mean"],
                    "latency": r["latency_s_per_input"]["mean"],
                    "vram": r.get("vram_peak_mib", "-"),
                    "errors": r.get("errors", 0),
                }
            )
        except Exception as e:
            console.print(f"  [yellow]Parse error in {f.name}: {e}[/yellow]")

    # Sort by tok/s descending
    rows.sort(key=lambda x: x["tok_s"], reverse=True)
    best_toks = rows[0]["tok_s"] if rows else 0

    for row in rows:
        tok_style = "bold bright_green" if row["tok_s"] == best_toks else "white"
        table.add_row(
            row["flavor"],
            str(row["gpu"]),
            f"[{tok_style}]{row['tok_s']:.1f}[/{tok_style}]",
            f"{row['ttft']:.0f}",
            f"{row['tpot']:.1f}",
            f"{row['latency']:.2f}",
            str(row["vram"]),
            str(row["errors"]),
        )

    console.print()
    console.print(table)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch vlmbench GPU sweep on HF Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--gpus", default=",".join(DEFAULT_GPUS), help="Comma-separated GPU flavors")
    p.add_argument("--results-repo", default=DEFAULT_RESULTS_REPO, help="HF dataset repo for results")
    p.add_argument("--serve-args", default="", help="Extra vLLM serve args")
    p.add_argument("--timeout", default=DEFAULT_TIMEOUT, help=f"Job timeout (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--output", default="./results/gpu-sweep", help="Local output directory")
    p.add_argument("--no-wait", action="store_true", help="Launch jobs and exit immediately")
    p.add_argument("--collate-only", action="store_true", help="Skip launching; just download and display results")
    p.add_argument("--logs", action="store_true", help="Print logs for all jobs (not just failed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        console.print("[red]Set HF_TOKEN environment variable first.[/red]")
        sys.exit(1)

    gpu_flavors = [g.strip() for g in args.gpus.split(",")]
    output_dir = Path(args.output)

    # ── Banner ────────────────────────────────────────────────────────────
    console.print()
    panel_lines = []
    panel_lines.append(f"[bold]Model[/bold]     {args.model}")
    panel_lines.append(f"[bold]GPUs[/bold]      {', '.join(gpu_flavors)}")
    panel_lines.append(f"[bold]Repo[/bold]      {args.results_repo}")
    panel_lines.append(f"[bold]Timeout[/bold]   {args.timeout}")
    console.print(
        Panel(
            "\n".join(panel_lines),
            title="[bold]vlmbench GPU sweep[/bold]",
            title_align="left",
            border_style="bright_magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    # ── Collate-only mode ─────────────────────────────────────────────────
    if args.collate_only:
        console.print("\n  [cyan]Downloading results...[/cyan]")
        downloaded = download_results(args.results_repo, hf_token, output_dir)
        if downloaded:
            console.print(f"  [green]{len(downloaded)} file(s) downloaded to {output_dir}[/green]")
            print_summary_table(downloaded)
            file_list = " ".join(str(f) for f in downloaded)
            console.print(f"\n  [dim]Compare with:[/dim]  vlmbench compare {file_list}")
        return

    # ── Launch ────────────────────────────────────────────────────────────
    console.print()
    jobs = launch_jobs(
        model=args.model,
        gpu_flavors=gpu_flavors,
        results_repo=args.results_repo,
        hf_token=hf_token,
        serve_args=args.serve_args,
        timeout=args.timeout,
    )

    if not jobs:
        console.print("[red]No jobs launched.[/red]")
        sys.exit(1)

    console.print(f"\n  [green bold]{len(jobs)} job(s) launched.[/green bold]")

    if args.no_wait:
        console.print("\n  Job IDs:")
        for flavor, job_id in jobs.items():
            console.print(f"    {flavor}: {job_id}")
        console.print("\n  [dim]Monitor at https://huggingface.co/settings/jobs[/dim]")
        console.print(
            f"  [dim]Collate later: uv run jobs/launch_gpu_sweep.py --collate-only --results-repo {args.results_repo}[/dim]"
        )
        return

    # ── Wait ──────────────────────────────────────────────────────────────
    console.print("\n  [dim]Polling jobs every 30s...[/dim]\n")
    statuses = poll_jobs(jobs, hf_token)

    # Show logs for failed jobs (or all if --logs)
    for flavor, job_id in jobs.items():
        if args.logs or statuses[flavor] in ("error", "failed"):
            show_logs(job_id, hf_token, flavor)

    # ── Collate ───────────────────────────────────────────────────────────
    succeeded = sum(1 for s in statuses.values() if s == "completed")
    failed = len(jobs) - succeeded

    console.print(f"\n  [bold]{succeeded} succeeded, {failed} failed[/bold]")

    if succeeded > 0:
        console.print("\n  [cyan]Downloading results...[/cyan]")
        downloaded = download_results(args.results_repo, hf_token, output_dir)
        if downloaded:
            console.print(f"  [green]{len(downloaded)} file(s) downloaded to {output_dir}[/green]")
            print_summary_table(downloaded)
            file_list = " ".join(str(f) for f in downloaded)
            console.print(f"\n  [dim]Compare with:[/dim]  vlmbench compare {file_list}")


if __name__ == "__main__":
    main()
