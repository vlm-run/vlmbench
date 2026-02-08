# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "rich>=13",
# ]
# ///
"""vlmbench GPU sweep — launch benchmarks across GPU types on HF Jobs.

Dispatches bench_worker.py to multiple HF Jobs GPU instances.
Each worker runs vlmbench, then pushes its result JSON to the
HF dataset repo.  Use a separate collation script to download
and compare results.

Usage:
    export HF_TOKEN=hf_...

    # Launch on default GPUs (T4, L4, A10G) with Qwen3-VL-2B
    uv run jobs/launch_gpu_sweep.py

    # Custom model and GPUs
    uv run jobs/launch_gpu_sweep.py --model Qwen/Qwen3-VL-8B-Instruct --gpus a10g-small,a100-large
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import run_uv_job
from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_GPUS = ["t4-small", "l4x1", "a10g-small"]
DEFAULT_RESULTS_REPO = "vlm-run/vlmbench-gpu-results"
DEFAULT_TIMEOUT = "45m"
WORKER_SCRIPT = Path(__file__).parent / "bench_worker.py"
WORKER_DEPS = ["vllm", "vlmbench"]

GPU_INFO = {
    "t4-small": "NVIDIA T4 16 GB ($0.40/hr)",
    "t4-medium": "NVIDIA T4 16 GB ($0.60/hr)",
    "l4x1": "NVIDIA L4 24 GB ($0.80/hr)",
    "l4x4": "4x NVIDIA L4 96 GB ($3.20/hr)",
    "a10g-small": "NVIDIA A10G 24 GB ($1.00/hr)",
    "a10g-large": "NVIDIA A10G 24 GB ($1.50/hr)",
    "a10g-largex2": "2x NVIDIA A10G 48 GB ($3.00/hr)",
    "a10g-largex4": "4x NVIDIA A10G 96 GB ($5.00/hr)",
    "a100-large": "NVIDIA A100 80 GB ($2.50/hr)",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Launch vlmbench GPU sweep on HF Jobs")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--gpus", default=",".join(DEFAULT_GPUS), help="Comma-separated GPU flavors")
    p.add_argument("--results-repo", default=DEFAULT_RESULTS_REPO, help="HF dataset repo for results")
    p.add_argument("--serve-args", default="", help="Extra vLLM serve args")
    p.add_argument("--timeout", default=DEFAULT_TIMEOUT, help=f"Job timeout (default: {DEFAULT_TIMEOUT})")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        console.print("[red]Set HF_TOKEN environment variable first.[/red]")
        sys.exit(1)

    gpu_flavors = [g.strip() for g in args.gpus.split(",")]

    # Banner
    console.print()
    console.print(
        Panel(
            f"[bold]Model[/bold]     {args.model}\n"
            f"[bold]GPUs[/bold]      {', '.join(gpu_flavors)}\n"
            f"[bold]Repo[/bold]      {args.results_repo}\n"
            f"[bold]Timeout[/bold]   {args.timeout}",
            title="[bold]vlmbench GPU sweep[/bold]",
            title_align="left",
            border_style="bright_magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()

    # Launch a job per GPU flavor
    jobs: dict[str, str] = {}
    for flavor in gpu_flavors:
        label = GPU_INFO.get(flavor, flavor)
        console.print(f"  Launching [bold cyan]{flavor}[/bold cyan] ({label})...")
        try:
            job = run_uv_job(
                str(WORKER_SCRIPT),
                script_args=[
                    "--model",
                    args.model,
                    "--gpu-flavor",
                    flavor,
                    "--hf-repo",
                    args.results_repo,
                    *(["--serve-args", args.serve_args] if args.serve_args else []),
                ],
                dependencies=WORKER_DEPS,
                flavor=flavor,
                timeout=args.timeout,
                secrets={"HF_TOKEN": hf_token},
                token=hf_token,
            )
            jobs[flavor] = job.id
            console.print(f"    [green]Job {job.id}[/green]")
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")

    if not jobs:
        console.print("[red]No jobs launched.[/red]")
        sys.exit(1)

    # Summary
    console.print(f"\n  [green bold]{len(jobs)} job(s) launched.[/green bold]\n")
    for flavor, job_id in jobs.items():
        console.print(f"    {flavor}: {job_id}")
    console.print("\n  [dim]Monitor: https://huggingface.co/settings/jobs[/dim]")
    console.print(f"  [dim]Results: https://huggingface.co/datasets/{args.results_repo}[/dim]")


if __name__ == "__main__":
    main()
