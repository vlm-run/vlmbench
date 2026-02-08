# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub",
#   "python-dotenv",
# ]
# ///
"""Sweep vlmbench across GPU flavors on HuggingFace Jobs."""

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vlmbench across GPU flavors on HF Jobs")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Model ID")
    parser.add_argument("--flavors", default="t4-small,l4x1,a10g-small", help="Comma-separated GPU flavors")
    parser.add_argument("--repo-id", default="vlm-run/vlmbench-results", help="HF dataset repo for results")
    parser.add_argument("--timeout", default="45m", help="Job timeout")
    parser.add_argument("--serve-args", default="", help="Extra vLLM serve args to forward")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set. Create a .env file or export HF_TOKEN.")
        sys.exit(1)

    from huggingface_hub import fetch_job_logs, inspect_job, run_uv_job

    script_path = str(Path(__file__).parent / "run_benchmark.py")
    flavors = [f.strip() for f in args.flavors.split(",")]

    # Launch jobs
    jobs: dict[str, object] = {}
    for flavor in flavors:
        print(f"Launching job on {flavor}...")
        script_args = ["--model", args.model, "--gpu-flavor", flavor, "--repo-id", args.repo_id]
        if args.serve_args:
            script_args.extend(["--serve-args", args.serve_args])

        job = run_uv_job(
            script_path,
            script_args=script_args,
            image="vllm/vllm-openai:latest",
            secrets={"HF_TOKEN": token},
            flavor=flavor,
            timeout=args.timeout,
        )
        jobs[flavor] = job
        print(f"  -> {job.id}: {job.url}")

    # Monitor jobs
    print(f"\nMonitoring {len(jobs)} jobs...")
    pending = dict(jobs)
    while pending:
        time.sleep(30)
        for flavor, job in list(pending.items()):
            info = inspect_job(job_id=job.id)
            stage = info.status.stage
            if stage in ("COMPLETED", "ERROR", "CANCELED"):
                icon = "OK" if stage == "COMPLETED" else "FAIL"
                print(f"  [{icon}] {flavor}: {stage}")
                if stage != "COMPLETED":
                    for line in fetch_job_logs(job_id=job.id):
                        print(f"    | {line}")
                del pending[flavor]

    # Summary
    print("\n--- Summary ---")
    for flavor, job in jobs.items():
        info = inspect_job(job_id=job.id)
        print(f"  {flavor}: {info.status.stage} -> {job.url}")


if __name__ == "__main__":
    main()
