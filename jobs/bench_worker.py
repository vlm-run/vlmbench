# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "Pillow>=10",
# ]
# ///
"""vlmbench GPU benchmark worker — runs on HF Jobs.

Runs vlmbench with --serve --backend vllm (vlmbench starts vLLM
implicitly), then optionally uploads the result JSON to HF Hub.

If --hf-repo and --hf-token are provided, the result JSON is uploaded
to the dataset repo under results/<gpu-flavor>/<filename>.json.
Otherwise results are saved locally only.

Usage:
    # Local run (no upload)
    uv run --with vllm,vlmbench jobs/bench_worker.py

    # Upload results to HF Hub
    uv run --with vllm,vlmbench jobs/bench_worker.py \
        --hf-repo vlm-run/vlmbench-gpu-results \
        --hf-token hf_...

    # Custom model
    uv run --with vllm,vlmbench jobs/bench_worker.py \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --hf-repo vlm-run/vlmbench-gpu-results \
        --hf-token hf_...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCHMARK_TIMEOUT = 1200  # 20 min for model download + load + benchmark


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def ensure_tmux() -> None:
    """Install tmux if not available (vlmbench --serve needs it)."""
    if shutil.which("tmux"):
        return
    log("Installing tmux...")
    subprocess.run(["apt-get", "update", "-qq"], check=False, capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "tmux"], check=True, capture_output=True)
    log("tmux installed.")


def create_test_image(output_dir: Path) -> Path:
    """Create a 1024x1024 synthetic document image for benchmarking."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw text-like gray bars to simulate a document
    y = 40
    for i in range(20):
        width = 200 + (i * 37) % 600
        gray = 40 + (i * 13) % 80
        draw.rectangle([60, y, 60 + width, y + 12], fill=(gray, gray, gray))
        y += 28

    path = output_dir / "synthetic_document.png"
    img.save(path)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vlmbench GPU benchmark worker")
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct"),
        help="HF model ID (default: $MODEL_ID or Qwen/Qwen3-VL-2B-Instruct)",
    )
    p.add_argument(
        "--gpu-flavor",
        default=os.environ.get("GPU_FLAVOR", "unknown"),
        help="GPU flavor label for tagging results (default: $GPU_FLAVOR)",
    )
    p.add_argument(
        "--serve-args",
        default=os.environ.get("SERVE_ARGS", ""),
        help="Extra vLLM serve args (default: $SERVE_ARGS)",
    )
    p.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_RESULTS_REPO", ""),
        help="HF dataset repo to upload results (e.g. vlm-run/vlmbench-gpu-results). "
        "If provided with --hf-token, results are uploaded.",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token for uploading results (default: $HF_TOKEN)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log(f"Model:       {args.model}")
    log(f"GPU flavor:  {args.gpu_flavor}")
    if args.hf_repo and args.hf_token:
        log(f"Upload to:   {args.hf_repo}")
    else:
        log("Upload:      disabled (pass --hf-repo and --hf-token to enable)")

    # GPU info
    subprocess.run(["nvidia-smi"], check=False)

    # vlmbench --serve --backend vllm needs tmux
    ensure_tmux()

    # Create test input
    input_dir = Path("/tmp/vlmbench-input")
    input_dir.mkdir(parents=True, exist_ok=True)
    create_test_image(input_dir)

    # Run vlmbench — it starts vLLM implicitly via --serve --backend vllm
    results_dir = Path("/tmp/vlmbench-results")
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_cmd = [
        "vlmbench",
        "run",
        "--model",
        args.model,
        "--input",
        str(input_dir),
        "--backend",
        "vllm",
        "--serve",
        "--runs",
        "3",
        "--warmup",
        "1",
        "--max-tokens",
        "2048",
        "--save",
        str(results_dir),
        "--tag",
        args.gpu_flavor,
    ]
    if args.serve_args:
        bench_cmd.extend(["--serve-args", args.serve_args])

    log(f"Running: {' '.join(bench_cmd)}")
    result = subprocess.run(bench_cmd, timeout=BENCHMARK_TIMEOUT)

    if result.returncode != 0:
        log(f"ERROR: vlmbench exited with code {result.returncode}")
        sys.exit(1)

    # Collect results
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        log("ERROR: No result JSON files found.")
        sys.exit(1)

    log(f"Found {len(json_files)} result file(s).")

    # Enrich with HF Job metadata
    for f in json_files:
        data = json.loads(f.read_text())
        data["hf_job"] = {"gpu_flavor": args.gpu_flavor, "source": "hf_jobs"}
        f.write_text(json.dumps(data, indent=2))

    # Upload to HF Hub if --hf-repo and --hf-token are both provided
    if args.hf_repo and args.hf_token:
        from huggingface_hub import HfApi

        api = HfApi(token=args.hf_token)

        try:
            api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
        except Exception as e:
            log(f"Warning: create_repo: {e}")

        for f in json_files:
            remote_path = f"results/{args.gpu_flavor}/{f.name}"
            log(f"Uploading {f.name} -> {args.hf_repo}/{remote_path}")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=remote_path,
                repo_id=args.hf_repo,
                repo_type="dataset",
            )
        log("Upload complete.")
    else:
        log("Results saved locally:")
        for f in json_files:
            log(f"  {f}")

    log("Done.")


if __name__ == "__main__":
    main()
