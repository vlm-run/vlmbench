# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.34",
#   "Pillow>=10",
# ]
# ///
"""vlmbench GPU benchmark worker — runs on HF Jobs.

Starts a native vLLM server, runs vlmbench against it,
and uploads the result JSON to a HF Hub dataset repo.

Expected env vars (set by the orchestrator):
    MODEL_ID        — HF model ID (e.g. Qwen/Qwen3-VL-2B-Instruct)
    GPU_FLAVOR      — HF Jobs hardware flavor (e.g. t4-small, a10g-small)
    HF_RESULTS_REPO — Dataset repo to upload results (e.g. vlm-run/vlmbench-gpu-results)
    SERVE_ARGS      — Extra vLLM serve CLI args (optional)
    HF_TOKEN        — HF token (passed as a secret)

Run locally for testing:
    HF_TOKEN=... MODEL_ID=Qwen/Qwen3-VL-2B-Instruct GPU_FLAVOR=local \
        uv run --with vllm,vlmbench jobs/bench_worker.py
"""

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://localhost:8000/v1"
VLLM_STARTUP_TIMEOUT = 600  # 10 minutes for model download + load
BENCHMARK_TIMEOUT = 600


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def wait_for_server(timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """Poll vLLM /v1/models until it responds 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{BASE_URL}/models", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def create_test_image(output_dir: Path) -> Path:
    """Create a 1024x1024 synthetic document image for benchmarking."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw some text-like content (lines of varying gray)
    y = 40
    for i in range(20):
        width = 200 + (i * 37) % 600
        gray = 40 + (i * 13) % 80
        draw.rectangle([60, y, 60 + width, y + 12], fill=(gray, gray, gray))
        y += 28

    path = output_dir / "synthetic_document.png"
    img.save(path)
    return path


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
    gpu_flavor = os.environ.get("GPU_FLAVOR", "unknown")
    results_repo = os.environ.get("HF_RESULTS_REPO", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    serve_args = os.environ.get("SERVE_ARGS", "")

    log(f"Model:       {model_id}")
    log(f"GPU flavor:  {gpu_flavor}")
    log(f"Results repo: {results_repo or '(none — local only)'}")

    # Print GPU info
    log("--- GPU Info ---")
    subprocess.run(["nvidia-smi"], check=False)
    log("----------------")

    # Create test input
    input_dir = Path("/tmp/vlmbench-input")
    input_dir.mkdir(parents=True, exist_ok=True)
    img_path = create_test_image(input_dir)
    log(f"Test image: {img_path}")

    # Start vLLM serve in background
    vllm_cmd = ["vllm", "serve", model_id]
    if serve_args:
        vllm_cmd.extend(shlex.split(serve_args))
    log(f"Starting vLLM: {' '.join(vllm_cmd)}")

    vllm_proc = subprocess.Popen(
        vllm_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Wait for server
    log(f"Waiting for vLLM (timeout {VLLM_STARTUP_TIMEOUT}s)...")
    if not wait_for_server():
        log("ERROR: vLLM failed to start.")
        vllm_proc.terminate()
        sys.exit(1)
    log("vLLM server ready.")

    # Run vlmbench
    results_dir = Path("/tmp/vlmbench-results")
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_cmd = [
        "vlmbench",
        "run",
        "--model",
        model_id,
        "--input",
        str(input_dir),
        "--base-url",
        BASE_URL,
        "--runs",
        "3",
        "--warmup",
        "1",
        "--max-tokens",
        "2048",
        "--save",
        str(results_dir),
        "--tag",
        gpu_flavor,
        "--no-serve",
    ]
    log(f"Running: {' '.join(bench_cmd)}")
    result = subprocess.run(bench_cmd, timeout=BENCHMARK_TIMEOUT)

    # Stop vLLM
    log("Stopping vLLM...")
    vllm_proc.terminate()
    try:
        vllm_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        vllm_proc.kill()

    if result.returncode != 0:
        log(f"ERROR: vlmbench exited with code {result.returncode}")
        sys.exit(1)

    # Collect result files
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        log("ERROR: No result JSON files found.")
        sys.exit(1)

    log(f"Found {len(json_files)} result file(s).")

    # Enrich results with HF Job metadata
    for f in json_files:
        data = json.loads(f.read_text())
        data["hf_job"] = {
            "gpu_flavor": gpu_flavor,
            "source": "hf_jobs",
        }
        f.write_text(json.dumps(data, indent=2))

    # Upload to HF Hub
    if results_repo and hf_token:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)

        try:
            api.create_repo(results_repo, repo_type="dataset", exist_ok=True)
        except Exception as e:
            log(f"Warning: create_repo: {e}")

        for f in json_files:
            remote_path = f"results/{gpu_flavor}/{f.name}"
            log(f"Uploading {f.name} -> {results_repo}/{remote_path}")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=remote_path,
                repo_id=results_repo,
                repo_type="dataset",
            )
        log("Upload complete.")
    else:
        log("No HF_RESULTS_REPO / HF_TOKEN — printing results locally:")
        for f in json_files:
            print(f.read_text())

    log("Done.")


if __name__ == "__main__":
    main()
