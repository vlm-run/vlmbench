# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "vlmbench",
#   "huggingface_hub",
#   "vllm",
#   "Pillow",
# ]
# ///
"""Run vlmbench on a HuggingFace Jobs GPU instance and upload results."""

import argparse
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def wait_for_server(url: str = "http://localhost:8000/v1/models", timeout: int = 600) -> None:
    """Poll vLLM server until it responds 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"vLLM server not ready after {timeout}s")


def create_test_image(path: Path) -> None:
    """Generate a simple test image with text for benchmarking."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), "Hello vlmbench\n2 + 2 = 4\nThe quick brown fox\njumps over the lazy dog.", fill=(0, 0, 0))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vlmbench on HF Jobs and upload results")
    parser.add_argument("--model", required=True, help="Model ID (e.g. Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--gpu-flavor", required=True, help="HF Jobs flavor (e.g. a10g-small)")
    parser.add_argument("--repo-id", default="vlm-run/vlmbench-results", help="HF dataset repo for results")
    parser.add_argument("--serve-args", default="", help="Extra vLLM serve args")
    args = parser.parse_args()

    save_dir = Path("/tmp/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    test_image = Path("/tmp/test_input.png")
    create_test_image(test_image)

    # Start vLLM server as background process
    cmd = ["vllm", "serve", args.model]
    if args.serve_args:
        cmd.extend(shlex.split(args.serve_args))

    print(f"Starting vLLM: {' '.join(cmd)}")
    server_proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    try:
        print("Waiting for vLLM server...")
        wait_for_server()
        print("vLLM ready.")

        # Run vlmbench
        bench_cmd = [
            "vlmbench", "run",
            "--model", args.model,
            "--input", str(test_image),
            "--base-url", "http://localhost:8000/v1",
            "--no-serve",
            "--tag", args.gpu_flavor,
            "--save", str(save_dir),
        ]
        print(f"Running: {' '.join(bench_cmd)}")
        subprocess.run(bench_cmd, check=True)

        # Find and upload result JSON
        result_files = sorted(save_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not result_files:
            print("ERROR: No result JSON files found")
            sys.exit(1)

        result_file = result_files[-1]
        path_in_repo = f"results/{args.gpu_flavor}/{result_file.name}"

        from huggingface_hub import upload_file

        print(f"Uploading {result_file.name} -> {args.repo_id}/{path_in_repo}")
        upload_file(
            path_or_fileobj=str(result_file),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        print("Done.")

    finally:
        server_proc.terminate()
        server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
