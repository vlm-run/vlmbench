#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "vlmbench @ git+https://github.com/vlm-run/vlmbench.git@spillai/hf-jobs-gpu-sweep",
# ]
# ///
"""Simple benchmark script for HF Jobs."""

import subprocess
import sys

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
INPUT = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/image.caption/car.jpg"

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "-m",
        "vlmbench",
        "run",
        "--model",
        MODEL,
        "--input",
        INPUT,
        "--serve",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
