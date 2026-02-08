"""Minimal end-to-end test for vlmbench CLI."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

# Vision model candidates in preference order
OLLAMA_VISION_KEYWORDS = ["vl", "vision", "llava"]


def _fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _find_ollama_vision_model() -> str | None:
    """Query Ollama for available models and pick first vision-capable one."""
    data = _fetch_json("http://localhost:11434/api/tags")
    if not data:
        return None
    for m in data.get("models", []):
        name = m["name"]
        if any(kw in name.lower() for kw in OLLAMA_VISION_KEYWORDS):
            return name
    return None


def _find_vllm_model() -> str | None:
    """Query vLLM for available models and pick first one."""
    data = _fetch_json("http://localhost:8000/v1/models")
    if not data:
        return None
    models = data.get("data", [])
    if models:
        return models[0].get("id")
    return None


@pytest.fixture()
def server():
    """Detect a running inference server with a vision model, or skip."""
    vllm_model = _find_vllm_model()
    if vllm_model:
        return {"base_url": "http://localhost:8000/v1", "model": vllm_model}

    ollama_model = _find_ollama_vision_model()
    if ollama_model:
        return {"base_url": "http://localhost:11434/v1", "model": ollama_model}

    pytest.skip("No inference server with a vision model available")


@pytest.fixture()
def test_image(tmp_path: Path) -> Path:
    """Create a tiny 64x64 white PNG."""
    img_path = tmp_path / "test.png"
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_path)
    return img_path


def test_cli_run(server: dict, test_image: Path, tmp_path: Path) -> None:
    save_dir = tmp_path / "results"
    result = subprocess.run(
        [
            "vlmbench",
            "run",
            "--model",
            server["model"],
            "--input",
            str(test_image),
            "--base-url",
            server["base_url"],
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "128",
            "--save",
            str(save_dir),
            "--no-serve",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    json_files = list(save_dir.glob("*.json"))
    assert len(json_files) == 1, f"Expected 1 result file, found {len(json_files)}"

    data = json.loads(json_files[0].read_text())
    assert data["schema_version"] == "0.1.0"
    assert data["model"]["model_id"] == server["model"]
    assert data["results"]["total_inputs_processed"] == 1
    assert data["results"]["errors"] == 0


def test_version():
    """Verify vlmbench --help shows version."""
    result = subprocess.run(
        ["vlmbench", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "v0.2.0" in result.stdout


def test_import():
    """Verify the package is importable."""
    from vlmbench import __version__

    assert __version__ == "0.2.0"
