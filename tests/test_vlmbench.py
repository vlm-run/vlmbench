"""Minimal end-to-end test for vlmbench CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

VLMBENCH_BIN = str(Path(sys.executable).parent / "vlmbench")

# Vision model candidates in preference order
OLLAMA_VISION_KEYWORDS = [
    "vl",
    "vision",
]


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


def test_cli_run_local_input(server: dict, test_image: Path, tmp_path: Path) -> None:
    """Test benchmark with local file input."""
    save_dir = tmp_path / "results"
    result = subprocess.run(
        [
            VLMBENCH_BIN,
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


def test_cli_run_local_directory(server: dict, tmp_path: Path) -> None:
    """Test benchmark with local directory input and --max-samples."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50)).save(img_dir / f"img_{i}.png")

    save_dir = tmp_path / "results"
    result = subprocess.run(
        [
            VLMBENCH_BIN,
            "run",
            "--model",
            server["model"],
            "--input",
            str(img_dir),
            "--base-url",
            server["base_url"],
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "128",
            "--max-samples",
            "2",
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
    assert data["results"]["total_inputs_processed"] == 2


def _has_datasets_library() -> bool:
    """Check if datasets library is available."""
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_datasets_library(), reason="datasets library not installed")
def test_cli_run_hf_dataset(server: dict, tmp_path: Path) -> None:
    """Test benchmark with HuggingFace dataset input and --max-samples."""
    save_dir = tmp_path / "results"
    result = subprocess.run(
        [
            VLMBENCH_BIN,
            "run",
            "--model",
            server["model"],
            "--dataset",
            "vlm-run/FineVision-vlmbench-mini",
            "--dataset-split",
            "train",
            "--base-url",
            server["base_url"],
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "128",
            "--max-samples",
            "2",
            "--save",
            str(save_dir),
            "--no-serve",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    json_files = list(save_dir.glob("*.json"))
    assert len(json_files) == 1, f"Expected 1 result file, found {len(json_files)}"

    data = json.loads(json_files[0].read_text())
    assert data["results"]["total_inputs_processed"] == 2


@pytest.mark.skipif(not _has_datasets_library(), reason="datasets library not installed")
def test_cli_run_hf_dataset_with_image_col(server: dict, tmp_path: Path) -> None:
    """Test benchmark with HuggingFace dataset and explicit --image-col."""
    save_dir = tmp_path / "results"
    result = subprocess.run(
        [
            VLMBENCH_BIN,
            "run",
            "--model",
            server["model"],
            "--dataset",
            "vlm-run/FineVision-vlmbench-mini",
            "--dataset-image-col",
            "images",
            "--base-url",
            server["base_url"],
            "--runs",
            "1",
            "--warmup",
            "0",
            "--max-tokens",
            "128",
            "--max-samples",
            "2",
            "--save",
            str(save_dir),
            "--no-serve",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    json_files = list(save_dir.glob("*.json"))
    assert len(json_files) == 1, f"Expected 1 result file, found {len(json_files)}"

    data = json.loads(json_files[0].read_text())
    assert data["results"]["total_inputs_processed"] == 2


def test_cli_input_mutual_exclusivity() -> None:
    """Verify --input and --dataset are mutually exclusive."""
    result = subprocess.run(
        [
            VLMBENCH_BIN,
            "run",
            "--model",
            "test-model",
            "--input",
            "/tmp/test.png",
            "--dataset",
            "some/dataset",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_help():
    """Verify vlmbench --help runs successfully."""
    result = subprocess.run(
        [VLMBENCH_BIN, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0


def test_import():
    """Verify the package is importable and has a version."""
    import importlib.metadata

    version = importlib.metadata.version("vlmbench")
    assert isinstance(version, str)
