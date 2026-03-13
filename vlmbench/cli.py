"""
██╗   ██╗██╗     ███╗   ███╗
██║   ██║██║     ████╗ ████║
██║   ██║██║     ██╔████╔██║
╚██╗ ██╔╝██║     ██║╚██╔╝██║
 ╚████╔╝ ███████╗██║ ╚═╝ ██║
  ╚═══╝  ╚══════╝╚═╝     ╚═╝
██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝

vlmbench — Single-file, drop-in VLM benchmark CLI for your agents.
by VLM Run · https://vlm.run
"""

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from types import UnionType
from typing import Any, Protocol, get_args, get_origin, get_type_hints, runtime_checkable

import httpx
import yaml
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from openai.types.create_embedding_response import CreateEmbeddingResponse
from PIL import Image
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential  # noqa: F401

from vlmbench.version import __version__ as VERSION

# ── Constants ─────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "0.1.0"
DEFAULT_PROMPT = "Extract all text from this document."
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RUNS = 3
DEFAULT_CONCURRENCY = 8
DEFAULT_SAVE_DIR = str(Path.home() / ".vlmbench" / "benchmarks")
DEFAULT_UPLOAD_REPO = "vlm-run/vlmbench-results"
DEFAULT_API_KEY = "no-key"
DEFAULT_IMAGE_URL = "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/image.caption/car.jpg"
DEFAULT_MAX_IMAGE_SIZE = 2048

DEFAULT_VLLM_IMAGE = "vllm-openai:latest"
DEFAULT_MAX_IMAGE_SIZE = 2048
DEFAULT_MAX_PDF_PAGES = 8
HF_DATASET_PREFIX = "hf://datasets/"
STEEL_BLUE = "#4682B4"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
# Sentinel prefix used to carry plain-text payloads through the inputs pipeline
TEXT_URI_PREFIX = "text:"

console = Console()


def print_error(title: str, message: str) -> None:
    """Print a styled error panel."""
    panel = Panel(
        f"[white]{message}[/white]",
        title=f"[bold]Error: {title}[/bold]",
        title_align="left",
        border_style="red",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)


def print_warning(message: str) -> None:
    """Print a styled warning."""
    console.print(f"  [yellow bold]![/yellow bold] [yellow]{message}[/yellow]")


# ── Schema ────────────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    model_id: str = ""
    revision: str = "main"
    quant: str | None = None
    params_b: float | None = None


@dataclass
class EnvironmentInfo:
    backend: str = ""
    backend_version: str | None = None
    base_url: str = ""
    accelerator: str | None = None
    gpu_name: str | None = None
    gpu_vram_mib: int | None = None
    gpu_driver: str | None = None
    ram_total_mib: int | None = None
    cpu: str | None = None
    os: str | None = None


@dataclass
class ImageStats:
    count: int = 0
    avg_width: int = 0
    avg_height: int = 0
    min_width: int = 0
    min_height: int = 0
    max_width: int = 0
    max_height: int = 0
    median_width: int = 0
    median_height: int = 0
    avg_pixels: int = 0
    min_pixels: int = 0
    max_pixels: int = 0
    median_pixels: int = 0


@dataclass
class InputInfo:
    hash: str = ""
    total_inputs: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)
    image_stats: ImageStats | None = None
    prompt: str = ""
    max_tokens: int = 0
    max_concurrency: int = 1
    task: str = "completion"


@dataclass
class StatBlock:
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


@dataclass
class TokenStat:
    mean: int = 0
    min: int = 0
    max: int = 0


@dataclass
class Results:
    ttft_ms: StatBlock = field(default_factory=StatBlock)
    tpot_ms: StatBlock = field(default_factory=StatBlock)
    itl_ms: StatBlock = field(default_factory=StatBlock)
    e2el_ms: StatBlock = field(default_factory=StatBlock)
    tokens_per_sec: float = 0.0
    input_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0
    inputs_per_sec: float = 0.0
    latency_s_per_input: StatBlock = field(default_factory=StatBlock)
    prompt_tokens: TokenStat = field(default_factory=TokenStat)
    completion_tokens: TokenStat = field(default_factory=TokenStat)
    cached_tokens: TokenStat | None = None
    reasoning_tokens: TokenStat | None = None
    vram_peak_mib: int | None = None
    embedding_dim: int | None = None
    total_inputs_processed: int = 0
    total_duration_s: float = 0.0
    errors: int = 0
    retries: int = 0


@dataclass
class RunRaw:
    input_idx: int = 0
    run: int = 0
    ttft_ms: float | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    embedding_dim: int = 0
    itl_ms: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkResult:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    timestamp: str = ""
    tag: str | None = None
    model: ModelInfo = field(default_factory=ModelInfo)
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    input: InputInfo = field(default_factory=InputInfo)
    results: Results = field(default_factory=Results)
    runs_raw: list[RunRaw] = field(default_factory=list)


def _coerce_field(ftype: type, val: Any) -> Any:
    """Coerce a JSON-deserialized value to match a dataclass field type."""
    if val is None:
        return None
    origin = get_origin(ftype)
    args = get_args(ftype)
    # X | None is UnionType in Python 3.10+
    if origin is UnionType:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _coerce_field(non_none[0], val)
        return val
    # list[X] where X is a dataclass
    if origin is list and args and dataclasses.is_dataclass(args[0]):
        return [_dc_from_dict(args[0], item) if isinstance(item, dict) else item for item in val]
    # Nested dataclass
    if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
        return _dc_from_dict(ftype, val)
    return val


def _dc_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, tolerating missing/extra keys."""
    if not isinstance(data, dict):
        return data
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue  # use field default
        kwargs[f.name] = _coerce_field(hints[f.name], data[f.name])
    return cls(**kwargs)


# ── Environment Detection ─────────────────────────────────────────────────────


def detect_backend(base_url: str) -> tuple[str, str | None]:
    """Detect the backend type and version from the base URL."""
    # Strip /v1 suffix for probing
    probe_url = base_url.rstrip("/")
    if probe_url.endswith("/v1"):
        probe_url = probe_url[:-3]

    # 1. Try vLLM: GET /version
    try:
        req = urllib.request.Request(f"{probe_url}/version", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return "vllm", data.get("version")
    except Exception:
        pass

    # 2. Try Ollama: GET /api/version
    try:
        req = urllib.request.Request(f"{probe_url}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return "ollama", data.get("version")
    except Exception:
        pass

    # 3. URL-based detection
    match base_url:
        case url if "openai.com" in url:
            return "openai", None
        case url if "together.xyz" in url:
            return "together", None
        case _:
            return "generic", None


def _nvidia_gpu_index() -> int:
    """Return the GPU index to query, respecting CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        try:
            return int(cvd.split(",")[0])
        except ValueError:
            pass
    return 0


def _collect_server_pids(port: int) -> set[int]:
    """Collect all PIDs associated with the server on *port* (host + Docker)."""
    pids: set[int] = set()

    # Host-native: lsof
    try:
        lsof = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5)
        if lsof.returncode == 0 and lsof.stdout.strip():
            pids.update(int(p) for p in lsof.stdout.strip().split("\n") if p.isdigit())
    except Exception:
        pass

    # Docker: find containers publishing this port, get their host PIDs
    try:
        docker_ps = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"publish={port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if docker_ps.returncode == 0 and docker_ps.stdout.strip():
            for cid in docker_ps.stdout.strip().split("\n"):
                cid = cid.strip()
                if not cid:
                    continue
                inspect = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Pid}}", cid],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if inspect.returncode == 0 and inspect.stdout.strip().isdigit():
                    pids.add(int(inspect.stdout.strip()))
    except Exception:
        pass

    # Expand to child processes (GPU workers are often forked)
    children: set[int] = set()
    for pid in pids:
        try:
            result = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                children.update(int(c) for c in result.stdout.strip().split("\n") if c.isdigit())
        except Exception:
            pass
    pids |= children
    return pids


def _bus_id_to_gpu_index(bus_id: str) -> int | None:
    """Map a PCI bus ID to an nvidia-smi GPU index."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1] == bus_id:
                return int(parts[0])
    except Exception:
        pass
    return None


def get_gpu_for_server_port(port: int = 8000) -> int | None:
    """Find which GPU is being used by the server on the given port.

    Handles both host-native processes and Docker containers by collecting
    PIDs from lsof and docker inspect, then matching against nvidia-smi
    compute apps.  Returns the GPU index, or None if not found.
    """
    try:
        all_pids = _collect_server_pids(port)
        if not all_pids:
            return None

        smi_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if smi_result.returncode != 0:
            return None

        for line in smi_result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    if int(parts[0]) in all_pids:
                        return _bus_id_to_gpu_index(parts[1])
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def get_nvidia_gpu_info(gpu_idx: int | None = None) -> dict[str, Any]:
    """Linux: parse nvidia-smi for GPU info.

    Args:
        gpu_idx: Specific GPU index to query. If None, uses CUDA_VISIBLE_DEVICES or 0.
    """
    try:
        if gpu_idx is None:
            gpu_idx = _nvidia_gpu_index()
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_idx}",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            return {
                "gpu_name": parts[0],
                "gpu_vram_mib": int(float(parts[1])),
                "gpu_driver": parts[3],
                "accelerator": "cuda",
            }
    except Exception:
        pass
    return {}


def get_apple_gpu_info() -> dict[str, Any]:
    """macOS: parse system_profiler for GPU info."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        output = result.stdout
        info: dict[str, Any] = {"accelerator": "metal"}

        # Extract chipset/GPU name
        for line in output.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("Chipset Model:"):
                info["gpu_name"] = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.startswith("VRAM") or "Total Number of Cores" in line_stripped:
                pass  # handled below

        # Try to get memory from sysctl (unified memory on Apple Silicon)
        try:
            mem_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if mem_result.returncode == 0:
                mem_bytes = int(mem_result.stdout.strip())
                info["gpu_vram_mib"] = mem_bytes // (1024 * 1024)
        except Exception:
            pass

        # GPU driver = macOS version
        info["gpu_driver"] = f"macOS {platform.mac_ver()[0]}"

        return info
    except Exception:
        pass
    return {}


def get_system_info() -> dict[str, Any]:
    """Cross-platform: RAM, CPU, OS."""
    info: dict[str, Any] = {"os": platform.platform()}
    system = platform.system()

    # CPU
    try:
        match system:
            case "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    info["cpu"] = result.stdout.strip()
            case "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            info["cpu"] = line.split(":", 1)[1].strip()
                            break
    except Exception:
        pass

    # RAM
    try:
        match system:
            case "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    info["ram_total_mib"] = int(result.stdout.strip()) // (1024 * 1024)
            case "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(re.search(r"\d+", line).group())
                            info["ram_total_mib"] = kb // 1024
                            break
    except Exception:
        pass

    return info


def collect_environment(base_url: str) -> EnvironmentInfo:
    """Collect full environment info."""
    backend, backend_version = detect_backend(base_url)

    # Start with system info
    sys_info = get_system_info()
    env_data: dict[str, Any] = {
        "backend": backend,
        "backend_version": backend_version,
        "base_url": base_url,
        "os": sys_info.get("os"),
        "cpu": sys_info.get("cpu"),
        "ram_total_mib": sys_info.get("ram_total_mib"),
    }

    # GPU info based on platform and backend
    match (backend, platform.system()):
        case ("openai" | "together", _):
            # Cloud APIs — no local GPU info
            env_data["accelerator"] = None
            env_data["gpu_name"] = None
            env_data["gpu_vram_mib"] = None
            env_data["gpu_driver"] = None
        case (_, "Darwin"):
            env_data.update(get_apple_gpu_info())
        case (_, "Linux"):
            # Try to detect which GPU the server is using
            try:
                from urllib.parse import urlparse

                parsed = urlparse(base_url)
                port = parsed.port or 8000
                server_gpu_idx = get_gpu_for_server_port(port)
            except Exception:
                server_gpu_idx = None
            env_data.update(get_nvidia_gpu_info(server_gpu_idx))

    return EnvironmentInfo(**env_data)


# ── Server Management ────────────────────────────────────────────────────────


@runtime_checkable
class ServerManager(Protocol):
    """Protocol for pluggable inference server backends."""

    def is_running(self) -> bool: ...
    def start(self, model: str, extra_args: str | None = None) -> None: ...
    def wait_ready(self, timeout: int = 120) -> bool: ...
    def get_base_url(self) -> str: ...


def _tmux_session_name(backend: str) -> str:
    """Generate a stable tmux session name: vlmbench-vllm, vlmbench-sglang, vlmbench-ollama."""
    return f"vlmbench-{backend}"


def _backend_category(backend: str) -> str:
    """Extract backend category: 'vllm-openai:latest' → 'vllm', 'sglang:latest' → 'sglang'."""
    return backend.split(":")[0].split("-")[0]


def _parse_docker_backend(backend: str) -> tuple[str, str]:
    """Parse a backend string like 'vllm-openai:nightly' into (docker_image, container_name).

    Mapping: 'vllm-openai:nightly' → image='vllm/vllm-openai:nightly', container='vlmbench-vllm-openai'
             'sglang:latest'       → image='lmsysorg/sglang:latest',   container='vlmbench-sglang'
    """
    # Known org prefixes for shorthand resolution
    KNOWN_ORGS = {
        "vllm": "vllm",
        "sglang": "lmsysorg",
    }

    # Split tag
    if ":" in backend:
        name_part, tag = backend.rsplit(":", 1)
    else:
        name_part, tag = backend, "latest"

    # If it already looks like a full image (contains /), use as-is
    if "/" in name_part:
        image = f"{name_part}:{tag}"
        container = f"vlmbench-{name_part.split('/')[-1]}"
        return image, container

    # Try to resolve org from the prefix (e.g., vllm-openai → vllm/vllm-openai)
    prefix = name_part.split("-")[0]
    org = KNOWN_ORGS.get(prefix, prefix)
    image = f"{org}/{name_part}:{tag}"
    container = f"vlmbench-{name_part}"
    return image, container


def _pull_docker_image(backend: str = DEFAULT_VLLM_IMAGE) -> bool:
    """Pull a Docker image for the given backend. Returns True on success."""
    if not shutil.which("docker"):
        print_warning("Docker not found on PATH. Install Docker to use GPU backends.")
        return False
    if backend == "vllm":
        backend = DEFAULT_VLLM_IMAGE
    image, _ = _parse_docker_backend(backend)
    console.print(f"  [cyan]Pulling [bold]{image}[/bold]...[/cyan]")
    try:
        result = subprocess.run(["docker", "pull", image], timeout=600)
        if result.returncode == 0:
            console.print(f"  [green]Image [bold]{image}[/bold] is ready.[/green]")
            return True
        print_warning(f"Failed to pull {image}.")
        return False
    except subprocess.TimeoutExpired:
        print_warning(f"Docker pull timed out for {image}.")
        return False
    except Exception as e:
        print_warning(f"Docker pull failed: {e}")
        return False


class DockerServerManager:
    """Manage an inference server via Docker with GPU support (tmux or subprocess)."""

    DEFAULT_PORT = 8000

    def __init__(self, backend: str, port: int | None = None) -> None:
        self.port = port or self.DEFAULT_PORT
        self._base_url = f"http://localhost:{self.port}/v1"
        self.docker_image, self.container_name = _parse_docker_backend(backend)
        self.category = _backend_category(backend)
        self.session_name = _tmux_session_name(self.category)
        self._subprocess: subprocess.Popen | None = None
        self._using_tmux = False

    def is_running(self) -> bool:
        try:
            OpenAI(base_url=self._base_url, api_key=DEFAULT_API_KEY, timeout=2).models.list()
            return True
        except Exception:
            return False

    def start(self, model: str, extra_args: str | None = None) -> None:
        _require_command("docker")

        cache_dir = Path.home() / ".vlmbench" / self.category
        cache_dir.mkdir(parents=True, exist_ok=True)
        cuda_env = "-e CUDA_DEVICE_ORDER=PCI_BUS_ID"
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None:
            cuda_env += f" -e CUDA_VISIBLE_DEVICES={cuda_visible}"

        # Build docker run command
        run_args = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-e",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        ]
        if cuda_visible is not None:
            run_args.extend(["-e", f"CUDA_VISIBLE_DEVICES={cuda_visible}"])
        run_args.extend(
            [
                "--name",
                self.container_name,
                "-p",
                f"{self.port}:8000",
                "--ipc=host",
                "-v",
                f"{cache_dir}:/root/.cache/huggingface",
                self.docker_image,
                "--model",
                model,
            ]
        )
        if extra_args:
            run_args.extend(shlex.split(extra_args))

        if shutil.which("tmux"):
            self._start_with_tmux(run_args)
        else:
            self._start_with_subprocess(run_args)

    def _start_with_tmux(self, run_args: list[str]) -> None:
        """Start server in tmux with GPU monitoring pane."""
        self._using_tmux = True
        pull_cmd = f"docker pull {shlex.quote(self.docker_image)}"
        run_cmd = shlex.join(run_args)
        server_cmd = f"{pull_cmd} && {run_cmd}"

        subprocess.run(["tmux", "kill-session", "-t", self.session_name], capture_output=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", self.session_name, server_cmd], check=True)

        monitor_cmd = _gpu_monitor_cmd()
        if monitor_cmd:
            subprocess.run(["tmux", "split-window", "-v", "-t", self.session_name, monitor_cmd], check=True)
            subprocess.run(["tmux", "select-pane", "-t", f"{self.session_name}:.0"], check=True)

        console.print(
            f"  [cyan]Started [bold]{self.docker_image}[/bold] in tmux session '[bold]{self.session_name}[/bold]'[/cyan]"
        )
        console.print(f"  [dim]Attach with: tmux attach -t {self.session_name}[/dim]")
        console.print(f"  [dim]Container:   docker logs -f {self.container_name}[/dim]")

    def _start_with_subprocess(self, run_args: list[str]) -> None:
        """Start server as background subprocess (no tmux)."""
        self._using_tmux = False
        console.print(f"  [cyan]Pulling [bold]{self.docker_image}[/bold]...[/cyan]")
        subprocess.run(["docker", "pull", self.docker_image], check=True)
        console.print(f"  [cyan]Starting [bold]{self.docker_image}[/bold] as subprocess...[/cyan]")
        self._subprocess = subprocess.Popen(
            run_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print(f"  [dim]Container:   docker logs -f {self.container_name}[/dim]")

    def wait_ready(self, timeout: int = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            if self._subprocess is not None and self._subprocess.poll() is not None:
                return False
            time.sleep(2)
        return False

    def get_base_url(self) -> str:
        return self._base_url


class OllamaServerManager:
    """Manage an Ollama inference server (tmux or subprocess)."""

    DEFAULT_PORT = 11434

    def __init__(self, port: int | None = None) -> None:
        self.port = port or self.DEFAULT_PORT
        self._base_url = f"http://localhost:{self.port}/v1"
        self.session_name = _tmux_session_name("ollama")
        self._subprocess: subprocess.Popen | None = None
        self._using_tmux = False

    def is_running(self) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{self.port}/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self, model: str, extra_args: str | None = None) -> None:
        if not self.is_running():
            _require_command("ollama")
            serve_args = ["ollama", "serve"]
            if extra_args:
                serve_args.extend(shlex.split(extra_args))

            if shutil.which("tmux"):
                self._start_with_tmux(serve_args)
            else:
                self._start_with_subprocess(serve_args)

            if not self.wait_ready(timeout=30):
                console.print("[red]Ollama failed to start within 30s.[/red]")
                sys.exit(1)

        console.print(f"  [cyan]Pulling model '{model}' (if needed)...[/cyan]")
        subprocess.run(["ollama", "pull", model], check=True)

    def _start_with_tmux(self, serve_args: list[str]) -> None:
        """Start Ollama in tmux with monitoring pane."""
        self._using_tmux = True
        server_cmd = shlex.join(serve_args)
        subprocess.run(["tmux", "kill-session", "-t", self.session_name], capture_output=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", self.session_name, server_cmd], check=True)

        monitor_cmd = _gpu_monitor_cmd()
        if monitor_cmd:
            subprocess.run(["tmux", "split-window", "-v", "-t", self.session_name, monitor_cmd], check=True)
            subprocess.run(["tmux", "select-pane", "-t", f"{self.session_name}:.0"], check=True)

        console.print(f"  [cyan]Started Ollama in tmux session '[bold]{self.session_name}[/bold]'[/cyan]")
        console.print(f"  [dim]Attach with: tmux attach -t {self.session_name}[/dim]")

    def _start_with_subprocess(self, serve_args: list[str]) -> None:
        """Start Ollama as background subprocess (no tmux)."""
        self._using_tmux = False
        console.print("  [cyan]Starting Ollama as subprocess...[/cyan]")
        self._subprocess = subprocess.Popen(
            serve_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: int = 120) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            if self._subprocess is not None and self._subprocess.poll() is not None:
                return False
            time.sleep(2)
        return False

    def get_base_url(self) -> str:
        return self._base_url


class NativeVllmServerManager:
    """Manage a native vLLM inference server (``vllm serve``) (tmux or subprocess)."""

    DEFAULT_PORT = 8000

    def __init__(self, port: int | None = None) -> None:
        self.port = port or self.DEFAULT_PORT
        self._base_url = f"http://localhost:{self.port}/v1"
        self.session_name = _tmux_session_name("vllm")
        self._subprocess: subprocess.Popen | None = None
        self._using_tmux = False

    def is_running(self) -> bool:
        try:
            OpenAI(base_url=self._base_url, api_key=DEFAULT_API_KEY, timeout=2).models.list()
            return True
        except Exception:
            return False

    def start(self, model: str, extra_args: str | None = None) -> None:
        _require_command("vllm")

        serve_args = ["vllm", "serve", model]
        if self.port != self.DEFAULT_PORT:
            serve_args.extend(["--port", str(self.port)])
        if extra_args:
            serve_args.extend(shlex.split(extra_args))

        if shutil.which("tmux"):
            self._start_with_tmux(serve_args)
        else:
            self._start_with_subprocess(serve_args)

    def _start_with_tmux(self, serve_args: list[str]) -> None:
        """Start vLLM in tmux with GPU monitoring pane."""
        self._using_tmux = True
        server_cmd = shlex.join(serve_args)
        subprocess.run(["tmux", "kill-session", "-t", self.session_name], capture_output=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", self.session_name, server_cmd], check=True)

        monitor_cmd = _gpu_monitor_cmd()
        if monitor_cmd:
            subprocess.run(["tmux", "split-window", "-v", "-t", self.session_name, monitor_cmd], check=True)
            subprocess.run(["tmux", "select-pane", "-t", f"{self.session_name}:.0"], check=True)

        console.print(f"  [cyan]Started native vLLM in tmux session '[bold]{self.session_name}[/bold]'[/cyan]")
        console.print(f"  [dim]Attach with: tmux attach -t {self.session_name}[/dim]")

    def _start_with_subprocess(self, serve_args: list[str]) -> None:
        """Start vLLM as background subprocess (no tmux)."""
        self._using_tmux = False
        console.print("  [cyan]Starting native vLLM as subprocess...[/cyan]")
        self._subprocess = subprocess.Popen(
            serve_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: int = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            if self._subprocess is not None and self._subprocess.poll() is not None:
                return False
            time.sleep(2)
        return False

    def get_base_url(self) -> str:
        return self._base_url


def _require_command(name: str) -> None:
    """Check that a CLI tool is available on PATH."""
    if shutil.which(name) is None:
        if name == "vllm":
            console.print("[red]'vllm' not found on PATH. Install with: uv pip install vllm[/red]")
        else:
            console.print(f"[red]'{name}' not found on PATH. Please install it first.[/red]")
        sys.exit(1)


def _build_server_cmd(backend: str, model: str, serve_args: str | None = None) -> tuple[str, str]:
    """Build the server command string and base_url for display purposes (without starting anything)."""
    if backend == "auto":
        backend = _auto_detect_backend()
    category = _backend_category(backend)

    match backend:
        case "ollama":
            cmd = "ollama serve"
            base_url = f"http://localhost:{OllamaServerManager.DEFAULT_PORT}/v1"
        case "vllm":
            cmd = f"vllm serve {shlex.quote(model)}"
            base_url = f"http://localhost:{NativeVllmServerManager.DEFAULT_PORT}/v1"
        case _:
            # Docker-based (vllm-openai, sglang)
            image, _container = _parse_docker_backend(backend)
            cache_dir = Path.home() / ".vlmbench" / category
            cmd = (
                f"docker run --rm --gpus all"
                f" -p {DockerServerManager.DEFAULT_PORT}:8000 --ipc=host"
                f" -v {shlex.quote(str(cache_dir))}:/root/.cache/huggingface"
                f" {shlex.quote(image)}"
                f" --model {shlex.quote(model)}"
            )
            base_url = f"http://localhost:{DockerServerManager.DEFAULT_PORT}/v1"

    if serve_args:
        cmd += f" {shlex.join(shlex.split(serve_args))}"
    return cmd, base_url


def _build_rerun_args(base_url: str) -> list[str]:
    """Reconstruct vlmbench CLI args for re-run, swapping --serve for --base-url."""
    rerun_args: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--serve", "--no-serve"):
            continue
        if arg in ("--backend", "--serve-args"):
            skip_next = True
            continue
        if arg.startswith("--backend=") or arg.startswith("--serve-args="):
            continue
        rerun_args.append(arg)
    rerun_args.extend(["--base-url", base_url])
    return rerun_args


def _print_server_instructions(
    server_cmd: str,
    base_url: str,
    *,
    reason: str,
    post_cmd: str | None = None,
    tip: str | None = None,
) -> None:
    """Print a Rich panel with server start command and vlmbench re-run command."""
    lines = Text()
    lines.append(f"  {reason}\n")
    lines.append("  Start the server manually, then re-run the benchmark.\n")

    step = 1
    lines.append(f"\n  {step}. Start the server in a separate terminal:\n\n")
    lines.append(f"     {server_cmd}\n", style="bold green")
    step += 1

    if post_cmd:
        lines.append(f"\n  {step}. Once the server is ready, run:\n\n")
        lines.append(f"     {post_cmd}\n", style="bold green")
        step += 1

    rerun_cmd = "vlmbench " + shlex.join(_build_rerun_args(base_url))
    lines.append(f"\n  {step}. Re-run the benchmark:\n\n")
    lines.append(f"     {rerun_cmd}\n", style="bold green")

    panel = Panel(
        lines,
        title="[bold]Server Setup[/bold]",
        title_align="left",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(1, 1),
    )
    console.print(panel)
    if tip:
        console.print(f"  [dim]{tip}[/dim]\n")


def _gpu_monitor_cmd() -> str | None:
    """Return the best available GPU/system monitor command for the bottom tmux pane."""
    match platform.system():
        case "Darwin":
            # macOS: prefer macmon (brew install macmon)
            return "macmon" if shutil.which("macmon") else None
        case _:
            # Linux: prefer nvitop via uvx
            if shutil.which("uvx"):
                return "uvx nvitop -m full -c"
            if shutil.which("nvitop"):
                return "nvitop -m full -c"
            return None


def _auto_detect_backend() -> str:
    """Pick a default backend based on the current platform."""
    match platform.system():
        case "Darwin":
            return "ollama"
        case _:
            return DEFAULT_VLLM_IMAGE  # Linux defaults to Docker vLLM


def _resolve_backend_for_monitor(backend: str) -> str:
    """Resolve backend string to a category: 'ollama', 'vllm', or 'sglang'."""
    match backend:
        case "auto":
            return "ollama" if platform.system() == "Darwin" else "vllm"
        case "ollama":
            return "ollama"
        case _ if backend.startswith("sglang"):
            return "sglang"
        case _:
            return "vllm"


def _create_server_manager(backend: str) -> DockerServerManager | OllamaServerManager | NativeVllmServerManager:
    """Create the appropriate server manager for the given backend."""
    match backend:
        case "ollama":
            return OllamaServerManager()
        case "vllm":
            return NativeVllmServerManager()
        case _:
            return DockerServerManager(backend)


def _find_docker_container(name_prefix: str = "vlmbench-") -> str | None:
    """Find a running Docker container matching the vlmbench naming convention."""
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name_prefix}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _server_log_cmd(backend: str) -> str | None:
    """Return a command to tail server logs for an already-running backend."""
    if backend == "ollama":
        log_path = Path.home() / ".ollama" / "logs" / "server.log"
        if log_path.exists():
            return f"tail -f {log_path}"
    else:
        # Docker-based backends (vllm, sglang, etc.)
        container = _find_docker_container("vlmbench-")
        if container:
            return f"docker logs -f {container}"
    return None


def _start_monitor_session(backend: str) -> str | None:
    """Start a tmux session with server logs (top) and GPU monitor (bottom).

    Returns the tmux session name, or None if tmux is unavailable.
    """
    if not shutil.which("tmux"):
        return None

    session_name = _tmux_session_name(backend)

    # Check if session already exists (e.g. from a prior run today)
    check = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    session_exists = check.returncode == 0

    if not session_exists:
        # Top pane: server logs (or just a label if no logs available)
        log_cmd = _server_log_cmd(backend)
        top_cmd = log_cmd or f"echo 'vlmbench monitor — {backend} server logs not available' && sleep infinity"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, top_cmd],
            check=True,
        )

    # Ensure bottom pane exists with GPU/system monitor
    monitor_cmd = _gpu_monitor_cmd()
    if monitor_cmd:
        # Count existing panes
        pane_count = subprocess.run(
            ["tmux", "list-panes", "-t", session_name],
            capture_output=True,
            text=True,
        )
        num_panes = len(pane_count.stdout.strip().split("\n")) if pane_count.stdout.strip() else 0
        if num_panes < 2:
            subprocess.run(
                ["tmux", "split-window", "-v", "-t", session_name, monitor_cmd],
                check=True,
            )
            subprocess.run(
                ["tmux", "select-pane", "-t", f"{session_name}:.0"],
                check=True,
            )

    return session_name


def _fetch_model_from_server(base_url: str, api_key: str | None = None) -> str | None:
    """Return the first model id from the server's model list, or None on failure."""
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key or DEFAULT_API_KEY,
            timeout=5,
        )
        models = client.models.list().data
        if models:
            return models[0].id
    except Exception:
        pass
    return None


def resolve_server(
    base_url: str | None,
    serve: bool,
    backend: str,
    model: str | None,
    serve_args: str | None,
) -> tuple[str, str | None]:
    """Resolve the base URL, optionally starting a server.

    Returns (base_url, tmux_session_name | None).
    """
    # If user passed an explicit --base-url, use it directly
    if base_url is not None:
        # Only start a monitor session for local servers (skip external APIs)
        is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if is_local:
            monitor_backend = _resolve_backend_for_monitor(backend)
            session = _start_monitor_session(monitor_backend)
        else:
            session = None
        return base_url, session

    # Try to auto-detect an already-running server
    detected_url, detected_backend = _try_detect_running_server()
    if detected_url is not None:
        if serve:
            console.print(f"  [green]Server already running at {detected_url}[/green]")
        session = _start_monitor_session(detected_backend)
        return detected_url, session

    # No server running and --serve not requested → show setup instructions
    if not serve:
        server_cmd, server_url = _build_server_cmd(backend, model or "<model>", serve_args)
        resolved_backend = backend if backend != "auto" else _auto_detect_backend()
        post_cmd = (
            f"ollama pull {shlex.quote(model)}" if model and _backend_category(resolved_backend) == "ollama" else None
        )
        _print_server_instructions(
            server_cmd,
            server_url,
            reason="No running inference server detected.",
            post_cmd=post_cmd,
        )
        sys.exit(1)

    # --serve requested, no server found → start one
    if not model:
        console.print("[red]--model is required when using --serve to start a new server.[/red]")
        sys.exit(1)

    if backend == "auto":
        backend = _auto_detect_backend()

    manager = _create_server_manager(backend)

    if isinstance(manager, NativeVllmServerManager):
        backend_label = "native vLLM"
    else:
        backend_label = getattr(manager, "docker_image", backend)
    console.print(f"  [cyan]Auto-starting {backend_label}...[/cyan]")
    manager.start(model, extra_args=serve_args)

    console.print("  [dim]Waiting for server to be ready...[/dim]")
    if not manager.wait_ready(timeout=600):
        console.print("[red]Server failed to become ready within 600s.[/red]")
        sys.exit(1)

    console.print("  [green]Server ready.[/green]")

    # Return session name only if using tmux
    session = manager.session_name if getattr(manager, "_using_tmux", False) else None
    return manager.get_base_url(), session


def _try_detect_running_server() -> tuple[str | None, str]:
    """Check common local ports for a running inference server.

    Returns (base_url | None, backend_name).
    """
    for base_url, backend in [
        ("http://localhost:8000/v1", "vllm"),
        ("http://localhost:11434/v1", "ollama"),
    ]:
        try:
            client = OpenAI(base_url=base_url, api_key=DEFAULT_API_KEY, timeout=2)
            client.models.list()
            return base_url, backend
        except Exception:
            pass

    return None, "unknown"


# ── Input Loading ─────────────────────────────────────────────────────────────


def compute_image_stats(inputs: list[list[str]]) -> ImageStats | None:
    """Compute image dimension statistics from base64 data URI inputs."""
    widths: list[int] = []
    heights: list[int] = []
    for group in inputs:
        for uri in group:
            try:
                if uri.startswith("data:image/"):
                    b64_data = uri.split(",", 1)[1]
                    img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                continue
    if not widths:
        return None
    pixels = [w * h for w, h in zip(widths, heights)]
    sw, sh, sp = sorted(widths), sorted(heights), sorted(pixels)
    n = len(widths)
    return ImageStats(
        count=n,
        avg_width=round(statistics.mean(widths)),
        avg_height=round(statistics.mean(heights)),
        min_width=min(widths),
        min_height=min(heights),
        max_width=max(widths),
        max_height=max(heights),
        median_width=round(statistics.median(sw)),
        median_height=round(statistics.median(sh)),
        avg_pixels=round(statistics.mean(pixels)),
        min_pixels=min(pixels),
        max_pixels=max(pixels),
        median_pixels=round(statistics.median(sp)),
    )


def downscale_size(size: tuple[int, int], max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> tuple[int, int]:
    """Downscale a size if it is larger than max_size (in pixels) on either width or height dimension."""
    w, h = size
    if w <= max_size and h <= max_size:
        return size
    if w > h:
        W = min(w, max_size)
        H = int(h * W / w)
    else:
        H = min(h, max_size)
        W = int(w * H / h)
    return (W, H)


def downscale_image(img: Image.Image, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> Image.Image:
    """Downscale an image if it is larger than max_size (in pixels) on either width or height dimension."""
    w, h = img.size
    if w <= max_size and h <= max_size:
        return img
    W, H = downscale_size((w, h), max_size)
    return img.resize((W, H), Image.LANCZOS)


def image_to_base64(image: Image.Image | Path, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> str:
    """Convert a PIL Image or image file to a base64 data URI (JPEG, quality=98, downscaled)."""
    img = Image.open(image) if isinstance(image, Path) else image
    img = downscale_image(img, max_size)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=98)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def pdf_to_base64(
    path: Path,
    dpi: int = 150,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
) -> list[str]:
    """Convert PDF pages to base64 data URIs using pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    n_pages = min(len(doc), max_pages)
    results = []
    for idx in range(n_pages):
        page = doc[idx]
        bitmap = page.render(scale=dpi / 72)
        img = bitmap.to_pil()
        results.append(image_to_base64(img, max_size))
    doc.close()
    return results


def video_to_base64_frames(path: Path) -> list[str]:
    """Extract video frames at 1fps using ffmpeg, return as base64 data URIs."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        console.print("[yellow]Warning: ffmpeg not found, skipping video.[/yellow]")
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                ffmpeg,
                "-i",
                str(path),
                "-vf",
                "fps=1",
                "-q:v",
                "2",
                os.path.join(tmpdir, "frame_%04d.jpg"),
            ],
            capture_output=True,
            timeout=300,
        )
        frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
        results = []
        for frame in frames:
            with open(frame, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                results.append(f"data:image/jpeg;base64,{b64}")
        return results


def pil_to_base64(img: Any, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> str:
    """Convert a PIL Image to a base64 data URI, with optional resizing.

    Args:
        img: PIL Image
        max_size: Maximum dimension (width or height). Images larger than this
                  are downscaled while preserving aspect ratio.
    """
    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if hasattr(img, "mode") and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downscale if larger than max_size
    if max(img.width, img.height) > max_size:
        img.thumbnail((max_size, max_size))

    # Encode as JPEG for smaller size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def numpy_to_base64(arr: Any, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> str:
    """Convert a numpy array to a base64 data URI via PIL."""
    from PIL import Image

    # Handle different array shapes
    if arr.ndim == 2:
        # Grayscale
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # RGB
        img = Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA -> RGB
        img = Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        # Best effort
        img = Image.fromarray(arr)
    return pil_to_base64(img, max_size=max_size)


def _is_base64_image(val: Any) -> bool:
    """Check if a value is a base64-encoded image string."""
    if not isinstance(val, str):
        return False
    # Check for data URI format
    if val.startswith("data:image/"):
        return True
    # Check for raw base64 (heuristic: long string with base64 chars)
    if len(val) > 100 and all(
        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in val[:100]
    ):
        return True
    return False


def _is_pil_image(val: Any) -> bool:
    """Check if a value is a PIL Image."""
    return hasattr(val, "save") and hasattr(val, "mode")


def _is_numpy_image(val: Any) -> bool:
    """Check if a value is a numpy array that looks like an image."""
    return hasattr(val, "ndim") and hasattr(val, "shape") and val.ndim in (2, 3)


def _convert_to_base64(img: Any, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> str:
    """Convert an image (PIL, numpy, base64 string, or bytes) to a base64 data URI."""
    # Already a data URI
    if isinstance(img, str) and img.startswith("data:image/"):
        return img

    # Raw base64 string (no data URI prefix)
    if isinstance(img, str) and _is_base64_image(img):
        # Assume JPEG if no mime type provided
        return f"data:image/jpeg;base64,{img}"

    # Bytes - decode and re-encode with resizing
    if isinstance(img, bytes):
        from PIL import Image

        pil_img = Image.open(io.BytesIO(img))
        return pil_to_base64(pil_img, max_size=max_size)

    # PIL Image
    if _is_pil_image(img):
        return pil_to_base64(img, max_size=max_size)

    # numpy array
    if _is_numpy_image(img):
        return numpy_to_base64(img, max_size=max_size)

    # Dict with 'bytes' key (common in HF datasets)
    if isinstance(img, dict) and "bytes" in img:
        from PIL import Image

        pil_img = Image.open(io.BytesIO(img["bytes"]))
        return pil_to_base64(pil_img, max_size=max_size)

    raise ValueError(f"Unsupported image type: {type(img)}")


def _detect_image_column(ds: Any, columns: list[str]) -> tuple[str | None, bool]:
    """Detect the image column in a dataset.

    Returns:
        (column_name, is_list) - column name and whether it contains lists of images
    """
    # First, check for common column names
    for col in columns:
        col_lower = col.lower()
        if col_lower in ("image", "img"):
            return col, False
        if col_lower in ("images", "imgs"):
            return col, True

    # Try to detect by inspecting the first row
    first_row = ds[0]
    for col in columns:
        val = first_row[col]
        if val is None:
            continue

        # Single image checks
        if _is_pil_image(val) or _is_numpy_image(val) or _is_base64_image(val):
            return col, False

        # Dict with bytes (HF image format)
        if isinstance(val, dict) and "bytes" in val:
            return col, False

        # List of images
        if isinstance(val, list) and len(val) > 0:
            first_item = val[0]
            if (
                _is_pil_image(first_item)
                or _is_numpy_image(first_item)
                or _is_base64_image(first_item)
                or (isinstance(first_item, dict) and "bytes" in first_item)
            ):
                return col, True

    return None, False


_HF_TEXT_COL_NAMES = {"text", "prompt", "input", "content", "query", "question", "instruction"}


def load_hf_dataset(
    dataset_path: str,
    image_col: str | None = None,
    text_col: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> tuple[list[list[str]], dict[str, int], str]:
    """Load inputs from a HuggingFace dataset.

    Supports image columns (PIL / numpy / base64 / bytes) and plain-text columns.
    Text values are stored as ``text:<value>`` sentinels so they travel through
    the same pipeline as visual inputs.  Pass ``text_col`` to load a text column;
    if neither ``image_col`` nor ``text_col`` is given, image columns are
    auto-detected first, then text columns as a fallback.

    When max_samples is specified, uses streaming mode to avoid downloading
    the entire dataset.

    Args:
        dataset_path: HuggingFace dataset path (org/name or org/name:config)
        image_col: Column name containing images. If None, auto-detect.
        text_col: Column name containing text prompts/documents. Takes precedence
                  over image_col when both are given.
        split: Dataset split to load (default: train)
        max_samples: Maximum number of samples to load. If set, uses streaming.
        max_size: Maximum image dimension (width or height)

    Returns:
        (inputs, breakdown, hash)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]datasets library required for HuggingFace inputs.[/red]")
        console.print("[dim]Install with: uv pip install datasets[/dim]")
        sys.exit(1)

    # Parse dataset path: org/name or org/name:config
    if ":" in dataset_path:
        dataset_name, config_name = dataset_path.split(":", 1)
    else:
        dataset_name = dataset_path
        config_name = None

    # Use streaming mode when max_samples is specified to avoid full download
    use_streaming = max_samples is not None

    if use_streaming:
        console.print(f"  [cyan]Streaming [bold]{dataset_name}[/bold] (split={split}, limit={max_samples})...[/cyan]")
    else:
        console.print(f"  [cyan]Loading [bold]{dataset_name}[/bold] (split={split})...[/cyan]")

    try:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=use_streaming)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        sys.exit(1)

    # Get column names (works for both streaming and regular datasets)
    columns = ds.column_names

    # Determine image column by peeking at first row
    is_list_col = False
    first_row = None

    if use_streaming:
        # For streaming, we need to peek at the first row
        ds_iter = iter(ds)
        try:
            first_row = next(ds_iter)
        except StopIteration:
            console.print("[red]Dataset is empty.[/red]")
            sys.exit(1)
    else:
        first_row = ds[0]

    is_list_col = False
    is_text_col = False

    if text_col is not None:
        # Explicit text column requested
        if text_col not in columns:
            console.print(f"[red]Text column '{text_col}' not found. Available: {columns}[/red]")
            sys.exit(1)
        is_text_col = True
        active_col = text_col
    elif image_col is not None:
        if image_col not in columns:
            console.print(f"[red]Column '{image_col}' not found. Available: {columns}[/red]")
            sys.exit(1)
        first_val = first_row[image_col]
        is_list_col = isinstance(first_val, list)
        active_col = image_col
    else:
        # Auto-detect: try image columns first, then fall back to known text column names
        active_col = None
        for col in columns:
            val = first_row[col]
            if val is None:
                continue
            if isinstance(val, list) and len(val) > 0:
                if _is_pil_image(val[0]) or _is_numpy_image(val[0]) or _is_base64_image(val[0]):
                    active_col = col
                    is_list_col = True
                    break
            elif _is_pil_image(val) or _is_numpy_image(val) or _is_base64_image(val):
                active_col = col
                break
            elif isinstance(val, dict) and "bytes" in val:
                active_col = col
                break

        if active_col is None:
            # Fall back to a well-known text column name
            for col in columns:
                if col.lower() in _HF_TEXT_COL_NAMES and isinstance(first_row[col], str):
                    active_col = col
                    is_text_col = True
                    break

        if active_col is None:
            console.print(f"[red]No image or text column found. Columns: {columns}[/red]")
            console.print("[dim]Use --dataset-image-col or --dataset-text-col to specify the column.[/dim]")
            sys.exit(1)

    col_type = "text" if is_text_col else ("image list" if is_list_col else "image")
    console.print(f"  [dim]Using column: {active_col} ({col_type})[/dim]")

    inputs: list[list[str]] = []
    breakdown = {"images": 0, "pdf_pages": 0, "video_frames": 0, "hf_images": 0, "text_inputs": 0}
    hasher = hashlib.sha256()

    def process_row(row: dict) -> bool:
        """Process a single row, return True if processed successfully."""
        val = row[active_col]
        if val is None:
            return False

        if is_text_col:
            text = str(val).strip()
            if not text:
                return False
            inputs.append([f"{TEXT_URI_PREFIX}{text}"])
            breakdown["text_inputs"] += 1
            hasher.update(text.encode("utf-8"))
            return True

        if is_list_col:
            if len(val) > 0:
                b64_list = [_convert_to_base64(img, max_size=max_size) for img in val]
                inputs.append(b64_list)
                breakdown["hf_images"] += len(b64_list)
                for b64 in b64_list:
                    hasher.update(b64.encode("utf-8"))
                return True
        else:
            b64 = _convert_to_base64(val, max_size=max_size)
            inputs.append([b64])
            breakdown["hf_images"] += 1
            hasher.update(b64.encode("utf-8"))
            return True
        return False

    if use_streaming:
        # Process first row we already fetched
        process_row(first_row)

        # Stream remaining rows with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total} samples"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading", total=max_samples)
            progress.update(task, completed=len(inputs))

            for row in ds_iter:
                if len(inputs) >= max_samples:
                    break
                if process_row(row):
                    progress.update(task, completed=len(inputs))
    else:
        # Full dataset with progress bar
        total = len(ds)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total} rows"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing", total=total)
            for i, row in enumerate(ds):
                process_row(row)
                progress.update(task, completed=i + 1)

    if not inputs:
        console.print("[red]No inputs found in dataset.[/red]")
        sys.exit(1)

    summary = (
        f"{breakdown['text_inputs']} text inputs" if breakdown["text_inputs"] else f"{breakdown['hf_images']} images"
    )
    console.print(f"  [green]Loaded [bold]{len(inputs)}[/bold] inputs ({summary})[/green]")

    input_hash = f"sha256:{hasher.hexdigest()[:16]}"
    return inputs, breakdown, input_hash


def load_local_inputs(
    input_path: str, max_size: int = DEFAULT_MAX_IMAGE_SIZE
) -> tuple[list[list[str]], dict[str, int], str]:
    """Load inputs from a local file or directory.

    Supports images, PDFs, and videos.

    Args:
        input_path: Path to file or directory
        max_size: Maximum image dimension (width or height)

    Returns:
        (inputs, breakdown, hash)
        - inputs: list of lists of data-URI strings (each inner list = one "input")
        - breakdown: {"images": N, "pdf_pages": N, "video_frames": N}
        - hash: SHA256 hex of all input contents
    """
    path = Path(input_path)
    breakdown = {"images": 0, "pdf_pages": 0, "video_frames": 0}
    inputs: list[list[str]] = []
    hasher = hashlib.sha256()

    if path.is_dir():
        files = sorted(path.rglob("*"))
        files = [f for f in files if f.is_file()]
    elif path.is_file():
        files = [path]
    else:
        console.print(f"[red]Input not found: {input_path}[/red]")
        sys.exit(1)

    for f in files:
        ext = f.suffix.lower()

        with open(f, "rb") as fh:
            hasher.update(fh.read())

        if ext in IMAGE_EXTENSIONS:
            b64 = image_to_base64(f, max_size=max_size)
            inputs.append([b64])
            breakdown["images"] += 1
        elif ext in PDF_EXTENSIONS:
            pages = pdf_to_base64(f, max_size=max_size)
            for page in pages:
                inputs.append([page])
            breakdown["pdf_pages"] += len(pages)
        elif ext in VIDEO_EXTENSIONS:
            frames = video_to_base64_frames(f)
            if frames:
                inputs.append(frames)
                breakdown["video_frames"] += len(frames)
        # else: skip unsupported file types

    if not inputs:
        console.print("[red]No supported inputs found.[/red]")
        sys.exit(1)

    input_hash = f"sha256:{hasher.hexdigest()}"
    return inputs, breakdown, input_hash


# ── Benchmark Core ────────────────────────────────────────────────────────────


# Whether the backend supports stream_options (auto-detected on first call)
_stream_options_supported: bool | None = None


async def call_completion_async(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> dict[str, Any]:
    """Call the completion API with async streaming and measure timing.

    Returns dict with keys: ttft_ms, latency_s, prompt_tokens, completion_tokens,
    cached_tokens, reasoning_tokens, error
    """
    global _stream_options_supported

    t_start = time.perf_counter()
    ttft: float | None = None
    most_recent_ts: float | None = None
    itl_values: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    full_content = ""

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if _stream_options_supported is not False:
        kwargs["stream_options"] = {"include_usage": True}

    try:
        stream = await client.chat.completions.create(**kwargs)
    except Exception:
        if _stream_options_supported is None:
            _stream_options_supported = False
            kwargs.pop("stream_options", None)
            t_start = time.perf_counter()
            stream = await client.chat.completions.create(**kwargs)
        else:
            raise

    if _stream_options_supported is None:
        _stream_options_supported = True

    async for chunk in stream:
        now = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = (now - t_start) * 1000
            elif most_recent_ts is not None:
                itl_values.append((now - most_recent_ts) * 1000)
            most_recent_ts = now
            full_content += chunk.choices[0].delta.content

        if hasattr(chunk, "usage") and chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            pd = getattr(chunk.usage, "prompt_tokens_details", None)
            if pd is not None:
                cached_tokens = getattr(pd, "cached_tokens", 0) or 0
            cd = getattr(chunk.usage, "completion_tokens_details", None)
            if cd is not None:
                reasoning_tokens = getattr(cd, "reasoning_tokens", 0) or 0

    t_end = time.perf_counter()
    latency_s = t_end - t_start

    if completion_tokens == 0 and full_content:
        completion_tokens = len(full_content.split())

    return {
        "ttft_ms": ttft if ttft is not None else latency_s * 1000,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "itl_ms": itl_values,
        "error": None,
    }


def call_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> dict[str, Any]:
    """Synchronous call_completion for warmup only."""
    global _stream_options_supported

    t_start = time.perf_counter()
    ttft: float | None = None
    most_recent_ts: float | None = None
    itl_values: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    full_content = ""

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if _stream_options_supported is not False:
        kwargs["stream_options"] = {"include_usage": True}

    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception:
        if _stream_options_supported is None:
            _stream_options_supported = False
            kwargs.pop("stream_options", None)
            t_start = time.perf_counter()
            stream = client.chat.completions.create(**kwargs)
        else:
            raise

    if _stream_options_supported is None:
        _stream_options_supported = True

    for chunk in stream:
        now = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = (now - t_start) * 1000
            elif most_recent_ts is not None:
                itl_values.append((now - most_recent_ts) * 1000)
            most_recent_ts = now
            full_content += chunk.choices[0].delta.content

        if hasattr(chunk, "usage") and chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            pd = getattr(chunk.usage, "prompt_tokens_details", None)
            if pd is not None:
                cached_tokens = getattr(pd, "cached_tokens", 0) or 0
            cd = getattr(chunk.usage, "completion_tokens_details", None)
            if cd is not None:
                reasoning_tokens = getattr(cd, "reasoning_tokens", 0) or 0

    t_end = time.perf_counter()
    latency_s = t_end - t_start

    if completion_tokens == 0 and full_content:
        completion_tokens = len(full_content.split())

    return {
        "ttft_ms": ttft if ttft is not None else latency_s * 1000,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "itl_ms": itl_values,
        "error": None,
    }


def _uri_to_content_block(uri: str) -> dict:
    """Convert a URI (data URI, HTTP URL, or text: sentinel) to an OpenAI content block."""
    if uri.startswith(TEXT_URI_PREFIX):
        return {"type": "text", "text": uri[len(TEXT_URI_PREFIX) :]}
    return {"type": "image_url", "image_url": {"url": uri}}


def build_messages(prompt: str, image_uris: list[str]) -> list[dict]:
    """Build OpenAI-compatible messages.

    Supports visual inputs (base64 data URIs / HTTP URLs) and plain-text inputs
    (stored as ``text:<content>`` sentinels).  When ``image_uris`` is empty the
    message contains only the prompt text, enabling pure text-only benchmarks.
    """
    content: list[dict] = [_uri_to_content_block(uri) for uri in image_uris]
    if prompt:
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_embedding_messages(
    prompt: str,
    image_uris: list[str],
    *,
    system_instruction: str | None = None,
) -> list[dict]:
    """Build chat-style messages for the vLLM embeddings API.

    When system_instruction is set, uses the system/user/assistant pattern
    required by models like Qwen3-VL-Embedding.  Supports text: sentinels.
    """
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})

    content: list[dict] = [_uri_to_content_block(uri) for uri in image_uris]
    if prompt:
        content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    if system_instruction:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})

    return messages


async def call_embedding_async(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    continue_final_message: bool = False,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Call the vLLM embeddings API (async) and measure timing."""
    t_start = time.perf_counter()

    body: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "encoding_format": "float",
    }
    if continue_final_message:
        body["continue_final_message"] = True
    if add_special_tokens:
        body["add_special_tokens"] = True

    response: CreateEmbeddingResponse = await client.post(  # type: ignore[misc]
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body=body,
    )

    t_end = time.perf_counter()
    latency_s = t_end - t_start

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    embedding_dim = len(response.data[0].embedding) if response.data else 0

    return {
        "ttft_ms": None,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "embedding_dim": embedding_dim,
        "error": None,
    }


def call_embedding(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    continue_final_message: bool = False,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Synchronous call_embedding for warmup only."""
    t_start = time.perf_counter()

    body: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "encoding_format": "float",
    }
    if continue_final_message:
        body["continue_final_message"] = True
    if add_special_tokens:
        body["add_special_tokens"] = True

    response: CreateEmbeddingResponse = client.post(  # type: ignore[misc]
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body=body,
    )

    t_end = time.perf_counter()
    latency_s = t_end - t_start

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    embedding_dim = len(response.data[0].embedding) if response.data else 0

    return {
        "ttft_ms": None,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "embedding_dim": embedding_dim,
        "error": None,
    }


async def call_score_async(
    http_client: httpx.AsyncClient,
    rerank_url: str,
    model: str,
    query: str,
    image_uris: list[str],
) -> dict[str, Any]:
    """Call the vLLM /rerank API (async) and measure timing."""
    content: list[dict] = [{"type": "image_url", "image_url": {"url": uri}} for uri in image_uris]

    t_start = time.perf_counter()
    resp = await http_client.post(
        rerank_url,
        json={"model": model, "query": query, "documents": [{"content": content}]},
    )
    resp.raise_for_status()
    t_end = time.perf_counter()

    data = resp.json()
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

    return {
        "ttft_ms": None,
        "latency_s": t_end - t_start,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "embedding_dim": 0,
        "error": None,
    }


def call_score(
    rerank_url: str,
    model: str,
    query: str,
    image_uris: list[str],
) -> dict[str, Any]:
    """Synchronous call_score for warmup only."""
    content: list[dict] = [{"type": "image_url", "image_url": {"url": uri}} for uri in image_uris]

    t_start = time.perf_counter()
    resp = httpx.post(
        rerank_url,
        json={"model": model, "query": query, "documents": [{"content": content}]},
        timeout=300,
    )
    resp.raise_for_status()
    t_end = time.perf_counter()

    data = resp.json()
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

    return {
        "ttft_ms": None,
        "latency_s": t_end - t_start,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "embedding_dim": 0,
        "error": None,
    }


def compute_stats(values: list[float]) -> StatBlock:
    """Compute mean, p50, p95, p99 from a list of values."""
    if not values:
        return StatBlock(mean=0, p50=0, p95=0, p99=0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return StatBlock(
        mean=round(statistics.mean(sorted_vals), 2),
        p50=round(statistics.median(sorted_vals), 2),
        p95=round(sorted_vals[min(int(n * 0.95), n - 1)], 2),
        p99=round(sorted_vals[min(int(n * 0.99), n - 1)], 2),
    )


def compute_token_stat(values: list[int]) -> TokenStat:
    """Compute mean, min, max for token counts."""
    if not values:
        return TokenStat(mean=0, min=0, max=0)
    return TokenStat(
        mean=round(statistics.mean(values)),
        min=min(values),
        max=max(values),
    )


async def _call_with_retry_async(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    max_attempts: int = 5,
) -> tuple[dict[str, Any], int]:
    """Call completion with exponential-backoff retry, return (result, retry_count)."""
    for attempt in range(max_attempts):
        try:
            result = await call_completion_async(client, model, messages, max_tokens)
            return result, attempt
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(2**attempt, 60))
    raise RuntimeError("unreachable")


async def _call_embedding_with_retry_async(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    continue_final_message: bool = False,
    add_special_tokens: bool = False,
    max_attempts: int = 5,
) -> tuple[dict[str, Any], int]:
    """Call embedding with exponential-backoff retry, return (result, retry_count)."""
    for attempt in range(max_attempts):
        try:
            result = await call_embedding_async(
                client,
                model,
                messages,
                continue_final_message=continue_final_message,
                add_special_tokens=add_special_tokens,
            )
            return result, attempt
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(2**attempt, 60))
    raise RuntimeError("unreachable")


async def _call_score_with_retry_async(
    http_client: httpx.AsyncClient,
    rerank_url: str,
    model: str,
    query: str,
    image_uris: list[str],
    max_attempts: int = 5,
) -> tuple[dict[str, Any], int]:
    """Call score/rerank with exponential-backoff retry, return (result, retry_count)."""
    for attempt in range(max_attempts):
        try:
            result = await call_score_async(http_client, rerank_url, model, query, image_uris)
            return result, attempt
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 503, 504) or attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(2**attempt, 60))
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(2**attempt, 60))
    raise RuntimeError("unreachable")


async def run_benchmark(
    client: AsyncOpenAI,
    model: str,
    inputs: list[list[str]],
    prompt: str | list[str],
    max_tokens: int,
    runs: int,
    max_concurrency: int,
    warmup: int = 1,
    progress_callback: Any = None,
    task: str = "completion",
    embedding_options: dict[str, Any] | None = None,
) -> tuple[list[RunRaw], int]:
    """Run the benchmark with true async concurrency: warmup + N timed runs.

    Uses an asyncio.Semaphore to cap in-flight requests at max_concurrency,
    keeping the GPU pipeline saturated without overwhelming the server.
    Returns (runs_raw, retries_count).
    """
    runs_raw: list[RunRaw] = []
    retries = 0
    retries_lock = asyncio.Lock()
    is_embedding = task == "embedding"
    is_score = task == "score"
    eopts = embedding_options or {}
    emb_continue = eopts.get("continue_final_message", False)
    emb_special = eopts.get("add_special_tokens", False)
    emb_system = eopts.get("system_instruction")

    # For score/rerank tasks, derive the rerank URL from the OpenAI client base
    rerank_url = f"{str(client.base_url).rstrip('/')}/rerank" if is_score else ""
    score_http: httpx.AsyncClient | None = None

    def _build_msgs(p: str, uris: list[str]) -> list[dict]:
        if is_embedding:
            return build_embedding_messages(p, uris, system_instruction=emb_system)
        return build_messages(p, uris)

    # Synchronous warmup via a temporary sync client
    sync_client = OpenAI(base_url=str(client.base_url), api_key=client.api_key)
    for w in range(warmup):
        if progress_callback:
            progress_callback("warmup", w + 1, 0)
        try:
            warmup_prompt = prompt[0] if isinstance(prompt, list) else prompt
            if is_score:
                call_score(rerank_url, model, warmup_prompt, inputs[0])
            elif is_embedding:
                messages = _build_msgs(warmup_prompt, inputs[0])
                call_embedding(
                    sync_client,
                    model,
                    messages,
                    continue_final_message=emb_continue,
                    add_special_tokens=emb_special,
                )
            else:
                messages = _build_msgs(warmup_prompt, inputs[0])
                call_completion(sync_client, model, messages, max_tokens)
        except Exception as e:
            err_str = str(e).lower()
            if any(
                kw in err_str
                for kw in ("not found", "404", "does not exist", "invalid model", "401", "403", "unauthorized")
            ):
                console.print()
                print_error("Warmup failed", str(e))
                sys.exit(1)
            if w == 0:
                console.print(f"  [yellow]Warmup warning: {e}[/yellow]")

    sem = asyncio.Semaphore(max_concurrency)
    if is_score:
        score_http = httpx.AsyncClient(timeout=httpx.Timeout(300))

    async def execute_single(input_idx: int, run_num: int) -> RunRaw:
        nonlocal retries
        input_prompt = prompt[input_idx] if isinstance(prompt, list) else prompt
        async with sem:
            try:
                if is_score:
                    assert score_http is not None
                    result, retry_count = await _call_score_with_retry_async(
                        score_http,
                        rerank_url,
                        model,
                        input_prompt,
                        inputs[input_idx],
                    )
                elif is_embedding:
                    messages = _build_msgs(input_prompt, inputs[input_idx])
                    result, retry_count = await _call_embedding_with_retry_async(
                        client,
                        model,
                        messages,
                        continue_final_message=emb_continue,
                        add_special_tokens=emb_special,
                    )
                else:
                    messages = _build_msgs(input_prompt, inputs[input_idx])
                    result, retry_count = await _call_with_retry_async(client, model, messages, max_tokens)
                if retry_count:
                    async with retries_lock:
                        retries += retry_count
                return RunRaw(
                    input_idx=input_idx,
                    run=run_num,
                    ttft_ms=round(result["ttft_ms"], 1) if result["ttft_ms"] is not None else None,
                    latency_s=round(result["latency_s"], 3),
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    cached_tokens=result["cached_tokens"],
                    reasoning_tokens=result["reasoning_tokens"],
                    embedding_dim=result.get("embedding_dim", 0),
                    itl_ms=[round(v, 2) for v in result.get("itl_ms", [])],
                    error=result["error"],
                )
            except Exception as e:
                return RunRaw(
                    input_idx=input_idx,
                    run=run_num,
                    ttft_ms=None,
                    latency_s=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    error=str(e),
                )

    completed = 0

    try:
        for run_num in range(1, runs + 1):
            if progress_callback:
                progress_callback("run", run_num, completed)

            tasks = [asyncio.create_task(execute_single(i, run_num)) for i in range(len(inputs))]

            error_count_run = 0
            for coro in asyncio.as_completed(tasks):
                run_result = await coro
                runs_raw.append(run_result)
                completed += 1
                if run_result.error:
                    error_count_run += 1
                if progress_callback:
                    progress_callback("progress", run_num, completed)

            if error_count_run == len(tasks):
                console.print()
                first_err = next(r.error for r in runs_raw if r.error)
                print_error("Benchmark aborted", f"All requests failed: {first_err}")
                sys.exit(1)
            elif error_count_run > 0:
                console.print(
                    f"  [yellow]Run {run_num}: {error_count_run}/{len(tasks)} samples failed (skipped)[/yellow]"
                )
    finally:
        if score_http:
            await score_http.aclose()

    return runs_raw, retries


# ── Display ───────────────────────────────────────────────────────────────────


def print_banner() -> None:
    """Print the vlmbench banner with a silver→blue gradient."""
    console.print()
    if console.width >= 50:
        art_lines: list[str] = []
        tag_lines: list[str] = []
        for line in (__doc__ or "").strip().splitlines():
            stripped = line.strip()
            if stripped and stripped[0] in ("█", "╚"):
                art_lines.append(line)
            elif stripped:
                tag_lines.append(stripped)

        # VLM (first 6) and BENCH (next 6) left-aligned
        _VLM_PAD = "  "
        _BENCH_PAD = "  "

        # Silver (#C0D0E8) → Steel blue (#4682B4)
        n = len(art_lines)
        s, e = (0xC0, 0xD0, 0xE8), (0x46, 0x82, 0xB4)
        for i, line in enumerate(art_lines):
            pad = _VLM_PAD if i < 6 else _BENCH_PAD
            t = i / max(n - 1, 1)
            r = int(s[0] + (e[0] - s[0]) * t)
            g = int(s[1] + (e[1] - s[1]) * t)
            b = int(s[2] + (e[2] - s[2]) * t)
            console.print(Text(f"{pad}{line}", style=f"#{r:02x}{g:02x}{b:02x}"))

        console.print(
            f"  [bold {STEEL_BLUE}]vlmbench (v{VERSION})[/bold {STEEL_BLUE}]"
            " — Single-file, drop-in VLM benchmark CLI for your agents."
        )
        console.print(
            f"  by [bold {STEEL_BLUE}][link=https://vlm.run]VLM Run[/link][/bold {STEEL_BLUE}] · https://vlm.run"
        )
    else:
        console.print(f"  [bold {STEEL_BLUE}]vlmbench (v{VERSION})[/bold {STEEL_BLUE}]")
        console.print("  by [link=https://vlm.run]VLM Run[/link]")
    console.print()


def print_config(
    model_id: str,
    revision: str,
    quant: str | None,
    base_url: str,
    backend: str,
    backend_version: str | None,
    env: EnvironmentInfo,
    total_inputs: int,
    breakdown: dict[str, int],
    input_path: str,
    max_tokens: int,
    runs: int,
    max_concurrency: int,
    tmux_session: str | None = None,
) -> None:
    """Print the configuration block in a Rich Panel."""
    tbl = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    tbl.add_column("key", style="bold cyan", no_wrap=True, width=12)
    tbl.add_column("val")

    # Model
    quant_str = f" ({quant})" if quant and quant != "auto" else ""
    tbl.add_row("model", f"{model_id}{quant_str}")
    tbl.add_row("revision", revision)

    # Backend
    backend_display = backend.capitalize() if backend != "vllm" else "vLLM"
    version_str = f" {backend_version}" if backend_version else ""
    tbl.add_row("backend", f"{backend_display}{version_str}")
    tbl.add_row("endpoint", base_url)

    # Blank separator
    tbl.add_row("", "")

    # Hardware
    if env.gpu_name:
        tbl.add_row("gpu", env.gpu_name)
        if env.gpu_vram_mib:
            if env.accelerator == "metal":
                tbl.add_row("vram", f"{env.gpu_vram_mib:,} MiB ({env.gpu_vram_mib // 1024} GB unified)")
            else:
                tbl.add_row("vram", f"{env.gpu_vram_mib:,} MiB")
        if env.gpu_driver:
            tbl.add_row("driver", env.gpu_driver)
    elif env.cpu:
        tbl.add_row("cpu", env.cpu)

    # Blank separator
    tbl.add_row("", "")

    # Dataset / inputs
    tbl.add_row("dataset", input_path)
    parts = []
    if breakdown.get("images"):
        parts.append(f"{breakdown['images']} images")
    if breakdown.get("pdf_pages"):
        parts.append(f"{breakdown['pdf_pages']} PDF pages")
    if breakdown.get("video_frames"):
        parts.append(f"{breakdown['video_frames']} video frames")
    if breakdown.get("text_inputs"):
        parts.append(f"{breakdown['text_inputs']} text inputs")
    if breakdown.get("hf_images"):
        parts.append(f"{breakdown['hf_images']} images")
    breakdown_str = ", ".join(parts) if parts else str(total_inputs)
    tbl.add_row("inputs", f"{total_inputs} ({breakdown_str})")

    # Blank separator
    tbl.add_row("", "")

    # Config
    tbl.add_row("max_tokens", str(max_tokens))
    tbl.add_row("runs", str(runs))
    tbl.add_row("concurrency", str(max_concurrency))

    # Monitor
    if tmux_session:
        tbl.add_row("", "")
        tbl.add_row("monitor", f"tmux attach -t {tmux_session}")

    panel = Panel(
        tbl,
        title="[bold]Configuration[/bold]",
        title_align="left",
        border_style=STEEL_BLUE,
        box=box.ROUNDED,
        padding=(1, 1),
    )
    console.print(panel)


def _format_mpixels(pixels: int) -> str:
    """Format pixel count as megapixels."""
    mp = pixels / 1_000_000
    return f"{mp:.2f} MP" if mp >= 0.01 else f"{pixels:,} px"


def print_results(result: BenchmarkResult, save_path: str, *, title_suffix: str = "") -> None:
    """Print the results card as a clean tabular Rich Panel."""
    r = result.results
    concurrency = result.input.max_concurrency
    img_stats = result.input.image_stats
    _H = "dim"

    def _tok_style(val: float) -> str:
        if val >= 50:
            return "bold green"
        if val >= 20:
            return "green"
        if val >= 10:
            return "yellow"
        return "red"

    is_embedding = r.embedding_dim is not None
    is_score = result.input.task == "score"
    is_pooling = is_embedding or is_score

    # ── Throughput section ────────────────────────────────────────────────
    throughput_tbl = Table(
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    throughput_tbl.add_column("Metric", style="bold cyan", no_wrap=True, width=20)
    throughput_tbl.add_column("Value", style="bold", no_wrap=True, width=16)
    throughput_tbl.add_column("p50", style=_H, justify="right", width=10)
    throughput_tbl.add_column("p95", style=_H, justify="right", width=10)
    throughput_tbl.add_column("p99", style=_H, justify="right", width=10)

    if is_pooling:
        unit = "emb/s" if is_embedding else "score/s"
        throughput_tbl.add_row("Throughput", f"{r.inputs_per_sec:.2f} {unit}", "—", "—", "—")
        if is_embedding and r.embedding_dim:
            throughput_tbl.add_row("Embedding dim", str(r.embedding_dim), "—", "—", "—")
    else:
        throughput_tbl.add_row("Throughput", f"{r.inputs_per_sec:.2f} req/s", "—", "—", "—")
        throughput_tbl.add_row(
            "Output tok/s",
            Text(f"{r.tokens_per_sec:.1f} tok/s", style=_tok_style(r.tokens_per_sec)),
            "—",
            "—",
            "—",
        )
        throughput_tbl.add_row("Total tok/s", f"{r.total_tokens_per_sec:.1f} tok/s", "—", "—", "—")

    # ── Latency section ───────────────────────────────────────────────────
    latency_tbl = Table(
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    latency_tbl.add_column("Metric", style="bold cyan", no_wrap=True, width=20)
    latency_tbl.add_column("Mean", style="bold", no_wrap=True, width=16)
    latency_tbl.add_column("p50", style=_H, justify="right", width=10)
    latency_tbl.add_column("p95", style=_H, justify="right", width=10)
    latency_tbl.add_column("p99", style=_H, justify="right", width=10)

    if is_pooling:
        lat_unit = "s/emb" if is_embedding else "s/score"
        latency_tbl.add_row(
            "E2E Latency",
            f"{r.latency_s_per_input.mean:.2f} {lat_unit}",
            f"{r.latency_s_per_input.p50:.2f} s",
            f"{r.latency_s_per_input.p95:.2f} s",
            f"{r.latency_s_per_input.p99:.2f} s",
        )
    else:

        def _lat_row(label: str, sb: StatBlock, fmt: str = ".0f", unit: str = "ms") -> None:
            if sb.mean == 0:
                latency_tbl.add_row(label, "-", "-", "-", "-")
            else:
                latency_tbl.add_row(
                    label,
                    f"{sb.mean:{fmt}} {unit}",
                    f"{sb.p50:{fmt}} {unit}",
                    f"{sb.p95:{fmt}} {unit}",
                    f"{sb.p99:{fmt}} {unit}",
                )

        _lat_row("TTFT", r.ttft_ms, ".0f", "ms")
        _lat_row("TPOT", r.tpot_ms, ".1f", "ms")
        _lat_row("ITL", r.itl_ms, ".1f", "ms")
        _lat_row("E2E Latency", r.e2el_ms, ".0f", "ms")

    # ── Details section ───────────────────────────────────────────────────
    details_tbl = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    details_tbl.add_column("Metric", style="bold cyan", no_wrap=True, width=20)
    details_tbl.add_column("Value", style="bold", no_wrap=True)

    if is_pooling:
        tok_val = f"prompt {r.prompt_tokens.mean:,}"
        details_tbl.add_row("Tokens (avg)", tok_val)
        tok_range = f"prompt {r.prompt_tokens.min:,}–{r.prompt_tokens.max:,}"
        details_tbl.add_row("Token ranges", tok_range)
    else:
        tok_val = f"prompt {r.prompt_tokens.mean:,}  •  completion {r.completion_tokens.mean:,}"
        details_tbl.add_row("Tokens (avg)", tok_val)
        tok_range = f"prompt {r.prompt_tokens.min:,}–{r.prompt_tokens.max:,}  •  completion {r.completion_tokens.min:,}–{r.completion_tokens.max:,}"
        if r.cached_tokens is not None and r.cached_tokens.mean > 0:
            tok_range += f"  •  {r.cached_tokens.mean:,} cached"
        if r.reasoning_tokens is not None and r.reasoning_tokens.mean > 0:
            tok_range += f"  •  {r.reasoning_tokens.mean:,} reasoning"
        details_tbl.add_row("Token ranges", tok_range)

    if img_stats and img_stats.count > 0:
        img_val = f"{img_stats.count:,}  •  avg {img_stats.avg_width}×{img_stats.avg_height} ({_format_mpixels(img_stats.avg_pixels)})"
        details_tbl.add_row("Images", img_val)
        res_val = f"min {img_stats.min_width}×{img_stats.min_height}  •  median {img_stats.median_width}×{img_stats.median_height}  •  max {img_stats.max_width}×{img_stats.max_height}"
        details_tbl.add_row("Resolution", res_val)

    if r.vram_peak_mib is not None:
        vram_gb = r.vram_peak_mib / 1024
        details_tbl.add_row("VRAM peak", f"{vram_gb:.1f} GB")

    total_ok = r.total_inputs_processed - r.errors
    if r.errors == 0:
        rel_text = Text(f"{total_ok}/{r.total_inputs_processed} ok", style="bold green")
    else:
        rel_text = Text()
        rel_text.append(f"{total_ok}/{r.total_inputs_processed} ok", style="bold yellow")
        rel_text.append(f", {r.errors} errors", style="red")
    if r.retries > 0:
        rel_text.append(f", {r.retries} retries", style="yellow")
    rel_text.append(f"  •  {r.total_duration_s:.1f}s total", style="dim")
    details_tbl.add_row("Reliability", rel_text)

    from rich.console import Group as RenderGroup

    body = RenderGroup(throughput_tbl, Text(), latency_tbl, Text(), details_tbl)

    title = f"[bold]Results{title_suffix}[/bold]"
    subtitle = f"[dim]workers: {concurrency}[/dim]"
    panel = Panel(
        body,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 1),
    )
    console.print(panel)
    console.print(f"  [green bold]>[/green bold] Saved [dim]->[/dim] {save_path}")
    console.print()


def print_concurrency_table(results: list[BenchmarkResult], saved_paths: list[str]) -> None:
    """Print a consolidated table for a concurrency sweep."""
    max_toks = max(r.results.tokens_per_sec for r in results)
    _UP = "\u2191"  # higher is better
    _DN = "\u2193"  # lower is better

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.ROUNDED,
        border_style="dim white",
        padding=(0, 1),
    )
    table.add_column("Workers", justify="right", min_width=7)
    table.add_column("Samples", justify="right", min_width=9)
    table.add_column(f"Tok/s {_UP}\n", justify="right", min_width=7)
    table.add_column(f"Img/s {_UP}\n", justify="right", min_width=7)
    table.add_column(f"TTFT {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"TPOT {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"ITL {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"E2EL {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"Duration {_DN}\n(s)", justify="right", min_width=8)
    table.add_column(f"VRAM {_DN}\n", justify="right", min_width=9)

    for r in results:
        c = r.input.max_concurrency
        toks = r.results.tokens_per_sec
        is_best = toks == max_toks
        style = "bold bright_green" if is_best else ""
        vram_gb = f"{r.results.vram_peak_mib / 1024:.1f} GB" if r.results.vram_peak_mib is not None else "-"
        total_ok = r.results.total_inputs_processed - r.results.errors
        samples_text = f"{total_ok}/{r.results.total_inputs_processed}"
        samples_style = "red" if r.results.errors > 0 else style

        def _ms_cell(sb: StatBlock, fmt: str = ".0f") -> Text:
            return Text("-", style="dim") if sb.mean == 0 else Text(f"{sb.mean:{fmt}}", style=style)

        table.add_row(
            Text(str(c), style=style),
            Text(samples_text, style=samples_style),
            Text(f"{toks:.1f}", style=style),
            Text(f"{r.results.inputs_per_sec:.2f}", style=style),
            _ms_cell(r.results.ttft_ms, ".0f"),
            _ms_cell(r.results.tpot_ms, ".1f"),
            _ms_cell(r.results.itl_ms, ".1f"),
            _ms_cell(r.results.e2el_ms, ".0f"),
            Text(f"{r.results.total_duration_s:.1f}", style=style),
            Text(vram_gb, style=style),
        )

    panel = Panel(
        table,
        title="[bold]Concurrency Sweep[/bold]",
        title_align="left",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 1),
    )
    console.print(panel)

    for path in saved_paths:
        console.print(f"  [green bold]>[/green bold] Saved [dim]->[/dim] {path}")
    console.print()


def print_compare_table(
    results: list[BenchmarkResult],
    *,
    max_rows_per_model: int | None = None,
    sort_by: str = "img/s",
) -> None:
    """Print the compare table grouped by model, sorted by the chosen metric descending."""
    console.print(f"  [bold]compare[/bold] [dim]({len(results)} runs, sorted by {sort_by})[/dim]")
    console.print()

    _UP = "\u2191"
    _DN = "\u2193"
    _BEST = "bold bright_green"

    def _total_tok_s(r: BenchmarkResult) -> float:
        return r.results.tokens_per_sec

    def _be_label(r: BenchmarkResult) -> str:
        be, ver = r.environment.backend, r.environment.backend_version or ""
        return f"vLLM {ver}".strip() if be == "vllm" else f"{be.capitalize()} {ver}".strip()

    def _img_s(r: BenchmarkResult) -> float:
        return r.results.inputs_per_sec

    _sort_key = _img_s if sort_by == "img/s" else _total_tok_s

    # Group by model_id, sort groups by best of chosen metric descending
    sorted_results = sorted(results, key=lambda r: (r.model.model_id, -_sort_key(r)))
    groups: list[tuple[str, list[BenchmarkResult]]] = [
        (model_id, list(runs)) for model_id, runs in groupby(sorted_results, key=lambda r: r.model.model_id)
    ]
    groups.sort(key=lambda g: max(_sort_key(r) for r in g[1]), reverse=True)

    # Global bests per metric (for per-cell highlighting); ignore zeros
    ttft_vals = [r.results.ttft_ms.mean for r in results if r.results.ttft_ms.mean > 0]
    tpot_vals = [r.results.tpot_ms.mean for r in results if r.results.tpot_ms.mean > 0]
    itl_vals = [r.results.itl_ms.mean for r in results if r.results.itl_ms.mean > 0]
    e2el_vals = [r.results.e2el_ms.mean for r in results if r.results.e2el_ms.mean > 0]
    best_ttft = min(ttft_vals) if ttft_vals else 0.0
    best_tpot = min(tpot_vals) if tpot_vals else 0.0
    best_itl = min(itl_vals) if itl_vals else 0.0
    best_e2el = min(e2el_vals) if e2el_vals else 0.0
    best_toks = max(_total_tok_s(r) for r in results)
    best_imgs = max(r.results.inputs_per_sec for r in results)
    best_dur = min(r.results.total_duration_s for r in results)
    vram_vals = [r.results.vram_peak_mib for r in results if r.results.vram_peak_mib is not None]
    best_vram = min(vram_vals) if vram_vals else None

    all_quants = {r.model.quant or "-" for r in results}
    show_quant = len(all_quants) > 1

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.ROUNDED,
        border_style="dim white",
        padding=(0, 1),
    )
    model_width = min(max(len(r.model.model_id) for r in results), 80)
    table.add_column("Model", width=model_width, no_wrap=True, style=f"bold {STEEL_BLUE}")
    table.add_column(f"TTFT {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"TPOT {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"ITL {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"E2EL {_DN}\n(ms)", justify="right", min_width=8)
    table.add_column(f"Tok/s {_UP}\n", justify="right", min_width=7)
    table.add_column(f"Img/s {_UP}\n", justify="right", min_width=6)
    table.add_column(f"Duration {_DN}\n(s)", justify="right", min_width=8)
    table.add_column("Workers", justify="right", min_width=4, style="dim")
    table.add_column(f"VRAM {_DN}\n", justify="right", min_width=9)
    if show_quant:
        table.add_column("Quant", min_width=6, style="dim")
    table.add_column("Backend", min_width=7, style="dim")
    table.add_column("Hardware", min_width=14, style="dim")

    for group_idx, (model_id, runs) in enumerate(groups):
        if group_idx > 0:
            table.add_section()

        visible_runs = runs[:max_rows_per_model] if max_rows_per_model else runs
        for row_idx, r in enumerate(visible_runs):
            total_toks = _total_tok_s(r)
            total_imgs = r.results.inputs_per_sec
            vram_mib = r.results.vram_peak_mib
            vram_gb = f"{vram_mib / 1024:.1f} GB" if vram_mib is not None else "-"

            model_cell = model_id if row_idx == 0 else ""

            ttft_v = r.results.ttft_ms.mean
            tpot_v = r.results.tpot_ms.mean
            itl_v = r.results.itl_ms.mean
            e2el_v = r.results.e2el_ms.mean

            def _cmp_cell(val: float, best: float, fmt: str = ".0f") -> Text:
                if val == 0:
                    return Text("-", style="dim")
                return Text(f"{val:{fmt}}", style=_BEST if val == best else "dim")

            cells: list[Any] = [
                model_cell,
                _cmp_cell(ttft_v, best_ttft, ".0f"),
                _cmp_cell(tpot_v, best_tpot, ".1f"),
                _cmp_cell(itl_v, best_itl, ".1f"),
                _cmp_cell(e2el_v, best_e2el, ".0f"),
                Text(f"{total_toks:.1f}", style=_BEST if total_toks == best_toks else "white"),
                Text(f"{total_imgs:.2f}", style=_BEST if total_imgs == best_imgs else "dim"),
                Text(
                    f"{r.results.total_duration_s:.1f}",
                    style=_BEST if r.results.total_duration_s == best_dur else "dim",
                ),
                str(r.input.max_concurrency),
                Text(
                    vram_gb,
                    style=_BEST if vram_mib is not None and best_vram is not None and vram_mib == best_vram else "dim",
                ),
            ]
            if show_quant:
                cells.append(r.model.quant or "-")
            cells.append(_be_label(r))
            hw = (r.environment.gpu_name or "-").replace("NVIDIA ", "")
            cells.append(hw[:20] + "…" if len(hw) > 20 else hw)

            table.add_row(*cells)

    table_width = console.measure(table).maximum
    console.print(table)
    console.print()

    # ── Results leaderboard ────────────────────────────────────────────────
    total_duration = sum(r.results.total_duration_s for r in results)
    total_errors = sum(r.results.errors for r in results)
    total_retries = sum(r.results.retries for r in results)

    lb = Table(show_header=True, header_style="bold white", box=None, padding=(0, 1), show_edge=False)
    lb.add_column("#", justify="right", style="dim", width=3)
    lb.add_column("Model", no_wrap=True, style=f"bold {STEEL_BLUE}", min_width=20)
    lb.add_column(f"Best Img/s {_UP}\n", justify="right", min_width=10)
    lb.add_column(f"Best Tok/s {_UP}\n", justify="right", min_width=10)
    lb.add_column("@ Workers\n", justify="right", style="dim", min_width=9)
    lb.add_column(f"TTFT {_DN}\n", justify="right", min_width=8)
    lb.add_column(f"TPOT {_DN}\n", justify="right", min_width=8)
    lb.add_column(f"ITL {_DN}\n", justify="right", min_width=8)
    lb.add_column(f"E2EL {_DN}\n", justify="right", min_width=8)

    for rank, (model_id, runs) in enumerate(groups, 1):
        best_run = max(runs, key=_sort_key)
        br = best_run.results
        img_style = _BEST if _img_s(best_run) == best_imgs else "white"
        tok_style = _BEST if _total_tok_s(best_run) == best_toks else "white"

        lb.add_row(
            str(rank),
            model_id,
            Text(f"{_img_s(best_run):.2f}", style=img_style),
            Text(f"{_total_tok_s(best_run):.1f}", style=tok_style),
            str(best_run.input.max_concurrency),
            "-" if br.ttft_ms.mean == 0 else f"{br.ttft_ms.mean:.0f} ms",
            "-" if br.tpot_ms.mean == 0 else f"{br.tpot_ms.mean:.1f} ms",
            "-" if br.itl_ms.mean == 0 else f"{br.itl_ms.mean:.1f} ms",
            "-" if br.e2el_ms.mean == 0 else f"{br.e2el_ms.mean:.0f} ms",
        )

    from rich.console import Group as RenderGroup

    footer = Text()
    footer.append(f"\n   {len(results)} runs", style="white")
    footer.append(f" across {len(groups)} model(s)", style="dim")
    footer.append(f"  \u00b7  total {total_duration:.1f}s", style="dim")
    error_style = "bright_green" if total_errors == 0 else "yellow"
    footer.append("  \u00b7  ", style="dim")
    footer.append(f"{total_errors} errors", style=error_style)
    if total_retries > 0:
        footer.append(f" ({total_retries} retries)", style="dim")

    panel = Panel(
        RenderGroup(lb, footer),
        title="[bold]Results[/bold]",
        title_align="left",
        subtitle=f"[dim]vlmbench v{VERSION}[/dim]",
        subtitle_align="right",
        border_style="dim white",
        box=box.ROUNDED,
        padding=(1, 0),
        width=table_width,
    )
    console.print(panel)
    console.print()


# ── CLI ───────────────────────────────────────────────────────────────────────

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


def load_profile(name: str) -> dict[str, Any]:
    """Load a model profile from vlmbench/profiles/<name>.yaml."""
    yaml_path = PROFILES_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
        console.print(f"[red]Profile '{name}' not found.[/red]")
        if available:
            console.print(f"  Available: {', '.join(available)}")
        sys.exit(1)
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _run_profile_setup(setup: str) -> None:
    """Execute a profile's inline setup script."""
    console.print(f"  [dim]$ {setup.strip().splitlines()[0]}…[/dim]")
    result = subprocess.run(["bash", "-c", setup])
    if result.returncode != 0:
        console.print(f"[red]Profile setup failed (exit {result.returncode})[/red]")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vlmbench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run subcommand ──
    run_parser = subparsers.add_parser("run", help="Run a VLM benchmark.")

    # Model (required unless --profile is set; validated in cmd_run)
    run_parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model ID (vLLM: Qwen/Qwen3-VL-2B-Instruct, Ollama: qwen3-vl:2b)",
    )
    run_parser.add_argument(
        "--profile",
        default=None,
        help="Model profile name (e.g. glm-ocr). See: vlmbench profiles",
    )

    # Input source (mutually exclusive: --input OR --dataset, defaults to sample image URL)
    input_group = run_parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input",
        "-i",
        dest="input_path",
        help="Local file, directory, or URL (images, PDFs, videos)",
    )
    input_group.add_argument(
        "--dataset",
        "-d",
        dest="dataset",
        help="HuggingFace dataset (e.g., vlm-run/FineVision-vlmbench-mini or org/name:config)",
    )
    run_parser.add_argument(
        "--dataset-image-col",
        default=None,
        help="Image column name in dataset (auto-detected if not specified)",
    )
    run_parser.add_argument(
        "--dataset-text-col",
        default=None,
        help="Text column name in dataset to use as prompt/document input (e.g. 'prompt', 'text')",
    )
    run_parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use (default: train)",
    )

    # Server
    run_parser.add_argument(
        "--backend",
        default="auto",
        help="Backend: auto, ollama, vllm (native), vllm-openai:<tag> (Docker), sglang:<tag>, etc.",
    )
    run_parser.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-start inference server if none detected",
    )
    run_parser.add_argument("--serve-args", default=None, help="Extra CLI args for the server command")
    run_parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    run_parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY), help="API key")

    # Model config
    run_parser.add_argument("--revision", default="main", help="Model revision (metadata)")
    run_parser.add_argument(
        "--quant", default="auto", help="Quantization: fp16, bf16, awq, gptq, fp8, q4_0, q4_K_M, etc."
    )
    run_parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with each input")
    run_parser.add_argument(
        "--task",
        default="completion",
        choices=["completion", "embedding", "score"],
        help="Task type: completion (default), embedding, or score/rerank (vLLM pooling models)",
    )
    run_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens")
    run_parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help=f"Max image dimension in pixels; larger images are downscaled (default: {DEFAULT_MAX_IMAGE_SIZE})",
    )

    # Benchmark
    run_parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Timed runs per input")
    run_parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (not recorded)")
    run_parser.add_argument(
        "--concurrency",
        default=str(DEFAULT_CONCURRENCY),
        help="Concurrency level or comma-separated sweep (e.g. 8 or 4,8,16,32,64)",
    )
    run_parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit number of input samples (for dry-runs)"
    )

    # Output
    run_parser.add_argument("--output-directory", default=DEFAULT_SAVE_DIR, help="Output directory")
    run_parser.add_argument("--tag", default=None, help="Custom label (used in result filename and metadata)")
    run_parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload results to a HuggingFace dataset repo (requires HF_TOKEN)",
    )
    run_parser.add_argument(
        "--upload-repo",
        default=DEFAULT_UPLOAD_REPO,
        help=f"HuggingFace dataset repo for uploads (default: {DEFAULT_UPLOAD_REPO})",
    )

    # ── compare subcommand ──
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results from multiple JSON files.")
    compare_parser.add_argument("files", nargs="*", help=f"JSON result files (default: all in {DEFAULT_SAVE_DIR}/)")
    compare_parser.add_argument(
        "--max-rows-per-model",
        type=int,
        default=None,
        help="Limit number of rows shown per model (default: all)",
    )
    compare_parser.add_argument(
        "--sort-by",
        choices=["img/s", "tok/s"],
        default="img/s",
        help="Metric to sort results by (default: img/s)",
    )

    # ── profiles subcommand ──
    subparsers.add_parser("profiles", help="List available model profiles.")

    return parser


def _build_save_path(
    save_dir: Path,
    env: EnvironmentInfo,
    model: str,
    tag: str | None,
) -> Path:
    """Build the result JSON save path from environment, model, and tag."""
    backend_slug = env.backend.lower()
    ver = env.backend_version
    version_slug = f"v{ver}" if ver and not ver.startswith("v") else (ver or "")
    model_slug = re.sub(r"[^a-zA-Z0-9]", "-", model.split("/")[-1]).strip("-").lower()
    gpu_slug = re.sub(r"[^a-zA-Z0-9]", "-", (env.gpu_name or "")).strip("-").lower() if env.gpu_name else ""
    gpu_slug = re.sub(r"-+", "-", gpu_slug.removeprefix("nvidia-"))
    tag_slug = tag or ""
    parts = [backend_slug, version_slug, model_slug, gpu_slug, tag_slug]
    filename = "-".join(p for p in parts if p) + ".json"
    return save_dir / filename


def _execute_benchmark(
    args: argparse.Namespace,
    client: AsyncOpenAI,
    env: EnvironmentInfo,
    inputs: list[list[str]],
    input_hash: str,
    image_stats: ImageStats | None,
    breakdown: dict[str, int],
    quant_resolved: str | None,
    server_gpu_idx: int | None,
    max_concurrency: int,
    tag: str | None,
    label: str | None = None,
) -> tuple[BenchmarkResult, Path]:
    """Run the benchmark at a single concurrency level. Returns (result, save_path)."""
    total_inputs = len(inputs)
    total_tasks = total_inputs * args.runs
    task_type = getattr(args, "task", "completion")
    embedding_options = getattr(args, "embedding_options", None)

    desc_prefix = f"[{label}] " if label else ""
    progress_unit = {"embedding": "embeddings", "score": "scores"}.get(task_type, "images")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} " + progress_unit),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"{desc_prefix}Warmup...", total=total_tasks)

        def progress_callback(phase: str, run_num: int, completed: int) -> None:
            if phase == "warmup":
                progress.update(task_id, description=f"{desc_prefix}Warmup...")
            elif phase == "run":
                progress.update(task_id, description=f"{desc_prefix}Run {run_num}/{args.runs}")
            elif phase == "progress":
                progress.update(task_id, completed=completed)

        t_start = time.perf_counter()
        runs_raw, retries = asyncio.run(
            run_benchmark(
                client=client,
                model=args.model,
                inputs=inputs,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                runs=args.runs,
                max_concurrency=max_concurrency,
                warmup=args.warmup,
                progress_callback=progress_callback,
                task=task_type,
                embedding_options=embedding_options,
            )
        )
        t_end = time.perf_counter()

    total_duration_s = round(t_end - t_start, 1)

    successful_runs = [r for r in runs_raw if r.error is None]
    error_count = len(runs_raw) - len(successful_runs)

    ttft_values = [r.ttft_ms for r in successful_runs if r.ttft_ms is not None]
    latency_values = [r.latency_s for r in successful_runs]
    prompt_token_values = [r.prompt_tokens for r in successful_runs if r.prompt_tokens > 0]
    completion_token_values = [r.completion_tokens for r in successful_runs if r.completion_tokens > 0]
    cached_token_values = [r.cached_tokens for r in successful_runs]
    reasoning_token_values = [r.reasoning_tokens for r in successful_runs]

    tpot_values = []
    for r in successful_runs:
        if r.completion_tokens > 1 and r.ttft_ms is not None:
            total_gen_time_s = r.latency_s - (r.ttft_ms / 1000.0)
            if total_gen_time_s > 0:
                tpot_ms = (total_gen_time_s / (r.completion_tokens - 1)) * 1000.0
                tpot_values.append(tpot_ms)

    # ITL: flatten per-token latencies across all successful requests (per-token, not per-request)
    itl_all: list[float] = []
    for r in successful_runs:
        itl_all.extend(r.itl_ms)

    # E2EL: end-to-end latency in milliseconds
    e2el_values = [r.latency_s * 1000.0 for r in successful_runs]

    total_completion_tokens = sum(r.completion_tokens for r in successful_runs)
    total_prompt_tokens = sum(r.prompt_tokens for r in successful_runs)
    tokens_per_sec = round(total_completion_tokens / total_duration_s, 1) if total_duration_s > 0 else 0
    input_tokens_per_sec = round(total_prompt_tokens / total_duration_s, 1) if total_duration_s > 0 else 0
    total_tokens_per_sec = (
        round((total_prompt_tokens + total_completion_tokens) / total_duration_s, 1) if total_duration_s > 0 else 0
    )
    inputs_per_sec = round(len(successful_runs) / total_duration_s, 2) if total_duration_s > 0 else 0

    vram_peak: int | None = None
    try:
        if platform.system() == "Linux":
            gpu_idx = server_gpu_idx if server_gpu_idx is not None else _nvidia_gpu_index()
            smi_result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_idx}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if smi_result.returncode == 0:
                vram_peak = int(float(smi_result.stdout.strip().split("\n")[0]))
    except Exception:
        pass

    # Embedding dimension from successful runs (constant per model)
    emb_dims = [r.embedding_dim for r in successful_runs if r.embedding_dim > 0]
    embedding_dim = emb_dims[0] if emb_dims else None

    run_id = secrets.token_hex(6)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = BenchmarkResult(
        run_id=run_id,
        timestamp=timestamp,
        tag=tag,
        model=ModelInfo(
            model_id=args.model,
            revision=args.revision,
            quant=quant_resolved,
        ),
        environment=env,
        input=InputInfo(
            hash=input_hash,
            total_inputs=total_inputs,
            breakdown=breakdown,
            image_stats=image_stats,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_concurrency=max_concurrency,
            task=task_type,
        ),
        results=Results(
            ttft_ms=compute_stats(ttft_values),
            tpot_ms=compute_stats(tpot_values),
            itl_ms=compute_stats(itl_all),
            e2el_ms=compute_stats(e2el_values),
            tokens_per_sec=tokens_per_sec,
            input_tokens_per_sec=input_tokens_per_sec,
            total_tokens_per_sec=total_tokens_per_sec,
            inputs_per_sec=inputs_per_sec,
            latency_s_per_input=compute_stats(latency_values),
            prompt_tokens=compute_token_stat(prompt_token_values),
            completion_tokens=compute_token_stat(completion_token_values),
            cached_tokens=compute_token_stat(cached_token_values),
            reasoning_tokens=compute_token_stat(reasoning_token_values),
            vram_peak_mib=vram_peak,
            embedding_dim=embedding_dim,
            total_inputs_processed=len(runs_raw),
            total_duration_s=total_duration_s,
            errors=error_count,
            retries=retries,
        ),
        runs_raw=runs_raw,
    )

    save_dir = Path(args.output_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = _build_save_path(save_dir, env, args.model, tag)

    with open(save_path, "w") as f:
        f.write(json.dumps(dataclasses.asdict(result), indent=2))

    return result, save_path


def _parse_concurrency_levels(args: argparse.Namespace) -> list[int]:
    """Parse --concurrency (single value or comma-separated sweep)."""
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    if not levels:
        console.print("[red]--concurrency requires at least one value[/red]")
        sys.exit(1)
    return sorted(levels)


def upload_results(paths: list[Path], repo_id: str) -> None:
    """Upload result JSON files to a HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        console.print("[red]huggingface_hub library required for --upload.[/red]")
        console.print("[dim]Install with: uv pip install huggingface_hub[/dim]")
        sys.exit(1)

    api = HfApi()
    for path in paths:
        remote_path = f"results/{path.name}"
        console.print(f"  [bold green]↑[/bold green] Uploading [dim]->[/dim] {repo_id}/{remote_path}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    console.print(
        f"  [bold green]✓[/bold green] Uploaded {len(paths)} file(s) to [link=https://huggingface.co/datasets/{repo_id}]{repo_id}[/link]"
    )
    console.print()


def cmd_profiles() -> None:
    """List available model profiles."""
    profile_yamls = sorted(PROFILES_DIR.glob("*.yaml"))
    if not profile_yamls:
        console.print("  No profiles found.")
        return
    table = Table(box=box.SIMPLE)
    table.add_column("Profile", style="bold")
    table.add_column("Model")
    table.add_column("Notes", style="dim")
    for p in profile_yamls:
        with open(p) as f:
            d = yaml.safe_load(f)
        notes: list[str] = []
        if d.get("task") in ("embedding", "score"):
            notes.append(d["task"])
        if d.get("setup"):
            notes.append("custom setup")
        if "image" in d:
            notes.append(d["image"])
        table.add_row(p.stem, d.get("model", "?"), ", ".join(notes) if notes else "")
    console.print(table)


def generate_dockerfile(name: str) -> Path:
    """Generate a self-contained Dockerfile for a profile under ~/.vlmbench/profiles/<name>/."""
    profile = load_profile(name)
    image = profile.get("image", "vllm/vllm-openai:v0.15.1")
    model = profile.get("model", "")
    setup = profile.get("setup", "").strip()

    out_dir = Path.home() / ".vlmbench" / "profiles" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = out_dir / "Dockerfile"

    lines = [f"FROM {image}", f"LABEL vlmbench.profile={name} vlmbench.model={model}"]
    if setup:
        lines.extend(['RUN <<"EOF"', "set -euo pipefail", setup, "EOF"])
    dockerfile.write_text("\n".join(lines) + "\n")
    return dockerfile


def cmd_run(args: argparse.Namespace) -> None:
    """Run a VLM benchmark."""
    # Apply profile defaults before anything else
    if args.profile:
        profile = load_profile(args.profile)
        if not args.model:
            args.model = profile["model"]
        if args.prompt == DEFAULT_PROMPT and "prompt" in profile:
            args.prompt = profile["prompt"]
        if args.serve_args is None and "serve_args" in profile:
            args.serve_args = profile["serve_args"]
        if args.task == "completion" and "task" in profile:
            args.task = profile["task"]
        if "embedding" in profile:
            args.embedding_options = profile["embedding"]
        if args.serve and profile.get("setup"):
            console.print(f"  [cyan]Running profile setup for '{args.profile}'...[/cyan]")
            _run_profile_setup(profile["setup"])
    if not hasattr(args, "embedding_options"):
        args.embedding_options = None

    if args.upload and not os.environ.get("HF_TOKEN"):
        console.print("[red]--upload requires HF_TOKEN environment variable.[/red]")
        console.print("[dim]Set it with: export HF_TOKEN=hf_...[/dim]")
        sys.exit(1)

    base_url, tmux_session = resolve_server(
        base_url=args.base_url,
        serve=args.serve,
        backend=args.backend,
        model=args.model,
        serve_args=args.serve_args,
    )

    # Auto-detect model from the server when --model is not provided
    if not args.model:
        args.model = _fetch_model_from_server(base_url, api_key=args.api_key)
        if args.model:
            console.print(f"  [dim]Auto-detected model: {args.model}[/dim]")
        else:
            console.print(
                "[red]Could not determine model: specify --model or ensure the server exposes GET /v1/models.[/red]"
            )
            sys.exit(1)

    env = collect_environment(base_url)

    server_gpu_idx: int | None = None
    try:
        if platform.system() == "Linux":
            from urllib.parse import urlparse

            parsed = urlparse(base_url)
            port = parsed.port or 8000
            server_gpu_idx = get_gpu_for_server_port(port)
    except Exception:
        pass

    if args.dataset:
        dataset_name = args.dataset.removeprefix("hf://")
        inputs, breakdown, input_hash = load_hf_dataset(
            dataset_name,
            image_col=args.dataset_image_col,
            text_col=args.dataset_text_col,
            split=args.dataset_split,
            max_samples=args.max_samples,
            max_size=args.max_size,
        )
        input_source = f"hf://{dataset_name}"
    elif args.input_path:
        inputs, breakdown, input_hash = load_local_inputs(args.input_path, max_size=args.max_size)
        input_source = args.input_path
        if args.max_samples is not None and args.max_samples < len(inputs):
            inputs = inputs[: args.max_samples]
    else:
        inputs = [[DEFAULT_IMAGE_URL]]
        breakdown = {"images": 1, "pdf_pages": 0, "video_frames": 0}
        input_hash = f"url:{DEFAULT_IMAGE_URL}"
        input_source = DEFAULT_IMAGE_URL

    total_inputs = len(inputs)
    image_stats = compute_image_stats(inputs)
    quant_resolved = args.quant if args.quant != "auto" else None

    concurrency_levels = _parse_concurrency_levels(args)
    is_sweep = len(concurrency_levels) > 1

    print_config(
        model_id=args.model,
        revision=args.revision,
        quant=quant_resolved,
        base_url=base_url,
        backend=env.backend,
        backend_version=env.backend_version,
        env=env,
        total_inputs=total_inputs,
        breakdown=breakdown,
        input_path=input_source,
        max_tokens=args.max_tokens,
        runs=args.runs,
        max_concurrency=concurrency_levels[-1] if is_sweep else concurrency_levels[0],
        tmux_session=tmux_session,
    )

    if is_sweep:
        console.print(f"  [bold]Concurrency sweep:[/bold] {', '.join(str(c) for c in concurrency_levels)}")
        console.print()

    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key)

    all_results: list[tuple[BenchmarkResult, Path]] = []
    for idx, c in enumerate(concurrency_levels):
        tag_parts = [args.tag] if args.tag else []
        tag_parts.append(f"c{c}")
        run_tag = "-".join(tag_parts)

        label = f"{idx + 1}/{len(concurrency_levels)} c={c}" if is_sweep else None
        if is_sweep:
            console.print(f"  [bold cyan]▸ concurrency={c}[/bold cyan]  ({idx + 1}/{len(concurrency_levels)})")

        result, save_path = _execute_benchmark(
            args=args,
            client=client,
            env=env,
            inputs=inputs,
            input_hash=input_hash,
            image_stats=image_stats,
            breakdown=breakdown,
            quant_resolved=quant_resolved,
            server_gpu_idx=server_gpu_idx,
            max_concurrency=c,
            tag=run_tag,
            label=label,
        )
        all_results.append((result, save_path))

        title_suffix = f"  (concurrency={c})" if is_sweep else ""
        print_results(result, str(save_path), title_suffix=title_suffix)

    if is_sweep:
        saved = [str(p) for _, p in all_results]
        results = [r for r, _ in all_results]
        print_concurrency_table(results, saved)

    if args.upload:
        upload_results([p for _, p in all_results], args.upload_repo)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare benchmark results from multiple JSON files."""
    files = args.files
    if not files:
        default_dir = Path(DEFAULT_SAVE_DIR)
        if not default_dir.is_dir():
            console.print(f"[red]No results directory found at {DEFAULT_SAVE_DIR}/[/red]")
            console.print("[dim]Run a benchmark first, or pass JSON files explicitly.[/dim]")
            sys.exit(1)
        files = sorted(str(p) for p in default_dir.glob("*.json"))
        if not files:
            console.print(f"[red]No JSON files in {DEFAULT_SAVE_DIR}/[/red]")
            sys.exit(1)
        console.print(f"  [dim]Loading {len(files)} results from {DEFAULT_SAVE_DIR}/[/dim]")
        console.print()

    results: list[BenchmarkResult] = []
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            console.print(f"[red]File not found: {filepath}[/red]")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        results.append(_dc_from_dict(BenchmarkResult, data))

    if not results:
        console.print("[red]No result files found.[/red]")
        sys.exit(1)

    # Sort by chosen metric descending
    results.sort(key=lambda r: r.results.inputs_per_sec, reverse=True)

    print_compare_table(results, max_rows_per_model=args.max_rows_per_model, sort_by=args.sort_by)


def main() -> None:
    """Entry point: default to 'run' subcommand if none specified."""
    # Always show the banner first
    print_banner()

    # If no subcommand given (first arg starts with -- or -), insert "run"
    argv = sys.argv[1:]
    subcommands = {"compare", "profiles"}
    if argv and argv[0] not in subcommands and not argv[0].startswith("--help"):
        if argv[0].startswith("--") or argv[0].startswith("-"):
            argv = ["run"] + argv
    elif not argv:
        # No args at all → show help
        pass

    parser = build_parser()
    parsed = parser.parse_args(argv)

    match parsed.command:
        case None:
            parser.print_help()
            sys.exit(0)
        case "run":
            cmd_run(parsed)
        case "compare":
            cmd_compare(parsed)
        case "profiles":
            cmd_profiles()


if __name__ == "__main__":
    main()
