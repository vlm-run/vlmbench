# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13",
#   "openai>=1.0",
#   "tenacity>=8",
#   "Pillow>=10",
#   "pypdfium2>=4",
# ]
# ///
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
built by VLM Run · https://vlm.run
"""

from __future__ import annotations

import argparse
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Union, get_args, get_origin, get_type_hints, runtime_checkable

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ── Constants ─────────────────────────────────────────────────────────────────

VERSION = "0.2.0"
SCHEMA_VERSION = "0.1.0"
DEFAULT_PROMPT = "Extract all text from this document."
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RUNS = 3
DEFAULT_CONCURRENCY = 1
DEFAULT_SAVE_DIR = "./results"
DEFAULT_API_KEY = "no-key"

DEFAULT_VLLM_IMAGE = "vllm-openai:latest"
STEEL_BLUE = "#4682B4"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

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
    quant: Optional[str] = None
    params_b: Optional[float] = None


@dataclass
class EnvironmentInfo:
    backend: str = ""
    backend_version: Optional[str] = None
    base_url: str = ""
    accelerator: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_vram_mib: Optional[int] = None
    gpu_driver: Optional[str] = None
    ram_total_mib: Optional[int] = None
    cpu: Optional[str] = None
    os: Optional[str] = None


@dataclass
class InputInfo:
    hash: str = ""
    total_inputs: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)
    prompt: str = ""
    max_tokens: int = 0
    max_concurrency: int = 1


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
    tokens_per_sec: float = 0.0
    inputs_per_sec: float = 0.0
    latency_s_per_input: StatBlock = field(default_factory=StatBlock)
    prompt_tokens: TokenStat = field(default_factory=TokenStat)
    completion_tokens: TokenStat = field(default_factory=TokenStat)
    cached_tokens: Optional[TokenStat] = None
    reasoning_tokens: Optional[TokenStat] = None
    vram_peak_mib: Optional[int] = None
    total_inputs_processed: int = 0
    total_duration_s: float = 0.0
    errors: int = 0
    retries: int = 0


@dataclass
class RunRaw:
    input_idx: int = 0
    run: int = 0
    ttft_ms: Optional[float] = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    timestamp: str = ""
    tag: Optional[str] = None
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
    # Optional[X] is Union[X, None]
    if origin is Union:
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


def detect_backend(base_url: str) -> tuple[str, Optional[str]]:
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
    if "openai.com" in base_url:
        return "openai", None
    if "together.xyz" in base_url:
        return "together", None

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


def get_nvidia_gpu_info() -> dict[str, Any]:
    """Linux: parse nvidia-smi for GPU info."""
    try:
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

    # CPU
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                info["cpu"] = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu"] = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass

    # RAM
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                info["ram_total_mib"] = int(result.stdout.strip()) // (1024 * 1024)
        elif platform.system() == "Linux":
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

    # GPU info based on platform
    if backend in ("openai", "together"):
        # Cloud APIs — no local GPU info
        env_data["accelerator"] = None
        env_data["gpu_name"] = None
        env_data["gpu_vram_mib"] = None
        env_data["gpu_driver"] = None
    elif platform.system() == "Darwin":
        gpu_info = get_apple_gpu_info()
        env_data.update(gpu_info)
    elif platform.system() == "Linux":
        gpu_info = get_nvidia_gpu_info()
        env_data.update(gpu_info)

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
    """Manage an inference server via Docker with GPU support in tmux."""

    DEFAULT_PORT = 8000

    def __init__(self, backend: str, port: int | None = None) -> None:
        self.port = port or self.DEFAULT_PORT
        self._base_url = f"http://localhost:{self.port}/v1"
        self.docker_image, self.container_name = _parse_docker_backend(backend)
        self.category = _backend_category(backend)
        self.session_name = _tmux_session_name(self.category)

    def is_running(self) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{self.port}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self, model: str, extra_args: str | None = None) -> None:
        _require_command("tmux")
        _require_command("docker")

        # Build docker run command (pull + run chained in tmux pane)
        cache_dir = Path.home() / ".vlmbench" / self.category
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Forward CUDA_VISIBLE_DEVICES from host so the user can select GPUs
        cuda_env = "-e CUDA_DEVICE_ORDER=PCI_BUS_ID"
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None:
            cuda_env += f" -e CUDA_VISIBLE_DEVICES={cuda_visible}"
        run_cmd = (
            f"docker run --rm --gpus all {cuda_env}"
            f" --name {shlex.quote(self.container_name)}"
            f" -p {self.port}:8000 --ipc=host"
            f" -v {shlex.quote(str(cache_dir))}:/root/.cache/huggingface"
            f" {shlex.quote(self.docker_image)}"
            f" --model {shlex.quote(model)}"
        )
        if extra_args:
            run_cmd += f" {shlex.join(shlex.split(extra_args))}"
        pull_cmd = f"docker pull {shlex.quote(self.docker_image)}"
        server_cmd = f"{pull_cmd} && {run_cmd}"

        # Kill any stale session with the same stable name
        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            capture_output=True,
        )

        # Top pane: docker run | Bottom pane: GPU monitor
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name, server_cmd],
            check=True,
        )
        monitor_cmd = _gpu_monitor_cmd()
        if monitor_cmd:
            subprocess.run(
                ["tmux", "split-window", "-v", "-t", self.session_name, monitor_cmd],
                check=True,
            )
            subprocess.run(
                ["tmux", "select-pane", "-t", f"{self.session_name}:.0"],
                check=True,
            )
        console.print(
            f"  [cyan]Started [bold]{self.docker_image}[/bold] in tmux session '[bold]{self.session_name}[/bold]'[/cyan]"
        )
        console.print(f"  [dim]Attach with: tmux attach -t {self.session_name}[/dim]")
        console.print(f"  [dim]Container:   docker logs -f {self.container_name}[/dim]")

    def wait_ready(self, timeout: int = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            time.sleep(2)
        return False

    def get_base_url(self) -> str:
        return self._base_url


class OllamaServerManager:
    """Manage an Ollama inference server via tmux."""

    DEFAULT_PORT = 11434

    def __init__(self, port: int | None = None) -> None:
        self.port = port or self.DEFAULT_PORT
        self._base_url = f"http://localhost:{self.port}/v1"
        self.session_name = _tmux_session_name("ollama")

    def is_running(self) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{self.port}/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self, model: str, extra_args: str | None = None) -> None:
        _require_command("tmux")
        if not self.is_running():
            _require_command("ollama")
            server_cmd = "ollama serve"
            if extra_args:
                server_cmd += f" {shlex.join(shlex.split(extra_args))}"
            # Kill any stale session with the same stable name
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                capture_output=True,
            )
            # Top pane: ollama serve | Bottom pane: system monitor
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self.session_name, server_cmd],
                check=True,
            )
            monitor_cmd = _gpu_monitor_cmd()
            if monitor_cmd:
                subprocess.run(
                    ["tmux", "split-window", "-v", "-t", self.session_name, monitor_cmd],
                    check=True,
                )
                subprocess.run(
                    ["tmux", "select-pane", "-t", f"{self.session_name}:.0"],
                    check=True,
                )
            console.print(f"  [cyan]Started Ollama in tmux session '[bold]{self.session_name}[/bold]'[/cyan]")
            console.print(f"  [dim]Attach with: tmux attach -t {self.session_name}[/dim]")
            # Wait for Ollama to be responsive before pulling
            if not self.wait_ready(timeout=30):
                console.print("[red]Ollama failed to start within 30s.[/red]")
                sys.exit(1)
        # Ensure the model is available
        console.print(f"  [cyan]Pulling model '{model}' (if needed)...[/cyan]")
        subprocess.run(["ollama", "pull", model], check=True)

    def wait_ready(self, timeout: int = 120) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            time.sleep(2)
        return False

    def get_base_url(self) -> str:
        return self._base_url


class NativeVllmServerManager:
    """Manage a native vLLM inference server (``vllm serve``) via tmux."""

    DEFAULT_PORT = 8000

    def __init__(self, port: int | None = None) -> None:
        self.port = port or self.DEFAULT_PORT
        self._base_url = f"http://localhost:{self.port}/v1"
        self.session_name = _tmux_session_name("vllm")

    def is_running(self) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{self.port}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self, model: str, extra_args: str | None = None) -> None:
        _require_command("tmux")
        _require_command("vllm")

        server_cmd = f"vllm serve {shlex.quote(model)}"
        if self.port != self.DEFAULT_PORT:
            server_cmd += f" --port {self.port}"
        if extra_args:
            server_cmd += f" {shlex.join(shlex.split(extra_args))}"

        # Kill any stale session with the same stable name
        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            capture_output=True,
        )

        # Top pane: vllm serve | Bottom pane: GPU monitor
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name, server_cmd],
            check=True,
        )
        monitor_cmd = _gpu_monitor_cmd()
        if monitor_cmd:
            subprocess.run(
                ["tmux", "split-window", "-v", "-t", self.session_name, monitor_cmd],
                check=True,
            )
            subprocess.run(
                ["tmux", "select-pane", "-t", f"{self.session_name}:.0"],
                check=True,
            )
        console.print(f"  [cyan]Started native vLLM in tmux session '[bold]{self.session_name}[/bold]'[/cyan]")
        console.print(f"  [dim]Attach with: tmux attach -t {self.session_name}[/dim]")

    def wait_ready(self, timeout: int = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            time.sleep(2)
        return False

    def get_base_url(self) -> str:
        return self._base_url


def _require_command(name: str) -> None:
    """Check that a CLI tool is available on PATH."""
    if shutil.which(name) is None:
        if name == "tmux":
            console.print("[red]tmux is required for --serve. Install with: brew install tmux / apt install tmux[/red]")
        elif name == "vllm":
            console.print("[red]'vllm' not found on PATH. Install with: uv pip install vllm[/red]")
        else:
            console.print(f"[red]'{name}' not found on PATH. Please install it first.[/red]")
        sys.exit(1)


def _gpu_monitor_cmd() -> str | None:
    """Return the best available GPU/system monitor command for the bottom tmux pane."""
    if platform.system() == "Darwin":
        # macOS: prefer macmon (brew install macmon)
        if shutil.which("macmon"):
            return "macmon"
    else:
        # Linux: prefer nvitop via uvx
        if shutil.which("uvx"):
            return "uvx nvitop -m full -c"
        if shutil.which("nvitop"):
            return "nvitop -m full -c"
    return None


def _auto_detect_backend() -> str:
    """Pick a default backend based on the current platform."""
    if platform.system() == "Darwin":
        return "ollama"
    return DEFAULT_VLLM_IMAGE  # Linux defaults to Docker vLLM


def _resolve_backend_for_monitor(backend: str) -> str:
    """Resolve backend string to a category: 'ollama', 'vllm', or 'sglang'."""
    if backend == "auto":
        return "ollama" if platform.system() == "Darwin" else "vllm"
    if backend == "ollama":
        return "ollama"
    if backend.startswith("sglang"):
        return "sglang"
    return "vllm"


def _create_server_manager(backend: str) -> DockerServerManager | OllamaServerManager | NativeVllmServerManager:
    """Create the appropriate server manager for the given backend."""
    if backend == "ollama":
        return OllamaServerManager()
    if backend == "vllm":
        return NativeVllmServerManager()
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


def resolve_server(
    base_url: str | None,
    serve: bool,
    backend: str,
    model: str,
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
        session = _start_monitor_session(detected_backend)
        return detected_url, session

    # No server running
    if not serve:
        console.print("[red]No server found. Pass --base-url explicitly or use --serve.[/red]")
        sys.exit(1)

    # Resolve which backend to use
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

    return manager.get_base_url(), getattr(manager, "session_name", None)


def _try_detect_running_server() -> tuple[str | None, str]:
    """Check common local ports for a running inference server.

    Returns (base_url | None, backend_name).
    """
    # Try vLLM default
    try:
        req = urllib.request.Request("http://localhost:8000/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return "http://localhost:8000/v1", "vllm"
    except Exception:
        pass

    # Try Ollama default
    try:
        req = urllib.request.Request("http://localhost:11434/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return "http://localhost:11434/v1", "ollama"
    except Exception:
        pass

    return None, "unknown"


# ── Input Loading ─────────────────────────────────────────────────────────────


def image_to_base64(path: Path) -> str:
    """Convert an image file to a base64 data URI."""
    with open(path, "rb") as f:
        data = f.read()

    # Determine MIME type from extension
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def pdf_to_base64_images(path: Path, dpi: int = 150) -> list[str]:
    """Convert PDF pages to base64 data URIs using pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    results = []
    for idx in range(len(doc)):
        page = doc[idx]
        bitmap = page.render(scale=dpi / 72)
        img = bitmap.to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        results.append(f"data:image/png;base64,{b64}")
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


def load_inputs(input_path: str) -> tuple[list[list[str]], dict[str, int], str]:
    """Load inputs from a file or directory.

    Returns:
        (inputs, breakdown, hash)
        - inputs: list of lists of base64 data URIs (each inner list = one "input")
        - breakdown: {"images": N, "pdf_pages": N, "video_frames": N}
        - hash: SHA256 hex of all input file contents
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

        # Update hash
        with open(f, "rb") as fh:
            hasher.update(fh.read())

        if ext in IMAGE_EXTENSIONS:
            b64 = image_to_base64(f)
            inputs.append([b64])
            breakdown["images"] += 1
        elif ext in PDF_EXTENSIONS:
            pages = pdf_to_base64_images(f)
            # Each page is a separate input
            for page in pages:
                inputs.append([page])
            breakdown["pdf_pages"] += len(pages)
        elif ext in VIDEO_EXTENSIONS:
            frames = video_to_base64_frames(f)
            if frames:
                # All frames from one video = one input
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


def call_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> dict[str, Any]:
    """Call the completion API with streaming and measure timing.

    Returns dict with keys: ttft_ms, latency_s, prompt_tokens, completion_tokens,
    cached_tokens, reasoning_tokens, error
    """
    global _stream_options_supported

    t_start = time.perf_counter()
    ttft: Optional[float] = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    full_content = ""

    # Try with stream_options first, fall back if unsupported
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
            # First call failed — retry without stream_options
            _stream_options_supported = False
            kwargs.pop("stream_options", None)
            t_start = time.perf_counter()
            stream = client.chat.completions.create(**kwargs)
        else:
            raise

    if _stream_options_supported is None:
        _stream_options_supported = True

    for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = (time.perf_counter() - t_start) * 1000  # ms

        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content

        # Final chunk with usage
        if hasattr(chunk, "usage") and chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            # Extract detailed token breakdown if available
            pd = getattr(chunk.usage, "prompt_tokens_details", None)
            if pd is not None:
                cached_tokens = getattr(pd, "cached_tokens", 0) or 0
            cd = getattr(chunk.usage, "completion_tokens_details", None)
            if cd is not None:
                reasoning_tokens = getattr(cd, "reasoning_tokens", 0) or 0

    t_end = time.perf_counter()
    latency_s = t_end - t_start

    # Fallback: estimate completion tokens from content if usage not provided
    if completion_tokens == 0 and full_content:
        completion_tokens = len(full_content.split())

    return {
        "ttft_ms": ttft if ttft is not None else latency_s * 1000,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "error": None,
    }


def build_messages(prompt: str, image_uris: list[str]) -> list[dict]:
    """Build OpenAI-compatible messages with image content."""
    content: list[dict] = []
    for uri in image_uris:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": uri},
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


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


def _call_with_retry_count(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> tuple[dict[str, Any], int]:
    """Call completion with retry, return (result, retry_count)."""
    attempt = 0

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
    )
    def _inner() -> dict[str, Any]:
        nonlocal attempt
        attempt += 1
        return call_completion(client, model, messages, max_tokens)

    result = _inner()
    return result, max(0, attempt - 1)


def run_benchmark(
    client: OpenAI,
    model: str,
    inputs: list[list[str]],
    prompt: str,
    max_tokens: int,
    runs: int,
    max_concurrency: int,
    warmup: int = 1,
    progress_callback: Any = None,
) -> tuple[list[RunRaw], int]:
    """Run the benchmark: warmup + N timed runs per input.

    Returns (runs_raw, retries_count).
    """
    runs_raw: list[RunRaw] = []
    retries = 0

    # Warmup runs on first input (not recorded)
    # First warmup is a validation pass — fail fast on obvious errors
    for w in range(warmup):
        if progress_callback:
            progress_callback("warmup", w + 1, 0)
        try:
            messages = build_messages(prompt, inputs[0])
            call_completion(client, model, messages, max_tokens)
        except Exception as e:
            err_str = str(e).lower()
            # Fail fast on errors that will never self-resolve
            if any(
                kw in err_str
                for kw in ("not found", "404", "does not exist", "invalid model", "401", "403", "unauthorized")
            ):
                console.print()
                print_error("Warmup failed", str(e))
                sys.exit(1)
            if w == 0:
                # First warmup failed with a non-obvious error — warn but continue
                console.print(f"  [yellow]Warmup warning: {e}[/yellow]")

    # Timed runs
    def execute_single(input_idx: int, run_num: int) -> RunRaw:
        nonlocal retries
        messages = build_messages(prompt, inputs[input_idx])
        try:
            result, retry_count = _call_with_retry_count(client, model, messages, max_tokens)
            retries += retry_count
            return RunRaw(
                input_idx=input_idx,
                run=run_num,
                ttft_ms=round(result["ttft_ms"], 1),
                latency_s=round(result["latency_s"], 3),
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                cached_tokens=result["cached_tokens"],
                reasoning_tokens=result["reasoning_tokens"],
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

    len(inputs) * runs
    completed = 0

    for run_num in range(1, runs + 1):
        if progress_callback:
            progress_callback("run", run_num, completed)

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(execute_single, i, run_num): i for i in range(len(inputs))}
            for future in as_completed(futures):
                run_result = future.result()
                runs_raw.append(run_result)
                completed += 1
                # Fail fast: if first completed request errors, abort
                if run_result.error and completed == 1:
                    console.print()
                    print_error("Benchmark aborted", f"First request failed: {run_result.error}")
                    sys.exit(1)
                if progress_callback:
                    progress_callback("progress", run_num, completed)

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
            f"  [bold {STEEL_BLUE}]vlmbench[/bold {STEEL_BLUE}] (v{VERSION})"
            " — Single-file, drop-in VLM benchmark CLI for your agents."
        )
        console.print(
            f"  built by [bold {STEEL_BLUE}][link=https://vlm.run]VLM Run[/link][/bold {STEEL_BLUE}] · https://vlm.run"
        )
    else:
        console.print(f"  [bold {STEEL_BLUE}]vlmbench[/bold {STEEL_BLUE}] (v{VERSION})")
        console.print("  built by [link=https://vlm.run]VLM Run[/link]")
    console.print()


def print_config(
    model_id: str,
    revision: str,
    quant: Optional[str],
    base_url: str,
    backend: str,
    backend_version: Optional[str],
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
    lines = Text()

    # Model line
    quant_str = f" ({quant})" if quant and quant != "auto" else ""
    lines.append("  Model      ", style="bold cyan")
    lines.append(f"{model_id} @ {revision}{quant_str}\n")

    # Server line
    version_str = f" {backend_version}" if backend_version else ""
    backend_display = backend.capitalize() if backend != "vllm" else "vLLM"
    lines.append("  Server     ", style="bold cyan")
    lines.append(f"{base_url} ")
    lines.append("• ", style="dim")
    lines.append(f"{backend_display}{version_str}\n")

    # Hardware line
    if env.gpu_name:
        lines.append("  Hardware   ", style="bold cyan")
        if env.accelerator == "metal":
            vram_gb = f"{env.gpu_vram_mib // 1024} GB unified" if env.gpu_vram_mib else ""
            lines.append(f"{env.gpu_name} ")
            lines.append("• ", style="dim")
            lines.append("Metal ")
            lines.append("• ", style="dim")
            lines.append(f"{vram_gb}\n")
        elif env.accelerator == "cuda":
            vram_str = f"({env.gpu_vram_mib} MiB)" if env.gpu_vram_mib else ""
            driver_str = f" • Driver {env.gpu_driver}" if env.gpu_driver else ""
            lines.append(f"{env.gpu_name} {vram_str} ")
            lines.append("• ", style="dim")
            lines.append(f"CUDA{driver_str}\n")
        else:
            lines.append(f"{env.gpu_name}\n")
    elif env.cpu:
        lines.append("  Hardware   ", style="bold cyan")
        lines.append(f"{env.cpu}\n")

    # Input line
    parts = []
    if breakdown.get("images"):
        parts.append(f"{breakdown['images']} images")
    if breakdown.get("pdf_pages"):
        parts.append(f"{breakdown['pdf_pages']} PDF pages")
    if breakdown.get("video_frames"):
        parts.append(f"{breakdown['video_frames']} video frames")
    breakdown_str = ", ".join(parts) if parts else "mixed"
    lines.append("  Input      ", style="bold cyan")
    lines.append(f"{input_path} ")
    lines.append("-> ", style="dim")
    lines.append(f"{total_inputs} images ({breakdown_str})\n")

    # Config line
    lines.append("  Config     ", style="bold cyan")
    lines.append(f"max_tokens={max_tokens} ")
    lines.append("• ", style="dim")
    lines.append(f"runs={runs} ")
    lines.append("• ", style="dim")
    lines.append(f"concurrency={max_concurrency}")

    # Tmux session line
    if tmux_session:
        lines.append("\n")
        lines.append("  Tmux       ", style="bold cyan")
        lines.append(f"{tmux_session} ", style="bold")
        lines.append("• ", style="dim")
        lines.append("tmux attach -t ", style="dim")
        lines.append(tmux_session, style="dim")

    panel = Panel(
        lines,
        title="[bold]Configuration[/bold]",
        title_align="left",
        border_style="bright_magenta",
        box=box.ROUNDED,
        padding=(1, 1),
    )
    console.print(panel)


def print_results(result: BenchmarkResult, save_path: str) -> None:
    """Print the results card in a styled Rich Panel."""
    r = result.results
    lines = Text()

    def _metric_color(label: str) -> str:
        """Pick color based on metric type."""
        return "bold cyan"

    def _value_color_throughput(val: float) -> str:
        if val >= 50:
            return "bold green"
        elif val >= 20:
            return "green"
        elif val >= 10:
            return "yellow"
        return "red"

    # TTFT
    lines.append("  TTFT         ", style=_metric_color("ttft"))
    lines.append(f"{r.ttft_ms.mean:>6.0f} ms", style="bold")
    lines.append(f"    (p95: {r.ttft_ms.p95:.0f} ms)\n", style="dim")

    # TPOT
    lines.append("  TPOT         ", style=_metric_color("tpot"))
    lines.append(f"{r.tpot_ms.mean:>6.1f} ms", style="bold")
    lines.append(f"    (p95: {r.tpot_ms.p95:.1f} ms)\n", style="dim")

    # Throughput
    lines.append("  Throughput   ", style=_metric_color("throughput"))
    lines.append(f"{r.tokens_per_sec:>6.1f} tok/s", style=_value_color_throughput(r.tokens_per_sec))
    lines.append(f"   {r.inputs_per_sec:.2f} images/s\n")

    # Latency
    lines.append("  Latency      ", style=_metric_color("latency"))
    lines.append(f"{r.latency_s_per_input.mean:>6.2f} s/img", style="bold")
    lines.append(f"  (p95: {r.latency_s_per_input.p95:.2f} s)\n", style="dim")

    # Tokens
    lines.append("  Tokens       ", style=_metric_color("tokens"))
    lines.append(f"{r.prompt_tokens.mean:>6d} prompt", style="bold")
    lines.append(f"   {r.completion_tokens.mean:>4d} completion (avg)")
    if r.cached_tokens is not None and r.cached_tokens.mean > 0:
        lines.append(f"   {r.cached_tokens.mean:>4d} cached", style="dim")
    if r.reasoning_tokens is not None and r.reasoning_tokens.mean > 0:
        lines.append(f"   {r.reasoning_tokens.mean:>4d} reasoning", style="dim")
    lines.append("\n")

    # VRAM
    if r.vram_peak_mib is not None:
        lines.append("  VRAM         ", style=_metric_color("vram"))
        lines.append(f"{r.vram_peak_mib:>6d} MiB peak\n", style="bold")

    # Reliability
    total_ok = r.total_inputs_processed - r.errors
    lines.append("  Reliability  ", style=_metric_color("reliability"))
    if r.errors == 0:
        lines.append(f"{total_ok}/{r.total_inputs_processed} ok", style="bold green")
    else:
        lines.append(f"{total_ok}/{r.total_inputs_processed} ok", style="bold yellow")
    lines.append(f", {r.retries} retries", style="dim" if r.retries == 0 else "yellow")

    panel = Panel(
        lines,
        title="[bold]Results[/bold]",
        title_align="left",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 1),
    )
    console.print(panel)
    console.print(f"  [green bold]>[/green bold] Saved [dim]->[/dim] {save_path}")
    console.print()


def print_compare_table(results: list[BenchmarkResult]) -> None:
    """Print the compare table grouped by model, sorted by tok/s descending."""
    console.print(f"  [bold]compare[/bold] [dim]({len(results)} runs)[/dim]")
    console.print()

    def _total_tok_s(r: BenchmarkResult) -> float:
        return r.results.tokens_per_sec * r.input.max_concurrency

    # Group by model_id, sort groups by best tok/s descending
    from itertools import groupby

    sorted_results = sorted(results, key=lambda r: (r.model.model_id, -_total_tok_s(r)))
    groups: list[tuple[str, list[BenchmarkResult]]] = [
        (model_id, list(runs)) for model_id, runs in groupby(sorted_results, key=lambda r: r.model.model_id)
    ]
    groups.sort(key=lambda g: max(_total_tok_s(r) for r in g[1]), reverse=True)

    # Global fastest
    max_toks = max(_total_tok_s(r) for r in results)

    # Collect all column values to detect uniform columns
    all_quants = {r.model.quant or "-" for r in results}
    all_drivers = {r.environment.gpu_driver or "-" for r in results}
    all_backends: set[str] = set()
    for r in results:
        be = r.environment.backend
        be_ver = r.environment.backend_version or ""
        all_backends.add(f"vLLM {be_ver}".strip() if be == "vllm" else f"{be.capitalize()} {be_ver}".strip())

    show_quant = len(all_quants) > 1
    len(all_drivers) > 1
    len(all_backends) > 1

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.ROUNDED,
        border_style="dim white",
        padding=(0, 1),
    )
    table.add_column("Model", min_width=24, no_wrap=True, style=f"bold {STEEL_BLUE}")
    table.add_column("TTFT (ms)", justify="right", min_width=8, style="dim")
    table.add_column("TPOT (ms)", justify="right", min_width=8, style="dim")
    table.add_column("Tok/s \u2193", justify="right", min_width=7)
    table.add_column("Img/s", justify="right", min_width=6, style="dim")
    table.add_column("Duration (s)", justify="right", min_width=8, style="dim")
    table.add_column("num_workers", justify="right", min_width=4, style="dim")
    table.add_column("VRAM", justify="right", min_width=9, style="dim")
    if show_quant:
        table.add_column("Quant", min_width=6, style="dim")
    table.add_column("Backend", min_width=7, style="dim")
    table.add_column("Hardware", min_width=14, style="dim")
    table.add_column("Driver", min_width=8, style="dim")

    for group_idx, (model_id, runs) in enumerate(groups):
        if group_idx > 0:
            table.add_section()

        for row_idx, r in enumerate(runs):
            be = r.environment.backend
            be_ver = r.environment.backend_version or ""
            be_label = f"vLLM {be_ver}".strip() if be == "vllm" else f"{be.capitalize()} {be_ver}".strip()

            concurrency = r.input.max_concurrency
            hardware = r.environment.gpu_name or "-"
            vram_gb = f"{r.results.vram_peak_mib / 1024:.2f} GB" if r.results.vram_peak_mib is not None else "-"
            total_toks = _total_tok_s(r)
            total_imgs = r.results.inputs_per_sec * concurrency
            is_fastest = total_toks == max_toks

            tok_s_text = f"{total_toks:.1f}"
            tok_s_style = "bold bright_green" if is_fastest else "white"

            # Show model name only on the first row of each group
            model_cell = model_id if row_idx == 0 else ""

            cells: list[Any] = [
                model_cell,
                f"{r.results.ttft_ms.mean:.0f}",
                f"{r.results.tpot_ms.mean:.1f}",
                Text(tok_s_text, style=tok_s_style),
                f"{total_imgs:.2f}",
                f"{r.results.total_duration_s:.1f}",
                str(concurrency),
                vram_gb,
            ]
            if show_quant:
                cells.append(r.model.quant or "-")
            cells.append(be_label)
            cells.append(hardware)
            cells.append(r.environment.gpu_driver or "-")

            table.add_row(*cells)

    console.print(table)
    console.print()

    # ── Summary statistics ────────────────────────────────────────────────
    unique_models = {r.model.model_id for r in results}
    all_tok_s = [_total_tok_s(r) for r in results]
    total_duration = sum(r.results.total_duration_s for r in results)
    total_errors = sum(r.results.errors for r in results)
    total_retries = sum(r.results.retries for r in results)

    lines = Text()
    lines.append("Runs       ", style=f"bold {STEEL_BLUE}")
    lines.append(f"{len(results)}", style="white")
    lines.append(f" across {len(unique_models)} model(s)", style="dim")
    lines.append(f"  total duration {total_duration:.1f}s\n", style="dim")

    lines.append("Tok/s      ", style=f"bold {STEEL_BLUE}")
    lines.append(f"{max(all_tok_s):.1f}", style="bright_green")
    lines.append(" best", style="dim")
    lines.append(f"   {min(all_tok_s):.1f}", style="white")
    lines.append(" worst", style="dim")
    lines.append(f"   {statistics.mean(all_tok_s):.1f}", style="white")
    lines.append(" avg\n", style="dim")

    reliability_style = "bright_green" if total_errors == 0 else "yellow"
    lines.append("Errors     ", style=f"bold {STEEL_BLUE}")
    lines.append(f"{total_errors}", style=reliability_style)
    if total_retries > 0:
        lines.append(f"  ({total_retries} retries)", style="dim")

    panel = Panel(
        lines,
        title="[bold]Summary[/bold]",
        title_align="left",
        subtitle=f"[dim]vlmbench v{VERSION}[/dim]",
        subtitle_align="right",
        border_style="dim white",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vlmbench",
        description="Single-file, drop-in VLM benchmark CLI for your agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run subcommand ──
    run_parser = subparsers.add_parser("run", help="Run a VLM benchmark.")

    # Required
    run_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model ID (vLLM: Qwen/Qwen3-VL-2B-Instruct, Ollama: qwen3-vl:2b)",
    )
    run_parser.add_argument(
        "--input", "-i", required=True, dest="input_path", help="File or directory (images, PDFs, videos)"
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
        default=True,
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
    run_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens")

    # Benchmark
    run_parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Timed runs per input")
    run_parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (not recorded)")
    run_parser.add_argument("--max-concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max parallel requests")

    # Output
    run_parser.add_argument("--save", default=DEFAULT_SAVE_DIR, help="Output directory")
    run_parser.add_argument("--tag", default=None, help="Custom grouping label")

    # ── compare subcommand ──
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results from multiple JSON files.")
    compare_parser.add_argument("files", nargs="+", help="JSON result files to compare")

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    """Run a VLM benchmark."""
    # Resolve base URL: auto-detect, or auto-start server
    base_url, tmux_session = resolve_server(
        base_url=args.base_url,
        serve=args.serve,
        backend=args.backend,
        model=args.model,
        serve_args=args.serve_args,
    )

    # Collect environment
    env = collect_environment(base_url)

    # Load inputs
    inputs, breakdown, input_hash = load_inputs(args.input_path)
    total_inputs = len(inputs)

    # Resolve quant
    quant_resolved = args.quant if args.quant != "auto" else None

    # Print configuration
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
        input_path=args.input_path,
        max_tokens=args.max_tokens,
        runs=args.runs,
        max_concurrency=args.max_concurrency,
        tmux_session=tmux_session,
    )

    # Create client
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    # Progress display
    total_tasks = total_inputs * args.runs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} images"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("  Warmup...", total=total_tasks)

        def progress_callback(phase: str, run_num: int, completed: int) -> None:
            if phase == "warmup":
                progress.update(task_id, description="  Warmup...")
            elif phase == "run":
                progress.update(task_id, description=f"  Run {run_num}/{args.runs}")
            elif phase == "progress":
                progress.update(task_id, completed=completed)

        t_benchmark_start = time.perf_counter()
        runs_raw, retries = run_benchmark(
            client=client,
            model=args.model,
            inputs=inputs,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            runs=args.runs,
            max_concurrency=args.max_concurrency,
            warmup=args.warmup,
            progress_callback=progress_callback,
        )
        t_benchmark_end = time.perf_counter()

    total_duration_s = round(t_benchmark_end - t_benchmark_start, 1)

    # Compute statistics from raw runs
    successful_runs = [r for r in runs_raw if r.error is None]
    error_count = len(runs_raw) - len(successful_runs)

    ttft_values = [r.ttft_ms for r in successful_runs if r.ttft_ms is not None]
    latency_values = [r.latency_s for r in successful_runs]
    prompt_token_values = [r.prompt_tokens for r in successful_runs if r.prompt_tokens > 0]
    completion_token_values = [r.completion_tokens for r in successful_runs if r.completion_tokens > 0]
    cached_token_values = [r.cached_tokens for r in successful_runs]
    reasoning_token_values = [r.reasoning_tokens for r in successful_runs]

    # TPOT calculation: for each run, tpot = (latency - ttft/1000) / completion_tokens * 1000
    tpot_values = []
    for r in successful_runs:
        if r.completion_tokens > 0 and r.ttft_ms is not None:
            total_gen_time_s = r.latency_s - (r.ttft_ms / 1000.0)
            if total_gen_time_s > 0:
                tpot_ms = (total_gen_time_s / r.completion_tokens) * 1000.0
                tpot_values.append(tpot_ms)

    # Tokens per second
    total_completion_tokens = sum(r.completion_tokens for r in successful_runs)
    total_gen_time = sum(r.latency_s for r in successful_runs)
    tokens_per_sec = round(total_completion_tokens / total_gen_time, 1) if total_gen_time > 0 else 0
    inputs_per_sec = round(len(successful_runs) / total_duration_s, 2) if total_duration_s > 0 else 0

    # VRAM peak (try nvidia-smi for the active GPU)
    vram_peak: Optional[int] = None
    try:
        if platform.system() == "Linux":
            gpu_idx = _nvidia_gpu_index()
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_idx}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                vram_peak = int(float(result.stdout.strip().split("\n")[0]))
    except Exception:
        pass

    # Build result
    run_id = secrets.token_hex(6)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = BenchmarkResult(
        run_id=run_id,
        timestamp=timestamp,
        tag=args.tag,
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
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_concurrency=args.max_concurrency,
        ),
        results=Results(
            ttft_ms=compute_stats(ttft_values),
            tpot_ms=compute_stats(tpot_values),
            tokens_per_sec=tokens_per_sec,
            inputs_per_sec=inputs_per_sec,
            latency_s_per_input=compute_stats(latency_values),
            prompt_tokens=compute_token_stat(prompt_token_values),
            completion_tokens=compute_token_stat(completion_token_values),
            cached_tokens=compute_token_stat(cached_token_values),
            reasoning_tokens=compute_token_stat(reasoning_token_values),
            vram_peak_mib=vram_peak,
            total_inputs_processed=len(runs_raw),
            total_duration_s=total_duration_s,
            errors=error_count,
            retries=retries,
        ),
        runs_raw=runs_raw,
    )

    # Save results
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model slug: take last part of model ID, replace non-alnum with dash
    model_slug = re.sub(r"[^a-zA-Z0-9]", "-", args.model.split("/")[-1]).strip("-").lower()
    ts_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"{model_slug}-{ts_str}.json"
    save_path = save_dir / filename

    with open(save_path, "w") as f:
        f.write(json.dumps(dataclasses.asdict(result), indent=2))

    # Print results
    print_results(result, str(save_path))


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare benchmark results from multiple JSON files."""
    results: list[BenchmarkResult] = []

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            console.print(f"[red]File not found: {filepath}[/red]")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        results.append(_dc_from_dict(BenchmarkResult, data))

    if not results:
        console.print("[red]No result files provided.[/red]")
        sys.exit(1)

    # Sort by total tokens_per_sec (across workers) descending
    results.sort(key=lambda r: r.results.tokens_per_sec * r.input.max_concurrency, reverse=True)

    print_compare_table(results)


def main() -> None:
    """Entry point: default to 'run' subcommand if none specified."""
    # Always show the banner first
    print_banner()

    # If no subcommand given (first arg starts with -- or -), insert "run"
    argv = sys.argv[1:]
    subcommands = {"compare"}
    if argv and argv[0] not in subcommands and not argv[0].startswith("--help"):
        if argv[0].startswith("--") or argv[0].startswith("-"):
            argv = ["run"] + argv
    elif not argv:
        # No args at all → show help
        pass

    parser = build_parser()
    parsed = parser.parse_args(argv)

    if parsed.command is None:
        parser.print_help()
        sys.exit(0)
    elif parsed.command == "run":
        cmd_run(parsed)
    elif parsed.command == "compare":
        cmd_compare(parsed)


if __name__ == "__main__":
    main()
