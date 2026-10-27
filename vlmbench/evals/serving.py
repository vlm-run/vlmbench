"""Local serves: one vLLM container at a time on GPU 0, started only when the card is free, removed after."""

import contextlib
import subprocess
import time
from pathlib import Path

import httpx

from vlmbench.evals.config import Serve, Vllm

GPU_FREE_MIB = 2000


def gpu_used_mib() -> int:
    out = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.decode().strip())


def container(serve: Serve) -> str:
    return f"vlmbench-eval-{serve.name}"


def start(serve: Serve, timeout: float = 1800) -> None:
    b = serve.provider
    assert isinstance(b, Vllm)
    if (used := gpu_used_mib()) > GPU_FREE_MIB:
        raise SystemExit(f"GPU 0 holds {used} MiB (> {GPU_FREE_MIB}); free it before serving {serve.name}")
    subprocess.run(["docker", "rm", "-f", container(serve)], capture_output=True)
    weights = b.weights or serve.model
    mounts = ["-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface"]
    if Path(weights).expanduser().is_dir():
        mounts += ["-v", f"{Path(weights).expanduser()}:/weights:ro"]
        weights = "/weights"
    env = [x for k, v in b.env.items() for x in ("-e", f"{k}={v}")]
    cmd = ["docker", "run", "-d", "--name", container(serve), "--gpus", "device=0", "--ipc=host"]
    cmd += ["-p", f"{b.port}:8000", *mounts, *env, b.image, "--model", weights, "--served-model-name", serve.model]
    subprocess.run(cmd + b.args, check=True, capture_output=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        running = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container(serve)], capture_output=True, text=True
        ).stdout.strip()
        if running != "true":
            raise SystemExit(f"{container(serve)} exited:\n{logs(serve, 40)}")
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(f"http://localhost:{b.port}/v1/models", timeout=5).status_code == 200:
                return
        time.sleep(10)
    stop(serve)
    raise SystemExit(f"{serve.name} not ready after {timeout:.0f} s")


def logs(serve: Serve, tail: int = 200) -> str:
    r = subprocess.run(["docker", "logs", "--tail", str(tail), container(serve)], capture_output=True, text=True)
    return r.stdout + r.stderr


def stop(serve: Serve) -> None:
    subprocess.run(["docker", "rm", "-f", container(serve)], capture_output=True)


@contextlib.contextmanager
def served(serve: Serve):
    """A local serve's container for the duration of the block; a remote serve needs nothing."""
    if not serve.local:
        yield
        return
    start(serve)
    try:
        yield
    finally:
        stop(serve)
