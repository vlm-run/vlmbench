"""Benchmark configs, validated by pydantic.

Three YAML kinds: `Serve` (how to reach one model, with its prompt), `Eval` (serves x datasets x metrics) and
`Metric` (an LLM judge). A file may `extends:` another file, deep-merged under it: `backend.args` appends, every
other list replaces, and a `null` removes the inherited key. A serve's `name` is its file stem. Files whose stem
starts with `_` are fragments: they are only ever extended, never loaded on their own.
"""

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Vllm(Strict):
    """A local vLLM container on one GPU."""

    type: Literal["vllm"]
    image: str = "vllm/vllm-openai:v0.30.0"
    weights: str | None = None
    """HF repo or a path inside the image; default: the serve's `model`."""
    args: list[str] = []
    env: dict[str, str] = {}
    port: int = 8000
    gpu_hour_usd: float | None = None
    """Prices a local page for the cost axis: this rate x wall time / pages."""


class Remote(Strict):
    """Any OpenAI-compatible endpoint: the VLM Run gateway, OpenRouter, a frontier API."""

    type: Literal["remote"]
    base_url: str
    api_key: str | None = None
    """A literal, `env:VAR`, or `toml:<file>` (its first `api_key`); read at request time, never stored."""
    user_agent: str | None = None
    rpm: int | None = None


class Request(Strict):
    prompt: str | None = None
    """A file (relative to the YAML that sets it) or inline text; the loader replaces it with the text."""
    method: str | None = None
    """A gateway OCR method, sent as `extra_body.method` with no prompt text (text would override it)."""
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = {}
    stream: bool = True
    concurrency: int = 8
    timeout: float = 600
    long_edge: int = 2048
    """Pages are rendered once at 2,048 px; a smaller value downsizes before sending."""


class Serve(Strict):
    kind: Literal["Serve"]
    name: str
    model: str
    backend: Annotated[Vllm | Remote, Field(discriminator="type")]
    request: Request = Request()
    postprocess: str | None = None
    """`module:function(text, image_path) -> text`, e.g. a model's own renderer."""
    prompt_file: str | None = None
    """Set by the loader: where the prompt came from."""

    @property
    def config_id(self) -> str:
        """Everything that changes a reply, nothing that doesn't (name, key, concurrency, pacing)."""
        d = self.model_dump(mode="json", exclude={"name", "prompt_file"})
        for k in ("api_key", "user_agent", "rpm", "port", "gpu_hour_usd"):
            d["backend"].pop(k, None)
        for k in ("concurrency", "timeout", "stream"):
            d["request"].pop(k, None)
        return "c-" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def local(self) -> bool:
        return isinstance(self.backend, Vllm)


Kind = Literal["omniparse", "olmocr_bench", "parsebench"]


class DatasetSpec(Strict):
    kind: Kind
    revision: str
    samples: int = 10
    """Pages per split: `test` reports, `dev` (the next, disjoint pages) is for tuning prompts."""


class Eval(Strict):
    kind: Literal["Eval"]
    name: str
    serves: list[str]
    datasets: dict[str, DatasetSpec]
    metrics: list[str] = []
    """`Metric` files (LLM judges), relative to the eval."""
    seed: int = 0


class Metric(Strict):
    """An LLM judge: a serve reads the page and a reply and answers `{"score": n, "reason": "..."}`."""

    kind: Literal["Metric"]
    name: str
    judge: str
    """A serve file, relative to the metric."""
    prompt: str
    scale: tuple[int, int] = (1, 5)
    prompt_file: str | None = None


Doc = TypeAdapter(Annotated[Serve | Eval | Metric, Field(discriminator="kind")])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLMBENCH_")
    home: Path = Path("~/.vlmbench").expanduser()


def _merge(base: dict, top: dict, path: str = "") -> dict:
    out = dict(base)
    for k, v in top.items():
        p = f"{path}.{k}" if path else k
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v, p)
        elif p == "backend.args" and isinstance(out.get(k), list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _raw(path: Path) -> dict:
    """A file with its `extends` chain merged under it, and file-relative prompts made absolute."""
    d = yaml.safe_load(path.read_text()) or {}
    for sect in (d, d.get("request") or {}):
        p = sect.get("prompt")
        if isinstance(p, str) and "\n" not in p and (path.parent / p).is_file():
            sect["prompt"] = str((path.parent / p).resolve())
    if j := d.get("judge"):
        d["judge"] = str((path.parent / j).resolve())
    parent = d.pop("extends", None)
    return _merge(_raw((path.parent / parent).resolve()), d) if parent else d


def load(path: str | Path) -> Serve | Eval | Metric:
    path = Path(path).resolve()
    d = _raw(path)
    d.setdefault("name", path.stem)
    for sect in (d, d.get("request") or {}):
        p = sect.get("prompt")
        if isinstance(p, str) and Path(p).is_absolute() and Path(p).is_file():
            sect["prompt"] = Path(p).read_text().rstrip("\n")
            d["prompt_file"] = p
    return Doc.validate_python(d)


def bench_files(bench: Path) -> tuple[Eval, dict[str, Serve], list[Metric]]:
    """A benchmark folder: its `<bench>.yaml` eval, the serves it lists (`<bench>-<serve>.yaml`), its metrics."""
    bench = bench.resolve()
    ev = load(bench / f"{bench.name}.yaml")
    assert isinstance(ev, Eval), f"{bench.name}.yaml is not an Eval"
    serves = {}
    for s in ev.serves:
        f = bench / f"{s}.yaml"
        if not f.exists():
            f = bench / f"{bench.name}-{s}.yaml"
        serves[f.stem] = load(f)
    return ev, serves, [load(bench / m) for m in ev.metrics]


def expand(stems: list[str] | None, bench: Path, known: dict) -> list[str]:
    """`-s gemma4-26b` names `<bench>-gemma4-26b`; a full stem works too."""
    if not stems:
        return list(known)
    out = []
    for s in stems:
        name = s if s in known else f"{bench.name}-{s}"
        if name not in known:
            raise SystemExit(f"no serve {s!r} in {bench}; have: {', '.join(known)}")
        out.append(name)
    return out


def api_key(ref: str | None) -> str | None:
    if not ref:
        return None
    kind, _, rest = ref.partition(":")
    if kind == "env":
        return os.environ[rest]
    if kind == "toml":
        d = tomllib.loads(Path(rest).expanduser().read_text())
        return d.get("api_key") or next(v["api_key"] for v in d.values() if isinstance(v, dict) and "api_key" in v)
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m[1]], ref)
