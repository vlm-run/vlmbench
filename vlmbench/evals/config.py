"""Benchmark configs, validated by pydantic. One root, `configs/`, three kinds of file, each referenced by name:

- `providers/<name>.yaml`: where requests go (the VLM Run gateway, OpenRouter, a local vLLM container).
- `models/<name>.yaml`: one model on one provider: its id, `provider:` (a name, or `{use: <name>, ...}` with
  overrides), its own `request:` defaults, `weights:`, `postprocess:`.
- `benchmarks/<bench>/<bench>.yaml`: an `Eval`: datasets, `request:` defaults, `serves:` (each key a model, or
  `model:` names one, with this benchmark's `request:` / `provider:` / `postprocess:` overrides) and LLM-judge
  `metrics:`. Prompt files sit next to it.

A serve resolves provider -> model -> the eval's `request:` defaults -> its own overrides, each deep-merged over the
last: a provider's `args` append (vLLM keeps an arg's last value), every other list replaces, `null` removes a key.
"""

import copy
import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
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
    """A file (next to the YAML that sets it) or inline text; the loader replaces it with the text."""
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
    pages_per_call: int | None = None
    """A document is read this many pages per request, the windows' JSON merged (arrays concatenated, scalars
    the first value found); default: the whole document in one request."""


class Serve(Strict):
    kind: Literal["Serve"] = "Serve"
    name: str
    model: str
    provider: Annotated[Vllm | Remote, Field(discriminator="type")]
    request: Request = Request()
    postprocess: str | None = None
    """`module:function(text, image_path) -> text`, e.g. a model's own renderer."""
    weights: Literal["open", "closed"] | None = None
    """Whether the model's weights are published, for grouping in reports; never part of the config id."""
    prompt_file: str | None = None
    """Set by the loader: where the prompt came from."""

    @property
    def config_id(self) -> str:
        """Everything that changes a reply, nothing that doesn't (name, key, concurrency, pacing)."""
        d = self.model_dump(mode="json", exclude={"name", "prompt_file", "weights"})
        d["backend"] = d.pop("provider")  # hashed under its earlier name, so ids (and caches) hold across the rename
        for k in ("api_key", "user_agent", "rpm", "port", "gpu_hour_usd"):
            d["backend"].pop(k, None)
        for k in ("concurrency", "timeout", "stream"):
            d["request"].pop(k, None)
        for k in ("pages_per_call",):  # added later: left out at its default so earlier ids (and caches) hold
            if d["request"].get(k) is None:
                d["request"].pop(k, None)
        return "c-" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def local(self) -> bool:
        return isinstance(self.provider, Vllm)

    @property
    def provider_name(self) -> str:
        return provider_name(self.provider.model_dump())


_HOSTS = {
    "gateway.vlm.run": "vlmrun",
    "openrouter.ai": "openrouter",
    "generativelanguage.googleapis.com": "gemini",
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
}


def provider_name(provider: dict) -> str:
    """What serves the model, for reports: `vllm` locally, else the endpoint (`vlmrun`, `openrouter`, its host)."""
    if provider["type"] == "vllm":
        return "vllm"
    from urllib.parse import urlparse

    host = urlparse(provider["base_url"]).hostname or provider["base_url"]
    return _HOSTS.get(host, host)


Kind = Literal[
    "omniparse",
    "olmocr_bench",
    "parsebench",
    "omnidocbench",
    "ccocr",
    "rdtablebench",
    "fintabnet",
    "mpdocbench",
    "marathondoc",
    "extractbench",
    "extractbench_pro",
    "longarray",
    "realdoc",
    "omniextract",
]


class DatasetSpec(Strict):
    kind: Kind
    revision: str
    samples: int = 10
    """Items per split: `test` reports, `dev` (the next, disjoint items) is for tuning prompts; 0 is every item."""
    subsets: list[str] | None = None
    """The benchmark's own sub-splits to draw from (e.g. ExtractBench `short`); default all."""
    long_edge: int = 2048
    """Render size of each page, in px."""


class Entry(Strict):
    """A serve in an eval: a model and this benchmark's overrides on it."""

    model: str | None = None
    """`models/<name>.yaml`; default: the serve's key."""
    provider: str | dict | None = None
    """A name swaps the model's provider; a mapping overrides its fields (`use:` starts from another provider)."""
    request: dict = {}
    postprocess: str | None = None


class Judge(Entry):
    """An LLM judge: a model reads the page and a reply and answers `{"score": n, "reason": "..."}`."""

    judge: str
    """`models/<name>.yaml`."""
    prompt: str
    """A file next to the eval."""
    scale: tuple[int, int] = (1, 5)


class Eval(Strict):
    kind: Literal["Eval"]
    name: str
    request: dict = {}
    """Defaults for every serve, under each serve's own `request:`."""
    serves: dict[str, Entry | None]
    datasets: dict[str, DatasetSpec]
    metrics: dict[str, Judge] = {}
    seed: int = 0
    budget_usd: float | None = None
    """The most a run may spend (billed cost, judges included); `--budget` overrides it. None: no cap."""


class Metric(Strict):
    """A resolved judge: its serve and its prompt text."""

    name: str
    judge: Serve
    prompt: str
    scale: tuple[int, int] = (1, 5)
    prompt_file: str | None = None


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
        elif p == "provider.args" and isinstance(out.get(k), list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _yaml(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"no {path.parent.name.rstrip('s')} {path.stem!r}: {path} does not exist")
    return yaml.safe_load(path.read_text()) or {}


def root_of(path: Path) -> Path:
    """The `configs/` folder a file or benchmark lives in (else `./configs`)."""
    path = path.resolve()
    for p in [path, *path.parents]:
        if p.name == "configs":
            return p
    return Path("configs").resolve()


def bench_dir(bench: str | Path) -> Path:
    """A benchmark by folder, or by name under `configs/benchmarks/`."""
    p = Path(bench)
    if (p / f"{p.name}.yaml").is_file():
        return p.resolve()
    q = Path("configs") / "benchmarks" / str(bench)
    if (q / f"{q.name}.yaml").is_file():
        return q.resolve()
    raise SystemExit(f"no benchmark {str(bench)!r}: not a folder, nor configs/benchmarks/{bench}/")


def _provider(root: Path, ref: str | dict | None, base: dict | None = None) -> dict:
    """A provider by name; or a mapping over `base` (or over `use:`, another provider)."""
    if ref is None:
        return copy.deepcopy(base or {})
    if isinstance(ref, str):
        return _yaml(root / "providers" / f"{ref}.yaml")
    ref = copy.deepcopy(ref)
    use = ref.pop("use", None)
    start = _yaml(root / "providers" / f"{use}.yaml") if use else copy.deepcopy(base or {})
    return _merge(start, ref, "provider")


def _prompt(request: dict, here: Path) -> str | None:
    """A prompt file (relative to `here`) replaced by its text; its path returned."""
    p = request.get("prompt")
    if isinstance(p, str) and "\n" not in p and (here / p).is_file():
        request["prompt"] = (here / p).read_text().rstrip("\n")
        return str((here / p).resolve())
    return None


def resolve(root: Path, here: Path, model: str, entry: Entry, defaults: dict, name: str) -> Serve:
    """provider -> model -> eval defaults -> the entry's overrides."""
    d = _yaml(root / "models" / f"{model}.yaml")
    d["provider"] = _provider(root, entry.provider, _provider(root, d.get("provider")))
    d["request"] = _merge(_merge(d.get("request") or {}, copy.deepcopy(defaults)), copy.deepcopy(entry.request))
    if entry.postprocess:
        d["postprocess"] = entry.postprocess
    d.update(name=name, prompt_file=_prompt(d["request"], here))
    return Serve.model_validate(d)


def load(path: str | Path) -> Serve:
    """One standalone model file, its stem the name (a file in `models/` reads as that model does)."""
    path = Path(path).resolve()
    if path.parent.name == "models":
        return resolve(root_of(path), path.parent, path.stem, Entry(), {}, path.stem)
    d = _yaml(path)
    d.setdefault("name", path.stem)
    d["provider"] = _provider(root_of(path), d.get("provider"))
    d["prompt_file"] = _prompt(d.setdefault("request", {}), path.parent)
    return Serve.model_validate(d)


def bench_files(bench: str | Path) -> tuple[Eval, dict[str, Serve], list[Metric]]:
    """A benchmark: its eval, its serves (`<bench>-<key>`) resolved, its judges."""
    b = bench_dir(bench)
    root, raw = root_of(b), _yaml(b / f"{b.name}.yaml")
    raw.setdefault("name", b.name)
    ev = Eval.model_validate(raw)
    serves = {}
    for k, e in ev.serves.items():
        e = e or Entry()
        serves[f"{b.name}-{k}"] = resolve(root, b, e.model or k, e, ev.request, f"{b.name}-{k}")
    metrics = []
    for n, j in ev.metrics.items():
        req = {"prompt": j.prompt}
        pf = _prompt(req, b)
        judge = resolve(root, b, j.judge, j, {}, f"{b.name}-judge-{n}")
        metrics.append(Metric(name=n, judge=judge, prompt=req["prompt"], scale=j.scale, prompt_file=pf))
    return ev, serves, metrics


def expand(keys: list[str] | None, bench: str | Path, known: dict) -> list[str]:
    """`-s gemma4-26b` names `<bench>-gemma4-26b`; the full name works too."""
    if not keys:
        return list(known)
    name, out = bench_dir(bench).name, []
    for s in keys:
        full = s if s in known else f"{name}-{s}"
        if full not in known:
            raise SystemExit(f"no serve {s!r} in {name}; have: {', '.join(k.removeprefix(name + '-') for k in known)}")
        out.append(full)
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
