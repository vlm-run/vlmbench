"""Typed catalog (Eval/Model/Prompt/Provider), async runner, scoring.

Catalog entries self-register by subclassing. All inference goes through AsyncOpenAI.
"""

import asyncio
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from cachetools import LRUCache
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from vlmbench.version import __version__ as VERSION

USER_AGENT = f"vlmbench/{VERSION}"

CACHE_ROOT = Path(os.environ.get("VLMBENCH_HOME", Path.home() / ".vlmbench"))
CATALOG_DIR = CACHE_ROOT / "catalog"
EVALS_DIR = CACHE_ROOT / "evals"
CORPORA_DIR = CACHE_ROOT / "corpora"
RESPONSE_CACHE = CACHE_ROOT / "cache"

_MEDIA_CACHE: LRUCache = LRUCache(maxsize=64)
"""Resolved data URIs are large; keep a bounded in-process cache alongside the disk one."""

HF_DATASET_PREFIX = "hf://datasets/"
_SEPARATORS = re.compile(r"[_:.]+")


class EvalError(RuntimeError):
    """Configuration or catalog resolution failure, reported without a traceback."""


def normalize_model_id(model_id: str) -> str:
    """Lowercase and collapse `_ : .` so provider casing differences stop mattering."""
    return _SEPARATORS.sub("-", model_id.strip().lower())


def parse_model_ref(ref: str) -> tuple[str | None, str, str | None]:
    """Split `provider::model_id::prompt`. Single colons always belong to the model id.

    Two fields are read as `provider::model_id`; a prompt needs all three.
    """
    autoload()
    parts = [part.strip() for part in ref.split("::")]
    if any(not part for part in parts) or len(parts) > 3:
        raise EvalError(f"malformed model ref {ref!r} — expected [provider::]model_id[::prompt]")
    match parts:
        case [model_id]:
            return None, model_id, None
        case [provider, model_id]:
            if provider not in _CATALOG["provider"]:
                known = ", ".join(sorted(_CATALOG["provider"])) or "none"
                raise EvalError(
                    f"unknown provider {provider!r} in {ref!r} — loaded: {known}. "
                    "To name a prompt, give all three: provider::model_id::prompt"
                )
            return provider, model_id, None
        case _:
            return parts[0], parts[1], parts[2]


def parse_dataset_ref(ref: str) -> tuple[str, str | None]:
    """Split an `hf://...@revision` ref into (ref, revision)."""
    base, sep, revision = ref.rpartition("@")
    return (base, revision) if sep and "/" not in revision else (ref, None)


class Sample(BaseModel):
    """One unit of work: a page, plus whatever the metric needs to judge it."""

    model_config = ConfigDict(frozen=True)

    id: str
    media: str
    """Data URI, http(s) URL, or a `text:`-prefixed payload."""
    split: str = "default"
    truth: Any = None
    """Ground truth, in whatever shape `Eval.score` expects."""
    context: Mapping[str, str] = Field(default_factory=dict)
    """Values a Prompt may interpolate, e.g. `{anchor_text}`."""
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class Prediction:
    """What one model returned for one sample."""

    sample_id: str
    text: str = ""
    latency_s: float = 0.0
    ttft_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    cached: bool = False
    error: str | None = None


Verdict = tuple[str, bool | float | None]
"""`(column, value)`: bool counts as a test, float is averaged, None is unsupported."""


_KINDS = ("eval", "model", "prompt", "provider")
_CATALOG: dict[str, dict[str, type]] = {kind: {} for kind in _KINDS}
_ORIGINS: dict[int, str] = {}
_loading_origin = "builtin"


def _register(kind: str, keys: Sequence[str], cls: type) -> None:
    for key in keys:
        if key:
            _CATALOG[kind][key] = cls
    _ORIGINS[id(cls)] = _loading_origin


def origin_of(cls: type) -> str:
    """Where a catalog entry was loaded from."""
    return _ORIGINS.get(id(cls), "builtin")


class Prompt:
    """A model-agnostic prompt. Catalog prompts are swept as a cross product."""

    name: ClassVar[str] = ""
    text: ClassVar[str] = ""
    system: ClassVar[str | None] = None
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _register("prompt", [cls.key()], cls)

    @classmethod
    def key(cls) -> str:
        return cls.name or cls.__name__

    @classmethod
    def render(cls, sample: Sample) -> str:
        """Interpolate `sample.context`. Escape literal braces as `{{` / `}}`."""
        try:
            return cls.text.format_map(dict(sample.context))
        except KeyError as exc:
            missing = exc.args[0]
            raise EvalError(
                f"prompt {cls.key()} references {{{missing}}} but sample {sample.id!r} has no context key {missing!r}"
            ) from exc

    @classmethod
    def sha256(cls) -> str:
        return hashlib.sha256(cls.text.encode()).hexdigest()[:16]


class Model:
    """Request mechanics for one model id. Carries no prompt."""

    name: ClassVar[str] = ""
    model: ClassVar[str] = ""
    provider: ClassVar[str | None] = None
    takes_prompt: ClassVar[bool] = True
    extra_body: ClassVar[Mapping[str, Any]] = {}
    max_tokens: ClassVar[int] = 4096
    temperature: ClassVar[float] = 0.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        keys = [normalize_model_id(cls.model)] if cls.model else []
        if cls.name:
            keys.append(cls.name)
        _register("model", keys, cls)

    @classmethod
    def key(cls) -> str:
        return cls.name or cls.model or cls.__name__


class Provider:
    """Where requests go. Every hook has an OpenAI-compatible default."""

    name: ClassVar[str] = ""
    base_url: ClassVar[str] = ""
    api_key_env: ClassVar[str | None] = None
    headers: ClassVar[Mapping[str, str]] = {}
    concurrency: ClassVar[int] = 4
    """In-flight request cap. Keep it under the provider's rate limit."""
    supports_stream_usage: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _register("provider", [cls.name], cls)

    @classmethod
    def api_key(cls) -> str:
        if cls.api_key_env:
            return os.environ.get(cls.api_key_env) or "vlmrun"
        return "vlmrun"

    @classmethod
    def client(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        max_retries: int = 5,
        timeout: float = 300.0,
    ) -> AsyncOpenAI:
        """One shared async client per provider. The SDK handles retries and pooling."""
        merged = {"User-Agent": USER_AGENT, **dict(cls.headers), **dict(extra_headers or {})}
        return AsyncOpenAI(
            base_url=base_url or cls.base_url,
            api_key=api_key or cls.api_key(),
            default_headers=merged,
            max_retries=max_retries,
            timeout=timeout,
        )

    @classmethod
    def request_kwargs(cls, subject: "Subject", sample: Sample) -> dict[str, Any]:
        """Extra kwargs merged into `chat.completions.create`."""
        return {}

    @classmethod
    def cost_usd(cls, usage: Any) -> float | None:
        """Read `usage.cost` when the provider reports it."""
        if usage is None:
            return None
        extra = getattr(usage, "model_extra", None) or {}
        for key in ("cost", "total_cost"):
            value = extra.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @classmethod
    async def list_models(cls, client: AsyncOpenAI) -> list[str]:
        """Model ids for `-m all`, via the SDK rather than a raw HTTP call."""
        page = await client.models.list()
        return [entry.id for entry in page.data]


class Eval:
    """A dataset plus how to score it. `samples` and `score` are the contract."""

    name: ClassVar[str] = ""
    dataset: ClassVar[str] = ""
    aliases: ClassVar[tuple[str, ...]] = ()
    description: ClassVar[str] = ""
    prompt: ClassVar[str] = ""
    """Default prompt, used when no catalog prompts are loaded."""
    thresholds: ClassVar[tuple[float, float]] = (85.0, 70.0)
    splits: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        keys = [cls.name, *cls.aliases]
        if cls.dataset:
            keys.append(cls.dataset)
        _register("eval", keys, cls)

    def __init__(self, revision: str | None = None) -> None:
        self.revision = revision

    @classmethod
    def key(cls) -> str:
        return cls.name or cls.__name__

    @property
    def cache_dir(self) -> Path:
        path = CORPORA_DIR / self.key()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def repo_id(self) -> str:
        if not self.dataset.startswith(HF_DATASET_PREFIX):
            raise EvalError(f"{self.key()}: dataset {self.dataset!r} is not an {HF_DATASET_PREFIX} ref")
        return self.dataset[len(HF_DATASET_PREFIX) :]

    def hf_file(self, path: str) -> Path:
        """Download one file from the eval's HF dataset repo."""
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision))

    def hf_rows(self, split: str = "train", config: str | None = None) -> Iterator[dict]:
        """Stream the eval's HF dataset rows."""
        from datasets import load_dataset

        yield from load_dataset(self.repo_id, config, split=split, streaming=True, revision=self.revision)

    def resolve_media(self, sample: Sample) -> str:
        """Materialise `sample.media` just before the request, so `--limit` discards cost nothing.

        `data:`, `http(s):` and `text:` pass through; anything else is `<repo path>[#page]`.
        """
        if sample.media.startswith(("data:", "http://", "https://", "text:")):
            return sample.media
        path, _, page = sample.media.rpartition("#")
        cached = self.cache_dir / f"{sample.id.replace('/', '__')}.b64"
        if cached.exists():
            return cached.read_text()
        uri = media_data_uri(self.hf_file(path or sample.media), int(page or 1))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(uri)
        return uri

    def samples(self) -> Iterator[Sample]:
        raise NotImplementedError

    def score(self, sample: Sample, output: str) -> Iterator[Verdict]:
        raise NotImplementedError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def media_data_uri(path: Path, page: int = 1, *, dpi: int = 150, max_long_edge: int = 2048) -> str:
    """Render a PDF page, or load an image, as a JPEG data URI."""
    import base64
    import io

    from PIL import Image

    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = Image.open(path).convert("RGB")
    else:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        try:
            index = min(max(page - 1, 0), len(document) - 1)
            image = document[index].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            document.close()

    if max(image.size) > max_long_edge:
        scale = max_long_edge / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@dataclass(slots=True, frozen=True)
class Subject:
    """One leaderboard row: provider + model + prompt."""

    provider: type[Provider]
    model_id: str
    model: type[Model]
    prompt: type[Prompt] | None
    default_prompt: str = ""

    @property
    def label(self) -> str:
        suffix = f" · {self.prompt.key()}" if self.prompt else ""
        return f"{self.provider.name}::{self.model_id}{suffix}"

    @property
    def slug(self) -> str:
        parts = [self.provider.name, normalize_model_id(self.model_id).replace("/", "-")]
        if self.prompt:
            parts.append(self.prompt.key().lower())
        return "--".join(parts)

    def prompt_text(self, sample: Sample) -> str:
        if not self.model.takes_prompt:
            return ""
        if self.prompt is not None:
            return self.prompt.render(sample)
        return self.default_prompt


def _import_file(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"vlmbench_catalog_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot import catalog file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_catalog(ref: str) -> list[type]:
    """Load a catalog file, directory, or `file.py:ClassName`, returning what it defined."""
    global _loading_origin

    target, _, selector = ref.partition(".py:")
    path = Path(target + ".py") if selector else Path(ref)
    if not path.exists():
        raise EvalError(f"catalog not found: {ref}")

    files = sorted(p for p in path.glob("*.py") if not p.name.startswith("_")) if path.is_dir() else [path]
    defined: list[type] = []
    for file in files:
        before = {kind: dict(entries) for kind, entries in _CATALOG.items()}
        previous, _loading_origin = _loading_origin, str(file)
        try:
            _import_file(file)
        finally:
            _loading_origin = previous
        for kind, entries in _CATALOG.items():
            for key, cls in entries.items():
                if before[kind].get(key) is not cls and cls not in defined:
                    defined.append(cls)

    if selector:
        chosen = [cls for cls in defined if cls.__name__ == selector]
        if not chosen:
            names = ", ".join(sorted(cls.__name__ for cls in defined)) or "nothing"
            raise EvalError(f"{path} defines no {selector!r} — found: {names}")
        return chosen
    return defined


_autoloaded = False


def autoload() -> None:
    """Load built-ins, then `~/.vlmbench/catalog/*.py`, then `$VLMBENCH_CATALOG`. Idempotent."""
    global _autoloaded
    if _autoloaded:
        return
    _autoloaded = True

    import vlmbench.catalog  # noqa: F401
    import vlmbench.evals  # noqa: F401

    if CATALOG_DIR.is_dir():
        load_catalog(str(CATALOG_DIR))
    for ref in filter(None, os.environ.get("VLMBENCH_CATALOG", "").split(os.pathsep)):
        load_catalog(ref)


def resolve_provider(name: str) -> type[Provider]:
    autoload()
    found = _CATALOG["provider"].get(name)
    if found is None:
        known = ", ".join(sorted(_CATALOG["provider"])) or "none"
        raise EvalError(f"unknown provider {name!r} — loaded: {known}")
    return found  # type: ignore[return-value]


def resolve_model(model_id: str) -> type[Model]:
    autoload()
    found = _CATALOG["model"].get(normalize_model_id(model_id)) or _CATALOG["model"].get(model_id)
    if found is not None:
        return found  # type: ignore[return-value]
    return type(f"Model_{normalize_model_id(model_id)}", (Model,), {"model": model_id})


def today() -> str:
    """Run date as `YY-MM-DD`, used as the dataset revision when none is pinned."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%y-%m-%d")


def resolve_eval(ref: str) -> Eval:
    """Resolve `-d` from a short name, an alias, a full `hf://` ref, or a path."""
    autoload()
    base, revision = parse_dataset_ref(ref)
    if base.endswith(".py") or ".py:" in base:
        for cls in load_catalog(base):
            if inspect.isclass(cls) and issubclass(cls, Eval):
                return cls(revision=revision)
        raise EvalError(f"{base} defines no Eval subclass")

    found = _CATALOG["eval"].get(base)
    if found is None:
        known = ", ".join(sorted(k for k in _CATALOG["eval"] if not k.startswith("hf://"))) or "none"
        raise EvalError(f"unknown dataset {base!r} — loaded evals: {known}")
    return found(revision=revision)  # type: ignore[operator]


def loaded(kind: Literal["eval", "model", "prompt", "provider"]) -> dict[str, type]:
    autoload()
    return dict(_CATALOG[kind])


def resolve_prompt(name: str) -> type[Prompt]:
    autoload()
    found = _CATALOG["prompt"].get(name)
    if found is None:
        known = ", ".join(sorted(_CATALOG["prompt"])) or "none — load one with -c"
        raise EvalError(f"unknown prompt {name!r} — registered: {known}")
    return found  # type: ignore[return-value]


def build_subjects(model_refs: Sequence[str], default_provider: str, default_prompt: str) -> list[Subject]:
    """One row per `-m`. The prompt is named explicitly or falls back to the eval's."""
    subjects: list[Subject] = []
    for ref in model_refs:
        provider_name, model_id, prompt_name = parse_model_ref(ref)
        model = resolve_model(model_id)
        provider = resolve_provider(provider_name or model.provider or default_provider)
        if prompt_name and not model.takes_prompt:
            raise EvalError(f"{model_id} takes no prompt, so {ref!r} cannot name {prompt_name!r}")
        subjects.append(
            Subject(
                provider=provider,
                model_id=model_id,
                model=model,
                prompt=resolve_prompt(prompt_name) if prompt_name else None,
                default_prompt=default_prompt,
            )
        )
    return subjects


def _merge(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Later layers win; dict values merge one level deep."""
    out: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                out[key] = {**out[key], **value}
            else:
                out[key] = value
    return out


def build_messages(prompt: str, media: str) -> list[dict[str, Any]]:
    if not media.startswith(("data:", "http://", "https://", "text:")):
        raise EvalError(f"media was never resolved to a URI: {media[:80]!r}")
    content: list[dict[str, Any]] = []
    if media.startswith("text:"):
        content.append({"type": "text", "text": media[len("text:") :]})
    else:
        content.append({"type": "image_url", "image_url": {"url": media}})
    if prompt:
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def response_cache_key(subject: "Subject", sample: Sample, prompt: str) -> str:
    """Fingerprint of everything that changes a response: edit the prompt, miss the cache."""
    parts = (
        subject.provider.name,
        subject.model_id,
        prompt,
        json.dumps(dict(subject.model.extra_body), sort_keys=True),
        sample.id,
        sample.media,
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


async def complete(
    client: AsyncOpenAI,
    subject: Subject,
    sample: Sample,
    media: str,
    *,
    extra_body: Mapping[str, Any],
    stream_usage: bool,
) -> Prediction:
    """One streamed completion: text plus TTFT, tokens and cost."""
    prompt = subject.prompt_text(sample)
    messages = build_messages(prompt, media)
    body = _merge(
        subject.provider.request_kwargs(subject, sample),
        {"extra_body": dict(subject.model.extra_body)},
        {"extra_body": dict(extra_body)},
    )
    kwargs: dict[str, Any] = {
        "model": subject.model_id,
        "messages": messages,
        "max_tokens": subject.model.max_tokens,
        "temperature": subject.model.temperature,
        "stream": True,
        **body,
    }
    if subject.prompt is not None and subject.prompt.system:
        kwargs["messages"] = [{"role": "system", "content": subject.prompt.system}, *messages]
    if stream_usage:
        kwargs["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    usage: Any = None
    try:
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                chunks.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
    except Exception as exc:  # noqa: BLE001 — one failed page must not end the run
        return Prediction(sample_id=sample.id, latency_s=time.perf_counter() - started, error=str(exc))

    return Prediction(
        sample_id=sample.id,
        text="".join(chunks),
        latency_s=time.perf_counter() - started,
        ttft_ms=ttft,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=subject.provider.cost_usd(usage),
    )


@dataclass(slots=True)
class ColumnScore:
    """Aggregate for one leaderboard column."""

    passed: int = 0
    tests: int = 0
    unsupported: int = 0
    values: list[float] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        if self.values:
            return round(sum(self.values) / len(self.values) * 100, 1)
        if self.tests:
            return round(self.passed / self.tests * 100, 1)
        return None


def aggregate(verdicts: Iterator[tuple[str, Verdict]]) -> dict[str, ColumnScore]:
    """Fold `(sample_id, (column, value))` pairs into per-column scores.

    A column is boolean (a pass rate) or continuous (a mean), never both.
    """
    columns: dict[str, ColumnScore] = {}
    kinds: dict[str, str] = {}
    for _sample_id, (column, value) in verdicts:
        entry = columns.setdefault(column, ColumnScore())
        match value:
            case None:
                entry.unsupported += 1
                continue
            case bool():
                kind = "bool"
                entry.tests += 1
                entry.passed += int(value)
            case _:
                kind = "float"
                entry.values.append(float(value))
        seen = kinds.setdefault(column, kind)
        if seen != kind:
            raise EvalError(
                f"column {column!r} mixes {seen} and {kind} verdicts — "
                "yield booleans and continuous scores under separate column names"
            )
    return columns


def overall(columns: Mapping[str, ColumnScore]) -> ColumnScore:
    total = ColumnScore()
    for entry in columns.values():
        total.passed += entry.passed
        total.tests += entry.tests
        total.unsupported += entry.unsupported
        total.values.extend(entry.values)
    return total


def apply_limit(samples: list[Sample], spec: str | None, *, stratify: bool = False) -> tuple[list[Sample], int]:
    """Apply `--limit N` or `--limit N%`, returning (kept, total)."""
    total = len(samples)
    if not spec:
        return samples, total
    if spec.endswith("%"):
        count = max(1, round(total * float(spec[:-1]) / 100))
    else:
        count = int(spec)
    if count >= total:
        return samples, total
    if not stratify:
        return samples[:count], total

    by_split: dict[str, list[Sample]] = {}
    for sample in samples:
        by_split.setdefault(sample.split, []).append(sample)
    kept: list[Sample] = []
    cursors = dict.fromkeys(by_split, 0)
    while len(kept) < count and any(cursors[name] < len(group) for name, group in by_split.items()):
        for name, group in by_split.items():
            if len(kept) >= count:
                break
            if cursors[name] < len(group):
                kept.append(group[cursors[name]])
                cursors[name] += 1
    return kept, total


async def run_subject(
    subject: Subject,
    samples: Sequence[Sample],
    *,
    client: AsyncOpenAI,
    cache_dir: Path | None,
    extra_body: Mapping[str, Any],
    concurrency: int,
    resolve_media: Any = None,
    skip_uncached: bool = False,
    use_cache: bool = True,
    on_done: Any = None,
) -> list[Prediction]:
    """Run one leaderboard row, reusing any cached response for the same fingerprint."""
    semaphore = asyncio.Semaphore(concurrency)
    stream_usage = subject.provider.supports_stream_usage

    async def one(sample: Sample) -> Prediction | None:
        blob = RESPONSE_CACHE / f"{response_cache_key(subject, sample, subject.prompt_text(sample))}.md"
        if use_cache and blob.exists():
            prediction = Prediction(sample_id=sample.id, text=blob.read_text(), cached=True)
        elif skip_uncached:
            return None
        else:
            async with semaphore:
                media = await asyncio.to_thread(resolve_media, sample) if resolve_media else sample.media
                prediction = await complete(
                    client, subject, sample, media, extra_body=extra_body, stream_usage=stream_usage
                )
            if use_cache and prediction.error is None:
                blob.parent.mkdir(parents=True, exist_ok=True)
                blob.write_text(prediction.text)
        if cache_dir is not None and prediction.error is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{sample.id.replace('/', '__')}.md").write_text(prediction.text)
        if on_done:
            on_done(prediction)
        return prediction

    done = await asyncio.gather(*(one(sample) for sample in samples))
    return [prediction for prediction in done if prediction is not None]


async def judge_predictions(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Not implemented: scoring a page with a VLM judge."""
    raise NotImplementedError("--judge is not implemented yet")


@dataclass(slots=True)
class SubjectResult:
    """Everything one leaderboard row contributes to the report."""

    subject: Subject
    columns: dict[str, ColumnScore] = field(default_factory=dict)
    total: ColumnScore = field(default_factory=ColumnScore)
    cost_usd: float | None = None
    latency_total_s: float = 0.0
    latency_p50_s: float = 0.0
    ttft_p50_ms: float | None = None
    errors: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[len(ordered) // 2], 3)


async def run_all(
    ev: Eval,
    subjects: Sequence[Subject],
    samples: Sequence[Sample],
    *,
    run_dir: Path | None = None,
    extra_body: Mapping[str, Any] | None = None,
    score_only: bool = False,
    concurrency: int | None = None,
    judge: str | None = None,
    use_cache: bool = True,
    on_progress: Any = None,
) -> list[SubjectResult]:
    """Run every subject over every sample, then score. One client per provider."""
    clients: dict[str, AsyncOpenAI] = {}
    results: list[SubjectResult] = []

    def cached_resolve(sample: Sample) -> str:
        key = f"{ev.key()}:{sample.id}"
        if key not in _MEDIA_CACHE:
            _MEDIA_CACHE[key] = ev.resolve_media(sample)
        return _MEDIA_CACHE[key]

    try:
        for subject in subjects:
            client = clients.get(subject.provider.name)
            if client is None:
                client = clients[subject.provider.name] = subject.provider.client()
            cache_dir = run_dir / "outputs" / subject.slug if run_dir else None
            predictions = await run_subject(
                subject,
                samples,
                client=client,
                cache_dir=cache_dir,
                extra_body=extra_body or {},
                concurrency=concurrency or subject.provider.concurrency,
                resolve_media=None if score_only else cached_resolve,
                skip_uncached=score_only,
                use_cache=use_cache,
                on_done=(lambda prediction, s=subject: on_progress(s, prediction)) if on_progress else None,
            )
            by_id = {sample.id: sample for sample in samples}
            verdicts = (
                (prediction.sample_id, verdict)
                for prediction in predictions
                if prediction.error is None
                for verdict in ev.score(by_id[prediction.sample_id], prediction.text)
            )
            judged = await judge_predictions(ev, judge, samples, predictions) if judge else {}
            columns = aggregate(itertools.chain(verdicts, (("judge", ("judge", score)) for score in judged.values())))
            costs = [p.cost_usd for p in predictions if p.cost_usd is not None]
            ttfts = [p.ttft_ms for p in predictions if p.ttft_ms is not None]
            results.append(
                SubjectResult(
                    subject=subject,
                    columns=columns,
                    total=overall(columns),
                    cost_usd=round(sum(costs), 6) if costs else None,
                    latency_total_s=round(sum(p.latency_s for p in predictions), 1),
                    latency_p50_s=_p50([p.latency_s for p in predictions if p.error is None]),
                    ttft_p50_ms=_p50(ttfts) if ttfts else None,
                    errors=sum(1 for p in predictions if p.error),
                    cached=sum(1 for p in predictions if p.cached),
                    prompt_tokens=sum(p.prompt_tokens for p in predictions),
                    completion_tokens=sum(p.completion_tokens for p in predictions),
                )
            )
    finally:
        for client in clients.values():
            await client.close()
    return results


def result_payload(
    ev: Eval,
    results: Sequence[SubjectResult],
    *,
    limit_spec: str | None,
    pages: int,
    total_pages: int,
    stratified: bool,
) -> dict[str, Any]:
    """The `result.json` body: one entry per leaderboard row, full refs recorded."""
    import secrets
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "schema_version": "0.2.0",
        "kind": "eval",
        "run_id": secrets.token_hex(6),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vlmbench": VERSION,
        "eval": {
            "name": ev.key(),
            "dataset": ev.dataset,
            "revision": ev.revision or now.strftime("%y-%m-%d"),
            "thresholds": {"good": ev.thresholds[0], "fair": ev.thresholds[1]},
        },
        "limit": {
            "spec": limit_spec,
            "mode": "stratified" if stratified else "first",
            "pages": pages,
            "of": total_pages,
            "partial": pages < total_pages,
        },
        "totals": {
            "subjects": len(results),
            "samples": pages,
            "requests": len(results) * pages,
            "cost_usd": round(sum(r.cost_usd or 0 for r in results), 6) or None,
            "latency_total_s": round(sum(r.latency_total_s for r in results), 1),
            "latency_p50_s": _p50([r.latency_p50_s for r in results]),
            "errors": sum(r.errors for r in results),
            "cached": sum(r.cached for r in results),
        },
        "rows": [
            {
                "provider": result.subject.provider.name,
                "base_url": result.subject.provider.base_url,
                "model": result.subject.model_id,
                "prompt": (
                    {
                        "name": result.subject.prompt.key(),
                        "sha256": result.subject.prompt.sha256(),
                        "text": result.subject.prompt.text,
                    }
                    if result.subject.prompt
                    else None
                ),
                "takes_prompt": result.subject.model.takes_prompt,
                "extra_body": dict(result.subject.model.extra_body),
                "scores": {
                    "overall": result.total.score,
                    "columns": {
                        name: {
                            "score": column.score,
                            "tests": column.tests,
                            "passed": column.passed,
                            "samples": len(column.values),
                            "unsupported": column.unsupported,
                        }
                        for name, column in sorted(result.columns.items())
                    },
                },
                "perf": {
                    "latency_total_s": result.latency_total_s,
                    "latency_p50_s": result.latency_p50_s,
                    "ttft_p50_ms": result.ttft_p50_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
                "cost": {
                    "usd": result.cost_usd,
                    "source": "usage" if result.cost_usd is not None else None,
                    "usd_per_1k_pages": (
                        round(result.cost_usd / pages * 1000, 4) if result.cost_usd and pages else None
                    ),
                },
                "errors": result.errors,
                "cached_pages": result.cached,
            }
            for result in results
        ],
    }
