"""One async path for every backend: cache, pacing, retries, streaming (for TTFT), usage and cost, tracing."""

import asyncio
import base64
import importlib
import io
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from vlmbench.evals.cache import DiskCache, key
from vlmbench.evals.config import Remote, Serve, api_key
from vlmbench.evals.trace import record, span

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:markdown|md|html)?\s*\n(.*?)\n```\s*$", re.S)


def clean(text: str) -> str:
    """The page alone: no leaked reasoning, no fence around the whole reply."""
    text = _THINK.sub("", text or "").strip()
    m = _FENCE.match(text)
    return (m.group(1) if m else text).strip()


def data_url(path: Path, long_edge: int) -> str:
    from PIL import Image

    img = Image.open(path)
    if max(img.size) > long_edge:
        img.thumbnail((long_edge, long_edge))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        raw = buf.getvalue()
    else:
        raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


class Pace:
    """At most `rpm` request starts a minute per endpoint, shared by every serve that calls it."""

    _all: dict[str, "Pace"] = {}

    def __init__(self, rpm: int | None):
        self.gap, self.next, self.lock = (60 / rpm if rpm else 0), 0.0, asyncio.Lock()

    @classmethod
    def of(cls, serve: Serve) -> "Pace":
        b = serve.backend
        url = b.base_url if isinstance(b, Remote) else f"local:{b.port}"
        return cls._all.setdefault(url, cls(getattr(b, "rpm", None)))

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.next > now:
                await asyncio.sleep(self.next - now)
            self.next = max(now, self.next) + self.gap


def _client(serve: Serve):
    from openai import AsyncOpenAI

    b = serve.backend
    if isinstance(b, Remote):
        headers = {"User-Agent": b.user_agent} if b.user_agent else None
        return AsyncOpenAI(base_url=b.base_url, api_key=api_key(b.api_key) or "none", default_headers=headers)
    return AsyncOpenAI(base_url=f"http://localhost:{b.port}/v1", api_key="none")


def _post(ref: str):
    mod, _, fn = ref.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _images(image: Path | list[Path]) -> list[Path]:
    return image if isinstance(image, list) else [image]


def reply_key(serve: Serve, image: Path | list[Path], text: str | None = None) -> str:
    """Everything that changes the reply: the serve's config id, the pages' bytes, any extra text."""
    import hashlib

    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in _images(image)]
    return key(serve.config_id, digests[0] if len(digests) == 1 else digests, text)


class Client:
    def __init__(self, cache: DiskCache, refresh: bool = False):
        self.cache, self.refresh, self.clients = cache, refresh, {}
        self.calls: dict[str, list[dict]] = defaultdict(list)
        """Live (uncached) calls per serve, for the performance table."""

    def cached(self, serve: Serve, image: Path | list[Path], text: str | None = None) -> bool:
        if self.refresh:
            return False
        try:
            self.cache[reply_key(serve, image, text)]
            return True
        except KeyError:
            return False

    async def complete(
        self, serve: Serve, image: Path | list[Path], text: str | None = None, item: str | None = None
    ) -> dict:
        """The serve's reply to one page (and `text`, appended after the serve's own prompt when given)."""
        k = reply_key(serve, image, text)
        with span("vlmbench.request", serve=serve.name, model=serve.model, config_id=serve.config_id, item=item) as s:
            rec = None
            if not self.refresh:
                try:
                    rec = {**self.cache[k], "cached": True}
                except KeyError:
                    pass
            if rec is None:
                rec = {**await self._call(serve, image, text), "cached": False}
                if "error" not in rec:  # a length-cut reply is kept (it would only be cut again); errors retry
                    self.cache[k] = {x: v for x, v in rec.items() if x != "cached"}
                self.calls[serve.name].append(rec)
            record(s, **{x: v for x, v in rec.items() if x != "text"}, chars=len(rec.get("text") or ""))
            return rec

    async def _call(self, serve: Serve, image: Path | list[Path], text: str | None, attempts: int = 3) -> dict:
        from openai import RateLimitError

        r = serve.request
        client = self.clients.setdefault(serve.name, _client(serve))
        parts = [{"type": "image_url", "image_url": {"url": data_url(p, r.long_edge)}} for p in _images(image)]
        prompt = "\n\n".join(x for x in (r.prompt, text) if x)
        if prompt and not (r.method and not text):
            parts.append({"type": "text", "text": prompt})
        body = {"model": serve.model, "messages": [{"role": "user", "content": parts}]}
        body |= {k: v for k, v in (("max_tokens", r.max_tokens), ("temperature", r.temperature)) if v is not None}
        extra = dict(r.extra_body) | ({"method": r.method} if r.method and not text else {})
        tries, limited = 0, 0
        while True:
            await Pace.of(serve).wait()
            t0 = time.perf_counter()
            try:
                rec = await (self._stream if r.stream else self._once)(client, body, extra, r.timeout, t0)
                break
            except RateLimitError:
                limited += 1
                if limited > 10:
                    return {"error": "rate limited 10 times", "latency_s": time.perf_counter() - t0}
                await asyncio.sleep(20 + 20 * random.random())
            except Exception as e:
                tries += 1
                if tries == attempts:
                    return {"error": f"{type(e).__name__}: {e}"[:400], "latency_s": time.perf_counter() - t0}
                await asyncio.sleep(2 * tries)
        if serve.postprocess and rec["text"]:
            rec["text"] = _post(serve.postprocess)(rec["text"], str(_images(image)[0]))
        rec["text"] = clean(rec["text"])
        return rec

    @staticmethod
    async def _once(client, body: dict, extra: dict, timeout: float, t0: float) -> dict:
        resp = await client.chat.completions.create(**body, extra_body=extra or None, timeout=timeout)
        return _record(resp.choices[0].message.content or "", resp.choices[0].finish_reason, resp.usage, t0, None)

    @staticmethod
    async def _stream(client, body: dict, extra: dict, timeout: float, t0: float) -> dict:
        stream = await client.chat.completions.create(
            **body, extra_body=extra or None, timeout=timeout, stream=True, stream_options={"include_usage": True}
        )
        chunks, finish, usage, ttft = [], None, None, None
        async for c in stream:
            if c.usage:
                usage = c.usage
            for ch in c.choices:
                d = ch.delta
                if ttft is None and (
                    d.content or getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                ):
                    ttft = time.perf_counter() - t0
                if d.content:
                    chunks.append(d.content)
                finish = ch.finish_reason or finish
        return _record("".join(chunks), finish, usage, t0, ttft)


def _record(text: str, finish: str | None, u, t0: float, ttft: float | None) -> dict:
    details = getattr(u, "completion_tokens_details", None) if u else None
    extra = (u.model_extra or {}) if u else {}
    return {
        "text": text,
        "finish": finish,
        "latency_s": round(time.perf_counter() - t0, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "prompt_tokens": u.prompt_tokens if u else 0,
        "completion_tokens": u.completion_tokens if u else 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        "cost_usd": getattr(u, "cost", None) or extra.get("cost"),
    }
