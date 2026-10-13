"""Built-in providers. Each is a base_url plus optional hooks; add more with `-c`."""

from typing import Any

from openai import AsyncOpenAI

from vlmbench.eval import Provider, Sample, Subject

_OCR_METHODS = {"markdown", "ocr", "grounding", "layout"}


def _extras(entry: Any) -> dict[str, Any]:
    return getattr(entry, "model_extra", None) or {}


class VlmRun(Provider):
    """VLM Run gateway — reports `usage.cost` and per-model `methods`."""

    name = "vlmrun"
    base_url = "https://gateway.vlm.run/v1/openai"
    api_key_env = "VLMRUN_API_KEY"
    concurrency = 8

    @classmethod
    async def list_models(cls, client: AsyncOpenAI) -> list[str]:
        page = await client.models.list()
        return [
            entry.id
            for entry in page.data
            if _OCR_METHODS & set(_extras(entry).get("methods") or ())
        ]


class OpenRouter(Provider):
    """OpenRouter — always returns `usage.cost`; upstream routing is pinned per run."""

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
    headers = {"HTTP-Referer": "https://vlm.run", "X-Title": "vlmbench"}

    @classmethod
    def request_kwargs(cls, subject: Subject, sample: Sample) -> dict[str, Any]:
        return {"extra_body": {"provider": {"allow_fallbacks": False}}}

    @classmethod
    async def list_models(cls, client: AsyncOpenAI) -> list[str]:
        page = await client.models.list()
        return [
            entry.id
            for entry in page.data
            if "image" in (_extras(entry).get("architecture") or {}).get("input_modalities", [])
        ]


class OpenAIProvider(Provider):
    name = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"


class Ollama(Provider):
    """Local Ollama. No cost is reported and none is imputed."""

    name = "ollama"
    base_url = "http://localhost:11434/v1"
    concurrency = 2
    supports_stream_usage = False


class Vllm(Provider):
    """Local vLLM. No cost is reported and none is imputed."""

    name = "vllm"
    base_url = "http://localhost:8000/v1"
    concurrency = 8
