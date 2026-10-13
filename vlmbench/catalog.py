"""Built-in catalog: providers and the few models with mechanics worth recording.

A bare `-m <id>` needs no Model entry; defaults apply. Add your own with `-c`.
"""

from typing import Any

from openai import AsyncOpenAI

from vlmbench.eval import Model, Prompt, Provider, Sample, Subject

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
        return [entry.id for entry in page.data if _OCR_METHODS & set(_extras(entry).get("methods") or ())]


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


class GlmOcr(Model):
    """One transcription prompt upstream, so `ocr` returns the same bytes as `markdown`."""

    model = "zai-org/glm-ocr"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class LightOnOcr3(Model):
    """Upstream ships two prompts and calls anything else out of distribution."""

    model = "lightonai/lightonocr-3-4b"
    takes_prompt = False


class VlmRunOcrAuto(Model):
    model = "vlm-run/ocr:auto"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class DotsOcr(Model):
    model = "rednote-hilab/dots.mocr"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class PaddleOcrVl(Model):
    model = "paddlepaddle/paddleocr-vl"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class Qwen38_27B(Model):
    """Reasoning model: thinking goes to `message.reasoning` and is off here.

    Left on, it spends ~75 tokens of chain-of-thought per page for no gain in
    transcription accuracy, and a `max_tokens` cap can be consumed by thinking
    alone — which returns `content: None`, i.e. a silently empty transcription.
    """

    model = "qwen/qwen3.8-27b"
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": False},
        "repetition_penalty": 1.05,
    }
    max_tokens = 8192


class Markdown(Prompt):
    """The naive ask, as a baseline for what prompt wording is worth."""

    name = "markdown"
    text = "Convert this page to markdown."


class MarkdownTuned(Prompt):
    """Baseline plus one clause per column the suites score weakest.

    Written against the observed weak columns: completeness, column flow,
    marginal text models drop, flat table rows the cell-neighbour checks can
    align, and no preamble. Measured against `markdown` on qwen3.8-27b at 10
    stratified pages it did NOT reliably win — see `MarkdownLean`.
    """

    name = "markdown-tuned"
    text = (
        "Transcribe the page to markdown, verbatim and complete.\n"
        "Read multi-column layouts one full column at a time.\n"
        "Keep captions, footnotes, headers, footers, page numbers.\n"
        "Tables: pipe rows, one cell per column, repeat merged values.\n"
        "No commentary, no code fences."
    )


class MarkdownLean(Prompt):
    """`markdown-tuned` with the clauses that cost or did nothing removed.

    One clause was dropped on evidence: "keep headers, footers, page numbers"
    fights olmOCR-bench's `headers_footers` split, whose `absent` rules exist
    to check those are *excluded*. Tuned scored 83.3 there against 100.0 for
    both other prompts, and dropping the clause restored it.

    The rest is token thrift, not a measured gain. Across four suites at 10
    stratified pages no prompt reliably beat the 6-token `markdown`: the
    columns that separated them are backed by 3-9 tests (parsebench `tables`
    n=3, omnidocbench `tables` n=6), so a few points is one test flipping.
    This prompt and `markdown-tuned` share a table clause yet scored 18.4 and
    29.5 on omnidocbench — the spread is noise, not instruction-following.
    Comparing prompts needs a far larger `--limit` than 10.
    """

    name = "markdown-lean"
    text = (
        "Transcribe the page to markdown, verbatim and complete.\n"
        "Tables: pipe rows, one cell per column, repeat merged values."
    )
