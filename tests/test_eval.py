"""Eval catalog, grammar and scoring tests. No network, no inference."""

import asyncio
from collections.abc import Iterator

import pytest

from vlmbench import Eval, Model, Prompt, Provider, Sample, checks
from vlmbench.eval import (
    EvalError,
    aggregate,
    apply_limit,
    build_subjects,
    normalize_model_id,
    overall,
    parse_dataset_ref,
    parse_model_ref,
    resolve_model,
)


class Fixture(Eval):
    name = "fixture-suite"
    dataset = "hf://datasets/acme/fixture"
    prompt = "Transcribe the page."

    def samples(self) -> Iterator[Sample]:
        for index in range(4):
            yield Sample(
                id=f"page-{index}",
                media=f"text:page {index} body",
                split="a" if index < 2 else "b",
                truth={"needle": f"page {index}"},
            )

    def score(self, sample: Sample, output: str) -> Iterator[tuple[str, bool | float | None]]:
        yield "text", checks.present(output, sample.truth["needle"])
        yield "charts", None


class Quirky(Model):
    model = "acme/fixed-ocr"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class Terse(Prompt):
    text = "Markdown only."


class Fake(Provider):
    name = "fake"
    base_url = "https://fake.test/v1"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("vlmrun::vlm-run/ocr:auto", ("vlmrun", "vlm-run/ocr:auto")),
        ("ollama::qwen3-vl:8b", ("ollama", "qwen3-vl:8b")),
        ("openrouter::deepseek/deepseek-r1:free", ("openrouter", "deepseek/deepseek-r1:free")),
        ("zai-org/glm-ocr", (None, "zai-org/glm-ocr")),
        ("qwen3-vl:8b", (None, "qwen3-vl:8b")),
    ],
)
def test_model_ref_splits_on_double_colon_only(ref, expected):
    assert parse_model_ref(ref) == expected


def test_malformed_model_ref_is_rejected():
    with pytest.raises(EvalError):
        parse_model_ref("openrouter::")


def test_dataset_ref_revision():
    assert parse_dataset_ref("parsebench") == ("parsebench", None)
    assert parse_dataset_ref("parsebench@3f9ab12") == ("parsebench", "3f9ab12")
    assert parse_dataset_ref("hf://datasets/a/b") == ("hf://datasets/a/b", None)


def test_model_id_normalisation_crosses_providers():
    ids = ["Qwen/Qwen3-VL-8B-Instruct", "qwen/qwen3-vl-8b-instruct", "qwen/qwen3_vl_8b_instruct"]
    assert len({normalize_model_id(i) for i in ids}) == 1


def test_quirky_model_resolves_by_any_casing():
    assert resolve_model("ACME/Fixed-OCR") is Quirky
    assert resolve_model("acme/unknown-model").takes_prompt is True


def test_cross_product_skips_prompts_for_models_that_take_none():
    subjects = build_subjects(["fake::acme/fixed-ocr", "fake::acme/general-vlm"], [Terse], "fake", "suite default")
    assert len(subjects) == 2
    fixed, general = subjects
    assert fixed.prompt is None and general.prompt is Terse
    sample = Sample(id="x", media="text:body")
    assert fixed.prompt_text(sample) == ""
    assert general.prompt_text(sample) == "Markdown only."


def test_suite_default_prompt_used_when_catalog_is_empty():
    (subject,) = build_subjects(["fake::acme/general-vlm"], [], "fake", "suite default")
    assert subject.prompt is None
    assert subject.prompt_text(Sample(id="x", media="text:body")) == "suite default"


def test_prompt_reports_missing_context_key():
    class Anchored(Prompt):
        text = "anchor: {anchor_text}"

    assert Anchored.render(Sample(id="x", media="m", context={"anchor_text": "abc"})) == "anchor: abc"
    with pytest.raises(EvalError, match="anchor_text"):
        Anchored.render(Sample(id="x", media="m"))


def test_scoring_counts_bools_and_flags_unsupported():
    suite = Fixture()
    samples = list(suite.samples())
    verdicts = ((s.id, v) for s in samples for v in suite.score(s, "page 0 body"))
    columns = aggregate(verdicts)
    assert columns["text"].tests == 4
    assert columns["text"].passed == 1
    assert columns["text"].score == 25.0
    assert columns["charts"].unsupported == 4
    assert columns["charts"].score is None
    assert overall(columns).tests == 4


def test_column_may_not_mix_bool_and_float():
    with pytest.raises(EvalError, match="mixes"):
        aggregate(iter([("s", ("x", True)), ("s", ("x", 0.5))]))


def test_limit_first_and_percent_and_stratified():
    samples = list(Fixture().samples())
    assert [s.id for s in apply_limit(samples, "2")[0]] == ["page-0", "page-1"]
    assert apply_limit(samples, "50%")[0] == samples[:2]
    assert apply_limit(samples, None) == (samples, 4)
    stratified = apply_limit(samples, "2", stratify=True)[0]
    assert {s.split for s in stratified} == {"a", "b"}


def test_user_agent_identifies_vlmbench():
    from vlmbench.eval import USER_AGENT

    assert USER_AGENT.startswith("vlmbench/")
    assert "openai-python/" in USER_AGENT


def test_provider_client_sends_user_agent_and_provider_headers():
    from vlmbench.eval import USER_AGENT

    class Branded(Provider):
        name = "branded"
        base_url = "https://branded.test/v1"
        headers = {"X-Title": "vlmbench"}

    async def check() -> None:
        client = Branded.client()
        try:
            assert client.default_headers["User-Agent"] == USER_AGENT
            assert client.default_headers["X-Title"] == "vlmbench"
        finally:
            await client.close()

    asyncio.run(check())
