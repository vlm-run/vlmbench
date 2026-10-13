"""Eval catalog, grammar and scoring tests. No network, no inference."""

import asyncio
from collections.abc import Iterator

import pytest

from vlmbench import Eval, Model, Prompt, Provider, Run, Sample, checks
from vlmbench.eval import (
    EvalError,
    aggregate,
    apply_limit,
    build_subjects,
    normalize_model_id,
    overall,
    parse_dataset_ref,
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
    from vlmbench import __version__
    from vlmbench.eval import USER_AGENT

    assert USER_AGENT == f"vlmbench/{__version__}"


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


def test_unresolved_media_never_reaches_the_request():
    from vlmbench.eval import build_messages

    assert build_messages("go", "text:body")[0]["content"][0]["type"] == "text"
    assert build_messages("go", "data:image/jpeg;base64,AAA")[0]["content"][0]["type"] == "image_url"
    with pytest.raises(EvalError, match="never resolved"):
        build_messages("go", "docs/layout/page.pdf#1")


def test_cache_key_changes_with_the_prompt():
    from vlmbench.eval import build_subjects, response_cache_key

    sample = Sample(id="p1", media="text:body")
    (plain,) = build_subjects([Run(model="acme/general-vlm")], "fake", "suite default")
    (terse,) = build_subjects([Run(model="acme/general-vlm", prompt=Terse)], "fake", "suite default")
    assert response_cache_key(plain, sample, plain.prompt_text(sample)) != response_cache_key(
        terse, sample, terse.prompt_text(sample)
    )


def test_cache_key_is_stable_for_identical_inputs():
    from vlmbench.eval import build_subjects, response_cache_key

    sample = Sample(id="p1", media="text:body")
    (subject,) = build_subjects([Run(model="acme/general-vlm", prompt=Terse)], "fake", "d")
    prompt = subject.prompt_text(sample)
    assert response_cache_key(subject, sample, prompt) == response_cache_key(subject, sample, prompt)


def test_judge_is_not_implemented():
    import asyncio as aio

    from vlmbench.eval import judge_predictions

    with pytest.raises(NotImplementedError):
        aio.run(judge_predictions())


def test_prompt_is_named_explicitly_per_row():
    fixed, general = build_subjects(
        [Run(model="acme/fixed-ocr"), Run(model="acme/general-vlm", prompt=Terse)], "fake", "suite default"
    )
    assert fixed.prompt is None and general.prompt is Terse
    sample = Sample(id="x", media="text:body")
    assert fixed.prompt_text(sample) == ""
    assert general.prompt_text(sample) == "Markdown only."


def test_suite_default_prompt_used_when_none_is_named():
    (subject,) = build_subjects([Run(model="acme/general-vlm")], "fake", "suite default")
    assert subject.prompt is None
    assert subject.prompt_text(Sample(id="x", media="text:body")) == "suite default"


def test_naming_a_prompt_for_a_promptless_model_is_an_error():
    with pytest.raises(EvalError, match="takes no prompt"):
        build_subjects([Run(model="acme/fixed-ocr", prompt=Terse)], "fake", "d")


def test_unknown_prompt_lists_what_is_registered():
    with pytest.raises(EvalError, match="unknown prompt"):
        build_subjects([Run(model="acme/general-vlm", prompt="Nope")], "fake", "d")


def test_revision_defaults_to_the_run_date():
    from vlmbench.eval import today

    assert len(today()) == 8 and today().count("-") == 2


def test_module_name_accepts_paths_and_dotted_names():
    from vlmbench.eval import module_name

    assert module_name("runs/ocr.py") == "runs.ocr"
    assert module_name("runs.ocr") == "runs.ocr"
    assert module_name("./runs/ocr.py") == "runs.ocr"


def test_run_file_is_loaded_as_an_ordinary_import(tmp_path, monkeypatch):
    from vlmbench.eval import load_runs

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "p.py").write_text(
        "from vlmbench import Prompt\n\n\nclass Terse(Prompt):\n    text = 'Markdown only.'\n"
    )
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "r.py").write_text(
        "from prompts.p import Terse\n\nfrom vlmbench import Run\n\n"
        "RUNS = [Run(model='a/one'), Run(model='a/two', prompt=Terse)]\n"
        "SMOKE = [Run(model='a/one')]\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    runs = load_runs("runs/r.py")
    assert [r.model for r in runs] == ["a/one", "a/two"]
    assert load_runs("runs/r.py:SMOKE")[0].model == "a/one"
    assert runs[1].prompt.text == "Markdown only."


def test_run_file_without_RUNS_says_so(tmp_path, monkeypatch):
    from vlmbench.eval import load_runs

    (tmp_path / "empty.py").write_text("X = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(EvalError, match="RUNS"):
        load_runs("empty.py")


def test_run_file_must_hold_runs(tmp_path, monkeypatch):
    from vlmbench.eval import load_runs

    (tmp_path / "bad.py").write_text("RUNS = ['not-a-run']\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(EvalError, match="must be a list of Run"):
        load_runs("bad.py")


def test_media_without_a_page_suffix_resolves(monkeypatch, tmp_path):
    """`pages/x.png` has no `#page`; rpartition used to put the whole path in the page slot."""
    seen = {}

    class Suite(Eval):
        name = "media-fixture"
        dataset = "hf://datasets/acme/fixture-media"

        def hf_file(self, path):
            seen["path"] = path
            target = tmp_path / "page.png"
            target.write_bytes(b"")
            return target

    monkeypatch.setattr("vlmbench.eval.media_data_uri", lambda path, page, **kw: f"data:image/jpeg;base64,{page}")
    suite = Suite()
    monkeypatch.setattr(type(suite), "cache_dir", property(lambda self: tmp_path))

    assert suite.resolve_media(Sample(id="a", media="images/x.png")).endswith("1")
    assert seen["path"] == "images/x.png"
    assert suite.resolve_media(Sample(id="b", media="docs/y.pdf#3")).endswith("3")
    assert seen["path"] == "docs/y.pdf"


def test_overall_is_the_mean_of_columns_not_pooled_verdicts():
    """A continuous column must not shadow the boolean ones in the headline number."""
    columns = aggregate(
        iter(
            [("s", ("tables", 1.0))]
            + [("s", ("text", False))] * 9
            + [("s", ("text", True))]
            + [("s", ("charts", None))]
        )
    )
    assert columns["tables"].score == 100.0
    assert columns["text"].score == 10.0
    assert overall(columns).score == 55.0
    assert overall(columns).unsupported == 1


def test_cached_predictions_keep_their_measured_cost(tmp_path, monkeypatch):
    """Without the sidecar, every repeat run reported no cost at all."""
    import json as json_module

    from vlmbench.eval import RESPONSE_CACHE, Prediction

    monkeypatch.setattr("vlmbench.eval.RESPONSE_CACHE", tmp_path)
    assert RESPONSE_CACHE is not None
    (tmp_path / "k.md").write_text("page text")
    (tmp_path / "k.json").write_text(json_module.dumps({"cost_usd": 0.0012, "prompt_tokens": 900}))

    measured = json_module.loads((tmp_path / "k.json").read_text())
    prediction = Prediction(sample_id="s", text=(tmp_path / "k.md").read_text(), cached=True, **measured)
    assert prediction.cost_usd == 0.0012
    assert prediction.prompt_tokens == 900
