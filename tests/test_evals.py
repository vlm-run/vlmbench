"""Evals: config resolution (providers, models, benchmarks; config ids), the disk cache, and the dev / test samples."""

from pathlib import Path

import pytest

from vlmbench.evals.cache import DiskCache
from vlmbench.evals.config import bench_files, expand, load
from vlmbench.evals.data import Item, sample

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _write(tmp: Path, name: str, text: str) -> Path:
    (tmp / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp / name).write_text(text)
    return tmp / name


def _tree(tmp: Path, bench: str) -> Path:
    """A `configs/` tree: providers `local` and `api`, a model `m` on `local`, and benchmark `mini`."""
    c = tmp / "configs"
    _write(c, "providers/local.yaml", "type: vllm\nargs: [--base]\n")
    _write(c, "providers/api.yaml", "type: remote\nbase_url: http://api\n")
    _write(
        c,
        "models/m.yaml",
        "model: org/m\nweights: open\nprovider: {use: local, args: [--a]}\nrequest: {temperature: 0.5}\n",
    )
    _write(c, "benchmarks/mini/p.md", "Read the page.\n")
    _write(c, "benchmarks/mini/mini.yaml", bench)
    return c / "benchmarks" / "mini"


MINI = """kind: Eval
request: {max_tokens: 100}
serves:
  m: {provider: {args: [--b]}, request: {prompt: p.md, temperature: null}}
  m-api: {model: m, provider: api}
datasets: {x: {kind: omniparse, revision: r}}
"""


def test_a_serve_resolves_provider_model_defaults_then_its_overrides(tmp_path):
    ev, serves, _ = bench_files(_tree(tmp_path, MINI))
    s, api = serves["mini-m"], serves["mini-m-api"]
    assert s.provider.args == ["--base", "--a", "--b"]  # args append across every layer
    assert s.request.temperature is None and s.request.max_tokens == 100  # null removes; defaults apply
    assert s.request.prompt == "Read the page." and s.prompt_file.endswith("p.md")
    assert s.weights == "open" and s.model == "org/m"
    assert api.provider.base_url == "http://api" and api.request.temperature == 0.5  # a provider swapped by name
    assert expand(["m-api"], tmp_path / "configs" / "benchmarks" / "mini", serves) == ["mini-m-api"]


def test_config_id_ignores_name_key_and_concurrency(tmp_path):
    a = load(
        _write(
            tmp_path,
            "a.yaml",
            "model: m\nprovider: {type: remote, base_url: http://x, api_key: k1}\nrequest: {concurrency: 4}\n",
        )
    )
    b = load(
        _write(
            tmp_path,
            "b.yaml",
            "model: m\nprovider: {type: remote, base_url: http://x, api_key: k2}\nrequest: {concurrency: 64}\n",
        )
    )
    c = load(
        _write(tmp_path, "c.yaml", "model: m\nprovider: {type: remote, base_url: http://x}\nrequest: {prompt: other}\n")
    )
    assert a.config_id == b.config_id != c.config_id


def test_unknown_keys_are_errors(tmp_path):
    with pytest.raises(Exception, match="extra"):
        load(_write(tmp_path, "x.yaml", "model: m\nprovider: {type: remote, base_url: http://x}\ntemprature: 0\n"))
    with pytest.raises(Exception, match="extra"):
        bench_files(_tree(tmp_path, MINI.replace("{model: m, provider: api}", "{model: m, prompt: x}")))


def test_shipped_benchmarks_resolve_and_short_names_expand():
    for b in ("ocr-bench", "extract-bench"):
        ev, serves, metrics = bench_files(CONFIGS / "benchmarks" / b)
        assert len(serves) == len(ev.serves) and all(s.name.startswith(f"{b}-") for s in serves.values())
    ev, serves, metrics = bench_files(CONFIGS / "benchmarks" / "ocr-bench")
    assert expand(["gemini-3.8-flash"], CONFIGS / "benchmarks" / "ocr-bench", serves) == ["ocr-bench-gemini-3.8-flash"]
    assert metrics and metrics[0].judge.model == "google/gemini-3.8-flash" and metrics[0].prompt


def test_disk_cache_survives_a_new_instance(tmp_path):
    c = DiskCache(tmp_path / "c.sqlite", maxsize=1)
    c["a"], c["b"] = {"text": "A"}, {"text": "B"}  # "a" is evicted from memory, kept on disk
    assert DiskCache(tmp_path / "c.sqlite")["a"] == {"text": "A"}
    with pytest.raises(KeyError):
        DiskCache(tmp_path / "c.sqlite")["missing"]


def test_dev_and_test_samples_are_disjoint_and_spread():
    items = [Item(f"{c}/{i}", c, "x.pdf") for c in "abc" for i in range(20)]
    test, dev = sample(items, 9, "test"), sample(items, 9, "dev")
    assert len(test) == len(dev) == 9
    assert not {i.id for i in test} & {i.id for i in dev}
    assert {i.category for i in test} == {"a", "b", "c"}
    assert sample(items, 9, "test") == test


def test_lint_passes_the_shipped_tree():
    from vlmbench.evals.lint import lint

    assert lint([CONFIGS], log=lambda m: None) == 0


def test_lint_flags_missing_models_missing_prompts_and_duplicates(tmp_path):
    from vlmbench.evals.lint import lint

    b = _tree(
        tmp_path,
        MINI.replace(
            "m-api: {model: m, provider: api}",
            "m-copy: {model: m, provider: {args: [--b]}, request: {prompt: p.md, temperature: null}}",
        ),
    )
    seen: list[str] = []
    assert lint([b], log=seen.append) == 1 and "same config" in seen[0]
    (b / "mini.yaml").write_text(MINI.replace("model: m, provider: api", "model: nope"))
    seen.clear()
    assert lint([b], log=seen.append) == 1 and "no model 'nope'" in seen[0]
    (b / "mini.yaml").write_text(MINI.replace("prompt: p.md", "prompt: gone.md"))
    seen.clear()
    assert lint([b], log=seen.append) == 1 and "not found" in seen[0]


def test_a_cached_request_is_a_bench_request_span_with_its_results(tmp_path, capfire):
    import asyncio

    import logfire
    from PIL import Image

    from vlmbench.evals import trace
    from vlmbench.evals.client import Client, reply_key

    trace._LOGFIRE = logfire  # capfire has configured logfire to capture in memory
    try:
        serve = load(_write(tmp_path, "s.yaml", "model: m\nprovider: {type: remote, base_url: http://x}\n"))
        png = tmp_path / "p.png"
        Image.new("RGB", (8, 8)).save(png)
        cache = DiskCache(tmp_path / "c.sqlite")
        cache[reply_key(serve, png)] = {"text": "hello", "finish": "stop", "completion_tokens": 3, "latency_s": 0.1}
        rec = asyncio.run(Client(cache).complete(serve, png, item="omniparse/p1"))
        assert rec["cached"] and rec["text"] == "hello"
        spans = {s["name"]: s["attributes"] for s in capfire.exporter.exported_spans_as_dict()}
        attrs = spans["vlmbench.request"]
        assert attrs["item"] == "omniparse/p1" and attrs["config_id"] == serve.config_id
        assert attrs["cached"] is True and attrs["completion_tokens"] == 3 and attrs["chars"] == 5
    finally:
        trace._LOGFIRE = None


def test_json_replies_parse_leniently_and_windows_merge():
    from vlmbench.evals.jsonx import merge, parse

    assert parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse('Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert parse("no json") is None
    assert parse('{"a": ' + "9" * 16288 + "}") is None  # a runaway reply is unparsed, never a crash
    a = {"name": "ACME", "rows": [{"x": 1}], "meta": {"date": None}}
    b = {"name": "other", "rows": [{"x": 2}], "meta": {"date": "2026-01-01"}}
    assert merge(a, b) == {"name": "ACME", "rows": [{"x": 1}, {"x": 2}], "meta": {"date": "2026-01-01"}}


def test_a_document_is_read_in_page_windows(tmp_path):
    from vlmbench.evals.runner import windows

    s = load(
        _write(
            tmp_path,
            "s.yaml",
            "model: m\nprovider: {type: remote, base_url: http://x}\nrequest: {pages_per_call: 2}\n",
        )
    )
    ws = windows(s, [tmp_path / f"p{i}.png" for i in range(5)], "schema")
    assert [len(w) for w, _ in ws] == [2, 2, 1] and ws[2][1].endswith("pages 5-5 of 5.")
    whole = load(_write(tmp_path, "w.yaml", "model: m\nprovider: {type: remote, base_url: http://x}\n"))
    assert windows(whole, [tmp_path / "p.png"] * 3, "schema") == [([tmp_path / "p.png"] * 3, "schema")]
    pages = [tmp_path / f"q{i}.png" for i in range(3)]
    assert windows(s, pages, None) == [
        ([p], None) for p in pages
    ]  # a text document: one page per request, no instruction


def test_longarray_scores_rows_order_insensitively():
    from vlmbench.evals.datasets.longarray import score_document

    schema = {"properties": {"t": {"type": "array", "items": {"properties": {"a": {}, "b": {}}}}}}
    gold = {"t": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}
    assert score_document({"t": [{"a": 2, "b": "y"}, {"a": 1, "b": "x"}]}, gold, schema, {}) == (4, 4)
    assert score_document({"t": [{"a": 1, "b": "x"}]}, gold, schema, {}) == (2, 4)  # a dropped row scores 0
    assert score_document({"value": {"t": []}}, gold, schema, {}) == (0, 4)


def test_a_field_added_at_its_default_keeps_earlier_config_ids(tmp_path):
    base = "model: m\nprovider: {type: remote, base_url: https://gateway.vlm.run/v1/openai}\n"
    a = load(_write(tmp_path, "a.yaml", base))
    b = load(_write(tmp_path, "b.yaml", base + "request: {pages_per_call: 2}\n"))
    assert a.config_id != b.config_id and a.provider_name == "vlmrun"
    d = a.model_dump(mode="json")
    assert "pages_per_call" in d["request"]  # in the dump, but not in the id at its default


def test_a_budget_is_never_crossed_by_calls_in_flight(tmp_path):
    import asyncio

    from vlmbench.evals.cache import DiskCache
    from vlmbench.evals.client import BUDGET, Budget, Client

    s = load(_write(tmp_path, "s.yaml", "model: m\nprovider: {type: remote, base_url: http://x}\n"))
    pngs = []
    for i in range(40):
        (p := tmp_path / f"p{i}.png").write_bytes(bytes([i]))
        pngs.append(p)
    client = Client(DiskCache(tmp_path / "c.sqlite"), budget=Budget(1.0))

    async def fake(serve, image, text, attempts=3):
        await asyncio.sleep(0.01)
        return {"text": "ok", "cost_usd": 0.15}

    client._call = fake

    async def go():
        return await asyncio.gather(*(client.complete(s, p) for p in pngs))

    recs = asyncio.run(go())
    assert client.budget.spent <= 1.0 and sum(r.get("error") == BUDGET for r in recs) == 34
    assert all(r.get("budget") for r in recs if "error" in r)
    assert not client.cached(s, pngs[-1])  # a skipped call is called again by a run with more budget


def test_dataset_draws_rebuild_the_interval_for_any_set_of_datasets():
    from vlmbench.evals.results import draws, interval

    a = {f"a{i}": {"x": [i % 3 == 0, 1]} for i in range(30)}
    b = {f"b{i}": {"y": [i % 2, 1], "z": [i % 5 == 0, 1]} for i in range(20)}
    da, db = draws(40.0, a, 4000, seed=1), draws(70.0, b, 4000, seed=2)
    m = sorted((x + y) / 2 for x, y in zip(da, db))
    lo, hi = interval([(40.0, a), (70.0, b)], draws=4000)
    assert abs(m[100] - lo) < 1.5 and abs(m[3899] - hi) < 1.5
    assert draws(50.0, {}) == []
