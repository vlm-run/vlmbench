"""Evals: config loading (extends, names, config ids), the disk cache, and the dev / test samples."""

from pathlib import Path

import pytest

from vlmbench.evals.cache import DiskCache
from vlmbench.evals.config import Eval, Serve, expand, load
from vlmbench.evals.data import Item, sample

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "ocr-bench"


def _write(tmp: Path, name: str, text: str) -> Path:
    (tmp / name).write_text(text)
    return tmp / name


def test_extends_merges_appends_args_and_removes_nulls(tmp_path):
    _write(
        tmp_path,
        "base.yaml",
        "kind: Serve\nmodel: m\nbackend: {type: vllm, args: [--a]}\nrequest: {temperature: 0.5}\n",
    )
    _write(tmp_path, "p.md", "Read the page.\n")
    s = load(
        _write(
            tmp_path,
            "bench-m.yaml",
            "kind: Serve\nextends: base.yaml\nbackend: {args: [--b]}\nrequest: {prompt: p.md, temperature: null}\n",
        )
    )
    assert isinstance(s, Serve)
    assert s.name == "bench-m"
    assert s.backend.args == ["--a", "--b"]
    assert s.request.temperature is None
    assert s.request.prompt == "Read the page."
    assert s.prompt_file.endswith("p.md")


def test_config_id_ignores_name_key_and_concurrency(tmp_path):
    a = load(
        _write(
            tmp_path,
            "a.yaml",
            "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x, api_key: k1}\nrequest: {concurrency: 4}\n",
        )
    )
    b = load(
        _write(
            tmp_path,
            "b.yaml",
            "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x, api_key: k2}\nrequest: {concurrency: 64}\n",
        )
    )
    c = load(
        _write(
            tmp_path,
            "c.yaml",
            "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\nrequest: {prompt: other}\n",
        )
    )
    assert a.config_id == b.config_id != c.config_id


def test_unknown_keys_are_errors(tmp_path):
    with pytest.raises(Exception, match="extra"):
        load(
            _write(
                tmp_path,
                "x.yaml",
                "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\ntemprature: 0\n",
            )
        )


def test_shipped_benchmark_loads_and_short_names_expand():
    ev = load(BENCH / "ocr-bench.yaml")
    assert isinstance(ev, Eval)
    known = {f"ocr-bench-{s}": None for s in ev.serves}
    assert expand(["gemini-3.8-flash"], BENCH, known) == ["ocr-bench-gemini-3.8-flash"]
    for s in ev.serves:
        assert isinstance(load(BENCH / f"ocr-bench-{s}.yaml"), Serve)


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


def _bench(tmp: Path, extra: dict[str, str]) -> Path:
    b = tmp / "mini"
    b.mkdir()
    _write(b, "mini.yaml", "kind: Eval\nserves: [a]\ndatasets: {x: {kind: omniparse, revision: r}}\n")
    _write(b, "mini-a.yaml", "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\n")
    for name, text in extra.items():
        _write(b, name, text)
    return b


def test_lint_passes_the_shipped_tree():
    from vlmbench.evals.lint import lint

    assert lint([ROOT / "configs", ROOT / "benchmarks"], log=lambda m: None) == 0


def test_lint_flags_unlisted_serves_missing_prompts_and_duplicates(tmp_path):
    from vlmbench.evals.lint import lint

    b = _bench(
        tmp_path,
        {
            "mini-b.yaml": "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\n",
            "other.yaml": "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\n",
        },
    )
    seen: list[str] = []
    assert lint([b], log=seen.append) == 2
    assert any("not listed" in m for m in seen) and any("named mini-<model>.yaml" in m for m in seen)
    (b / "mini.yaml").write_text("kind: Eval\nserves: [a, b]\ndatasets: {x: {kind: omniparse, revision: r}}\n")
    (b / "other.yaml").unlink()
    (b / "mini-b.yaml").write_text(
        "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\nrequest: {prompt: gone.md}\n"
    )
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
        serve = load(_write(tmp_path, "s.yaml", "kind: Serve\nmodel: m\nbackend: {type: remote, base_url: http://x}\n"))
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
