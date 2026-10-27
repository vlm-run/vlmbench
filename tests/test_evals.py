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
