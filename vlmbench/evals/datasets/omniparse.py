"""OmniParseBench (datalab-to/omni_parse_bench), graded by `omni-parse-bench`: the mean of its text and tables
families, on the HTML its harness derives from a markdown reply (layout needs boxes, so it is unsupported)."""

from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "datalab-to/omni_parse_bench"


@cache
def _root(revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision, max_workers=8))


def items(spec: DatasetSpec) -> list[Item]:
    from omni_parse_bench import benchmark

    root, pages = benchmark.manifest(str(_root(spec.revision)))
    return [Item(p.page_id, p.suite, str(root / p.doc_path)) for p in pages]


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    from omni_parse_bench import benchmark, metric, vocab
    from omni_parse_bench.harness.providers.llm import as_html

    _, pages, by_page = benchmark.load(str(_root(spec.revision)))
    verdicts, tests, units = [], [], {}
    for it in its:
        ts, text = by_page[it.id], replies.get(it.id)
        vs = metric.score({"html": as_html(text) if text else None}, ts, unsupported={"blocks"})
        for t, v in zip(ts, vs):  # a family scores flat over its scored (pass / fail) tests
            if v.result in ("pass", "fail") and (f := vocab.FAMILIES[t.test_type]) in ("text", "tables"):
                c = units.setdefault(it.id, {}).setdefault(f, [0, 0])
                c[0], c[1] = c[0] + (v.result == "pass"), c[1] + 1
        verdicts += vs
        tests += ts
    ids = {it.id for it in its}
    s = metric.summarize(verdicts, tests, [p for p in pages if p.page_id in ids])
    fam = {k: 100 * v["score"] for k, v in s["headline"]["families"].items() if v["score"] is not None}
    md = [fam[k] for k in ("text", "tables") if k in fam]
    return {
        "score": sum(md) / len(md) if md else 0.0,
        "metrics": {k: v for k, v in fam.items() if k != "layout"},
        "by_category": {k: 100 * v for k, v in s.get("per_suite", {}).items()},
        "units": units,
    }
