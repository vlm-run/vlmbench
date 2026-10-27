"""OmniExtractBench (datalab-to/omni_extract_bench): schema-guided JSON extraction from 620 whole documents pooled
from four suites (ExtractBench, LongArray, Micro1, Datalab's own `internal`), graded by `omni-extract-bench`'s field
matching: every value gets a key path, array rows are Hungarian-paired by content, and a document's accuracy is
matched / (matched + misread + unfound + fabricated + invented). Nulls, `""` and absent keys state nothing and are not
scored, so a blank reply earns nothing and invented rows cost.

The score is the benchmark's UNIFIED number: the mean over suites of the mean accuracy within each (its METRIC_SPEC
§7), so the units are grouped by suite. A reply with no JSON object is a failed document, accuracy 0, as its
`summarize` headline counts it. The schema is sent as its harness sends it: the gold conventions stated in field
descriptions (`apply_overlay`), the grader-only keys removed.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.jsonx import parse

REPO = "datalab-to/omni_extract_bench"
DOCUMENT = True
ERRORS = ("misread", "unfound", "fabricated", "invented_item", "invented_field")


@cache
def _docs(revision: str) -> dict:
    """doc_id -> the benchmark's `Doc` (pdf, gold path, schema, suite), read by its own manifest reader."""
    from huggingface_hub import snapshot_download
    from omni_extract_bench import data

    patterns = ["manifest.parquet", "pdfs/*", "gold/*"]
    snap = snapshot_download(REPO, repo_type="dataset", revision=revision, allow_patterns=patterns, max_workers=8)
    return {d.doc_id: d for d in data.read_manifest(snap)[1]}


def items(spec: DatasetSpec) -> list[Item]:
    return [Item(i, d.suite, str(d.doc_path), doc=True) for i, d in _docs(spec.revision).items()]


@cache
def _sent(revision: str, doc_id: str) -> str:
    from omni_extract_bench.harness import schema as S

    return json.dumps(S.strip_benchmark_keys(S.apply_overlay(_docs(revision)[doc_id].schema)))


def instruction(spec: DatasetSpec, item: Item) -> str:
    return (
        "Extract the data from this document according to the following JSON Schema. Read the field descriptions; "
        "copy values exactly as written; extract every item of every array; return numbers as numbers and "
        "booleans as true/false; use null where the document does not say.\n\nJSON Schema:\n"
        + _sent(spec.revision, item.id)
    )


def _score_one(job: tuple[str, dict | None, dict]) -> dict | None:
    from types import SimpleNamespace

    from omni_extract_bench import gold, score

    gt, pred, schema = job
    if pred is None:
        return None
    m = score(pred, gold(SimpleNamespace(doc_id=gt, gt_path=gt)), schema)
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "total", "matching_exact", *ERRORS)}


@cache
def _pool() -> ProcessPoolExecutor:
    import multiprocessing

    return ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    docs = _docs(spec.revision)
    preds = [parse(replies.get(it.id)) for it in its]
    jobs = [
        (str(docs[i.id].gt_path), p if isinstance(p, dict) else None, docs[i.id].schema) for i, p in zip(its, preds)
    ]
    got = list(_pool().map(_score_one, jobs))
    acc = [g["accuracy"] if g else 0.0 for g in got]
    by: dict[str, list[float]] = {}
    for it, a in zip(its, acc):
        by.setdefault(it.category, []).append(100 * a)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    scored = [g for g in got if g]
    over = lambda k: 100 * mean([g[k] for g in scored])  # noqa: E731
    return {
        "score": mean([mean(v) for v in by.values()]),
        "metrics": {
            "unified": mean([mean(v) for v in by.values()]),
            "accuracy": 100 * mean(acc),
            "coverage": 100 * len(scored) / len(got) if got else 0.0,
            **{k: over(k) for k in ("precision", "recall", "f1")},
            **{f"{e}_rate": 100 * mean([g[e] / g["total"] if g["total"] else 0.0 for g in scored]) for e in ERRORS},
            "inexact_docs": sum(not g["matching_exact"] for g in scored),
        },
        "by_category": {k: mean(v) for k, v in by.items()},
        "units": {it.id: {it.category: [a, 1]} for it, a in zip(its, acc)},
    }
