"""ExtractBench (llamaindex/ExtractBench): schema-guided JSON extraction from whole enterprise documents, graded by
`extract_bench` itself: unified value F1 per document, the split score their mean.

Each document is a test case with its own JSON Schema, sent as the instruction. The cases are materialised with
ExtractBench's own sidecar writer, from the pinned snapshot, so its loader and scorer read them unchanged.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.jsonx import parse

REPO = "llamaindex/ExtractBench"
DOCUMENT = True
SUBSETS = ("short", "medium", "long")


@cache
def _cases(revision: str, subsets: tuple[str, ...], repo: str = REPO) -> dict[str, tuple[Path, Path, dict]]:
    """id -> (pdf, test.json, row) for every case in the subsets, written once beside the HF cache."""
    from extract_bench.data.download import _write_sidecar
    from huggingface_hub import constants, snapshot_download

    patterns = [f"{s}.jsonl" for s in subsets] + [f"docs/{s}/**" for s in subsets]
    snap = Path(snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=patterns, max_workers=8))
    out = {}
    for s in subsets:
        split_dir = Path(constants.HF_HOME) / "vlmbench" / repo.split("/")[1].lower() / revision[:12] / s
        for line in (snap / f"{s}.jsonl").read_text().splitlines():
            row = json.loads(line)
            sidecar = split_dir / Path(row["pdf"]).relative_to(Path("docs") / s).with_suffix(".test.json")
            if not sidecar.exists():
                sidecar = _write_sidecar(row, split_dir, snap)
            out[row["id"]] = (sidecar.with_name(sidecar.name.removesuffix(".test.json") + ".pdf"), sidecar, row)
    return out


def _all(spec: DatasetSpec, repo: str = REPO, subsets: tuple[str, ...] = SUBSETS):
    return _cases(spec.revision, tuple(spec.subsets or subsets), repo)


def items(spec: DatasetSpec, repo: str = REPO, subsets: tuple[str, ...] = SUBSETS) -> list[Item]:
    return [Item(i, row["category"], str(pdf), doc=True) for i, (pdf, _, row) in _all(spec, repo, subsets).items()]


def instruction(spec: DatasetSpec, item: Item, repo: str = REPO, subsets: tuple[str, ...] = SUBSETS) -> str:
    schema = _all(spec, repo, subsets)[item.id][2]["data_schema"]
    return (
        "Extract the data this JSON Schema describes from the document. Fill every field the document supports, "
        "use null where it does not, and include every record of every array.\n\nJSON Schema:\n" + schema
    )


def _score_one(job: tuple[str, str]) -> tuple[float, float, float]:
    from extract_bench.evaluation.score_case import load_extract_test_case, score_extract_prediction

    sidecar, reply = job
    case = load_extract_test_case(Path(sidecar))
    m = score_extract_prediction(case, parse(reply) or {})["metrics"]
    return tuple(100 * (m.get(f"extract_unified_value_{k}") or 0.0) for k in ("f1", "precision", "recall"))


@cache
def _pool() -> ProcessPoolExecutor:
    import multiprocessing

    return ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))


def score(
    spec: DatasetSpec, its: list[Item], replies: dict[str, str], repo: str = REPO, subsets: tuple[str, ...] = SUBSETS
) -> dict:
    cases = _all(spec, repo, subsets)
    jobs = [(str(cases[it.id][1]), replies.get(it.id) or "") for it in its]
    got = list(_pool().map(_score_one, jobs))
    units = {it.id: {"value_f1": [f1 / 100, 1]} for it, (f1, _, _) in zip(its, got)}
    by: dict[str, list[float]] = {}
    for it, (f1, _, _) in zip(its, got):
        by.setdefault(it.category, []).append(f1)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "score": mean([g[0] for g in got]),
        "metrics": {
            "value_f1": mean([g[0] for g in got]),
            "precision": mean([g[1] for g in got]),
            "recall": mean([g[2] for g in got]),
        },
        "by_category": {k: mean(v) for k, v in by.items()},
        "units": units,
    }
