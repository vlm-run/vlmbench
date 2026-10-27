"""LongArray-Extract (Extend-AI/LongArray-Extract): one long array per document (27 to 2,205 rows), scored as its
README's reference implementation does: data-point accuracy with order-insensitive (Hungarian) row matching and
fuzzy matching for two free-text fields; the score is the mean per-document accuracy (macro). Accuracy does not
penalise extra rows, so the predicted / expected row ratio is reported beside it."""

import json
import re
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.jsonx import parse

REPO = "Extend-AI/LongArray-Extract"
DOCUMENT = True
CONFIGS = ("clinical", "financial", "legal")
FUZZY = {"financial": {"description": 0.95}, "legal": {"court": 0.85}, "clinical": {}}
_PUNCT, _WS = re.compile(r"[^\w\s]", re.U), re.compile(r"\s+", re.U)


@cache
def _rows(revision: str, configs: tuple[str, ...]) -> dict[str, dict]:
    """id -> row (ground truth, target field, schema, pdf path), PDFs written once beside the HF cache."""
    import pyarrow.parquet as pq
    from huggingface_hub import constants, snapshot_download

    snap = Path(snapshot_download(REPO, repo_type="dataset", revision=revision))
    out = {}
    for c in configs:
        schema = json.loads((snap / "schemas" / f"{c}.json").read_text())
        folder = Path(constants.HF_HOME) / "vlmbench" / "longarray" / revision[:12] / c
        folder.mkdir(parents=True, exist_ok=True)
        for r in pq.read_table(snap / c / "test-00000-of-00001.parquet").to_pylist():
            pdf = folder / f"{r['document_id']}.pdf"
            if not pdf.exists():
                pdf.write_bytes(r["pdf"]["bytes"])
            gt = json.loads(r["ground_truth"])
            out[f"{c}/{r['document_id']}"] = {
                "config": c,
                "pdf": pdf,
                "gt": gt,
                "target": r["target_field"],
                "schema": schema,
            }
    return out


def _configs(spec: DatasetSpec) -> tuple[str, ...]:
    return tuple(spec.subsets or CONFIGS)


def items(spec: DatasetSpec) -> list[Item]:
    return [Item(i, r["config"], str(r["pdf"]), doc=True) for i, r in _rows(spec.revision, _configs(spec)).items()]


def instruction(spec: DatasetSpec, item: Item) -> str:
    r = _rows(spec.revision, _configs(spec))[item.id]
    return (
        f"Extract the data this JSON Schema describes from the document. `{r['target']}` is a long list that runs "
        "across many pages: include every row, in document order, none merged, skipped or repeated.\n\n"
        "JSON Schema:\n" + json.dumps(r["schema"])
    )


def _norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip().lower()


def _cell(a, b, field: str, fuzzy: dict) -> bool:
    if field in fuzzy and isinstance(a, str) and isinstance(b, str):
        from rapidfuzz import fuzz

        na, nb = _norm(a), _norm(b)
        return na == nb or (bool(na) and bool(nb) and fuzz.ratio(na, nb) >= fuzzy[field] * 100.0)
    return a == b


def _array(pred: list, exp: list, sub: list[str], fuzzy: dict) -> tuple[int, int]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    total = len(exp) * len(sub)
    if not exp or not sub or not pred:
        return 0, total
    cost = np.empty((len(pred), len(exp)))
    for i, p in enumerate(pred):
        p = p if isinstance(p, dict) else {}
        for j, e in enumerate(exp):
            cost[i, j] = sum(not _cell(p.get(f), e.get(f), f, fuzzy) for f in sub)
    correct = 0
    for i, j in zip(*linear_sum_assignment(cost)):
        p = pred[i] if isinstance(pred[i], dict) else {}
        correct += sum(_cell(p.get(f), exp[j].get(f), f, fuzzy) for f in sub)
    return correct, total


def score_document(pred: dict, gold: dict, schema: dict, fuzzy: dict) -> tuple[int, int]:
    """(correct, total) data points, as the dataset's reference implementation."""
    if isinstance(pred, dict) and set(pred) == {"value"}:
        pred = pred["value"]
    pred = pred if isinstance(pred, dict) else {}
    props = schema.get("properties", {})
    correct = total = 0
    for f in sorted(set(gold) | set(pred)):
        fs = props.get(f, {})
        if fs.get("type") == "array" or isinstance(gold.get(f), list):
            sub = list(fs.get("items", {}).get("properties", {}).keys())
            c, t = _array(pred.get(f) or [], gold.get(f) or [], sub, fuzzy)
        else:
            c, t = int(_cell(pred.get(f), gold.get(f), f, fuzzy)), 1
        correct += c
        total += t
    return correct, total


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    rows = _rows(spec.revision, _configs(spec))
    per, num, den, ratio, by, units = [], 0, 0, [], {}, {}
    for it in its:
        r, pred = rows[it.id], parse(replies.get(it.id)) or {}
        c, t = score_document(pred, r["gt"], r["schema"], FUZZY[r["config"]])
        acc = c / t if t else 0.0
        per.append(acc)
        num, den = num + c, den + t
        got = pred.get(r["target"]) if isinstance(pred.get(r["target"]), list) else []
        ratio.append(len(got) / max(len(r["gt"][r["target"]]), 1))
        by.setdefault(r["config"], []).append(100 * acc)
        units[it.id] = {"accuracy": [acc, 1]}
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "score": 100 * mean(per),
        "metrics": {"macro": 100 * mean(per), "micro": 100 * num / den if den else 0.0, "rows_ratio": mean(ratio)},
        "by_category": {k: mean(v) for k, v in by.items()},
        "units": units,
    }
