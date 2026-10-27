"""olmOCR-bench (allenai/olmOCR-bench), graded by `olmocr.bench`: a pass rate per test file, overall their mean.

olmOCR-bench wants running headers and footers absent while the other benchmarks want them present, so a reply may
mark them `<page_header>` / `<page_footer>`; this grader drops the marked spans for every model alike.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "allenai/olmOCR-bench"
_FURNITURE = re.compile(r"<page_(header|footer)>.*?</page_\1>", re.S | re.I)


@cache
def _root(revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision)) / "bench_data"


@cache
def _rows(revision: str) -> dict[str, list[dict]]:
    return {
        f.stem: [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        for f in sorted(_root(revision).glob("*.jsonl"))
    }


def items(spec: DatasetSpec) -> list[Item]:
    root, out = _root(spec.revision), {}
    for name, rows in _rows(spec.revision).items():
        for r in rows:
            out.setdefault(r["pdf"], Item(r["pdf"], name, str(root / "pdfs" / r["pdf"])))
    return list(out.values())


def _tests(revision: str, pdfs: set[str]) -> dict[str, list]:
    from olmocr.bench.tests import BaselineTest, load_single_test

    tests: dict[str, list] = {}
    for name, rows in _rows(revision).items():
        tests[name] = [load_single_test(r) for r in rows if r["pdf"] in pdfs]
    covered = {t.pdf for ts in tests.values() for t in ts if t.type == "baseline"}
    tests["baseline"] = [
        BaselineTest(id=f"{p}_baseline", pdf=p, page=1, type="baseline") for p in sorted(pdfs - covered)
    ]
    return {k: v for k, v in tests.items() if v}


def _passes(test, md: str) -> bool:
    """As olmocr.bench's runner: a test that raises on a reply (e.g. an empty colspan) fails."""
    try:
        return bool(test.run(md)[0])
    except Exception:
        return False


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    md = {it.id: _FURNITURE.sub("", replies.get(it.id) or "") for it in its}
    tests = _tests(spec.revision, set(md))
    with ThreadPoolExecutor(32) as pool:  # as olmocr.bench: math tests render KaTeX in a browser pool
        ok = {g: list(pool.map(lambda t: _passes(t, md.get(t.pdf, "")), ts)) for g, ts in tests.items()}
    rates = {g: 100 * sum(v) / len(v) for g, v in ok.items()}
    units: dict[str, dict[str, list[int]]] = {}
    for g, ts in tests.items():
        for t, passed in zip(ts, ok[g]):
            c = units.setdefault(t.pdf, {}).setdefault(g, [0, 0])
            c[0], c[1] = c[0] + passed, c[1] + 1
    return {"score": sum(rates.values()) / max(len(rates), 1), "metrics": {}, "by_category": rates, "units": units}
