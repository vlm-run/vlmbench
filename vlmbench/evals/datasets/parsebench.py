"""ParseBench (llamaindex/ParseBench), graded by `parse_bench`: the four dimensions a markdown reply can answer,
each by the metric its leaderboard reports, overall their mean. Visual grounding needs boxes, so it is left out."""

import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "llamaindex/ParseBench"
HEADLINE = {
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
    "table": "grits_trm_composite",
    "chart": "rule_pass_rate",
}
GRADED = {"text": ("text_content", "text_formatting"), "table": ("table",), "chart": ("chart",)}


@cache
def _root(revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision))


@cache
def _cases(revision: str) -> dict:
    from parse_bench.test_cases.loader import load_test_cases

    return {(tc.group, tc.test_id): tc for tc in load_test_cases(_root(revision))}


def items(spec: DatasetSpec) -> list[Item]:
    """Single-page documents of the graded categories."""
    import pypdfium2 as pdfium

    out = []
    for f in sorted((_root(spec.revision) / "docs").glob("*/*")):
        if f.parent.name not in GRADED:
            continue
        if f.suffix.lower() == ".pdf" and len(pdfium.PdfDocument(str(f))) != 1:
            continue
        out.append(Item(f"{f.parent.name}/{f.stem}", f.parent.name, str(f)))
    return out


def _evaluate(job: tuple, limit_s: int = 120) -> tuple:
    """(item, dimension, result): a reply the evaluator cannot parse, or that runs it past `limit_s` (a runaway,
    repeated reply), is graded as an empty reply."""

    def expire(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(limit_s)
    try:
        return job[1], job[2], _evaluate_one(job)
    except Exception:
        signal.alarm(0)
        return job[1], job[2], _evaluate_one((*job[:3], ""))
    finally:
        signal.alarm(0)


def _evaluate_one(job: tuple):
    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.schemas.parse_output import PageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType

    revision, item, dim, md = job
    tc = _cases(revision)[dim, item]
    now = datetime.now()
    res = InferenceResult(
        request=InferenceRequest(example_id=item, source_file_path=str(tc.file_path), product_type=ProductType.PARSE),
        pipeline_name="vlmbench",
        product_type=ProductType.PARSE,
        raw_output={},
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
        output=ParseOutput(
            example_id=item, pipeline_name="vlmbench", markdown=md, pages=[PageIR(page_index=0, markdown=md)]
        ),
    )
    return ParseEvaluator().evaluate(res, tc)


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    from parse_bench.evaluation.runner import EvaluationRunner

    cases = _cases(spec.revision)
    jobs = [
        (spec.revision, it.id, dim, (replies.get(it.id) or "").strip())
        for it in its
        for dim in GRADED[it.category]
        if (dim, it.id) in cases
    ]
    agg = EvaluationRunner(output_dir=Path("/tmp"))
    by: dict[str, list] = {}
    units: dict[str, dict[str, list[float]]] = {}
    # spawn: forking a process that runs olmOCR's browser threads deadlocks the workers
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        for item, dim, res in pool.map(_evaluate, jobs):
            by.setdefault(dim, []).append(res)
            if (v := agg._aggregate_metrics([res]).get(f"avg_{HEADLINE[dim]}")) is not None:
                units.setdefault(item, {})[dim] = [v, 1]  # the page's own value; a dimension averages pages
    dims = {d: 100 * agg._aggregate_metrics(rs)[f"avg_{HEADLINE[d]}"] for d, rs in by.items()}
    return {"score": sum(dims.values()) / max(len(dims), 1), "metrics": dims, "by_category": dims, "units": units}
