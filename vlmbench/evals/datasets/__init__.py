"""One module per benchmark, each graded by the benchmark's own package.

A module has `REPO`, `items(spec) -> list[Item]` (ids are the benchmark's own),
`score(spec, items, replies) -> {"score", "metrics", "by_category", "units"}` (`replies` maps an item id to the
reply text; `units[item][group] = [passed, tests]` rebuilds the score for bootstrap intervals), and optionally
`normalize(text) -> text`: the benchmark's own conventions, applied to every model's reply before grading.
A document-level benchmark (`DOCUMENT = True`) sends every page of an item, with `instruction(spec, item)` (its
JSON schema) after the serve's prompt. A long document parsed to markdown (`DOCUMENT = "text"`) ships its page images
from `pages(spec, item)`; each page is read alone with the serve's own prompt and the replies joined in page order.
A model's own output format is the serve's `postprocess`, applied once when the reply arrives.
"""

from vlmbench.evals.datasets import (
    ccocr,
    extractbench,
    extractbench_pro,
    fintabnet,
    longarray,
    marathondoc,
    mpdocbench,
    olmocr_bench,
    omnidocbench,
    omniparse,
    parsebench,
    rdtablebench,
    realdoc,
)

DATASETS = {
    "omniparse": omniparse,
    "olmocr_bench": olmocr_bench,
    "parsebench": parsebench,
    "omnidocbench": omnidocbench,
    "ccocr": ccocr,
    "rdtablebench": rdtablebench,
    "fintabnet": fintabnet,
    "mpdocbench": mpdocbench,
    "marathondoc": marathondoc,
    "extractbench": extractbench,
    "extractbench_pro": extractbench_pro,
    "longarray": longarray,
    "realdoc": realdoc,
}


def normalize(kind: str, text: str) -> str:
    return getattr(DATASETS[kind], "normalize", lambda t: t)(text or "")
