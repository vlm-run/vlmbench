"""One module per benchmark, each graded by the benchmark's own package.

A module has `REPO`, `items(spec) -> list[Item]` (ids are the benchmark's own),
`score(spec, items, replies) -> {"score", "metrics", "by_category", "units"}` (`replies` maps an item id to the
reply text; `units[item][group] = [passed, tests]` rebuilds the score for bootstrap intervals), and optionally
`normalize(text) -> text`: the benchmark's own conventions, applied to every model's reply before grading.
A model's own output format is the serve's `postprocess`, applied once when the reply arrives.
"""

from vlmbench.evals.datasets import olmocr_bench, omniparse, parsebench

DATASETS = {"omniparse": omniparse, "olmocr_bench": olmocr_bench, "parsebench": parsebench}


def normalize(kind: str, text: str) -> str:
    return getattr(DATASETS[kind], "normalize", lambda t: t)(text or "")
