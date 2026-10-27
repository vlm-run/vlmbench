"""One module per benchmark, each graded by the benchmark's own package.

A module has `REPO`, `items(spec) -> list[Item]` (ids are the benchmark's own) and
`score(spec, items, replies) -> {"score", "metrics", "by_category", "units"}`, where `replies` maps an item id to
the reply text and `units[item][group] = [passed, tests]` rebuilds the score for bootstrap intervals.
"""

from vlmbench.evals.datasets import olmocr_bench, omniparse, parsebench

DATASETS = {"omniparse": omniparse, "olmocr_bench": olmocr_bench, "parsebench": parsebench}
