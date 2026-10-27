"""ExtractBench Pro (llamaindex/ExtractBenchPro): the hard companion to ExtractBench, same task and scorer
(unified value F1 per document), over six challenge categories: document conventions, house vocabulary, house
counting rules, drawn relations, charts inside documents and long chart sheets. Its rows have ExtractBench's shape,
so ExtractBench's module reads them, pointed at this repo."""

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.datasets import extractbench as _eb

REPO = "llamaindex/ExtractBenchPro"
DOCUMENT = True
SUBSETS = (
    "document_conventions",
    "house_vocabulary",
    "house_counting_rules",
    "drawn_relations",
    "charts_inside_documents",
    "long_chart_sheets",
)


def items(spec: DatasetSpec) -> list[Item]:
    return _eb.items(spec, REPO, SUBSETS)


def instruction(spec: DatasetSpec, item: Item) -> str:
    return _eb.instruction(spec, item, REPO, SUBSETS)


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    return _eb.score(spec, its, replies, REPO, SUBSETS)
