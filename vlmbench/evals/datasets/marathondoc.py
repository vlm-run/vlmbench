"""MarathonDocBench (baidu/MarathonDocBench, from the Unlimited OCR Works 2 report): 100 documents of 103-298 pages
(18,272 page images, mixed Chinese / English, cross-page text and tables throughout), graded by the MPDocBench-Parse
evaluator it ships for (a drop-in ground truth for Qwen-Doc's `MPDocBench/` toolkit), with the three files Baidu
replaces (`utils/mpdocbench_patch/`): ported once, in `mpdocbench`, the patch's metric changes switched on here:
- T-TextEdit covers every cross-page text chain whose first block is a text block, title, reference or code (the
  toolkit: text blocks only);
- malformed `rowspan` / `colspan` values are read as their digits before TEDS;
- inline LaTeX nested deeper than ten braces is left as text, not converted to unicode.
The leaderboard settings are kept where they change scores: the matcher's timeout is 1,800 s
(`MPDOC_MATCH_TIMEOUT`), a table past 300 s of TEDS scores 0 (`MPDOC_TEDS_TIMEOUT`). Metrics, Overall and the
deviations (the CDM stand-in) are `mpdocbench`'s.

Every document is in one data source (`multipage_3lan_refined`, language `mixed`), so the category is the script
that dominates its annotated text: `zh` (CJK above 70% of letters), `en` (below 30%), else `mixed`. A sampled
document's pages are fetched alone (`images/<doc>/`), never the 12 GB snapshot.
"""

import json
import re
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.datasets.mpdocbench import grade

REPO = "baidu/MarathonDocBench"
DOCUMENT = "text"


def _download(revision: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision, allow_patterns=patterns))


@cache
def _docs(revision: str) -> dict[str, dict]:
    root = _download(revision, ["MarathonDocBench.json"])
    return {
        d["page_info"]["image_path"].removesuffix(".pdf"): d
        for d in json.loads((root / "MarathonDocBench.json").read_text())
    }


def _script(doc: dict) -> str:
    text = "".join(x.get("text") or "" for x in doc["layout_dets"])
    cjk, latin = len(re.findall(r"[\u4e00-\u9fff]", text)), len(re.findall(r"[A-Za-z]", text))
    share = cjk / (cjk + latin) if cjk + latin else 0.0
    return "zh" if share > 0.7 else "en" if share < 0.3 else "mixed"


def items(spec: DatasetSpec) -> list[Item]:
    return [
        Item(i, _script(d), f"hf://datasets/{REPO}@{spec.revision[:12]}/images/{i}/", doc=True)
        for i, d in _docs(spec.revision).items()
    ]


def instruction(spec: DatasetSpec, item: Item) -> str:
    return ""


def pages(spec: DatasetSpec, item: Item) -> list[Path]:
    """The document's page images in page order, downloaded (to the HF cache) on first use."""
    root = _download(spec.revision, [f"images/{item.id}/*"])
    return [root / p for p in _docs(spec.revision)[item.id]["page_info"]["images_list"]]


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    return grade(_docs(spec.revision), its, replies, patched=True, timeout_s=1800)
