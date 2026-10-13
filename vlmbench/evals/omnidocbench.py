"""OmniDocBench: 1,651 annotated pages across ten document sources."""

import json
from collections.abc import Iterator

from vlmbench import checks
from vlmbench.eval import Eval, Sample, Verdict

TEXT_CATEGORIES = {
    "text_block",
    "title",
    "list_group",
    "reference",
    "code_txt",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "figure_footnote",
    "equation_caption",
    "page_footnote",
    "equation_explanation",
}
"""Blocks a transcription must contain. Headers, footers, page numbers and masks
are excluded: models are routinely asked to drop them."""


class OmniDocBench(Eval):
    """Annotation-based: each block's text must appear, in reading order."""

    name = "omnidocbench"
    dataset = "hf://datasets/opendatalab/OmniDocBench"
    aliases = ("omni-doc-bench",)
    description = "1,651 pages — per-block text, reading order and tables"
    prompt = "Convert this page to markdown. Preserve reading order and tables; omit headers, footers and page numbers."

    def samples(self) -> Iterator[Sample]:
        pages = json.loads(self.hf_file("OmniDocBench.json").read_text())
        for page in sorted(pages, key=lambda item: item["page_info"]["image_path"]):
            info = page["page_info"]
            attributes = info.get("page_attribute") or {}
            yield Sample(
                id=info["image_path"].rsplit(".", 1)[0],
                media=f"images/{info['image_path']}",
                split=str(attributes.get("data_source") or "unknown"),
                truth={"blocks": page.get("layout_dets") or []},
                tags=(str(attributes.get("language") or ""), str(attributes.get("layout") or "")),
            )

    def score(self, sample: Sample, output: str) -> Iterator[Verdict]:
        ordered: list[tuple[int, str]] = []
        for block in sample.truth["blocks"]:
            category = block.get("category_type")
            if category == "table":
                records = checks.html_table_records(block.get("html") or "")
                yield "tables", checks.table_cell_recall(records, output) if records else None
                continue
            if category in ("equation_isolated", "equation_semantic"):
                yield "equations", None
                continue
            if category not in TEXT_CATEGORIES:
                continue
            text = (block.get("text") or "").strip()
            if len(text) < 4:
                continue
            yield "text", checks.present(output, text, max_diffs=max(2, len(text) // 20))
            if isinstance(block.get("order"), int):
                ordered.append((block["order"], text))

        ordered.sort()
        for (_, before), (_, after) in zip(ordered, ordered[1:]):
            yield "order", checks.in_order(output, before, after)
