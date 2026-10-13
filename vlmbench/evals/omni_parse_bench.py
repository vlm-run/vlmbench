"""Datalab omni_parse_bench: 16,288 tests over 2,937 pages."""

import json
from collections.abc import Iterator

from vlmbench import checks
from vlmbench.eval import Eval, Sample, Verdict

_SCORABLE_OUTPUTS = {"html", "markdown", "text", ""}
"""`blocks` tests ask about a structured layout response, which markdown cannot answer."""


class OmniParseBench(Eval):
    """Rule-based tests over pre-rendered pages; gold files load lazily per sample."""

    name = "omni-parse-bench"
    dataset = "hf://datasets/datalab-to/omni_parse_bench"
    aliases = ("omniparse",)
    description = "16,288 tests over 2,937 pages across 22 suites"
    prompt = "Convert this page to markdown. Preserve reading order and tables."

    def samples(self) -> Iterator[Sample]:
        import pyarrow.parquet as pq

        table = pq.read_table(self.hf_file("manifest.parquet")).to_pylist()
        for entry in sorted(table, key=lambda row: row["page_id"]):
            yield Sample(
                id=entry["page_id"],
                media=entry["doc_path"],
                split=entry["suite"],
                truth={"gt_path": entry["gt_path"]},
            )

    def score(self, sample: Sample, output: str) -> Iterator[Verdict]:
        gold = json.loads(self.hf_file(sample.truth["gt_path"]).read_text())
        for test in gold.get("tests", []):
            kind = test["test_type"]
            raw = test.get("args") or {}
            args = raw.get("content") or {}
            diffs = int(args.get("max_diffs") or 0)
            if test.get("output_type") not in _SCORABLE_OUTPUTS:
                yield kind, None
                continue
            match kind:
                case "present":
                    needle = args.get("text") or args.get("latex") or ""
                    yield "present", checks.present(output, needle, max_diffs=diffs) if needle else None
                case "order":
                    before = (raw.get("before") or {}).get("text", "")
                    after = (raw.get("after") or {}).get("text", "")
                    yield "order", checks.in_order(output, before, after)
                case "repeat":
                    yield "repeat", checks.repeated_ngram_fraction(output) < 0.5
                case "table_cell":
                    cell = args.get("text") or args.get("cell") or ""
                    if not cell:
                        yield "tables", None
                        continue
                    yield "tables", checks.table_cell_matches(
                        output,
                        cell,
                        top_heading=args.get("top_heading"),
                        left_heading=args.get("left_heading"),
                        max_diffs=diffs,
                    )
                case _:
                    yield kind, None
