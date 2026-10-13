"""olmOCR-Bench: 7,010 unit tests over 1,403 PDF pages."""

import json
from collections.abc import Iterator

from vlmbench import checks
from vlmbench.eval import Eval, Sample, Verdict

SPLITS = (
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
)


def _window(output: str, first_n: int | None, last_n: int | None) -> str:
    if first_n:
        return output[:first_n]
    return output[-last_n:] if last_n else output


class OlmOCRBench(Eval):
    """AllenAI olmOCR-Bench. Math tests need rendered-glyph geometry and report unsupported."""

    name = "olmocr-bench"
    dataset = "hf://datasets/allenai/olmOCR-bench"
    aliases = ("olmocr",)
    description = "7,010 unit tests over 1,403 pages: presence, order, tables, math"
    prompt = "Convert this page to markdown. Preserve reading order and tables; omit headers and footers."
    splits = SPLITS

    def samples(self) -> Iterator[Sample]:
        grouped: dict[tuple[str, int], list[dict]] = {}
        splits: dict[tuple[str, int], str] = {}
        for split in self.splits:
            with self.hf_file(f"bench_data/{split}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (row["pdf"], int(row.get("page") or 1))
                    grouped.setdefault(key, []).append(row)
                    splits[key] = split

        for (pdf, page), tests in sorted(grouped.items()):
            yield Sample(
                id=pdf.removesuffix(".pdf").replace("/", "__") + f"_p{page}",
                media=f"bench_data/pdfs/{pdf}#{page}",
                split=splits[(pdf, page)],
                truth={"tests": tests},
            )

    def score(self, sample: Sample, output: str) -> Iterator[Verdict]:
        for test in sample.truth["tests"]:
            kind = test["type"]
            diffs = int(test.get("max_diffs") or 0)
            match kind:
                case "present" | "absent":
                    found = checks.present(
                        _window(output, test.get("first_n"), test.get("last_n")),
                        test["text"],
                        max_diffs=diffs,
                        case_sensitive=bool(test.get("case_sensitive")),
                    )
                    yield kind, found if kind == "present" else not found
                case "order":
                    yield "order", checks.in_order(output, test["before"], test["after"])
                case "table":
                    yield "tables", checks.table_cell_matches(
                        output,
                        test["cell"],
                        up=test.get("up"),
                        down=test.get("down"),
                        left=test.get("left"),
                        right=test.get("right"),
                        top_heading=test.get("top_heading"),
                        left_heading=test.get("left_heading"),
                        max_diffs=diffs,
                    )
                case "baseline":
                    yield "baseline", bool(output.strip()) and checks.repeated_ngram_fraction(output) < 0.5
                case _:
                    yield kind, None
