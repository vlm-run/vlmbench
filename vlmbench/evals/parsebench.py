"""ParseBench: 141k rule-based checks over 506 single-page PDFs.

Rules come from the repo's JSONL manifests; pages are rendered on demand.
"""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from vlmbench import checks
from vlmbench.eval import Eval, Sample, Verdict, render_pdf_page

MANIFESTS = ("text_content.jsonl", "table.jsonl", "layout.jsonl")
UNSUPPORTED_MANIFESTS = ("chart.jsonl", "text_formatting.jsonl")

_RECALL_RULES = {"missing_word_percent", "missing_sentence_percent", "bag_of_digit_percent"}
_PRECISION_RULES = {
    "unexpected_word_percent",
    "too_many_word_occurence_percent",
    "unexpected_sentence_percent",
    "too_many_sentence_occurence_percent",
}
"""Precision rules pass vacuously on an empty page, so they get their own column
rather than propping up `content`."""
_SENTENCE_SPLIT = ". "


def _bag_recall(bag: dict[str, int], produced: Counter[str]) -> float:
    total = sum(bag.values())
    if not total:
        return 1.0
    found = sum(min(count, produced[token.lower()]) for token, count in bag.items())
    return found / total


def _lowered(bag: dict[str, int]) -> dict[str, int]:
    folded: Counter[str] = Counter()
    for token, count in bag.items():
        folded[token.lower()] += count
    return dict(folded)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in checks.normalize(text).split(_SENTENCE_SPLIT)]
    return [part for part in parts if len(part) > 8]


class ParseBench(Eval):
    """LlamaIndex ParseBench — content faithfulness, tables and layout."""

    name = "parsebench"
    dataset = "hf://datasets/llamaindex/ParseBench"
    aliases = ("parse-bench",)
    description = "1,509 pages — 141k text-content rules, 503 tables, 500 layouts"
    prompt = "Convert this page to markdown. Preserve reading order, tables and all text exactly."
    splits = ("text_content", "table", "layout")
    manifests: tuple[str, ...] = MANIFESTS

    def samples(self) -> Iterator[Sample]:
        grouped: dict[tuple[str, int | None], list[dict]] = {}
        categories: dict[tuple[str, int | None], str] = {}
        tags: dict[tuple[str, int | None], set[str]] = {}
        for manifest in self.manifests:
            with self.hf_file(manifest).open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    pdf = row.get("pdf")
                    if not pdf:
                        continue
                    key = (pdf, row.get("page"))
                    grouped.setdefault(key, []).append(row)
                    categories[key] = str(row.get("category") or "unknown")
                    tags.setdefault(key, set()).update(row.get("tags") or ())

        for (pdf, page), rules in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
            slug = Path(pdf).with_suffix("").as_posix().removeprefix("docs/")
            yield Sample(
                id=f"{slug}_p{page}" if page else slug,
                media=f"{pdf}#{page or 1}",
                split=categories[(pdf, page)],
                truth={"rules": rules},
                tags=tuple(sorted(tags[(pdf, page)])),
            )

    def resolve_media(self, sample: Sample) -> str:
        """Download the source PDF and render the referenced page, cached on disk."""
        path, _, page = sample.media.rpartition("#")
        cached = self.cache_dir / f"{sample.id.replace('/', '__')}.jpg.b64"
        if cached.exists():
            return cached.read_text()
        uri = render_pdf_page(self.hf_file(path), int(page))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(uri)
        return uri

    def score(self, sample: Sample, output: str) -> Iterator[Verdict]:
        rules = sample.truth["rules"]
        produced_words = Counter(checks.words(output))
        produced_total = sum(produced_words.values())

        for rule in rules:
            kind = rule["type"]
            payload = json.loads(rule.get("rule") or "{}")

            if kind in _RECALL_RULES:
                yield "content", self._content_score(kind, payload, output, produced_words, produced_total)
            elif kind in _PRECISION_RULES:
                yield "noise", self._content_score(kind, payload, output, produced_words, produced_total)
            elif kind == "missing_specific_word":
                yield "text", checks.present(output, payload["word"])
            elif kind == "missing_specific_sentence":
                yield "text", checks.present(output, payload["sentence"], max_diffs=2)
            elif kind in ("is_header", "is_footer"):
                yield "text", checks.present(output, payload["text"], max_diffs=4)
            elif kind == "order":
                yield "order", checks.in_order(output, payload["before"], payload["after"])
            elif kind == "expected_markdown":
                records = checks.html_table_records(rule.get("expected_markdown") or "")
                yield "tables", checks.table_cell_recall(records, output)
            elif kind == "layout":
                text = (payload.get("content") or {}).get("text") or ""
                yield "layout", checks.present(output, text, max_diffs=2) if len(text) > 3 else None
            else:
                yield kind, None

        yield from self._layout_order(rules, output)

    def _content_score(
        self,
        kind: str,
        payload: dict,
        output: str,
        produced_words: Counter[str],
        produced_total: int,
    ) -> float:
        if kind == "bag_of_digit_percent":
            expected = {str(digit): count for digit, count in (payload.get("bag_of_digit") or {}).items()}
            actual = Counter(char for char in output if char.isdigit())
            total = sum(expected.values())
            if not total:
                return 1.0
            return sum(min(count, actual[digit]) for digit, count in expected.items()) / total

        if "word" in kind:
            bag = _lowered(payload.get("bag_of_word") or {})
            if kind == "missing_word_percent":
                return _bag_recall(bag, produced_words)
            if not produced_total:
                return 0.0
            if kind == "unexpected_word_percent":
                unexpected = sum(count for token, count in produced_words.items() if token not in bag)
                return 1 - unexpected / produced_total
            excess = sum(max(0, produced_words[token] - count) for token, count in bag.items())
            return 1 - excess / produced_total

        bag = _lowered(payload.get("bag_of_sentence") or {})
        if kind == "missing_sentence_percent":
            if not bag:
                return 1.0
            found = sum(1 for sentence in bag if checks.present(output, sentence, max_diffs=2))
            return found / len(bag)

        produced_sentences = _sentences(output)
        if not produced_sentences:
            return 0.0
        matched = sum(
            1
            for sentence in produced_sentences
            if any(checks.present(sentence, expected, max_diffs=2) for expected in bag)
        )
        if kind == "unexpected_sentence_percent":
            return matched / len(produced_sentences)
        duplicates = len(produced_sentences) - len(set(produced_sentences))
        return 1 - duplicates / len(produced_sentences)

    def _layout_order(self, rules: list[dict], output: str) -> Iterator[Verdict]:
        """Check consecutive reading-order pairs from the layout manifest."""
        ordered: list[tuple[int, str]] = []
        for rule in rules:
            if rule["type"] != "layout":
                continue
            payload = json.loads(rule.get("rule") or "{}")
            text = (payload.get("content") or {}).get("text") or ""
            index = payload.get("ro_index")
            if text and len(text) > 8 and isinstance(index, int):
                ordered.append((index, text))
        ordered.sort()
        for (_, before), (_, after) in zip(ordered, ordered[1:]):
            yield "order", checks.in_order(output, before, after)
