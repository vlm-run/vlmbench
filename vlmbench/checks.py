"""Deterministic checks shared by evals: fuzzy presence, reading order, table records.

Plain functions — import what you need; nothing here registers in the catalog.
"""

import re
from collections import Counter
from html.parser import HTMLParser

from rapidfuzz import fuzz

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏ']+")
_MARKDOWN_NOISE = re.compile(r"[#*_`~|]+")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_PUNCTUATION_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ",
    "\u2026": "...",
})
"""Typographic punctuation varies between ground truth and model output; fold it."""
_WHITESPACE = re.compile(r"\s+")
_MIN_SIMILARITY = 60.0
"""Floor on fuzzy similarity, so a short needle can never match everything."""


def normalize(text: str, *, case_sensitive: bool = False) -> str:
    """Strip HTML tags and markdown syntax, collapse whitespace: compare content, not notation.

    A model emitting `<table>` and one emitting a markdown pipe table must score alike.
    """
    folded = text.translate(_PUNCTUATION_FOLD)
    cleaned = _WHITESPACE.sub(" ", _MARKDOWN_NOISE.sub(" ", _HTML_TAG.sub(" ", folded))).strip()
    return cleaned if case_sensitive else cleaned.lower()


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def present(output: str, needle: str, *, max_diffs: int = 0, case_sensitive: bool = False) -> bool:
    """True when `needle` appears in `output`, allowing up to `max_diffs` character edits.

    Fuzzy matching uses rapidfuzz `partial_ratio`, so `max_diffs` is a similarity
    threshold rather than an exact edit-distance bound.
    """
    haystack = normalize(output, case_sensitive=case_sensitive)
    target = normalize(needle, case_sensitive=case_sensitive)
    if not target:
        return True
    if not haystack:
        return False
    if max_diffs <= 0:
        return target in haystack
    allowed = min(max_diffs, max(0, len(target) - 1))
    threshold = max(_MIN_SIMILARITY, (1 - allowed / len(target)) * 100)
    return fuzz.partial_ratio(target, haystack) >= threshold


def absent(output: str, needle: str, *, max_diffs: int = 0, case_sensitive: bool = False) -> bool:
    return not present(output, needle, max_diffs=max_diffs, case_sensitive=case_sensitive)


def in_order(output: str, first: str, second: str, *, case_sensitive: bool = False) -> bool:
    """True when `first` appears before `second` in the output's reading order."""
    haystack = normalize(output, case_sensitive=case_sensitive)
    left = haystack.find(normalize(first, case_sensitive=case_sensitive))
    right = haystack.rfind(normalize(second, case_sensitive=case_sensitive))
    return left != -1 and right != -1 and left < right


def _bag_overlap(expected: list[str], actual: list[str]) -> int:
    return sum((Counter(expected) & Counter(actual)).values())


def word_recall(expected: str, output: str) -> float:
    """Fraction of the expected words that survived into the output (1.0 = none missing)."""
    wanted = words(expected)
    if not wanted:
        return 1.0
    return _bag_overlap(wanted, words(output)) / len(wanted)


def word_precision(expected: str, output: str) -> float:
    """Fraction of the output's words that were expected (1.0 = nothing hallucinated)."""
    produced = words(output)
    if not produced:
        return 0.0
    return _bag_overlap(produced, words(expected)) / len(produced)


def digit_recall(expected: str, output: str) -> float:
    """Word recall restricted to numeric tokens — the costliest OCR errors."""
    wanted = [token for token in words(expected) if any(char.isdigit() for char in token)]
    if not wanted:
        return 1.0
    produced = [token for token in words(output) if any(char.isdigit() for char in token)]
    return _bag_overlap(wanted, produced) / len(wanted)


def repeated_ngram_fraction(output: str, *, size: int = 8) -> float:
    """Share of n-grams that are duplicates — detects decoder degeneration loops."""
    tokens = words(output)
    if len(tokens) < size * 2:
        return 0.0
    grams = [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
    return 1 - len(set(grams)) / len(grams)


class _TableParser(HTMLParser):
    """Collect `<table>` rows as lists of cell strings, stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_WHITESPACE.sub(" ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_table_records(markup: str) -> list[dict[str, str]]:
    """Parse an HTML table into records keyed by its header row."""
    parser = _TableParser()
    parser.feed(markup)
    rows = [row for row in parser.rows if any(cell for cell in row)]
    if len(rows) < 2:
        return []
    header = rows[0]
    records = []
    for row in rows[1:]:
        record = {header[i]: cell for i, cell in enumerate(row) if i < len(header) and cell}
        if record:
            records.append(record)
    return records


def table_cell_recall(records: list[dict[str, str]], output: str, *, max_diffs: int = 2) -> float:
    """Fraction of ground-truth table cell values that appear in the output.

    Cell-level rather than all-or-nothing per row: one mangled cell should not
    void an otherwise correct table, and the score stays diagnostic.
    """
    values = [value for record in records for value in record.values() if value]
    if not values:
        return 1.0
    haystack = normalize(output)
    return sum(present(haystack, value, max_diffs=max_diffs) for value in values) / len(values)
