"""Deterministic checks shared by evals: fuzzy presence, reading order, table records.

Plain functions — import what you need; nothing here registers in the catalog.
"""

import re
from html.parser import HTMLParser

from rapidfuzz import fuzz

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏ']+")
_MARKDOWN_NOISE = re.compile(r"[#*_`~|]+")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_PUNCTUATION_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u2026": "...",
    }
)
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
    """True when `needle` appears in `output`, within `max_diffs` edits.

    `max_diffs` becomes a similarity threshold, not an exact edit-distance bound.
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


def in_order(output: str, first: str, second: str, *, case_sensitive: bool = False) -> bool:
    """True when `first` appears before `second`. An empty needle never passes."""
    haystack = normalize(output, case_sensitive=case_sensitive)
    before = normalize(first, case_sensitive=case_sensitive)
    after = normalize(second, case_sensitive=case_sensitive)
    if not before or not after or not haystack:
        return False
    left = haystack.find(before)
    right = haystack.rfind(after)
    return left != -1 and right != -1 and left < right


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


def _markdown_grids(text: str) -> list[list[list[str]]]:
    """Parse pipe tables into grids, dropping separator rows."""
    grids, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not all(set(cell) <= set("-: ") for cell in cells):
                current.append(cells)
            continue
        if current:
            grids.append(current)
            current = []
    if current:
        grids.append(current)
    return grids


def table_grids(text: str) -> list[list[list[str]]]:
    """Every table in the text as a row-major grid, from HTML or markdown alike."""
    parser = _TableParser()
    parser.feed(text)
    grids = [parser.rows] if parser.rows else []
    return grids + _markdown_grids(text)


def table_cell_matches(
    text: str,
    cell: str,
    *,
    up: str | None = None,
    down: str | None = None,
    left: str | None = None,
    right: str | None = None,
    top_heading: str | None = None,
    left_heading: str | None = None,
    max_diffs: int = 0,
) -> bool:
    """True when `cell` sits in some table with the named neighbours around it."""

    def same(value: str, expected: str | None) -> bool:
        return expected is None or present(value, expected, max_diffs=max_diffs)

    for grid in table_grids(text):
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                if not present(value, cell, max_diffs=max_diffs):
                    continue
                above = grid[row_index - 1][col_index] if row_index > 0 and col_index < len(grid[row_index - 1]) else ""
                below = (
                    grid[row_index + 1][col_index]
                    if row_index + 1 < len(grid) and col_index < len(grid[row_index + 1])
                    else ""
                )
                if not (
                    same(above, up)
                    and same(below, down)
                    and same(row[col_index - 1] if col_index > 0 else "", left)
                    and same(row[col_index + 1] if col_index + 1 < len(row) else "", right)
                    and same(grid[0][col_index] if col_index < len(grid[0]) else "", top_heading)
                    and same(row[0], left_heading)
                ):
                    continue
                return True
    return False


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
    """Fraction of ground-truth table cell values present in the output.

    Cell-level, so one mangled cell does not void an otherwise correct table.
    """
    values = [value for record in records for value in record.values() if value]
    if not values:
        return 1.0
    haystack = normalize(output)
    return sum(present(haystack, value, max_diffs=max_diffs) for value in values) / len(values)
