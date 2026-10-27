"""CC-OCR (wulipc/CC-OCR), its document parsing track only, graded by a port of its own `ParsingEvaluator`
(AdvancedLiterateMachinery/Benchmarks/CC-OCR, not on PyPI): 1 - normalized edit distance for documents and
formulas, TEDS for tables; each subset is the mean over its items, overall the mean over subsets.

CC-OCR asks each subset for its own format (documents as LaTeX, tables as HTML, formulas as LaTeX); the reply here
is one Markdown page, so each subset reads its part of it before CC-OCR's metric:
- `table`: the reply's `<table>`s, as CC-OCR extracts them, with `<th>` read as `<td>` (its gold has no `<th>`) and
  `<thead>` / `<tbody>` ignored on both sides (TEDS's own `ignore_nodes`).
- `formula`: the reply's math spans (`$`, `$$`, `\\(`, `\\[`, display environments) joined, the whole reply when
  it has none; math delimiters and display environments dropped on both sides.
- `doc`: the gold is Mathpix-style LaTeX (`\\section*`, `tabular`, `\\(`), the reply Markdown with HTML tables, so
  both are reduced to their text before the edit distance: markup dropped (headings, emphasis, HTML tags, sectioning
  commands, `tabular` rules and `&` / `\\\\` separators, math delimiters), LaTeX escapes unescaped, math kept.
`molecular` (SMILES) is left out by default: a Markdown prompt never asks for it.
"""

import csv
import html
import json
import re
import sys
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "wulipc/CC-OCR"
SUBSETS = ("doc", "table", "formula")
_SPLITS = {
    "doc": ("doc_photo_chn", "doc_photo_eng", "doc_scan_chn", "doc_scan_eng"),
    "table": ("table_photo_chn", "table_photo_eng", "table_scan_chn", "table_scan_eng"),
    "formula": ("formula_handwriting",),
    "molecular": ("molecular_handwriting",),
}

# CC-OCR's own LaTeX preamble patterns, removed from a document reply
_PREAMBLE = [
    r"\\documentclass\{.*?\}",
    r"\\usepackage\[.*?\]\{.*?\}",
    r"\\usepackage\{.*?\}",
    r"\\geometry\{.*?\}",
    r"\\begin\{document\}",
    r"\\end\{document\}",
    r"\\noindent",
]
_DISPLAY = r"(?:equation|align|eqnarray|multline|gather|flalign|alignat|displaymath)\*?"
_MATH = re.compile(r"\\begin\{(" + _DISPLAY + r")\}.*?\\end\{\1\}|\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|\$.+?\$", re.S)
_DELIMS = re.compile(r"\\(?:begin|end)\{" + _DISPLAY + r"\}(?:\{\d+\})?|\$|\\[()\[\]]")
_TAGS = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th|br|sup|sub|u|b|i|s|em|strong|del|ins|mark|span|div|p|img|caption"
    r"|figure|page_header|page_footer)\b[^<>]*/?>",
    re.I,
)
_ARG = r"\{((?:[^{}]|\{[^{}]*\})*)\}"
_FULL = "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～"
_HALF = str.maketrans(_FULL, "".join(chr(ord(c) - 0xFEE0) for c in _FULL))


def _subset(split: str) -> str:
    return next(s for s, xs in _SPLITS.items() if split in xs)


@cache
def _rows(revision: str) -> dict[str, dict]:
    """id -> row (subset, category, image path, gold), images and gold written once beside the HF cache."""
    from huggingface_hub import constants, snapshot_download

    folder = Path(constants.HF_HOME) / "vlmbench" / "ccocr" / revision[:12]
    gold = folder / "gold.json"
    if not gold.exists():
        import base64

        snap = Path(snapshot_download(REPO, repo_type="dataset", revision=revision, allow_patterns="doc_parsing/*"))
        csv.field_size_limit(sys.maxsize)
        out = {}
        for f in sorted((snap / "doc_parsing").glob("*/*.tsv")):
            with f.open(newline="") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    img = folder / r["split"] / r["image_name"]
                    img.parent.mkdir(parents=True, exist_ok=True)
                    img.write_bytes(base64.b64decode(r["image"]))
                    out[f"{r['split']}/{Path(r['image_name']).stem}"] = {
                        "split": r["split"],
                        "image": str(img),
                        "answer": r["answer"],
                    }
        tmp = gold.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False))
        tmp.replace(gold)
    rows = json.loads(gold.read_text())
    for r in rows.values():
        r["subset"] = s = _subset(r["split"])
        r["category"] = f"{s}_{r['split'].rsplit('_', 1)[1]}" if s in ("doc", "table") else s
    return rows


def _selected(spec: DatasetSpec) -> set[str]:
    """Splits named by `spec.subsets`, each a subset (`doc`, `table`, `formula`, `molecular`) or a split name."""
    want = spec.subsets or SUBSETS
    return {x for s in want for x in _SPLITS.get(s, (s,))}


def items(spec: DatasetSpec) -> list[Item]:
    keep = _selected(spec)
    return [Item(i, r["category"], r["image"]) for i, r in _rows(spec.revision).items() if r["split"] in keep]


def _ned(pred: str, gt: str) -> float:
    """CC-OCR's 1 - nltk.edit_distance / max length (rapidfuzz computes the same Levenshtein distance)."""
    from rapidfuzz.distance import Levenshtein

    n = max(len(pred), len(gt))
    return 1 - Levenshtein.distance(pred, gt) / n if n else 0.0


# --- doc --------------------------------------------------------------------------------------------------------


def _text(s: str) -> str:
    """A LaTeX or Markdown page reduced to its text, math kept (applied to the gold and the reply alike)."""
    s = html.unescape(_TAGS.sub(" ", s))
    s = re.sub(r"^\s*```.*$", "", s, flags=re.M)  # code fences
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)  # images
    s = re.sub(r"^\s*#{1,6}\s+", "", s, flags=re.M)  # headings
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.M)  # bullets
    s = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", "", s, flags=re.M)  # pipe-table rules
    s = re.sub(r"\*\*|__|~~|\|", "", s)
    s = re.sub(r"\\multi(?:column|row)\{[^}]*\}\{[^}]*\}(?:" + _ARG + ")?", r"\1", s)
    s = re.sub(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\(?:begin|end)\{abstract\}", "", s)
    s = re.sub(r"\\(?:hline|maketitle)\b|\\cline\{[^}]*\}", "", s)
    for _ in range(2):  # nested once, e.g. \section*{\textbf{..}}
        s = re.sub(
            r"\\(?:(?:sub)*section|title|author|textbf|textit|emph|underline|footnotetext|footnote)\*?" + _ARG,
            r"\1",
            s,
        )
    s = re.sub(r"\\\\", "", s)  # row / line breaks
    s = re.sub(r"\\([%$_#&{}])", r"\1", s)  # escapes
    return _DELIMS.sub("", s).replace("&", "")


def _doc(gt: str, pred: str) -> float:
    for p in _PREAMBLE:
        pred = re.sub(p, "", pred)
    pred, gt = _text(pred), _text(gt)
    return _ned(pred.replace(" ", "").replace("\n", ""), gt.replace(" ", "").replace("\n", ""))


# --- formula / molecular ----------------------------------------------------------------------------------------


def _formula(gt: str, pred: str) -> float:
    spans = [m.group(0) for m in _MATH.finditer(pred)]
    pred = "".join(spans) if spans else pred
    pred = pred.replace("\n", " ").replace("```latex", "").replace("```", "").replace("\t", " ").replace(" ", "")
    return _ned(_DELIMS.sub("", pred), _DELIMS.sub("", gt.replace(" ", "")))


def _molecular(gt: str, pred: str) -> float:
    pred = re.sub(r"```\w*", "", pred).replace("\n", "").replace(" ", "").replace("<smiles>", "")
    return _ned(pred.replace("</smiles>", ""), gt.replace(" ", ""))


# --- table: CC-OCR's TEDS (IBM, Apache 2.0) ---------------------------------------------------------------------


def _clean_tables(text: str) -> str:
    """CC-OCR's `extract_and_clean_tables`."""
    if "</table>" not in text:
        text += "</table>"
    out = []
    for t in re.findall(r"<table.*?>.*?</table>", text, re.DOTALL):
        t = re.sub(r"<table.*?>", "<table>", t)
        t = re.sub(r">\s+<", "><", t)
        t = re.sub(r">(.*?)<", lambda m: ">" + m.group(1).replace("\n", "").replace(" ", "") + "<", t, flags=re.S)
        out.append(t.replace("\n", "").strip())
    return "".join(out)


def _teds(pred: str, true: str) -> float:
    from apted import APTED, Config
    from lxml import etree
    from lxml import html as lhtml
    from rapidfuzz.distance import Levenshtein

    class Node:
        def __init__(self, tag, colspan=None, rowspan=None, content=None):
            self.tag, self.colspan, self.rowspan, self.content, self.children = tag, colspan, rowspan, content, []

    class Cfg(Config):
        def rename(self, a, b):
            if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            if a.tag == "td" and (a.content or b.content):
                return Levenshtein.distance(a.content, b.content) / max(len(a.content), len(b.content))
            return 0.0

    def tokens(n, out):
        out.append(f"<{n.tag}>")
        out += list(n.text or "")
        for c in n:
            tokens(c, out)
        if n.tag != "unk":
            out.append(f"</{n.tag}>")
        if n.tag != "td" and n.tail is not None:
            out += list(n.tail)
        return out

    def tree(n):
        if n.tag == "td":
            return Node("td", int(n.attrib.get("colspan", "1")), int(n.attrib.get("rowspan", "1")), tokens(n, [])[1:-1])
        t = Node(n.tag)
        t.children = [tree(c) for c in n]
        return t

    if not pred or not true:
        return 0.0
    parser = lhtml.HTMLParser(remove_comments=True, encoding="utf-8")
    pred, true = lhtml.fromstring(pred, parser=parser), lhtml.fromstring(true, parser=parser)
    if not (pred.xpath("body/table") and true.xpath("body/table")):
        return 0.0
    pred, true = pred.xpath("body/table")[0], true.xpath("body/table")[0]
    for t in (pred, true):
        etree.strip_tags(t, "thead", "tbody")
    n = max(len(pred.xpath(".//*")), len(true.xpath(".//*")))
    return 1.0 - APTED(tree(pred), tree(true), Cfg()).compute_edit_distance() / n


def _table(gt: str, pred: str) -> float:
    m = re.search(r"```html(.+?)```", pred, re.S)
    pred = m.group(1) if m else pred.split("```html")[1] if "```html" in pred else pred
    pred = re.sub(r"<(/?)th\b", r"<\1td", pred, flags=re.I)  # the gold has no <th>
    pred = _clean_tables(pred).translate(_HALF)
    gt = _clean_tables(gt).translate(_HALF)
    return _teds(f"<html><body>{pred}</body></html>", f"<html><body>{gt}</body></html>")


_GRADE = {"doc": _doc, "table": _table, "formula": _formula, "molecular": _molecular}


def _grade(job: tuple[str, str, str]) -> float:
    subset, gt, pred = job
    return _GRADE[subset](gt, pred) if pred else 0.0


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    rows = _rows(spec.revision)
    jobs = [(rows[it.id]["subset"], rows[it.id]["answer"], replies.get(it.id) or "") for it in its]
    # TEDS (pure-Python APTED) on a 1,000-cell table is ~17 s; spawn, not fork, beside other graders' threads
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        vals = list(pool.map(_grade, jobs))
    by_sub, by_cat, units = {}, {}, {}
    for it, v in zip(its, vals):
        r = rows[it.id]
        by_sub.setdefault(r["subset"], []).append(v)
        by_cat.setdefault(r["category"], []).append(v)
        units[it.id] = {r["subset"]: [v, 1]}
    mean = lambda xs: 100 * sum(xs) / len(xs)  # noqa: E731
    metrics = {k: mean(v) for k, v in by_sub.items()}
    return {
        "score": sum(metrics.values()) / len(metrics) if metrics else 0.0,
        "metrics": metrics,
        "by_category": {k: mean(v) for k, v in by_cat.items()},
        "units": units,
    }
