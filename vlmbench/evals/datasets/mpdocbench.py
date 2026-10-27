"""MPDocBench-Parse (Tongyi-Zhiwen/Qwen-Doc, `MPDocBench/`; arXiv 2605.22100): 420 real-world multi-page PDFs
(2-31 pages, 3,135 page images) from 15 domains, Chinese and English, graded by a port of its own end-to-end
evaluator (`MPDocBench/` @ the pinned Qwen-Doc commit: `dataset/end2end_dataset.py`, `utils/match_quick.py`,
`utils/extract.py`, `utils/data_preprocess.py`, `metrics/cal_metric.py`, `metrics/head_metric.py`). The evaluator is
a script tree, not a package (it imports `func_timeout`, `loguru`, `evaluate`, `mmeval`, `skimage`, matplotlib), so
its definitions are reproduced here; `marathondoc` grades with the same port (Baidu ships MarathonDocBench as a
drop-in ground truth for this toolkit).

Ground truth (`MPDocBench.json`) comes from the pinned Qwen-Doc commit (`spec.revision`); page images from the
official ModelScope archive (`zhoubb/MPDocBench`, `MPDocBench_data.zip`, pinned below), read by HTTP range so only a
sampled document's pages are fetched. The 11 `pptSlides_eng` documents the toolkit pulls from SlideVQA (a separate,
login-gated dataset) are not in the archive and are left out: 409 documents.

A document's reply (its page replies joined in page order) is graded as the evaluator grades one `<doc>.md`:
- the reply is split by `md_tex_filter` (HTML, LaTeX and pipe tables, `$$` / `\\[` / `$` / `\\(` math, code fences,
  paragraphs on blank lines) and matched to the document's annotated blocks by `match_gt2pred_quick`, cross-page
  `truncated` chains joined into one block first;
- TextEdit: normalized edit distance over the matched text blocks (letters, digits and CJK only, inline math as
  unicode), per document sum of distances / sum of lengths, the mean over documents (`ALL_page_avg`);
- T-TextEdit: the same over the cross-page merged text blocks only, each matched one-to-one;
- TableTEDS / T-TableTEDS: TEDS per annotated table (and per cross-page merged table), the mean over tables;
- FormulaCDM: per display formula, the mean over formulas (stand-in, below);
- ReadOrderEdit: edit distance between the matched text blocks' annotated order and the reply's, per document;
- HeadingTEDS: tree edit similarity between the reply's Markdown heading tree (`#` lines) and the annotated title
  hierarchy (`parent_son` relations), per document;
- FigureF1: figures as `![](pageN_x1_y1_x2_y2.jpg)` tags (N the 1-based page, coordinates on a 1000x1000 grid)
  matched to the annotated figure boxes on the same page at IoU >= 0.5, F1 per document with figures.
Score, the leaderboard's Overall: the mean of the eight, edit distances as 100 * (1 - edit), each the mean over the
documents (or tables, formulas) the sample has; a component the sample lacks is left out of the mean. Metrics are each
component, 0-100, higher better.

Deviations from the evaluator:
- FormulaCDM needs xelatex, ImageMagick and Node; it stands in as 1 - normalized edit distance between the
  `normalized_formula` forms (an empty prediction scores 0, as CDM does).
- Its matcher runs under a timeout (30 s here, as in the toolkit; `marathondoc` uses its leaderboard's 1,800 s) and
  falls back to its one-to-one `match_gt2pred_simple` past it; the same fallback is kept, but the port computes the
  edit-distance matrices in C (rapidfuzz, the same values), so it reaches the fallback far less often.
- A table whose TEDS runs past 300 s scores 0 (the MarathonDocBench leaderboard's `MPDOC_TEDS_TIMEOUT`); identical
  normalized tables score 1 without the tree edit (the same value).
- LaTeX tables score 0 as in the evaluator (it blanks their predictions); `latexmlc` is not needed for that.
- BLEU / METEOR and the relation F1s are off in the toolkit's `end2end.yaml` and not computed.
The evaluator's own quirks are kept (paragraph positions that skip the blank-line separators, the reading order read
from misaligned indices when a block normalizes to nothing, the entry skipped after a removed inline-formula match).
"""

import io
import json
import os
import re
import signal
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "Tongyi-Zhiwen/Qwen-Doc"
DOCUMENT = "text"
IMAGES = "zhoubb/MPDocBench"
IMAGES_REVISION = "48428421ee985717fbda3a6521165aaedfc42077"
IMAGES_SHA256 = "f9a4c05d586b11a749e60ce13565090fdfbf8f4387fd35e5511a4a1a3d682d22"
"""`MPDocBench_data.zip` at that ModelScope revision; its LFS object is addressed by this hash."""

TEXT = ["text_block", "title", "code_txt", "code_txt_caption", "reference", "equation_caption", "figure_caption",
        "figure_footnote", "table_caption", "table_footnote", "code_algorithm", "code_algorithm_caption", "header",
        "footer", "page_footnote", "page_number"]  # fmt: skip
IGNORES = ["figure_caption", "figure_footnote", "table_caption", "table_footnote", "code_algorithm",
           "code_algorithm_caption", "header", "footer", "page_footnote", "page_number", "equation_caption"]  # fmt: skip
GROUPS = ("text", "truncated_text", "table_teds", "truncated_table_teds", "formula", "reading_order", "figure_f1",
          "heading_teds")  # fmt: skip
IGNORE_THRES = 0.1


# --- data -----------------------------------------------------------------------------------------------------------


def _home(*parts: str) -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HOME).joinpath("vlmbench", *parts)


def _fetch(url: str, out: Path) -> Path:
    import urllib.request

    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f"{out.suffix}.{os.getpid()}.tmp")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(out)
    return out


@cache
def _docs(revision: str) -> dict[str, dict]:
    url = f"https://raw.githubusercontent.com/{REPO}/{revision}/MPDocBench/MPDocBench.json"
    path = _fetch(url, _home("mpdocbench", revision[:12], "MPDocBench.json"))
    return {d["page_info"]["image_path"].removesuffix(".pdf"): d for d in json.loads(path.read_text())}


class _Remote(io.RawIOBase):
    """A remote file read by HTTP range requests (seekable, for `zipfile`); the CDN's signed redirect is resolved
    again when it expires."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size, self.location = self._resolve()

    def _resolve(self) -> tuple[int, str]:
        import urllib.request

        with urllib.request.urlopen(urllib.request.Request(self.url, method="HEAD")) as r:
            return int(r.headers["Content-Length"]), r.url

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        import urllib.error
        import urllib.request

        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(3):
            req = urllib.request.Request(self.location, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
            try:
                with urllib.request.urlopen(req) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError:
                if attempt == 2:
                    raise
                self.location = self._resolve()[1]
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


@cache
def _zip():
    import zipfile

    url = f"https://www.modelscope.cn/datasets/{IMAGES}/resolve/{IMAGES_REVISION}/MPDocBench_data.zip"
    remote = _Remote(url)
    if f"/{IMAGES_SHA256[:2]}/{IMAGES_SHA256[2:4]}/{IMAGES_SHA256[4:]}" not in remote.location:
        raise RuntimeError(f"{IMAGES}@{IMAGES_REVISION}: the archive is not the pinned {IMAGES_SHA256[:12]}")
    return zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))


def _members() -> set[str]:
    """The archive's file names, listed once (its central directory) and kept beside the cache."""
    out = _home("mpdocbench", "images", IMAGES_REVISION[:12], "members.txt")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text("\n".join(_zip().namelist()))
        tmp.replace(out)
    return set(out.read_text().split("\n"))


def items(spec: DatasetSpec) -> list[Item]:
    """One item per document in the archive; the category is its domain (`data_source`)."""
    have = {m.split("/")[0] for m in _members()}
    src = f"modelscope://{IMAGES}@{IMAGES_REVISION[:12]}/MPDocBench_data.zip"
    return [
        Item(i, d["page_info"]["page_attribute"]["data_source"], f"{src}#{i}/", doc=True)
        for i, d in _docs(spec.revision).items()
        if i in have
    ]


def instruction(spec: DatasetSpec, item: Item) -> str:
    return ""


def pages(spec: DatasetSpec, item: Item) -> list[Path]:
    """The document's page images in page order, each extracted from the archive once."""
    folder = _home("mpdocbench", "images", IMAGES_REVISION[:12])
    out = []
    for rel in _docs(spec.revision)[item.id]["page_info"]["images_list"]:
        name = rel.removeprefix("images/")
        path = folder / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
            tmp.write_bytes(_zip().read(name))
            tmp.replace(path)
        out.append(path)
    return out


# --- normalization (utils/data_preprocess.py) -----------------------------------------------------------------------

_INLINE = re.compile(r"\$(.*?)\$|\\\((.*?)\\\)")
_FORMULA_FILTER = ["\\mathbf", "\\mathrm", "\\mathnormal", "\\mathit", "\\mathbb", "\\mathcal", "\\mathscr",
                   "\\mathfrak", "\\mathsf", "\\mathtt", "\\textbf", "\\text", "\\boldmath", "\\boldsymbol",
                   "\\operatorname", "\\bm", "\\symbfit", "\\mathbfcal", "\\symbf", "\\scriptscriptstyle", "\\notag",
                   "\\setlength", "\\coloneqq", "\\space", "\\thickspace", "\\thinspace", "\\medspace",
                   "\\nobreakspace", "\\negmedspace", "\\quad", "\\qquad", "\\enspace", "\\substackw", " ", "$$",
                   "\\left", "\\right", "\\displaystyle", "\\text"]  # fmt: skip


def _too_deep(s: str, limit: int = 10) -> bool:
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
            if depth > limit:
                return True
        elif ch == "}":
            depth -= 1
    return False


def _latex2text(s: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text

    return LatexNodes2Text().latex_to_text(s)


def textblock2unicode(text: str, patched: bool = False) -> str:
    """Inline math holding `\\`, `^` or `_` as unicode text (the MarathonDocBench patch skips braces nested > 10)."""
    found = []
    for m in _INLINE.finditer(text):
        body = re.sub(r"\\([\\_&%^])", "", m[1] if m[1] is not None else m[2])
        try:
            if any(c in body for c in "\\^_") and not (patched and _too_deep(body)):
                if body.endswith("\\"):
                    body += " "
                found.append((m.start(), m.end(), _latex2text(body)))
        except Exception:
            continue
    for a, b, u in sorted(found, reverse=True):
        text = text[:a] + u.strip() + text[b:]
    return text


def clean_string(s: str) -> str:
    for t in ("\\t", "\\n", "\t", "\n", "/t", "/n", "_"):
        s = s.replace(t, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s)


def normalized_formula(text: str) -> str:
    text = text.strip().strip("$").strip("\n")
    if m := re.search(r"\\\[(.+?)(?<!\\)\\\]", text):
        text = m[1].strip()
    for p in (r"\\tag\{.*?\}", r"\\hspace\{.*?\}", r"\\begin\{.*?\}", r"\\end\{.*?\}", r"\\arraycolsep.*?\}"):
        text = re.sub(p, "", text)
    text = text.strip(".")
    for f in _FORMULA_FILTER:
        text = text.replace(f, "")
    return text.lower()


def normalized_html_table(text: str) -> str:
    import html
    import unicodedata

    from bs4 import BeautifulSoup

    if "<table" not in text.replace(" ", "").replace("'", '"'):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for th in soup.find_all("th"):
        th.name = "td"
    for t in soup.find_all("thead"):
        t.unwrap()
    for m in soup.find_all("math"):
        m.replace_with(f"${m.get('alttext', '')}$")
    for s in soup.find_all("span"):
        s.unwrap()
    t = unicodedata.normalize("NFKC", html.unescape(str(soup)).replace("\n", "")).strip()
    t = "".join(re.findall(r"<table\b[^>]*>(.*)</table>", t, re.DOTALL | re.IGNORECASE))
    for a in ("style", "height", "width", "align", "class"):
        t = re.sub(f'( {a}=".*?")', "", t)
    t = re.sub(r"\s+", " ", re.sub("</?tbody>", "", t))
    t = '<html><body><table border="1" >' + t + "</table></body></html>"
    for tag in ("sup", "sub", "span", "div", "p"):
        t = t.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
    t = t.replace('<spandata-span-identity="">', "")
    return re.sub("<colgroup>.*?</colgroup>", "", t)


# --- a reply split into blocks (utils/extract.py, utils/table_utils.py) ---------------------------------------------

_IMG = re.compile(r"!\[.*?\]\(.*?\)")
_FIG = re.compile(r"page(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.\w+")
_DISPLAY = re.compile(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\$(.*?)\$|\\\((.*?)\\\)", re.DOTALL)
_DOLLAR = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)", re.DOTALL)
_MD_TABLE = re.compile(r"\|\s*.*?\s*\|\n", re.DOTALL)
_HTML_TABLE = re.compile(r"(<table.*?</table>)", re.DOTALL)
_CODE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)


def figure_boxes(content: str) -> list[dict]:
    """`extract_figure_boxes`: every `![](pageN_x1_y1_x2_y2.ext)`."""
    out = []
    for m in _IMG.finditer(content):
        if f := _FIG.search(m[0]):
            out.append({"page_id": int(f[1]), "bbox_norm": [int(f[i]) for i in range(2, 6)]})
    return out


_TABULAR_BEGIN = re.compile(r"\\begin{tabular}")
_HTML_BEGIN = re.compile(r"<table(?:[^>]*)>")


def _tex_tables(content: str) -> tuple[list[str], list[tuple[int, int]]]:
    tables, positions = [], []
    for m in re.finditer(r"\\begin{table}(.*?)\\end{table}", content, re.DOTALL):
        tables.append(m[0])
        positions.append((m.start(), m.end()))
        content = content[: m.start()] + " " * (m.end() - m.start()) + content[m.end() :]
    t2, p2 = _nested_tex(content)
    pairs = sorted(zip(positions + p2, tables + t2), key=lambda x: x[0][0])
    return [t for _, t in pairs], [p for p, _ in pairs]


def _nested_tex(text: str):
    # `extract_tabular` re-enters past `len(begin_pattern)`, the raw pattern string `\\begin{tabular}` (15 chars)
    return _nested_raw(text, _TABULAR_BEGIN, r"\end{tabular}", len(r"\\begin{tabular}"))


def _nested_html(text: str):
    return _nested_raw(text, _HTML_BEGIN, "</table>", len(r"<table(?:[^>]*)>"))


def _nested_raw(text: str, begin: re.Pattern, end: str, reenter: int):
    end_re = re.compile(re.escape(end))
    end_len = len(end) if end == "</table>" else len(r"\\end{tabular}")
    out, pos, cur, stack = [], [], 0, []
    while cur < len(text):
        b, e = begin.search(text, cur), end_re.search(text, cur)
        if not b and not e:
            break
        if b and (not e or b.start() < e.start()):
            stack.append(b.start())
            cur = b.start() + end_len
        elif e:
            if stack:
                start = stack.pop()
                if not stack:
                    out.append(text[start : e.start() + end_len])
                    pos.append((start, e.start() + end_len))
            cur = e.start() + end_len
        else:
            cur += 1
    if stack:
        new = stack[0] + reenter
        sub_out, sub_pos = _nested_raw(text[new:], begin, end, reenter)
        out += sub_out
        pos += [(a + new, b + new) for a, b in sub_pos]
    return out, pos


def _md_to_html(table: str) -> str:
    rows = [r.strip() for r in table.strip().split("\n")]
    out = "<table>\n  <thead>\n    <tr>\n"
    for c in [c.strip() for c in rows[0].split("|")[1:-1]]:
        out += f"      <th>{c}</th>\n"
    out += "    </tr>\n  </thead>\n  <tbody>\n"
    for row in rows[2:]:
        out += "    <tr>\n"
        for c in [c.strip() for c in row.split("|")[1:-1]]:
            out += f"      <td>{c}</td>\n"
        out += "    </tr>\n"
    return out + "  </tbody>\n</table>\n"


def _md_table_mode(line: str) -> bool:
    return bool(re.search(r"-*?:", line) or re.search(r"---", line) or re.search(r":-*?", line))


def _merge_md_tables(s: str) -> str:
    s = re.sub(r"<!--[\s\S]*?-->", "", s)
    lines = []
    for block in re.findall(r"<table>[\s\S]*?</table>", s):
        bl = [x.replace("<th>", "<td>").replace("</th>", "</td>") if "<th>" in x else x for x in block.split("\n")]
        kept = [x for x in bl if not re.search(r"</?t(able|head|body)>", x)]
        if len(kept) > 2:
            lines += kept
    return "\n\n" + "<table>\n{}\n</table>".format("\n".join(lines)) + "\n\n"


def _placeholder(s: str) -> str:
    """`replace_table_with_placeholder`: consecutive converted pipe tables merged into one."""
    out, in_block, block, last = [], False, "", ""
    for line in s.split("\n"):
        if "<table>" in line:
            in_block = True
            block += last
        elif in_block:
            if not _md_table_mode(last) and "</thead>" not in last:
                block += "\n" + last
            if "</table>" in last and "<table>" not in line:
                in_block = False
                out.append(_merge_md_tables(block))
                block = ""
        else:
            out.append(last)
        last = line
    if last:
        if in_block or "</table>" in last:
            block += "\n" + last
            out.append(_merge_md_tables(block))
        else:
            out.append(last)
    return "\n".join(out)


def _convert_md_tables(content: str) -> str:
    content = content.replace("\r", "") + "\n"
    for m in _MD_TABLE.findall(content):
        content = content.replace(m, _md_to_html(m), 1)
    out = _placeholder(content)
    return out.replace("<table>", '<table border="1" >').replace("<td>", '<td colspan="1" rowspan="1">')


def _blank(s: str, a: int, b: int) -> str:
    return s[:a] + " " * (b - a) + s[b:]


def md_tex_filter(content: str) -> dict[str, list[dict]]:
    """The reply as blocks by category (`latex_table`, `html_table`, `equation_isolated`, `text_all`), each list in
    reply order, the categories in order of first appearance."""
    content = _IMG.sub("", content)
    for p in (r"^```markdown\n?", r"^```html\n?", r"^```latex\n?", r"```\n?$"):
        content = re.sub(p, "", content, flags=re.MULTILINE)
    content = re.sub(r" {4,}", "    ", re.sub(r"_{4,}", "____", content))
    for t in ("<html>", "</html>", "<body>", "</body>"):
        content = content.replace(t, "")
    out = []
    tables, positions = _tex_tables(content)
    for t, (a, _) in zip(tables, positions):
        out.append({"category_type": "latex_table", "position": [a, a + len(t)], "content": t})
        content = _blank(content, a, a + len(t))
    tables, positions = _nested_html(content)
    for t, (a, _) in zip(tables, positions):
        out.append({"category_type": "html_table", "position": [a, a + len(t)], "content": t})
        content = _blank(content, a, a + len(t))
    for m in list(_DISPLAY.finditer(content)):
        if not m[0]:
            continue
        line, pos = " ".join(m[0].strip().split("\n")), [m.start(), m.end()]
        sub = _DOLLAR.search(line)
        if sub is None:
            content = _blank(content, *pos)
            out.append({"category_type": "equation_isolated", "position": pos, "content": line})
        elif sub[1]:
            content = _blank(content, *pos)
            out.append(
                {"category_type": "equation_isolated", "position": pos, "content": _DOLLAR.sub(r"\\[\1\\]", line)}
            )
        else:  # inline math stays in its paragraph, and is matched on its own too
            out.append({"category_type": "equation_isolated", "position": pos,
                        "content": _DOLLAR.sub(r"\\[\2\3\\]", line), "fine_category_type": "equation_inline"})  # fmt: skip
    if len(_MD_TABLE.findall(content + "\n")) >= 2:
        content = _convert_md_tables(content)
        for m in _HTML_TABLE.finditer(content):
            content = _blank(content, m.start(), m.end())
            out.append({"category_type": "html_table", "position": [m.start(), m.end()], "content": m[0].strip(),
                        "fine_category_type": "md2html_table"})  # fmt: skip
    for m in _CODE.finditer(content):
        content = _blank(content, m.start(), m.end())
        out.append({"category_type": "text_all", "position": [m.start(), m.end()], "content": m[2].strip(),
                    "language": m[1], "fine_category_type": "code"})  # fmt: skip
    content = re.sub(r"\\title\{(.*?)\}", r"\1", content)
    content = re.sub(r"\\title\s*\{\s*(.*?)\s*\}", r"\1", content, flags=re.DOTALL)
    content = re.sub(r"\\text\s*\{\s*(.*?)\s*\}", r"\1", content, flags=re.DOTALL)
    content = re.sub(r"\\section\*?\{(.*?)\}", r"\1", content)
    content = re.sub(r"\\section\*?\{\s*(.*?)\s*\}", r"\1", content, flags=re.DOTALL)
    parts = content.split("\n\n")
    if len(parts) == 1:
        parts = content.split("\n")
    cursor = 0
    for text in parts:
        pos = [cursor, cursor + len(text)]  # the evaluator's cursor skips the separators
        cursor += len(text)
        text = "\n".join(x.strip() for x in text.strip().strip("\n").split("\n") if x.strip())
        if not text:
            continue
        if text.startswith("<table") and text.endswith("</table>"):
            out.append({"category_type": "html_table", "position": pos, "content": text})
        elif text.startswith("$") and text.endswith("$"):
            if text.replace("$", "").strip():
                out.append({"category_type": "equation_isolated", "position": pos, "content": text.strip()})
        else:
            out.append(
                {"category_type": "text_all", "position": pos, "content": text, "fine_category_type": "text_block"}
            )
    by: dict[str, list[dict]] = defaultdict(list)
    for b in sorted(out, key=lambda x: x["position"][0]):
        by[b["category_type"]].append(b)
    return by


# --- matching (utils/match.py, utils/match_quick.py) ----------------------------------------------------------------


def _np():
    import numpy as np

    return np


def _ned_matrix(a: list[str], b: list[str]):
    """`compute_edit_distance_matrix_new`: Levenshtein / max length (both empty: 0), here in C."""
    np = _np()
    if not a or not b:
        return np.zeros((len(a), len(b)))
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    return cdist(a, b, scorer=Levenshtein.normalized_distance, dtype=np.float64, workers=4)


def _lev(a, b) -> int:
    from Levenshtein import distance

    return distance(a, b)


def _ned(a: str, b: str) -> float:
    return _lev(a, b) / max(len(a), len(b))


def _window_min(long: str, short: str) -> int:
    """The least edit distance between `short` and any window of `long` of its length."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    k = len(short)
    wins = [long[i : i + k] for i in range(len(long) - k + 1)]
    return int(cdist([short], wins, scorer=Levenshtein.distance, dtype=_np().int64, workers=1).min())


def _pred_category(i, items: list[dict]) -> str:
    return items[i].get("fine_category_type") or items[i]["category_type"]


def _gt_pred_lines(gt_mix: list[dict], pred_mix: list[dict], line_type):
    """`get_gt_pred_lines`; the item lists come back unfiltered, as the evaluator returns them."""
    gt_lines, norm_gt, cats, pred_lines, norm_pred = [], [], [], [], []
    patched = _PATCHED[0]
    if line_type in ("html_table", "latex_table"):
        for it in gt_mix:
            cats.append(it.get("fine_category_type") or it["category_type"])
            gt_lines.append(str(it["content"]) if it.get("content") else str(it.get("html", "")))
        pred_lines = [str(it["content"]) for it in pred_mix]
        norm_gt, norm_pred = gt_lines, pred_lines
    else:
        for it in pred_mix:
            c = str(it["content"])
            pred_lines.append(c)
            if it["category_type"] == "text_all":
                norm_pred.append(clean_string(textblock2unicode(c, patched)))
            elif it["category_type"] == "equation_isolated":
                norm_pred.append(normalized_formula(c))
            else:
                norm_pred.append(c)
        for it in gt_mix:
            cat = it.get("fine_category_type") or it["category_type"]
            if it.get("content"):
                gt_lines.append(str(it["content"]))
                norm_gt.append(clean_string(textblock2unicode(str(it["content"]), patched))
                               if it["category_type"] == "text_all" else it["content"])  # fmt: skip
                cats.append(cat)
            elif it["category_type"] in TEXT:
                gt_lines.append(str(it.get("text", "")))
                norm_gt.append(clean_string(textblock2unicode(str(it.get("text", "")), patched)))
                cats.append(cat)
            elif it["category_type"] == "equation_isolated":
                gt_lines.append(str(it.get("latex", "")))
                norm_gt.append(normalized_formula(str(it.get("latex", ""))))
                cats.append(cat)
    g = [(a, b, c) for a, b, c in zip(gt_lines, norm_gt, cats) if a and b]
    p = [(a, b) for a, b in zip(pred_lines, norm_pred) if a and b]
    return ([x[0] for x in g], [x[1] for x in g], [x[2] for x in g], [x[0] for x in p], [x[1] for x in p],
            gt_mix, pred_mix)  # fmt: skip


_PATCHED = [False]  # the MarathonDocBench patch's guards, set per document job


def _gt_position(it: dict):
    return it.get("order") if it.get("order") else it.get("position", [""])[0]


def match_simple(gt_items, pred_items, line_type, img):
    """`match_gt2pred_simple`: one-to-one by least total normalized edit distance."""
    from scipy.optimize import linear_sum_assignment

    gl, ngl, cats, pl, npl, gt_items, pred_items = _gt_pred_lines(gt_items, pred_items, line_type)
    out = []
    if not ngl and not npl:
        return out, None
    if not ngl:
        idx = list(range(len(npl)))
        return [{"gt_idx": [""], "gt": "", "pred_idx": idx, "pred": "".join(pl), "gt_position": [""],
                 "pred_position": pred_items[0]["position"][0], "norm_gt": "", "norm_pred": "".join(npl),
                 "gt_category_type": "", "pred_category_type": _pred_category(0, pred_items), "edit": 1,
                 "img_id": img}], None  # fmt: skip
    if not npl:
        return [{"gt_idx": [i], "gt": gl[i], "pred_idx": [""], "pred": "", "gt_position": [_gt_position(gt_items[i])],
                 "pred_position": "", "norm_gt": ngl[i], "norm_pred": "", "gt_category_type": cats[i],
                 "pred_category_type": "", "edit": 1, "img_id": img} for i in range(len(ngl))], None  # fmt: skip
    cost = _ned_matrix(ngl, npl)
    rows, cols = linear_sum_assignment(cost)
    hit = dict(zip(rows.tolist(), cols.tolist()))
    for i in range(len(ngl)):
        j = hit.get(i, "")
        out.append({"gt_idx": [i], "gt": gl[i], "norm_gt": ngl[i], "gt_category_type": cats[i],
                    "gt_position": [_gt_position(gt_items[i])], "pred_idx": [j], "pred": pl[j] if j != "" else "",
                    "norm_pred": npl[j] if j != "" else "", "edit": cost[i][j] if j != "" else 1,
                    "pred_category_type": _pred_category(j, pred_items) if j else "",
                    "pred_position": pred_items[j]["position"][0] if j else "", "img_id": img})  # fmt: skip
    used = set(cols.tolist())
    left = [j for j in range(len(npl)) if j not in used]
    if left:
        if line_type in ("html_table", "latex_table"):
            from bs4 import BeautifulSoup

            cells = []
            for j in left:
                soup = BeautifulSoup(pred_items[j].get("content"), "html.parser")
                for t in [re.sub(r"\$\\cdot\$", "", x.string).strip() for x in soup.find_all("td") if x.string]:
                    new = deepcopy(pred_items[j])
                    new["content"], new["category_type"] = t, "text_all"
                    cells.append(new)
            return out, cells
        out.append({"gt_idx": [""], "gt": "", "pred_idx": left, "pred": "".join(pl[j] for j in left),
                    "gt_position": [""], "pred_position": pred_items[left[0]]["position"][0], "norm_gt": "",
                    "norm_pred": "".join(npl[j] for j in left), "gt_category_type": "",
                    "pred_category_type": _pred_category(left[0], pred_items), "edit": 1, "img_id": img})  # fmt: skip
    return out, None


def _sub_pred(g: str, p: str):
    """`sub_pred_fuzzy_matching`: the best window of `g` against all of `p`, or False."""
    if len(g) >= len(p) > 0:
        return _window_min(g, p) / len(p)
    return False


def _sub_gt(p: str, g: str) -> float:
    """`sub_gt_fuzzy_matching`: the best window of `p` against all of `g` (1 when `p` is shorter)."""
    if len(p) >= len(g) > 0:
        return _window_min(p, g) / len(g)
    return 1


def _judge(g: str, parts: list[str], memo: dict, js: list[int], threshold: float = 0.6):
    """`judge_pred_merge`; `memo` holds `_sub_pred(g, pred[j])` by j (`js` the parts' indices)."""
    if len(parts) == 1:
        return False, False
    cur, merged = " ".join(parts[:-1]), " ".join(parts)
    if _lev(g, merged) / max(len(g), len(merged)) > _lev(g, cur) / max(len(g), len(cur)):
        return False, False

    def sub(j, p):
        if j not in memo:
            memo[j] = _sub_pred(g, p)
        return memo[j]

    for j, p in zip(js[:-1], parts[:-1]):
        d = sub(j, p)
        if d is False or d > threshold:
            return False, False
    d = sub(js[-1], parts[-1])
    if d is False:
        return False, False
    return d < threshold, len(merged) <= len(g)


def _final_subset(subsets: list[list[int]], costs: list[float]) -> list[list[int]]:
    """`get_final_subset`: overlapping runs grouped; per group, the path of disjoint runs with the least mean cost."""
    if not subsets or not costs:
        return []
    pairs = sorted(zip(subsets, costs), key=lambda x: x[0][0])
    groups = defaultdict(list)
    gi = 0
    groups[gi].append(pairs[0])
    for item in pairs[1:]:
        if any(i in s[0] for s in groups[gi] for i in item[0]):
            groups[gi].append(item)
        else:
            gi += 1
            groups[gi].append(item)
    final = []
    for group in groups.values():
        if len(group) == 1:
            final.append(group[0][0])
            continue
        paths = defaultdict(list)
        paths[0].append(group[0])
        for sub in group[1:]:
            new_path = True
            for k, items_ in paths.items():
                is_dup = is_same = False
                for it in items_:  # the live list, as the evaluator iterates it while replacing
                    if it[0] == sub[0]:
                        is_dup = is_same = True
                        if it[1] > sub[1]:
                            items_.pop(items_.index(it))
                            items_.append(sub)
                    elif any(a == b for a in it[0] for b in sub[0]):
                        is_dup = True
                if not is_dup:
                    items_.append(sub)
                    new_path = False
                if is_same:
                    new_path = False
            if new_path:
                paths[len(paths)].append(sub)
        best, best_cost = [], float("inf")
        for path in paths.values():
            c = sum(x[1] for x in path) / len(path)
            if c < best_cost:
                best, best_cost = [x[0] for x in path], c
        final.extend(best)
    return final


def _deal_with_truncated(cost, ngl: list[str], npl: list[str]):
    """`deal_with_truncated`: a block with no near (< 0.1) paragraph takes the run of consecutive free paragraphs
    that best covers it; runs chosen by `get_final_subset` replace their paragraphs. The first merge test (does adding
    the next paragraph bring the edit distance down) is computed for every pair at once; the rest per pair."""
    np = _np()
    near = np.argwhere(cost < IGNORE_THRES)
    masked_g, masked_p = set(near[:, 0].tolist()), set(near[:, 1].tolist())
    free_g = [i for i in range(cost.shape[0]) if i not in masked_g]
    free_p = [j for j in range(cost.shape[1]) if j not in masked_p]
    subsets, costs = [], []
    if free_g and free_p:
        ext = [j for j in free_p if j + 1 < len(npl) and j + 1 not in masked_p]  # pairs that can try a merge
        pair = [npl[j] + " " + npl[j + 1] for j in ext]
        merged = _ned_matrix([ngl[i] for i in free_g], pair) if ext else None
        col = {j: k for k, j in enumerate(ext)}
        for r, gi in enumerate(free_g):
            g, memo = ngl[gi], {}
            best_d, best_run = None, None
            for j in free_p:
                step = 1
                k = col.get(j)
                if (
                    k is not None
                    and merged[r, k] <= cost[gi, j]
                    and len(npl[j]) <= len(g)
                    and len(npl[j + 1]) <= len(g)
                ):
                    parts, js = [npl[j]], [j]
                    while True:
                        if j + step in masked_p or j + step >= len(npl):
                            break
                        parts.append(npl[j + step])
                        js.append(j + step)
                        ok, more = _judge(g, parts, memo, js)
                        if not ok:
                            break
                        step += 1
                        if not more:
                            break
                d = cost[gi, j] if step == 1 else _ned(g, " ".join(npl[j : j + step]))
                if best_d is None or d < best_d:
                    best_d, best_run = d, list(range(j, j + step))
            subsets.append(best_run)
            costs.append(best_d)
    final = _final_subset(subsets, costs)
    if not final:
        return cost, npl, list(range(len(npl)))
    idx: list = list(range(len(npl)))
    for sub in final:
        at = idx.index(sub[0])
        for _ in sub:
            idx.pop(at)
        idx.insert(at, sub)
    lines = [" ".join(npl[x[0] : x[-1] + 1]) if isinstance(x, list) else npl[x] for x in idx]
    return _ned_matrix(ngl, lines), lines, idx


def _quick_core(ngl: list[str], npl: list[str], opl: list[str]):
    """`cal_final_match` .. `convert_final_matches` on the non-ignored lines: [{gt_idx, pred_idx, ...}]."""
    from scipy.optimize import linear_sum_assignment

    cost = _ned_matrix(ngl, npl)
    new_cost, _, idx = _deal_with_truncated(cost, ngl, npl)
    rows, cols = linear_sum_assignment(new_cost)
    at = {r: k for k, r in enumerate(rows.tolist())}
    matched = [idx[c] for c in cols.tolist()]
    costs = [new_cost[r][c] for r, c in zip(rows.tolist(), cols.tolist())]
    matches, unmatched_g = {}, []  # `process_matches`
    for i in range(len(ngl)):
        if i not in at:
            unmatched_g.append(i)
            continue
        p = matched[at[i]]
        rng = list(range(p[0], p[-1] + 1)) if isinstance(p, list) else [p]
        if costs[at[i]] > 0.7:
            unmatched_g.append(i)
        else:
            matches[i] = {"pred_indices": rng, "edit_distance": costs[at[i]]}
    fuzzy = {}  # `fuzzy_match_unmatched_items`
    for j, p in enumerate(npl):
        hits = [g for g in unmatched_g if len(p) >= len(ngl[g]) > 0 and _sub_gt(p, ngl[g]) < 0.4]
        if hits:
            fuzzy[j] = hits
    final = _merge_matches(matches, fuzzy)
    _recalculate(final, ngl, npl)
    return _convert_final(final, ngl, npl)


def _merge_matches(matches: dict, fuzzy: dict) -> dict:
    final, done = {}, set()
    for gi, m in matches.items():
        key = tuple(sorted(m["pred_indices"]))
        if key in final:
            if gi not in done:
                final[key]["gt_indices"].append(gi)
                done.add(gi)
        else:
            final[key] = {"gt_indices": [gi], "edit_distance": m["edit_distance"]}
            done.add(gi)
    for pj, gis in fuzzy.items():
        key = (pj,)
        if key in final:
            for gi in gis:
                if gi not in done:
                    final[key]["gt_indices"].append(gi)
                    done.add(gi)
        else:
            final[key] = {"gt_indices": [gi for gi in gis if gi not in done], "edit_distance": None}
            done.update(final[key]["gt_indices"])
    return final


def _recalculate(final: dict, ngl: list[str], npl: list[str]) -> None:
    for key, info in final.items():
        gis = sorted(set(info["gt_indices"]))
        if not gis:
            info["edit_distance"] = 1
            continue
        if len(gis) > 1:
            g, p = "".join(ngl[i] for i in gis), npl[key[0]] if isinstance(key[0], int) else ""
        else:
            g, p = ngl[gis[0]], " ".join(npl[j] for j in key if isinstance(j, int))
        info["edit_distance"] = _lev(g, p) / max(len(g), len(p)) if max(len(g), len(p)) else 1


def _convert_final(final: dict, ngl: list[str], npl: list[str]) -> list[dict]:
    from scipy.optimize import linear_sum_assignment

    out = []
    all_g, all_p = set(range(len(ngl))), set(range(len(npl)))
    for key, info in final.items():
        p = " ".join(npl[j] for j in key if isinstance(j, int))
        for gi in sorted(set(info["gt_indices"])):
            out.append(
                {"gt_idx": int(gi), "gt": ngl[gi], "pred_idx": list(key), "pred": p, "edit": info["edit_distance"]}
            )
    matched_g = set().union(*[set(i["gt_indices"]) for i in final.values()])
    left_g = all_g - matched_g
    left_p = all_p - {j for key in final for j in key if isinstance(j, int)}
    if left_p:
        if left_g:
            lg, lp = list(left_g), list(left_p)
            r, c = linear_sum_assignment(_ned_matrix([ngl[i] for i in lg], [npl[j] for j in lp]))
            for a, b in zip(r.tolist(), c.tolist()):
                out.append({"gt_idx": int(lg[a]), "gt": ngl[lg[a]], "pred_idx": [lp[b]], "pred": npl[lp[b]], "edit": 1})
        else:
            out.append({"gt_idx": "", "gt": "", "pred_idx": list(left_p), "pred": " ".join(npl[j] for j in left_p),
                        "edit": 1})  # fmt: skip
    else:
        out += [{"gt_idx": int(gi), "gt": ngl[gi], "pred_idx": "", "pred": "", "edit": 1} for gi in left_g]
    return out


def _merge_duplicates(conv: list[dict], ngl: list[str], gl: list[str]) -> list[dict]:
    """`merge_duplicates_add_unmatched`: entries sharing a prediction joined; unmatched blocks appended."""
    key = lambda e: tuple(e["pred_idx"]) if isinstance(e["pred_idx"], list) else (e["pred_idx"],)  # noqa: E731
    groups: dict[tuple, list[dict]] = {}
    for e in conv:
        if key(e) != ("",):
            groups.setdefault(key(e), []).append(e)
    out, done_g = [], set()
    for k, es in groups.items():
        first = es[0]
        m = {"gt_idx": [first["gt_idx"]], "gt": first["gt"], "pred_idx": first["pred_idx"], "pred": first["pred"],
             "edit": first["edit"]}  # fmt: skip
        for o in es[1:]:
            m["gt_idx"].append(o["gt_idx"])
            m["gt"] += o["gt"]
            done_g.add(o["gt_idx"])
        out.append(m)
        done_g.add(first["gt_idx"])
    for gi in range(len(ngl)):
        if gi not in done_g:
            out.append({"gt_idx": [gi], "gt": gl[gi], "pred_idx": [""], "pred": "", "edit": 1})
    return out


def _array(lines: list[str], ks: list[int]) -> str:
    """Several formulas matched as one, written as the evaluator joins them."""
    return (
        "\\begin{array}{l} " + " \\\\ ".join("{" + lines[k].strip("$$").strip("\n") + "}" for k in ks) + " \\end{array}"
    )


def _finish(results: list[dict], ngl, gl, gcats, gitems, npl, pl, pitems, gidx, pidx, img) -> list[dict]:
    """The per-entry fields `match_gt2pred_quick` fills in, with its quirks: removing an unmatched inline formula
    while iterating skips the entry after it."""
    i = 0
    while i < len(results):
        e = results[i]
        if e["gt_idx"] != [""]:
            idxs = e["gt_idx"] if isinstance(e["gt_idx"], list) else [e["gt_idx"]]
            clean = [gcats[k] for k in idxs if gcats[k] not in IGNORES]
            e["gt_category_type"] = Counter(clean or [gcats[k] for k in idxs]).most_common(1)[0][0]
        else:
            e["gt_category_type"] = ""
            e["pred_category_type"] = _pred_category(e["pred_idx"][0], pitems) if e["pred_idx"] != [""] else ""
            if e["pred_category_type"] == "equation_inline":
                results.pop(i)
        e["gt_idx"] = e["gt_idx"] if isinstance(e["gt_idx"], list) else [e["gt_idx"]]
        e["pred_idx"] = e["pred_idx"] if isinstance(e["pred_idx"], list) else [e["pred_idx"]]
        e["gt_position"] = [_gt_position(gitems[k]) for k in e["gt_idx"]] if e["gt_idx"] != [""] else [""]
        e["pred_position"] = pitems[e["pred_idx"][0]]["position"][0] if e["pred_idx"] != [""] else ""
        if e["gt_category_type"] == "equation_isolated" and len(e["gt_idx"]) > 1:
            e["gt"] = _array(gl, e["gt_idx"])
        else:
            e["gt"] = "".join(gl[k] for k in e["gt_idx"]) if e["gt_idx"] != [""] else ""
        e["pred_category_type"] = _pred_category(e["pred_idx"][0], pitems) if e["pred_idx"] != [""] else ""
        e["img_id"] = img
        if "equation" in e["pred_category_type"] and len(e["pred_idx"]) > 1:
            e["pred"] = _array(pl, e["pred_idx"])
        else:
            e["pred"] = "".join(pl[k] for k in e["pred_idx"]) if e["pred_idx"] != [""] else ""
        e["norm_gt"] = "".join(ngl[k] for k in e["gt_idx"]) if e["gt_idx"] != [""] else ""
        e["norm_pred"] = "".join(npl[k] for k in e["pred_idx"]) if e["pred_idx"] != [""] else ""
        i += 1  # after a removal the next entry sits at i, so the evaluator's for-loop skips it
    for e in results:
        if isinstance(e["gt_idx"], list) and e["gt_idx"] != [""]:
            e["gt_idx"] = [gidx[k] for k in e["gt_idx"]]
        if isinstance(e["pred_idx"], list) and e["pred_idx"] != [""]:
            e["pred_idx"] = [int(pidx[k]) for k in e["pred_idx"]]
    return results


def match_quick(gt_items: list[dict], pred_items: list[dict], img: str) -> list[dict]:
    """`match_gt2pred_quick`: ignored categories (headers, captions, ...) absorb near (< 0.1) paragraphs first; the
    rest are matched with merges of consecutive paragraphs for split blocks."""
    order = sorted(enumerate(pred_items), key=lambda p: (p[1].get("fine_category_type") == "equation_inline",
                                                        p[1]["position"][0], p[0]))  # fmt: skip
    pred_items = [x for _, x in order]
    gl, ngl, cats, pl, npl, gt_items, pred_items = _gt_pred_lines(gt_items, pred_items, None)
    if not ngl:
        return [{"gt_idx": [""], "gt": "", "pred_idx": [j], "pred": pl[j], "gt_position": [""],
                 "pred_position": pred_items[j]["position"][0], "norm_gt": "", "norm_pred": npl[j],
                 "gt_category_type": "", "pred_category_type": _pred_category(j, pred_items), "edit": 1,
                 "img_id": img} for j in range(len(npl))]  # fmt: skip
    if not npl:
        return [{"gt_idx": [i], "gt": gl[i], "pred_idx": [""], "pred": "", "gt_position": [_gt_position(gt_items[i])],
                 "pred_position": "", "norm_gt": ngl[i], "norm_pred": "", "gt_category_type": cats[i],
                 "pred_category_type": "", "edit": 1, "img_id": img} for i in range(len(ngl))]  # fmt: skip
    if len(ngl) == 1 and len(npl) == 1:
        return [{"gt_idx": [0], "gt": gl[0], "pred_idx": [0], "pred": pl[0], "gt_position": [_gt_position(gt_items[0])],
                 "pred_position": pred_items[0]["position"][0], "norm_gt": ngl[0], "norm_pred": npl[0],
                 "gt_category_type": cats[0], "pred_category_type": _pred_category(0, pred_items),
                 "edit": _ned(ngl[0], npl[0]), "img_id": img}]  # fmt: skip
    ig = [i for i in range(len(ngl)) if cats[i] in IGNORES]
    keep_g = [i for i in range(len(ngl)) if cats[i] not in IGNORES]
    ig_pred = []
    if ig:  # their own matches are graded nowhere (every metric drops ignored categories), so only this is kept
        ig_pred = [int(x[1]) for x in _np().argwhere(_ned_matrix([ngl[i] for i in ig], npl) < IGNORE_THRES)]
    taken = set(ig_pred)
    keep_p = [j for j in range(len(npl)) if j not in taken]
    k_ngl, k_gl = [ngl[i] for i in keep_g], [gl[i] for i in keep_g]
    k_npl, k_pl, k_items = [npl[j] for j in keep_p], [pl[j] for j in keep_p], [pred_items[j] for j in keep_p]
    conv = _quick_core(k_ngl, k_npl, k_pl)
    res = _merge_duplicates(conv, k_ngl, k_gl)
    res = _finish(res, k_ngl, k_gl, [cats[i] for i in keep_g], [gt_items[i] for i in keep_g], k_npl, k_pl, k_items,
                  keep_g, keep_p, img)  # fmt: skip
    return res


# --- ground truth (dataset/end2end_dataset.py) ----------------------------------------------------------------------


def _order0(x: dict):
    return x.get("order") or 0


def gt_elements(doc: dict, patched: bool) -> dict[str, list[dict]]:
    """`get_page_elements`: blocks by category, `truncated` chains joined into one block (a table to its
    `merged_html`), the merged text marked for T-TextEdit (text blocks; titles, references and code too when
    patched)."""
    related, trunc = [], {}
    for r in doc["extra"]["relation"]:
        if r["relation_type"] != "truncated":
            continue
        s, t = r["source_anno_id"], r["target_anno_id"]
        trunc[s], trunc[t] = "", ""
        hit = False
        for ml in related:
            if s in ml or t in ml:
                ml += [s, t]
                hit = True
        if not hit:
            related.append([s, t])
    out: dict[str, list[dict]] = defaultdict(list)
    for it in doc["layout_dets"]:
        if it["anno_id"] in trunc:
            trunc[it["anno_id"]] = it
        else:
            out[it["category_type"]].append(it)
    merged_kinds = ("text_block", "title", "reference", "code_txt", "code_txt_caption") if patched else ("text_block",)
    for ml in related:
        ml = [x for k, x in enumerate(ml) if k == 0 or x != ml[k - 1]]
        blocks = sorted((trunc[k] for k in ml), key=_order0)
        first, pages_ = blocks[0], {b["page_id"] for b in blocks}
        if first["category_type"] == "table":
            m = {"category_type": "table", "order": first["order"], "anno_id": first["anno_id"],
                 "html": first.get("merged_html") or "".join(b.get("html", "") for b in blocks), "merge_list": blocks,
                 "is_merged": "table", "attribute": {"table_merge_cross_pages": str(len(pages_))}}  # fmt: skip
        else:
            m = {"category_type": first["category_type"], "order": first["order"], "anno_id": first["anno_id"],
                 "text": "".join(b.get("text", "") for b in blocks), "merge_list": blocks,
                 "is_merged": "text_block" if first["category_type"] in merged_kinds else "unknown"}  # fmt: skip
            if m["is_merged"] == "text_block":
                m["attribute"] = {"text_merge_cross_pages": str(len(pages_))}
        out[first["category_type"]].append(m)
    return out


def _sorted(items_: list[dict]) -> list[dict]:
    return [x for _, x in sorted(((x.get("order") or 0, x) for x in items_), key=lambda p: p[0])]


def _figures(doc: dict, preds: list[dict]) -> tuple[int, int, int] | None:
    """`get_figure_matched`: (gt, pred, matched) figures, greedy by IoU >= 0.5 on the same page; None without any."""
    info, gts = doc["page_info"], []
    w, h = info.get("width", []), info.get("height", [])
    for d in doc["layout_dets"]:
        if d.get("category_type") != "figure":
            continue
        pid, poly = d.get("page_id"), d.get("poly", [])
        if pid is None or len(poly) < 6 or not (0 <= pid - 1 < min(len(w), len(h))):
            continue
        pw, ph = w[pid - 1], h[pid - 1]
        if pw and ph:
            gts.append((pid, [poly[0] / pw * 1000, poly[1] / ph * 1000, poly[4] / pw * 1000, poly[5] / ph * 1000]))
    if not gts:
        return None

    def iou(a, b):
        inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0

    pairs = sorted(((gi, pj, v) for gi, (gp, gb) in enumerate(gts) for pj, p in enumerate(preds)
                    if p["page_id"] == gp and (v := iou(gb, p["bbox_norm"])) >= 0.5), key=lambda x: x[2], reverse=True)  # fmt: skip
    mg, mp = set(), set()
    for gi, pj, _ in pairs:
        if gi not in mg and pj not in mp:
            mg.add(gi)
            mp.add(pj)
    return len(gts), len(preds), len(mg)


def _heads(doc: dict) -> tuple[list[dict], dict]:
    nodes = {it["anno_id"]: it for it in doc["layout_dets"] if it["category_type"] == "title"}
    rel = [r for r in doc["extra"]["relation"] if r["relation_type"] == "parent_son"
           and (r["source_anno_id"] in nodes or r["source_anno_id"] == "root") and r["target_anno_id"] in nodes]  # fmt: skip
    return rel, nodes


def _order_edit(entries: list[dict]) -> tuple[int, int] | None:
    """`get_order_paired` then `Edit_dist`: (edit, length) between the annotated and the reply's block order."""
    matched = [
        (e["gt_position"], e["pred_position"]) for e in entries if e["gt_position"] != [""] and e["pred_position"] != ""
    ]
    gt = sorted(x for x in sum((e["gt_position"] for e in entries if e["gt_position"] != [""]), []) if x)
    pred = [x for x in sum((g for g, _ in sorted(matched, key=lambda x: x[1])), []) if x]
    if not (gt or pred):
        return None
    return _lev(pred, gt), max(len(pred), len(gt))


# --- metrics (metrics/head_metric.py, metrics/table_metric.py) ------------------------------------------------------


def head_teds(pred: list[str], rel: list[dict], nodes: dict) -> float:
    """`HeadTEDS`: the reply's `#` heading tree against the annotated title tree, 1 - APTED / max node count."""
    from apted import APTED, Config

    class Node:
        def __init__(self, content, depth=0, order=0):
            self.content, self.depth, self.order, self.parent, self.children = content, depth, order, None, []

    class Cost(Config):
        def rename(self, a, b):
            if a.content or b.content:
                return _lev(a.content, b.content) / max(len(a.content), len(b.content))
            return 0.0

    def md_tree(lines: list[str]) -> Node:
        stack = [Node("root")]
        for i, line in enumerate(lines):
            line = line.strip()
            if re.match(r"^#+\s", line):
                label, title = line.split(" ", 1)
                level, title = len(label), title.strip()
            else:
                level, title = line.count("#"), line.replace("#", "").strip()
                if level == 0:
                    continue
            node = Node(title, level, i + 1)
            while level <= stack[-1].depth:
                stack.pop()
            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)
        return stack[0]

    def json_tree() -> Node:
        root, by = Node("root"), {"root": None}
        by["root"] = root
        for r in rel:
            for k in (r["source_anno_id"], r["target_anno_id"]):
                if k not in by:
                    by[k] = Node(nodes[k]["text"], 0, nodes[k]["anno_id"])
        for r in rel:
            p, c = by[r["source_anno_id"]], by[r["target_anno_id"]]
            p.children.append(c)
            c.parent = p
        for k, n in by.items():
            if k != "root" and n.parent is None:
                root.children.append(n)
                n.parent = root
        for n in by.values():
            n.children.sort(key=lambda x: x.order)
        return root

    def count(n: Node) -> int:
        return 1 + sum(count(c) for c in n.children)

    try:
        p, g = md_tree(pred), json_tree()
        return 1.0 - APTED(p, g, Cost()).compute_edit_distance() / max(count(p), count(g))
    except Exception:
        return 0.0


_SPAN = re.compile(r"\b(row|col)span\s*=\s*([^\s>]*)", re.I)


def _fix_spans(h: str) -> str:
    if not h or "span" not in h:
        return h
    return _SPAN.sub(lambda m: '%sspan="%s"' % (m[1], d[0] if (d := re.search(r"\d+", m[2])) else "1"), h)


def table_teds(job: tuple[str, str, bool], limit_s: int = 300) -> float:
    """TEDS of one matched table (its normalized HTML), 0 past `limit_s`; identical tables score 1 directly."""
    from vlmbench.evals.datasets.omnidocbench import _teds

    pred, gt, patched = job
    if patched:
        pred, gt = _fix_spans(pred), _fix_spans(gt)
    if not pred or not gt:
        return 0.0
    if pred == gt:
        return 1.0

    def expire(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(limit_s)
    try:
        return _teds(pred, gt)
    except TimeoutError:
        return 0.0
    except Exception:
        return 0.0
    finally:
        signal.alarm(0)


def _cdm_strip(s: str, pred: bool) -> str:
    """The delimiters CDM's sample preparation strips before rendering (`_process_single_cdm_sample`)."""
    if pred:
        s = s.split("```latex")[-1].split("```")[0]
    s = s.lstrip("$$").rstrip("$$").strip().lstrip("$").rstrip("$").strip()
    s = re.sub(r"^\\\[|\\\]$", "", s.strip()).strip()
    return re.sub(r"^\\\(|\\\)$", "", s.strip()).strip()


def formula_score(gt: str, pred: str) -> float:
    """The CDM stand-in: CDM's delimiter stripping, then 1 - normalized edit distance of the `normalized_formula`
    forms; an empty prediction scores 0, as in CDM."""
    gt, pred = _cdm_strip(gt, False), _cdm_strip(pred, True)
    if not pred.strip():
        return 0.0
    g, p = normalized_formula(gt), normalized_formula(pred)
    return 1.0 - _lev(p, g) / max(len(p), len(g)) if (g or p) else 1.0


# --- one document ---------------------------------------------------------------------------------------------------


def _match_doc(doc: dict, reply: str, patched: bool, timeout_s: int) -> dict:
    """`process_get_matched_elements`, plus the figure match, for one document."""
    _PATCHED[0] = patched
    img = doc["page_info"]["image_path"]
    figure = _figures(doc, figure_boxes(reply))
    pred = md_tex_filter(reply)
    gt = gt_elements(doc, patched)
    rel, nodes = _heads(doc)
    mix = [x for c, xs in pred.items() if c not in ("html_table", "latex_table", "md2html_table") for x in xs]
    gt_mix = [x for c in [*TEXT, "equation_isolated"] for x in gt.get(c, [])]
    gt_mix = _sorted(gt_mix) if gt_mix else gt_mix
    tables, merged_tables = [], []
    if gt.get("table"):
        gt_table = _sorted(gt["table"])
        merged_gt = [t for t in gt_table if t.get("is_merged") == "table"]
        n_latex, n_html = len(pred.get("latex_table") or []), len(pred.get("html_table") or [])
        kind = "latex_table" if n_latex > n_html else "html_table"
        preds_ = [] if n_latex == n_html == 0 else pred[kind]
        t, extra = match_simple(gt_table, preds_, kind, img)
        tables = [x for x in t if x["gt_idx"] != [""]]
        merged_tables = [x for x in match_simple(merged_gt, preds_, kind, img)[0] if x["gt_idx"] != [""]]
        if kind == "latex_table":  # the evaluator blanks LaTeX table predictions (they would need latexmlc)
            for x in tables + merged_tables:
                x["pred"] = ""
        if extra:
            mix += extra
    mix.sort(key=lambda x: x["position"][0])  # `get_head_paired` sorts the list it is given
    heads = [x["content"].strip() for x in mix if x["content"].strip().startswith("#")]

    def expire(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(timeout_s)
    try:
        match = match_quick(gt_mix, mix, img)
    except TimeoutError:
        match = match_simple(gt_mix, mix, "text_all", img)[0]
    finally:
        signal.alarm(0)
    merged_gt_mix = [g for g in gt_mix if g.get("is_merged") == "text_block"]
    merged_text = [x for x in match_simple(merged_gt_mix, mix, "text_all", img)[0] if x["gt_idx"] != [""]]
    plain = [x for x in match if x.get("gt_category_type") in TEXT]
    formulas = [x for x in match if x.get("gt_category_type") == "equation_isolated" and x["gt_idx"] != [""]]
    plain = [x for x in plain if x["gt_category_type"] not in IGNORES]
    return {
        "text": [(x.get("norm_gt") or x["gt"], x.get("norm_pred") or x["pred"]) for x in plain],
        "truncated_text": [(x.get("norm_gt") or x["gt"], x.get("norm_pred") or x["pred"]) for x in merged_text],
        "formulas": [(x["gt"], x["pred"], x.get("pred_category_type"), x.get("norm_pred") or "") for x in formulas],
        "tables": [
            (normalized_html_table(x["pred"]) or x["pred"], normalized_html_table(x["gt"]) or x["gt"]) for x in tables
        ],
        "truncated_tables": [
            (normalized_html_table(x["pred"]) or x["pred"], normalized_html_table(x["gt"]) or x["gt"])
            for x in merged_tables
        ],  # fmt: skip
        "order": _order_edit(plain) if plain else None,
        "heading": head_teds(heads, rel, nodes),
        "figure": figure,
    }


def _edit_sums(pairs: list[tuple[str, str]]) -> tuple[int, int]:
    e = n = 0
    for g, p in pairs:
        if g or p:
            e, n = e + _lev(p, g), n + max(len(p), len(g))
    return e, n


def _match_job(job: tuple[dict, str, bool, int]) -> dict:
    return _match_doc(*job)


@cache
def _pool() -> ProcessPoolExecutor:
    import multiprocessing

    return ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))


def _rate(units: dict[str, dict[str, list]], ids) -> dict[str, float]:
    tot: dict[str, list[float]] = {}
    for i in ids:
        for g, (s, n) in units.get(i, {}).items():
            a = tot.setdefault(g, [0.0, 0.0])
            a[0], a[1] = a[0] + s, a[1] + n
    return {g: 100 * s / n for g, (s, n) in tot.items() if n}


def grade(docs: dict[str, dict], its: list[Item], replies: dict[str, str], patched: bool, timeout_s: int) -> dict:
    """The eight leaderboard metrics over the documents, as `score` returns them. Units per document: edit-based and
    per-document metrics `[value, 1]`, per-table and per-formula ones `[sum, count]`, so pooling them reproduces the
    evaluator's means."""
    jobs = [(docs[it.id], replies.get(it.id) or "", patched, timeout_s) for it in its]
    matched = list(_pool().map(_match_job, jobs))
    table_jobs = sorted({(p, g, patched) for m in matched for p, g in m["tables"] + m["truncated_tables"]})
    teds = dict(zip(table_jobs, _pool().map(table_teds, table_jobs)))
    any_text = any(m["text"] for m in matched)
    units: dict[str, dict[str, list]] = {}
    for it, m in zip(its, matched):
        u: dict[str, list] = {}
        text = list(m["text"])
        f_scores = []
        for g, p, cat, norm_p in m["formulas"]:
            if cat not in ("equation_inline", "equation_isolated", "") and any_text:  # matched to text: graded as text
                try:
                    g2 = _latex2text(g)
                except Exception:
                    g2 = g or ""
                text.append((clean_string(g2) or g2, norm_p or p))
            elif cat in ("equation_inline", "equation_isolated", ""):
                f_scores.append(formula_score(g, p))
        for name, pairs in (("text", text), ("truncated_text", m["truncated_text"])):
            e, n = _edit_sums(pairs)
            if n:
                u[name] = [1 - e / n, 1]
        for name, key in (("table_teds", "tables"), ("truncated_table_teds", "truncated_tables")):
            if m[key]:
                u[name] = [sum(teds[(p, g, patched)] for p, g in m[key]), len(m[key])]
        if f_scores:
            u["formula"] = [sum(f_scores), len(f_scores)]
        if m["order"] and m["order"][1]:
            u["reading_order"] = [1 - m["order"][0] / m["order"][1], 1]
        if m["figure"]:
            gt_n, pred_n, hit = m["figure"]
            prec, rec = (hit / pred_n if pred_n else 0), hit / gt_n
            u["figure_f1"] = [2 * prec * rec / (prec + rec) if prec + rec else 0.0, 1]
        u["heading_teds"] = [m["heading"], 1]
        units[it.id] = u
    overall = lambda r: sum(r.values()) / len(r) if r else 0.0  # noqa: E731
    metrics = _rate(units, [it.id for it in its])
    by: dict[str, list[str]] = {}
    for it in its:
        by.setdefault(it.category, []).append(it.id)
    return {
        "score": overall(metrics),
        "metrics": {g: metrics[g] for g in GROUPS if g in metrics},
        "by_category": {k: overall(_rate(units, ids)) for k, ids in by.items()},
        "units": units,
    }


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    return grade(_docs(spec.revision), its, replies, patched=False, timeout_s=30)
