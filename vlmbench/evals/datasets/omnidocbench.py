"""OmniDocBench (opendatalab/OmniDocBench): 1,651 annotated pages of 10 document types, graded by a port of its own
end-to-end evaluator (`omnidocbench-eval` 1.6, github.com/opendatalab/OmniDocBench @ f133a71e). The evaluator is not
installable here (Python <3.12, pins numpy 1.24, a top-level `src` package), so its definitions are reproduced.

A reply is split as its `md_tex_filter` does (HTML and pipe tables, `$$` / `\\[` display math, paragraphs on blank
lines) and matched to the page's annotated blocks, then graded per page as its leaderboard does:
- text: 1 - normalized edit distance (sum of distances / sum of lengths) over the matched text blocks, on text with
  inline math turned to unicode and everything but letters, digits and CJK dropped;
- formula: the same over display formulas, LaTeX normalized as the evaluator does. It stands in for CDM, which needs
  TeX Live, ImageMagick and Ghostscript; the leaderboard reports both, overall uses CDM;
- table: TEDS (PubTabNet's, as the evaluator vendors it) per annotated table, a page their mean, a missing table 0;
- reading order: 1 - edit distance between the matched blocks' annotated order and the reply's (reported, not in
  the score).
Score, the leaderboard's Overall: ((1 - text edit) * 100 + table TEDS + formula) / 3, each component a mean over the
pages that have it (a sample without tables or formulas averages what it has). Metrics are each component, 0-100.

As in the evaluator: headers, footers, page numbers, captions and footnotes absorb the reply's copy but are not
graded; reply text that matches no annotated block (figure text, a description) is not penalised; text blocks are
matched by its multi-granularity rule (consecutive reply paragraphs merge into one block, blocks merge into one
paragraph). Simplified: formulas match one-to-one (no array splitting, no rescue of formulas written as text), LaTeX
`tabular` replies are not converted to HTML (the evaluator needs latexmlc), inline `$...$` stays in its paragraph, and
a page that runs past 120 s is graded as an empty reply (the evaluator's TEDS timeout scores 0 too).
"""

import html as htmllib
import json
import re
import signal
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "opendatalab/OmniDocBench"
TEXT = {"text_block", "title", "code_txt", "code_txt_caption", "reference"}
IGNORED = {"equation_caption", "figure_caption", "figure_footnote", "table_caption", "table_footnote",
           "code_algorithm", "code_algorithm_caption", "header", "footer", "page_footnote", "page_number"}  # fmt: skip
_FURNITURE = re.compile(r"<page_(header|footer)>\s*(.*?)\s*</page_\1>", re.S | re.I)
_INLINE_HTML = re.compile(r"</?(u|b|i|s|em|strong|del|span|mark)\b[^>]*>|<br\s*/?>", re.I)
_DISPLAY = re.compile(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\$(.*?)\$|\\\((.*?)\\\)", re.S)
_INLINE = re.compile(r"\$(.*?)\$|\\\((.*?)\\\)")
_TABLE_TAG = re.compile(r"<table\b[^>]*>|</table>", re.I)
_CIRCLED = {chr(0x2460 + i): str(i + 1) for i in range(20)} | {chr(0x24D0 + i): chr(97 + i) for i in range(26)}
_CIRCLED |= {chr(0x24B6 + i): chr(65 + i) for i in range(26)}
_FORMULA_DROP = ["\\mathbf", "\\mathrm", "\\mathnormal", "\\mathit", "\\mathbb", "\\mathcal", "\\mathscr", "\\mathfrak",
                 "\\mathsf", "\\mathtt", "\\textbf", "\\text", "\\boldmath", "\\boldsymbol", "\\operatorname", "\\bm",
                 "\\symbfit", "\\mathbfcal", "\\symbf", "\\scriptscriptstyle", "\\notag", "\\setlength", "\\coloneqq",
                 "\\space", "\\thickspace", "\\thinspace", "\\medspace", "\\nobreakspace", "\\negmedspace", "\\quad",
                 "\\qquad", "\\enspace", "\\substackw", " ", "$$", "\\left", "\\right", "\\displaystyle"]  # fmt: skip
_CELL_SUBS = [("\\checkmark", "√"), ("\\checked", "√"), ("\\surd", "√"), ("\\times", "×"), ("\\xmark", "×"),
              ("\\crossmark", "×"), ("\\pm", "±"), ("\\mp", "∓"), ("\\alpha", "α"), ("\\beta", "β"),
              ("\\gamma", "γ"), ("\\gama", "γ"), ("\\mu", "μ"), ("\\lambda", "λ"), ("\\theta", "θ"),
              ("\\eta", "η"), ("\\pi", "π"), ("\\rho", "ρ"), ("\\sigma", "σ"), ("\\omega", "ω"), ("\\delta", "δ"),
              ("\\Delta", "Δ"), ("\\epsilon", "ε"), ("\\varepsilon", "ε"), ("\\phi", "φ"), ("\\varphi", "φ"),
              ("\\tau", "τ"), ("\\partial", "∂"), ("\\varnothing", "∅"), ("\\emptyset", "∅"), ("\\sim", "～"),
              ("✓", "√"), ("✔", "√"), ("☑", "√"), ("✅", "√"), ("🗸", "√"), ("✗", "×"), ("✘", "×"), ("✕", "×"),
              ("✖", "×"), ("☒", "×"), ("❌", "×"), ("╳", "×"), ("⨯", "×"), ("Ø", "∅"), ("∼", "～"), ("〜", "～"),
              ("\\%", "%"), ("\\#", "#"), ("\\&", "&"), ("\\_", "_")]  # fmt: skip


@cache
def _root(revision: str) -> Path:
    from huggingface_hub import snapshot_download

    files = ["OmniDocBench.json", "images/*"]
    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision, allow_patterns=files, max_workers=8))


@cache
def _pages(revision: str) -> dict[str, dict]:
    pages = json.loads((_root(revision) / "OmniDocBench.json").read_text())
    return {Path(p["page_info"]["image_path"]).stem: p for p in pages}


def normalize(text: str) -> str:
    """Header / footer tags unwrapped (OmniDocBench matches that text, ungraded), and outside tables `<sup>` / `<sub>`
    written as its annotation's LaTeX and other inline HTML dropped, so tag names are not graded as text."""
    text = _FURNITURE.sub(r"\2", text)
    parts = re.split(r"(<table\b.*?</table>)", text, flags=re.S | re.I)
    for i in range(0, len(parts), 2):
        p = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", parts[i], flags=re.S | re.I)
        p = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", p, flags=re.S | re.I)
        parts[i] = _INLINE_HTML.sub(lambda m: " " if m[0].lower().startswith("<br") else "", p)
    return "".join(parts)


def items(spec: DatasetSpec) -> list[Item]:
    root, keep = _root(spec.revision), set(spec.subsets or ())
    return [
        Item(i, p["page_info"]["page_attribute"]["data_source"], str(root / "images" / p["page_info"]["image_path"]))
        for i, p in _pages(spec.revision).items()
        if not keep or p["page_info"]["page_attribute"]["subset"] in keep
    ]


# --- normalization (src/core/preprocess/data_preprocess.py) ---------------------------------------------------------


def _lev(a, b) -> int:
    from Levenshtein import distance

    return distance(a, b)


def _ned(a: str, b: str) -> float:
    return _lev(a, b) / max(len(a), len(b)) if a or b else 0.0


def _circled(text: str) -> str:
    text = re.sub(r"\\textcircled\s*\{\s*([^{}]+?)\s*\}", lambda m: m[1].strip(), str(text))
    return "".join(_CIRCLED.get(c, c) for c in text)


def _balanced(s: str) -> bool:
    depth, i = 0, 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in "{}[]()":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth < 0:
            return False
        i += 1
    return depth == 0


def _to_unicode(latex: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text

    if not _balanced(latex) or len(latex) > 8000:
        return latex
    try:
        return LatexNodes2Text().latex_to_text(latex)
    except Exception:
        return latex


def _ntext(text: str) -> str:
    """`normalized_text`: inline math with `\\`, `^` or `_` as unicode, then letters, digits and CJK only."""
    text = _circled(text)
    out, last = [], 0
    for m in _INLINE.finditer(text):
        body = re.sub(r"\\([\\_&%^])", "", m[1] if m[1] is not None else m[2])
        complex_ = len(body) > 128 or any(t in body for t in ("\\begin{", "\\end{", "\\array", "\\matrix", "\\\\", "&"))
        if any(c in body for c in "\\^_") and not complex_ and body.count("{") == body.count("}"):
            out += [text[last : m.start()], _to_unicode(body + (" " if body.endswith("\\") else "")).strip()]
            last = m.end()
    text = _circled("".join(out) + text[last:])
    for s in ("\\t", "\\n", "\t", "\n", "/t", "/n"):
        text = text.replace(s, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text)


def _drop_group(text: str, cmd: str) -> str:
    """`cmd{...}` removed, braces balanced (`\\tag*{..}`, `\\phantom{..}`)."""
    for m in reversed(list(re.finditer(re.escape(cmd) + r"(?![A-Za-z])\s*\*?\s*\{", text))):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        text = text[: m.start()] + text[i:] if depth == 0 else text[: m.start()] + text[m.start() + len(cmd) :]
    return text


def _strip_delims(text: str) -> str:
    text, changed = text.strip(), True
    while text and changed:
        changed = False
        for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
            if text.startswith(a) and text.endswith(b) and len(text) >= len(a) + len(b):
                text, changed = text[len(a) : len(text) - len(b)].strip(), True
                break
    return text


def _nformula(text: str) -> str:
    """`normalized_formula`: delimiters, tags, fonts, spacing and alignment dropped, braces flattened, lower case."""
    text = _strip_delims(_circled(text))
    text = _drop_group(text, "\\tag").replace("\\notag", "").replace("\\nonumber", "")
    if m := re.search(r"\\\[(.+?)(?<!\\)\\\]|\\\((.+?)(?<!\\)\\\)", text):
        text = (m[1] or m[2] or "").strip()
    text = _drop_group(text, "\\phantom")
    text = re.sub(r"\\[!,;:]", "", text)
    text = re.sub(r"(?<!\\)&", "", text)
    text = text.replace("\\mid", "|").replace("\\vert", "|").replace(r"\{", "").replace(r"\}", "").replace("~", "")
    for p in (r"\\hspace\{.*?\}", r"\\begin\{.*?\}", r"\\end\{.*?\}", r"\\arraycolsep.*?\}", r"\{[lcr| ]+\}"):
        text = re.sub(p, "", text)
    text = text.strip(".")
    for f in _FORMULA_DROP:
        text = text.replace(f, "")
    ops = ("=", "+", "-", "\\\\", "^", "_", "[", "]", "(", ")")
    prev = None
    while text != prev:
        prev = text
        text = re.sub(r"\{([A-Za-z0-9.+\-]+)\}", r"\1", text).replace("{}", "")
        text = re.sub(r"\}\}+", "}", re.sub(r"\{\{+", "{", text))
        text = re.sub(r"\{([^{}]{1,160})\}", lambda m: m[1] if any(t in m[1] for t in ops) else m[0], text)
    return re.sub(r"\s+", "", text.lower())


def _cell(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ").replace("\u200b", "")
    for _ in range(2):
        for a, b in _CELL_SUBS:
            text = text.replace(a, b)
        t, changed = text.strip(), True
        while t and changed:
            changed = False
            for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
                inner = t[len(a) : len(t) - len(b)].strip()
                if t.startswith(a) and t.endswith(b) and len(t) >= len(a) + len(b) and 0 < len(inner) <= 128:
                    if "<" not in inner and ">" not in inner and "$" not in inner:
                        t, changed = inner, True
                        break
        text = t
    if text.strip() in ("-", "–", "−", "－", "—"):
        text = "—"
    return re.sub(r"\s+", " ", text).strip()


def _ntable(text: str) -> str:
    """`normalized_html_table`: `th` as `td`, sections and spans unwrapped, cells normalized, attributes but spans
    dropped, inline tags removed, wrapped as `<html><body><table border="1" >`."""
    from bs4 import BeautifulSoup, NavigableString

    if "<table" not in text.replace(" ", "").replace("'", '"'):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for th in soup.find_all("th"):
        th.name = "td"
    for t in soup.find_all(["thead", "tbody", "span"]):
        t.unwrap()
    for m in soup.find_all("math"):
        m.replace_with(f"${m.get('alttext', '')}$")
    for c in soup.find_all(["td"]):
        for n in list(c.descendants):
            if isinstance(n, NavigableString) and (v := _cell(str(n))) != str(n):
                n.replace_with(v)
    t = unicodedata.normalize("NFKC", htmllib.unescape(str(soup)).replace("\n", "")).strip()
    t = "".join(re.findall(r"<table\b[^>]*>(.*)</table>", t, re.S | re.I))
    for a in ("style", "height", "width", "align", "class"):
        t = re.sub(f'( {a}=".*?")', "", t)
    t = re.sub(r"\s+", " ", re.sub("</?tbody>", "", t))
    for tag in ("sup", "sub", "span", "div", "p"):
        t = t.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
    t = re.sub("<colgroup>.*?</colgroup>", "", t.replace('<spandata-span-identity="">', ""))
    return '<html><body><table border="1" >' + t + "</table></body></html>"


# --- a reply split into blocks (src/core/preprocess/extract.py: md_tex_filter) ------------------------------------


def _blank(s: str, a: int, b: int) -> str:
    return s[:a] + " " * (b - a) + s[b:]


def _pipe_tables(content: str) -> list[tuple[int, int, str]]:
    """Runs of `| ... |` lines as HTML, the first row the header, the second (the rule) dropped."""
    out, lines, pos = [], content.split("\n"), 0
    run: list[tuple[int, str]] = []
    for line in lines + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|") and len(line.strip()) > 1:
            run.append((pos, line))
        else:
            if len(run) >= 2:
                rows = [r for k, (_, r) in enumerate(run) if k != 1]
                body = "".join(
                    "<tr>" + "".join(f"<td>{c.strip()}</td>" for c in r.strip()[1:-1].split("|")) + "</tr>"
                    for r in rows
                )
                out.append((run[0][0], run[-1][0] + len(run[-1][1]), f"<table>{body}</table>"))
            run = []
        pos += len(line) + 1
    return out


def _blocks(md: str) -> list[dict]:
    """The reply as `{kind: text | formula | table, content, pos}`, in reply order."""
    md = re.sub(r"!\[.*?\]\(.*?\)", lambda m: " " * len(m[0]), md)
    for p in (r"^```markdown\n?", r"^```html\n?", r"^```latex\n?", r"```\n?$"):
        md = re.sub(p, "", md, flags=re.M)
    md = re.sub(r" {4,}", "    ", re.sub(r"_{4,}", "____", md))
    for t in ("<html>", "</html>", "<body>", "</body>"):
        md = md.replace(t, "")
    out, depth, start = [], 0, 0
    for m in _TABLE_TAG.finditer(md):
        if not m[0].startswith("</"):
            depth, start = depth + 1, m.start() if depth == 0 else start
        elif depth:
            depth -= 1
            if not depth:
                out.append({"kind": "table", "content": md[start : m.end()], "pos": start})
    for b in out:
        md = _blank(md, b["pos"], b["pos"] + len(b["content"]))
    for m in list(_DISPLAY.finditer(md)):
        if m[1] is not None or m[2] is not None:
            body = " ".join((m[1] or m[2]).split("\n"))
            out.append({"kind": "formula", "content": f"\\[{body}\\]", "pos": m.start()})
            md = _blank(md, m.start(), m.end())
    for a, b, table in _pipe_tables(md):
        out.append({"kind": "table", "content": table, "pos": a})
        md = _blank(md, a, b)
    for p in (r"\\title\s*\{\s*(.*?)\s*\}", r"\\text\s*\{\s*(.*?)\s*\}", r"\\section\*?\{\s*(.*?)\s*\}"):
        md = re.sub(p, lambda m: m[1] + " " * (len(m[0]) - len(m[1])), md, flags=re.S)
    sep = "\n\n" if "\n\n" in md else "\n"
    pos = 0
    for chunk in md.split(sep):
        text = "\n".join(x.strip() for x in chunk.strip().split("\n") if x.strip())
        if text:
            out.append({"kind": "text", "content": text, "pos": pos})
        pos += len(chunk) + len(sep)
    return sorted(out, key=lambda b: b["pos"])


def _order(b: dict) -> float:
    return b.get("order") or 0


# --- ground truth (src/dataset/end2end_dataset.py: get_page_elements) -----------------------------------------------


def _gt(page: dict) -> dict[str, list[dict]]:
    """Blocks by category, ignored ones out, `truncated` chains joined into their first block, sorted by order."""
    groups: list[set] = []
    for r in page["extra"]["relation"]:
        if r["relation_type"] == "truncated":
            g = {r["source_anno_id"], r["target_anno_id"]}
            groups = [h for h in groups if not (h & g)] + [g.union(*[h for h in groups if h & g])]
    linked = set().union(*groups)
    out, held = defaultdict(list), {}
    for b in page["layout_dets"]:
        if b["anno_id"] in linked:
            held[b["anno_id"]] = b
        elif not b.get("ignore"):
            out[b["category_type"]].append(b)
    for g in groups:
        bs = sorted((held[k] for k in g if k in held), key=_order)
        if bs and not any(b.get("ignore") for b in bs):
            out[bs[0]["category_type"]].append({**bs[0], "text": "".join(b.get("text", "") for b in bs)})
    return {k: sorted(v, key=_order) for k, v in out.items()}


# --- matching (src/core/matching/match_quick.py: match_gt2pred_quick) ------------------------------------------------


def _sub_pred(g: str, p: str) -> float | None:
    """`sub_pred_fuzzy_matching`: the best window of `g` against all of `p`."""
    if not (len(g) >= len(p) > 0):
        return None
    return min(_lev(g[i : i + len(p)], p) for i in range(len(g) - len(p) + 1)) / len(p)


def _sub_gt(p: str, g: str) -> float:
    """`sub_gt_fuzzy_matching`: the best window of `p` against all of `g`."""
    if not (len(p) >= len(g) > 0):
        return 1.0
    return min(_lev(p[i : i + len(g)], g) for i in range(len(p) - len(g) + 1)) / len(g)


def _judge(g: str, parts: list[str]) -> tuple[bool, bool]:
    """`judge_pred_merge`: (merge the last part in, keep extending)."""
    cur, merged = " ".join(parts[:-1]), " ".join(parts)
    if _ned(g, merged) > _ned(g, cur):
        return False, False
    if any((d := _sub_pred(g, p)) is None or d > 0.6 for p in parts[:-1]):
        return False, False
    if (d := _sub_pred(g, parts[-1])) is None:
        return False, False
    return d < 0.6, len(merged) <= len(g)


def _merge_runs(cost, gl: list[str], pl: list[str]) -> list[list[int]]:
    """`deal_with_truncated`: a block with no near match (< 0.25) takes the run of consecutive free paragraphs that
    best covers it; overlapping runs go to the cheaper block. Returns the reply's paragraphs as runs."""
    import numpy as np

    near = np.argwhere(cost < 0.25)
    held_g, held_p = set(near[:, 0].tolist()), set(near[:, 1].tolist())
    cands = []
    for gi in (g for g in range(len(gl)) if g not in held_g):
        best = None
        for pj in (p for p in range(len(pl)) if p not in held_p):
            step = 1
            while step < 160 and pj + step < len(pl) and pj + step not in held_p:
                ok, more = _judge(gl[gi], pl[pj : pj + step + 1])
                if not ok:
                    break
                step += 1
                if not more:
                    break
            d = _ned(gl[gi], " ".join(pl[pj : pj + step]))
            if best is None or d < best[0]:
                best = (d, list(range(pj, pj + step)))
        if best and len(best[1]) > 1:
            cands.append(best)
    taken, starts = set(), {}
    for _, run in sorted(cands):
        if not taken & set(run):
            taken |= set(run)
            starts[run[0]] = run
    runs, j = [], 0
    while j < len(pl):
        runs.append(starts.get(j, [j]))
        j += len(runs[-1])
    return runs


def _match(gn: list[str], gcat: list[str], pn: list[str]) -> list[tuple[list[int], list[int]]]:
    """(block indices, paragraph indices) pairs for the non-ignored blocks; ignored blocks only absorb paragraphs."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not gn or not pn:
        return [([i], []) for i in range(len(gn)) if gcat[i] not in IGNORED]
    if len(gn) == 1 and len(pn) == 1:
        return [([0], [0])] if gcat[0] not in IGNORED else []
    absorbed = {j for i in range(len(gn)) if gcat[i] in IGNORED for j in range(len(pn)) if _ned(gn[i], pn[j]) < 0.25}
    G = [i for i in range(len(gn)) if gcat[i] not in IGNORED]
    P = [j for j in range(len(pn)) if j not in absorbed]
    gl, pl = [gn[i] for i in G], [pn[j] for j in P]
    if not G or not P:
        return [([i], []) for i in G]
    ned = lambda a, b: np.array([[_ned(x, y) for y in b] for x in a])  # noqa: E731
    runs = _merge_runs(ned(gl, pl), gl, pl)
    cost = ned(gl, [" ".join(pl[k] for k in r) for r in runs])
    rows, cols = linear_sum_assignment(cost)
    hit = dict(zip(rows.tolist(), cols.tolist()))
    matches, free_g = {}, []
    for gi in range(len(gl)):  # `process_matches`: a pair past 0.7 is no match
        if gi in hit and cost[gi, hit[gi]] <= 0.7:
            matches[gi] = runs[hit[gi]]
        else:
            free_g.append(gi)
    final, done = {}, set()  # `merge_matches`: unmatched blocks found inside a paragraph join it
    for gi, run in matches.items():
        final.setdefault(tuple(run), []).append(gi)
        done.add(gi)
    for pj in range(len(pl)):
        hits = [g for g in free_g if gcat[G[g]] != "equation_isolated" and _sub_gt(pl[pj], gl[g]) < 0.4]
        if hits:
            new = [g for g in hits if g not in done]
            final.setdefault((pj,), []).extend(new)
            done.update(new)
    entries = [(g, list(k)) for k, gs in final.items() for g in sorted(set(gs))]
    used_p = {j for k in final for j in k}
    left_g = [g for g in range(len(gl)) if g not in done]
    left_p = [j for j in range(len(pl)) if j not in used_p]
    if left_p and left_g:  # `convert_final_matches`: leftovers paired off, the surplus paragraphs dropped
        r, c = linear_sum_assignment(ned([gl[g] for g in left_g], [pl[j] for j in left_p]))
        entries += [(left_g[a], [left_p[b]]) for a, b in zip(r.tolist(), c.tolist())]
    by_pred: dict[tuple, list[int]] = {}
    for g, ps in entries:  # `merge_duplicates_add_unmatched`
        by_pred.setdefault(tuple(ps), []).append(g)
    out = [([G[g] for g in gs], [P[j] for j in ps]) for ps, gs in by_pred.items()]
    paired = {g for gs in by_pred.values() for g in gs}
    return out + [([G[g]], []) for g in range(len(gl)) if g not in paired]


def _simple(gn: list[str], pn: list[str]) -> list[int | None]:
    """`match_gt2pred_simple`: one-to-one by least total edit distance, the paragraph per block (None: unmatched)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not gn or not pn:
        return [None] * len(gn)
    r, c = linear_sum_assignment(np.array([[_ned(g, p) for p in pn] for g in gn]))
    hit = dict(zip(r.tolist(), c.tolist()))
    return [hit.get(i) for i in range(len(gn))]


# --- TEDS (src/metrics/table_metric.py, PubTabNet's) ----------------------------------------------------------------


def _teds(pred: str, true: str) -> float:
    from apted import APTED, Config
    from apted.helpers import Tree
    from lxml import html

    class Node(Tree):
        def __init__(self, tag, colspan=None, rowspan=None, content=None):
            self.tag, self.colspan, self.rowspan, self.content, self.children = tag, colspan, rowspan, content, []

    class Cost(Config):
        def rename(self, a, b):
            if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            if a.tag == "td" and (a.content or b.content):
                return _lev(a.content, b.content) / max(len(a.content), len(b.content))
            return 0.0

    def tokens(n) -> list[str]:
        out = [f"<{n.tag}>", *(n.text or "")]
        for c in n:
            out += tokens(c)
        if n.tag != "unk":
            out.append(f"</{n.tag}>")
        if n.tag != "td" and n.tail is not None:
            out += list(n.tail)
        return out

    def tree(n) -> Node:
        if n.tag == "td":
            return Node("td", int(n.attrib.get("colspan", "1")), int(n.attrib.get("rowspan", "1")), tokens(n)[1:-1])
        node = Node(n.tag)
        node.children = [tree(c) for c in n]
        return node

    if not pred or not true:
        return 0.0
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    p, t = (html.fromstring(x, parser=parser).xpath("body/table") for x in (pred, true))
    if not p or not t:
        return 0.0
    n = max(len(p[0].xpath(".//*")), len(t[0].xpath(".//*")))
    return 1.0 - APTED(tree(p[0]), tree(t[0]), Cost()).compute_edit_distance() / n


# --- one page ------------------------------------------------------------------------------------------------------


def _grade_page(page: dict, reply: str) -> dict:
    """{"text" / "formula": (edit, length) sums, "table": [TEDS per table], "order": edit} for what the page has."""
    gt, blocks = _gt(page), _blocks(reply)
    mix = sorted((b for c in (*TEXT, *IGNORED, "equation_isolated") for b in gt.get(c, [])), key=_order)
    mix = [(b, n) for b in mix if (n := _nformula(b.get("latex") or "") if b["category_type"] == "equation_isolated"
                                   else _ntext(b.get("text") or ""))]  # fmt: skip
    preds = [(b, n) for b in blocks if b["kind"] != "table"
             if (n := _nformula(b["content"]) if b["kind"] == "formula" else _ntext(b["content"]))]  # fmt: skip
    out: dict = {}
    pairs = []  # (block orders, reply position) for reading order
    text = [0, 0]
    for gs, ps in _match([n for _, n in mix], [b["category_type"] for b, _ in mix], [n for _, n in preds]):
        cat = Counter(mix[g][0]["category_type"] for g in gs).most_common(1)[0][0]
        if cat not in TEXT:
            continue
        g, p = "".join(mix[i][1] for i in gs), "".join(preds[j][1] for j in ps)
        text[0], text[1] = text[0] + _lev(p, g), text[1] + max(len(p), len(g))
        pairs.append(([mix[i][0].get("order") for i in gs], preds[ps[0]][0]["pos"] if ps else None))
    if text[1]:
        out["text"] = text
    gf = [b for b in gt.get("equation_isolated", []) if b.get("latex")]
    if gf:
        pf = [b for b in blocks if b["kind"] == "formula"]
        gn, pn = [_nformula(b["latex"]) for b in gf], [_nformula(b["content"]) for b in pf]
        f = [0, 0]
        for b, g, k in zip(gf, gn, _simple(gn, pn)):
            p = pn[k] if k is not None else ""
            f[0], f[1] = f[0] + _lev(p, g), f[1] + max(len(p), len(g))
            pairs.append(([b.get("order")], pf[k]["pos"] if k is not None else None))
        if f[1]:
            out["formula"] = f
    if gt.get("table"):
        gtab = gt["table"]
        ptab = [b for b in blocks if b["kind"] == "table"]
        gn, pn = [_ntable(b.get("html", "")) for b in gtab], [_ntable(b["content"]) for b in ptab]
        out["table"] = []
        for b, g, k in zip(gtab, gn, _simple(gn, pn)):
            out["table"].append(_teds(pn[k], g) if k is not None else 0.0)
            pairs.append(([b.get("order")], ptab[k]["pos"] if k is not None else None))
    gt_order = sorted(x for os_, _ in pairs for x in os_ if x)  # `get_order_paired`
    pred_order = [x for os_, _ in sorted((q for q in pairs if q[1] is not None), key=lambda q: q[1]) for x in os_ if x]
    if gt_order or pred_order:
        out["order"] = _lev(gt_order, pred_order) / max(len(gt_order), len(pred_order))
    return out


def _grade(job: tuple[dict, str], limit_s: int = 120) -> dict:
    """A page that runs past `limit_s` (a runaway reply, a huge table) is graded as an empty reply."""

    def expire(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(limit_s)
    try:
        return _grade_page(*job)
    except TimeoutError:
        return _grade_page(job[0], "")
    finally:
        signal.alarm(0)


@cache
def _pool() -> ProcessPoolExecutor:
    import multiprocessing

    return ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"))


def _components(graded: list[dict]) -> dict[str, float]:
    """Each component 0-100, a mean over the pages that have it."""
    mean = lambda xs: 100 * sum(xs) / len(xs)  # noqa: E731
    out = {}
    if t := [1 - e / n for g in graded if (x := g.get("text")) for e, n in [x]]:
        out["text"] = mean(t)
    if t := [sum(x) / len(x) for g in graded if (x := g.get("table"))]:
        out["table_teds"] = mean(t)
    if t := [1 - e / n for g in graded if (x := g.get("formula")) for e, n in [x]]:
        out["formula"] = mean(t)
    return out


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    pages = _pages(spec.revision)
    graded = list(_pool().map(_grade, [(pages[it.id], replies.get(it.id) or "") for it in its]))
    units = {}
    for it, g in zip(its, graded):
        u = {}
        if x := g.get("text"):
            u["text"] = [1 - x[0] / x[1], 1]
        if x := g.get("table"):
            u["table"] = [sum(x) / len(x), 1]
        if x := g.get("formula"):
            u["formula"] = [1 - x[0] / x[1], 1]
        units[it.id] = u
    overall = lambda c: sum(c.values()) / len(c) if c else 0.0  # noqa: E731
    comps = _components(graded)
    metrics = dict(comps)
    if order := [1 - g["order"] for g in graded if "order" in g]:
        metrics["reading_order"] = 100 * sum(order) / len(order)
    by: dict[str, list[dict]] = {}
    for it, g in zip(its, graded):
        by.setdefault(it.category, []).append(g)
    return {
        "score": overall(comps),
        "metrics": metrics,
        "by_category": {k: overall(_components(v)) for k, v in by.items()},
        "units": units,
    }
