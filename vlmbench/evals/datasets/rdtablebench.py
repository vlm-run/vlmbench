"""RD-TableBench (reducto/rd-tablebench, CC-BY-NC-ND-4.0: internal evaluation only), graded as Reducto's
`grading.py` (github.com/reductoai/rd-tablebench @ 1cae108): each table as a 2D string array (`convert.py`, spans
repeated), rows and cells aligned by Needleman-Wunsch with Levenshtein partial credit; the score is the mean
per-table similarity. As their `parsing.py`, the largest table in a reply is graded; a markdown pipe table counts,
and a reply with no table scores 0. Items are the zip's `_images` (table crops), categorised by the gold's spans."""

import multiprocessing
import re
import signal
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item

REPO = "reducto/rd-tablebench"
_SPAN = re.compile(r"""(row|col)span\s*=\s*["']?\s*([2-9]|\d{2,})""", re.I)
S_ROW_MATCH, G_ROW, S_CELL_MATCH, P_CELL_MISMATCH, G_COL = 5, -3, 1, -1, -1


@cache
def _root(revision: str) -> Path:
    """`_images/` and `groundtruth/`, unzipped once beside the HF cache."""
    from huggingface_hub import constants, hf_hub_download

    out = Path(constants.HF_HOME) / "vlmbench" / "rdtablebench" / revision[:12]
    if not (out / "done").exists():
        zf = hf_hub_download(REPO, "rd-tablebench.zip", repo_type="dataset", revision=revision)
        with zipfile.ZipFile(zf) as z:
            for n in z.namelist():
                if n.split("/")[1:2] in (["_images"], ["groundtruth"]) and not n.endswith("/"):
                    (out / n.split("/", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
                    (out / n.split("/", 1)[1]).write_bytes(z.read(n))
        (out / "done").touch()
    return out


@cache
def _gold(revision: str) -> dict[str, str]:
    return {f.stem: f.read_text() for f in sorted((_root(revision) / "groundtruth").glob("*.html"))}


def items(spec: DatasetSpec) -> list[Item]:
    root = _root(spec.revision)
    return [
        Item(i, "merged" if _SPAN.search(h) else "flat", str(next((root / "_images").glob(f"{i}.*"))))
        for i, h in _gold(spec.revision).items()
    ]


def tables(text: str) -> list[str]:
    """Every outermost `<table>` in a reply (unclosed ones closed); pipe tables via `markdown` when none."""
    import markdown
    from lxml import etree, html

    if not text or not text.strip():
        return []
    if not re.search(r"<table\b", text, re.I):
        text = markdown.markdown(text, extensions=["tables"])
        if "<table" not in text:
            return []
    doc = html.fromstring(f"<div>{text}</div>")
    return [etree.tostring(t, encoding="unicode", with_tail=False) for t in doc.xpath("//table[not(ancestor::table)]")]


# --- convert.py (verbatim logic) ---
def html_to_numpy(html_string: str):
    import numpy as np
    from lxml import etree

    dom_tree = etree.HTML(html_string, parser=etree.HTMLParser())
    table_rows: list[list[str]] = []
    span_info: dict[int, tuple[str, int]] = {}

    def carry(row: list[str], col: int) -> int:
        while span_info.get(col, (None, 0))[1] > 0:
            row.append(span_info[col][0])
            span_info[col] = (span_info[col][0], span_info[col][1] - 1)
            if span_info[col][1] == 0:
                del span_info[col]
            col += 1
        return col

    for table_row in dom_tree.xpath("//tr"):
        current_row: list[str] = []
        column_index = carry(current_row, 0)
        for table_cell in table_row.xpath("td|th"):
            column_index = carry(current_row, column_index)
            row_span = int(table_cell.get("rowspan", "1"))
            col_span = int(table_cell.get("colspan", "1"))
            cell_text = "".join(table_cell.itertext()).strip()
            if row_span > 1:
                for i in range(col_span):
                    span_info[column_index + i] = (cell_text, row_span - 1)
            current_row += [cell_text] * col_span
            column_index += col_span
        carry(current_row, column_index)
        table_rows.append(current_row)

    max_columns = max(map(len, table_rows)) if table_rows else 0
    for row in table_rows:
        row.extend([""] * (max_columns - len(row)))
    return np.array(table_rows)


# --- grading.py (verbatim logic; the column pass keeps only the score, which is all `table_similarity` reads) ---
def cell_match_score(cell1: str | None, cell2: str | None) -> float:
    from Levenshtein import distance

    if cell1 is None or cell2 is None:
        return P_CELL_MISMATCH
    if cell1 == cell2:
        return S_CELL_MATCH
    max_len = max(len(cell1), len(cell2))
    similarity = 1.0 - (distance(cell1, cell2) / max_len if max_len else 0.0)
    return P_CELL_MISMATCH + similarity * (S_CELL_MATCH - P_CELL_MISMATCH)


def _nw(score_of, m: int, n: int, gap: float):
    """Needleman-Wunsch with free end gaps: (float32 score matrix, traceback, best end cell)."""
    import numpy as np

    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)
    traceback = np.full((m + 1, n + 1), None)
    traceback[1:, 0], traceback[0, 1:] = "up", "left"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1, j - 1] + score_of(i - 1, j - 1)
            delete = score_matrix[i - 1, j] + gap
            insert = score_matrix[i, j - 1] + gap
            max_score = max(match, delete, insert)
            score_matrix[i, j] = max_score
            traceback[i, j] = "diag" if max_score == match else "up" if max_score == delete else "left"
    max_i, max_j, max_score = m, n, score_matrix[m, n]
    last_row, last_col = score_matrix[:, n], score_matrix[m, :]
    if last_row.max() > max_score:
        max_i, max_j, max_score = last_row.argmax(), n, last_row[last_row.argmax()]
    if last_col.max() > max_score:
        max_i, max_j, max_score = m, last_col.argmax(), last_col[last_col.argmax()]
    return traceback, max_i, max_j, max_score


def table_similarity(ground_truth, prediction) -> float:
    import numpy as np

    norm = np.vectorize(lambda c: "".join(c.replace("\n", " ").replace("-", "").split()).replace(" ", ""))
    gt_rows = [list(r) for r in norm(ground_truth)]
    pred_rows = [list(r) for r in norm(prediction)]
    m, n = len(gt_rows), len(pred_rows)
    row_match_scores = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            g, p = gt_rows[i], pred_rows[j]
            col_score = _nw(lambda a, b: cell_match_score(g[a], p[b]), len(g), len(p), G_COL)[3]
            row_match_scores[i, j] = col_score + S_ROW_MATCH
    traceback, i, j, max_score = _nw(lambda a, b: row_match_scores[a, b], m, n, G_ROW)
    aligned = 0  # rows in the traced alignment, gaps included
    while i > 0 or j > 0:
        d = traceback[i, j]
        if i > 0 and j > 0 and d == "diag":
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or d == "up"):
            i -= 1
        elif j > 0 and (i == 0 or d == "left"):
            j -= 1
        else:
            break
        aligned += 1
    if aligned == 0:
        return 0.0
    return max(0.0, min(max_score / (aligned * (S_ROW_MATCH + len(gt_rows[0]) * S_CELL_MATCH)), 1.0))


def similarity(gold: str, reply: str) -> float:
    """The largest table in `reply` against the gold HTML, 0 to 1."""
    ts = tables(reply)
    if not ts:
        return 0.0
    pred = html_to_numpy(max(ts, key=len))
    if pred.size == 0:
        return 0.0
    return float(table_similarity(html_to_numpy(gold), pred))


def _job(job: tuple[str, str], limit_s: int = 120) -> float:
    """A reply the grader cannot read (e.g. `colspan="2px"`), or that runs it past `limit_s` (a runaway, repeated
    table), scores 0, as an empty one."""

    def expire(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(limit_s)
    try:
        return similarity(*job)
    except Exception:
        return 0.0
    finally:
        signal.alarm(0)


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    gold = _gold(spec.revision)
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:  # olmOCR threads: no fork
        sims = list(pool.map(_job, [(gold[it.id], replies.get(it.id) or "") for it in its]))
    by: dict[str, list[float]] = {}
    for it, s in zip(its, sims):
        by.setdefault(it.category, []).append(100 * s)
    mean = 100 * sum(sims) / len(sims) if sims else 0.0
    return {
        "score": mean,
        "metrics": {"similarity": mean},
        "by_category": {k: sum(v) / len(v) for k, v in by.items()},
        "units": {it.id: {"similarity": [s, 1]} for it, s in zip(its, sims)},
    }
