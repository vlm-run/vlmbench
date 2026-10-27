"""FinTabNet bench (datalab-to/fintabnet_bench): 999 table crops from financial reports, graded by TEDS as marker's
table benchmark (github.com/datalab-to/marker @ v1.10.1, `benchmarks/table/scoring.py`, itself from IBM's EDD):
tree edit distance over `<table>/<tr>/<td>` with span-aware renames and character-level cell distance; the score is
the mean per-table TEDS, `teds_struct` the same tree without cell text. As marker's `fix_table_html`, a reply's
`<th>` becomes `<td>` and `<br>` is dropped; `<thead>` / `<tbody>` / `<tfoot>` are unwrapped too (the gold has none,
and a markdown reply always has them). The gold is prettified, so whitespace is collapsed and trimmed at tag
boundaries on both sides. The largest table in a reply is graded (a pipe table counts); none scores 0."""

import multiprocessing
import re
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.datasets.rdtablebench import tables

REPO = "datalab-to/fintabnet_bench"
_SPAN = re.compile(r"""(row|col)span\s*=\s*["']?\s*([2-9]|\d{2,})""", re.I)


@cache
def _rows(revision: str) -> dict[str, dict]:
    """id (row index) -> gold HTML and image path, images written once beside the HF cache."""
    import pyarrow.parquet as pq
    from huggingface_hub import constants, hf_hub_download

    f = hf_hub_download(REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=revision)
    folder = Path(constants.HF_HOME) / "vlmbench" / "fintabnet" / revision[:12]
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, r in enumerate(pq.read_table(f, columns=["html", "image"]).to_pylist()):
        png = folder / f"{i:04d}.png"
        if not png.exists():
            png.write_bytes(r["image"]["bytes"])
        out[f"{i:04d}"] = {"html": r["html"], "image": png}
    return out


def items(spec: DatasetSpec) -> list[Item]:
    return [
        Item(i, "merged" if _SPAN.search(r["html"]) else "flat", str(r["image"]))
        for i, r in _rows(spec.revision).items()
    ]


def _compact(table: str) -> str:
    return re.sub(r"\s*(<[^>]*>)\s*", r"\1", re.sub(r"\s+", " ", table)).strip()


def _fix(table: str) -> str:
    """marker's `fix_table_html`, plus `<thead>` / `<tfoot>` unwrapped."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(table, "html.parser")
    for t in soup.find_all(["thead", "tbody", "tfoot"]):
        t.unwrap()
    for th in soup.find_all("th"):
        th.name = "td"
    for br in soup.find_all("br"):
        br.replace_with(soup.new_string(""))
    return str(soup).replace("\n", " ")


# --- marker benchmarks/table/scoring.py (TEDS) ---
def _tree(node, cells: bool):
    from apted.helpers import Tree

    class TableTree(Tree):
        def __init__(self, tag, colspan=None, rowspan=None, content=None):
            self.tag, self.colspan, self.rowspan, self.content = tag, colspan, rowspan, content
            super().__init__(tag)

    def tokens(n, out: list) -> list:
        out.append(f"<{n.tag}>")
        if n.text is not None:
            out += list(n.text)
        for c in list(n):
            tokens(c, out)
        if n.tag != "unk":
            out.append(f"</{n.tag}>")
        if n.tag != "td" and n.tail is not None:
            out += list(n.tail)
        return out

    def convert(n):
        if n.tag == "td":
            cell = tokens(n, [])[1:-1] if cells else []
            return TableTree(n.tag, int(n.attrib.get("colspan", "1")), int(n.attrib.get("rowspan", "1")), cell)
        t = TableTree(n.tag)
        t.children = [convert(c) for c in list(n)]
        return t

    return convert(node)


def teds(pred: str, true: str, structure_only: bool = False) -> float:
    from apted import APTED, Config
    from Levenshtein import distance
    from lxml import html

    class CustomConfig(Config):
        def rename(self, node1, node2):
            if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
                return 1.0
            if node1.tag == "td" and (node1.content or node2.content):
                return float(distance(node1.content, node2.content)) / max(len(node1.content), len(node2.content))
            return 0.0

    pred, true = (html.fromstring(f"<html><body>{x}</body></html>") for x in (pred, true))
    if not (pred.xpath("body/table") and true.xpath("body/table")):
        return 0.0
    pred, true = pred.xpath("body/table")[0], true.xpath("body/table")[0]
    n_nodes = max(len(pred.xpath(".//*")), len(true.xpath(".//*")))
    d = APTED(_tree(pred, not structure_only), _tree(true, not structure_only), CustomConfig()).compute_edit_distance()
    return 1.0 - float(d) / n_nodes


def _job(job: tuple[str, str], limit_s: int = 120) -> tuple[float, float]:
    """(TEDS, structure-only TEDS) of the reply's largest table; one the grader cannot read (e.g. `colspan="2px"`),
    or that runs it past `limit_s` (a runaway, repeated table), scores 0, as an empty one."""

    def expire(*_):
        raise TimeoutError

    gold, reply = job
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(limit_s)
    try:
        ts = tables(reply)
        if not ts:
            return 0.0, 0.0
        pred, true = _compact(_fix(max(ts, key=len))), _compact(gold)
        return teds(pred, true), teds(pred, true, structure_only=True)
    except Exception:
        return 0.0, 0.0
    finally:
        signal.alarm(0)


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    rows = _rows(spec.revision)
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:  # olmOCR threads: no fork
        res = list(pool.map(_job, [(rows[it.id]["html"], replies.get(it.id) or "") for it in its]))
    by: dict[str, list[float]] = {}
    for it, (t, _) in zip(its, res):
        by.setdefault(it.category, []).append(100 * t)
    mean = lambda xs: 100 * sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "score": mean([t for t, _ in res]),
        "metrics": {"teds": mean([t for t, _ in res]), "teds_struct": mean([s for _, s in res])},
        "by_category": {k: sum(v) / len(v) for k, v in by.items()},
        "units": {it.id: {"teds": [t, 1]} for it, (t, _) in zip(its, res)},
    }
