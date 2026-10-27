"""RealDoc-Bench QA (Extend-AI/RealDoc-Bench): 1,356 questions over 581 real documents (finance, medical_healthcare,
mortgage, supply_chain), each answered as a small typed JSON object and graded field by field, as Extend's
`realdoc-bench` scorer (github.com/extend-hq/realdoc-bench@fb26a68, `evaluate/score.py` + `jsonify.py`, ported here:
it is not on PyPI). The score is its leaderboard's per-field accuracy (micro over 3,742 fields); per-question
accuracy (every field right) beside it.

Their pipeline has a parser write markdown and Gemini answer from it; here the model reads the pages and answers
itself, with the same question, typed template and extractor rules. Comparison is `deep_equal` (numbers by value,
`"$1,234"` == 1234, booleans exact, strings after quote / dash / whitespace / markdown normalisation) and, for plain
`<string>` fields only, a fuzzy fall-back (punctuation-insensitive, or rapidfuzz >= 92 when both sides have 5+
words). A null gold is matched by a missing or null answer, so an empty reply still scores the 188 blank fields (~5).
The card says 1,359 questions; the pinned bank holds 1,356 (the leaderboard's count). Every document is one page.
`units` are `[fields right, fields]`, so intervals resample the same micro accuracy.
"""

import json
import re
from datetime import datetime
from functools import cache
from pathlib import Path

from vlmbench.evals.config import DatasetSpec
from vlmbench.evals.data import Item
from vlmbench.evals.jsonx import parse

REPO = "Extend-AI/RealDoc-Bench"
DOCUMENT = True
BUCKETS = {  # realdoc_bench/evaluate/capability_buckets.yaml: bucket -> capability tags (they overlap)
    "checkboxes": {"checkbox_state", "handdrawn_check", "form_checkbox_grid"},
    "tables": set(
        "table_structure table_row_binding table_column_alignment table_parsing row_binding multi_column_grid "
        "parallel_columns merged_cells total_row repeated_records".split()
    ),
    "forms": {"field_value_pairing", "form_key_value", "form_region", "scanned_form"},
    "multi_column": {"parallel_columns", "multi_column_grid", "column_alignment", "reading_order"},
    "charts": {"chart_title", "chart_axis", "chart_legend", "chart_value", "figure_content"},
    "handwriting": {"handdrawn_check"},
    "blank_field": {"blank_field"},
    "small_text": {"small_text"},
}
RULES = (  # score.py SYSTEM_PROMPT, the extractor's rules, with "markdown" read as "document"
    "Return ONE JSON object whose keys and value types exactly match the answer template. The type hint inside "
    "`< >` is the source of truth for each field; the key name does not change typing or formatting. Rules: "
    "(1) numbers must be JSON numbers (no commas, no '$', no '%', no quotes); (2) only fields whose template says "
    "`<boolean>` may be JSON true / false; (3) fields whose template says `<one of: a | b | ...>` MUST return one of "
    "the listed tokens as a JSON string, copied verbatim including case; (4) ONLY fields whose type hint is "
    "`<date string YYYY-MM-DD>` get normalized to ISO YYYY-MM-DD; (5) all other fields, including any `<string>`, "
    "copy the page text verbatim, trimmed (do not normalize dates, numbers, capitalization or separators); "
    "(6) for any field whose value is absent, blank, illegible, or genuinely ambiguous in the document, use JSON "
    "null; treat redaction markers like `[Redacted]` as absent. Do not include keys outside the template."
)


# ── jsonify.py: response_format -> typed template (and the gold widened to match) ──────────────────────────────
_INT = {"digits", "digit", "year", "qty", "quantity", "count", "integer", "int"}
_NUM = set(
    "amount money cost price fee balance total subtotal premium payment deposit percent percentage rate decimal "
    "float number".split()
)
_DATE = {"date", "mm/dd/yyyy", "mm-dd-yyyy", "yyyy-mm-dd", "yyyy/mm/dd"}
_DATE_FORMATS = (
    "%Y-%m-%d|%Y/%m/%d|%Y.%m.%d|%m/%d/%Y|%m-%d-%Y|%m.%d.%Y|%d/%m/%Y|%d-%m-%Y|%m/%d/%y|%m-%d-%y|%B %d, %Y|%b %d, %Y|"
    "%d %B %Y|%d %b %Y|%B %d %Y|%b %d %Y"
).split("|")
_RF_PREFIX, _PAIR = re.compile(r"^\s*Return exactly[^:]*:\s*"), re.compile(r"([^=<>;|]+)=<([^>]+)>")


def _hint(inner: str) -> str:
    opts = [o.strip() for o in inner.strip().split("|") if o.strip()]
    nullable = "null" in [o.lower() for o in opts]
    opts = [o for o in opts if o.lower() != "null"]
    lo = [o.lower() for o in opts]
    if len(opts) == 1:
        t = lo[0]
        h = (
            "boolean"
            if t in {"boolean", "bool"}
            else "integer"
            if t in _INT
            else "number"
            if t in _NUM
            else "date string YYYY-MM-DD"
            if t in _DATE
            else "string"
        )
    elif opts and set(lo) == {"true", "false"} and len(lo) == 2:
        h = "boolean"
    elif opts:
        h = "one of: " + " | ".join(o for i, o in enumerate(opts) if o.lower() not in lo[:i])
    else:
        h = "string"
    return f"{h} | null" if nullable else h


def _iso(text: str) -> str | None:
    t = re.sub(r"\bSept\b", "Sep", text.strip().rstrip(".,;:"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _widen(kt: dict[str, str], gold: dict) -> None:
    """A key's type widened where the gold does not fit it (a date gold under `<number>` becomes a date)."""
    for k, v in list(gold.items()):
        ht = kt.get(k, "")
        base, null = ht.replace(" | null", "").strip(), " | null" if "null" in ht else ""
        if base in {"integer", "number"} and isinstance(v, str) and v.strip():
            if iso := _iso(v):
                kt[k], gold[k] = "date string YYYY-MM-DD" + null, iso
            else:
                kt[k] = "string" + null
        elif base.startswith("date") and isinstance(v, str) and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            kt[k] = "string" + null
        elif base.startswith("one of:") and isinstance(v, str):
            if v.lower() not in {t.strip().lower() for t in base[len("one of:") :].split("|") if t.strip()}:
                kt[k] = "string" + null


def _template(rf: str, gold: dict) -> dict[str, str]:
    pairs = _PAIR.findall(_RF_PREFIX.sub("", rf or ""))
    if not pairs:
        return {k: "list of strings" if isinstance(v, list) else "string" for k, v in gold.items()} or {
            "answer": "string"
        }
    kt = {k.strip(): _hint(t) for k, t in pairs}
    _widen(kt, gold)
    return kt


def _render(kt: dict[str, str]) -> str:
    return "\n".join(
        ["{", ",\n".join(f"  {json.dumps(k)}: <{t}>" for k, t in kt.items() if not k.startswith("__")), "}"]
    )


# ── score.py: deep_equal / fuzzy_equal ─────────────────────────────────────────────────────────────────────────
_SEP, _WS = re.compile(r"\s*([/\-,:])\s*"), re.compile(r"\s+")
_DASH, _TAG, _STYLE = re.compile(r"[‐‑‒–—―−]"), re.compile(r"<br\s*/?>|</?[a-z][^>]*>", re.I), re.compile(r"\*\*|__|`")
_TYPE = re.compile(r'"([^"]+)":\s*<([^>]+)>')
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "′": "'", "“": '"', "”": '"', "„": '"', "″": '"'})


def _norm(s: str) -> str:
    s = _STYLE.sub("", _TAG.sub(" ", _DASH.sub("-", s.translate(_QUOTES))))
    return _WS.sub(" ", _SEP.sub(r"\1", s)).strip()


def deep_equal(a, b) -> bool:
    if isinstance(a, list) and len(a) == 1 and isinstance(b, dict) and isinstance(a[0], dict):
        return deep_equal(a[0], b)
    if isinstance(b, list) and len(b) == 1 and isinstance(a, dict) and isinstance(b[0], dict):
        return deep_equal(a, b[0])
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        na, nb = _norm(a), _norm(b)
        return na == nb or _WS.sub("", na) == _WS.sub("", nb)
    if isinstance(a, (int, float)) and isinstance(b, str):
        num, s = a, b
    elif isinstance(b, (int, float)) and isinstance(a, str):
        num, s = b, a
    else:
        return a == b
    try:
        return float(re.sub(r"[,$%\s]", "", s)) == float(num)
    except ValueError:
        return False


def fuzzy_equal(a, b) -> bool:
    if deep_equal(a, b):
        return True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    ca, cb = (re.sub(r"[^0-9a-z]+", " ", x.lower()).strip() for x in (a, b))
    if ca and ca == cb:
        return True
    if min(len(ca.split()), len(cb.split())) < 5:
        return False
    from rapidfuzz import fuzz

    return bool(ca) and fuzz.ratio(ca, cb) >= 92


def score_fields(answer, gold: dict, kt: dict[str, str]) -> dict[str, bool]:
    """score_typed: fuzzy on plain `<string>` fields, deep_equal on the rest."""
    strs = {k for k, t in _TYPE.findall(_render(kt)) if t.replace("| null", "").strip() == "string"}
    ans = answer if isinstance(answer, dict) else {}
    return {k: (fuzzy_equal if k in strs else deep_equal)(ans.get(k), g) for k, g in gold.items()}


# ── the module interface ───────────────────────────────────────────────────────────────────────────────────────
@cache
def _bank(revision: str) -> dict[str, dict]:
    """question id -> bank row with its typed template, the PDFs from the pinned snapshot."""
    from huggingface_hub import snapshot_download

    snap = Path(
        snapshot_download(
            REPO, repo_type="dataset", revision=revision, allow_patterns=["qa_bank.json", "docs/**"], max_workers=8
        )
    )
    out = {}
    for row in json.loads((snap / "qa_bank.json").read_text())["items"]:
        row["types"] = _template(row.get("response_format", ""), row["gold_dict"])
        row["pdf"] = snap / "docs" / f"{row['source_file']}.pdf"
        out[row["question_id"]] = row
    return out


def items(spec: DatasetSpec) -> list[Item]:
    return [Item(q, r["domain"], str(r["pdf"]), doc=True) for q, r in _bank(spec.revision).items()]


def instruction(spec: DatasetSpec, item: Item) -> str:
    r = _bank(spec.revision)[item.id]
    return (
        f"Answer this question about the document.\n\nQuestion:\n{r['question']}\n\n"
        f"Answer template (return JSON with exactly these keys and value types):\n{_render(r['types'])}\n\n{RULES}"
    )


def score(spec: DatasetSpec, its: list[Item], replies: dict[str, str]) -> dict:
    bank = _bank(spec.revision)
    tally: dict[str, list[int]] = {}  # group -> [fields right, fields, questions right, questions]
    units = {}

    def add(g: str, ok: int, n: int) -> None:
        t = tally.setdefault(g, [0, 0, 0, 0])
        t[0], t[1], t[2], t[3] = t[0] + ok, t[1] + n, t[2] + (ok == n), t[3] + 1

    for it in its:
        r = bank[it.id]
        fm = score_fields(parse(replies.get(it.id)), r["gold_dict"], r["types"])
        ok, n = sum(fm.values()), len(fm)
        units[it.id] = {"field": [ok, n]}
        caps = set(r["capabilities"])
        for g in ["all", f"domain:{r['domain']}", *(f"bucket:{b}" for b, tags in BUCKETS.items() if caps & tags)]:
            add(g, ok, n)
    pct = lambda t, i=0: 100 * t[i] / t[i + 1] if t[i + 1] else 0.0  # noqa: E731
    total = tally.get("all", [0, 0, 0, 0])
    return {
        "score": pct(total),
        "metrics": {"field_accuracy": pct(total), "question_accuracy": pct(total, 2)}
        | {f"field_accuracy/{g[7:]}": pct(t) for g, t in tally.items() if g.startswith("bucket:")},
        "by_category": {g[7:]: pct(t) for g, t in tally.items() if g.startswith("domain:")},
        "units": units,
    }
