"""Leaderboard as self-contained HTML, markdown or CSV.

Driven only by `result.json`, so catalog-defined columns render unmodified.
"""

import csv
import html
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BANDS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
"""Fixed status palette. CVD-separable; the visible number in each cell is the
relief for `warning`, which sits below 3:1 on a light surface."""

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
_FIXED_COLUMNS = ("content", "noise", "text", "order", "tables", "layout")


@dataclass
class Row:
    """One leaderboard row, flattened from a result file."""

    provider: str
    model: str
    prompt: str
    overall: float | None
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    usd_per_1k: float | None = None
    eval_name: str = ""
    partial: bool = False
    pages: int = 0
    of_pages: int = 0
    errors: int = 0
    good: float = 85.0
    fair: float = 70.0


def band(score: float | None, good: float, fair: float) -> str:
    if score is None:
        return "none"
    if score >= good:
        return "good"
    return "warning" if score >= fair else "critical"


def band_label(name: str) -> str:
    return {"good": "strong", "warning": "adequate", "critical": "weak", "none": "not measured"}[name]


def load_documents(paths: list[Path]) -> list[dict]:
    """Read result files, keeping only eval documents and dropping duplicate runs."""
    documents, seen = [], set()
    for path in paths:
        payload = json.loads(path.read_text())
        run_id = payload.get("run_id")
        if payload.get("kind") == "eval" and run_id not in seen:
            seen.add(run_id)
            documents.append(payload)
    return documents


def load_rows(paths: list[Path]) -> list[Row]:
    """Flatten one or more `result.json` files into leaderboard rows."""
    rows: list[Row] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("kind") != "eval":
            continue
        limit = payload.get("limit") or {}
        thresholds = (payload.get("eval") or {}).get("thresholds") or {}
        for entry in payload.get("rows", []):
            scores = entry.get("scores") or {}
            cost = entry.get("cost") or {}
            per_1k = cost.get("usd_per_1k_pages")
            rows.append(
                Row(
                    provider=entry.get("provider", ""),
                    model=entry.get("model", ""),
                    prompt=(entry.get("prompt") or {}).get("name") or "—",
                    overall=scores.get("overall"),
                    columns=scores.get("columns") or {},
                    usd_per_1k=round(per_1k, 3) if per_1k else None,
                    eval_name=(payload.get("eval") or {}).get("name", ""),
                    partial=bool(limit.get("partial")),
                    pages=int(limit.get("pages") or 0),
                    of_pages=int(limit.get("of") or 0),
                    errors=int(entry.get("errors") or 0),
                    good=float(thresholds.get("good", 85.0)),
                    fair=float(thresholds.get("fair", 70.0)),
                )
            )
    return sorted(rows, key=lambda row: (row.overall is None, -(row.overall or 0)))


def column_order(rows: list[Row]) -> list[str]:
    """Known ParseBench columns first, then anything a catalog eval contributed."""
    present = {name for row in rows for name in row.columns}
    fixed = [name for name in _FIXED_COLUMNS if name in present]
    return fixed + sorted(present - set(fixed))


def _cell_value(row: Row, name: str) -> float | None:
    entry = row.columns.get(name)
    return entry.get("score") if entry else None


def _cell_detail(row: Row, name: str) -> str:
    entry = row.columns.get(name)
    if not entry:
        return "not reported"
    parts = []
    if entry.get("tests"):
        parts.append(f"{entry.get('passed', 0)}/{entry['tests']} tests passed")
    if entry.get("samples"):
        parts.append(f"mean of {entry['samples']} samples")
    if entry.get("unsupported"):
        parts.append(f"{entry['unsupported']} unsupported")
    return " · ".join(parts) or "no data"


def informative(rows: list[Row]) -> tuple[bool, bool]:
    """Whether the provider and prompt columns tell the reader anything."""
    return len({row.provider for row in rows}) > 1, any(row.prompt not in ("—", "") for row in rows)


def to_markdown(rows: list[Row]) -> str:
    names = column_order(rows)
    show_provider, show_prompt = informative(rows)
    header = (
        (["Provider"] if show_provider else [])
        + ["Model"]
        + (["Prompt"] if show_prompt else [])
        + ["Overall", *(n.title() for n in names), "$/1K pages"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        values = [f"{v:.1f}" if (v := _cell_value(row, name)) is not None else "—" for name in names]
        cells = (
            ([row.provider] if show_provider else [])
            + [f"`{row.model}`"]
            + ([row.prompt] if show_prompt else [])
            + [
                f"**{row.overall:.1f}**" if row.overall is not None else "—",
                *values,
                f"${row.usd_per_1k:.2f}" if row.usd_per_1k is not None else "—",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    if any(row.partial for row in rows):
        sample = next(row for row in rows if row.partial)
        lines.append("")
        lines.append(f"> ⚠ PARTIAL — {sample.pages} of {sample.of_pages} pages. Not comparable to published numbers.")
    return "\n".join(lines)


def to_csv(rows: list[Row]) -> str:
    names = column_order(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Provider", "Model", "Prompt", "Overall", *(n.title() for n in names), "Cost_Usd_Per_1k_Pages"])
    for row in rows:
        writer.writerow(
            [
                row.provider,
                row.model,
                row.prompt,
                "" if row.overall is None else f"{row.overall:.2f}",
                *["" if (v := _cell_value(row, n)) is None else f"{v:.2f}" for n in names],
                "" if row.usd_per_1k is None else f"{row.usd_per_1k:.3f}",
            ]
        )
    return buffer.getvalue()


_CSS = r"""
:root {
  color-scheme: light dark;
  --surface: %(light)s; --panel: #ffffff; --ink: #1a1a19; --ink-dim: #5f5f5c;
  --rule: #e4e4e0; --track: #efefeb; --accent: #4682B4;
  --good: %(good)s; --warning: %(warning)s; --critical: %(critical)s;
}
:root[data-theme="dark"], :root:not([data-theme="light"]) {
  @media (prefers-color-scheme: dark) {
    --surface: %(dark)s; --panel: #232322; --ink: #f2f2ef; --ink-dim: #a1a19c;
    --rule: #343432; --track: #2b2b29;
  }
}
:root[data-theme="dark"] {
  --surface: %(dark)s; --panel: #232322; --ink: #f2f2ef; --ink-dim: #a1a19c;
  --rule: #343432; --track: #2b2b29;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px; background: var(--surface); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--ink-dim); margin: 0 0 20px; font-size: 13px; }
.banner {
  border: 1px solid var(--warning); border-left-width: 3px; border-radius: 6px;
  padding: 10px 14px; margin-bottom: 20px; font-size: 13px;
}
.bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
.legend { display: flex; gap: 14px; align-items: center; color: var(--ink-dim); font-size: 12px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
button {
  font: inherit; font-size: 12px; padding: 5px 11px; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--panel); color: var(--ink); cursor: pointer;
}
button:hover { border-color: var(--accent); }
button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
#drop {
  border: 1.5px dashed var(--rule); border-radius: 8px; padding: 18px; text-align: center;
  color: var(--ink-dim); font-size: 13px; margin-bottom: 18px; cursor: pointer;
}
#drop.over { border-color: var(--accent); color: var(--accent); }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { width: 100%%; min-width: 760px; border-collapse: collapse; background: var(--panel);
        border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--rule); }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
     color: var(--ink-dim); font-weight: 600; background: var(--panel);
     position: sticky; top: 0; cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { color: var(--ink); }
th[aria-sort] { color: var(--accent); }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; }
td.model { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
td.prompt { color: var(--ink-dim); }
.score { display: flex; flex-direction: column; gap: 3px; align-items: flex-end; min-width: 68px; }
.score .v { font-variant-numeric: tabular-nums; }
.score .v.lead { font-weight: 650; }
.track { display: block; width: 100%%; height: 5px; border-radius: 2px;
         background: var(--track); overflow: hidden; }
.fill { display: block; height: 100%%; border-radius: 2px; }
.fill.good { background: var(--good); }
.fill.warning { background: var(--warning); }
.fill.critical { background: var(--critical); }
.muted { color: var(--ink-dim); }
tr.older td { background: color-mix(in srgb, var(--ink) 4%%, transparent); }
tr.older td.model { padding-left: 24px; }
.older-toggle {
  font-size: 10.5px; padding: 1px 6px; margin-left: 8px; border-radius: 10px;
  color: var(--ink-dim); vertical-align: 1px;
}
.empty { padding: 36px; text-align: center; color: var(--ink-dim); }
#tip {
  position: fixed; z-index: 9; pointer-events: none; opacity: 0; transition: opacity .1s;
  background: var(--panel); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px; padding: 6px 9px; font-size: 12px; box-shadow: 0 4px 14px rgb(0 0 0 / 0.12);
  max-width: 300px;
}
@media (max-width: 720px) { body { padding: 20px 16px; } }
"""

_JS = r"""
const FIXED = %(fixed)s;
const BAND_LABEL = {good: 'strong', warning: 'adequate', critical: 'weak'};
const docs = [];
const seen = new Set();
let sortKey = 'overall', sortDesc = true, suite = 'all';

const band = (score, good, fair) =>
  score === null ? null : score >= good ? 'good' : score >= fair ? 'warning' : 'critical';

function addDocuments(list) {
  let added = 0;
  for (const doc of list) {
    if (doc && doc.kind === 'eval' && !seen.has(doc.run_id)) {
      seen.add(doc.run_id);
      docs.push(doc);
      added++;
    }
  }
  return added;
}

function rows() {
  const out = [];
  for (const doc of docs) {
    const limit = doc.limit || {};
    const meta = doc.eval || {};
    const th = meta.thresholds || {};
    if (suite !== 'all' && meta.name !== suite) continue;
    for (const entry of doc.rows || []) {
      const scores = entry.scores || {};
      const cost = entry.cost || {};
      out.push({
        provider: entry.provider || '',
        model: entry.model || '',
        prompt: (entry.prompt || {}).name || '\u2014',
        overall: scores.overall ?? null,
        columns: scores.columns || {},
        usdPer1k: cost.usd_per_1k_pages ?? null,
        suite: meta.name || '',
        revision: meta.revision || '',
        timestamp: doc.timestamp || '',
        date: (doc.timestamp || '').slice(0, 10) || meta.revision || '',
        partial: !!limit.partial,
        pages: limit.pages || 0,
        ofPages: limit.of || 0,
        errors: entry.errors || 0,
        good: th.good ?? 85,
        fair: th.fair ?? 70,
      });
    }
  }
  return out;
}

const columnOrder = list => {
  const present = new Set();
  list.forEach(row => Object.keys(row.columns).forEach(name => present.add(name)));
  const fixed = FIXED.filter(name => present.has(name));
  return fixed.concat([...present].filter(name => !fixed.includes(name)).sort());
};

const cellValue = (row, name) => (row.columns[name] || {}).score ?? null;

function cellDetail(row, name) {
  const entry = row.columns[name];
  if (!entry) return 'not reported';
  const parts = [];
  if (entry.tests) parts.push(`${entry.passed || 0}/${entry.tests} tests passed`);
  if (entry.samples) parts.push(`mean of ${entry.samples} samples`);
  if (entry.unsupported) parts.push(`${entry.unsupported} unsupported`);
  return parts.join(' \u00b7 ') || 'no data';
}

function scoreCell(row, score, detail, lead) {
  if (score === null || score === undefined)
    return `<td class="num"><span class="muted" data-tip="${esc(detail)}">\u2014</span></td>`;
  const name = band(score, row.good, row.fair);
  const tip = `${score.toFixed(1)} \u2014 ${BAND_LABEL[name]} \u00b7 ${detail}`;
  const width = Math.max(0, Math.min(100, score)).toFixed(1);
  return `<td class="num" data-tip="${esc(tip)}"><div class="score">` +
    `<span class="v${lead ? ' lead' : ''}">${score.toFixed(1)}</span>` +
    `<span class="track"><span class="fill ${name}" style="width:${width}%%"></span></span></div></td>`;
}

const esc = text => String(text).replace(/[&<>"]/g, ch =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch]));

function sortRows(list, names) {
  const value = row => sortKey === 'overall' ? row.overall
    : sortKey === 'cost' ? row.usdPer1k
    : sortKey === 'model' ? row.model
    : sortKey === 'suite' ? row.suite
    : sortKey === 'date' ? row.date
    : names.includes(sortKey) ? cellValue(row, sortKey) : null;
  return list.slice().sort((a, b) => {
    const x = value(a), y = value(b);
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    const cmp = typeof x === 'string' ? x.localeCompare(y) : x - y;
    return sortDesc ? -cmp : cmp;
  });
}

function toCsv(list, names) {
  const header = ['Provider', 'Model', 'Prompt', 'Suite', 'Revision', 'Overall',
    ...names.map(n => n[0].toUpperCase() + n.slice(1)), 'Cost_Usd_Per_1k_Pages'];
  const lines = [header.join(',')];
  for (const row of list) {
    lines.push([row.provider, row.model, row.prompt, row.suite, row.revision,
      row.overall ?? '', ...names.map(n => cellValue(row, n) ?? ''),
      row.usdPer1k ?? ''].map(v => `"${String(v).replace(/"/g, '""')}"`).join(','));
  }
  return lines.join('\n');
}

function groupRuns(list) {
  const groups = new Map();
  for (const row of list) {
    const key = [row.suite, row.provider, row.model, row.prompt].join('\u0000');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  for (const rows of groups.values()) {
    rows.sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
  }
  return [...groups.entries()].map(([key, rows]) => ({key, rows}));
}

function render() {
  const all = rows();
  const suitesShown = new Set(all.map(r => r.suite)).size;
  const names = suitesShown > 1 ? [] : columnOrder(all);
  const groups = groupRuns(all);
  const latest = sortRows(groups.map(g => g.rows[0]), names);
  const order = new Map(latest.map((row, index) => [row, index]));
  groups.sort((a, b) => order.get(a.rows[0]) - order.get(b.rows[0]));
  const list = all;
  const suites = [...new Set(docs.map(d => (d.eval || {}).name).filter(Boolean))];
  const showPrompt = all.some(r => r.prompt !== '\u2014');
  const showSuite = suitesShown > 1;
  const partial = all.find(r => r.partial);

  document.getElementById('sub').textContent = docs.length
    ? `${suites.join(', ')} \u00b7 ${list.length} rows \u00b7 ${docs.length} run${docs.length > 1 ? 's' : ''}` +
      (suitesShown > 1 ? ' \u00b7 pick a suite to see its columns' : ' \u00b7 higher is better except $/1K pages')
    : 'Drop result.json files below to build a leaderboard.';

  document.getElementById('banner').innerHTML = partial
    ? `<div class="banner">\u26a0 <strong>Partial run</strong> \u2014 ${partial.pages} of ${partial.ofPages} pages. ` +
      'Useful for iterating on a prompt; not comparable to published numbers.</div>'
    : '';

  document.getElementById('suites').innerHTML = suites.length > 1
    ? ['all', ...suites].map(name =>
        `<button data-suite="${esc(name)}" aria-pressed="${suite === name}">${esc(name)}</button>`).join('')
    : '';

  if (!list.length) {
    document.getElementById('board').innerHTML =
      '<div class="empty">No results loaded yet.</div>';
    return;
  }

  const sortMark = key => sortKey === key ? ` aria-sort="${sortDesc ? 'descending' : 'ascending'}"` : '';
  const head =
    (showSuite ? `<th data-sort="suite"${sortMark('suite')}>Suite</th>` : '') +
    '<th>Provider</th>' +
    `<th data-sort="model"${sortMark('model')}>Model</th>` +
    (showPrompt ? '<th>Prompt</th>' : '') +
    `<th class="num" data-sort="overall"${sortMark('overall')}>Overall</th>` +
    names.map(n => `<th class="num" data-sort="${esc(n)}"${sortMark(n)}>${esc(n)}</th>`).join('') +
    `<th class="num" data-sort="cost"${sortMark('cost')}>$/1K pages</th>` +
    `<th data-sort="date"${sortMark('date')}>Evaluated</th>`;

  const renderRow = (row, group, older) => {
    const toggle = !older && group.rows.length > 1
      ? ` <button class="older-toggle" data-group="${esc(group.key)}">` +
        `+${group.rows.length - 1} older</button>`
      : '';
    const cells = [
      showSuite ? `<td class="prompt">${esc(row.suite)}</td>` : '',
      `<td>${esc(row.provider)}</td>`,
      `<td class="model">${esc(row.model)}${toggle}</td>`,
      showPrompt ? `<td class="prompt">${esc(row.prompt)}</td>` : '',
      scoreCell(row, row.overall,
        `${row.pages} pages${row.errors ? ` \u00b7 ${row.errors} errors` : ''} \u00b7 rev ${row.revision}`, !older),
      ...names.map(name => scoreCell(row, cellValue(row, name), cellDetail(row, name), false)),
      row.usdPer1k === null
        ? '<td class="num muted" data-tip="provider reported no cost">\u2014</td>'
        : `<td class="num muted">$${row.usdPer1k.toFixed(2)}</td>`,
      `<td class="muted">${esc(row.date)}</td>`,
    ];
    const attrs = older
      ? ` class="older" data-of="${esc(group.key)}" hidden`
      : '';
    return `<tr${attrs}>${cells.join('')}</tr>`;
  };

  const body = groups.map(group =>
    group.rows.map((row, index) => renderRow(row, group, index > 0)).join('')
  ).join('');

  document.getElementById('board').innerHTML =
    `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  window.__csv = () => toCsv(list, names);
}

function readFiles(files) {
  const pending = [...files].filter(file => file.name.endsWith('.json'));
  let done = 0, added = 0;
  if (!pending.length) return;
  pending.forEach(file => {
    const reader = new FileReader();
    reader.onload = () => {
      try { added += addDocuments([JSON.parse(reader.result)]); }
      catch (err) { console.warn(file.name, err); }
      if (++done === pending.length) render();
    };
    reader.readAsText(file);
  });
}

const drop = document.getElementById('drop');
const picker = document.getElementById('picker');
drop.addEventListener('click', () => picker.click());
picker.addEventListener('change', event => readFiles(event.target.files));
['dragenter', 'dragover'].forEach(name => drop.addEventListener(name, event => {
  event.preventDefault(); drop.classList.add('over');
}));
['dragleave', 'drop'].forEach(name => drop.addEventListener(name, event => {
  event.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', event => readFiles(event.dataTransfer.files));

document.addEventListener('click', event => {
  const header = event.target.closest('th[data-sort]');
  if (header) {
    const key = header.dataset.sort;
    sortDesc = sortKey === key ? !sortDesc : true;
    sortKey = key;
    render();
  }
  const chip = event.target.closest('button[data-suite]');
  if (chip) { suite = chip.dataset.suite; render(); }
  const older = event.target.closest('button.older-toggle');
  if (older) {
    event.stopPropagation();
    const rows = document.querySelectorAll(`tr.older[data-of="${CSS.escape(older.dataset.group)}"]`);
    const show = rows.length && rows[0].hidden;
    rows.forEach(row => { row.hidden = !show; });
    older.textContent = show ? 'hide older' : `+${rows.length} older`;
  }
});

document.getElementById('csv').addEventListener('click', () => {
  const blob = new Blob([window.__csv ? window.__csv() : ''], {type: 'text/csv'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'vlmbench-leaderboard.csv';
  link.click();
  URL.revokeObjectURL(link.href);
});

document.getElementById('clear').addEventListener('click', () => {
  docs.length = 0; seen.clear(); render();
});

const tip = document.getElementById('tip');
document.addEventListener('pointerover', event => {
  const cell = event.target.closest('[data-tip]');
  if (!cell) { tip.style.opacity = '0'; return; }
  tip.textContent = cell.dataset.tip;
  tip.style.opacity = '1';
  const box = cell.getBoundingClientRect();
  tip.style.left = Math.min(box.left, window.innerWidth - 320) + 'px';
  tip.style.top = (box.bottom + 6) + 'px';
});

addDocuments(JSON.parse(document.getElementById('data').textContent));
render();
"""


def to_html(documents: list[dict], *, title: str = "vlmbench leaderboard") -> str:
    """Self-contained page: embedded runs, plus drop-in import of more result.json files."""
    legend = "".join(
        f'<span><span class="swatch" style="background:var(--{key})"></span>{label}</span>'
        for key, label in (("good", "\u2265 85 strong"), ("warning", "\u2265 70 adequate"), ("critical", "< 70 weak"))
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS % {"light": SURFACE_LIGHT, "dark": SURFACE_DARK, **BANDS}}</style>
</head><body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub" id="sub"></p>
<div id="banner"></div>
<div id="drop">Drop <code>result.json</code> files here, or click to choose \u2014 everything stays in your browser</div>
<input type="file" id="picker" accept="application/json,.json" multiple hidden>
<div class="bar">
  <div class="legend">{legend}</div>
  <span id="suites"></span>
  <button id="csv">Download CSV</button>
  <button id="clear">Clear</button>
</div>
<div id="board"></div>
</div>
<div id="tip"></div>
<script type="application/json" id="data">{json.dumps(documents)}</script>
<script>{_JS % {"fixed": json.dumps(list(_FIXED_COLUMNS))}}</script>
</body></html>
"""
