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
                f"{row.usd_per_1k:.2f}" if row.usd_per_1k is not None else "—",
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


_CSS = """
:root {
  color-scheme: light dark;
  --surface: %(light)s; --panel: #ffffff; --ink: #1a1a19; --ink-dim: #5f5f5c;
  --rule: #e4e4e0; --track: #efefeb;
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
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { width: 100%%; min-width: 760px; border-collapse: collapse; background: var(--panel);
        border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--rule); }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
     color: var(--ink-dim); font-weight: 600; background: var(--panel);
     position: sticky; top: 0; }
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
#tip {
  position: fixed; z-index: 9; pointer-events: none; opacity: 0; transition: opacity .1s;
  background: var(--panel); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px; padding: 6px 9px; font-size: 12px; box-shadow: 0 4px 14px rgb(0 0 0 / 0.12);
  max-width: 280px;
}
@media (max-width: 720px) { body { padding: 20px 16px; } .sub { font-size: 12px; } }
"""

_JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(cell => {
  cell.addEventListener('pointerenter', event => {
    tip.textContent = cell.dataset.tip;
    tip.style.opacity = '1';
    const box = cell.getBoundingClientRect();
    tip.style.left = Math.min(box.left, window.innerWidth - 300) + 'px';
    tip.style.top = (box.bottom + 6) + 'px';
  });
  cell.addEventListener('pointerleave', () => { tip.style.opacity = '0'; });
});
document.getElementById('csv').addEventListener('click', () => {
  const blob = new Blob([document.getElementById('csvdata').textContent], {type: 'text/csv'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'vlmbench-leaderboard.csv';
  link.click();
  URL.revokeObjectURL(link.href);
});
"""


def _score_cell(row: Row, score: float | None, detail: str, *, lead: bool = False) -> str:
    if score is None:
        return f'<td class="num"><span class="muted" data-tip="{html.escape(detail)}">—</span></td>'
    name = band(score, row.good, row.fair)
    tip = f"{score:.1f} — {band_label(name)} · {detail}"
    classes = "v lead" if lead else "v"
    return (
        f'<td class="num" data-tip="{html.escape(tip)}"><div class="score">'
        f'<span class="{classes}">{score:.1f}</span>'
        f'<span class="track"><span class="fill {name}" style="width:{max(0.0, min(100.0, score)):.1f}%"></span></span>'
        f"</div></td>"
    )


def to_html(rows: list[Row], *, title: str = "vlmbench leaderboard") -> str:
    """Self-contained page: no CDN, no fonts, light and dark both selected."""
    names = column_order(rows)
    suites = sorted({row.eval_name for row in rows if row.eval_name})
    partial = next((row for row in rows if row.partial), None)
    show_provider, show_prompt = informative(rows)

    head = "".join(f'<th class="num">{html.escape(name.title())}</th>' for name in names)
    body = []
    for row in rows:
        cells = [
            *([f"<td>{html.escape(row.provider)}</td>"] if show_provider else []),
            f'<td class="model">{html.escape(row.model)}</td>',
            *([f'<td class="prompt">{html.escape(row.prompt)}</td>'] if show_prompt else []),
            _score_cell(
                row, row.overall, f"{row.pages} pages" + (f" · {row.errors} errors" if row.errors else ""), lead=True
            ),
        ]
        cells += [_score_cell(row, _cell_value(row, name), _cell_detail(row, name)) for name in names]
        cost = (
            f'<td class="num muted">{row.usd_per_1k:.2f}</td>'
            if row.usd_per_1k is not None
            else '<td class="num muted" data-tip="provider reported no cost">—</td>'
        )
        body.append("<tr>" + "".join(cells) + cost + "</tr>")

    banner = (
        f'<div class="banner">⚠ <strong>Partial run</strong> — {partial.pages} of {partial.of_pages} pages. '
        f"Useful for iterating on a prompt; not comparable to published numbers.</div>"
        if partial
        else ""
    )
    legend = "".join(
        f'<span><span class="swatch" style="background:var(--{key})"></span>{label}</span>'
        for key, label in (("good", "≥ 85 strong"), ("warning", "≥ 70 adequate"), ("critical", "< 70 weak"))
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS % {"light": SURFACE_LIGHT, "dark": SURFACE_DARK, **BANDS}}</style>
</head><body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(", ".join(suites) or "eval")} · {len(rows)} rows · higher is better except $/1K pages</p>
{banner}
<div class="bar"><div class="legend">{legend}</div><button id="csv">Download CSV</button></div>
<div class="scroll"><table><thead><tr>
{"<th>Provider</th>" if show_provider else ""}<th>Model</th>{"<th>Prompt</th>" if show_prompt else ""}<th class="num">Overall</th>{head}<th class="num">$/1K pages</th>
</tr></thead><tbody>
{"".join(body)}
</tbody></table></div>
</div>
<div id="tip"></div>
<script type="application/csv" id="csvdata">{html.escape(to_csv(rows))}</script>
<script>{_JS}</script>
</body></html>
"""
