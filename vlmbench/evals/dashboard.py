"""`vlmbench dashboard`: one self-contained HTML page over everything collected, with two tabs: accuracy evals
(the latest score per serve and dataset, with intervals, and a radar per eval) and runs (throughput sweeps from
`vlmbench run`, plus the throughput each eval measured)."""

import functools
import http.server
import json
import zlib
from pathlib import Path

from vlmbench.evals.config import Settings, backend_name
from vlmbench.evals.datasets import DATASETS
from vlmbench.evals.results import draws, interval, latest

FIRST = ("ocr-bench",)
"""The eval the page opens on; the rest follow by name."""


def collect(home: Path) -> dict:
    evals = sorted((d for d in (home / "evals").glob("*") if d.is_dir()), key=lambda d: (d.name not in FIRST, d.name))
    return {"evals": [_eval(d) for d in evals], "runs": _runs(home)}


def _links(root: Path, kinds: dict[str, str]) -> dict[str, dict]:
    """Each dataset's Hugging Face repo and the revision its newest result pinned."""
    out = {}
    for f in sorted(root.glob("*/result.json"), reverse=True):
        for name, spec in json.loads(f.read_text())["run"].get("datasets", {}).items():
            repo = getattr(DATASETS.get(kinds.get(name, "")), "REPO", None)
            if name in kinds and name not in out and repo:
                out[name] = {"repo": repo, "revision": spec.get("revision")}
        if len(out) == len(kinds):
            break
    return out


def _eval(root: Path) -> dict:
    rows = latest(root)
    configs = {k[1]: v for k, v in rows.items() if k[0] == "cfg"}
    acc = [v for k, v in rows.items() if k[0] == "acc"]
    perf = [v for k, v in rows.items() if k[0] == "perf"]
    datasets = sorted({a["dataset"] for a in acc if "judge" not in a})
    doc = all(getattr(DATASETS.get(a["metric"]), "DOCUMENT", False) for a in acc if "judge" not in a)
    out = {"name": root.name, "datasets": datasets, "splits": {}, "unit": "docs" if doc and acc else "pages"}
    out["links"] = _links(root, {a["dataset"]: a["metric"] for a in acc if "judge" not in a})
    for split in sorted({a["split"] for a in acc}):
        serves = []
        for cid, cfg in configs.items():
            mine = [a for a in acc if a["config_id"] == cid and a["split"] == split]
            bench = {a["dataset"]: a for a in mine if "judge" not in a}
            if not bench:
                continue
            judged = [a["score"] for a in mine if "judge" in a]
            ps = [p for p in perf if p["config_id"] == cid and p["split"] == split]
            full = all(d in bench for d in datasets)
            ci = interval([(a["score"], a["units"]) for a in bench.values()]) if full else None
            serves.append(
                {
                    "serve": cfg["serve"].removeprefix(f"{root.name}-"),
                    "model": cfg["model_id"],
                    "config_id": cid,
                    "backend": backend_name(cfg["backend"]),
                    "prompt": Path(cfg["prompt_file"]).name if cfg.get("prompt_file") else cfg["request"].get("method"),
                    "mean": round(sum(a["score"] for a in bench.values()) / len(datasets), 1) if full else None,
                    "ci": list(ci) if ci else None,
                    "scores": {
                        d: {
                            "score": a["score"],
                            "ci": a.get("ci95"),
                            "metrics": a.get("metrics") or {},
                            "cats": a.get("by_category") or {},
                            "draws": draws(a["score"], a.get("units") or {}, seed=zlib.crc32(d.encode())),
                        }
                        for d, a in bench.items()
                    },
                    "judge_by": {a["dataset"]: a["score"] for a in mine if "judge" in a},
                    "perf_by": {
                        p["dataset"]: {
                            "pages_s": p["items_per_s"],
                            "usd_1k": p["usd_per_1k_items"],
                            "cost": p["cost_usd"],
                        }
                        for p in ps
                    },
                    "judge": round(sum(judged) / len(judged), 1) if judged else None,
                    "pages_s": _mean([p["items_per_s"] for p in ps]),
                    "usd_1k": _mean([p["usd_per_1k_items"] for p in ps]),
                    "cost": round(sum(p["cost_usd"] or 0 for p in ps), 4) if any(p["cost_usd"] for p in ps) else None,
                    "gpu_rate": cfg["backend"].get("gpu_hour_usd"),
                    "ttft_p50": _mean([(p.get("ttft_s") or {}).get("p50") for p in ps]),
                }
            )
        out["splits"][split] = sorted(serves, key=lambda s: -(s["mean"] or 0))
    out["perf"] = [
        p
        | {
            "serve": p["serve"].removeprefix(f"{root.name}-"),
            "gpu_rate": configs.get(p["config_id"], {}).get("backend", {}).get("gpu_hour_usd"),
        }
        for p in perf
    ]
    return out


def _runs(home: Path) -> list[dict]:
    out = []
    for f in sorted((home / "benchmarks").glob("*.json")):
        try:
            d = json.loads(f.read_text())
            r, env = d["results"], d["environment"]
        except (KeyError, json.JSONDecodeError):
            continue
        out.append(
            {
                "model": Path(d["model"]["model_id"]).name
                if "/mnt/" in d["model"]["model_id"]
                else d["model"]["model_id"],
                "backend": f"{env.get('backend')} {env.get('backend_version') or ''}".strip(),
                "gpu": env.get("gpu_name") or env.get("accelerator") or "",
                "concurrency": d["input"].get("max_concurrency"),
                "inputs": d["input"].get("total_inputs"),
                "items_s": r.get("inputs_per_sec"),
                "tok_s": r.get("tokens_per_sec"),
                "ttft_p50": (r.get("ttft_ms") or {}).get("p50"),
                "ttft_p95": (r.get("ttft_ms") or {}).get("p95"),
                "e2e_p50": (r.get("e2el_ms") or {}).get("p50"),
                "vram_mib": r.get("vram_peak_mib"),
                "when": d.get("timestamp", "")[:16].replace("T", " "),
            }
        )
    return sorted(out, key=lambda x: x["when"], reverse=True)


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def build(out: Path | None = None) -> Path:
    home = Settings().home
    out = out or home / "dashboard" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(collect(home), ensure_ascii=False).replace("</", "<\\/")
    out.write_text(PAGE.replace("/*DATA*/null", data))
    return out


def serve(path: Path, host: str, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path.parent))
    http.server.ThreadingHTTPServer((host, port), handler).serve_forever()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vlmbench dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
.viz-root{color-scheme:light;--surface-1:#fbfbfa;--hair:#e7e6e1;--ink-1:#0b0b0b;--ink-2:#52514e;--ink-3:#9a9993;--hover:#f3f2ee;
 --good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--accent:#0b0b0b;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#121211;
 --hair:#2a2a27;--ink-1:#f0efec;--ink-2:#c3c2b7;--ink-3:#77766f;--hover:#1b1b19;--accent:#f0efec;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#121211;--hair:#2a2a27;--ink-1:#f0efec;--ink-2:#c3c2b7;
 --ink-3:#77766f;--hover:#1b1b19;--accent:#f0efec;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 "Space Grotesk",ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.mono,td,.v,.tick{font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.viz-root{background:var(--surface-1);color:var(--ink-1);min-height:100vh}
main{max-width:1280px;margin:0 auto;padding:40px 40px 96px}
header{display:flex;align-items:center;gap:28px}
h1{font-size:18px;font-weight:700;letter-spacing:-.01em;margin:0}
.brand{display:inline-flex;align-items:center;gap:10px;color:inherit;text-decoration:none}.logo{height:22px;width:auto;display:block}
footer{display:flex;gap:10px;margin-top:24px;padding-top:20px;border-top:1px solid var(--hair);font:500 12px "Geist Mono",monospace;color:var(--ink-3)}
footer a{color:var(--ink-2);text-decoration:none}footer a:hover{color:var(--ink-1);text-decoration:underline}
.sub{color:var(--ink-3);font-size:12px}
nav.tabs{display:flex;gap:22px}
nav.tabs button,.seg button,select{font:500 13px "Space Grotesk",sans-serif;color:var(--ink-3);background:none;border:0;padding:4px 0;cursor:pointer}
nav.tabs button[aria-selected=true]{color:var(--ink-1);box-shadow:inset 0 -1.5px 0 var(--ink-1)}
.seg{display:inline-flex;gap:14px}.seg button[aria-pressed=true]{color:var(--ink-1)}
.chips[hidden]{display:none}
.frame.box{max-height:calc(100vh - 96px);overflow:auto}
input.q{font:500 13px "Geist Mono",monospace;color:var(--ink-1);background:none;border:0;border-bottom:1px solid var(--hair);padding:4px 0;width:220px;outline:none}
input.q:focus{border-bottom-color:var(--ink-1)}input.q::placeholder{color:var(--ink-3)}.qrow{margin:18px 0 6px}
td.none{text-align:left;color:var(--ink-3);padding:28px 10px}
.frame thead th{position:sticky;top:0;z-index:2;background:var(--surface-1);border-bottom:0;box-shadow:inset 0 -1px 0 var(--ink-1);padding-top:8px}
.frame table.dense th:nth-child(2){z-index:3;box-shadow:inset 0 -1px 0 var(--ink-1),8px 0 8px -8px var(--hair)}
.radar svg{overflow:visible}
.pop{position:fixed;z-index:20;display:flex;visibility:hidden;flex-direction:column;gap:3px;padding:6px 9px;background:#fff;border:1px solid #e7e6e1;
 box-shadow:0 2px 8px rgba(0,0,0,.08);border-radius:4px;font:500 11px "Geist Mono",monospace}
.pop.on{visibility:visible}.pop a{color:#0b0b0b;text-decoration:none}.pop a:hover{text-decoration:underline}.pop a.rev{color:#9a9993}
th[data-ds]{cursor:help}
.chips{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;margin:-28px 0 40px}
.chips button{font:500 12.5px "Geist Mono",monospace;color:var(--ink-3);background:none;border:0;padding:2px 0;cursor:pointer;display:inline-flex;align-items:center;gap:7px}
.chips button::before{content:"";width:9px;height:9px;border:1px solid var(--ink-3);border-radius:2px}
.chips button[aria-pressed=true]{color:var(--ink-1)}.chips button[aria-pressed=true]::before{background:var(--ink-1);border-color:var(--ink-1)}
select{color:var(--ink-1);border-bottom:1px solid var(--hair);padding-right:4px}
.controls{display:flex;gap:28px;align-items:center;margin:40px 0 28px}
.lab{font:500 10.5px "Space Grotesk",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.card{margin:0 0 64px}
.card h2{font:600 13px "Space Grotesk",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-2);margin:0 0 4px}
.key{font-size:12px;color:var(--ink-3);display:flex;gap:18px;align-items:center;margin:0 0 18px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);margin:0 0 64px;border-top:1px solid var(--hair);border-bottom:1px solid var(--hair)}
.tile{padding:20px 0 22px 24px;border-left:1px solid var(--hair)}.tile:first-child{border-left:0;padding-left:0}
.tile .v{font-size:40px;font-weight:500;letter-spacing:-.03em;line-height:1.1;margin:8px 0 6px}
.tile .v small{font-size:13px;color:var(--ink-3);letter-spacing:0;margin-left:6px}
.tile .w{font-size:13px;color:var(--ink-2)}
table{border-collapse:collapse;width:100%}
th.cat{font-style:italic}
th{font:500 10.5px "Space Grotesk",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);
 text-align:right;padding:0 10px 10px;border-bottom:1px solid var(--ink-1);cursor:pointer;user-select:none;white-space:nowrap}
td{padding:11px 10px;border-bottom:1px solid var(--hair);text-align:right;white-space:nowrap;font-size:13px}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th[aria-sort=descending]::after{content:" ↓"}th[aria-sort=ascending]::after{content:" ↑"}
tbody tr:hover td,tbody tr.on td{background:var(--hover)}
svg g.s{transition:opacity .15s}
td.rank{color:var(--ink-3);width:28px;padding-right:0}
td.name{font-family:"Space Grotesk",sans-serif;font-size:14px;font-weight:500}
td.name .meta{display:block;font:400 11px "Geist Mono",monospace;color:var(--ink-3);margin-top:1px}
.cell{display:flex;align-items:center;gap:10px;justify-content:flex-end}
.meter{position:relative;display:inline-block;width:84px;height:3px}.meter.sm{width:48px}
table.dense th,table.dense td{padding-left:6px;padding-right:6px}table.dense th{letter-spacing:.04em;white-space:normal;line-height:1.3;vertical-align:bottom}
.cell.stack{display:inline-flex;flex-direction:column;align-items:flex-end;gap:5px}.stack .meter.sm{width:44px}.stack .meter:not(.sm){width:64px}
table.dense td.name,table.dense th:nth-child(2){position:sticky;left:0;z-index:1;background:var(--surface-1);box-shadow:8px 0 8px -8px var(--hair)}
table.dense tr:hover td.name{background:var(--hover)}
.meter>i{position:absolute;left:0;top:0;bottom:0;border-radius:2px}
.meter>b{position:absolute;top:1px;height:1px;background:var(--ink-3);opacity:.6}
.meter>u{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--ink-3);opacity:.6}
.big{font-size:15px;font-weight:700}
.muted,.note{color:var(--ink-3)}.note{font-size:12px}
.radar{display:grid;grid-template-columns:420px minmax(0,1fr);gap:40px;align-items:center}
@media (max-width:900px){.radar{grid-template-columns:1fr}.tiles{grid-template-columns:1fr 1fr}}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:4px 24px;align-content:start}
.legend label{display:flex;gap:10px;align-items:center;cursor:pointer;color:var(--ink-2);font-size:13px}
.legend input{accent-color:var(--ink-1)}
.legend .sw{width:14px;height:2px;border-radius:1px;display:inline-block}
.legend input:disabled+span{opacity:.35}
.colm{min-width:0}
.bars .key .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}.bars .key .psw{margin-right:6px;vertical-align:-1px}
.bars .key .sw.mean{height:2px;background:var(--ink-1);border-radius:0;vertical-align:3px}
.bars .key .sw.lead{height:0;border-top:1px dashed #1E5ACA;border-radius:0;vertical-align:3px}
.barsvg svg{display:block;overflow:visible}.barsvg .ax{font:500 10.5px "Geist Mono",monospace;fill:var(--ink-3)}
.barsvg text{text-transform:none;letter-spacing:0}.barsvg .xl{font:500 11px "Geist Mono",monospace;fill:var(--ink-2)}.barsvg .mv{font:700 11px "Geist Mono",monospace;fill:var(--ink-1)}
.barsvg .top{fill:#3B5FD9}.barsvg .ld{stroke:#3B5FD9}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .barsvg .top{fill:#7aa5f5}:root:where(:not([data-theme="light"])) .barsvg .ld{stroke:#7aa5f5}}
:root[data-theme="dark"] .barsvg .top{fill:#7aa5f5}:root[data-theme="dark"] .barsvg .ld{stroke:#7aa5f5}.barsvg g.grp{transition:opacity .12s}.barsvg.hov g.grp:not(.on){opacity:.3}
/* the profile sits above the leaderboard; on a wide screen, beside it and pinned, so a hovered row's profile stays in view */
.grid2{display:flex;flex-direction:column}.grid2>.side{order:-1}
@media (min-width:1760px){main{max-width:1720px}
 .grid2{display:grid;grid-template-columns:400px minmax(0,1fr);gap:56px;align-items:start}
 .grid2>.side{position:sticky;top:24px;margin:0;overflow:visible}
 .grid2>.side .radar{grid-template-columns:1fr;gap:12px}
 .grid2>.side .legend{grid-template-columns:1fr;max-height:calc(100vh - 560px);overflow:auto}}
svg text{fill:var(--ink-3);font:500 10.5px "Space Grotesk",sans-serif;letter-spacing:.1em;text-transform:uppercase}
svg .tick{font:400 9.5px "Geist Mono",monospace;letter-spacing:0;text-transform:none}
svg polygon.s{transition:opacity .15s}
.tip{position:fixed;pointer-events:none;background:#fff;color:#0b0b0b;border:1px solid #e7e6e1;box-shadow:0 2px 8px rgba(0,0,0,.08);
 font:500 11px "Geist Mono",monospace;padding:3px 7px;border-radius:4px;opacity:0;transition:opacity .08s;white-space:nowrap;z-index:9}
.empty{color:var(--ink-3);padding:24px 0}
</style></head>
<body><div class="viz-root"><main>
<header><a class="brand" href="https://vlm.run" target="_blank" rel="noopener"><svg class="logo" role="img" aria-label="VLM Run" viewBox="0 0 171 130" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M10.4983 16.086C10.4983 7.20196 17.7003 0 26.5843 0H47.581V10.724H26.5843C23.623 10.724 21.2223 13.1247 21.2223 16.086V58.1921C21.2223 61.1535 18.8217 63.5541 15.8603 63.5541L15.8603 66.1222C18.8217 66.1222 21.2223 68.5229 21.2223 71.4842V113.59C21.2223 116.552 23.623 118.952 26.5843 118.952H47.581V129.676H26.5843C17.7003 129.676 10.4983 122.474 10.4983 113.59V70.2002H0V59.4762H10.4983V16.086Z" fill="#1E5ACA"/><path d="M44.3593 60.8161L32.4557 27.7432H42.1716L50.1503 51.5506L58.0003 27.7432H67.5232L55.6195 60.8161H44.3593Z" fill="#1E5ACA"/><path d="M69.26 60.8161V15.7752H78.3325V60.8161H69.26Z" fill="#1E5ACA"/><path d="M122.94 27.3572C129.76 27.3572 135.165 31.2178 135.165 41.8346V60.8161H126.092V42.3493C126.092 37.3949 124.291 34.8211 120.366 34.8211C116.441 34.8211 113.803 37.7166 113.803 42.7354V60.8161H104.73V42.3493C104.73 37.3949 102.929 34.8211 98.9392 34.8211C95.0142 34.8211 92.4404 37.8453 92.4404 42.7998V60.8161H83.3679V27.7432H91.2822L92.1187 31.8613C94.1134 29.2875 97.0089 27.4215 101.835 27.3572C105.888 27.2928 109.685 28.7727 111.744 32.9551C114.06 29.4162 117.921 27.3572 122.94 27.3572Z" fill="#1E5ACA"/><path d="M58.3526 73.3203H59.8969V81.5564H56.2936C50.8887 81.5564 48.8297 85.1596 48.8297 90.1785V106.393H39.7572V73.3203H47.9932L48.8297 78.2748C50.6313 75.315 53.2051 73.3203 58.3526 73.3203Z" fill="#1E5ACA"/><path d="M84.7322 89.8567V73.3203H93.8047V106.393H85.7617L84.9252 102.404C82.8019 104.978 79.9064 106.779 75.2093 106.779C68.3888 106.779 62.0831 103.369 62.0831 91.6584V73.3203H71.1556V90.4358C71.1556 96.3555 73.0859 99.251 77.59 99.251C82.0941 99.251 84.7322 95.9694 84.7322 89.8567Z" fill="#1E5ACA"/><path d="M117.807 72.9342C124.949 72.9342 131.062 76.7305 131.062 88.0551V106.393H121.989V88.9559C121.989 83.358 119.866 80.3982 115.298 80.3982C110.601 80.3982 107.898 83.744 107.898 89.4707V106.393H98.8256V73.3203H106.74L107.576 77.4383C109.7 74.8646 112.724 72.9342 117.807 72.9342Z" fill="#1E5ACA"/><path d="M159.732 16.086C159.732 7.20196 152.53 0 143.646 0H122.649V10.724H143.646C146.607 10.724 149.008 13.1247 149.008 16.086V58.1921C149.008 61.1535 151.408 63.5541 154.37 63.5541V66.1222C151.408 66.1222 149.008 68.5229 149.008 71.4842V113.59C149.008 116.552 146.607 118.952 143.646 118.952H122.649V129.676H143.646C152.53 129.676 159.732 122.474 159.732 113.59V70.2002H170.23V59.4762H159.732V16.086Z" fill="#1E5ACA"/></svg><h1>vlmbench</h1></a>
<nav class="tabs" role="tablist"><button role="tab" data-tab="acc" aria-selected="true">Accuracy</button>
<button role="tab" data-tab="runs" aria-selected="false">Runs</button></nav>
<span style="flex:1"></span><span class="sub mono" id="sub"></span><div class="seg" id="theme"><button data-t="light">Light</button><button data-t="dark">Dark</button></div></header>
<section id="acc"></section><section id="runs" hidden></section>
<footer><a href="https://github.com/vlm-run/vlmbench" target="_blank" rel="noopener">github.com/vlm-run/vlmbench</a><span>·</span><span>open-source, by <a href="https://vlm.run" target="_blank" rel="noopener">VLM Run</a></span></footer>
</main><div class="tip" id="tip"></div></div>
<script>
const D=/*DATA*/null;
const $=(s,r=document)=>r.querySelector(s), el=(t,a={},h="")=>{const e=document.createElement(t);Object.assign(e,a);if(h)e.innerHTML=h;return e};
const usd=(v,rate)=>{if(v==null)return"—";const t="$"+(+v).toFixed(v>=1?2:v>=.01?3:4);
 return rate?`<span class="cell" data-tip="GPU time at $${f(rate,2)}/h, not billed">~${t}</span>`:t};
const f=(v,d=1)=>v==null?"—":(+v).toFixed(d), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tip=$("#tip");
// a table scrolls in its own box, at most a screen tall, so its header row stays in view
const frame=t=>{const w=el("div",{className:"frame"});w.append(t);
 new ResizeObserver(()=>w.classList.toggle("box",t.scrollWidth>w.clientWidth+1)).observe(w);return w};
// search: every word must appear in a row's model, backend, prompt, eval, dataset or split; ranks are kept
const Q={terms:[],text:""}, hay=r=>[r.serve,r.model,r.backend,r.prompt,r.eval,r.dataset,r.split,r.gpu].filter(Boolean).join(" ").toLowerCase();
const match=r=>Q.terms.every(t=>hay(r).includes(t));
function searchBox(){ // one query, shown above the leaderboard and above the runs tables
 const q=el("input",{type:"search",className:"q",placeholder:"search models   /",autocomplete:"off",spellcheck:false,value:Q.text});
 q.setAttribute("aria-label","Search models, backends, prompts, datasets");
 q.oninput=()=>{Q.text=q.value;Q.terms=q.value.toLowerCase().split(/\s+/).filter(Boolean);
  document.querySelectorAll("input.q").forEach(o=>{if(o!==q)o.value=q.value});document.querySelectorAll("table,.bars").forEach(t=>t.redraw?.())};
 return el("div",{className:"qrow"}).appendChild(q).parentNode}
document.addEventListener("keydown",e=>{const q=[...document.querySelectorAll("input.q")].find(x=>x.offsetParent);if(!q)return;
 if(e.key==="/"&&document.activeElement!==q&&!/input|select|textarea/i.test(document.activeElement?.tagName)){e.preventDefault();q.focus()}
 else if(e.key==="Escape"&&document.activeElement===q){q.value="";q.oninput();q.blur()}});
// a dataset header opens a card with its Hugging Face repo and pinned revision (hover; it stays while the pointer is on it)
const pop=el("div",{className:"pop"});(document.querySelector(".viz-root")||document.body).append(pop);let popT=0;
pop.onmouseenter=()=>clearTimeout(popT);pop.onmouseleave=()=>{popT=setTimeout(()=>pop.classList.remove("on"),150)};
function link(th,l){if(!l)return;
 const hf=`https://huggingface.co/datasets/${l.repo}`, rev=l.revision?`${hf}/tree/${l.revision}`:null;
 th.onmouseenter=()=>{clearTimeout(popT);
  pop.innerHTML=`<a href="${hf}" target="_blank" rel="noopener">${esc(l.repo)} ↗</a>`+(rev?`<a class="rev" href="${rev}" target="_blank" rel="noopener">pinned @ ${esc(l.revision.slice(0,10))}</a>`:"");
  above(pop,th.getBoundingClientRect());pop.classList.add("on")};
 th.onmouseleave=()=>{popT=setTimeout(()=>pop.classList.remove("on"),150)}}
window.addEventListener("scroll",()=>pop.classList.remove("on"),true);
// a tooltip sits above what it describes (below only when there is no room), never over the numbers beside it
function above(node,b,gap=6){const top=b.top-node.offsetHeight-gap;node.style.top=`${top<4?b.bottom+gap:top}px`;
 node.style.left=`${Math.max(8,Math.min(innerWidth-node.offsetWidth-8,b.left+b.width/2-node.offsetWidth/2))}px`}
document.addEventListener("mouseover",e=>{const t=e.target.closest("[data-tip]");if(!t||!t.dataset.tip){tip.style.opacity=0;return}
 tip.textContent=t.dataset.tip;above(tip,t.getBoundingClientRect());tip.style.opacity=1});
// theme toggle: an explicit choice beats the OS setting both ways
document.querySelectorAll("#theme button").forEach(b=>b.onclick=()=>{document.documentElement.dataset.theme=b.dataset.t;
 document.querySelectorAll("#theme button").forEach(x=>x.setAttribute("aria-pressed",x===b))});
document.querySelectorAll("nav.tabs button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav.tabs button").forEach(x=>x.setAttribute("aria-selected",x===b));
 $("#acc").hidden=b.dataset.tab!=="acc";$("#runs").hidden=b.dataset.tab!=="runs"});
$("#sub").textContent=`${D.evals.length} eval${D.evals.length==1?"":"s"} · ${D.runs.length} runs`;

const tone=v=>v==null?"var(--ink-3)":v>=80?"var(--good)":v>=60?"var(--warn)":"var(--bad)";
function meter(v,ci,sm,color){ // a 0-100 score: status colour by default; the 95% interval as a hairline
 const w=Math.max(0,Math.min(100,v??0)), c=color||tone(v);
 const b=ci?`<b style="left:${Math.max(0,ci[0])}%;width:${Math.max(0,Math.min(100,ci[1])-Math.max(0,ci[0]))}%"></b><u style="left:${Math.max(0,ci[0])}%"></u><u style="left:${Math.min(100,ci[1])}%"></u>`:"";
 return `<span class="meter${sm?" sm":""}"><i style="width:${w}%;background:${c}"></i>${b}</span>`}
function sortable(tbl,rows,cols,render){let key=null,dir=-1;
 const head=tbl.tHead.rows[0];[...head.cells].forEach((th,i)=>{if(!cols[i].key)return;th.onclick=()=>{dir=key===cols[i].key?-dir:-1;key=cols[i].key;
  [...head.cells].forEach(c=>c.removeAttribute("aria-sort"));th.setAttribute("aria-sort",dir<0?"descending":"ascending");draw()}});
 function val(r,k){const v=k.split(".").reduce((o,p)=>o?.[p],r);return v==null?-Infinity:v}
 function draw(){const rs=key?[...rows].sort((a,b)=>{const x=val(a,key),y=val(b,key);return (typeof x==="string"?x.localeCompare(y):x-y)*dir}):rows;
  const shown=rs.map((r,i)=>[r,i]).filter(([r])=>match(r));
  tbl.tBodies[0].replaceChildren(...(shown.length?shown.map(([r,i])=>render(r,i)):
   [el("tr",{},`<td class="none" colspan="${cols.length}">Nothing matches “${esc(Q.terms.join(" "))}”</td>`)]))}
 tbl.redraw=draw;draw()}

// ── accuracy tab ──
function accTab(){const root=$("#acc");root.replaceChildren();
 if(!D.evals.length){root.append(el("p",{className:"empty"},"","No eval results yet: <code>vlmbench eval benchmarks/&lt;bench&gt;</code>."));return}
 const ctl=el("div",{className:"controls"});root.append(ctl);
 const pick=el("select");D.evals.forEach((e,i)=>pick.append(el("option",{value:i,textContent:e.name})));
 ctl.append(el("span",{className:"lab",textContent:"Eval"}),pick);
 const seg=el("div",{className:"seg"});ctl.append(el("span",{className:"lab",textContent:"Split"}),seg);
 const dpick=el("select");ctl.append(el("span",{className:"lab",textContent:"Dataset"}),dpick);
 const chips=el("div",{className:"chips"});root.append(chips);
 const body=el("div");root.append(body);let split;
 // the datasets the "all" view averages: a checklist, kept per eval in this browser
 const DEFAULT=["olmocr","omnidocbench","omniparse","parsebench"];
 const chosen=ev=>{try{const v=JSON.parse(localStorage.getItem("vlmbench.sel."+ev.name));if(Array.isArray(v)){const k=v.filter(d=>ev.datasets.includes(d));if(k.length)return k}}catch(e){}
  const d=DEFAULT.filter(x=>ev.datasets.includes(x));return d.length?d:[...ev.datasets]};
 const keep=(ev,sel)=>localStorage.setItem("vlmbench.sel."+ev.name,JSON.stringify(sel));
 function pooled(r,sel){ // mean over the chosen datasets, and its 95% interval from their bootstrap draws
  const ss=sel.map(d=>r.scores[d]);if(ss.some(s=>s==null))return {mean:null,ci:null};
  const mean=ss.reduce((a,s)=>a+s.score,0)/ss.length, n=Math.min(...ss.map(s=>s.draws?.length||0));
  if(!n)return {mean,ci:null};
  const m=Array.from({length:n},(_,k)=>ss.reduce((a,s)=>a+s.draws[k],0)/ss.length).sort((a,b)=>a-b);
  return {mean,ci:[m[Math.floor(.025*n)],m[Math.min(n-1,Math.ceil(.975*n)-1)]]}}
 function draw(){const ev=D.evals[pick.value], splits=Object.keys(ev.splits);
  if(!splits.includes(split))split=splits.includes("test")?"test":splits[0];
  seg.replaceChildren(...splits.map(s=>{const b=el("button",{textContent:s});b.setAttribute("aria-pressed",s===split);b.onclick=()=>{split=s;draw()};return b}));
  const ds=ev.datasets.includes(dpick.value)?dpick.value:"";
  dpick.replaceChildren(el("option",{value:"",textContent:"all"}),...ev.datasets.map(d=>el("option",{value:d,textContent:d})));dpick.value=ds;
  const all=ev.splits[split]||[];body.replaceChildren();
  MCOL.clear();all.forEach((r,i)=>MCOL.set(r.serve,PAL[i%PAL.length]));
  const sel=chosen(ev);chips.hidden=!!ds;
  chips.replaceChildren(el("span",{className:"lab",textContent:"Averaged"}),...ev.datasets.map(d=>{
   const b=el("button",{textContent:d});b.setAttribute("aria-pressed",sel.includes(d));
   b.onclick=()=>{const on=sel.includes(d);if(on&&sel.length===1)return;
    keep(ev,on?sel.filter(x=>x!==d):ev.datasets.filter(x=>x===d||sel.includes(x)));draw()};return b}));
  // one dataset: its score is the mean, its metrics and categories the columns, its own judge, speed and cost
  const rows=!ds?all.map(r=>({...r,...pooled(r,sel)})).sort((a,b)=>(b.mean??-1)-(a.mean??-1)):all.map(r=>{const s=r.scores[ds],p=r.perf_by?.[ds]||{};
   return {...r,mean:s?s.score:null,ci:s?.ci,judge:r.judge_by?.[ds]??null,pages_s:p.pages_s??null,usd_1k:p.usd_1k??null,cost:p.cost??null}})
   .sort((a,b)=>(b.mean??-1)-(a.mean??-1));
  const ok=rows.filter(r=>r.mean!=null), best=ok[0];
  const value=ok.filter(r=>r.usd_1k).sort((a,b)=>b.mean/b.usd_1k-a.mean/a.usd_1k)[0];
  const fast=ok.filter(r=>r.pages_s).sort((a,b)=>b.pages_s-a.pages_s)[0];
  const tile=(k,v,w)=>`<div class="tile"><div class="lab">${k}</div><div class="v">${v}</div><div class="w">${esc(w)}</div></div>`;
  if(best)body.append(el("div",{className:"tiles"},tile("Most accurate",f(best.mean),best.serve)+
   (value?tile("Best value",`${f(value.mean)}<small>$${f(value.usd_1k,2)} / 1k ${ev.unit}</small>`,value.serve):"")+
   (fast?tile("Fastest",`${f(fast.pages_s,2)}<small>${ev.unit}/s</small>`,fast.serve):"")+
   tile("Models",`${ok.length}<small>${ds?esc(ds):`× ${sel.length} benchmarks`}</small>`,`${split} split`)));
  const g=el("div",{className:"grid2"});body.append(g);
  const colm=el("div",{className:"colm"});g.append(colm);
  const card=el("div",{className:"card"});colm.append(card);
  card.append(el("h2",{textContent:"Leaderboard"}),el("div",{className:"key"},`<span><span class="dot" style="background:var(--good)"></span>≥ 80</span><span><span class="dot" style="background:var(--warn)"></span>60–80</span><span><span class="dot" style="background:var(--bad)"></span>&lt; 60</span><span>hairline = 95% CI</span><span>${ds?`${esc(ds)} score, then its metrics and categories`:`mean of ${sel.length} benchmarks; judge not averaged in`}</span>`));
  card.append(searchBox());
  // between score and judge: every dataset, or one dataset's metrics then its categories (each name once)
  const mid=[];
  if(!ds)sel.forEach(d=>mid.push({t:d,ds:d,key:`scores.${d}.score`,get:r=>r.scores[d]?.score,ci:r=>r.scores[d]?.ci}));
  else{const seen=new Set();
   for(const part of ["metrics","cats"])for(const r of all)for(const k of Object.keys(r.scores[ds]?.[part]||{})){
    if(seen.has(k))continue;seen.add(k);
    if(part==="metrics"&&all.every(r=>{const v=r.scores[ds]?.metrics?.[k];return v==null||Math.abs(v-r.scores[ds].score)<.05}))continue; // the score again
    mid.push({t:k,key:`scores.${ds}.${part}.${k}`,get:r=>r.scores[ds]?.[part]?.[k],cat:part==="cats"})}}
  const ratio=k=>/ratio/i.test(k);
  const cols=[{t:"#"},{t:"model",key:"serve",l:1},{t:"backend",key:"backend",l:1},{t:ds?"score":"mean",key:"mean",ds:ds||null},...mid,
   {t:"judge",key:"judge"},{t:`${ev.unit}/s`,key:"pages_s"},{t:`$/1k ${ev.unit}`,key:"usd_1k"}];
  const dense=mid.length>3, tbl=el("table",{className:dense?"dense":""});tbl.createTHead().append(el("tr",{},cols.map(c=>`<th class="${c.l?"l":""}${c.cat?" cat":""}"${c.cat?' title="category"':""}${c.ds?` data-ds="${esc(c.ds)}"`:""}>${esc(dense?c.t.replace(/_/g," "):c.t)}</th>`).join("")));tbl.createTBody();card.append(frame(tbl));
  tbl.tHead.querySelectorAll("th[data-ds]").forEach(th=>link(th,ev.links?.[th.dataset.ds]));
  let rad=null;
  sortable(tbl,rows,cols,(r,i)=>hover(el("tr",{},`<td class="rank">${String(i+1).padStart(2,"0")}</td>`+
   `<td class="l name">${esc(r.serve)}<span class="meta">${esc(r.prompt||"")}</span></td><td class="l note">${esc(r.backend)}</td>`+
   `<td><span class="cell stack" data-tip="${r.ci?`95% CI ${f(r.ci[0])}–${f(r.ci[1])}`:ds?"":"not every dataset graded"}"><span class="big">${f(r.mean)}</span>${r.mean!=null?meter(r.mean,r.ci):""}</span></td>`+
   mid.map(c=>{const v=c.get(r),ci=c.ci?.(r);
    if(ratio(c.t))return `<td>${v==null?"—":f(v,2)+"×"}</td>`;
    return `<td><span class="cell stack" data-tip="${esc(c.t)} ${f(v)}${ci?` · 95% CI ${f(ci[0])}–${f(ci[1])}`:""}">${f(v)}${v!=null?meter(v,ci,1):""}</span></td>`}).join("")+
   `<td><span class="cell stack" data-tip="LLM judge, 0–100">${f(r.judge)}${r.judge!=null?meter(r.judge,null,1):""}</span></td>`+
   `<td>${f(r.pages_s,2)}</td><td>${r.usd_1k==null?"—":(r.gpu_rate?`<span class="cell" data-tip="GPU time at $${f(r.gpu_rate,2)}/h, not billed">~$${f(r.usd_1k,2)}</span>`:"$"+f(r.usd_1k,2))}</td>`),r));
  function hover(tr,r){tr.onmouseenter=()=>rad?.focus(r.serve);tr.onmouseleave=()=>rad?.focus(null);return tr}
  rad=radar(ds?ev:{...ev,datasets:sel},rows);g.append(rad.card);
  colm.append(bars(ev,rows,mid,ds,rad))}
 pick.onchange=draw;dpick.onchange=draw;draw()}

// a model keeps one colour everywhere (bars, radar): its place in the full ranking when the page is built, never its
// rank in a filtered view. The reference chart's palette, in its order.
const PAL=["#3B5FD9","#E2C384","#86AFD9","#E0969C","#9C9C9C","#94C29C","#BBA6D8","#8ACBC8","#D3AA84","#7B9FD2"];
const MCOL=new Map(), mcol=s=>MCOL.get(s)||"var(--ink-3)";
// a dataset keeps one pattern: the default benchmarks first (solid, ///, \\\, crosshatch), then the rest
const PATS=["solid","fwd","back","cross","dots","horiz","vert","tint"];
function patOrder(ev){const DEF=["olmocr","omnidocbench","omniparse","parsebench"];
 return [...DEF.filter(d=>ev.datasets.includes(d)),...ev.datasets.filter(d=>!DEF.includes(d))]}
function patDef(id,kind,c){
 const L=(d)=>`<path d="${d}" stroke="${c}" stroke-width="1.6"/>`, bg=`<rect width="8" height="8" fill="${c}" fill-opacity=".32"/>`;
 const body={fwd:L("M-2,2l4,-4M0,8l8,-8M6,10l4,-4"),back:L("M-2,6l4,4M0,0l8,8M6,-2l4,4"),cross:L("M0,8l8,-8M0,0l8,8"),
  dots:`<circle cx="2" cy="2" r="1.3" fill="${c}"/><circle cx="6" cy="6" r="1.3" fill="${c}"/>`,horiz:L("M0,4h8"),vert:L("M4,0v8"),tint:""}[kind]??"";
 return `<pattern id="${id}" width="8" height="8" patternUnits="userSpaceOnUse">${bg}${body}</pattern>`}
const patFill=(id,kind,c)=>kind==="solid"?c:`url(#${id})`;

// grouped bars: one group per model (ranked as the leaderboard), one bar per benchmark (or, for one dataset, per metric
// and category); the leader banded, every group's mean a tick and its one label, each bar's value on hover
function bars(ev,rows,mid,ds,rad){
 const card=el("div",{className:"card bars"}), ser=mid.filter(c=>!/ratio/i.test(c.t)), order=patOrder(ev);
 const pat=(c,i)=>PATS[(ds?i:Math.max(0,order.indexOf(c.t)))%PATS.length];
 const sw=(kind,i)=>`<svg width="16" height="11" class="psw"><defs>${patDef("lg"+i,kind,"#8f8f8f")}</defs><rect width="16" height="11" rx="2" fill="${patFill("lg"+i,kind,"#8f8f8f")}"/></svg>`;
 card.append(el("h2",{textContent:ds?`${ds}, by metric and category`:"Scores by benchmark"}),
  el("div",{className:"key"},ser.map((c,i)=>`<span>${sw(pat(c,i),i)}${esc(c.t)}</span>`).join("")+
   `<span><span class="sw mean"></span>mean</span><span><span class="sw lead"></span>leader's mean</span><span>colour = model</span>`));
 const wrap=el("div",{className:"barsvg"});card.append(wrap);
 function draw(){
  const rs=rows.filter(match).filter(r=>r.mean!=null||ser.some(c=>c.get(r)!=null));
  if(!rs.length){wrap.innerHTML=`<p class="empty">Nothing to chart.</p>`;return}
  const reach=Math.max(...rs.map(r=>r.serve.length))*6.6*Math.SQRT1_2; // a 45° label's horizontal and vertical extent
  const W=1200,T=30,gw0=(W-40)/rs.length,L=Math.max(34,Math.round(reach-gw0/2+8)),B=Math.round(reach+28),H=300+B,pw=W-L-6,ph=H-T-B,y=v=>T+ph*(1-Math.max(0,Math.min(100,v))/100);
  const gw=pw/rs.length,bw=Math.max(3,Math.min(18,(gw*.8-2*(ser.length-1))/ser.length)),cw=ser.length*bw+2*(ser.length-1);
  const lead=rs.filter(r=>r.mean!=null).sort((a,b)=>b.mean-a.mean)[0];
  let defs="", o="";
  for(const v of [0,20,40,60,80,100])o+=`<line x1="${L}" x2="${W-6}" y1="${y(v)}" y2="${y(v)}" stroke="var(--hair)" stroke-width="1"/>`+
   `<text x="${L-8}" y="${y(v)+4}" text-anchor="end" class="ax">${v}</text>`;
  rs.forEach((r,gi)=>{const x0=L+gi*gw, bx=x0+(gw-cw)/2, top=r===lead, col=mcol(r.serve);
   o+=`<g class="grp" data-serve="${esc(r.serve)}"><rect x="${x0}" y="${T}" width="${gw}" height="${ph}" fill="${top?"rgba(59,95,217,.08)":"transparent"}"/>`;
   let hi=r.mean??0;
   ser.forEach((c,i)=>{const v=c.get(r);if(v==null)return;hi=Math.max(hi,v);
    const x=bx+i*(bw+2),yt=y(v),yb=y(0),rr=Math.min(3,bw/2,yb-yt),id=`bp${gi}_${i}`,k=pat(c,i);
    if(k!=="solid")defs+=patDef(id,k,col);
    o+=`<path d="M${x},${yb}V${yt+rr}Q${x},${yt} ${x+rr},${yt}H${x+bw-rr}Q${x+bw},${yt} ${x+bw},${yt+rr}V${yb}Z" fill="${patFill(id,k,col)}" stroke="${col}" stroke-width="${k==="solid"?0:1}" data-tip="${esc(r.serve)} · ${esc(c.t)} ${f(v)}"/>`});
   if(r.mean!=null)o+=`<line x1="${bx-3}" x2="${bx+cw+3}" y1="${y(r.mean)}" y2="${y(r.mean)}" stroke="var(--ink-1)" stroke-width="1.5" data-tip="${esc(r.serve)} · mean ${f(r.mean)}"/>`+
    `<text x="${x0+gw/2}" y="${y(hi)-8}" text-anchor="middle" class="mv${top?" top":""}">${f(r.mean)}</text>`;
   o+=`<text transform="translate(${x0+gw/2+3},${H-B+14}) rotate(-45)" text-anchor="end" class="xl${top?" top":""}">${esc(r.serve)}</text></g>`});
  if(lead)o+=`<line x1="${L}" x2="${W-6}" y1="${y(lead.mean)}" y2="${y(lead.mean)}" class="ld" stroke-width="1" stroke-dasharray="4 4" opacity=".6"/>`;
  wrap.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Scores per model and benchmark"><defs>${defs}</defs>${o}</svg>`;
  wrap.querySelectorAll("g.grp").forEach(g=>{g.onmouseenter=()=>{wrap.classList.add("hov");g.classList.add("on");rad?.focus(g.dataset.serve)};
   g.onmouseleave=()=>{wrap.classList.remove("hov");g.classList.remove("on");rad?.focus(null)}})}
 card.redraw=draw;draw();return card}

function radar(ev,rows){ // 5 axes: each benchmark, the judge, and speed relative to the fastest serve
 const card=el("div",{className:"card side"});card.append(el("h2",{textContent:"Profile"}),
  el("div",{className:"key",textContent:"0–100 per axis · speed relative to the fastest · up to 8 · hover a name to isolate it"}));
 const wrap=el("div",{className:"radar"});card.append(wrap);
 const axes=[...ev.datasets.map(d=>({k:d,get:r=>r.scores[d]?.score})),{k:"judge",get:r=>r.judge}];
 const fast=Math.max(...rows.map(r=>r.pages_s||0));if(fast>0)axes.push({k:"speed",get:r=>r.pages_s!=null?100*r.pages_s/fast:null});
 const W=400,H=330,cx=W/2,cy=H/2+6,R=118,n=axes.length,ang=i=>-Math.PI/2+2*Math.PI*i/n,pt=(i,v)=>[cx+R*v/100*Math.cos(ang(i)),cy+R*v/100*Math.sin(ang(i))];
 const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.setAttribute("width","100%");
 svg.setAttribute("role","img");svg.setAttribute("aria-label","Radar of each selected serve across "+axes.map(a=>a.k).join(", "));wrap.append(svg);
 const slots=new Map(), free=[1,2,3,4,5,6,7,8]; // a serve keeps its colour while it is selected
 // default: the best serve of each kind (prompted API, OCR method, local), then the next best, four in all
 const kind=r=>r.backend==="vllm"?"local":r.prompt&&/\.(md|txt)$/.test(r.prompt)?"prompted":"method";
 const pickd=[];for(const r of rows)if(r.mean!=null&&!pickd.some(p=>kind(p)===kind(r)))pickd.push(r);
 for(const r of rows)if(pickd.length<4&&!pickd.includes(r))pickd.push(r);
 pickd.slice(0,4).forEach(r=>slots.set(r.serve,free.shift()));
 const leg=el("div",{className:"legend"});wrap.append(leg);
 function drawLegend(){leg.replaceChildren(...rows.map(r=>{const on=slots.has(r.serve);
  const lab=el("label",{},`<input type="checkbox" ${on?"checked":""} ${!on&&!free.length?"disabled":""}><span><span class="sw" style="background:${on?mcol(r.serve):"var(--hair)"}"></span> ${esc(r.serve)}</span>`);
  lab.onmouseenter=()=>focus(r.serve);lab.onmouseleave=()=>focus(null);
  lab.querySelector("input").onchange=e=>{if(e.target.checked){slots.set(r.serve,free.shift())}else{free.push(slots.get(r.serve));free.sort((a,b)=>a-b);slots.delete(r.serve)}drawLegend();drawPlot()};return lab}))}
 let preview=null;
 function drawPlot(){let s="";for(const v of [25,50,75,100])s+=`<polygon points="${axes.map((_,i)=>pt(i,v).join(",")).join(" ")}" fill="none" stroke="var(--hair)" stroke-width="1"/>`;
  axes.forEach((a,i)=>{const [x,y]=pt(i,100),[lx,ly]=pt(i,118);s+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--hair)"/>`+
   `<text x="${lx}" y="${ly}" text-anchor="${Math.abs(lx-cx)<8?"middle":lx>cx?"start":"end"}" dominant-baseline="middle">${esc(a.k)}</text>`});
  s+=`<text class="tick" x="${cx+4}" y="${cy-R/2}">50</text><text class="tick" x="${cx+4}" y="${cy-R}">100</text>`;
  const series=(r,c,dash)=>{const ps=axes.map((a,i)=>pt(i,a.get(r)??0));
   let g=`<g class="s" data-serve="${esc(r.serve)}"><polygon points="${ps.map(p=>p.join(",")).join(" ")}" fill="none" stroke="${c}" stroke-width="1.75" stroke-linejoin="round"${dash?' stroke-dasharray="4 3"':""}/>`;
   ps.forEach((p,i)=>{const v=axes[i].get(r);g+=`<circle cx="${p[0]}" cy="${p[1]}" r="3" fill="${c}" stroke="var(--surface-1)" stroke-width="1.5"/><circle cx="${p[0]}" cy="${p[1]}" r="10" fill="transparent" data-tip="${esc(r.serve)} · ${esc(axes[i].k)}: ${v==null?"—":f(v)}"/>`});
   return g+"</g>"};
  for(const r of rows)if(slots.has(r.serve))s+=series(r,mcol(r.serve));
  if(preview&&!slots.has(preview.serve))s+=series(preview,"var(--ink-1)",1);  // a hovered row not on the radar
  svg.innerHTML=s}
 function focus(serve){ // one serve at full strength, every other at 15%; an unselected one drawn dashed
  const r=rows.find(x=>x.serve===serve)||null, redraw=(r&&!slots.has(r.serve))||(preview&&preview!==r);
  preview=r&&!slots.has(r.serve)?r:null;if(redraw)drawPlot();
  svg.querySelectorAll("g.s").forEach(g=>{const on=!serve||g.dataset.serve===serve;g.style.opacity=on?1:.15;
   g.querySelector("polygon").setAttribute("stroke-width",serve&&on?2.75:1.75)});
  leg.querySelectorAll("label").forEach((l,i)=>l.style.color=serve&&rows[i].serve===serve?"var(--ink-1)":"")}
 drawLegend();drawPlot();return {card,focus}}

// ── runs tab ──
function runsTab(){const root=$("#runs");root.append(searchBox());
 const ev=D.evals.flatMap(e=>e.perf.map(p=>({...p,eval:e.name})));
 if(ev.length){const c=el("div",{className:"card"});root.append(c);
  c.append(el("h2",{textContent:"Eval throughput"}),el("div",{className:"key",textContent:"per serve × dataset · live calls only"}));
  const top=Math.max(...ev.map(p=>p.items_per_s||0));
  const cols=[{t:"serve",key:"serve",l:1},{t:"dataset",key:"dataset",l:1},{t:"split",key:"split",l:1},{t:"conc.",key:"concurrency"},
   {t:"items/s",key:"items_per_s"},{t:"cost",key:"cost_usd"},{t:"$/1k items",key:"usd_per_1k_items"},{t:"out tok/s",key:"output_tok_per_s"},
   {t:"TTFT p50 s",key:"ttft_s.p50"},{t:"latency p50 · p95 s",key:"latency_s.p50"},{t:"errors",key:"errors"},{t:"cut",key:"truncated"}];
  const t=el("table");t.createTHead().append(el("tr",{},cols.map(x=>`<th class="${x.l?"l":""}">${x.t}</th>`).join("")));t.createTBody();c.append(frame(t));
  sortable(t,ev,cols,p=>el("tr",{},`<td class="l name">${esc(p.serve)}<span class="meta">${esc(p.eval)}</span></td><td class="l">${esc(p.dataset)}</td><td class="l">${esc(p.split)}</td><td>${p.concurrency}</td>`+
   `<td><span class="cell"><span class="num">${f(p.items_per_s,2)}</span>${meter(top?100*(p.items_per_s||0)/top:0,null,1,"var(--ink-2)")}</span></td>`+
   `<td>${usd(p.cost_usd,p.gpu_rate)}</td><td>${p.usd_per_1k_items==null?"—":"$"+f(p.usd_per_1k_items,2)}</td><td>${f(p.output_tok_per_s,0)}</td>`+
   `<td>${f(p.ttft_s?.p50,2)}</td><td>${f(p.latency_s?.p50,1)} · ${f(p.latency_s?.p95,1)}</td><td>${p.errors}</td><td>${p.truncated}</td>`))}
 const c=el("div",{className:"card"});root.append(c);
 c.append(el("h2",{textContent:"Throughput runs"}),el("div",{className:"key",textContent:"vlmbench run · one row per concurrency"}));
 if(!D.runs.length){c.append(el("p",{className:"empty",textContent:"No runs yet."}));return}
 const top=Math.max(...D.runs.map(r=>r.tok_s||0));
 const cols=[{t:"model",key:"model",l:1},{t:"backend",key:"backend",l:1},{t:"GPU",key:"gpu",l:1},{t:"conc.",key:"concurrency"},{t:"inputs",key:"inputs"},
  {t:"items/s",key:"items_s"},{t:"tok/s",key:"tok_s"},{t:"TTFT p50 ms",key:"ttft_p50"},{t:"TTFT p95 ms",key:"ttft_p95"},{t:"e2e p50 ms",key:"e2e_p50"},{t:"VRAM MiB",key:"vram_mib"},{t:"when",key:"when",l:1}];
 const t=el("table");t.createTHead().append(el("tr",{},cols.map(x=>`<th class="${x.l?"l":""}">${x.t}</th>`).join("")));t.createTBody();c.append(frame(t));
 sortable(t,D.runs,cols,r=>el("tr",{},`<td class="l">${esc(r.model)}</td><td class="l">${esc(r.backend)}</td><td class="l note">${esc(r.gpu)}</td><td>${r.concurrency??"—"}</td><td>${r.inputs??"—"}</td>`+
  `<td>${f(r.items_s,2)}</td><td><span class="cell"><span class="num">${f(r.tok_s,0)}</span>${meter(top?100*(r.tok_s||0)/top:0,null,1,"var(--ink-2)")}</span></td>`+
  `<td>${f(r.ttft_p50,0)}</td><td>${f(r.ttft_p95,0)}</td><td>${f(r.e2e_p50,0)}</td><td>${r.vram_mib??"—"}</td><td class="l note">${esc(r.when)}</td>`))}
accTab();runsTab();
</script></body></html>
"""
