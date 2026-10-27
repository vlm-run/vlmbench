"""`vlmbench dashboard`: one self-contained HTML page over everything collected, with two tabs: accuracy evals
(the latest score per serve and dataset, with intervals, and a radar per eval) and runs (throughput sweeps from
`vlmbench run`, plus the throughput each eval measured)."""

import functools
import http.server
import json
from pathlib import Path

from vlmbench.evals.config import Settings
from vlmbench.evals.results import interval, latest


def collect(home: Path) -> dict:
    return {"evals": [_eval(d) for d in sorted((home / "evals").glob("*")) if d.is_dir()], "runs": _runs(home)}


def _eval(root: Path) -> dict:
    rows = latest(root)
    configs = {k[1]: v for k, v in rows.items() if k[0] == "cfg"}
    acc = [v for k, v in rows.items() if k[0] == "acc"]
    perf = [v for k, v in rows.items() if k[0] == "perf"]
    datasets = sorted({a["dataset"] for a in acc if "judge" not in a})
    out = {"name": root.name, "datasets": datasets, "splits": {}}
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
                    "backend": cfg["backend"]["type"],
                    "prompt": Path(cfg["prompt_file"]).name if cfg.get("prompt_file") else cfg["request"].get("method"),
                    "mean": round(sum(a["score"] for a in bench.values()) / len(datasets), 1) if full else None,
                    "ci": list(ci) if ci else None,
                    "scores": {d: {"score": a["score"], "ci": a.get("ci95")} for d, a in bench.items()},
                    "judge": round(sum(judged) / len(judged), 1) if judged else None,
                    "pages_s": _mean([p["items_per_s"] for p in ps]),
                    "usd_1k": _mean([p["usd_per_1k_items"] for p in ps]),
                    "ttft_p50": _mean([(p.get("ttft_s") or {}).get("p50") for p in ps]),
                }
            )
        out["splits"][split] = sorted(serves, key=lambda s: -(s["mean"] or 0))
    out["perf"] = [p | {"serve": p["serve"].removeprefix(f"{root.name}-")} for p in perf]
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
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f4f3f0;--grid:#e4e2dc;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#8a8984;--bar:#2a78d6;--bar-track:#e9edf3;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--surface-2:#222220;--grid:#33332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#8f8e86;--bar:#3987e5;--bar-track:#2a2f37;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#222220;--grid:#33332f;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;--bar:#3987e5;--bar-track:#2a2f37;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--surface-1);color:var(--text-primary);min-height:100vh}
main{max-width:1320px;margin:0 auto;padding:24px 24px 64px}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
h1{font-size:20px;margin:0}.sub{color:var(--text-secondary)}
nav.tabs{display:flex;gap:4px;margin:20px 0 16px;border-bottom:1px solid var(--grid)}
nav.tabs button{background:none;border:0;border-bottom:2px solid transparent;padding:8px 14px;font:inherit;
 color:var(--text-secondary);cursor:pointer}
nav.tabs button[aria-selected=true]{color:var(--text-primary);border-bottom-color:var(--bar);font-weight:600}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 12px}
select,.seg button{font:inherit;color:var(--text-primary);background:var(--surface-2);border:1px solid var(--grid);
 border-radius:6px;padding:4px 10px}
.seg{display:inline-flex}.seg button{border-radius:0;cursor:pointer}.seg button:first-child{border-radius:6px 0 0 6px}
.seg button:last-child{border-radius:0 6px 6px 0}.seg button[aria-pressed=true]{background:var(--bar);color:#fff;border-color:var(--bar)}
.card{background:var(--surface-2);border:1px solid var(--grid);border-radius:10px;padding:14px 16px;margin:0 0 16px;overflow-x:auto}
.card h2{font-size:15px;margin:0 0 4px}.note{color:var(--text-muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--text-secondary);font-weight:600;font-size:12px;cursor:pointer;user-select:none}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th[aria-sort=descending]::after{content:" ↓"}th[aria-sort=ascending]::after{content:" ↑"}
tr:hover td{background:color-mix(in srgb,var(--bar) 6%,transparent)}
.cell{display:flex;align-items:center;gap:8px;justify-content:flex-end}
.meter{position:relative;width:96px;height:8px;background:var(--bar-track);border-radius:4px}
.meter.sm{width:56px;height:6px}
.meter>i{position:absolute;left:0;top:0;bottom:0;background:var(--bar);border-radius:4px}
.meter>b{position:absolute;top:-3px;bottom:-3px;border-left:2px solid var(--text-secondary);border-right:2px solid var(--text-secondary);opacity:.55}
.num{min-width:38px;display:inline-block}
.tag{font-size:11px;color:var(--text-muted);margin-left:6px}
.grid2{display:block}
.radar{display:grid;grid-template-columns:440px minmax(0,1fr);gap:16px;align-items:center}
@media (max-width:900px){.radar{grid-template-columns:1fr}}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:2px 16px;align-content:start}
.legend label{display:flex;gap:8px;align-items:center;cursor:pointer;color:var(--text-secondary)}
.legend .sw{width:12px;height:3px;border-radius:2px;display:inline-block}
.legend input:disabled+span{opacity:.4}
svg text{fill:var(--text-secondary);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);font-size:12px;
 padding:4px 8px;border-radius:6px;opacity:0;transition:opacity .08s;white-space:nowrap;z-index:9}
.empty{color:var(--text-muted);padding:24px 0}
</style></head>
<body><div class="viz-root"><main>
<header><h1>vlmbench</h1><span class="sub" id="sub"></span>
<span style="flex:1"></span><div class="seg" id="theme"><button data-t="light">Light</button><button data-t="dark">Dark</button></div></header>
<nav class="tabs" role="tablist"><button role="tab" data-tab="acc" aria-selected="true">Accuracy evals</button>
<button role="tab" data-tab="runs" aria-selected="false">Runs</button></nav>
<section id="acc"></section><section id="runs" hidden></section>
</main><div class="tip" id="tip"></div></div>
<script>
const D=/*DATA*/null;
const $=(s,r=document)=>r.querySelector(s), el=(t,a={},h="")=>{const e=document.createElement(t);Object.assign(e,a);if(h)e.innerHTML=h;return e};
const f=(v,d=1)=>v==null?"—":(+v).toFixed(d), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tip=$("#tip");
document.addEventListener("mousemove",e=>{const t=e.target.closest("[data-tip]");if(!t){tip.style.opacity=0;return}
 tip.textContent=t.dataset.tip;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";tip.style.opacity=1});
// theme toggle: an explicit choice beats the OS setting both ways
document.querySelectorAll("#theme button").forEach(b=>b.onclick=()=>{document.documentElement.dataset.theme=b.dataset.t;
 document.querySelectorAll("#theme button").forEach(x=>x.setAttribute("aria-pressed",x===b))});
document.querySelectorAll("nav.tabs button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav.tabs button").forEach(x=>x.setAttribute("aria-selected",x===b));
 $("#acc").hidden=b.dataset.tab!=="acc";$("#runs").hidden=b.dataset.tab!=="runs"});
$("#sub").textContent=`${D.evals.length} eval${D.evals.length==1?"":"s"} · ${D.runs.length} throughput runs`;

function meter(v,ci,sm){ // a 0-100 score: fill, with the 95% interval as a bracket
 const w=Math.max(0,Math.min(100,v??0)), b=ci?`<b style="left:${Math.max(0,ci[0])}%;width:${Math.max(0,Math.min(100,ci[1])-Math.max(0,ci[0]))}%"></b>`:"";
 return `<span class="meter${sm?" sm":""}"><i style="width:${w}%"></i>${b}</span>`}
function sortable(tbl,rows,cols,render){let key=null,dir=-1;
 const head=tbl.tHead.rows[0];[...head.cells].forEach((th,i)=>{if(!cols[i].key)return;th.onclick=()=>{dir=key===cols[i].key?-dir:-1;key=cols[i].key;
  [...head.cells].forEach(c=>c.removeAttribute("aria-sort"));th.setAttribute("aria-sort",dir<0?"descending":"ascending");draw()}});
 function val(r,k){const v=k.split(".").reduce((o,p)=>o?.[p],r);return v==null?-Infinity:v}
 function draw(){const rs=key?[...rows].sort((a,b)=>{const x=val(a,key),y=val(b,key);return (typeof x==="string"?x.localeCompare(y):x-y)*dir}):rows;
  tbl.tBodies[0].replaceChildren(...rs.map(render))}
 draw()}

// ── accuracy tab ──
function accTab(){const root=$("#acc");root.replaceChildren();
 if(!D.evals.length){root.append(el("p",{className:"empty"},"","No eval results yet: <code>vlmbench eval benchmarks/&lt;bench&gt;</code>."));return}
 const ctl=el("div",{className:"controls"});root.append(ctl);
 const pick=el("select");D.evals.forEach((e,i)=>pick.append(el("option",{value:i,textContent:e.name})));
 ctl.append(el("span",{className:"note",textContent:"Eval"}),pick);
 const seg=el("div",{className:"seg"});ctl.append(el("span",{className:"note",textContent:"Split"}),seg);
 const body=el("div");root.append(body);let split;
 function draw(){const ev=D.evals[pick.value], splits=Object.keys(ev.splits);
  if(!splits.includes(split))split=splits.includes("test")?"test":splits[0];
  seg.replaceChildren(...splits.map(s=>{const b=el("button",{textContent:s});b.setAttribute("aria-pressed",s===split);b.onclick=()=>{split=s;draw()};return b}));
  const rows=ev.splits[split]||[];body.replaceChildren();
  const g=el("div",{className:"grid2"});body.append(g);
  const card=el("div",{className:"card"});g.append(card);
  card.append(el("h2",{textContent:`${ev.name} · ${split}`}),el("div",{className:"note",textContent:"Each benchmark graded by its own package; bars show the score, brackets its 95% page-bootstrap interval. Judge: an LLM judge, shown beside the benchmarks, never averaged in."}));
  const cols=[{t:"serve",key:"serve",l:1},{t:"mean",key:"mean"},...ev.datasets.map(d=>({t:d,key:`scores.${d}.score`})),
   {t:"judge",key:"judge"},{t:"pages/s",key:"pages_s"},{t:"$/1k pages",key:"usd_1k"},{t:"prompt",key:"prompt",l:1}];
  const tbl=el("table");tbl.createTHead().append(el("tr",{},cols.map(c=>`<th class="${c.l?"l":""}">${esc(c.t)}</th>`).join("")));tbl.createTBody();card.append(tbl);
  sortable(tbl,rows,cols,r=>el("tr",{},`<td class="l">${esc(r.serve)}<span class="tag">${esc(r.backend)}</span></td>`+
   `<td><span class="cell" data-tip="${r.ci?`95% CI ${f(r.ci[0])}–${f(r.ci[1])}`:"not every dataset graded"}"><span class="num"><b>${f(r.mean)}</b></span>${meter(r.mean,r.ci)}</span></td>`+
   ev.datasets.map(d=>{const s=r.scores[d];return `<td><span class="cell" data-tip="${esc(d)}: ${s?f(s.score):"—"}${s&&s.ci?` (95% CI ${f(s.ci[0])}–${f(s.ci[1])})`:""}"><span class="num">${s?f(s.score):"—"}</span>${s?meter(s.score,s.ci,1):""}</span></td>`}).join("")+
   `<td><span class="cell" data-tip="LLM judge, 0-100"><span class="num">${f(r.judge)}</span>${r.judge!=null?meter(r.judge,null,1):""}</span></td>`+
   `<td>${f(r.pages_s,2)}</td><td>${r.usd_1k==null?"—":"$"+f(r.usd_1k,2)}</td><td class="l note">${esc(r.prompt)}</td>`));
  g.append(radar(ev,rows))}
 pick.onchange=draw;draw()}

function radar(ev,rows){ // 5 axes: each benchmark, the judge, and speed relative to the fastest serve
 const card=el("div",{className:"card"});card.append(el("h2",{textContent:"Profile"}),
  el("div",{className:"note",textContent:"Each axis 0–100. Speed: pages/s as a share of the fastest serve here. Pick up to 8 to overlay."}));
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
  const lab=el("label",{},`<input type="checkbox" ${on?"checked":""} ${!on&&!free.length?"disabled":""}><span><span class="sw" style="background:${on?`var(--s${slots.get(r.serve)})`:"var(--grid)"}"></span> ${esc(r.serve)}</span>`);
  lab.querySelector("input").onchange=e=>{if(e.target.checked){slots.set(r.serve,free.shift())}else{free.push(slots.get(r.serve));free.sort((a,b)=>a-b);slots.delete(r.serve)}drawLegend();drawPlot()};return lab}))}
 function drawPlot(){let s="";for(const v of [25,50,75,100])s+=`<polygon points="${axes.map((_,i)=>pt(i,v).join(",")).join(" ")}" fill="none" stroke="var(--grid)" stroke-width="1"/>`;
  axes.forEach((a,i)=>{const [x,y]=pt(i,100),[lx,ly]=pt(i,118);s+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--grid)"/>`+
   `<text x="${lx}" y="${ly}" text-anchor="${Math.abs(lx-cx)<8?"middle":lx>cx?"start":"end"}" dominant-baseline="middle">${esc(a.k)}</text>`});
  s+=`<text x="${cx+3}" y="${cy-R/2}" font-size="10">50</text><text x="${cx+3}" y="${cy-R}" font-size="10">100</text>`;
  for(const r of rows){if(!slots.has(r.serve))continue;const c=`var(--s${slots.get(r.serve)})`;
   const ps=axes.map((a,i)=>pt(i,a.get(r)??0));s+=`<polygon points="${ps.map(p=>p.join(",")).join(" ")}" fill="${c}" fill-opacity=".04" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
   ps.forEach((p,i)=>{const v=axes[i].get(r);s+=`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${c}" stroke="var(--surface-2)" stroke-width="2"/><circle cx="${p[0]}" cy="${p[1]}" r="10" fill="transparent" data-tip="${esc(r.serve)} · ${esc(axes[i].k)}: ${v==null?"—":f(v)}"/>`})}
  svg.innerHTML=s}
 drawLegend();drawPlot();return card}

// ── runs tab ──
function runsTab(){const root=$("#runs");
 const ev=D.evals.flatMap(e=>e.perf.map(p=>({...p,eval:e.name})));
 if(ev.length){const c=el("div",{className:"card"});root.append(c);
  c.append(el("h2",{textContent:"Eval throughput"}),el("div",{className:"note",textContent:"Per serve and dataset, from the most complete live pass (cache hits are never timed)."}));
  const top=Math.max(...ev.map(p=>p.items_per_s||0));
  const cols=[{t:"eval",key:"eval",l:1},{t:"serve",key:"serve",l:1},{t:"dataset",key:"dataset",l:1},{t:"split",key:"split",l:1},{t:"conc.",key:"concurrency"},
   {t:"pages/s",key:"items_per_s"},{t:"out tok/s",key:"output_tok_per_s"},{t:"TTFT p50 s",key:"ttft_s.p50"},{t:"latency p50 s",key:"latency_s.p50"},
   {t:"latency p95 s",key:"latency_s.p95"},{t:"errors",key:"errors"},{t:"cut",key:"truncated"},{t:"$/1k pages",key:"usd_per_1k_items"}];
  const t=el("table");t.createTHead().append(el("tr",{},cols.map(x=>`<th class="${x.l?"l":""}">${x.t}</th>`).join("")));t.createTBody();c.append(t);
  sortable(t,ev,cols,p=>el("tr",{},`<td class="l">${esc(p.eval)}</td><td class="l">${esc(p.serve)}</td><td class="l">${esc(p.dataset)}</td><td class="l">${esc(p.split)}</td><td>${p.concurrency}</td>`+
   `<td><span class="cell"><span class="num">${f(p.items_per_s,2)}</span>${meter(top?100*(p.items_per_s||0)/top:0,null,1)}</span></td><td>${f(p.output_tok_per_s,0)}</td>`+
   `<td>${f(p.ttft_s?.p50,2)}</td><td>${f(p.latency_s?.p50,1)}</td><td>${f(p.latency_s?.p95,1)}</td><td>${p.errors}</td><td>${p.truncated}</td><td>${p.usd_per_1k_items==null?"—":"$"+f(p.usd_per_1k_items,2)}</td>`))}
 const c=el("div",{className:"card"});root.append(c);
 c.append(el("h2",{textContent:"Throughput runs"}),el("div",{className:"note",textContent:"vlmbench run results in ~/.vlmbench/benchmarks, one row per concurrency level."}));
 if(!D.runs.length){c.append(el("p",{className:"empty",textContent:"No runs yet."}));return}
 const top=Math.max(...D.runs.map(r=>r.tok_s||0));
 const cols=[{t:"model",key:"model",l:1},{t:"backend",key:"backend",l:1},{t:"GPU",key:"gpu",l:1},{t:"conc.",key:"concurrency"},{t:"inputs",key:"inputs"},
  {t:"items/s",key:"items_s"},{t:"tok/s",key:"tok_s"},{t:"TTFT p50 ms",key:"ttft_p50"},{t:"TTFT p95 ms",key:"ttft_p95"},{t:"e2e p50 ms",key:"e2e_p50"},{t:"VRAM MiB",key:"vram_mib"},{t:"when",key:"when",l:1}];
 const t=el("table");t.createTHead().append(el("tr",{},cols.map(x=>`<th class="${x.l?"l":""}">${x.t}</th>`).join("")));t.createTBody();c.append(t);
 sortable(t,D.runs,cols,r=>el("tr",{},`<td class="l">${esc(r.model)}</td><td class="l">${esc(r.backend)}</td><td class="l note">${esc(r.gpu)}</td><td>${r.concurrency??"—"}</td><td>${r.inputs??"—"}</td>`+
  `<td>${f(r.items_s,2)}</td><td><span class="cell"><span class="num">${f(r.tok_s,0)}</span>${meter(top?100*(r.tok_s||0)/top:0,null,1)}</span></td>`+
  `<td>${f(r.ttft_p50,0)}</td><td>${f(r.ttft_p95,0)}</td><td>${f(r.e2e_p50,0)}</td><td>${r.vram_mib??"—"}</td><td class="l note">${esc(r.when)}</td>`))}
accTab();runsTab();
</script></body></html>
"""
