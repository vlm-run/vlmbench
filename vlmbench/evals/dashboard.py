"""`vlmbench dashboard`: one self-contained HTML page over everything collected, with two tabs: accuracy evals
(the latest score per serve and dataset, with intervals, and a radar per eval) and runs (throughput sweeps from
`vlmbench run`, plus the throughput each eval measured)."""

import functools
import http.server
import json
from pathlib import Path

from vlmbench.evals.config import Settings, backend_name
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
                    "backend": backend_name(cfg["backend"]),
                    "prompt": Path(cfg["prompt_file"]).name if cfg.get("prompt_file") else cfg["request"].get("method"),
                    "mean": round(sum(a["score"] for a in bench.values()) / len(datasets), 1) if full else None,
                    "ci": list(ci) if ci else None,
                    "scores": {d: {"score": a["score"], "ci": a.get("ci95")} for d, a in bench.items()},
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
.sub{color:var(--ink-3);font-size:12px}
nav.tabs{display:flex;gap:22px}
nav.tabs button,.seg button,select{font:500 13px "Space Grotesk",sans-serif;color:var(--ink-3);background:none;border:0;padding:4px 0;cursor:pointer}
nav.tabs button[aria-selected=true]{color:var(--ink-1);box-shadow:inset 0 -1.5px 0 var(--ink-1)}
.seg{display:inline-flex;gap:14px}.seg button[aria-pressed=true]{color:var(--ink-1)}
select{color:var(--ink-1);border-bottom:1px solid var(--hair);padding-right:4px}
.controls{display:flex;gap:28px;align-items:center;margin:40px 0 28px}
.lab{font:500 10.5px "Space Grotesk",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.card{margin:0 0 64px;overflow-x:auto}
.card h2{font:600 13px "Space Grotesk",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-2);margin:0 0 4px}
.key{font-size:12px;color:var(--ink-3);display:flex;gap:18px;align-items:center;margin:0 0 18px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);margin:0 0 64px;border-top:1px solid var(--hair);border-bottom:1px solid var(--hair)}
.tile{padding:20px 0 22px 24px;border-left:1px solid var(--hair)}.tile:first-child{border-left:0;padding-left:0}
.tile .v{font-size:40px;font-weight:500;letter-spacing:-.03em;line-height:1.1;margin:8px 0 6px}
.tile .v small{font-size:13px;color:var(--ink-3);letter-spacing:0;margin-left:6px}
.tile .w{font-size:13px;color:var(--ink-2)}
table{border-collapse:collapse;width:100%}
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
.meter>i{position:absolute;left:0;top:0;bottom:0;border-radius:2px}
.meter>b{position:absolute;top:1px;height:1px;background:var(--ink-3);opacity:.6}
.meter>u{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--ink-3);opacity:.6}
.big{font-size:15px;font-weight:600}
.muted,.note{color:var(--ink-3)}.note{font-size:12px}
.radar{display:grid;grid-template-columns:420px minmax(0,1fr);gap:40px;align-items:center}
@media (max-width:900px){.radar{grid-template-columns:1fr}.tiles{grid-template-columns:1fr 1fr}}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:4px 24px;align-content:start}
.legend label{display:flex;gap:10px;align-items:center;cursor:pointer;color:var(--ink-2);font-size:13px}
.legend input{accent-color:var(--ink-1)}
.legend .sw{width:14px;height:2px;border-radius:1px;display:inline-block}
.legend input:disabled+span{opacity:.35}
svg text{fill:var(--ink-3);font:500 10.5px "Space Grotesk",sans-serif;letter-spacing:.1em;text-transform:uppercase}
svg .tick{font:400 9.5px "Geist Mono",monospace;letter-spacing:0;text-transform:none}
svg polygon.s{transition:opacity .15s}
.tip{position:fixed;pointer-events:none;background:var(--ink-1);color:var(--surface-1);font:500 12px "Geist Mono",monospace;
 padding:5px 9px;border-radius:4px;opacity:0;transition:opacity .08s;white-space:nowrap;z-index:9}
.empty{color:var(--ink-3);padding:24px 0}
</style></head>
<body><div class="viz-root"><main>
<header><h1>vlmbench</h1>
<nav class="tabs" role="tablist"><button role="tab" data-tab="acc" aria-selected="true">Accuracy</button>
<button role="tab" data-tab="runs" aria-selected="false">Runs</button></nav>
<span style="flex:1"></span><span class="sub mono" id="sub"></span><div class="seg" id="theme"><button data-t="light">Light</button><button data-t="dark">Dark</button></div></header>
<section id="acc"></section><section id="runs" hidden></section>
</main><div class="tip" id="tip"></div></div>
<script>
const D=/*DATA*/null;
const $=(s,r=document)=>r.querySelector(s), el=(t,a={},h="")=>{const e=document.createElement(t);Object.assign(e,a);if(h)e.innerHTML=h;return e};
const usd=(v,rate)=>{if(v==null)return"—";const t="$"+(+v).toFixed(v>=1?2:v>=.01?3:4);
 return rate?`<span class="cell" data-tip="GPU time at $${f(rate,2)}/h, not billed">~${t}</span>`:t};
const f=(v,d=1)=>v==null?"—":(+v).toFixed(d), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tip=$("#tip");
document.addEventListener("mousemove",e=>{const t=e.target.closest("[data-tip]");if(!t){tip.style.opacity=0;return}
 tip.textContent=t.dataset.tip;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";tip.style.opacity=1});
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
  tbl.tBodies[0].replaceChildren(...rs.map((r,i)=>render(r,i)))}
 draw()}

// ── accuracy tab ──
function accTab(){const root=$("#acc");root.replaceChildren();
 if(!D.evals.length){root.append(el("p",{className:"empty"},"","No eval results yet: <code>vlmbench eval benchmarks/&lt;bench&gt;</code>."));return}
 const ctl=el("div",{className:"controls"});root.append(ctl);
 const pick=el("select");D.evals.forEach((e,i)=>pick.append(el("option",{value:i,textContent:e.name})));
 ctl.append(el("span",{className:"lab",textContent:"Eval"}),pick);
 const seg=el("div",{className:"seg"});ctl.append(el("span",{className:"lab",textContent:"Split"}),seg);
 const body=el("div");root.append(body);let split;
 function draw(){const ev=D.evals[pick.value], splits=Object.keys(ev.splits);
  if(!splits.includes(split))split=splits.includes("test")?"test":splits[0];
  seg.replaceChildren(...splits.map(s=>{const b=el("button",{textContent:s});b.setAttribute("aria-pressed",s===split);b.onclick=()=>{split=s;draw()};return b}));
  const rows=ev.splits[split]||[];body.replaceChildren();
  const ok=rows.filter(r=>r.mean!=null), best=ok[0];
  const value=ok.filter(r=>r.usd_1k).sort((a,b)=>b.mean/b.usd_1k-a.mean/a.usd_1k)[0];
  const fast=ok.filter(r=>r.pages_s).sort((a,b)=>b.pages_s-a.pages_s)[0];
  const tile=(k,v,w)=>`<div class="tile"><div class="lab">${k}</div><div class="v">${v}</div><div class="w">${esc(w)}</div></div>`;
  if(best)body.append(el("div",{className:"tiles"},tile("Most accurate",f(best.mean),best.serve)+
   (value?tile("Best value",`${f(value.mean)}<small>$${f(value.usd_1k,2)} / 1k</small>`,value.serve):"")+
   (fast?tile("Fastest",`${f(fast.pages_s,2)}<small>items/s</small>`,fast.serve):"")+
   tile("Models",`${ok.length}<small>× ${ev.datasets.length} benchmarks</small>`,`${split} split`)));
  const g=el("div",{className:"grid2"});body.append(g);
  const card=el("div",{className:"card"});g.append(card);
  card.append(el("h2",{textContent:"Leaderboard"}),el("div",{className:"key"},`<span><span class="dot" style="background:var(--good)"></span>≥ 80</span><span><span class="dot" style="background:var(--warn)"></span>60–80</span><span><span class="dot" style="background:var(--bad)"></span>&lt; 60</span><span>hairline = 95% CI</span><span>mean of ${ev.datasets.length} benchmarks; judge not averaged in</span>`));
  const cols=[{t:"#"},{t:"model",key:"serve",l:1},{t:"backend",key:"backend",l:1},{t:"mean",key:"mean"},...ev.datasets.map(d=>({t:d,key:`scores.${d}.score`})),
   {t:"judge",key:"judge"},{t:"items/s",key:"pages_s"},{t:"$ / 1k",key:"usd_1k"},{t:"cost",key:"cost"}];
  const tbl=el("table");tbl.createTHead().append(el("tr",{},cols.map(c=>`<th class="${c.l?"l":""}">${esc(c.t)}</th>`).join("")));tbl.createTBody();card.append(tbl);
  let rad=null;
  sortable(tbl,rows,cols,(r,i)=>hover(el("tr",{},`<td class="rank">${String(i+1).padStart(2,"0")}</td>`+
   `<td class="l name">${esc(r.serve)}<span class="meta">${esc(r.prompt||"")}</span></td><td class="l note">${esc(r.backend)}</td>`+
   `<td><span class="cell" data-tip="${r.ci?`95% CI ${f(r.ci[0])}–${f(r.ci[1])}`:"not every dataset graded"}"><span class="big">${f(r.mean)}</span>${meter(r.mean,r.ci)}</span></td>`+
   ev.datasets.map(d=>{const s=r.scores[d];return `<td><span class="cell" data-tip="${esc(d)} ${s?f(s.score):"—"}${s&&s.ci?` · 95% CI ${f(s.ci[0])}–${f(s.ci[1])}`:""}">${s?f(s.score):"—"}${s?meter(s.score,s.ci,1):""}</span></td>`}).join("")+
   `<td><span class="cell" data-tip="LLM judge, 0–100">${f(r.judge)}${r.judge!=null?meter(r.judge,null,1):""}</span></td>`+
   `<td>${f(r.pages_s,2)}</td><td>${r.usd_1k==null?"—":f(r.usd_1k,2)}</td><td>${usd(r.cost,r.gpu_rate)}</td>`),r));
  function hover(tr,r){tr.onmouseenter=()=>rad?.focus(r.serve);tr.onmouseleave=()=>rad?.focus(null);return tr}
  rad=radar(ev,rows);g.append(rad.card)}
 pick.onchange=draw;draw()}

function radar(ev,rows){ // 5 axes: each benchmark, the judge, and speed relative to the fastest serve
 const card=el("div",{className:"card"});card.append(el("h2",{textContent:"Profile"}),
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
  const lab=el("label",{},`<input type="checkbox" ${on?"checked":""} ${!on&&!free.length?"disabled":""}><span><span class="sw" style="background:${on?`var(--s${slots.get(r.serve)})`:"var(--hair)"}"></span> ${esc(r.serve)}</span>`);
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
  for(const r of rows)if(slots.has(r.serve))s+=series(r,`var(--s${slots.get(r.serve)})`);
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
function runsTab(){const root=$("#runs");
 const ev=D.evals.flatMap(e=>e.perf.map(p=>({...p,eval:e.name})));
 if(ev.length){const c=el("div",{className:"card"});root.append(c);
  c.append(el("h2",{textContent:"Eval throughput"}),el("div",{className:"key",textContent:"per serve × dataset · live calls only"}));
  const top=Math.max(...ev.map(p=>p.items_per_s||0));
  const cols=[{t:"serve",key:"serve",l:1},{t:"dataset",key:"dataset",l:1},{t:"split",key:"split",l:1},{t:"conc.",key:"concurrency"},
   {t:"items/s",key:"items_per_s"},{t:"cost",key:"cost_usd"},{t:"$/1k items",key:"usd_per_1k_items"},{t:"out tok/s",key:"output_tok_per_s"},
   {t:"TTFT p50 s",key:"ttft_s.p50"},{t:"latency p50 · p95 s",key:"latency_s.p50"},{t:"errors",key:"errors"},{t:"cut",key:"truncated"}];
  const t=el("table");t.createTHead().append(el("tr",{},cols.map(x=>`<th class="${x.l?"l":""}">${x.t}</th>`).join("")));t.createTBody();c.append(t);
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
 const t=el("table");t.createTHead().append(el("tr",{},cols.map(x=>`<th class="${x.l?"l":""}">${x.t}</th>`).join("")));t.createTBody();c.append(t);
 sortable(t,D.runs,cols,r=>el("tr",{},`<td class="l">${esc(r.model)}</td><td class="l">${esc(r.backend)}</td><td class="l note">${esc(r.gpu)}</td><td>${r.concurrency??"—"}</td><td>${r.inputs??"—"}</td>`+
  `<td>${f(r.items_s,2)}</td><td><span class="cell"><span class="num">${f(r.tok_s,0)}</span>${meter(top?100*(r.tok_s||0)/top:0,null,1,"var(--ink-2)")}</span></td>`+
  `<td>${f(r.ttft_p50,0)}</td><td>${f(r.ttft_p95,0)}</td><td>${f(r.e2e_p50,0)}</td><td>${r.vram_mib??"—"}</td><td class="l note">${esc(r.when)}</td>`))}
accTab();runsTab();
</script></body></html>
"""
