"""`vlmbench eval`: every serve reads every sampled page, each dataset grades the replies with its own package, the
judges score them, and one result document holds performance and accuracy rows joined on (model_id, config_id)."""

import asyncio
import contextlib
import json
import platform
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from vlmbench.evals import serving
from vlmbench.evals.cache import DiskCache
from vlmbench.evals.client import Client
from vlmbench.evals.config import Eval, Metric, Serve, Settings, bench_files, expand, load
from vlmbench.evals.data import Item, render, sample
from vlmbench.evals.datasets import DATASETS
from vlmbench.evals.results import SCHEMA, interval, performance, write
from vlmbench.evals.trace import configure, record, span
from vlmbench.version import __version__


def pages(ev: Eval, names: list[str], split: str, home: Path) -> dict[str, list[tuple[Item, Path]]]:
    out = {}
    for d in names:
        spec = ev.datasets[d]
        its = sample(DATASETS[spec.kind].items(spec), spec.samples, split)
        out[d] = [(it, render(it, home / "pages" / d)) for it in its]
    return out


async def _predict(client: Client, serve: Serve, todo: dict[str, list[tuple[Item, Path]]], log) -> dict:
    """Each dataset in turn at the serve's concurrency, so its wall time is its own."""
    sem, out = asyncio.Semaphore(serve.request.concurrency), {}

    async def one(d: str, it: Item, png: Path) -> dict:
        async with sem:
            return await client.complete(serve, png, item=f"{d}/{it.id}")

    for d, its in todo.items():
        with span("bench.dataset", dataset=d, serve=serve.name, model=serve.model, config_id=serve.config_id) as s:
            t0 = time.perf_counter()
            recs = await asyncio.gather(*(one(d, it, png) for it, png in its))
            out[d] = (recs, time.perf_counter() - t0)
            record(
                s,
                items=len(recs),
                cached=sum(r["cached"] for r in recs),
                errors=sum("error" in r for r in recs),
                truncated=sum(r.get("finish") == "length" for r in recs),
                wall_s=round(out[d][1], 2),
            )
        bad = sum("error" in r for r in recs)
        log(f"{serve.name} · {d}: {len(recs)} pages, {sum(r['cached'] for r in recs)} cached, {bad} errors")
    return out


def _judge_score(text: str, lo: int, hi: int) -> float | None:
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text or "")
    if not m:
        return None
    return max(0.0, min(1.0, (float(m[1]) - lo) / (hi - lo)))


async def _judge(client: Client, metric: Metric, judge: Serve, sem: asyncio.Semaphore, its, recs) -> dict[str, list]:
    async def one(it: Item, png: Path, rec: dict):
        async with sem:
            text = metric.prompt.replace("{reply}", rec.get("text") or "(empty reply)")
            r = await client.complete(judge, png, text, item=f"judge:{metric.name}:{it.id}")
            return it.id, _judge_score(r.get("text"), *metric.scale)

    with span("bench.judge", metric=metric.name, judge=judge.model, items=len(recs)) as js:
        pairs = await asyncio.gather(*(one(it, png, rec) for (it, png), rec in zip(its, recs)))
        scored = {i: [s, 1] for i, s in pairs if s is not None}
        record(
            js,
            scored=len(scored),
            score=round(100 * sum(v for v, _ in scored.values()) / len(scored), 2) if scored else None,
        )
        return scored


def run(bench: Path, only: list[str] | None, datasets: list[str] | None, split: str, refresh: bool, log=print) -> Path:
    """One event loop for the whole run: the pacing locks and HTTP clients are bound to it."""
    return asyncio.run(_run(bench, only, datasets, split, refresh, log))


async def _run(bench: Path, only, datasets, split: str, refresh: bool, log) -> Path:
    configure()
    home = Settings().home
    ev, serves, metrics = bench_files(bench)
    names = expand(only, bench, serves)
    ds_names = datasets or list(ev.datasets)
    todo = pages(ev, ds_names, split, home)
    client = Client(DiskCache(home / "cache.sqlite"), refresh)
    started = datetime.now(UTC)
    replies: dict[str, dict] = {}

    async def remote():
        got = await asyncio.gather(*(_predict(client, serves[n], todo, log) for n in names if not serves[n].local))
        return dict(zip([n for n in names if not serves[n].local], got))

    with span("bench.eval", eval=ev.name, split=split, serves=names, datasets=ds_names) as root:
        replies |= await remote()
        for n in [n for n in names if serves[n].local]:
            every = all(client.cached(serves[n], png) for its in todo.values() for _, png in its)
            with serving.served(serves[n]) if not every else contextlib.nullcontext():  # no GPU when all is cached
                replies[n] = await _predict(client, serves[n], todo, log)
        judges = {m.name: (m, load(m.judge)) for m in metrics}
        keys = [(n, d, m) for n in names for d in ds_names for m in judges]
        sems = {m: asyncio.Semaphore(j.request.concurrency) for m, (_, j) in judges.items()}  # one per judge
        judged = dict(
            zip(
                keys,
                await asyncio.gather(
                    *(_judge(client, *judges[m], sems[m], todo[d], replies[n][d][0]) for n, d, m in keys)
                ),
            )
        )
        perf, acc, summary, reply_rows, unit_rows = [], [], [], [], []
        for n in names:
            s = serves[n]
            graded = []
            for d in ds_names:
                spec, its = ev.datasets[d], todo[d]
                recs, wall = replies[n][d]
                perf.append(performance(s, d, recs, wall))
                texts = {it.id: r.get("text") or "" for (it, _), r in zip(its, recs)}
                with span("bench.grade", dataset=d, serve=n, grader=spec.kind, config_id=s.config_id) as gs:
                    g = DATASETS[spec.kind].score(spec, [it for it, _ in its], texts)
                    record(
                        gs, score=round(g["score"], 2), **{f"metric.{k}": round(v, 2) for k, v in g["metrics"].items()}
                    )
                graded.append((g["score"], g["units"]))
                lo, hi = interval([(g["score"], g["units"])])
                base = {"model_id": s.model, "config_id": s.config_id, "serve": n, "dataset": d}
                acc.append(
                    base
                    | {
                        "metric": spec.kind,
                        "score": round(g["score"], 2),
                        "ci95": [lo, hi],
                        "items": len(its),
                        "missing": sum(1 for r in recs if "error" in r or not r.get("text")),
                        "by_category": {k: round(v, 1) for k, v in g["by_category"].items()},
                        "metrics": {k: round(v, 1) for k, v in g["metrics"].items()},
                    }
                )
                unit_rows += [base | {"item": i, "metric": spec.kind, "units": u} for i, u in g["units"].items()]
                for m, judge in judges.values():
                    ju = judged[n, d, m.name]
                    js = 100 * sum(v for v, _ in ju.values()) / len(ju) if ju else 0.0
                    jlo, jhi = interval([(js, {i: {"judge": u} for i, u in ju.items()})])
                    acc.append(
                        base
                        | {
                            "metric": m.name,
                            "score": round(js, 2),
                            "ci95": [jlo, jhi],
                            "items": len(its),
                            "missing": len(its) - len(ju),
                            "judge": judge.model,
                            "self_judged": judge.model == s.model,
                        }
                    )
                    unit_rows += [base | {"item": i, "metric": m.name, "units": {"judge": u}} for i, u in ju.items()]
                reply_rows += [
                    base | {"item": it.id, **{k: v for k, v in r.items() if k != "text"}, "text": r.get("text")}
                    for (it, _), r in zip(its, recs)
                ]
            lo, hi = interval(graded)
            rows = [p for p in perf if p["config_id"] == s.config_id]
            summary.append(
                {
                    "model_id": s.model,
                    "config_id": s.config_id,
                    "serve": n,
                    "mean_score": round(sum(x for x, _ in graded) / len(graded), 2),
                    "ci95": [lo, hi],
                    "items_per_s": _mean([p["items_per_s"] for p in rows]),
                    "usd_per_1k_items": _mean([p["usd_per_1k_items"] for p in rows]),
                    "datasets": len(graded),
                }
            )
        record(root, **{f"mean.{x['serve']}": x["mean_score"] for x in summary})
    doc = {
        "schema": SCHEMA,
        "run": {
            "id": f"{started:%Y%m%dT%H%M%SZ}-{ev.name}-{split}",
            "eval": ev.name,
            "split": split,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "vlmbench": __version__,
            "host": {"node": platform.node(), "python": platform.python_version()},
            "datasets": {d: ev.datasets[d].model_dump() for d in ds_names},
        },
        "configs": [_config(serves[n]) for n in names],
        "performance": perf,
        "accuracy": acc,
        "summary": summary,
    }
    return write(home / "evals" / ev.name / doc["run"]["id"], doc, reply_rows, unit_rows)


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _config(s: Serve) -> dict:
    d = s.model_dump(mode="json")
    d["backend"].pop("api_key", None)
    return {"config_id": s.config_id, "model_id": s.model, "serve": s.name} | d


def board(bench: Path, split: str, log=print) -> list[dict]:
    """Per serve, each dataset's latest score; the mean (and its interval) only when every dataset has one;
    throughput and cost from each dataset's most complete live pass. Best mean first."""
    from vlmbench.evals.results import latest

    ev, serves, metrics = bench_files(bench)
    rows, kinds = latest(Settings().home / "evals" / ev.name), {d: s.kind for d, s in ev.datasets.items()}
    out = []
    for n, s in serves.items():
        acc = {(k[2], k[4]): v for k, v in rows.items() if k[0] == "acc" and k[1] == s.config_id and k[3] == split}
        bench_acc = {d: acc[d, kind] for d, kind in kinds.items() if (d, kind) in acc}
        if not bench_acc:
            continue
        perf = [v for k, v in rows.items() if k[0] == "perf" and k[1] == s.config_id and k[3] == split]
        judged = [v["score"] for (d, m), v in acc.items() if m != kinds.get(d)]
        full = len(bench_acc) == len(kinds)
        ci = interval([(v["score"], v["units"]) for v in bench_acc.values()]) if full else None
        out.append(
            {
                "serve": n.removeprefix(f"{ev.name}-"),
                "mean": round(sum(v["score"] for v in bench_acc.values()) / len(kinds), 1) if full else None,
                "ci95": "{:.0f}–{:.0f}".format(*ci) if ci else None,
                **{d: v["score"] for d, v in bench_acc.items()},
                "judge": round(sum(judged) / len(judged), 1) if judged else None,
                "pages/s": _mean([p["items_per_s"] for p in perf]),
                "$/1k pages": _mean([p["usd_per_1k_items"] for p in perf]),
            }
        )
    return sorted(out, key=lambda r: -(r["mean"] or 0))


def dump(path: Path) -> str:
    return json.dumps(json.loads(path.read_text())["summary"], indent=1)
