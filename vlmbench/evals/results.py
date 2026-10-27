"""The result document (`vlmbench.result/v1`): configs, performance and accuracy rows joined on (model_id, config_id),
a summary per config, and 95% page-bootstrap intervals rebuilt from per-item units."""

import json
import random
import statistics
from pathlib import Path

SCHEMA = "vlmbench.result/v1"


def _rate(units: dict[str, dict[str, list]], ids: list[str]) -> float:
    """A dataset's score from its items' units: the mean over groups of each group's pooled rate."""
    tot: dict[str, list[float]] = {}
    for i in ids:
        for g, (s, n) in units.get(i, {}).items():
            a = tot.setdefault(g, [0.0, 0.0])
            a[0] += s
            a[1] += n
    rates = [s / n for s, n in tot.values() if n]
    return 100 * sum(rates) / len(rates) if rates else 0.0


def interval(datasets: list[tuple[float, dict]], draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean over `datasets` [(reported score, units)], each dataset's items resampled; shifted
    so the full sample gives the reported score (a grader's own aggregate need not equal the pooled one)."""
    rng = random.Random(seed)
    prepared = [(score, units, list(units), _rate(units, list(units))) for score, units in datasets if units]
    if not prepared:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(draws):
        vals = [score + _rate(u, rng.choices(ids, k=len(ids))) - full for score, u, ids, full in prepared]
        means.append(sum(vals) / len(vals))
    q = statistics.quantiles(means, n=40)
    return (round(q[0], 1), round(q[-1], 1))


def _dist(xs: list[float]) -> dict | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    pick = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"mean": round(statistics.fmean(xs), 3), "p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99)}


def performance(serve, dataset: str, recs: list[dict], wall_s: float) -> dict:
    """One row per (config, dataset): timing from live calls only (a cache hit is counted, never timed)."""
    live = [r for r in recs if not r.get("cached")]
    ok = [r for r in live if "error" not in r]
    gpu_rate = getattr(serve.backend, "gpu_hour_usd", None)
    cost = sum(r.get("cost_usd") or 0 for r in ok) or None
    if gpu_rate and live:
        cost = gpu_rate * wall_s / 3600
    out_tok = sum(r.get("completion_tokens", 0) for r in ok)
    return {
        "model_id": serve.model,
        "config_id": serve.config_id,
        "serve": serve.name,
        "dataset": dataset,
        "concurrency": serve.request.concurrency,
        "items": len(recs),
        "live": len(live),
        "cached": len(recs) - len(live),
        "errors": sum("error" in r for r in recs),
        "truncated": sum(r.get("finish") == "length" for r in recs),
        "wall_s": round(wall_s, 2),
        "items_per_s": round(len(live) / wall_s, 3) if live and wall_s else None,
        "output_tok_per_s": round(out_tok / wall_s, 1) if live and wall_s else None,
        "ttft_s": _dist([r.get("ttft_s") for r in ok]),
        "latency_s": _dist([r.get("latency_s") for r in ok]),
        "prompt_tokens": _dist([r.get("prompt_tokens") for r in ok]),
        "output_tokens": _dist([r.get("completion_tokens") for r in ok]),
        "reasoning_tokens": _dist([r.get("reasoning_tokens") for r in ok]),
        "cost_usd": round(cost, 5) if cost else None,
        "usd_per_1k_items": round(1000 * cost / len(live), 3) if cost and live else None,
    }


def write(run_dir: Path, doc: dict, replies: list[dict], units: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    (run_dir / "replies.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in replies))
    (run_dir / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    return run_dir / "result.json"


def latest(root: Path) -> dict[tuple, dict]:
    """Across every run of an eval: the newest accuracy row per (config, dataset, split, metric), with its per-item
    units; and per (config, dataset, split) the performance row with the most live calls (a pass that re-called
    a few uncached pages says nothing about throughput)."""
    rows: dict[tuple, dict] = {}
    for f in sorted(root.glob("*/result.json")):
        doc = json.loads(f.read_text())
        split = doc["run"]["split"]
        units: dict[tuple, dict] = {}
        if (u := f.parent / "units.jsonl").exists():
            for line in u.read_text().splitlines():
                r = json.loads(line)
                units.setdefault((r["config_id"], r["dataset"], r["metric"]), {})[r["item"]] = r["units"]
        for a in doc["accuracy"]:
            rows[("acc", a["config_id"], a["dataset"], split, a["metric"])] = a | {
                "split": split,
                "units": units.get((a["config_id"], a["dataset"], a["metric"]), {}),
            }
        for p in doc["performance"]:
            k = ("perf", p["config_id"], p["dataset"], split)
            if p["live"] and p["live"] >= rows.get(k, {}).get("live", 0):
                rows[k] = p | {"split": split}
        for c in doc["configs"]:
            rows[("cfg", c["config_id"])] = c
    return rows
