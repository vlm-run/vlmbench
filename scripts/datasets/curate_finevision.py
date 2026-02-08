# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0", "huggingface-hub>=0.20", "Pillow>=10", "rich>=13"]
# ///
"""Curate a small, diverse VLM benchmark dataset from HuggingFaceM4/FineVision.

Usage:
    uv run scripts/datasets/curate_finevision.py --dry-run   # preview stats only
    uv run scripts/datasets/curate_finevision.py              # push to HuggingFace
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Config: which FineVision subsets to sample per category
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[tuple[str, int]]] = {
    "natural": [
        ("LLaVA_Instruct_150K", 6),
        ("a_okvqa", 5),
        ("cocoqa", 5),
        ("localized_narratives", 5),
        ("lvis_instruct4v", 5),
        ("sharegpt4v(coco)", 6),
    ],
    "multi_image": [
        ("nlvr2", 11),
        ("DoclingMatix", 11),
        ("slidevqa", 10),
    ],
    "ocr": [
        ("ocrvqa", 6),
        ("st_vqa", 6),
        ("cocotext", 5),
        ("rendered_text", 5),
        ("iiit5k", 5),
        ("olmOCR-mix-0225-documents", 5),
    ],
    "document": [
        ("docvqa", 6),
        ("dvqa", 5),
        ("funsd", 4),
        ("infographic_vqa", 5),
        ("invoices_receipts", 4),
        ("chartqa", 4),
        ("CoSyn_400k_document", 4),
    ],
}

MIN_DIM = 256  # minimum min(width, height) to accept
SEED = 42
DEFAULT_REPO = "vlm-run/FineVision-vlmbench-mini"

# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------


def reservoir_sample[T](
    stream,
    n: int,
    max_scan: int,
    predicate=None,
    rng: random.Random | None = None,
) -> list[T]:
    """Reservoir-sample *n* items from *stream*, scanning at most *max_scan* total rows."""
    rng = rng or random.Random(SEED)
    reservoir: list[T] = []
    matched = 0
    for total, item in enumerate(stream, 1):
        if total > max_scan:
            break
        if predicate and not predicate(item):
            continue
        matched += 1
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, matched - 1)
            if j < n:
                reservoir[j] = item
    return reservoir


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _first_image_size(images: list[PILImage.Image]) -> tuple[int, int]:
    """Return (width, height) of the first image."""
    img = images[0]
    return img.size  # (w, h)


def _is_high_res(images: list[PILImage.Image]) -> bool:
    w, h = _first_image_size(images)
    return min(w, h) >= MIN_DIM


def transform_row(row: dict[str, Any], config: str, category: str) -> dict[str, Any]:
    """Convert a raw FineVision row into the output schema."""
    images: list[PILImage.Image] = row["images"]
    # Normalize to RGB
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    texts = row.get("texts") or []
    prompt = texts[0]["user"] if texts else ""
    response = texts[0]["assistant"] if texts else ""

    w, h = _first_image_size(images)

    return {
        "images": images,
        "prompt": prompt,
        "response": response,
        "image_height": h,
        "image_width": w,
        "num_images": len(images),
        "source": config,
        "category": category,
        "prompt_length": len(prompt),
        "response_length": len(response),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_from_config(
    config: str,
    n: int,
    category: str,
    *,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Stream *n* diverse rows from a FineVision config."""
    require_multi = category == "multi_image"
    rng = random.Random(seed + hash(config))

    console.print(f"  Streaming [cyan]{config}[/cyan] (want {n})…")
    ds = load_dataset("HuggingFaceM4/FineVision", name=config, split="train", streaming=True)

    min_dim = 128 if require_multi else MIN_DIM

    def predicate(row):
        imgs = row.get("images")
        if not imgs:
            return False
        texts = row.get("texts")
        if not texts or not texts[0].get("user"):
            return False
        if require_multi and len(imgs) <= 1:
            return False
        if not require_multi and len(imgs) != 1:
            return False
        w, h = _first_image_size(imgs)
        if min(w, h) < min_dim:
            return False
        return True

    scan_limit = max(n * 200, 1000) if require_multi else max(n * 100, 500)
    raw = reservoir_sample(ds, n=n, max_scan=scan_limit, predicate=predicate, rng=rng)
    console.print(f"    → got {len(raw)} rows")
    return [transform_row(r, config, category) for r in raw]


# ---------------------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------------------


def print_stats(rows: list[dict[str, Any]]) -> None:
    """Print a Rich summary of the curated dataset."""
    console.rule("[bold]Dataset Distribution")

    # --- Category × Source table ---
    tbl = Table(title="Rows by Category & Source", show_lines=False)
    tbl.add_column("Category", style="bold")
    tbl.add_column("Source")
    tbl.add_column("Count", justify="right")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["category"]][r["source"]] += 1

    for cat in ("natural", "multi_image", "ocr", "document"):
        sources = counts.get(cat, {})
        for src, cnt in sorted(sources.items()):
            tbl.add_row(cat, src, str(cnt))
        tbl.add_row(f"[bold]{cat} total", "", f"[bold]{sum(sources.values())}")
        tbl.add_section()

    tbl.add_row("[bold green]TOTAL", "", f"[bold green]{len(rows)}")
    console.print(tbl)

    # --- Image dimension stats ---
    dim_tbl = Table(title="Image Dimensions by Category")
    dim_tbl.add_column("Category", style="bold")
    dim_tbl.add_column("Avg W", justify="right")
    dim_tbl.add_column("Avg H", justify="right")
    dim_tbl.add_column("Min WxH", justify="right")
    dim_tbl.add_column("Max WxH", justify="right")

    for cat in ("natural", "multi_image", "ocr", "document"):
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        ws = [r["image_width"] for r in cat_rows]
        hs = [r["image_height"] for r in cat_rows]
        dim_tbl.add_row(
            cat,
            str(round(statistics.mean(ws))),
            str(round(statistics.mean(hs))),
            f"{min(ws)}x{min(hs)}",
            f"{max(ws)}x{max(hs)}",
        )

    console.print(dim_tbl)

    # --- Text length stats ---
    txt_tbl = Table(title="Text Length by Category")
    txt_tbl.add_column("Category", style="bold")
    txt_tbl.add_column("Avg Prompt", justify="right")
    txt_tbl.add_column("Max Prompt", justify="right")
    txt_tbl.add_column("Avg Response", justify="right")
    txt_tbl.add_column("Max Response", justify="right")

    for cat in ("natural", "multi_image", "ocr", "document"):
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        pls = [r["prompt_length"] for r in cat_rows]
        rls = [r["response_length"] for r in cat_rows]
        txt_tbl.add_row(
            cat,
            str(round(statistics.mean(pls))),
            str(max(pls)),
            str(round(statistics.mean(rls))),
            str(max(rls)),
        )

    console.print(txt_tbl)

    # --- Num images stats for multi_image ---
    mi_rows = [r for r in rows if r["category"] == "multi_image"]
    if mi_rows:
        ni = [r["num_images"] for r in mi_rows]
        console.print(
            f"\n[bold]Multi-image stats:[/bold] "
            f"min={min(ni)}, max={max(ni)}, mean={statistics.mean(ni):.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(rows: list[dict[str, Any]]) -> Dataset:
    """Convert list of row dicts into a HuggingFace Dataset."""
    # Transpose row-oriented → column-oriented
    columns: dict[str, list] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            columns[k].append(v)

    features = Features(
        {
            "images": Sequence(Image()),
            "prompt": Value("string"),
            "response": Value("string"),
            "image_height": Value("int32"),
            "image_width": Value("int32"),
            "num_images": Value("int32"),
            "source": Value("string"),
            "category": Value("string"),
            "prompt_length": Value("int32"),
            "response_length": Value("int32"),
        }
    )
    return Dataset.from_dict(dict(columns), features=features)


def main(seed: int = SEED, dry_run: bool = False, repo_id: str = DEFAULT_REPO) -> None:
    console.rule(f"[bold]Curating FineVision → {repo_id}")

    all_rows: list[dict[str, Any]] = []

    for category, configs in CATEGORIES.items():
        console.print(f"\n[bold magenta]Category: {category}[/bold magenta]")
        for config, n in configs:
            try:
                rows = sample_from_config(config, n, category, seed=seed)
                all_rows.extend(rows)
            except Exception as e:
                console.print(f"  [red]FAILED {config}: {e}[/red]")

    console.print()
    print_stats(all_rows)

    if dry_run:
        console.print("\n[yellow]Dry run — not pushing.[/yellow]")
        return

    console.print(f"\n[bold]Building dataset ({len(all_rows)} rows)…[/bold]")
    ds = build_dataset(all_rows)
    console.print(ds)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        # Fall back to cached huggingface-cli token
        for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
            if p.is_file():
                token = p.read_text().strip()
                break
    if not token:
        console.print("[red]No HF token found. Run `huggingface-cli login` or set HF_TOKEN.[/red]")
        return

    # Ensure the repo exists
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    console.print(f"\n[bold]Pushing to [cyan]{repo_id}[/cyan] (private)…[/bold]")
    ds.push_to_hub(repo_id, private=True, token=token)
    console.print(f"[bold green]Done![/bold green] https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate FineVision-vlmbench-mini dataset")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without pushing")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="HF repo ID")
    args = parser.parse_args()
    main(seed=args.seed, dry_run=args.dry_run, repo_id=args.repo_id)
