#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0", "huggingface_hub>=0.20", "Pillow>=10", "pandas>=2", "rich>=13"]
# ///
"""
Build the vlm-run/FineVision-vlmbench dataset.

Curates 128 rows from HuggingFaceM4/FineVision:
  - 32 single-image natural images   (General VQA)
  - 32 multi-image examples          (multi-image subsets)
  - 32 dense text / OCR              (OCR / text-heavy)
  - 32 document with rich structure   (DocVQA / structured docs)

Usage:
    uv run scripts/build_finevision_vlmbench.py            # build + show distribution
    uv run scripts/build_finevision_vlmbench.py --push      # build + push to HF
    uv run scripts/build_finevision_vlmbench.py --dry-run   # preview configs only
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Category → FineVision config mapping
# We pick diverse subsets for each category to maximise variety.
# ---------------------------------------------------------------------------

CATEGORY_CONFIGS: dict[str, list[str]] = {
    # General VQA — natural photos with visual questions
    "natural_image": [
        "vqav2",
        "aokvqa",
        "ok_vqa",
        "gqa",
        "visual7w",
        "allava_laion",
        "allava_vflan",
        "clevr",
        "super_clevr_mathv360k_",
        "tallyqa",
        "a_okvqa",
    ],
    # Multi-image — subsets known to have >1 image per row
    "multi_image": [
        "spot_the_diff",
        "slidevqa",
        "birds_to_words",
        "coinstruct",
        "contrastive_caption",
        "dreamsim",
        "iconqa",
        "nlvr2",
        "multi30k",
        "visual_story_telling",
    ],
    # Dense text / OCR — text-heavy images
    "ocr": [
        "textvqa",
        "textcaps",
        "st_vqa",
        "synthdog",
        "tal_ocr_eng",
        "textocr_gpt4v_",
        "ocr_vqa",
        "iiit5k",
        "llavar_instruct",
        "rendered_text",
    ],
    # Documents with rich structure — invoices, forms, tables, PDFs
    "document": [
        "docvqa",
        "infovqa",
        "tat_dqa",
        "sroie",
        "tab_fact",
        "tabmwp",
        "wtq",
        "deepform",
        "kleister_charity",
        "funsd",
        "cord",
    ],
}

ROWS_PER_CATEGORY = 32
TARGET_REPO = "vlm-run/FineVision-vlmbench"
SOURCE_DATASET = "HuggingFaceM4/FineVision"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class SampledRow:
    """One curated row for our benchmark dataset."""

    images: list[PILImage.Image]
    prompt: str
    response: str
    category: str
    source: str  # FineVision config name
    num_images: int
    image_height: int  # first image
    image_width: int  # first image
    num_turns: int
    prompt_length: int
    response_length: int


def image_hash(img: PILImage.Image) -> str:
    """Quick perceptual hash for dedup."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()[:12]


def extract_row(raw: dict, category: str, config_name: str) -> SampledRow | None:
    """Convert a FineVision row to our schema. Returns None if unusable."""
    images = raw.get("images") or []
    texts = raw.get("texts") or []

    # Filter out rows with no images or no text
    if not images or not texts:
        return None

    # Ensure images are PIL Image objects
    valid_images = []
    for img in images:
        if img is None:
            continue
        if isinstance(img, PILImage.Image):
            valid_images.append(img)
    if not valid_images:
        return None

    # For multi_image category, require >1 image
    if category == "multi_image" and len(valid_images) < 2:
        return None

    # For non-multi_image, take only single image rows (or first image)
    if category != "multi_image" and len(valid_images) > 1:
        valid_images = [valid_images[0]]

    # Extract prompt/response from texts (list of dicts with 'user'/'assistant')
    prompt_parts = []
    response_parts = []
    for turn in texts:
        if isinstance(turn, dict):
            if "user" in turn and turn["user"]:
                prompt_parts.append(turn["user"])
            if "assistant" in turn and turn["assistant"]:
                response_parts.append(turn["assistant"])

    prompt = prompt_parts[0] if prompt_parts else ""
    response = "\n".join(response_parts) if response_parts else ""

    if not prompt:
        return None

    first_img = valid_images[0]
    w, h = first_img.size

    return SampledRow(
        images=valid_images,
        prompt=prompt,
        response=response,
        category=category,
        source=config_name,
        num_images=len(valid_images),
        image_height=h,
        image_width=w,
        num_turns=len(texts),
        prompt_length=len(prompt),
        response_length=len(response),
    )


def try_config(config_name: str, category: str, needed: int, max_stream: int = 200) -> list[SampledRow]:
    """Stream from a FineVision config and sample rows."""
    console.print(f"  [dim]Streaming {config_name}...[/dim]", end=" ")
    try:
        ds = load_dataset(
            SOURCE_DATASET,
            name=config_name,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        console.print(f"[red]SKIP[/red] ({e.__class__.__name__})")
        return []

    candidates: list[SampledRow] = []
    seen_hashes: set[str] = set()

    for i, raw in enumerate(ds):
        if i >= max_stream:
            break
        row = extract_row(raw, category, config_name)
        if row is None:
            continue

        # Dedup by image hash
        h = image_hash(row.images[0])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        candidates.append(row)

    # Sample for diversity (don't just take the first N)
    if len(candidates) > needed:
        # Stratified: pick evenly across the stream
        step = len(candidates) // needed
        candidates = [candidates[i * step] for i in range(needed)]

    console.print(f"[green]{len(candidates)} rows[/green]")
    return candidates


def collect_category(category: str, configs: list[str]) -> list[SampledRow]:
    """Collect ROWS_PER_CATEGORY rows for one category, spread across configs."""
    console.print(f"\n[bold cyan]Category: {category}[/bold cyan] (target: {ROWS_PER_CATEGORY} rows)")
    all_rows: list[SampledRow] = []
    remaining = ROWS_PER_CATEGORY

    # Distribute evenly across configs, then fill from extras
    per_config = max(1, ROWS_PER_CATEGORY // len(configs))

    for config_name in configs:
        if remaining <= 0:
            break
        want = min(per_config, remaining)
        rows = try_config(config_name, category, needed=want, max_stream=300)
        all_rows.extend(rows[:want])
        remaining -= len(rows[:want])

    # If still short, try remaining configs with larger batches
    if remaining > 0:
        for config_name in configs:
            if remaining <= 0:
                break
            rows = try_config(config_name, category, needed=remaining, max_stream=500)
            new_rows = [r for r in rows if r not in all_rows]
            all_rows.extend(new_rows[:remaining])
            remaining -= len(new_rows[:remaining])

    console.print(f"  [bold]Collected {len(all_rows)}/{ROWS_PER_CATEGORY} rows[/bold]")
    return all_rows[:ROWS_PER_CATEGORY]


def rows_to_dataset(rows: list[SampledRow]) -> Dataset:
    """Convert sampled rows into a HuggingFace Dataset."""
    data = {
        "images": [r.images for r in rows],
        "prompt": [r.prompt for r in rows],
        "response": [r.response for r in rows],
        "category": [r.category for r in rows],
        "source": [r.source for r in rows],
        "num_images": [r.num_images for r in rows],
        "image_height": [r.image_height for r in rows],
        "image_width": [r.image_width for r in rows],
        "num_turns": [r.num_turns for r in rows],
        "prompt_length": [r.prompt_length for r in rows],
        "response_length": [r.response_length for r in rows],
    }

    features = Features(
        {
            "images": Sequence(Image()),
            "prompt": Value("string"),
            "response": Value("string"),
            "category": Value("string"),
            "source": Value("string"),
            "num_images": Value("int32"),
            "image_height": Value("int32"),
            "image_width": Value("int32"),
            "num_turns": Value("int32"),
            "prompt_length": Value("int32"),
            "response_length": Value("int32"),
        }
    )

    return Dataset.from_dict(data, features=features)


def show_distribution(ds: Dataset) -> None:
    """Print a full distribution analysis."""
    import pandas as pd

    df = pd.DataFrame(ds.remove_columns("images"))

    console.print("\n[bold underline]Dataset Distribution[/bold underline]\n")

    # Overall stats
    console.print(f"  Total rows: [bold]{len(df)}[/bold]")
    console.print(f"  Categories: {df['category'].nunique()}")
    console.print(f"  Sources: {df['source'].nunique()}")

    # Category breakdown
    t = Table(title="By Category", show_lines=True)
    t.add_column("Category", style="cyan")
    t.add_column("Count", justify="right")
    t.add_column("Sources", justify="right")
    t.add_column("Avg Images", justify="right")
    t.add_column("Avg Prompt Len", justify="right")
    t.add_column("Avg Response Len", justify="right")
    for cat, g in df.groupby("category"):
        t.add_row(
            str(cat),
            str(len(g)),
            str(g["source"].nunique()),
            f"{g['num_images'].mean():.1f}",
            f"{g['prompt_length'].mean():.0f}",
            f"{g['response_length'].mean():.0f}",
        )
    console.print(t)

    # Source breakdown
    t2 = Table(title="By Source", show_lines=True)
    t2.add_column("Source", style="green")
    t2.add_column("Category", style="cyan")
    t2.add_column("Count", justify="right")
    t2.add_column("Avg W×H", justify="right")
    t2.add_column("Avg Imgs", justify="right")
    for (src, cat), g in df.groupby(["source", "category"]):
        t2.add_row(
            str(src),
            str(cat),
            str(len(g)),
            f"{g['image_width'].mean():.0f}×{g['image_height'].mean():.0f}",
            f"{g['num_images'].mean():.1f}",
        )
    console.print(t2)

    # Image dimension distribution
    t3 = Table(title="Image Dimensions", show_lines=True)
    t3.add_column("Stat", style="bold")
    t3.add_column("Width", justify="right")
    t3.add_column("Height", justify="right")
    for stat in ["min", "mean", "median", "max", "std"]:
        if stat == "median":
            w_val = df["image_width"].median()
            h_val = df["image_height"].median()
        elif stat == "std":
            w_val = df["image_width"].std()
            h_val = df["image_height"].std()
        else:
            w_val = getattr(df["image_width"], stat)()
            h_val = getattr(df["image_height"], stat)()
        t3.add_row(stat, f"{w_val:.0f}", f"{h_val:.0f}")
    console.print(t3)

    # Prompt/response length distribution
    t4 = Table(title="Text Lengths", show_lines=True)
    t4.add_column("Stat", style="bold")
    t4.add_column("Prompt", justify="right")
    t4.add_column("Response", justify="right")
    for stat in ["min", "mean", "median", "max"]:
        if stat == "median":
            p_val = df["prompt_length"].median()
            r_val = df["response_length"].median()
        else:
            p_val = getattr(df["prompt_length"], stat)()
            r_val = getattr(df["response_length"], stat)()
        t4.add_row(stat, f"{p_val:.0f}", f"{r_val:.0f}")
    console.print(t4)


def discover_available_configs() -> list[str]:
    """List all available FineVision configs."""
    from datasets import get_dataset_config_names

    console.print("[bold]Discovering FineVision configs...[/bold]")
    configs = get_dataset_config_names(SOURCE_DATASET)
    console.print(f"  Found [bold]{len(configs)}[/bold] configs")
    return configs


def resolve_configs(available: list[str]) -> dict[str, list[str]]:
    """Resolve our desired configs against what's actually available.

    FineVision config names can be slightly different from what we expect
    (e.g., underscores, suffixes). Do fuzzy matching.
    """
    available_set = set(available)
    resolved: dict[str, list[str]] = {}

    for category, desired in CATEGORY_CONFIGS.items():
        found = []
        for name in desired:
            if name in available_set:
                found.append(name)
            else:
                # Try fuzzy: check if any available config contains the name
                matches = [c for c in available if name.rstrip("_") in c]
                if matches:
                    found.append(matches[0])
                    console.print(f"  [yellow]Fuzzy match: {name} → {matches[0]}[/yellow]")
        resolved[category] = found
        console.print(f"  {category}: {len(found)}/{len(desired)} configs matched")

    return resolved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build FineVision-vlmbench dataset")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--dry-run", action="store_true", help="Only discover configs, don't download")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-local", type=str, default=None, help="Save dataset locally to this path")
    args = parser.parse_args()

    random.seed(args.seed)

    console.print("[bold magenta]Building FineVision-vlmbench[/bold magenta]")
    console.print(f"  Source: {SOURCE_DATASET}")
    console.print(f"  Target: {TARGET_REPO}")
    console.print(f"  Rows: {ROWS_PER_CATEGORY} × {len(CATEGORY_CONFIGS)} = {ROWS_PER_CATEGORY * len(CATEGORY_CONFIGS)}")

    # Step 1: Discover available configs
    available = discover_available_configs()

    # Step 2: Resolve our config names against available ones
    resolved = resolve_configs(available)

    if args.dry_run:
        console.print("\n[bold yellow]Dry run — stopping before download.[/bold yellow]")
        for cat, configs in resolved.items():
            console.print(f"  [cyan]{cat}[/cyan]: {configs}")
        sys.exit(0)

    # Step 3: Collect samples from each category
    all_rows: list[SampledRow] = []
    for category, configs in resolved.items():
        if not configs:
            console.print(f"[red]No configs found for {category}![/red]")
            continue
        rows = collect_category(category, configs)
        all_rows.extend(rows)

    if not all_rows:
        console.print("[red]No rows collected! Check network connectivity to HuggingFace.[/red]")
        sys.exit(1)

    # Shuffle for good measure
    random.shuffle(all_rows)

    # Step 4: Build the HF Dataset
    console.print("\n[bold]Building HuggingFace Dataset...[/bold]")
    ds = rows_to_dataset(all_rows)
    console.print(f"  Dataset: {ds}")

    # Step 5: Show distribution
    show_distribution(ds)

    # Step 6: Save locally if requested
    if args.save_local:
        save_path = Path(args.save_local)
        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))
        console.print(f"\n[green]Saved locally to {save_path}[/green]")

    # Step 7: Push to HuggingFace Hub
    if args.push:
        console.print(f"\n[bold]Pushing to {TARGET_REPO} (private)...[/bold]")
        api = HfApi()
        try:
            api.create_repo(TARGET_REPO, repo_type="dataset", private=True, exist_ok=True)
        except Exception as e:
            console.print(f"[yellow]Repo creation note: {e}[/yellow]")

        ds.push_to_hub(TARGET_REPO, private=True)
        console.print(f"[bold green]Pushed to https://huggingface.co/datasets/{TARGET_REPO}[/bold green]")
    else:
        console.print(
            f"\n[yellow]Dataset ready. Run with --push to upload to {TARGET_REPO}.[/yellow]"
        )


if __name__ == "__main__":
    main()
