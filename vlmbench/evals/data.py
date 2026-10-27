"""Pages: every benchmark item rendered once to a PNG, and the `test` / `dev` samples drawn from them."""

from dataclasses import dataclass
from pathlib import Path

LONG_EDGE = 2048


@dataclass(frozen=True)
class Item:
    id: str
    """`<dataset>/<the benchmark's own id>`."""
    category: str
    source: str
    """The PDF or image the benchmark ships."""
    page: int = 0
    doc: bool = False
    """A whole document (every page sent), not one page."""


def sample(items: list[Item], n: int, split: str) -> list[Item]:
    """`test`: `n` items spread over the categories in turn, evenly spaced within each, the same every run.
    `dev`: the next `n` the same way from what `test` left, so tuning never sees a reported page.
    `n` of 0 is every item (all in `test`)."""
    if n <= 0:
        return items if split == "test" else []
    test = _spread(items, n)
    if split == "test":
        return test
    taken = {it.id for it in test}
    return _spread([it for it in items if it.id not in taken], n)


def _spread(items: list[Item], n: int) -> list[Item]:
    by: dict[str, list[Item]] = {}
    for it in items:
        by.setdefault(it.category, []).append(it)
    k = -(-n // max(len(by), 1))
    cols = [[xs[i * len(xs) // min(k, len(xs))] for i in range(min(k, len(xs)))] for xs in by.values()]
    return [x for i in range(k) for xs in cols if i < len(xs) for x in [xs[i]]][:n]


def render_doc(item: Item, root: Path, long_edge: int = LONG_EDGE) -> list[Path]:
    """Every page of a document as a PNG (`<id>/p0001.png`, ...), written once."""
    import pypdfium2 as pdfium

    folder = root / item.id.replace("/", "__") / str(long_edge)
    done = folder / "pages.txt"
    if done.exists():
        return [folder / n for n in done.read_text().split()]
    folder.mkdir(parents=True, exist_ok=True)
    pdf, names = pdfium.PdfDocument(item.source), []
    for i in range(len(pdf)):
        page = pdf[i]
        img = page.render(scale=long_edge / max(page.get_size())).to_pil().convert("RGB")
        names.append(f"p{i + 1:04d}.png")
        img.save(folder / names[-1], "PNG")
    pdf.close()
    done.write_text(" ".join(names))
    return [folder / n for n in names]


def render(item: Item, root: Path) -> Path:
    """The page as a PNG with a 2,048 px long edge (PDFs via pdfium, images only downsized), written once."""
    from PIL import Image

    out = root / (item.id.replace("/", "__") + ".png")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    if item.source.lower().endswith(".pdf"):
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(item.source)
        p = pdf[item.page]
        img = p.render(scale=LONG_EDGE / max(p.get_size())).to_pil()
        pdf.close()
    else:
        img = Image.open(item.source)
        img.thumbnail((LONG_EDGE, LONG_EDGE))
    tmp = out.with_suffix(".tmp")
    img.convert("RGB").save(tmp, "PNG")
    tmp.replace(out)
    return out
