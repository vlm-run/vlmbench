"""A model's native output -> the markdown every grader reads (a serve's `postprocess:`)."""

import sys
from functools import cache


@cache
def _nemotron():
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        "nvidia/Nemotron-3.5-Parse", allow_patterns=["postprocessing.py", "latex2html.py", "nemotron_parse/**"]
    )
    sys.path.insert(0, root)
    import nemotron_parse

    return nemotron_parse


def nemotron(raw: str, image: str) -> str:
    """NVIDIA's renderer for Nemotron Parse's box/class/text blocks; its page header and footer blocks are tagged
    as page furniture, as the prompt asks of every other model."""
    from PIL import Image

    doc = _nemotron().postprocess_document(raw, image_size=Image.open(image).size)
    md = doc["markdown"]
    for b in doc["blocks"]:
        kind = b["class"].lower().replace("_", "-")
        tag = {"page-header": "page_header", "page-footer": "page_footer"}.get(kind)
        if tag and (c := b["content"].strip()) and c in md:
            md = md.replace(c, f"<{tag}>{c}</{tag}>", 1)
    return md
