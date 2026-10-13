"""Built-in model mechanics — only for models with quirks worth recording once.

A bare `-m <id>` needs no entry here; defaults apply.
"""

from vlmbench.eval import Model


class GlmOcr(Model):
    """One transcription prompt upstream, so `ocr` returns the same bytes as `markdown`."""

    model = "zai-org/glm-ocr"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class LightOnOcr3(Model):
    """Upstream ships two prompts and calls anything else out of distribution."""

    model = "lightonai/lightonocr-3-4b"
    takes_prompt = False


class VlmRunOcrAuto(Model):
    model = "vlm-run/ocr:auto"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class DotsOcr(Model):
    model = "rednote-hilab/dots.mocr"
    takes_prompt = False
    extra_body = {"method": "markdown"}


class PaddleOcrVl(Model):
    model = "paddlepaddle/paddleocr-vl"
    takes_prompt = False
    extra_body = {"method": "markdown"}
