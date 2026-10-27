"""JSON replies: parsed leniently (fences, prose around the object), and a document's windows merged into one."""

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse(text: str | None) -> dict | None:
    """The first JSON object in a reply, or None."""
    if not text:
        return None
    for candidate in (text, *_FENCE.findall(text)):
        try:
            v = json.loads(candidate)
            return v if isinstance(v, dict) else {"value": v}
        except (ValueError, TypeError, RecursionError):  # also a runaway reply: a 16,000-digit number, deep nesting
            pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        v, _ = json.JSONDecoder().raw_decode(text[start:])
        return v if isinstance(v, dict) else None
    except (ValueError, RecursionError):
        return None


def _empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def merge(a, b):
    """`b` (a later window) folded into `a`: arrays concatenate, objects merge key by key, a scalar keeps the
    first non-empty value."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if k in out else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return b if _empty(a) else a
