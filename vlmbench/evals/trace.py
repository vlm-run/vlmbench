"""Logfire spans, in vlm-playground's shape: `with span("vlmbench.x", **attrs) as s:` then `record(s, **results)`.
Enabled when `vlmbench[logfire]` is installed and `LOGFIRE_TOKEN` is set; a no-op otherwise."""

import os
from contextlib import nullcontext
from typing import Any

_LOGFIRE: Any = None


def configure() -> bool:
    global _LOGFIRE
    if _LOGFIRE is not None:
        return True
    if not os.environ.get("LOGFIRE_TOKEN"):
        return False
    try:
        import logfire
    except ImportError:
        return False
    logfire.configure(service_name="vlmbench", send_to_logfire=True, console=False)
    logfire.instrument_openai()
    _LOGFIRE = logfire
    return True


def span(name: str, **attributes: Any):
    """A Logfire span, or a no-op context manager (yielding None) when tracing is off."""
    return nullcontext() if _LOGFIRE is None else _LOGFIRE.span(name, **attributes)


def record(s: Any, **attributes: Any) -> None:
    """Back-fill results onto an open span; None values and a disabled span are skipped."""
    if s is not None and hasattr(s, "set_attribute"):
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
