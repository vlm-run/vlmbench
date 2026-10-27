"""Logfire tracing when `vlmbench[logfire]` is installed and a token is set; a no-op otherwise."""

from contextlib import nullcontext

try:
    import logfire
except ImportError:
    logfire = None


def configure() -> None:
    if logfire:
        logfire.configure(service_name="vlmbench", send_to_logfire="if-token-present", console=False)
        logfire.instrument_openai()


def span(msg: str, **attrs):
    return logfire.span(msg, **attrs) if logfire else nullcontext()
