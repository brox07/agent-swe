"""Turning exceptions into something a job record can usefully carry."""

from __future__ import annotations


def describe(exc: BaseException) -> str:
    """A failure string that is never blank.

    httpx's timeout exceptions stringify to the empty string, so a job killed by
    a Qdrant read timeout recorded `error = ""` and looked like a mystery. The
    type name alone is enough to point at the cause.
    """
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
