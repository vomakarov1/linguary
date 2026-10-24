"""Normalization of false truncation signals at the end of a translation."""

import re


_COLON_DASH = re.compile(r":\s*[—–-]\s*$")
_EXTRA_DOTS = re.compile(r"\.{4,}\s*$")


def normalized_tail(text: str) -> str:
    """Removes trailing dash after a colon and collapses four or more dots into an ellipsis."""
    text = _COLON_DASH.sub(":", text)
    return _EXTRA_DOTS.sub("…", text)


__all__ = ["normalized_tail"]
