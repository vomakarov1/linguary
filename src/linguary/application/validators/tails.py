"""Нормализация ложных признаков обрыва на конце перевода."""

import re


_COLON_DASH = re.compile(r":\s*[—–-]\s*$")
_EXTRA_DOTS = re.compile(r"\.{4,}\s*$")


def normalized_tail(text: str) -> str:
    """Убирает завершающее тире после двоеточия и заменяет четыре точки многоточием."""
    text = _COLON_DASH.sub(":", text)
    return _EXTRA_DOTS.sub("…", text)


__all__ = ["normalized_tail"]
