"""Validation against words mixing Latin and Cyrillic scripts (PRD § 15, ALG-04).

The model occasionally glues the original name spelling to a Russian inflectional suffix —
"Selenой Роу" instead of "Селеной Роу". Such a word belongs to neither language and offers
no value to the reader, so a result containing it is rejected and retried with clarification.

A word here is defined as a contiguous sequence of letters: digits, hyphens, apostrophes,
whitespace, and punctuation divide words. Therefore Latin names and terms in Russian
text ("Wren", "Duskmere"), footnote markers `[^1]`, addresses, numbers with units ("5kg"),
and hyphenated compounds of two words ("Wonka-фабрика", "mp3-плеер") are not violations:
in each, every individual word is written entirely in one script. Only words where letters
of both alphabets appear adjacent are rejected. The exception is a Latin word whose only
Cyrillic characters are yers "ь" and "ъ": scientific Latin uses them for Proto-Slavic
reconstructions ("vьlkodlakъ"), and a language note discussing word origins legitimately quotes them.

All result texts written by the model in the target language are inspected: translation,
glossary hint translation and explanation, language note and its translated excerpt.
Source excerpts (`original`, `original_excerpt`) are original quotes and are not checked.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse

#: Word: a continuous sequence of letters without digits or symbols.
_LETTERS = re.compile(r"[^\W\d_]+")

#: Yers — the only Cyrillic letters used in scientific Latin transliteration of Slavic forms.
_YERS = frozenset("ьъ")


def mixed_script_violation(payload: TranslationResultResponse) -> str | None:
    """Violation description of the first word with mixed Latin and Cyrillic script, or `None`."""
    for block in payload.blocks:
        for text in _texts_of(payload, block=block):
            word = mixed_script_word(text)
            if word is not None:
                return (
                    f"результат блока {block.source_block_id} содержит слово из смеси "
                    f"латиницы и кириллицы: {word!r}. Каждое слово пиши одним "
                    f"алфавитом: имя из глоссария — в его варианте перевода с русским "
                    f"окончанием, исходное написание — без русского окончания"
                )
    return None


def mixed_script_word(text: str) -> str | None:
    """First word in the text containing both Latin and Cyrillic characters."""
    for match in _LETTERS.finditer(text):
        word = match.group()
        cyrillic = {letter for letter in word if _is_cyrillic(letter)}
        if any(_is_latin(letter) for letter in word) and cyrillic - _YERS:
            return word
    return None


def _is_latin(letter: str) -> bool:
    """Whether the character is Latin, including letters with diacritics (é, ß)."""
    return letter.isascii() or "À" <= letter <= "ɏ"


def _is_cyrillic(letter: str) -> bool:
    """Whether the character is Cyrillic."""
    return "Ѐ" <= letter <= "ӿ"


def _texts_of(
    payload: TranslationResultResponse, *, block: BlockResultResponse
) -> Iterator[str]:
    """Block result texts written by the model in the target language."""
    if block.translated_text is not None:
        yield block.translated_text
    for hint in block.glossary_hints:
        yield hint.translation
        if hint.note is not None:
            yield hint.note
    for note in payload.notes_for(block.source_block_id):
        yield note.note
        if note.translated_excerpt is not None:
            yield note.translated_excerpt


__all__ = ["mixed_script_violation", "mixed_script_word"]
