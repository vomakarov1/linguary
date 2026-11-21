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

#: Frequent English words that leak into Russian NOTE prose when left unmarked between Russian words.
_LEAK_EN_WORDS = frozenset(
    {
        "very",
        "strong",
        "really",
        "just",
        "means",
        "shows",
        "because",
        "when",
        "here",
        "also",
        "only",
        "always",
        "never",
        "something",
        "someone",
    }
)

#: Closed whitelist of grammar/linguary terms permitted unmarked in Russian notes (P2.T1).
_NOTE_WHITELIST = re.compile(
    r"(?:"
    r"\b(?:Past Perfect|Present Simple|Present Continuous|Past Simple|"
    r"Past Continuous|Future Simple|Present Perfect|"
    r"Modal|Infinitive|UK|US|IPA|[ABC][12]|to)\b"
    r"|(?<!\w)-(?:ing|ed|s)\b"
    r")",
    re.IGNORECASE,
)

#: Marked quotation / italic / code spans inside notes.
_MARKED_SPAN = re.compile(
    r"«([^«»\n]+)»"
    r'|"([^"\n]+)"'
    r"|“([^“”\n]+)”"
    r"|(?<!\w)‘((?:[^’\n]|(?<=\w)’(?=\w))+)’(?!\w)"
    r"|(?<!\w)'((?:[^'\n]|(?<=\w)'(?=\w))+)'(?!\w)"
    r"|\*([^*\n]+)\*"
    r"|(?<!\w)_([^_\n]+)_(?!\w)"
    r"|`([^`\n]+)`"
)

#: Parenthetical glosses and grammar formula tails (`+ инфинитив`) allowed inside quoted patterns.
_PARENTHETICAL_GLOSS = re.compile(r"\([^)]*\)")
_FORMULA_CYRILLIC_TAIL = re.compile(r"\+\s*[А-ЯЁа-яё-]+")


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


def note_language_leak_violation(
    payload: TranslationResultResponse,
    *,
    target_language: str,
) -> str | None:
    """Asymmetric check for language leaks inside NOTE texts (P2.T1).

    Strictly catches:
    1. Cyrillic words leaked inside English quotes of >= 2 words
       (e.g. `«was strongly reminded of Дадли»`);
    2. Unmarked (outside quotes/italics) frequent English words between Russian words,
       respecting the closed whitelist of grammar/linguary terms.
    """
    if target_language.strip().lower().split("-")[0] != "ru":
        return None
    for subject, text in _note_texts_of(payload):
        leak = _note_text_leak(text)
        if leak is not None:
            return f"{subject} содержит языковое просачивание: {leak}"
    return None


def _note_text_leak(text: str) -> str | None:
    """Returns a Russian description of the first language leak in one note text, or `None`."""
    marked_spans: list[tuple[int, int]] = []
    for match in _MARKED_SPAN.finditer(text):
        marked_spans.append(match.span())
        inner = next((group for group in match.groups() if group is not None), "")
        if _has_cyrillic_in_english_quote(inner):
            return (
                f"кириллица внутри английской цитаты {inner!r}: "
                "цитату оригинала пиши полностью на английском, а перевод — отдельно на русском"
            )

    whitelist_spans = [match.span() for match in _NOTE_WHITELIST.finditer(text)]
    tokens = list(_LETTERS.finditer(text))
    cyrillic_tokens = [
        token
        for token in tokens
        if _is_cyrillic_word(token.group())
    ]
    if len(cyrillic_tokens) < 2:
        return None
    first_cyr_end = cyrillic_tokens[0].end()
    last_cyr_start = cyrillic_tokens[-1].start()

    for token in tokens:
        start, end = token.span()
        if start < first_cyr_end or end > last_cyr_start:
            continue
        word = token.group()
        if not _is_latin_word(word):
            continue
        if _overlaps_any(start, end, marked_spans) or _overlaps_any(start, end, whitelist_spans):
            continue
        if word.casefold() in _LEAK_EN_WORDS:
            return (
                f"немаркированное английское слово {word!r} между русскими словами: "
                "пиши пояснение по-русски, а цитируемые английские слова заключай в кавычки"
            )
    return None


def _has_cyrillic_in_english_quote(inner: str) -> bool:
    """Whether a quoted/italicized English phrase of >= 2 words contains leaked Cyrillic."""
    cleaned = _PARENTHETICAL_GLOSS.sub(" ", inner)
    cleaned = _FORMULA_CYRILLIC_TAIL.sub(" ", cleaned)
    cleaned = _NOTE_WHITELIST.sub(" ", cleaned)
    latin_words = 0
    cyrillic_words = 0
    for match in _LETTERS.finditer(cleaned):
        word = match.group()
        if _is_latin_word(word):
            latin_words += 1
        elif _is_cyrillic_word(word):
            cyrillic_words += 1
    return latin_words >= 2 and cyrillic_words >= 1


def _is_latin_word(word: str) -> bool:
    return bool(word) and all(_is_latin(ch) for ch in word)


def _is_cyrillic_word(word: str) -> bool:
    return bool(word) and all(_is_cyrillic(ch) for ch in word)


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start < end and start < span_end for span_start, span_end in spans)


def _note_texts_of(payload: TranslationResultResponse) -> Iterator[tuple[str, str]]:
    """Note-bearing texts scoped to `payload.blocks` when present, or all notes/assistances otherwise."""
    if payload.blocks:
        for block in payload.blocks:
            subject = f"примечание блока {block.source_block_id}"
            for hint in block.glossary_hints:
                if hint.note is not None:
                    yield subject, hint.note
            for note in payload.notes_for(block.source_block_id):
                yield subject, note.note
                if note.translated_excerpt is not None:
                    yield subject, note.translated_excerpt
        return
    for note in payload.language_notes:
        subject = f"примечание блока {note.source_block_id}"
        yield subject, note.note
        if note.translated_excerpt is not None:
            yield subject, note.translated_excerpt
    for assistance in payload.fragment_assistances:
        if assistance.text is not None:
            yield f"помощь к трудности {assistance.difficulty_id}", assistance.text


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


__all__ = [
    "mixed_script_violation",
    "mixed_script_word",
    "note_language_leak_violation",
]
