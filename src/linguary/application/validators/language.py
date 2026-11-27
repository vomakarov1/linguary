"""Processing result language validation (PRD § 15, ALG-04).

Language compliance is determined by the share of target alphabet letters among all letters
in the text. The check applies only to full translations longer than a configurable minimum:
a short string may legitimately consist of a proper name, and a glossary hint consists of
the source language word it was written for. A segmented block is checked once, on the
same join the stored result keeps.

The alphabet is defined only for the MVP target language — Russian (PRD § 1). For other
target languages, the share of "target alphabet characters" is undefined and validation
is skipped: this is safer than counting a foreign alphabet's share and rejecting a valid translation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.validators.mixed_script import note_language_leak_violation
from linguary.application.validators.treatment import (
    joined_segment_translation,
    planned_segment_modes,
)
from linguary.domain import (
    BlockTreatment,
    ConceptType,
    GlossarySnapshotEntry,
    TranslationUnit,
    TranslationUnitItem,
)

#: Target languages whose alphabet is known to the application. Key is the book language code without region.
_TARGET_ALPHABETS: dict[str, tuple[str, str]] = {"ru": ("Ѐ", "ӿ")}

#: Source `dozen` (12) erroneously rendered as Russian `десяток` (10).
_DOZEN_SOURCE = re.compile(r"\bdozen\b", re.IGNORECASE)
_DESYATOK_TRANSLATION = re.compile(
    r"(?<![А-ЯЁа-яё])(?:пол)?десят(?:ок|ка|ку|ком|ке|ки|ков|кам|ками|ках)(?![А-ЯЁа-яё])",
    re.IGNORECASE,
)

#: Hyphenated spelling error `что-бы`.
_CHTO_BY = re.compile(r"(?<![А-ЯЁа-яё-])что-бы(?![А-ЯЁа-яё-])", re.IGNORECASE)

#: Invalid Russian preposition `со` before sibilant + vowel (does not match consonant clusters like `со зла`, `со знанием`).
_SO_SIBILANT_VOWEL = re.compile(
    r"(?<![А-ЯЁа-яё-])(со\s+[зсжш][аеёиоуыэюя][а-яё]*)",
    re.IGNORECASE,
)

#: Indeclinable `Сам-Знаешь-Кто` kept in nominative after a Russian preposition.
_INDECLINABLE_YOU_KNOW_WHO = re.compile(
    r"(?<![А-ЯЁа-яё-])"
    r"(?:о|об|обо|от|ото|к|ко|с|со|у|про|для|без|безо|из-за|из|против|"
    r"перед|передо|над|надо|под|подо|после|вместо|кроме|ради|вокруг|"
    r"возле|около|на|за|до|при|по|в|во)"
    r"\s+сам-знаешь-кто(?![А-ЯЁа-яё-])",
    re.IGNORECASE,
)

#: Word tokens used for Russian inflection-aware mandatory glossary matching.
_TERM_WORD = re.compile(r"[А-ЯЁа-яёA-Za-z0-9]+")
_PARENS = re.compile(r"\([^)]*\)")
_CASE_SENSITIVE_CONCEPT_TYPES = frozenset(
    {ConceptType.PERSON, ConceptType.PLACE, ConceptType.ORGANIZATION}
)
_LEADING_ARTICLE = re.compile(r"^(the|an?)\s+(.+)$", re.IGNORECASE)
_ADJ_ENDINGS = (
    "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой",
    "ую", "юю", "ым", "им", "ом", "ем",
)
_IRREGULAR_RUSSIAN_FORMS: dict[str, str] = {
    "кто": "(?:кто|кого|кому|кем|ком)",
    "что": "(?:что|чего|чему|чем|ч[её]м)",
    "тот": "(?:тот|того|тому|тем|том|та|ту|той|то|те|тех|теми)",
}


def language_violation(
    payload: TranslationResultResponse,
    *,
    target_language: str,
    min_share: float,
    min_length: int,
) -> str | None:
    """Violation description of translation language mismatch with target language, or `None`."""
    alphabet = _TARGET_ALPHABETS.get(_language_key(target_language))
    if alphabet is None:
        return None
    for block in payload.blocks:
        # One check on the join a segmented result stores, not on each slice.
        text = joined_segment_translation(block)
        if text is None or len(text) <= min_length:
            continue
        share = _alphabet_share(text, alphabet=alphabet)
        if share is not None and share < min_share:
            return (
                f"перевод блока {block.source_block_id} не на языке {target_language}: "
                f"доля целевого алфавита {share:.2f} ниже порога {min_share:.2f}"
            )
    return None


def translation_quality_warnings(
    payload: TranslationResultResponse,
    *,
    unit: TranslationUnit,
    target_language: str,
    mandatory_glossary: Sequence[GlossarySnapshotEntry] = (),
) -> tuple[str, ...]:
    """Non-blocking Russian translation quality warnings (P2.T1).

    Flags:
    - `dozen` rendered as `десяток`;
    - hyphenated typo `что-бы`;
    - `со` before `[зсжш][аеёиоуыэюя]` (preserving consonant clusters like `со зла`, `со знанием`);
    - calque initial `Х. <Surname>` when canonical glossary entry maps `H*` to `Г*`;
    - indeclinable `Сам-Знаешь-Кто` after Russian prepositions;
    - drift from `mandatory_glossary` when `source_text` appears in the block's original text.
    """
    if _language_key(target_language) != "ru":
        return ()
    items = {item.block.id: item for item in unit.items}
    warnings: list[str] = []

    def add(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    for block in payload.blocks:
        block_id = block.source_block_id
        item = items.get(block_id)
        source_text = _translated_source_text(block, item)
        translated_text = joined_segment_translation(block)
        prose_texts = _russian_prose_texts(payload, block=block, translated_text=translated_text)

        if (
            translated_text is not None
            and source_text
            and _DOZEN_SOURCE.search(source_text)
            and _DESYATOK_TRANSLATION.search(translated_text)
        ):
            add(
                f"блок {block_id}: слово 'dozen' (дюжина = 12) переведено как «десяток» (10)"
            )

        for text in prose_texts:
            if _CHTO_BY.search(text):
                add(
                    f"блок {block_id}: орфографическая ошибка «что-бы» (нужно «чтобы» или «что бы»)"
                )
            so_match = _SO_SIBILANT_VOWEL.search(text)
            if so_match is not None:
                add(
                    f"блок {block_id}: предлог «со » перед шипящей/свистящей с гласной ({so_match.group(1)!r})"
                )
            indec_match = _INDECLINABLE_YOU_KNOW_WHO.search(text)
            if indec_match is not None:
                add(
                    f"блок {block_id}: несклоняемое «Сам-Знаешь-Кто» после предлога ({indec_match.group(0)!r})"
                )

        if translated_text is not None and mandatory_glossary:
            for initial_warning in _calque_initial_warnings(
                block_id=block_id,
                translated_text=translated_text,
                mandatory_glossary=mandatory_glossary,
            ):
                add(initial_warning)
            if source_text:
                for glossary_warning in _glossary_mismatch_warnings(
                    block_id=block_id,
                    source_text=source_text,
                    translated_text=translated_text,
                    mandatory_glossary=mandatory_glossary,
                ):
                    add(glossary_warning)

    return tuple(warnings)


def _translated_source_text(
    block: BlockResultResponse, item: TranslationUnitItem | None
) -> str:
    """Source text corresponding to translated content of `block`."""
    if item is None:
        return ""
    modes = planned_segment_modes(item)
    if not modes or not block.segments:
        return item.block.text
    parts = [
        item.block.text[span.start : span.end]
        for span, mode in zip(item.block.segment_spans, modes, strict=True)
        if mode is BlockTreatment.TRANSLATE
    ]
    return " ".join(parts)


def _russian_prose_texts(
    payload: TranslationResultResponse,
    *,
    block: BlockResultResponse,
    translated_text: str | None,
) -> list[str]:
    """Russian prose texts of a block inspected for spelling and grammar warnings."""
    texts: list[str] = []
    if translated_text is not None:
        texts.append(translated_text)
    for hint in block.glossary_hints:
        if hint.note is not None:
            texts.append(hint.note)
    for note in payload.notes_for(block.source_block_id):
        texts.append(note.note)
    return texts


def _calque_initial_warnings(
    *,
    block_id: str,
    translated_text: str,
    mandatory_glossary: Sequence[GlossarySnapshotEntry],
) -> list[str]:
    """Detects calque initial `Х. <Surname>` when canonical Russian name begins with `Г*`."""
    warnings: list[str] = []
    h_to_g_surnames: list[tuple[str, str]] = []
    has_h_to_g_first = False
    single_surnames: list[str] = []

    for entry in mandatory_glossary:
        src_words = _TERM_WORD.findall(entry.source_text)
        tr_words = _TERM_WORD.findall(_PARENS.sub(" ", entry.translation))
        if not src_words or not tr_words:
            continue
        if len(src_words) >= 2 and len(tr_words) >= 2:
            if src_words[0][0].casefold() == "h" and tr_words[0][0].casefold() == "г":
                h_to_g_surnames.append((tr_words[-1], entry.translation))
        elif len(src_words) == 1 and len(tr_words) == 1:
            if src_words[0][0].casefold() == "h" and tr_words[0][0].casefold() == "г":
                has_h_to_g_first = True
            elif (
                entry.type is ConceptType.PERSON
                and src_words[0][0].isupper()
                and tr_words[0][0].isupper()
            ):
                single_surnames.append(tr_words[0])

    if has_h_to_g_first:
        for surname in single_surnames:
            h_to_g_surnames.append((surname, f"Г. {surname}"))

    for surname, canonical in h_to_g_surnames:
        surname_pattern = _word_stem_pattern(surname)
        match = re.search(
            rf"(?<![А-ЯЁа-яё])Х\.\s*(?i:{surname_pattern})(?![А-ЯЁа-яё])",
            translated_text,
        )
        if match is not None:
            warnings.append(
                f"блок {block_id}: калька инициала «{match.group(0)}» при каноническом «{canonical}» (нужен инициал «Г.»)"
            )
    return warnings


def _glossary_mismatch_warnings(
    *,
    block_id: str,
    source_text: str,
    translated_text: str,
    mandatory_glossary: Sequence[GlossarySnapshotEntry],
) -> list[str]:
    """Flags mandatory glossary terms present in `source_text` but missing from `translated_text`."""
    warnings: list[str] = []
    ordered_entries = sorted(
        mandatory_glossary,
        key=lambda entry: (
            -len(entry.source_text.strip()),
            not entry.status.is_user_decision,
            entry.source_text,
        ),
    )
    covered_spans: list[tuple[int, int]] = []
    for entry in ordered_entries:
        source_term = entry.source_text.strip()
        if not source_term:
            continue
        spans = _source_term_spans(source_text, source_term, entry_type=entry.type)
        uncovered = [
            span
            for span in spans
            if not any(
                covered_start <= span[0] and span[1] <= covered_end
                for covered_start, covered_end in covered_spans
            )
        ]
        if not uncovered:
            continue
        covered_spans.extend(uncovered)
        if not _translation_contains_term(translated_text, entry.translation):
            glossary_kind = (
                "обязательным глоссарием"
                if entry.status.is_user_decision
                else "предложенным глоссарием"
            )
            warnings.append(
                f"блок {block_id}: расхождение с {glossary_kind} для {source_term!r} "
                f"(ожидалось «{entry.translation}»)"
            )
    return warnings


def _source_term_spans(
    source_text: str,
    term: str,
    *,
    entry_type: ConceptType | None,
) -> list[tuple[int, int]]:
    """Match spans of `term` in `source_text` with hyphen-safe and case/article-aware boundaries."""
    if not term:
        return []
    left = r"(?<!\w)" if term.startswith("-") else r"(?<![\w-])"
    right = r"(?!\w)" if term.endswith("-") else r"(?![\w-])"
    case_sensitive = (entry_type in _CASE_SENSITIVE_CONCEPT_TYPES) or term[0].isupper()
    if case_sensitive:
        article_match = _LEADING_ARTICLE.match(term)
        if article_match is not None:
            article, tail = article_match.group(1), article_match.group(2)
            base = rf"(?i:{re.escape(article)})\s+{re.escape(tail)}"
        else:
            base = re.escape(term)
        upper_term = term.upper()
        body = rf"(?:{base}|{re.escape(upper_term)})" if upper_term != term else base
        flags = 0
    else:
        body = re.escape(term)
        flags = re.IGNORECASE
    return [
        (match.start(), match.end())
        for match in re.finditer(rf"{left}{body}{right}", source_text, flags)
    ]


def _translation_contains_term(translated_text: str, expected_translation: str) -> bool:
    cleaned = _PARENS.sub(" ", expected_translation)
    alternatives = [part.strip() for part in cleaned.split("/") if part.strip()]
    if not alternatives:
        return True
    for alternative in alternatives:
        words = _TERM_WORD.findall(alternative)
        if not words:
            continue
        significant = [word for word in words if len(word) >= 3] or words
        if all(
            re.search(
                rf"(?<!\w){_word_stem_pattern(word)}(?!\w)",
                translated_text,
                re.IGNORECASE,
            )
            for word in significant
        ):
            return True
    return False


def _word_stem_pattern(word: str) -> str:
    """Inflection-tolerant Russian/Latin regex pattern for a single glossary word."""
    lower = word.casefold()
    if lower in _IRREGULAR_RUSSIAN_FORMS:
        return _IRREGULAR_RUSSIAN_FORMS[lower]
    is_cyrillic = all("Ѐ" <= ch <= "ӿ" for ch in lower)
    if not is_cyrillic:
        return re.escape(lower)

    stem = lower
    if len(stem) >= 5 and stem.endswith(_ADJ_ENDINGS):
        stem = stem[:-2]
    elif len(stem) >= 4 and stem[-1] in "аяоеыиуюьй":
        stem = stem[:-1]

    if len(stem) >= 4 and stem[-2] in "оеё" and stem[-1] in "кцлрнмс":
        prefix = _escape_yo(stem[:-2])
        vowel = "[её]?" if stem[-2] in "её" else f"{re.escape(stem[-2])}?"
        return f"{prefix}{vowel}{re.escape(stem[-1])}[а-яё]*"
    return f"{_escape_yo(stem)}[а-яё]*"


def _escape_yo(text: str) -> str:
    """Escapes regex metacharacters while matching Russian `е` and `ё` interchangeably."""
    return "".join("[её]" if ch in "её" else re.escape(ch) for ch in text)


def _language_key(target_language: str) -> str:
    """Book language code normalized without region or case: `ru-RU` and `RU` both represent Russian."""
    return target_language.strip().lower().split("-")[0]


def _alphabet_share(text: str, *, alphabet: tuple[str, str]) -> float | None:
    """Share of target alphabet characters among text letters; `None` if text contains no letters."""
    first, last = alphabet
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return None
    target = sum(1 for character in letters if first <= character <= last)
    return target / len(letters)


__all__ = [
    "language_violation",
    "note_language_leak_violation",
    "translation_quality_warnings",
]
