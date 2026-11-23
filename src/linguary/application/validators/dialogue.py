"""Russian direct-speech punctuation validation (P2.T1).

Checks Russian translations of prose blocks for structural dialogue punctuation defects:
- hybrid dash+guillemet openings (`— «`, `«—`);
- unmatched / odd guillemets `«` / `»` (while preserving intentional multi-paragraph
  unclosed opening quotes mirrored from the source);
- author narration with a spoken speech verb embedded inside `«…»`;
- stray `— ` inserted between sentences of a single speaker without author narration
  when the source is a single unbroken speech (without flagging ordinary prose dashes);
- paragraph-initial `«…»` followed by an explicit spoken speech verb (`сказал`,
  `спросил`, `крикнул`, `прошептал`, ...), while allowing internal thoughts
  (`«…», — подумал он`).

Validation runs only for Russian (`ru`) and skips `BlockType.POEM`.
"""

from __future__ import annotations

from itertools import pairwise
import re

from linguary.application.llm_dto import BlockResultResponse, TranslationResultResponse
from linguary.application.validators.quotes import (
    _outer_unclosed_open,
    british_ends_with_closing_quote,
    british_quote_counts,
    has_direct_speech_quotes,
    has_unclosed_british_quote,
)
from linguary.application.validators.treatment import joined_segment_translation
from linguary.domain import BlockType, TranslationUnit, TranslationUnitItem

#: Spoken-aloud speech/dialogue verbs (explicitly excluding internal-thought verbs).
_SPOKEN_VERB_ALT = (
    r"(?:"
    r"сказал[аои]?|скаж(?:ет|ут)|говорил[аои]?|говорит|говорят|заговорил[аои]?|проговорил[аои]?|"
    r"спросил[аои]?|спрашивал[аои]?|спрашивает|переспросил[аои]?|"
    r"ответил[аои]?|отвечал[аои]?|отвечает|"
    r"крикнул[аои]?|кричал[аои]?|закричал[аои]?|выкрикнул[аои]?|вскрикнул[аои]?|прокричал[аои]?|"
    r"прошептал[аои]?|шепнул[аои]?|шептал[аои]?|зашептал[аои]?|"
    r"пробормотал[аои]?|бормотал[аои]?|забормотал[аои]?|буркнул[аои]?|пробурчал[аои]?|"
    r"воскликнул[аои]?|произн[ёе]с(?:л[аои])?|"
    r"добавил[аои]?|прибавил[аои]?|продолжил[аои]?|перебил[аои]?|возразил[аои]?|"
    r"заметил[аои]?|пояснил[аои]?|объяснил[аои]?|уточнил[аои]?|сообщил[аои]?|"
    r"объявил[аои]?|заявил[аои]?|подтвердил[аои]?|"
    r"рявкнул[аои]?|гаркнул[аои]?|фыркнул[аои]?|хмыкнул[аои]?|"
    r"выпалил[аои]?|отрезал[аои]?|огрызнул(?:ся|ась|ись)|протянул[аои]?|"
    r"позвал[аои]?|окликнул[аои]?|откликнул(?:ся|ась|ись)|отозвал(?:ся|ась|ись)|"
    r"повторил[аои]?|напомнил[аои]?|предложил[аои]?|попросил[аои]?|"
    r"приказал[аои]?|велел[аои]?|потребовал[аои]?|предупредил[аои]?|пообещал[аои]?|"
    r"пожаловал(?:ся|ась|ись)|простонал[аои]?|прохрипел[аои]?|прошипел[аои]?|"
    r"промычал[аои]?|пропищал[аои]?|пискнул[аои]?|завопил[аои]?|заорал[аои]?|"
    r"проворчал[аои]?|промямлил[аои]?|пролепетал[аои]?|подхватил[аои]?|"
    r"усмехнул(?:ся|ась|ись)|рассмеял(?:ся|ась|ись)|засмеял(?:ся|ась|ись)|хихикнул[аои]?|"
    r"вздохнул[аои]?|ахнул[аои]?|охнул[аои]?"
    r")"
)

#: Internal-thought verbs allowed with guillemets (`«…», — подумал он`).
_THOUGHT_VERB_START = re.compile(
    r"^(?:(?:[А-ЯЁа-яё-]+\s+){0,2})?"
    r"(?:подумал[аои]?|думал[аои]?|думает|решил[аои]?|вспомнил[аои]?|"
    r"почувствовал[аои]?|понял[аои]?|представил[аои]?|сообразил[аои]?|"
    r"догадал(?:ся|ась|ись)|удивил(?:ся|ась|ись)|размышлял[аои]?)\b",
    re.IGNORECASE,
)

#: Words that must not be skipped as subject/adverb prefixes before a spoken verb.
_NON_PREFIX_WORD = (
    r"(?:и|но|а|что|чтобы|когда|если|"
    r"подумал[аои]?|думал[аои]?|решил[аои]?|вспомнил[аои]?|"
    r"почувствовал[аои]?|понял[аои]?|представил[аои]?|сообразил[аои]?)"
)

#: Optional 0..2 subject/adverb words before a spoken verb (e.g. `он`, `тихо`, `мистер Дурсль`).
_PREFIX_WORDS = rf"(?:(?:(?!{_NON_PREFIX_WORD}\b)[А-ЯЁа-яё-]+\s+){{0,2}})"

#: (a) Dialogue hybrid patterns combining dashes and guillemets.
_HYBRID_GUILLEMET_DASH = re.compile(r"«\s*[—–]")
_HYBRID_LEADING_DASH_GUILLEMET = re.compile(r"(?:^|\n)\s*[—–]\s*«")
_HYBRID_PUNCT_DASH_GUILLEMET = re.compile(r"[:.,!?…]\s*[—–]\s*«")
_HYBRID_DASH_GUILLEMET_ATTRIBUTION = re.compile(
    rf"[—–]\s*«[^«»\n]+»(?:\s*,\s*[—–]\s*|\s+[—–]\s*){_PREFIX_WORDS}{_SPOKEN_VERB_ALT}\b",
    re.IGNORECASE,
)

#: (c) Matched `«…»` span and spoken author-attribution dash inside it.
_GUILLEMET_SPAN = re.compile(r"«([^«»]+)»")
_INNER_AUTHOR_SPEECH = re.compile(
    rf"[—–]\s*{_PREFIX_WORDS}{_SPOKEN_VERB_ALT}\b",
    re.IGNORECASE,
)

#: (d) Stray dialogue dash between sentences of a single speaker.
_STRAY_INLINE_DASH = re.compile(r"[.!?…]\s+[—–]\s+[А-ЯЁ]")
_LOWERCASE_AUTHOR_DASH = re.compile(r"(?:[,!?…]|\.\.\.)\s*[—–]\s*[а-яё]")
_ABBREVIATION_BEFORE_DASH = re.compile(
    r"(?:"
    r"(?<![A-Za-zА-ЯЁа-яё])(?i:Mr|Mrs|Ms|Dr|St|Prof)\."
    r"|(?<![A-Za-zА-ЯЁа-яё])(?i:т\.\s*д|т\.\s*п|т\.\s*е|н\.\s*э)\."
    r"|(?<![A-Za-zА-ЯЁа-яё])(?i:г|св|др|пр|стр|гл)\."
    r"|(?<![A-Za-zА-ЯЁа-яё])[A-ZА-ЯЁ]\."
    r")$"
)

#: (e) Paragraph-initial `«…»` followed by attribution tail.
_PARAGRAPH_INITIAL_GUILLEMET = re.compile(
    r"^\s*«[^«»]+»(?:\s*,\s*[—–]\s*|\s+[—–]\s*|\s*,\s+)(.+)$"
)
_SPOKEN_ATTRIBUTION_START = re.compile(
    rf"^{_PREFIX_WORDS}{_SPOKEN_VERB_ALT}\b",
    re.IGNORECASE,
)


def dialogue_violation(
    payload: TranslationResultResponse,
    *,
    target_language: str,
    unit: TranslationUnit | None = None,
    strict_v3: bool = False,
) -> str | None:
    """Violation description for Russian dialogue punctuation defects, or `None`."""
    if _language_key(target_language) != "ru":
        return None
    items = {item.block.id: item for item in unit.items} if unit is not None else {}
    for block in payload.blocks:
        item = items.get(block.source_block_id)
        if item is not None and item.block.type is BlockType.POEM:
            continue
        text = joined_segment_translation(block)
        if text is None or not text.strip():
            continue
        source_text = item.block.text if item is not None else None
        violation = _text_dialogue_violation(
            text,
            block_id=block.source_block_id,
            source_text=source_text,
            strict_v3=strict_v3,
            block=block if strict_v3 else None,
            item=item if strict_v3 else None,
        )
        if violation is not None:
            return violation
    return None


def _language_key(target_language: str) -> str:
    """Normalized language code without region or case."""
    return target_language.strip().lower().split("-")[0]


def _text_dialogue_violation(
    text: str,
    *,
    block_id: str,
    source_text: str | None,
    strict_v3: bool = False,
    block: BlockResultResponse | None = None,
    item: TranslationUnitItem | None = None,
) -> str | None:
    """Applies rules (a)–(e) and strict v3 checks to one translated Russian block."""
    if _has_dialogue_hybrid(text):
        return (
            f"в переводе блока {block_id} смешаны тире и кавычки-ёлочки (гибрид «—» и ««»»): "
            "используй для прямой речи тире без ёлочек"
        )
    if _has_unmatched_guillemets(text, source_text=source_text):
        return (
            f"в переводе блока {block_id} непарное число кавычек « и »: "
            "проверь оформление прямой речи и цитат"
        )
    if _has_author_speech_inside_guillemets(text):
        return (
            f"в переводе блока {block_id} слова автора с глаголом речи оказались внутри кавычек «…»: "
            "оформляй прямую речь с ремаркой автора через тире без ёлочек"
        )
    if strict_v3 and block is not None and item is not None:
        seg_violation = _segmented_monologue_violation(block, item=item)
        if seg_violation is not None:
            return seg_violation
    if _has_stray_dialogue_dash(text, source_text=source_text):
        return (
            f"в переводе блока {block_id} лишнее тире «— » между предложениями одного говорящего "
            "без слов автора: в цельной реплике не ставь тире перед следующим предложением"
        )
    if strict_v3 and _has_inline_dialogue_dash(text, source_text=source_text):
        return (
            f"в переводе блока {block_id} лишнее тире «— » в строке после точки без переноса строки: "
            "прямую речь после слов автора начинай с новой строки и «— » либо оформляй через двоеточие"
        )
    if _has_guillemet_spoken_dialogue(text):
        return (
            f"в переводе блока {block_id} прямая речь вслух с начала абзаца оформлена в «…» вместо тире: "
            "реплики вслух с глаголом речи начинай с «— », ёлочки оставь только для мыслей"
        )
    return None


def _segmented_monologue_violation(
    block: BlockResultResponse, *, item: TranslationUnitItem
) -> str | None:
    """Under strict v3: rejects leading `—`/`–` on subsequent segments inside one unbroken quote."""
    if len(block.segments) < 2:
        return None
    source_text = item.block.text
    if "\n" in source_text:
        return None
    spans = item.block.segment_spans
    ordered = sorted(block.segments, key=lambda segment: segment.index)
    for prev_seg, curr_seg in pairwise(ordered):
        if not (1 <= prev_seg.index <= len(spans) and 1 <= curr_seg.index <= len(spans)):
            continue
        if not curr_seg.translated_text.lstrip().startswith(("—", "–")):
            continue
        prev_span = spans[prev_seg.index - 1]
        curr_span = spans[curr_seg.index - 1]
        prev_open = _outer_unclosed_open(source_text, prev_span.end)
        if prev_open is not None and prev_open == _outer_unclosed_open(source_text, curr_span.start):
            return (
                f"в переводе блока {block.source_block_id} лишнее тире в начале сегмента {curr_seg.index}: "
                "внутри одной непрерывной реплики тире ставь только в начале первого сегмента"
            )
    return None


def _has_dialogue_hybrid(text: str) -> bool:
    """Rule (a): detects `— «` and `«—` dialogue hybrids without flagging prose dashes before titles."""
    return bool(
        _HYBRID_GUILLEMET_DASH.search(text)
        or _HYBRID_LEADING_DASH_GUILLEMET.search(text)
        or _HYBRID_PUNCT_DASH_GUILLEMET.search(text)
        or _HYBRID_DASH_GUILLEMET_ATTRIBUTION.search(text)
    )


def _has_unmatched_guillemets(text: str, *, source_text: str | None) -> bool:
    """Rule (b): detects odd/unmatched `«`/`»`, mirroring multi-paragraph unclosed openings from source."""
    depth = 0
    for character in text:
        if character == "«":
            depth += 1
        elif character == "»":
            depth -= 1
            if depth < 0:
                return True
    if depth == 0:
        return False
    if (
        depth == 1
        and text.lstrip().startswith("«")
        and source_text is not None
        and _source_leaves_opening_quote_unclosed(source_text)
    ):
        return False
    return True


def _source_leaves_opening_quote_unclosed(source: str) -> bool:
    """Whether `source` opens a multi-paragraph quote left unclosed at paragraph end."""
    stripped = source.lstrip()
    if stripped.startswith('"'):
        return source.count('"') % 2 == 1
    if stripped.startswith("“"):
        return source.count("“") > source.count("”")
    if stripped.startswith("‘"):
        return has_unclosed_british_quote(source)
    return False


def _has_author_speech_inside_guillemets(text: str) -> bool:
    """Rule (c): detects author narration / spoken verb inside `«…»`."""
    for match in _GUILLEMET_SPAN.finditer(text):
        inner = match.group(1)
        if _INNER_AUTHOR_SPEECH.search(inner):
            return True
    return False


def _has_stray_dialogue_dash(text: str, *, source_text: str | None) -> bool:
    """Rule (d): detects stray `— ` between sentences of a single speaker without author narration."""
    if source_text is not None and not _is_single_unbroken_speech(source_text):
        return False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith(("—", "–")):
            continue
        body = line[1:].lstrip()
        match = _STRAY_INLINE_DASH.search(body)
        if match is None:
            continue
        prefix = body[: match.start() + 1]
        if _LOWERCASE_AUTHOR_DASH.search(prefix):
            continue
        return True
    return False


def _has_inline_dialogue_dash(text: str, *, source_text: str | None) -> bool:
    """Under strict v3: detects non-leading inline dialogue dash `[.!?…]\\s+[—–]\\s+[А-ЯЁ]`."""
    if source_text is None or not has_direct_speech_quotes(source_text):
        return False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("—", "–")):
            continue
        for match in _STRAY_INLINE_DASH.finditer(line):
            if (
                line[match.start()] == "."
                and _ABBREVIATION_BEFORE_DASH.search(line[: match.start() + 1]) is not None
            ):
                continue
            return True
    return False


def _is_single_unbroken_speech(source: str) -> bool:
    """Whether `source` is one continuous spoken quote without author interruption or second speaker."""
    stripped = source.strip()
    if not stripped:
        return False
    if stripped.startswith('"'):
        count = stripped.count('"')
        return (count == 2 and stripped.endswith('"')) or count == 1
    if stripped.startswith("“"):
        opens = stripped.count("“")
        closes = stripped.count("”")
        return opens == 1 and ((closes == 1 and stripped.endswith("”")) or closes == 0)
    if stripped.startswith("‘"):
        opens, closes = british_quote_counts(stripped)
        return opens == 1 and (
            (closes == 1 and british_ends_with_closing_quote(stripped)) or closes == 0
        )
    return False


def _has_guillemet_spoken_dialogue(text: str) -> bool:
    """Rule (e): detects paragraph-initial `«…»` with an explicit spoken speech verb, allowing thoughts."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("«"):
            continue
        match = _PARAGRAPH_INITIAL_GUILLEMET.match(line)
        if match is None:
            continue
        tail = match.group(1).strip()
        if _THOUGHT_VERB_START.match(tail):
            continue
        if _SPOKEN_ATTRIBUTION_START.match(tail):
            return True
    return False


__all__ = ["_outer_unclosed_open", "dialogue_violation"]
