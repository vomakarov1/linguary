"""One book projection and the inputs shared by export and the web.

The projection calls the chapter and block helpers. It does not call a model
and it does not write. A missing active run is an empty result map, not an
error. Export readiness is not required.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from linguary.application.errors import ApplicationError
from linguary.application.layout.ipa_reference import IpaSymbol, ipa_key
from linguary.application.layout.legend import LayoutFeatures, legend_lines
from linguary.application.layout.notes import block_notes
from linguary.application.layout.parts import LayoutPart, compose_parts
from linguary.application.layout.review import ReviewEntry, chapter_review
from linguary.application.layout.units import LayoutUnit, _whole_block_translation, block_units
from linguary.application.layout.who_is_who import (
    WhoEntry,
    _content_chapter_numbers,
    book_who_is_who,
    chapter_who_is_who,
)
from linguary.application.pronunciation_projection import project_pronunciations
from linguary.domain import (
    BlockProcessingResult,
    BlockTreatment,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    DocumentBlock,
    TermConcept,
    glossary_key,
)
from linguary.domain.content_plan import ContentPlan, ContextFact, Difficulty, SourceAnchor

_SHOWN_CERTAINTY = frozenset({"explicit", "inferred"})
_UNIT_SPEAKER_CERTAINTY = frozenset({"inferred"})
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_OPEN_QUOTES = {"“": "”", "‘": "’", "«": "»", '"': '"'}
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "prof"})
_REMARK_VERBS = frozenset({
    "say", "says", "said",
    "ask", "asks", "asked",
    "reply", "replies", "replied",
    "answer", "answers", "answered",
    "tell", "tells", "told",
    "demand", "demands", "demanded",
    "murmur", "murmurs", "murmured",
    "whisper", "whispers", "whispered",
    "snap", "snaps", "snapped",
    "mutter", "mutters", "muttered",
    "shout", "shouts", "shouted",
    "cry", "cries", "cried",
})
_REMARK_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.IGNORECASE)
_TITLE_TRANSLATIONS = {
    "mr": "мистер",
    "mrs": "миссис",
    "miss": "мисс",
    "ms": "мис",
    "uncle": "дядя",
    "aunt": "тётя",
    "professor": "профессор",
    "dr": "доктор",
    "sir": "сэр",
    "madam": "мадам",
    "lord": "лорд",
}


@dataclass(frozen=True, slots=True)
class LayoutInputs:
    """Stored data one projection reads. Nothing here is computed from a model."""

    document: CanonicalDocument
    results: Mapping[str, BlockProcessingResult]
    analyses: Sequence[ChapterAnalysis]
    concepts: Sequence[TermConcept]
    reader_level: str | None
    english_help: bool = False


@dataclass(frozen=True, slots=True)
class ChapterLayout:
    """One section: reading units, and extras only when the section is content.

    `number` is none for front matter and for a notes-only section. Those
    sections still carry a unit for every block.
    """

    section_id: str
    number: int | None
    units: Mapping[str, tuple[LayoutUnit, ...]]
    who_is_who: tuple[WhoEntry, ...]
    review: tuple[ReviewEntry, ...]


@dataclass(frozen=True, slots=True)
class BookLayout:
    """The finished book: chapters in reading order, then the book-level pages."""

    chapters: Mapping[str, ChapterLayout]
    legend: tuple[str, ...]
    ipa_key: tuple[IpaSymbol, ...]
    who_is_who: tuple[WhoEntry, ...]
    features: LayoutFeatures


def build_layout_inputs(unit_of_work, book_id: str) -> LayoutInputs:
    """Load the document, active-run results, analyses, glossary, and conditions.

    `read_transaction` is used when the unit of work has it, otherwise
    `transaction`. No active run leaves `results` empty. No conditions leaves
    `reader_level` unset and `english_help` false. A missing book or document
    is an error. The run does not have to be finished, and chapters do not
    have to have a content plan.
    """
    with getattr(unit_of_work, "read_transaction", unit_of_work.transaction)() as transaction:
        if transaction.books.get(book_id) is None:
            raise ApplicationError(f"книги {book_id} нет в хранилище")
        document = transaction.documents.get(book_id)
        if document is None:
            raise ApplicationError(f"книга {book_id} не импортирована")
        run = transaction.runs.active_for_book(book_id)
        if run is None:
            results: dict[str, BlockProcessingResult] = {}
        else:
            results = {
                item.block_id: item for item in transaction.results.list_for_run(run.id)
            }
        analyses = tuple(transaction.analyses.list_for_book(book_id))
        concepts = tuple(transaction.glossary.list_for_book(book_id))
        reader_level, english_help = _reader_settings(transaction.work.latest_conditions(book_id))
    return LayoutInputs(document, results, analyses, concepts, reader_level, english_help)


def build_book_layout(inputs: LayoutInputs) -> BookLayout:
    """Project `inputs` into chapters, legend lines, and the book-level lists.

    Unit labels use inferred speaker facts only. Repeated speech styles still
    count explicit and inferred facts. A repeated speech-style
    difficulty for the same speaker is removed from the plan copy and from the
    assistance on the result copy passed to `block_notes`, so the later note
    is absent. The id is dropped only in the section that owns that later
    style. The stored plan and the stored result are not changed. There is
    no Previously list.
    """
    document = inputs.document
    numbers = _content_chapter_numbers(document)
    facts = _speaker_facts(inputs.analyses)
    dropped_by_section = _repeated_speech_style_ids(document, inputs.analyses, facts)
    plans = _plans_by_section(inputs.analyses)
    pronunciations = project_pronunciations(tuple(document.blocks()), inputs.results)
    chapters: dict[str, ChapterLayout] = {}
    for section in document.ordered_sections:
        dropped = dropped_by_section.get(section.id, frozenset())
        plan = _plan_without(plans.get(section.id), dropped)
        units: dict[str, tuple[LayoutUnit, ...]] = {}
        for block in section.ordered_blocks:
            result = inputs.results.get(block.id)
            notes = ()
            if result is not None:
                notes = block_notes(
                    block,
                    _result_without(result, dropped),
                    plan,
                    pronunciations,
                    english_help=inputs.english_help,
                )
            units[block.id] = block_units(
                block,
                result,
                _treatments(block, result),
                notes,
                _speakers_for_block(
                    block,
                    facts,
                    inputs.concepts,
                    inputs.analyses,
                    document.target_language,
                    _treatments(block, result),
                    result,
                    document=document,
                ),
            )
        chapters[section.id] = ChapterLayout(
            section_id=section.id,
            number=numbers.get(section.id),
            units=units,
            who_is_who=chapter_who_is_who(
                document,
                inputs.analyses,
                inputs.concepts,
                section_id=section.id,
            ),
            review=chapter_review(
                document,
                inputs.results,
                section_id=section.id,
                reader_level=inputs.reader_level,
            ),
        )
    features = _features(chapters, inputs.analyses)
    return BookLayout(
        chapters=chapters,
        legend=legend_lines(features),
        ipa_key=ipa_key(_ipa_strings(chapters)),
        who_is_who=book_who_is_who(document, inputs.analyses, inputs.concepts),
        features=features,
    )


def _reader_settings(conditions: object) -> tuple[str | None, bool]:
    """Reader level and English help from conditions. Both default when absent."""
    if conditions is None:
        return None, False
    payload = getattr(conditions, "payload", None)
    if not isinstance(payload, Mapping):
        return None, False
    level = payload.get("reader_level")
    reader_level = level if isinstance(level, str) else None
    return reader_level, bool(payload.get("english_help", False))


def _treatments(
    block: DocumentBlock,
    result: BlockProcessingResult | None,
) -> tuple[BlockTreatment | None, ...]:
    """One treatment per span. No result is all none. An empty result repeats the block mode."""
    count = len(block.segment_spans)
    if result is None:
        return (None,) * count
    if not result.segment_treatments:
        return (result.treatment,) * count
    return result.segment_treatments


def _plans_by_section(analyses: Sequence[ChapterAnalysis]) -> dict[str, ContentPlan]:
    """Latest content plan for each section. An analysis without one is skipped."""
    found: dict[str, ContentPlan] = {}
    for analysis in analyses:
        if analysis.content_plan is not None:
            found[analysis.section_id] = analysis.content_plan
    return found


def _speaker_facts(analyses: Sequence[ChapterAnalysis]) -> tuple[ContextFact, ...]:
    """Speaker facts with a name. `unknown` and other kinds are not shown."""
    found: list[ContextFact] = []
    for analysis in analyses:
        plan = analysis.content_plan
        if plan is None:
            continue
        for fact in plan.context_facts:
            if fact.kind != "speaker" or fact.certainty not in _SHOWN_CERTAINTY:
                continue
            if not isinstance(fact.value, str) or not fact.value.strip():
                continue
            found.append(fact)
    return tuple(found)


def _speaker_name(
    value: str,
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    *,
    document: CanonicalDocument | None = None,
) -> str | None:
    """Russian (or other target) name for `value`, or none when it stays unresolved.

    Keys are the whole value, the text before `(`, and the text inside `(...)`.
    A user decision wins, then a person candidate from a content section, then
    for Russian a title or kinship prefix plus the same lookup on the remainder.
    The English source is never returned on its own.
    """
    variants = _name_variants(value)
    for variant in variants:
        found = _user_translation(variant, concepts, language)
        if found is not None:
            return found
    for variant in variants:
        found = _candidate_translation(variant, analyses, language, document)
        if found is not None:
            return found
    if _normalized_language(language) == "ru":
        for variant in variants:
            found = _titled_name(variant, concepts, analyses, language, document)
            if found is not None:
                return found
    return None


def _speakers_for_block(
    block: DocumentBlock,
    facts: Sequence[ContextFact],
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    treatments: Sequence[BlockTreatment | None],
    result: BlockProcessingResult | None = None,
    *,
    document: CanonicalDocument | None = None,
) -> dict[tuple[int, int], str]:
    """Inferred subject anchors whose shown part opens the line or continues its quote.

    Parts follow `compose_parts`, including the single unit used when a whole-block
    translation replaces those parts. Secondary anchors are ignored. A part that
    starts as narrative, with a quote only later, is absent. The first resolved
    name keeps a repeated part.
    """
    speakers: dict[tuple[int, int], str] = {}
    parts = _shown_parts(block, treatments, result)
    for fact in facts:
        if fact.certainty not in _UNIT_SPEAKER_CERTAINTY or not isinstance(fact.value, str):
            continue
        anchor = fact.subject_anchor
        if anchor.block_id != block.id:
            continue
        name = _speaker_name(
            fact.value, concepts, analyses, language, document=document,
        )
        if name is None:
            continue
        for part in parts:
            if _part_shows_speaker(block.text, part, anchor.start):
                speakers.setdefault((part.start, part.end), name)
    return speakers


def _name_variants(value: str) -> tuple[str, ...]:
    """Whole name, the part before parentheses, and the part inside them."""
    text = value.strip()
    if not text:
        return ()
    found = [text]
    if "(" in text and ")" in text:
        before, _, rest = text.partition("(")
        inside, _, _ = rest.partition(")")
        if before.strip():
            found.append(before.strip())
        if inside.strip():
            found.append(inside.strip())
    variants: list[str] = []
    seen: set[str] = set()
    for variant in found:
        key = glossary_key(variant)
        if not key or key in seen:
            continue
        seen.add(key)
        variants.append(variant)
    return tuple(variants)


def _user_translation(
    variant: str,
    concepts: Sequence[TermConcept],
    language: str,
) -> str | None:
    """Approved or locked variant for this key. A suggestion does not count."""
    key = glossary_key(variant)
    if not key:
        return None
    normalized = _normalized_language(language)
    for concept in concepts:
        if glossary_key(concept.canonical_source_text) != key:
            continue
        chosen = concept.variant_for(normalized)
        if chosen is not None and chosen.status.is_user_decision and chosen.text.strip():
            return chosen.text.strip()
    return None


def _candidate_translation(
    variant: str,
    analyses: Sequence[ChapterAnalysis],
    language: str,
    document: CanonicalDocument | None,
) -> str | None:
    """Person suggestion from a content section. Russian text must use Cyrillic."""
    key = glossary_key(variant)
    if not key:
        return None
    russian = _normalized_language(language) == "ru"
    allowed = _content_section_ids(document)
    for analysis in _analyses_in_reading_order(analyses, document, allowed):
        for candidate in analysis.glossary_candidates:
            if candidate.type is not ConceptType.PERSON:
                continue
            if glossary_key(candidate.source_text) != key:
                continue
            translation = candidate.suggested_translation.strip()
            if not translation:
                continue
            if russian and _CYRILLIC.search(translation) is None:
                continue
            return translation
    return None


def _titled_name(
    variant: str,
    concepts: Sequence[TermConcept],
    analyses: Sequence[ChapterAnalysis],
    language: str,
    document: CanonicalDocument | None,
) -> str | None:
    """Russian title or kinship word plus a resolved remainder. No English fallback."""
    if _normalized_language(language) != "ru":
        return None
    token, separator, rest = glossary_key(variant).partition(" ")
    if not separator or token not in _TITLE_TRANSLATIONS or not rest.strip():
        return None
    translated = _user_translation(rest, concepts, language)
    if translated is None:
        translated = _candidate_translation(rest, analyses, language, document)
    if translated is None:
        return None
    return f"{_TITLE_TRANSLATIONS[token]} {translated}"


def _normalized_language(language: str) -> str:
    return language.strip().lower()


def _content_section_ids(document: CanonicalDocument | None) -> frozenset[str] | None:
    """Content section ids, or none when the document was not passed in."""
    if document is None:
        return None
    return frozenset(
        section.id
        for section in document.ordered_sections
        if not document.is_front_matter(section) and not section.is_notes_only
    )


def _analyses_in_reading_order(
    analyses: Sequence[ChapterAnalysis],
    document: CanonicalDocument | None,
    allowed: frozenset[str] | None,
) -> tuple[ChapterAnalysis, ...]:
    """Content analyses in section order. Without a document, keep the given order."""
    if document is None:
        return tuple(analyses)
    grouped: dict[str, list[ChapterAnalysis]] = {}
    for analysis in analyses:
        if allowed is not None and analysis.section_id not in allowed:
            continue
        grouped.setdefault(analysis.section_id, []).append(analysis)
    ordered: list[ChapterAnalysis] = []
    for section in document.ordered_sections:
        ordered.extend(grouped.get(section.id, ()))
    return tuple(ordered)


def _shown_parts(
    block: DocumentBlock,
    treatments: Sequence[BlockTreatment | None],
    result: BlockProcessingResult | None,
) -> tuple[LayoutPart, ...]:
    """Parts `block_units` will build, so a collapsed translation is one part."""
    chosen = tuple(treatments)
    parts = compose_parts(block, chosen)
    if _whole_block_translation(result, chosen) is None:
        return parts
    spans = block.segment_spans
    return (
        LayoutPart(
            start=spans[0].start,
            end=spans[-1].end,
            treatment=BlockTreatment.TRANSLATE,
            segments=tuple(range(len(spans))),
        ),
    )


def _part_shows_speaker(text: str, part: LayoutPart, anchor_start: int) -> bool:
    """Whether `anchor_start` belongs to the dialogue that opens this part."""
    opening_end = _opening_speech_end(text, part.start, part.end)
    if opening_end is not None and part.start <= anchor_start < opening_end:
        return True
    continued = _continued_speech_bounds(text, part.start)
    if continued is None:
        return False
    start, end = continued
    return start <= anchor_start < end


def _opening_speech_end(text: str, part_start: int, part_end: int) -> int | None:
    """End of the opening replica and its remark, when the part starts with one."""
    open_at = _first_nonspace(text, part_start, part_end)
    if open_at is None or not _is_dialogue_opener(text, open_at, part_end):
        return None
    if text[open_at] not in _OPEN_QUOTES:
        return _sentence_end(text, open_at, part_end)
    close = _matching_close(text, open_at, part_end)
    if close is None:
        return part_end
    return min(part_end, _remark_end(text, close, part_end))


def _continued_speech_bounds(text: str, part_start: int) -> tuple[int, int] | None:
    """Quote opened before this part and still open there, through its remark."""
    if part_start <= 0:
        return None
    open_at = _outer_unclosed_open(text[:part_start])
    if open_at is None:
        return None
    close = _matching_close(text, open_at, len(text))
    if close is None:
        return (open_at, len(text))
    if close < part_start:
        return None
    return (open_at, _remark_end(text, close, len(text)))


def _outer_unclosed_open(text: str) -> int | None:
    """Index of the outermost quote that `text` does not close."""
    stack: list[tuple[int, str]] = []
    for index, char in enumerate(text):
        if char in _OPEN_QUOTES:
            if char == '"' and stack and stack[-1][1] == '"':
                stack.pop()
            else:
                stack.append((index, _OPEN_QUOTES[char]))
        elif stack and char == stack[-1][1]:
            stack.pop()
    if not stack:
        return None
    return stack[0][0]


def _matching_close(text: str, open_index: int, limit: int) -> int | None:
    """Index of the quote that closes the opener at `open_index`, inside `limit`."""
    opener = text[open_index]
    if opener not in _OPEN_QUOTES:
        return None
    stack = [_OPEN_QUOTES[opener]]
    for index in range(open_index + 1, limit):
        char = text[index]
        if char in _OPEN_QUOTES:
            if char == '"' and stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append(_OPEN_QUOTES[char])
        elif stack and char == stack[-1]:
            stack.pop()
        if not stack:
            return index
    return None


def _remark_end(text: str, close_index: int, limit: int) -> int:
    """End of the reporting clause after a closing quote, before the next sentence.

    A capital right after the quote is still that remark when the same sentence
    has a speech verb (`Harry said`, `I said`). It starts the next sentence
    only when no such verb follows (`Harry left`).
    """
    index = close_index + 1
    while index < limit and text[index].isspace():
        index += 1
    if index >= limit:
        return limit
    if text[index] in _OPEN_QUOTES:
        return close_index + 1
    sentence_end = _sentence_end(text, close_index + 1, limit)
    if text[index].isupper() and not _speech_verb_in(text, index, sentence_end):
        return close_index + 1
    return sentence_end


def _speech_verb_in(text: str, start: int, end: int) -> bool:
    """Whether the sentence slice contains a finite speech verb."""
    return any(
        match.group(0).casefold() in _REMARK_VERBS
        for match in _REMARK_WORD.finditer(text, start, end)
    )


def _sentence_end(text: str, start: int, limit: int) -> int:
    """Exclusive end of the sentence containing `start`, capped at `limit`."""
    index = start
    while index < limit:
        if text[index] in ".!?" and _ends_sentence(text, index, limit):
            return index + 1
        index += 1
    return limit


def _ends_sentence(text: str, index: int, limit: int) -> bool:
    """A terminator that is not an abbreviation dot and is followed by a break."""
    if text[index] == "." and _abbreviation_dot(text, index):
        return False
    next_index = index + 1
    return next_index >= limit or text[next_index].isspace()


def _abbreviation_dot(text: str, index: int) -> bool:
    word_start = index
    while word_start > 0 and text[word_start - 1].isalpha():
        word_start -= 1
    return text[word_start:index].casefold() in _ABBREVIATIONS


def _first_nonspace(text: str, start: int, end: int) -> int | None:
    index = start
    while index < end and text[index].isspace():
        index += 1
    if index >= end:
        return None
    return index


def _is_dialogue_opener(text: str, index: int, end: int) -> bool:
    """Opening quote or a dialogue dash, not a hyphen glued to a word."""
    char = text[index]
    if char in _OPEN_QUOTES or char in "—–":
        return True
    return char == "-" and index + 1 < end and text[index + 1] in " —–"


def _repeated_speech_style_ids(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
    facts: Sequence[ContextFact],
) -> dict[str, frozenset[str]]:
    """Later speech-style difficulties of one speaker, by owning section.

    The same source name is one speaker across chapters, whether the fact is
    explicit or inferred. The earliest anchor wins, not the difficulty id.
    A style that overlaps no shown speaker is not repeated and is not listed.
    Difficulty ids repeat across plans, so each dropped id is recorded only
    for the section that owns that later style.
    """
    positions = _block_positions(document)
    anchors_by_speaker: dict[str, list[SourceAnchor]] = {}
    for fact in facts:
        if not isinstance(fact.value, str):
            continue
        anchors_by_speaker.setdefault(glossary_key(fact.value), []).extend(fact.anchors)
    grouped: dict[str, list[tuple[tuple[int, int, int, str], str, str]]] = {}
    for analysis in analyses:
        plan = analysis.content_plan
        if plan is None:
            continue
        for difficulty in plan.difficulties:
            if difficulty.kind != "speech_style":
                continue
            position = _appearance(difficulty, positions)
            for key, anchors in anchors_by_speaker.items():
                if not _overlaps(difficulty.anchors, anchors):
                    continue
                grouped.setdefault(key, []).append((position, analysis.section_id, difficulty.id))
    kept: set[tuple[str, str]] = set()
    later: set[tuple[str, str]] = set()
    for items in grouped.values():
        items.sort()
        kept.add((items[0][1], items[0][2]))
        later.update((section_id, difficulty_id) for _, section_id, difficulty_id in items[1:])
    dropped: dict[str, set[str]] = {}
    for section_id, difficulty_id in later - kept:
        dropped.setdefault(section_id, set()).add(difficulty_id)
    return {section_id: frozenset(ids) for section_id, ids in dropped.items()}


def _block_positions(document: CanonicalDocument) -> dict[str, tuple[int, int]]:
    """Section index and block index for every block, in reading order."""
    positions: dict[str, tuple[int, int]] = {}
    for section_index, section in enumerate(document.ordered_sections):
        for block_index, block in enumerate(section.ordered_blocks):
            positions[block.id] = (section_index, block_index)
    return positions


def _appearance(
    difficulty: Difficulty,
    positions: Mapping[str, tuple[int, int]],
) -> tuple[int, int, int, str]:
    """Earliest anchor of a difficulty. An anchor outside the document sorts last."""
    ranked: list[tuple[int, int, int]] = []
    for anchor in difficulty.anchors:
        located = positions.get(anchor.block_id)
        if located is None:
            continue
        ranked.append((located[0], located[1], anchor.start))
    if not ranked:
        return (10**9, 10**9, 10**9, difficulty.id)
    section_index, block_index, start = min(ranked)
    return (section_index, block_index, start, difficulty.id)


def _overlaps(
    left: Sequence[SourceAnchor],
    right: Sequence[SourceAnchor],
) -> bool:
    """Whether two half-open anchor ranges share a block and an offset."""
    for anchor in left:
        for other in right:
            if (
                anchor.block_id == other.block_id
                and anchor.start < other.end
                and other.start < anchor.end
            ):
                return True
    return False


def _plan_without(plan: ContentPlan | None, dropped: frozenset[str]) -> ContentPlan | None:
    """Copy of `plan` without the dropped difficulties. The stored plan is not changed.

    Evidence that pointed at a dropped difficulty is left off the copy so the
    plan stays valid. Other difficulties and the speaker facts stay.
    """
    if plan is None or not dropped:
        return plan
    kept = tuple(item for item in plan.difficulties if item.id not in dropped)
    if len(kept) == len(plan.difficulties):
        return plan
    evidence = tuple(
        item for item in plan.before_you_read_evidence if item.difficulty_id not in dropped
    )
    return replace(plan, difficulties=kept, before_you_read_evidence=evidence)


def _result_without(
    result: BlockProcessingResult,
    dropped: frozenset[str],
) -> BlockProcessingResult:
    """Copy without assistance for a dropped speech style. The stored result stays.

    `block_notes` still emits that assistance after the difficulty leaves the
    plan. Omitting it drops the note, label and body. Other help stays.
    """
    if not dropped:
        return result
    kept = tuple(item for item in result.fragment_assistances if item.difficulty_id not in dropped)
    if len(kept) == len(result.fragment_assistances):
        return result
    return replace(result, fragment_assistances=kept)


def _features(
    chapters: Mapping[str, ChapterLayout],
    analyses: Sequence[ChapterAnalysis],
) -> LayoutFeatures:
    """Layers and section keys that the finished chapters actually contain.

    `previously` is not set: that list is not projected. Before You Read is a
    section key when a content chapter's analysis has entries. Labels are the
    strings written on notes, not kind ids that lost their label. `has_notes`
    is true only when some note has a number. An unnumbered «Говорит …» note
    sets `has_speakers` and does not turn on the numbered-notes line.
    """
    units = [
        unit
        for chapter in chapters.values()
        for grouped in chapter.units.values()
        for unit in grouped
    ]
    notes = [item.note for unit in units for item in unit.notes]
    content_ids = {
        section_id for section_id, chapter in chapters.items() if chapter.number is not None
    }
    sections: set[str] = set()
    if any(chapter.who_is_who for chapter in chapters.values()):
        sections.add("who_is_who")
    if any(chapter.review for chapter in chapters.values()):
        sections.add("review")
    if any(
        analysis.before_you_read and analysis.section_id in content_ids for analysis in analyses
    ):
        sections.add("before_you_read")
    return LayoutFeatures(
        has_translation=any(unit.translation for unit in units),
        has_notes=any(item.number is not None for unit in units for item in unit.notes),
        labels=frozenset(label for note in notes for label in note.labels),
        has_ipa=any(note.ipa for note in notes),
        has_poems=any(unit.lines for unit in units),
        has_speakers=any(unit.speaker for unit in units),
        sections=frozenset(sections),
    )


def _ipa_strings(chapters: Mapping[str, ChapterLayout]) -> tuple[str, ...]:
    """IPA strings carried by notes, in reading order."""
    return tuple(
        item.note.ipa
        for chapter in chapters.values()
        for grouped in chapter.units.values()
        for unit in grouped
        for item in unit.notes
        if item.note.ipa
    )
