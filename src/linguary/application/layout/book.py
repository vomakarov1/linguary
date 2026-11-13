"""One book projection and the inputs shared by export and the web.

The projection calls the chapter and block helpers. It does not call a model
and it does not write. A missing active run is an empty result map, not an
error. Export readiness is not required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from linguary.application.errors import ApplicationError
from linguary.application.layout.ipa_reference import IpaSymbol, ipa_key
from linguary.application.layout.legend import LayoutFeatures, legend_lines
from linguary.application.layout.notes import block_notes
from linguary.application.layout.review import ReviewEntry, chapter_review
from linguary.application.layout.units import LayoutUnit, block_units
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
    DocumentBlock,
    TermConcept,
    glossary_key,
)
from linguary.domain.content_plan import ContentPlan, ContextFact, Difficulty, SourceAnchor

_SHOWN_CERTAINTY = frozenset({"explicit", "inferred"})


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

    Speakers are explicit or inferred speaker facts. A repeated speech-style
    difficulty for the same speaker is removed from the plan copy passed to
    `block_notes`. The stored plan is not changed. There is no Previously list.
    """
    document = inputs.document
    numbers = _content_chapter_numbers(document)
    facts = _speaker_facts(inputs.analyses)
    dropped = _repeated_speech_style_ids(document, inputs.analyses, facts)
    plans = _plans_by_section(inputs.analyses)
    pronunciations = project_pronunciations(tuple(document.blocks()), inputs.results)
    chapters: dict[str, ChapterLayout] = {}
    for section in document.ordered_sections:
        plan = _plan_without(plans.get(section.id), dropped)
        units: dict[str, tuple[LayoutUnit, ...]] = {}
        for block in section.ordered_blocks:
            result = inputs.results.get(block.id)
            notes = ()
            if result is not None:
                notes = block_notes(
                    block,
                    result,
                    plan,
                    pronunciations,
                    english_help=inputs.english_help,
                )
            units[block.id] = block_units(
                block,
                result,
                _treatments(block, result),
                notes,
                _speakers_for_block(block, facts, inputs.concepts, document.target_language),
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


def _speaker_name(value: str, concepts: Sequence[TermConcept], language: str) -> str:
    """User-decision variant for this source value, otherwise the value itself."""
    key = glossary_key(value)
    for concept in concepts:
        if glossary_key(concept.canonical_source_text) != key:
            continue
        variant = concept.variant_for(language)
        if variant is not None and variant.status.is_user_decision:
            return variant.text
    return value


def _speakers_for_block(
    block: DocumentBlock,
    facts: Sequence[ContextFact],
    concepts: Sequence[TermConcept],
    language: str,
) -> dict[tuple[int, int], str]:
    """Anchors of shown speaker facts that fall in this block.

    A fact with no anchor here is absent. The first name keeps a repeated span.
    """
    speakers: dict[tuple[int, int], str] = {}
    for fact in facts:
        if not isinstance(fact.value, str):
            continue
        name = _speaker_name(fact.value, concepts, language)
        for anchor in fact.anchors:
            if anchor.block_id != block.id:
                continue
            speakers.setdefault((anchor.start, anchor.end), name)
    return speakers


def _repeated_speech_style_ids(
    document: CanonicalDocument,
    analyses: Sequence[ChapterAnalysis],
    facts: Sequence[ContextFact],
) -> frozenset[str]:
    """Later speech-style difficulties of one speaker, in reading order.

    The same source name is one speaker across chapters, whether the fact is
    explicit or inferred. The earliest anchor wins, not the difficulty id.
    A style that overlaps no shown speaker is not repeated and is not listed.
    """
    positions = _block_positions(document)
    anchors_by_speaker: dict[str, list[SourceAnchor]] = {}
    for fact in facts:
        if not isinstance(fact.value, str):
            continue
        anchors_by_speaker.setdefault(glossary_key(fact.value), []).extend(fact.anchors)
    grouped: dict[str, list[tuple[tuple[int, int, int, str], str]]] = {}
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
                grouped.setdefault(key, []).append((position, difficulty.id))
    kept: set[str] = set()
    later: set[str] = set()
    for items in grouped.values():
        items.sort()
        kept.add(items[0][1])
        later.update(difficulty_id for _, difficulty_id in items[1:])
    return frozenset(later - kept)


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


def _features(
    chapters: Mapping[str, ChapterLayout],
    analyses: Sequence[ChapterAnalysis],
) -> LayoutFeatures:
    """Layers and section keys that the finished chapters actually contain.

    `previously` is not set: that list is not projected. Before You Read is a
    section key when a content chapter's analysis has entries. Labels are the
    strings written on notes, not kind ids that lost their label.
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
        has_notes=any(unit.notes for unit in units),
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
