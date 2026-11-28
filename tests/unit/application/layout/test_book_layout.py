"""Book layout projected from stored chapters, and the shared input loader.

No model is called and nothing is written. A block without a result stays one
unit. A repeated speech-style note is decided by speaker and reading order.
Front matter and notes-only sections keep their units and lose the extras.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from linguary.application.errors import ApplicationError
from linguary.application.layout.book import (
    BookLayout,
    ChapterLayout,
    LayoutInputs,
    build_book_layout,
    build_layout_inputs,
)
from linguary.application.layout.ipa_reference import ipa_key
from linguary.application.layout.legend import LayoutFeatures, legend_lines
from linguary.application.layout.parts import PART_MAX_CHARS
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    BookMetadata,
    BookStatus,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    CoverageStatus,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    GlossaryHint,
    GlossarySnapshot,
    LLMSettingsSnapshot,
    ProcessingRun,
    ProcessingRunStatus,
    SourceReference,
    TermConcept,
    TermStatus,
    TermVariant,
    TreatmentPlan,
)
from linguary.domain.content_plan import (
    ContentPlan,
    ContextFact,
    Difficulty,
    RequiredPoint,
    SourceAnchor,
)
from linguary.domain.learning import BeforeYouReadEntry, FragmentAssistance
from linguary.domain.pronunciation import PronunciationAssistance, PronunciationEvidence

DOCUMENT_ID = "book-1"


def _block(
    section_id: str,
    order: int,
    text: str,
    *,
    kind: BlockType = BlockType.PARAGRAPH,
    segments: tuple[BlockSegment, ...] = (),
) -> DocumentBlock:
    """One block. The resource path keeps its identifier unique per section."""
    return DocumentBlock.create(
        document_id=DOCUMENT_ID,
        section_id=section_id,
        type=kind,
        order=order,
        text=text,
        source_reference=SourceReference(f"{section_id}.xhtml", f"/body/p[{order + 1}]"),
        segments=segments,
    )


def _section(
    section_id: str,
    order: int,
    title: str,
    blocks: Sequence[DocumentBlock],
) -> DocumentSection:
    return DocumentSection(
        id=section_id,
        document_id=DOCUMENT_ID,
        title=title,
        order=order,
        source_resource_path=f"{section_id}.xhtml",
        blocks=tuple(blocks),
    )


def _document(
    sections: Sequence[DocumentSection],
    *,
    target_language: str = "ru",
) -> CanonicalDocument:
    return CanonicalDocument(
        id=DOCUMENT_ID,
        metadata=BookMetadata(title="Wonderland", authors=("Carroll",)),
        source_language="en",
        target_language=target_language,
        sections=tuple(sections),
    )


def _anchor(block: DocumentBlock, quote: str, start: int | None = None) -> SourceAnchor:
    """An anchor whose quote sits at `start` in this block."""
    if start is None:
        start = block.text.find(quote)
    assert start >= 0
    return SourceAnchor(block.id, start, start + len(quote), quote)


def _difficulty(
    block: DocumentBlock,
    quote: str,
    kind: str,
    difficulty_id: str,
    *,
    help_kind: str = "explanation",
) -> Difficulty:
    anchor = _anchor(block, quote)
    return Difficulty(
        difficulty_id,
        (anchor,),
        block.id,
        kind,
        "useful",
        help_kind,
        (RequiredPoint(f"p-{difficulty_id}", "Explain the point"),),
    )


def _assistance(difficulty: Difficulty, text: str, kind: str) -> FragmentAssistance:
    return FragmentAssistance(
        difficulty.id,
        tuple(point.id for point in difficulty.required_points),
        kind,
        text,
        difficulty.anchors,
    )


def _plan(*difficulties: Difficulty, facts: tuple[ContextFact, ...] = ()) -> ContentPlan:
    """A v3 plan when it carries speaker facts, otherwise the stored default."""
    if facts:
        return ContentPlan(
            version=3,
            profile="content-plan-v3",
            difficulties=difficulties,
            context_facts=facts,
        )
    return ContentPlan(difficulties=difficulties)


def _analysis(
    analysis_id: str,
    section_id: str,
    *,
    candidates: Sequence[GlossaryCandidate] = (),
    plan: ContentPlan | None = None,
    before: Sequence[BeforeYouReadEntry] = (),
    chapter_summary: str = "Кратко.",
) -> ChapterAnalysis:
    return ChapterAnalysis(
        id=analysis_id,
        section_id=section_id,
        coverage_status=CoverageStatus.INCOMPLETE,
        treatment_plan=TreatmentPlan(section_id),
        translation_instructions="Держи имена.",
        chapter_summary=chapter_summary,
        glossary_candidates=tuple(candidates),
        before_you_read=tuple(before),
        content_plan=plan,
    )


def _candidate(
    analysis_id: str,
    candidate_id: str,
    source: str,
    translation: str,
    *,
    description: str = "",
) -> GlossaryCandidate:
    return GlossaryCandidate(
        id=candidate_id,
        analysis_id=analysis_id,
        source_text=source,
        type=ConceptType.PERSON,
        suggested_translation=translation,
        confidence=0.9,
        model_flagged_review=False,
        requires_review=True,
        description=description,
    )


def _concept(concept_id: str, source: str, translation: str, status: TermStatus) -> TermConcept:
    return TermConcept(
        id=concept_id,
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text=source,
        status=status,
        variants=(TermVariant(concept_id, "ru", translation, status),),
    )


def _fact(
    fact_id: str,
    block: DocumentBlock,
    quote: str,
    value: str | None,
    certainty: str,
    *,
    kind: str = "speaker",
) -> ContextFact:
    anchor = _anchor(block, quote)
    return ContextFact(fact_id, kind, (anchor,), anchor, value=value, certainty=certainty)


def _inputs(
    document: CanonicalDocument,
    results: dict[str, BlockProcessingResult] | None = None,
    analyses: Sequence[ChapterAnalysis] = (),
    concepts: Sequence[TermConcept] = (),
    *,
    reader_level: str | None = None,
    english_help: bool = False,
) -> LayoutInputs:
    return LayoutInputs(
        document,
        {} if results is None else results,
        tuple(analyses),
        tuple(concepts),
        reader_level,
        english_help,
    )


def _notes(layout: BookLayout, block_id: str) -> tuple:
    found = []
    for chapter in layout.chapters.values():
        for unit in chapter.units.get(block_id, ()):
            found.extend(item.note for item in unit.notes)
    return tuple(found)


def _labels(layout: BookLayout, block_id: str) -> tuple[str, ...]:
    return tuple(label for note in _notes(layout, block_id) for label in note.labels)


class _Ctx:
    """Context manager that yields the unit of work the repositories live on."""

    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def __enter__(self) -> _UnitOfWork:
        return self._unit_of_work

    def __exit__(self, *args: object) -> bool:
        return False


class _Books:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def get(self, book_id: str) -> object:
        self._unit_of_work.calls.append(f"books.get:{book_id}")
        if book_id != DOCUMENT_ID:
            return None
        return self._unit_of_work.book


class _Documents:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def get(self, book_id: str) -> CanonicalDocument | None:
        self._unit_of_work.calls.append(f"documents.get:{book_id}")
        if book_id != DOCUMENT_ID:
            return None
        return self._unit_of_work.document


class _Runs:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def active_for_book(self, book_id: str) -> ProcessingRun | None:
        self._unit_of_work.calls.append(f"runs.active_for_book:{book_id}")
        return self._unit_of_work.run


class _Results:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def list_for_run(self, run_id: str) -> tuple[BlockProcessingResult, ...]:
        self._unit_of_work.calls.append(f"results.list_for_run:{run_id}")
        return self._unit_of_work.result_rows


class _Analyses:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def list_for_book(self, book_id: str) -> tuple[ChapterAnalysis, ...]:
        self._unit_of_work.calls.append(f"analyses.list_for_book:{book_id}")
        return self._unit_of_work.analysis_rows


class _Glossary:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def list_for_book(self, book_id: str) -> tuple[TermConcept, ...]:
        self._unit_of_work.calls.append(f"glossary.list_for_book:{book_id}")
        return self._unit_of_work.concept_rows


class _Work:
    def __init__(self, unit_of_work: _UnitOfWork) -> None:
        self._unit_of_work = unit_of_work

    def latest_conditions(self, book_id: str) -> object:
        self._unit_of_work.calls.append(f"work.latest_conditions:{book_id}")
        return self._unit_of_work.conditions


class _UnitOfWork:
    """Fake unit of work. Repositories are available inside the transaction."""

    def __init__(
        self,
        *,
        book: object,
        document: CanonicalDocument | None,
        run: ProcessingRun | None,
        results: tuple[BlockProcessingResult, ...],
        analyses: tuple[ChapterAnalysis, ...],
        concepts: tuple[TermConcept, ...],
        conditions: object,
        readable: bool,
    ) -> None:
        self.book = book
        self.document = document
        self.run = run
        self.result_rows = results
        self.analysis_rows = analyses
        self.concept_rows = concepts
        self.conditions = conditions
        self.calls: list[str] = []
        self.books = _Books(self)
        self.documents = _Documents(self)
        self.runs = _Runs(self)
        self.results = _Results(self)
        self.analyses = _Analyses(self)
        self.glossary = _Glossary(self)
        self.work = _Work(self)
        if readable:
            self.read_transaction = self._read  # type: ignore[method-assign]

    def transaction(self) -> _Ctx:
        self.calls.append("transaction")
        return _Ctx(self)

    def _read(self) -> _Ctx:
        self.calls.append("read_transaction")
        return _Ctx(self)


def _unit_of_work(
    *,
    book: object,
    document: CanonicalDocument | None,
    run: ProcessingRun | None = None,
    results: tuple[BlockProcessingResult, ...] = (),
    analyses: tuple[ChapterAnalysis, ...] = (),
    concepts: tuple[TermConcept, ...] = (),
    conditions: object = None,
    readable: bool = True,
) -> _UnitOfWork:
    return _UnitOfWork(
        book=book,
        document=document,
        run=run,
        results=results,
        analyses=analyses,
        concepts=concepts,
        conditions=conditions,
        readable=readable,
    )


def _book() -> object:
    return SimpleNamespace(
        id=DOCUMENT_ID,
        status=BookStatus.PROCESSING,
        metadata=BookMetadata(title="Wonderland"),
        source_language="en",
        target_language="ru",
        source_epub_path="/tmp/book.epub",
        created_at=datetime(2026, 11, 13, tzinfo=UTC),
    )


def _run() -> ProcessingRun:
    return ProcessingRun(
        "run-1",
        DOCUMENT_ID,
        ProcessingRunStatus.ACTIVE,
        GlossarySnapshot(),
        LLMSettingsSnapshot("local", "test-model"),
    )


def test_structure_only_layout_whole_blocks_no_notes() -> None:
    """No result keeps every block whole, including a long multi-segment one.

    A whole `translated_text` with no segment translations is shown once on a
    unit that covers the block. Neither block grows notes.
    """
    plain = _block("chapter", 0, "Just the words.")
    long_text = "a" * 300 + "b" * 300
    bare = _block(
        "chapter",
        1,
        long_text,
        segments=(BlockSegment(0, 300, 0), BlockSegment(300, 600, 1)),
    )
    repeated = _block(
        "chapter",
        2,
        long_text,
        segments=(BlockSegment(0, 300, 0), BlockSegment(300, 600, 1)),
    )
    section = _section("chapter", 0, "Chapter 1", (plain, bare, repeated))
    repeated_result = BlockProcessingResult(
        repeated.id,
        BlockTreatment.TRANSLATE,
        translated_text="готовый перевод",
    )
    layout = build_book_layout(_inputs(_document((section,)), {repeated.id: repeated_result}))

    assert isinstance(layout, BookLayout)
    chapter = layout.chapters[section.id]
    assert isinstance(chapter, ChapterLayout)
    assert chapter.number == 1
    assert tuple(chapter.units) == (plain.id, bare.id, repeated.id)
    whole = chapter.units[plain.id]
    assert len(whole) == 1
    assert (whole[0].start, whole[0].end) == (0, len(plain.text))
    assert whole[0].translation is None
    assert whole[0].notes == ()
    assert whole[0].speaker is None
    assert whole[0].continues is False and whole[0].ends_paragraph is True
    untouched = chapter.units[bare.id]
    assert len(untouched) == 1
    assert (untouched[0].start, untouched[0].end, untouched[0].translation, untouched[0].notes) == (
        0,
        600,
        None,
        (),
    )
    shown = chapter.units[repeated.id]
    assert [(unit.start, unit.end, unit.translation) for unit in shown] == [
        (0, 600, "готовый перевод"),
    ]
    assert shown[0].notes == ()
    assert (shown[0].continues, shown[0].ends_paragraph) == (False, True)
    assert layout.features == LayoutFeatures(
        True, False, frozenset(), False, False, False, frozenset(), fades_translation=False
    )
    assert layout.legend == (
        "Абзац оригинала всегда идёт первым.",
        "Строка с линией слева — перевод этой части абзаца. "
        "Переведены не все абзацы: остальные вы читаете сами, с подсказками или без них. "
        "Старайтесь сначала прочитать абзац оригинала целиком и только потом сверяться с переводом.",
    )
    assert layout.ipa_key == ()
    assert layout.who_is_who == ()


def test_repeated_speech_style_same_speaker_dropped() -> None:
    """The later speech style of one speaker leaves no note. The plan stays.

    `a-drawl` sorts before `z-drawl`, and the second fact is only inferred.
    Reading order still keeps the earlier explicit occurrence, label and body.
    The stored plans are the same objects and still contain both difficulties.
    """
    spoken = "“Softly,” Alice drawled softly."
    first = _block("one", 0, spoken)
    second = _block("two", 0, spoken)
    early = _difficulty(first, "drawled", "speech_style", "z-drawl")
    late = _difficulty(second, "drawled", "speech_style", "a-drawl")
    early_fact = _fact("f-explicit", first, spoken, "Alice", "explicit")
    late_fact = _fact("f-inferred", second, spoken, "Alice", "inferred")
    first_plan = _plan(early, facts=(early_fact,))
    second_plan = _plan(late, facts=(late_fact,))
    first_section = _section("one", 0, "Chapter 1", (first,))
    second_section = _section("two", 1, "Chapter 2", (second,))
    analyses = (
        _analysis("a-one", first_section.id, plan=first_plan),
        _analysis("a-two", second_section.id, plan=second_plan),
    )
    results = {
        first.id: BlockProcessingResult(
            first.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(early, "первое появление", "sentence_rephrase"),),
        ),
        second.id: BlockProcessingResult(
            second.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(late, "повтор", "sentence_rephrase"),),
        ),
    }
    layout = build_book_layout(_inputs(
        _document((first_section, second_section)),
        results,
        analyses,
        concepts=(_concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED),),
    ))

    assert analyses[0].content_plan is first_plan
    assert analyses[1].content_plan is second_plan
    assert [item.id for item in first_plan.difficulties] == ["z-drawl"]
    assert [item.id for item in second_plan.difficulties] == ["a-drawl"]
    assert _labels(layout, first.id) == ("Речь персонажа.",)
    assert _labels(layout, second.id) == ()
    kept = [note for note in _notes(layout, first.id) if note.origin != "speaker"]
    dropped = [note for note in _notes(layout, second.id) if note.origin != "speaker"]
    assert kept[0].labels == ("Речь персонажа.",)
    assert kept[0].difficulty_kinds == ("speech_style",)
    assert kept[0].body == ("первое появление",)
    assert dropped == []
    assert tuple(note.body for note in _notes(layout, second.id)) == (("Говорит Алиса",),)
    assert layout.chapters[first_section.id].units[first.id][0].speaker is None
    assert layout.chapters[second_section.id].units[second.id][0].speaker == "Алиса"


def test_repeated_speech_style_id_scoped_to_its_chapter() -> None:
    """A difficulty id is unique only inside one plan.

    Both chapters use ``d1`` for Alice's speech style, so the later note is
    dropped in chapter 2 and the earlier note stays. Chapter 1's grammar
    difficulty uses ``d3``, the same id as chapter 2's other dropped style.
    That grammar note stays.
    """
    first = _block("one", 0, "Alice drawled softly.")
    grammar_block = _block("one", 1, "She had a curious dream.")
    repeated = _block("two", 0, "Alice drawled softly.")
    again = _block("two", 1, "Alice drawled once more.")
    early = _difficulty(first, "drawled", "speech_style", "d1")
    grammar = _difficulty(grammar_block, "curious", "grammar", "d3")
    late = _difficulty(repeated, "drawled", "speech_style", "d1")
    late_style = _difficulty(again, "drawled", "speech_style", "d3")
    early_fact = _fact("f-early", first, "Alice drawled softly.", "Alice", "explicit")
    late_fact = _fact("f-late", repeated, "Alice drawled softly.", "Alice", "inferred")
    again_fact = _fact("f-again", again, "Alice drawled once more.", "Alice", "explicit")
    first_plan = _plan(early, grammar, facts=(early_fact,))
    second_plan = _plan(late, late_style, facts=(late_fact, again_fact))
    first_section = _section("one", 0, "Chapter 1", (first, grammar_block))
    second_section = _section("two", 1, "Chapter 2", (repeated, again))
    analyses = (
        _analysis("a-one", first_section.id, plan=first_plan),
        _analysis("a-two", second_section.id, plan=second_plan),
    )
    results = {
        first.id: BlockProcessingResult(
            first.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(early, "первое появление", "sentence_rephrase"),),
        ),
        grammar_block.id: BlockProcessingResult(
            grammar_block.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(grammar, "странное время", "sentence_rephrase"),),
        ),
        repeated.id: BlockProcessingResult(
            repeated.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(late, "повтор", "sentence_rephrase"),),
        ),
        again.id: BlockProcessingResult(
            again.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(late_style, "ещё стиль", "sentence_rephrase"),),
        ),
    }
    layout = build_book_layout(_inputs(
        _document((first_section, second_section)),
        results,
        analyses,
    ))

    assert analyses[0].content_plan is first_plan
    assert analyses[1].content_plan is second_plan
    assert [item.id for item in first_plan.difficulties] == ["d1", "d3"]
    assert [item.id for item in second_plan.difficulties] == ["d1", "d3"]
    kept = [note for note in _notes(layout, first.id) if note.origin != "speaker"]
    grammar_notes = [note for note in _notes(layout, grammar_block.id) if note.origin != "speaker"]
    assert kept[0].labels == ("Речь персонажа.",)
    assert kept[0].body == ("первое появление",)
    assert grammar_notes[0].labels == ()
    assert grammar_notes[0].difficulty_kinds == ("grammar",)
    assert grammar_notes[0].body == ("странное время",)
    assert [note for note in _notes(layout, repeated.id) if note.origin != "speaker"] == []
    assert [note for note in _notes(layout, again.id) if note.origin != "speaker"] == []
    assert layout.who_is_who == ()
    assert layout.chapters[first_section.id].review == ()
    assert layout.chapters[second_section.id].review == ()
    assert "review" not in layout.features.sections
    assert layout.features.has_speakers is False


def test_speech_style_without_speaker_not_dropped() -> None:
    """Speech styles that overlap no speaker fact both stay, in either order."""
    rough = _block("one", 0, "The sailor spoke roughly today.")
    again = _block("one", 1, "The sailor spoke roughly again.")
    named = _block("one", 2, "“Yes,” Alice nodded once.")
    first = _difficulty(rough, "roughly", "speech_style", "z-rough")
    second = _difficulty(again, "roughly", "speech_style", "a-rough")
    fact = _fact("f-alice", named, "“Yes,”", "Alice", "inferred")
    plan = _plan(first, second, facts=(fact,))
    section = _section("one", 0, "Chapter 1", (rough, again, named))
    results = {
        rough.id: BlockProcessingResult(
            rough.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(first, "грубо", "sentence_rephrase"),),
        ),
        again.id: BlockProcessingResult(
            again.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(second, "снова грубо", "sentence_rephrase"),),
        ),
        named.id: BlockProcessingResult(named.id, BlockTreatment.READ_AS_IS),
    }
    layout = build_book_layout(_inputs(
        _document((section,)),
        results,
        (_analysis(
            "a-one",
            section.id,
            candidates=(_candidate("a-one", "cand-alice", "Alice", "Алиса"),),
            plan=plan,
        ),),
    ))

    assert _labels(layout, rough.id) == ("Речь персонажа.",)
    assert _labels(layout, again.id) == ("Речь персонажа.",)
    assert layout.chapters[section.id].units[named.id][0].speaker == "Алиса"
    assert layout.chapters[section.id].units[rough.id][0].speaker is None
    assert plan.difficulties[0].id == "a-rough"


def test_explicit_speaker_not_shown_on_unit() -> None:
    """A unit label is only an inferred speaker. An explicit fact on the same line is not.

    Both lines open with a quote and the name is an approved glossary decision.
    """
    line = "“Wait,” she said."
    explicit = _block("one", 0, line)
    inferred = _block("one", 1, line)
    explicit_fact = _fact("f-explicit", explicit, line, "Alice", "explicit")
    inferred_fact = _fact("f-inferred", inferred, line, "Alice", "inferred")
    plan = _plan(facts=(explicit_fact, inferred_fact))
    section = _section("one", 0, "Chapter 1", (explicit, inferred))
    layout = build_book_layout(_inputs(
        _document((section,)),
        analyses=(_analysis("a-one", section.id, plan=plan),),
        concepts=(_concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED),),
    ))
    units = layout.chapters[section.id].units

    assert units[explicit.id][0].speaker is None
    assert units[explicit.id][0].notes == ()
    assert units[inferred.id][0].speaker == "Алиса"
    assert units[inferred.id][0].notes[0].note.body == ("Говорит Алиса",)


def test_unmatched_speaker_not_shown() -> None:
    """Unknown, gender, and an unresolved English name are not speakers.

    An approved decision replaces the source name. A suggested variant does not,
    and the English source is not shown in its place.
    """
    alice = _block("one", 0, "“Wait,” said Alice.")
    hatter = _block("one", 1, "“Wait,” said Hatter.")
    unknown = _block("one", 2, "“Wait,” said someone.")
    alice_fact = _fact("f-alice", alice, "“Wait,” said Alice.", "Alice", "inferred")
    hatter_fact = _fact("f-hatter", hatter, "“Wait,” said Hatter.", "Hatter", "inferred")
    unknown_anchor = _anchor(unknown, "“Wait,” said someone.")
    unknown_fact = ContextFact(
        "f-unknown",
        "speaker",
        (unknown_anchor,),
        unknown_anchor,
        value=None,
        certainty="unknown",
    )
    gender = _fact("f-gender", unknown, "“Wait,” said someone.", "female", "explicit", kind="gender")
    plan = _plan(facts=(alice_fact, hatter_fact, unknown_fact, gender))
    section = _section("one", 0, "Chapter 1", (alice, hatter, unknown))
    concepts = (
        _concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED),
        _concept("c-hatter", "Hatter", "Шляпник", TermStatus.SUGGESTED),
    )
    layout = build_book_layout(_inputs(
        _document((section,)),
        analyses=(_analysis("a-one", section.id, plan=plan),),
        concepts=concepts,
    ))
    units = layout.chapters[section.id].units

    assert units[alice.id][0].speaker == "Алиса"
    assert units[hatter.id][0].speaker is None
    assert units[unknown.id][0].speaker is None
    assert units[unknown.id][0].notes == ()
    assert units[hatter.id][0].speaker != "Шляпник"
    assert all(unit.speaker != "female" for block_units in units.values() for unit in block_units)
    assert "f-alice" in {fact.id for fact in plan.context_facts}
    assert units[alice.id][0].notes[0].number is None
    assert units[alice.id][0].notes[0].note.body == ("Говорит Алиса",)
    assert layout.features.has_speakers is True
    assert layout.features.has_notes is False
    assert "Маленькая цифра после слова" not in "\n".join(layout.legend)


def test_speaker_name_resolved_from_candidate_and_title_prefix_and_parentheses() -> None:
    """A Russian label comes from a decision, a person candidate, or a title plus the rest.

    The whole value, the text before parentheses, and the text inside them are all keys.
    A front-matter candidate and a Latin suggestion are not used. Another target language
    does not invent a Russian title.
    """
    line = "“Hello,” said the person."
    names = (
        "Alice",
        "Bob",
        "Mr. Kent",
        "Mrs. Kent",
        "Miss Kent",
        "Ms. Kent",
        "Uncle Kent",
        "Aunt Kent",
        "Professor Kent",
        "Dr. Kent",
        "Sir Kent",
        "Madam Kent",
        "Lord Kent",
        "Mr. Smith",
        "Petunia Dursley (Mrs Dursley)",
        "Vernon Dursley (Uncle Vernon)",
        "Nobody",
        "Carl",
        "Ghost",
    )
    blocks = tuple(_block("one", index, line) for index in range(len(names)))
    facts = tuple(
        _fact(f"f-{index}", block, "“Hello,”", name, "inferred")
        for index, (block, name) in enumerate(zip(blocks, names, strict=True))
    )
    section = _section("one", 1, "Chapter 1", blocks)
    front_block = _block("front", 0, "Table of contents.")
    front = _section("front", 0, "Contents", (front_block,))
    candidates = (
        _candidate("a-one", "cand-alice", "Alice", "Элис"),
        _candidate("a-one", "cand-bob", "Bob", "Бобби"),
        _candidate("a-one", "cand-kent", "Kent", "Кент"),
        _candidate("a-one", "cand-petunia", "Petunia Dursley", "Петуния Дерсли"),
        _candidate("a-one", "cand-vernon", "Vernon", "Вернон"),
        _candidate("a-one", "cand-carl", "Carl", "Carl"),
    )
    analyses = (
        _analysis(
            "a-front",
            front.id,
            candidates=(_candidate("a-front", "cand-ghost", "Ghost", "Призрак"),),
        ),
        _analysis("a-one", section.id, candidates=candidates, plan=_plan(facts=facts)),
    )
    concepts = (
        _concept("c-bob", "Bob", "Роберт", TermStatus.APPROVED),
        _concept("c-smith", "Mr. Smith", "господин Смит", TermStatus.LOCKED),
    )
    layout = build_book_layout(_inputs(
        _document((front, section)),
        analyses=analyses,
        concepts=concepts,
    ))
    shown = {
        name: layout.chapters[section.id].units[block.id][0].speaker
        for name, block in zip(names, blocks, strict=True)
    }

    assert shown["Alice"] == "Элис"
    assert shown["Bob"] == "Роберт"
    assert shown["Mr. Kent"] == "мистер Кент"
    assert shown["Mrs. Kent"] == "миссис Кент"
    assert shown["Miss Kent"] == "мисс Кент"
    assert shown["Ms. Kent"] == "мис Кент"
    assert shown["Uncle Kent"] == "дядя Кент"
    assert shown["Aunt Kent"] == "тётя Кент"
    assert shown["Professor Kent"] == "профессор Кент"
    assert shown["Dr. Kent"] == "доктор Кент"
    assert shown["Sir Kent"] == "сэр Кент"
    assert shown["Madam Kent"] == "мадам Кент"
    assert shown["Lord Kent"] == "лорд Кент"
    assert shown["Mr. Smith"] == "господин Смит"
    assert shown["Petunia Dursley (Mrs Dursley)"] == "Петуния Дерсли"
    assert shown["Vernon Dursley (Uncle Vernon)"] == "дядя Вернон"
    assert shown["Nobody"] is None
    assert shown["Carl"] is None
    assert shown["Ghost"] is None

    titled = _block("de", 0, line)
    plain = _block("de", 1, line)
    german_section = _section("de", 0, "Chapter 1", (titled, plain))
    german_layout = build_book_layout(_inputs(
        _document((german_section,), target_language="de"),
        analyses=(_analysis(
            "a-de",
            german_section.id,
            candidates=(
                _candidate("a-de", "cand-kent", "Kent", "Кент"),
                _candidate("a-de", "cand-alice", "Alice", "Alicia"),
            ),
            plan=_plan(facts=(
                _fact("f-title", titled, "“Hello,”", "Mr. Kent", "inferred"),
                _fact("f-plain", plain, "“Hello,”", "Alice", "inferred"),
            )),
        ),),
    ))
    german_units = german_layout.chapters[german_section.id].units
    assert german_units[titled.id][0].speaker is None
    assert german_units[plain.id][0].speaker == "Alicia"


def test_secondary_anchor_does_not_attach_speaker_to_other_block() -> None:
    """Only the subject anchor labels its block. A secondary anchor does not."""
    spoken = _block("one", 0, "“Hello,” she said.")
    other = _block("one", 1, "“Hello,” he said.")
    subject = _anchor(spoken, "“Hello,” she said.")
    secondary = _anchor(other, "“Hello,” he said.")
    fact = ContextFact(
        "f-alice",
        "speaker",
        (subject, secondary),
        subject,
        value="Alice",
        certainty="inferred",
    )
    section = _section("one", 0, "Chapter 1", (spoken, other))
    layout = build_book_layout(_inputs(
        _document((section,)),
        analyses=(_analysis(
            "a-one",
            section.id,
            candidates=(_candidate("a-one", "cand-alice", "Alice", "Алиса"),),
            plan=_plan(facts=(fact,)),
        ),),
    ))
    units = layout.chapters[section.id].units

    assert units[spoken.id][0].speaker == "Алиса"
    assert units[other.id][0].speaker is None


def test_mid_paragraph_quote_does_not_attach_speaker_to_narrative_unit() -> None:
    """A quote inside a narrative part is not a label. An opening line and a continued quote are.

    A dash opens a line the same way a quote does. A dash later in the paragraph does not.
    """
    narrative = _block("one", 0, "The boy looked around. “Hello,” he said.")
    opening = _block("one", 1, "“Hello,” he said.")
    dashed = _block("one", 2, "— Hello, she said.")
    buried = _block("one", 3, "She paused — Hello, she said.")
    continued_text = "He shouted, “Hello there, friend."
    cut = continued_text.index("there")
    continued = _block(
        "one",
        4,
        continued_text,
        segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(continued_text), 1)),
    )
    facts = (
        _fact("f-narrative", narrative, "“Hello,”", "Alice", "inferred"),
        _fact("f-opening", opening, "“Hello,”", "Alice", "inferred"),
        _fact("f-dashed", dashed, "— Hello, she said.", "Alice", "inferred"),
        _fact("f-buried", buried, "Hello", "Alice", "inferred"),
        _fact("f-continued", continued, "Hello", "Alice", "inferred"),
    )
    section = _section("one", 0, "Chapter 1", (narrative, opening, dashed, buried, continued))
    continued_result = BlockProcessingResult(
        continued.id,
        BlockTreatment.TRANSLATE,
        translated_text="там.",
        segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        segment_translations=(None, "там."),
    )
    layout = build_book_layout(_inputs(
        _document((section,)),
        {continued.id: continued_result},
        analyses=(_analysis(
            "a-one",
            section.id,
            candidates=(_candidate("a-one", "cand-alice", "Alice", "Алиса"),),
            plan=_plan(facts=facts),
        ),),
    ))
    units = layout.chapters[section.id].units
    continued_units = units[continued.id]

    assert units[narrative.id][0].speaker is None
    assert units[opening.id][0].speaker == "Алиса"
    assert units[dashed.id][0].speaker == "Алиса"
    assert units[buried.id][0].speaker is None
    assert [(unit.start, unit.speaker) for unit in continued_units] == [
        (0, None),
        (cut, "Алиса"),
    ]


def test_name_before_speech_verb_stays_inside_the_opening_remark() -> None:
    """A capital after the quote stays in the remark when a speech verb follows.

    The anchor is the name, not the reply. “Hello,” Harry said. and
    “Hello,” I said. keep the label. “Hello!” Harry left. starts the next
    sentence, and so does a name after that remark has already ended.
    A period before the closing quote ends the sentence, so a speech verb in
    the following sentence does not attach to the first quote.
    """
    said_first = _block("one", 0, "“Hello,” said Harry.")
    name_first = _block("one", 1, "“Hello,” Harry said.")
    pronoun = _block("one", 2, "“Hello,” I said.")
    asked = _block("one", 3, "“Hello?” Harry asked.")
    titled = _block("one", 4, "“Hello,” Mr. Dursley said.")
    left = _block("one", 5, "“Hello!” Harry left.")
    later = _block("one", 6, "“Hello,” Harry said. Harry left.")
    mumbled = _block("one", 7, "‘Funny stuff on the news,’ Mr Dursley mumbled.")
    yelled = _block("one", 8, "‘Mine!’ Dudley yelled.")
    gasped = _block("one", 9, "‘Oh!’ I gasped.")
    roared = _block("one", 10, "‘Out!’ Uncle Vernon roared.")
    period_next = _block(
        "one",
        11,
        "‘Wait here.’ Harry walked to the door and asked, ‘Who is it?’",
    )
    dumbledore = _block(
        "one",
        12,
        "‘It’s lucky it’s dark. I haven’t blushed so much since Madam Pomfrey "
        "told me she liked my new earmuffs.’",
    )
    possessive = _block(
        "one",
        13,
        "‘Check the boys’ room and get ’em now.’ Look at them.",
    )
    split_quote = "‘Wait for me in the hall,’ "
    split_remark = "she demanded from the stairs."
    split_text = split_quote + split_remark
    split_cut = len(split_quote)
    split_block = _block(
        "one",
        14,
        split_text,
        segments=(BlockSegment(0, split_cut, 0), BlockSegment(split_cut, len(split_text), 0)),
    )
    word_close = _block("one", 15, "‘Boys’ he didn’t say. Someone left.")
    called_boys_text = "He called them ‘boys’ and didn’t wait. Then he spoke."
    called_boys_cut = called_boys_text.index("Then")
    called_boys = _block(
        "one",
        16,
        called_boys_text,
        segments=(
            BlockSegment(0, called_boys_cut, 0),
            BlockSegment(called_boys_cut, len(called_boys_text), 1),
        ),
    )
    unclosed_many_text = "‘" + " ".join(["don’t"] * 28) + " wait."
    unclosed_many = _block("one", 17, unclosed_many_text)
    facts = (
        _fact("f-said-first", said_first, "Harry", "Harry", "inferred"),
        _fact("f-name-first", name_first, "Harry", "Harry", "inferred"),
        _fact("f-pronoun", pronoun, "I", "Alice", "inferred"),
        _fact("f-asked", asked, "Harry", "Harry", "inferred"),
        _fact("f-titled", titled, "Dursley", "Dursley", "inferred"),
        _fact("f-left", left, "Harry", "Harry", "inferred"),
        _fact("f-later", later, "Harry left", "Harry", "inferred"),
        _fact("f-mumbled", mumbled, "Mr Dursley mumbled", "Mr. Dursley", "inferred"),
        _fact("f-yelled", yelled, "Dudley yelled", "Dudley", "inferred"),
        _fact("f-gasped", gasped, "I gasped", "Alice", "inferred"),
        _fact("f-roared", roared, "Uncle Vernon roared", "Uncle Vernon", "inferred"),
        _fact("f-period-next", period_next, "Harry walked to the door and asked", "Harry", "inferred"),
        _fact("f-dumbledore", dumbledore, "I haven’t blushed so much", "Dumbledore", "inferred"),
        _fact("f-possessive", possessive, "get ’em now", "Harry", "inferred"),
        _fact("f-split", split_block, "she demanded", "Alice", "inferred"),
        _fact("f-word-close", word_close, "Someone left", "Harry", "inferred"),
        _fact("f-called-boys", called_boys, "Then he spoke", "Harry", "inferred"),
        _fact("f-unclosed-many", unclosed_many, "wait", "Harry", "inferred"),
    )
    section = _section(
        "one",
        0,
        "Chapter 1",
        (
            said_first,
            name_first,
            pronoun,
            asked,
            titled,
            left,
            later,
            mumbled,
            yelled,
            gasped,
            roared,
            period_next,
            dumbledore,
            possessive,
            split_block,
            word_close,
            called_boys,
            unclosed_many,
        ),
    )
    split_result = BlockProcessingResult(
        split_block.id,
        BlockTreatment.TRANSLATE,
        translated_text="— Подожди меня в холле,",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
        segment_translations=("— Подожди меня в холле,", None),
    )
    called_boys_result = BlockProcessingResult(
        called_boys.id,
        BlockTreatment.TRANSLATE,
        translated_text="Затем он заговорил.",
        segment_treatments=(BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE),
        segment_translations=(None, "Затем он заговорил."),
    )
    layout = build_book_layout(_inputs(
        _document((section,)),
        {split_block.id: split_result, called_boys.id: called_boys_result},
        analyses=(_analysis(
            "a-one",
            section.id,
            candidates=(
                _candidate("a-one", "cand-harry", "Harry", "Гарри"),
                _candidate("a-one", "cand-alice", "Alice", "Алиса"),
                _candidate("a-one", "cand-dursley", "Dursley", "Дёрсли"),
                _candidate("a-one", "cand-dudley", "Dudley", "Дадли"),
                _candidate("a-one", "cand-vernon", "Vernon", "Вернон"),
                _candidate("a-one", "cand-dumbledore", "Dumbledore", "Дамблдор"),
            ),
            plan=_plan(facts=facts),
        ),),
    ))
    units = layout.chapters[section.id].units

    assert units[said_first.id][0].speaker == "Гарри"
    assert units[name_first.id][0].speaker == "Гарри"
    assert units[pronoun.id][0].speaker == "Алиса"
    assert units[asked.id][0].speaker == "Гарри"
    assert units[titled.id][0].speaker == "Дёрсли"
    assert units[left.id][0].speaker is None
    assert units[later.id][0].speaker is None
    assert units[mumbled.id][0].speaker == "мистер Дёрсли"
    assert units[yelled.id][0].speaker == "Дадли"
    assert units[gasped.id][0].speaker == "Алиса"
    assert units[roared.id][0].speaker == "дядя Вернон"
    assert units[period_next.id][0].speaker is None
    assert units[dumbledore.id][0].speaker == "Дамблдор"
    assert units[possessive.id][0].speaker == "Гарри"
    assert units[split_block.id][0].speaker == "Алиса"
    assert units[word_close.id][0].speaker is None
    assert [unit.speaker for unit in units[called_boys.id]] == [None, None]
    assert units[unclosed_many.id][0].speaker == "Гарри"


def test_collapsed_whole_block_translation_hides_mid_paragraph_speaker() -> None:
    """One shown translation of a narrative paragraph does not keep a later quote's speaker.

    Segment cuts would open a later part on the quote, but the block is shown as one
    unit from the narrative start.
    """
    narrative = "The boy looked around. "
    quote = "“Hello,” he said."
    filler = "word " * ((PART_MAX_CHARS // 5) + 2)
    text = narrative + filler + quote
    cut = len(narrative + filler)
    assert text[cut:].startswith("“")
    assert len(text) > PART_MAX_CHARS
    block = _block(
        "one",
        0,
        text,
        segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(text), 1)),
    )
    fact = _fact("f-alice", block, "“Hello,”", "Alice", "inferred")
    section = _section("one", 0, "Chapter 1", (block,))
    result = BlockProcessingResult(
        block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Один перевод абзаца.",
    )
    layout = build_book_layout(_inputs(
        _document((section,)),
        {block.id: result},
        analyses=(_analysis(
            "a-one",
            section.id,
            candidates=(_candidate("a-one", "cand-alice", "Alice", "Алиса"),),
            plan=_plan(facts=(fact,)),
        ),),
    ))
    units = layout.chapters[section.id].units[block.id]

    assert [(unit.start, unit.end, unit.speaker) for unit in units] == [
        (0, len(text), None),
    ]


def test_features_collected_for_legend() -> None:
    """Flags, labels, IPA, and section keys come from the finished projection.

    `english_help` hides the lexical-hint label, so it is not a legend label.
    `previously` is set because chapter 1 (non-last content chapter) has a summary.
    Before You Read counts only as a section key.
    """
    poem = _block("one", 0, "Twinkle, twinkle, little bat.", kind=BlockType.POEM)
    grammar_block = _block("one", 1, "She had a curious dream.")
    wordplay_block = _block("one", 2, "The mouse's tale was long.")
    pronounced = _block("one", 3, "The cat sat.")
    speaker_block = _block("one", 4, "“Hello,” said Alice.")
    hinted = _block("one", 5, "Past prose.")
    current = _block("two", 0, "She saw a lantern.")
    grammar = _difficulty(grammar_block, "curious", "grammar", "d-grammar")
    wordplay = _difficulty(
        wordplay_block, "tale", "wordplay", "d-wordplay", help_kind="lexical_hint"
    )
    speaker = _fact("f-alice", speaker_block, "“Hello,” said Alice.", "Alice", "inferred")
    plan = _plan(grammar, wordplay, facts=(speaker,))
    first = _section(
        "one",
        0,
        "Chapter 1",
        (poem, grammar_block, wordplay_block, pronounced, speaker_block, hinted),
    )
    second = _section("two", 1, "Chapter 2", (current,))
    pronunciation = PronunciationAssistance(
        _anchor(pronounced, "cat"),
        "cat",
        "kæt",
        "uk",
        "hint-cat",
        PronunciationEvidence(
            "synthetic-private",
            "private-version",
            "entry-1",
            "private-sense",
            "noun",
            "cat",
            "kæt",
            "Invented test data",
        ),
    )
    translated = "Мерцай."
    hello = "Алиса сказала привет."
    results = {
        poem.id: BlockProcessingResult(
            poem.id,
            BlockTreatment.TRANSLATE,
            translated_text=translated,
            segment_treatments=(BlockTreatment.TRANSLATE,),
            segment_translations=(translated,),
        ),
        grammar_block.id: BlockProcessingResult(
            grammar_block.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(grammar, "странное время", "sentence_rephrase"),),
        ),
        wordplay_block.id: BlockProcessingResult(
            wordplay_block.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(wordplay, "каламбур", "lexical_hint"),),
        ),
        pronounced.id: BlockProcessingResult(
            pronounced.id,
            BlockTreatment.READ_AS_IS,
            pronunciation_assistances=(pronunciation,),
        ),
        speaker_block.id: BlockProcessingResult(
            speaker_block.id,
            BlockTreatment.TRANSLATE,
            translated_text=hello,
            segment_treatments=(BlockTreatment.TRANSLATE,),
            segment_translations=(hello,),
        ),
        hinted.id: BlockProcessingResult(
            hinted.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint("lantern", "фонарь"),),
        ),
    }
    analyses = (
        _analysis(
            "a-one",
            first.id,
            candidates=(_candidate("a-one", "cand-alice", "Alice", "Элис", description="девочка"),),
            plan=plan,
        ),
        _analysis(
            "a-two",
            second.id,
            before=(BeforeYouReadEntry("lantern", "Снова встретится."),),
        ),
    )
    concepts = (_concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED),)
    layout = build_book_layout(_inputs(
        _document((first, second)),
        results,
        analyses,
        concepts,
        reader_level="a2",
        english_help=True,
    ))

    speaker_unit = layout.chapters[first.id].units[speaker_block.id][0]
    assert layout.chapters[first.id].units[poem.id][0].lines is True
    assert layout.chapters[first.id].units[poem.id][0].translation == translated
    assert speaker_unit.speaker == "Алиса"
    assert all(item.note.origin != "speaker" for item in speaker_unit.notes)
    grammar_notes = [note for note in _notes(layout, grammar_block.id) if note.origin != "speaker"]
    assert grammar_notes[0].labels == ()
    assert grammar_notes[0].difficulty_kinds == ("grammar",)
    assert grammar_notes[0].body == ("странное время",)
    assert _labels(layout, wordplay_block.id) == ()
    assert any(note.ipa == "kæt" for note in _notes(layout, pronounced.id))
    assert layout.features == LayoutFeatures(
        has_translation=True,
        has_notes=True,
        labels=frozenset(),
        has_ipa=True,
        has_poems=True,
        has_speakers=False,
        sections=frozenset({"previously", "before_you_read"}),
        fades_translation=False,
    )
    assert "previously" in layout.features.sections
    assert "review" not in layout.features.sections
    assert layout.legend[0] == "Абзац оригинала всегда идёт первым."
    assert "что было раньше" in "\n".join(layout.legend)
    assert "Говорит" not in "\n".join(layout.legend)
    assert layout.ipa_key == ipa_key(("kæt",))
    assert layout.legend == legend_lines(layout.features)
    assert layout.who_is_who == ()
    assert layout.chapters[first.id].who_is_who == ()
    assert layout.chapters[second.id].review == ()
    assert layout.chapters[second.id].who_is_who == ()


def test_front_matter_has_no_extras() -> None:
    """Front matter and notes keep units, no number, and no who-is-who or review.

    The notes page between the chapters does not move chapter 2 off distance 1.
    A person introduced on the contents page is not a book entry.
    """
    front_block = _block("front", 0, "Table of contents.")
    past = _block("one", 0, "Past prose.")
    note = _block("notes", 0, "A footnote.", kind=BlockType.FOOTNOTE)
    current = _block("two", 0, "She saw a lantern.")
    front = _section("front", 0, "Contents", (front_block,))
    first = _section("one", 1, "Chapter 1", (past,))
    notes = _section("notes", 2, "Notes", (note,))
    second = _section("two", 3, "Chapter 2", (current,))
    analyses = (
        _analysis(
            "a-front",
            front.id,
            candidates=(_candidate("a-front", "cand-bob", "Bob", "Боб", description="из оглавления"),),
        ),
        _analysis(
            "a-one",
            first.id,
            candidates=(_candidate("a-one", "cand-alice", "Alice", "Элис", description="девочка"),),
        ),
    )
    results = {
        past.id: BlockProcessingResult(
            past.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint("lantern", "фонарь"),),
        ),
    }
    layout = build_book_layout(_inputs(
        _document((front, first, notes, second)),
        results,
        analyses,
        reader_level="a2",
    ))

    assert tuple(layout.chapters) == (front.id, first.id, notes.id, second.id)
    assert layout.chapters[front.id].number is None
    assert layout.chapters[notes.id].number is None
    assert layout.chapters[first.id].number == 1
    assert layout.chapters[second.id].number == 2
    assert layout.chapters[front.id].who_is_who == ()
    assert layout.chapters[notes.id].who_is_who == ()
    assert layout.chapters[front.id].review == ()
    assert layout.chapters[notes.id].review == ()
    assert layout.chapters[first.id].who_is_who == ()
    assert layout.chapters[first.id].review == ()
    assert layout.chapters[second.id].review == ()
    assert layout.who_is_who == ()
    front_unit = layout.chapters[front.id].units[front_block.id][0]
    note_unit = layout.chapters[notes.id].units[note.id][0]
    assert (front_unit.start, front_unit.end, front_unit.notes) == (0, len(front_block.text), ())
    assert (note_unit.start, note_unit.end, note_unit.notes) == (0, len(note.text), ())
    assert "previously" in layout.features.sections
    assert "who_is_who" not in layout.features.sections
    assert "review" not in layout.features.sections
    assert layout.features.sections == frozenset({"previously"})


def test_book_layout_omits_who_is_who_and_review() -> None:
    """Book assembly drops who-is-who and «Повторим». Speakers are untranslated notes.

    A2 without English help also omits `before_you_read`. A2 with English help,
    another level, and an unset level keep that key. A translated replica keeps
    the name on the unit and does not set `has_speakers`. An untranslated
    replica adds «Говорит …» and does.
    """
    block = _block("one", 0, "Past prose.")
    section = _section("one", 0, "Chapter 1", (block,))
    analysis = _analysis(
        "a-one",
        section.id,
        candidates=(_candidate("a-one", "cand-alice", "Alice", "Элис", description="девочка"),),
        before=(BeforeYouReadEntry("lantern", "Снова встретится."),),
    )
    document = _document((section,))

    def sections(level: str | None, *, english_help: bool) -> frozenset[str]:
        layout = build_book_layout(_inputs(
            document,
            analyses=(analysis,),
            reader_level=level,
            english_help=english_help,
        ))
        assert layout.chapters[section.id].who_is_who == ()
        assert layout.chapters[section.id].review == ()
        assert layout.who_is_who == ()
        assert "who_is_who" not in layout.features.sections
        assert "review" not in layout.features.sections
        assert layout.features.has_speakers is False
        return layout.features.sections

    assert sections("a2", english_help=False) == frozenset()
    assert sections("a2", english_help=True) == frozenset({"before_you_read"})
    assert sections("b1", english_help=False) == frozenset({"before_you_read"})
    assert sections(None, english_help=False) == frozenset({"before_you_read"})

    spoken = "“Hello,” said Alice."
    translated = _block("talk", 0, spoken)
    plain = _block("talk", 1, spoken)
    fact_translated = _fact("f-tr", translated, spoken, "Alice", "inferred")
    fact_plain = _fact("f-plain", plain, spoken, "Alice", "inferred")
    talk = _section("talk", 0, "Chapter 1", (translated, plain))
    hello = "Алиса сказала привет."
    translated_result = BlockProcessingResult(
        translated.id,
        BlockTreatment.TRANSLATE,
        translated_text=hello,
        segment_treatments=(BlockTreatment.TRANSLATE,),
        segment_translations=(hello,),
    )
    both = build_book_layout(_inputs(
        _document((talk,)),
        {translated.id: translated_result},
        analyses=(_analysis(
            "a-talk",
            talk.id,
            candidates=(_candidate("a-talk", "cand-alice", "Alice", "Элис"),),
            plan=_plan(facts=(fact_translated, fact_plain)),
        ),),
        concepts=(_concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED),),
    ))
    translated_unit = both.chapters[talk.id].units[translated.id][0]
    plain_unit = both.chapters[talk.id].units[plain.id][0]
    assert translated_unit.speaker == "Алиса"
    assert translated_unit.notes == ()
    assert plain_unit.speaker == "Алиса"
    assert plain_unit.notes[0].note.origin == "speaker"
    assert plain_unit.notes[0].note.body == ("Говорит Алиса",)
    assert both.features.has_speakers is True
    assert both.who_is_who == ()
    assert both.chapters[talk.id].review == ()
    assert "review" not in both.features.sections

    only_translated = build_book_layout(_inputs(
        _document((_section("talk", 0, "Chapter 1", (translated,)),)),
        {translated.id: translated_result},
        analyses=(_analysis(
            "a-talk",
            talk.id,
            plan=_plan(facts=(fact_translated,)),
        ),),
        concepts=(_concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED),),
    ))
    assert only_translated.chapters[talk.id].units[translated.id][0].speaker == "Алиса"
    assert only_translated.features.has_speakers is False


def test_build_layout_inputs_unifies_data_loading() -> None:
    """One read loads the document, the active run, analyses, and conditions.

    The run may still be active and the analysis incomplete. No active run
    leaves the results empty. Missing conditions leave the level unset and
    English help off. A missing book or document is a Russian error. Without
    `read_transaction`, the write transaction is the boundary.
    """
    plain = _block("chapter", 0, "Just the words.")
    section = _section("chapter", 0, "Chapter 1", (plain,))
    document = _document((section,))
    stored = BlockProcessingResult(plain.id, BlockTreatment.READ_AS_IS)
    extra = BlockProcessingResult("outside", BlockTreatment.READ_AS_IS)
    analysis = _analysis("partial", section.id)
    concept = _concept("c-alice", "Alice", "Алиса", TermStatus.APPROVED)
    assert analysis.coverage_status is CoverageStatus.INCOMPLETE
    assert analysis.content_plan is None
    loaded = _unit_of_work(
        book=_book(),
        document=document,
        run=_run(),
        results=(extra, stored),
        analyses=(analysis,),
        concepts=(concept,),
        conditions=SimpleNamespace(payload={"reader_level": "b1", "english_help": True}),
    )
    inputs = build_layout_inputs(loaded, DOCUMENT_ID)

    assert inputs.document is document
    assert inputs.results == {stored.block_id: stored, extra.block_id: extra}
    assert inputs.analyses == (analysis,)
    assert inputs.concepts == (concept,)
    assert inputs.reader_level == "b1"
    assert inputs.english_help is True
    assert loaded.calls[0] == "read_transaction"
    assert "transaction" not in loaded.calls
    assert f"results.list_for_run:{_run().id}" in loaded.calls
    assert f"runs.active_for_book:{DOCUMENT_ID}" in loaded.calls
    assert loaded.book.status is BookStatus.PROCESSING

    quiet = _unit_of_work(
        book=_book(),
        document=document,
        run=None,
        results=(stored,),
        analyses=(analysis,),
        concepts=(concept,),
        conditions=SimpleNamespace(payload={"reader_level": "c1"}),
    )
    without_run = build_layout_inputs(quiet, DOCUMENT_ID)
    assert without_run.results == {}
    assert without_run.reader_level == "c1"
    assert without_run.english_help is False
    assert without_run.analyses == (analysis,)
    assert not any(call.startswith("results.list_for_run:") for call in quiet.calls)

    bare = _unit_of_work(book=_book(), document=document, conditions=None)
    defaults = build_layout_inputs(bare, DOCUMENT_ID)
    assert defaults.reader_level is None
    assert defaults.english_help is False
    assert defaults.results == {}

    missing_book = _unit_of_work(book=None, document=document)
    with pytest.raises(ApplicationError, match="книги book-1 нет в хранилище"):
        build_layout_inputs(missing_book, DOCUMENT_ID)

    missing_document = _unit_of_work(book=_book(), document=None)
    with pytest.raises(ApplicationError, match="книга book-1 не импортирована"):
        build_layout_inputs(missing_document, DOCUMENT_ID)

    writing = _unit_of_work(book=_book(), document=document, readable=False)
    assert build_layout_inputs(writing, DOCUMENT_ID).document is document
    assert writing.calls[0] == "transaction"
    assert not hasattr(_UnitOfWork, "read_transaction") or "read_transaction" not in writing.calls


def test_repeated_speech_style_drops_language_notes_and_hints_backed_by_evidence() -> None:
    """Repeated speech_style removes Russian language_notes and glossary_hints, not only assistances."""
    from linguary.domain import LanguageNote
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    first = _block("one", 0, "‘Yeh’ll be late,’ Hagrid said.")
    second = _block("two", 0, "‘Yeh’ll see,’ Hagrid said.")
    early = _difficulty(first, "Yeh’ll", "speech_style", "d-early")
    late = _difficulty(second, "Yeh’ll", "speech_style", "d-late")
    kept_grammar = _difficulty(second, "see", "grammar", "d-gram")
    first_fact = _fact("f1", first, first.text, "Hagrid", "explicit")
    second_fact = _fact("f2", second, second.text, "Hagrid", "explicit")
    first_plan = _plan(early, facts=(first_fact,))
    second_plan = _plan(late, kept_grammar, facts=(second_fact,))
    first_section = _section("one", 0, "Chapter 1", (first,))
    second_section = _section("two", 1, "Chapter 2", (second,))

    early_note = LanguageNote("n1", first.id, "Yeh’ll", "Просторечное yeh = you.")
    late_note = LanguageNote("n2", second.id, "Yeh’ll", "Повтор про yeh = you.")
    kept_note = LanguageNote("n3", second.id, "see", "Будущее время.")
    late_hint = GlossaryHint("Yeh’ll", "ты будешь", "Повтор про диалект.")

    def _note_key(note: LanguageNote) -> str:
        return artifact_key(
            "language_note.note",
            note.block_id,
            {
                "original_excerpt": note.original_excerpt,
                "note": note.note,
                "translated_excerpt": note.translated_excerpt,
            },
        )

    def _hint_key(block_id: str, hint: GlossaryHint) -> str:
        return artifact_key(
            "glossary_hint.note",
            block_id,
            {"original": hint.original, "translation": hint.translation, "note": hint.note},
        )

    results = {
        first.id: BlockProcessingResult(
            first.id,
            BlockTreatment.TRANSLATE,
            translated_text="— Мы опоздали, — сказал Хагрид.",
            language_notes=(early_note,),
            content_evidence=ContentEvidence(
                first_plan.sha256,
                (
                    HelpEvidence(
                        early.id,
                        early.required_points[0].id,
                        "language_note.note",
                        first.id,
                        _note_key(early_note),
                        early_note.note,
                    ),
                ),
            ),
        ),
        second.id: BlockProcessingResult(
            second.id,
            BlockTreatment.TRANSLATE,
            translated_text="— Сам увидишь, — сказал Хагрид.",
            glossary_hints=(late_hint,),
            language_notes=(late_note, kept_note),
            content_evidence=ContentEvidence(
                second_plan.sha256,
                (
                    HelpEvidence(
                        late.id,
                        late.required_points[0].id,
                        "language_note.note",
                        second.id,
                        _note_key(late_note),
                        late_note.note,
                    ),
                    HelpEvidence(
                        late.id,
                        late.required_points[0].id,
                        "glossary_hint.note",
                        second.id,
                        _hint_key(second.id, late_hint),
                        late_hint.note or "",
                    ),
                    HelpEvidence(
                        kept_grammar.id,
                        kept_grammar.required_points[0].id,
                        "language_note.note",
                        second.id,
                        _note_key(kept_note),
                        kept_note.note,
                    ),
                ),
            ),
        ),
    }

    layout = build_book_layout(
        _inputs(
            _document((first_section, second_section)),
            results,
            (
                _analysis("a1", first_section.id, plan=first_plan),
                _analysis("a2", second_section.id, plan=second_plan),
            ),
        )
    )

    first_notes = _notes(layout, first.id)
    second_notes = _notes(layout, second.id)
    assert len(first_notes) == 1
    assert first_notes[0].labels == ("Речь персонажа.",)
    assert [note.body for note in second_notes] == [("Будущее время.",)]
    assert [note.labels for note in second_notes] == [()]


def test_repeated_speech_style_on_glossary_only_keeps_translation_and_shared_evidence() -> None:
    """GLOSSARY_ONLY does not crash with BlockResultError; shared evidence and unbacked notes stay."""
    from linguary.domain import LanguageNote
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    first = _block("one", 0, "‘An’ yeh’ll be late,’ Hagrid said.")
    second = _block("two", 0, "‘Yeh’ll see,’ Hagrid said.")
    third = _block("two", 1, "‘An’ don’t forget,’ Hagrid said.")
    early = _difficulty(first, "An’ yeh’ll", "speech_style", "d-early")
    late_glossary = _difficulty(second, "Yeh’ll", "speech_style", "d-late-g", help_kind="lexical_hint")
    late_shared_style = _difficulty(third, "An’", "speech_style", "d-late-s")
    kept_idiom = _difficulty(third, "don’t forget", "idiom", "d-keep-idiom")
    first_plan = _plan(early, facts=(_fact("f1", first, first.text, "Hagrid", "explicit"),))
    second_plan = _plan(
        late_glossary,
        late_shared_style,
        kept_idiom,
        facts=(
            _fact("f2", second, second.text, "Hagrid", "explicit"),
            _fact("f3", third, third.text, "Hagrid", "explicit"),
        ),
    )
    first_section = _section("one", 0, "Chapter 1", (first,))
    second_section = _section("two", 1, "Chapter 2", (second, third))

    yeh_hint = GlossaryHint("Yeh’ll", "ты будешь", "Повтор про диалект Хагрида.")
    shared_note = LanguageNote("n-shared", third.id, "don’t forget", "Устойчивое напоминание.")
    unbacked_note = LanguageNote("n-unbacked", third.id, "An’", "Заметка без evidence.")

    def _hint_tr_key(block_id: str, hint: GlossaryHint) -> str:
        return artifact_key(
            "glossary_hint.translation",
            block_id,
            {"original": hint.original, "translation": hint.translation, "note": hint.note},
        )

    def _note_key(note: LanguageNote) -> str:
        return artifact_key(
            "language_note.note",
            note.block_id,
            {
                "original_excerpt": note.original_excerpt,
                "note": note.note,
                "translated_excerpt": note.translated_excerpt,
            },
        )

    results = {
        first.id: BlockProcessingResult(
            first.id,
            BlockTreatment.TRANSLATE,
            translated_text="— Мы опоздали, — сказал Хагрид.",
            language_notes=(LanguageNote("n1", first.id, "We was", "Просторечие."),),
        ),
        second.id: BlockProcessingResult(
            second.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(yeh_hint,),
            content_evidence=ContentEvidence(
                second_plan.sha256,
                (
                    HelpEvidence(
                        late_glossary.id,
                        late_glossary.required_points[0].id,
                        "glossary_hint.translation",
                        second.id,
                        _hint_tr_key(second.id, yeh_hint),
                        yeh_hint.translation,
                    ),
                ),
            ),
        ),
        third.id: BlockProcessingResult(
            third.id,
            BlockTreatment.TRANSLATE,
            translated_text="— И не забудь, — сказал Хагрид.",
            language_notes=(shared_note, unbacked_note),
            content_evidence=ContentEvidence(
                second_plan.sha256,
                (
                    HelpEvidence(
                        late_shared_style.id,
                        late_shared_style.required_points[0].id,
                        "language_note.note",
                        third.id,
                        _note_key(shared_note),
                        shared_note.note,
                    ),
                    HelpEvidence(
                        kept_idiom.id,
                        kept_idiom.required_points[0].id,
                        "language_note.note",
                        third.id,
                        _note_key(shared_note),
                        shared_note.note,
                    ),
                ),
            ),
        ),
    }

    layout = build_book_layout(
        _inputs(
            _document((first_section, second_section)),
            results,
            (
                _analysis("a1", first_section.id, plan=first_plan),
                _analysis("a2", second_section.id, plan=second_plan),
            ),
        )
    )

    second_notes = _notes(layout, second.id)
    assert len(second_notes) == 1
    assert second_notes[0].term == "Yeh’ll"
    assert second_notes[0].body == ("ты будешь",)
    assert second_notes[0].labels == ()

    third_notes = _notes(layout, third.id)
    assert [note.body for note in third_notes] == [
        ("Заметка без evidence.",),
        ("Устойчивое напоминание.",),
    ]


def test_display_translation_typography_applies_only_to_russian_target_language() -> None:
    """German quotes and spaced en-dashes are normalized for Russian display, preserved for German."""
    block = _block("one", 0, "He said hello.")
    section = _section("one", 0, "Chapter 1", (block,))
    raw_text = (
        "Сказал: „Привет“ и „Ещё” – потом «Он назвал его „волшебником“» "
        "и „обрыв и „Точка“ и требовать\u00adся."
    )
    result = BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, translated_text=raw_text)

    ru_layout = build_book_layout(
        _inputs(_document((section,), target_language="ru"), {block.id: result})
    )
    ru_translation = ru_layout.chapters[section.id].units[block.id][0].translation
    assert ru_translation == (
        "Сказал: «Привет» и «Ещё» — потом «Он назвал его „волшебником“» "
        "и „обрыв и «Точка» и требоваться."
    )

    de_text = "Er sagte: „Hallo“ – und „Welt” – und tren\u00adnen."
    de_result = BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, translated_text=de_text)
    de_layout = build_book_layout(
        _inputs(_document((section,), target_language="de"), {block.id: de_result})
    )
    de_translation = de_layout.chapters[section.id].units[block.id][0].translation
    assert de_translation == "Er sagte: „Hallo“ – und „Welt” – und trennen."


def test_book_layout_includes_previously_only_with_multiple_content_chapters() -> None:
    """`previously` appears only with >= 2 content chapters and a non-last chapter summary."""
    front = _section("front", 0, "Contents", (_block("front", 0, "Table of contents."),))
    ch1 = _section("ch1", 1, "Chapter 1", (_block("ch1", 0, "First chapter text."),))
    notes = _section("notes", 2, "Notes", (_block("notes", 0, "Footnote.", kind=BlockType.FOOTNOTE),))
    ch2 = _section("ch2", 3, "Chapter 2", (_block("ch2", 0, "Second chapter text."),))

    # Only 1 content chapter (+ front matter + notes-only) -> no `previously`.
    single = build_book_layout(_inputs(
        _document((front, ch1, notes)),
        analyses=(
            _analysis("a-front", front.id, chapter_summary="Оглавление."),
            _analysis("a-ch1", ch1.id, chapter_summary="Первая глава."),
            _analysis("a-notes", notes.id, chapter_summary="Примечания."),
        ),
    ))
    assert "previously" not in single.features.sections
    assert "что было раньше" not in "\n".join(single.legend)

    # 2 content chapters, but only the LAST content chapter has a non-empty summary -> no `previously`.
    only_last_summary = build_book_layout(_inputs(
        _document((ch1, ch2)),
        analyses=(
            _analysis("a-ch1", ch1.id, chapter_summary="   "),
            _analysis("a-ch2", ch2.id, chapter_summary="Итог последней главы."),
        ),
    ))
    assert "previously" not in only_last_summary.features.sections
    assert "что было раньше" not in "\n".join(only_last_summary.legend)

    # 2 content chapters, first (non-last) content chapter has a non-empty summary -> `previously` set.
    multi = build_book_layout(_inputs(
        _document((front, ch1, notes, ch2)),
        analyses=(
            _analysis("a-ch1", ch1.id, chapter_summary="Гарри получает письмо."),
            _analysis("a-ch2", ch2.id, chapter_summary=""),
        ),
    ))
    assert "previously" in multi.features.sections
    assert "who_is_who" not in multi.features.sections
    assert "review" not in multi.features.sections
    assert "«Previously in…» — что было раньше" in "\n".join(multi.legend)


def test_book_layout_three_stage_vocabulary_deduplication_and_11_grammar_caps() -> None:
    """3-stage book-wide glossary deduplication and 11 grammar pattern caps work without mutating results."""
    from linguary.domain import LanguageNote
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    # Build 7 content chapters with spacing blocks so we can control both 20-block window and chapter gaps.
    ch1_blocks = [
        _block("c1", 0, "He wore a heavy cloak."),
        _block("c1", 1, "The cloak swirled near a lantern."),
        _block("c1", 2, "Another cloak hung nearby."),
    ]
    # Add 22 filler blocks in ch1 so block index jumps > 20 before ch2.
    ch1_blocks.extend(_block("c1", idx, f"Filler sentence {idx}.") for idx in range(3, 25))
    ch2_blocks = [
        _block("c2", 0, "He grabbed the cloak in the hall."),
        _block("c2", 1, "She saw the cloak on the chair."),
    ]
    ch2_blocks.extend(_block("c2", idx, f"Filler c2 {idx}.") for idx in range(2, 24))
    ch3_blocks = [
        _block("c3", 0, "The cloak fluttered in the wind."),
    ]
    ch3_blocks.extend(_block("c3", idx, f"Filler c3 {idx}.") for idx in range(1, 23))
    ch4_blocks = [
        _block("c4", 0, "A cloak and a wand lay there."),
        _block("c4", 1, "Only the cloak remained."),
    ]
    ch5_blocks = [_block("c5", 0, "Quiet chapter five.")]
    ch6_blocks = [_block("c6", 0, "Quiet chapter six.")]
    ch7_blocks = [_block("c7", 0, "Quiet chapter seven.")]
    ch8_blocks = [_block("c8", 0, "The cloak appeared again after a long break.")]

    # Attach grammar language notes across chapters 1, 2, 3, 5, 7 to test all 11 closed regex keys.
    grammar_cases = (
        ("used_to", "used to live", "Конструкция used to обозначает привычку в прошлом."),
        ("modal_must_have", "must have been", "Модальный глагол must have + V3: наверняка был."),
        ("modal_would_have", "would have done", "would have + V3: сделал бы в прошлом."),
        ("modal_could_have", "could have sworn", "could have + V3: мог бы поклясться."),
        ("modal_should_have", "should have known", "should have + V3: следовало знать."),
        ("modal_might_have", "might have heard", "might have + V3: возможно, слышал."),
        ("past_perfect", "had already left", "Past Perfect: действие случилось раньше в прошлом."),
        ("wish_past", "I wish I were", "После wish используется прошедшее время were."),
        ("passive_infinitive", "to be seen", "Пассивный инфинитив to be + V3: быть увиденным."),
        ("double_negative", "didn't know nothing", "Двойное отрицание в просторечной речи."),
        ("preposition_stranding", "talking to", "Предлог в конце вопроса (preposition stranding)."),
    )

    grammar_blocks: dict[tuple[int, str], DocumentBlock] = {}
    extra_sections: dict[int, list[DocumentBlock]] = {1: ch1_blocks, 2: ch2_blocks, 3: ch3_blocks, 4: ch4_blocks, 5: ch5_blocks, 6: ch6_blocks, 7: ch7_blocks, 8: ch8_blocks}
    results: dict[str, BlockProcessingResult] = {}

    # Vocabulary hints for `cloak`:
    cloak_hint_1 = GlossaryHint("cloak", "плащ", "Длинная накидка с капюшоном.")
    lantern_hint = GlossaryHint("lantern", "фонарь", "Переносной фонарь.")
    cloak_hint_2_tr = GlossaryHint("cloak", "плащ", "Повтор в переведённом блоке.")
    cloak_hint_2_gl = GlossaryHint("cloak", "плащ", "Пояснение должно исчезнуть при 2-3-м появлении.")
    cloak_hint_3_gl = GlossaryHint("cloak", "плащ", "Пояснение при 3-м появлении.")
    cloak_hint_4_with_other = GlossaryHint("cloak", "плащ", "4-е появление с другой подсказкой.")
    wand_hint = GlossaryHint("wand", "палочка", "Волшебная палочка.")
    cloak_hint_4_sole = GlossaryHint("cloak", "плащ", "Единственная подсказка в GLOSSARY_ONLY.")
    cloak_hint_reset = GlossaryHint("cloak", "плащ", "Возвращение после перерыва > 3 глав.")

    # Attach HelpEvidence on c2_blocks[1] (stripped hint) to verify artifact_key preservation through book.py.
    c2_gl_block = ch2_blocks[1]
    c2_diff = _difficulty(c2_gl_block, "cloak", "vocabulary", "d-cloak-c2")
    c2_plan = _plan(c2_diff)
    c2_hint_key = artifact_key(
        "glossary_hint.note",
        c2_gl_block.id,
        {"original": cloak_hint_2_gl.original, "translation": cloak_hint_2_gl.translation, "note": cloak_hint_2_gl.note},
    )

    # c1[0]: 1st appearance of cloak -> FULL
    results[ch1_blocks[0].id] = BlockProcessingResult(
        ch1_blocks[0].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_1,),
    )
    # c1[1]: within 20 blocks, GLOSSARY_ONLY with another hint -> cloak suppressed, lantern kept
    results[ch1_blocks[1].id] = BlockProcessingResult(
        ch1_blocks[1].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_1, lantern_hint),
    )
    # c1[2]: within 20 blocks, GLOSSARY_ONLY with ONLY cloak -> MUST keep >= 1 compact hint!
    results[ch1_blocks[2].id] = BlockProcessingResult(
        ch1_blocks[2].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_1,),
    )
    # c2[0]: > 20 blocks later, 2nd appearance in TRANSLATE -> suppressed!
    results[ch2_blocks[0].id] = BlockProcessingResult(
        ch2_blocks[0].id,
        BlockTreatment.TRANSLATE,
        translated_text="Он схватил плащ в прихожей.",
        glossary_hints=(cloak_hint_2_tr,),
    )
    # c2[1]: > 20 blocks from last shown (c1[2]), 3rd appearance in GLOSSARY_ONLY -> stripped (translation only, artifact_key kept!)
    results[c2_gl_block.id] = BlockProcessingResult(
        c2_gl_block.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_2_gl,),
        content_evidence=ContentEvidence(
            c2_plan.sha256,
            (
                HelpEvidence(
                    c2_diff.id,
                    c2_diff.required_points[0].id,
                    "glossary_hint.note",
                    c2_gl_block.id,
                    c2_hint_key,
                    cloak_hint_2_gl.note or "",
                ),
            ),
        ),
    )
    # c3[0]: > 20 blocks later, 4th appearance (last shown c2, gap=1 <= 3) in TRANSLATE -> suppressed!
    results[ch3_blocks[0].id] = BlockProcessingResult(
        ch3_blocks[0].id,
        BlockTreatment.TRANSLATE,
        translated_text="Плащ развевался на ветру.",
        glossary_hints=(cloak_hint_3_gl,),
    )
    # c4[0]: 4th+ appearance (last shown c2, gap=2 <= 3) in GLOSSARY_ONLY with another hint -> cloak suppressed, wand kept
    results[ch4_blocks[0].id] = BlockProcessingResult(
        ch4_blocks[0].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_4_with_other, wand_hint),
    )
    # c4[1]: 4th+ appearance in GLOSSARY_ONLY with ONLY cloak -> fallback keeps >= 1 compact hint!
    results[ch4_blocks[1].id] = BlockProcessingResult(
        ch4_blocks[1].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_4_sole,),
    )
    # c8[0]: after > 3 chapters gap (8 - 4 = 4 > 3) -> reset, shown FULL again!
    results[ch8_blocks[0].id] = BlockProcessingResult(
        ch8_blocks[0].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_hint_reset,),
    )

    # Add grammar notes in chapters 1, 2, 3, 5, 7 (plus a duplicate in ch1 for used_to).
    for ch_num in (1, 2, 3, 5, 7):
        bucket = extra_sections[ch_num]
        for g_key, excerpt, explanation in grammar_cases:
            blk = _block(f"c{ch_num}", len(bucket), f"Sample {excerpt} text.")
            bucket.append(blk)
            grammar_blocks[(ch_num, g_key)] = blk
            results[blk.id] = BlockProcessingResult(
                blk.id,
                BlockTreatment.READ_AS_IS,
                language_notes=(LanguageNote(f"ln-{ch_num}-{g_key}", blk.id, excerpt, explanation),),
            )
    # Add an extra same-chapter used_to in ch1 to verify same-chapter suppression (gap 0 < 2).
    same_ch_used_to = _block("c1", len(ch1_blocks), "He used to walk.")
    ch1_blocks.append(same_ch_used_to)
    results[same_ch_used_to.id] = BlockProcessingResult(
        same_ch_used_to.id,
        BlockTreatment.READ_AS_IS,
        language_notes=(LanguageNote("ln-c1-used-dup", same_ch_used_to.id, "used to walk", "Конструкция used to — привычка."),),
    )

    sections = tuple(
        _section(f"c{n}", n - 1, f"Chapter {n}", extra_sections[n])
        for n in range(1, 9)
    )
    analyses = (_analysis("a-c2", "c2", plan=c2_plan),)
    layout = build_book_layout(_inputs(_document(sections), results, analyses))

    # --- Vocabulary assertions ---
    # c1[0]: 1st appearance -> full translation + note
    assert [(n.term, n.body) for n in _notes(layout, ch1_blocks[0].id)] == [
        ("cloak", ("плащ", "Длинная накидка с капюшоном.")),
    ]
    # c1[1]: within 20 blocks -> cloak suppressed, lantern kept full
    assert [(n.term, n.body) for n in _notes(layout, ch1_blocks[1].id)] == [
        ("lantern", ("фонарь", "Переносной фонарь.")),
    ]
    # c1[2]: within 20 blocks, sole hint in untranslated GLOSSARY_ONLY -> rescued as compact (translation only)
    assert [(n.term, n.body) for n in _notes(layout, ch1_blocks[2].id)] == [
        ("cloak", ("плащ",)),
    ]
    # c2[0]: 2nd appearance in TRANSLATE -> suppressed
    assert _notes(layout, ch2_blocks[0].id) == ()
    # c2[1]: 3rd appearance in GLOSSARY_ONLY -> stripped note, preserved artifact_key -> difficulty_kinds=("vocabulary",)
    c2_notes = _notes(layout, c2_gl_block.id)
    assert [(n.term, n.body, n.difficulty_kinds) for n in c2_notes] == [
        ("cloak", ("плащ",), ("vocabulary",)),
    ]
    # c3[0]: 4th appearance in TRANSLATE (gap <= 3) -> suppressed
    assert _notes(layout, ch3_blocks[0].id) == ()
    # c4[0]: 4th+ in GLOSSARY_ONLY with wand -> cloak suppressed, wand kept
    assert [(n.term, n.body) for n in _notes(layout, ch4_blocks[0].id)] == [
        ("wand", ("палочка", "Волшебная палочка.")),
    ]
    # c4[1]: 4th+ in GLOSSARY_ONLY with ONLY cloak -> fallback keeps compact translation
    assert [(n.term, n.body) for n in _notes(layout, ch4_blocks[1].id)] == [
        ("cloak", ("плащ",)),
    ]
    # c8[0]: gap 8 - 4 = 4 > 3 chapters -> reset, shown full again!
    assert [(n.term, n.body) for n in _notes(layout, ch8_blocks[0].id)] == [
        ("cloak", ("плащ", "Возвращение после перерыва > 3 глав.")),
    ]
    # Stored results are never mutated.
    assert results[c2_gl_block.id].glossary_hints == (cloak_hint_2_gl,)
    assert results[ch2_blocks[0].id].glossary_hints == (cloak_hint_2_tr,)

    # --- 11 Grammar keys assertions ---
    # Same-chapter duplicate of used_to is suppressed.
    assert _notes(layout, same_ch_used_to.id) == ()
    # used_to: cap 2, min gap 2 -> shown in ch1 (1), suppressed ch2, shown ch3 (2), suppressed ch5 & ch7.
    assert len(_notes(layout, grammar_blocks[(1, "used_to")].id)) == 1
    assert len(_notes(layout, grammar_blocks[(2, "used_to")].id)) == 0
    assert len(_notes(layout, grammar_blocks[(3, "used_to")].id)) == 1
    assert len(_notes(layout, grammar_blocks[(5, "used_to")].id)) == 0
    assert len(_notes(layout, grammar_blocks[(7, "used_to")].id)) == 0
    # All other 10 keys (including 5 distinct modal perfects): cap 3, min gap 2 ->
    # shown in ch1 (1), suppressed ch2 (gap 1), shown ch3 (2), shown ch5 (3), suppressed ch7 (cap 3).
    for g_key, _, _ in grammar_cases[1:]:
        assert len(_notes(layout, grammar_blocks[(1, g_key)].id)) == 1, g_key
        assert len(_notes(layout, grammar_blocks[(2, g_key)].id)) == 0, g_key
        assert len(_notes(layout, grammar_blocks[(3, g_key)].id)) == 1, g_key
        assert len(_notes(layout, grammar_blocks[(5, g_key)].id)) == 1, g_key
        assert len(_notes(layout, grammar_blocks[(7, g_key)].id)) == 0, g_key


def test_book_layout_keeps_first_two_speech_style_and_unique_idioms() -> None:
    """First 2 speech_style notes in speaker's first scene/chapter stay, plus unique dialect lexemes later."""
    from linguary.domain.content_plan import Scene

    b1 = _block("one", 0, "‘Yeh’re a wizard,’ Hagrid said.")
    b2 = _block("one", 1, "‘Yeh’ll see,’ Hagrid said.")
    b3 = _block("one", 2, "‘Yeh’d better hurry,’ Hagrid said.")
    b4 = _block("one", 3, "‘Gallopin’ Gorgons!’ Hagrid cried.")
    b5 = _block("two", 0, "‘Yeh know what I mean,’ Hagrid said.")
    b6 = _block("two", 1, "‘Codswallop!’ Hagrid barked.")
    b7 = _block("three", 0, "‘Codswallop again!’ Hagrid laughed.")

    d1 = _difficulty(b1, "Yeh’re", "speech_style", "d-s1")
    d2 = _difficulty(b2, "Yeh’ll", "speech_style", "d-s2")
    d3 = _difficulty(b3, "Yeh’d", "speech_style", "d-s3")
    d4 = _difficulty(b4, "Gallopin’ Gorgons", "speech_style", "d-s4")
    d5 = _difficulty(b5, "Yeh", "speech_style", "d-s5")
    d6 = _difficulty(b6, "Codswallop", "speech_style", "d-s6")
    d7 = _difficulty(b7, "Codswallop", "speech_style", "d-s7")

    p1 = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(d1, d2, d3, d4),
        context_facts=(
            _fact("f1", b1, b1.text, "Hagrid", "explicit"),
            _fact("f2", b2, b2.text, "Hagrid", "explicit"),
            _fact("f3", b3, b3.text, "Hagrid", "explicit"),
            _fact("f4", b4, b4.text, "Hagrid", "explicit"),
        ),
        scenes=(
            Scene("sc-1", 0, b1.id, b4.id, "Hagrid arrives at the hut."),
        ),
    )
    p2 = _plan(
        d5,
        d6,
        facts=(
            _fact("f5", b5, b5.text, "Hagrid", "explicit"),
            _fact("f6", b6, b6.text, "Hagrid", "explicit"),
        ),
    )
    p3 = _plan(d7, facts=(_fact("f7", b7, b7.text, "Hagrid", "explicit"),))

    s1 = _section("one", 0, "Chapter 1", (b1, b2, b3, b4))
    s2 = _section("two", 1, "Chapter 2", (b5, b6))
    s3 = _section("three", 2, "Chapter 3", (b7,))

    results = {
        blk.id: BlockProcessingResult(
            blk.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(diff, f"пояснение {diff.id}", "sentence_rephrase"),),
        )
        for blk, diff in ((b1, d1), (b2, d2), (b3, d3), (b4, d4), (b5, d5), (b6, d6), (b7, d7))
    }

    layout = build_book_layout(
        _inputs(
            _document((s1, s2, s3)),
            results,
            (
                _analysis("a1", s1.id, plan=p1),
                _analysis("a2", s2.id, plan=p2),
                _analysis("a3", s3.id, plan=p3),
            ),
        )
    )

    # First 2 in Hagrid's first scene stay (even though both use `yeh`).
    assert _labels(layout, b1.id) == ("Речь персонажа.",)
    assert _labels(layout, b2.id) == ("Речь персонажа.",)
    # 3rd in first scene repeats `yeh` -> dropped!
    assert _notes(layout, b3.id) == ()
    # 4th in first scene has unique dialect idiom (`Gallopin' Gorgons`) -> kept!
    assert _labels(layout, b4.id) == ("Речь персонажа.",)
    # Chapter 2: `Yeh` repeats already-shown `yeh` -> dropped!
    assert _notes(layout, b5.id) == ()
    # Chapter 2: `Codswallop` is a new unique dialect lexeme -> kept!
    assert _labels(layout, b6.id) == ("Речь персонажа.",)
    # Chapter 3: `Codswallop` repeats already-shown `codswallop` -> dropped!
    assert _notes(layout, b7.id) == ()


def test_vocab_dedup_no_translate_flip_or_double_count_and_mixed_segment_preserved() -> None:
    """Covers review findings 1, 2, and 3:
    1) Suppressed TRANSLATE never flips to full on the next block and adjacent TRANSLATE
       blocks do not double-count occurrences to starve a following GLOSSARY_ONLY block.
    2) Mixed TRANSLATE + GLOSSARY_ONLY blocks strip (not suppress) 2nd-3rd hints on the
       untranslated segment and rescue >= 1 hint on the untranslated segment inside windows.
    3) Russian preposition_stranding phrasing naming the preposition (e.g. «to», about, (with))
       is matched and capped without requiring the English label.
    """
    from linguary.domain import LanguageNote

    # --- Scenario 1a & 1b: adjacent TRANSLATE blocks + Ch1 -> Ch2 -> Ch5 reset ---
    c1_blocks = [_block("c1", 0, "He grabbed the cloak and the lantern.")]
    c1_blocks.extend(_block("c1", i, f"Filler c1 {i}.") for i in range(1, 23))

    c2_tr_1 = _block("c2", 0, "The cloak billowed.")
    c2_tr_2 = _block("c2", 1, "Another cloak passed.")
    c2_gl_after = _block("c2", 2, "A cloak lay near a wand.")
    c2_tr_lantern = _block("c2", 3, "The lantern flickered.")
    c2_blocks = [c2_tr_1, c2_tr_2, c2_gl_after, c2_tr_lantern]
    c2_blocks.extend(_block("c2", i, f"Filler c2 {i}.") for i in range(4, 26))

    # Mixed segment block in ch3: segment 0 = TRANSLATE ("He opened the goblet."),
    # segment 1 = GLOSSARY_ONLY (" The lantern glowed.")
    mixed_text_1 = "He opened the goblet. The lantern glowed."
    cut_1 = mixed_text_1.index(" The lantern")
    c3_mixed_strip = _block(
        "c3",
        0,
        mixed_text_1,
        segments=(
            BlockSegment(0, cut_1, 0),
            BlockSegment(cut_1, len(mixed_text_1), 1),
        ),
    )
    # Mixed block within 20-block window right after c3_mixed_strip:
    # segment 0 = TRANSLATE with brand-new hint "quill", segment 1 = GLOSSARY_ONLY with "lantern" (within window!)
    mixed_text_2 = "He held a quill. The lantern shone."
    cut_2 = mixed_text_2.index(" The lantern")
    c3_mixed_rescue = _block(
        "c3",
        1,
        mixed_text_2,
        segments=(
            BlockSegment(0, cut_2, 0),
            BlockSegment(cut_2, len(mixed_text_2), 1),
        ),
    )
    c3_blocks = [c3_mixed_strip, c3_mixed_rescue]

    c4_blocks = [_block("c4", 0, "Quiet chapter four.")]

    # Ch 5: "cauldron" was shown in Ch 1 (count=1), suppressed in Ch 2 (count=2),
    # and now appears in two adjacent TRANSLATE blocks in Ch 5 (gap 5 - 1 = 4 > 3).
    # First Ch 5 block should reintroduce cleanly; second adjacent block must stay suppressed!
    c5_tr_1 = _block("c5", 0, "A cauldron gleamed.")
    c5_tr_2 = _block("c5", 1, "The cauldron fell.")
    c5_blocks = [c5_tr_1, c5_tr_2]

    c6_blocks = [_block("c6", 0, "Quiet chapter six.")]
    c7_blocks = [_block("c7", 0, "Quiet chapter seven.")]

    # Add Ch 1 & Ch 2 goblet + cauldron blocks for Scenario 1a & 2.
    c1_goblet = _block("c1", len(c1_blocks), "He lifted a goblet and a cauldron.")
    c1_blocks.insert(1, c1_goblet)
    c2_goblet = _block("c2", len(c2_blocks), "She took the goblet and the cauldron.")
    c2_blocks.insert(4, c2_goblet)

    # Russian preposition_stranding notes without English label across chapters 1, 2, 3, 5, 7.
    prep_samples = (
        (1, c1_blocks, "talking to", "Предлог «to» стоит в конце вопроса."),
        (2, c2_blocks, "thinking about", "Предлог about вынесен в конец предложения."),
        (3, c3_blocks, "playing with", "Предлог (with) остаётся в конце."),
        (5, c5_blocks, "looking for", "Предлог for обычно ставится в конце вопроса."),
        (7, c7_blocks, "staring at", "Предлог «at» стоит в конце фразы."),
    )
    prep_blocks: dict[int, DocumentBlock] = {}
    results: dict[str, BlockProcessingResult] = {}

    for ch_num, bucket, excerpt, note_text in prep_samples:
        blk = _block(f"c{ch_num}", len(bucket), f"Who is she {excerpt}?")
        bucket.append(blk)
        prep_blocks[ch_num] = blk
        results[blk.id] = BlockProcessingResult(
            blk.id,
            BlockTreatment.READ_AS_IS,
            language_notes=(LanguageNote(f"ln-prep-{ch_num}", blk.id, excerpt, note_text),),
        )

    cloak_h = GlossaryHint("cloak", "плащ", "Длинная накидка.")
    lantern_h = GlossaryHint("lantern", "фонарь", "Переносной фонарь.")
    wand_h = GlossaryHint("wand", "палочка", "Волшебная палочка.")
    goblet_h = GlossaryHint("goblet", "кубок", "Большая чаша.")
    cauldron_h = GlossaryHint("cauldron", "котёл", "Большой металлический котёл.")
    quill_h = GlossaryHint("quill", "перо", "Писчее перо.")

    results[c1_blocks[0].id] = BlockProcessingResult(
        c1_blocks[0].id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_h, lantern_h),
    )
    results[c1_goblet.id] = BlockProcessingResult(
        c1_goblet.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(goblet_h, cauldron_h),
    )
    # Adjacent TRANSLATE blocks in Ch 2 for cloak: must NOT double-count!
    results[c2_tr_1.id] = BlockProcessingResult(
        c2_tr_1.id,
        BlockTreatment.TRANSLATE,
        translated_text="Плащ развевался.",
        glossary_hints=(cloak_h,),
    )
    results[c2_tr_2.id] = BlockProcessingResult(
        c2_tr_2.id,
        BlockTreatment.TRANSLATE,
        translated_text="Другой плащ мелькнул.",
        glossary_hints=(cloak_h,),
    )
    # Following GLOSSARY_ONLY block with wand + cloak: cloak is 3rd (not 4th!), so shown stripped!
    results[c2_gl_after.id] = BlockProcessingResult(
        c2_gl_after.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(cloak_h, wand_h),
    )
    results[c2_tr_lantern.id] = BlockProcessingResult(
        c2_tr_lantern.id,
        BlockTreatment.TRANSLATE,
        translated_text="Фонарь мигнул.",
        glossary_hints=(lantern_h,),
    )
    results[c2_goblet.id] = BlockProcessingResult(
        c2_goblet.id,
        BlockTreatment.TRANSLATE,
        translated_text="Она взяла кубок и котёл.",
        glossary_hints=(goblet_h, cauldron_h),
    )
    # Mixed block 1: segment 0 TRANSLATE ("goblet" -> 3rd occurrence in TRANSLATE -> suppressed),
    # segment 1 GLOSSARY_ONLY ("lantern" -> 3rd occurrence in GLOSSARY_ONLY -> stripped!).
    results[c3_mixed_strip.id] = BlockProcessingResult(
        c3_mixed_strip.id,
        BlockTreatment.TRANSLATE,
        translated_text="Он открыл кубок.",
        glossary_hints=(goblet_h, lantern_h),
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        segment_translations=("Он открыл кубок.", None),
    )
    # Mixed block 2 (within 20-block window of lantern):
    # segment 0 TRANSLATE ("quill" -> 1st, shown), segment 1 GLOSSARY_ONLY ("lantern" -> rescued compact!).
    results[c3_mixed_rescue.id] = BlockProcessingResult(
        c3_mixed_rescue.id,
        BlockTreatment.TRANSLATE,
        translated_text="Он держал перо.",
        glossary_hints=(quill_h, lantern_h),
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        segment_translations=("Он держал перо.", None),
    )
    # Scenario 1a in Ch 5: cauldron was shown in Ch 1 (count=1), suppressed in Ch 2 (count=2), gap 5-1=4 > 3.
    results[c5_tr_1.id] = BlockProcessingResult(
        c5_tr_1.id,
        BlockTreatment.TRANSLATE,
        translated_text="Котёл сверкнул.",
        glossary_hints=(cauldron_h,),
    )
    results[c5_tr_2.id] = BlockProcessingResult(
        c5_tr_2.id,
        BlockTreatment.TRANSLATE,
        translated_text="Котёл упал.",
        glossary_hints=(cauldron_h,),
    )

    sections = (
        _section("c1", 0, "Chapter 1", c1_blocks),
        _section("c2", 1, "Chapter 2", c2_blocks),
        _section("c3", 2, "Chapter 3", c3_blocks),
        _section("c4", 3, "Chapter 4", c4_blocks),
        _section("c5", 4, "Chapter 5", c5_blocks),
        _section("c6", 5, "Chapter 6", c6_blocks),
        _section("c7", 6, "Chapter 7", c7_blocks),
    )
    layout = build_book_layout(_inputs(_document(sections), results))

    # Finding 1b: adjacent TRANSLATE blocks did not double-count; c2_gl_after keeps stripped cloak + full wand.
    assert _notes(layout, c2_tr_1.id) == ()
    assert _notes(layout, c2_tr_2.id) == ()
    assert [(n.term, n.body) for n in _notes(layout, c2_gl_after.id)] == [
        ("cloak", ("плащ",)),
        ("wand", ("палочка", "Волшебная палочка.")),
    ]

    # Finding 1a: Ch 5 blk 0 reintroduces cauldron cleanly; Ch 5 blk 1 stays suppressed (no suppress->full flip!).
    assert [(n.term, n.body) for n in _notes(layout, c5_tr_1.id)] == [
        ("cauldron", ("котёл", "Большой металлический котёл.")),
    ]
    assert _notes(layout, c5_tr_2.id) == ()

    # Finding 2: Mixed-segment blocks preserve hints on untranslated GLOSSARY_ONLY segments.
    assert [(n.term, n.body) for n in _notes(layout, c3_mixed_strip.id)] == [
        ("lantern", ("фонарь",)),
    ]
    assert [(n.term, n.body) for n in _notes(layout, c3_mixed_rescue.id)] == [
        ("quill", ("перо", "Писчее перо.")),
        ("lantern", ("фонарь",)),
    ]

    # Finding 3: Russian preposition_stranding notes naming the preposition are capped (cap 3, gap >= 2).
    assert len(_notes(layout, prep_blocks[1].id)) == 1
    assert len(_notes(layout, prep_blocks[2].id)) == 0
    assert len(_notes(layout, prep_blocks[3].id)) == 1
    assert len(_notes(layout, prep_blocks[5].id)) == 1
    assert len(_notes(layout, prep_blocks[7].id)) == 0


def test_gringotts_paragraph_stanza_sets_lines_and_has_poems() -> None:
    """Implicit enjambed verse stanza in PARAGRAPH blocks sets `unit.lines=True` and `has_poems=True`."""
    intro = _block("c1", 0, "Words were carved beside the doors:")
    stanza = (
        _block("c1", 1, "Enter, stranger, but take heed"),
        _block("c1", 2, "Of what awaits the sin of greed,"),
        _block("c1", 3, "For those who take, but do not earn,"),
        _block("c1", 4, "Must pay most dearly in their turn."),
    )
    outro = _block("c1", 5, "Harry looked up at the doors.")
    section = _section("c1", 0, "Chapter 1", (intro, *stanza, outro))
    results = {
        b.id: BlockProcessingResult(b.id, BlockTreatment.TRANSLATE, translated_text=f"Перевод {i}.")
        for i, b in enumerate(stanza, start=1)
    }
    layout = build_book_layout(_inputs(_document((section,)), results, reader_level="a2"))
    units = layout.chapters[section.id].units

    assert units[intro.id][0].lines is False
    assert all(units[b.id][0].lines is True for b in stanza)
    assert units[outro.id][0].lines is False
    assert layout.features.has_poems is True


def test_fades_translation_threshold_by_content_word_count() -> None:
    """`fades_translation` is True only for A2 without english_help, >= 2 content chapters, and >= 25_000 words before last."""
    ten_words = "word " * 10
    big_block_15k = "word " * 15_000
    big_block_10k = "word " * 10_000
    block_9999 = "word " * 9_999

    def _build(
        sections: tuple[DocumentSection, ...],
        *,
        reader_level: str | None = "a2",
        english_help: bool = False,
    ) -> BookLayout:
        first_block = sections[0].ordered_blocks[0]
        results = {
            first_block.id: BlockProcessingResult(
                first_block.id, BlockTreatment.TRANSLATE, translated_text="слово"
            )
        }
        return build_book_layout(
            _inputs(
                _document(sections),
                results,
                reader_level=reader_level,
                english_help=english_help,
            )
        )

    # 1 content chapter with >= 25_000 words -> False
    single_ch = _section("ch1", 0, "Chapter 1", (_block("ch1", 0, big_block_15k + big_block_10k),))
    assert _build((single_ch,)).features.fades_translation is False

    # 2 content chapters: 24_999 words before last -> False
    ch1_under = _section("ch1", 0, "Chapter 1", (_block("ch1", 0, big_block_15k + block_9999),))
    ch2_tail = _section("ch2", 1, "Chapter 2", (_block("ch2", 0, ten_words),))
    assert _build((ch1_under, ch2_tail)).features.fades_translation is False

    # Front matter + notes-only words do NOT count toward the 25_000 threshold
    front = _section("front", 0, "Title", (_block("front", 0, big_block_15k + big_block_15k),))
    ch1_small = _section("ch1", 1, "Chapter 1", (_block("ch1", 0, ten_words),))
    notes = _section(
        "notes",
        2,
        "Notes",
        (_block("notes", 0, big_block_15k, kind=BlockType.FOOTNOTE),),
    )
    ch2_small = _section("ch2", 3, "Chapter 2", (_block("ch2", 0, ten_words),))
    assert _build((front, ch1_small, notes, ch2_small)).features.fades_translation is False

    # Exactly 25_000 words across ch1 + ch2 before ch3 (last) -> True for A2 without english_help
    ch1_15k = _section("ch1", 0, "Chapter 1", (_block("ch1", 0, big_block_15k),))
    ch2_10k = _section("ch2", 1, "Chapter 2", (_block("ch2", 0, big_block_10k),))
    ch3_last = _section("ch3", 2, "Chapter 3", (_block("ch3", 0, ten_words),))
    assert _build((ch1_15k, ch2_10k, ch3_last)).features.fades_translation is True
    assert _build((ch1_15k, ch2_10k, ch3_last), reader_level="b2").features.fades_translation is False
    assert _build((ch1_15k, ch2_10k, ch3_last), english_help=True).features.fades_translation is False


def test_a2_dedup_filters_trivial_a1_hints_per_segment_with_fallbacks_and_protections() -> None:
    """A1 hint filtering inside `_deduplicate_glossary_hints` for `reader_level == 'a2'`."""
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    def _hint_ev(diff: Difficulty, block_id: str, hint: GlossaryHint) -> HelpEvidence:
        key = artifact_key(
            "glossary_hint.translation",
            block_id,
            {"original": hint.original, "translation": hint.translation, "note": hint.note},
        )
        return HelpEvidence(
            diff.id,
            diff.required_points[0].id,
            "glossary_hint.translation",
            block_id,
            key,
            hint.translation,
        )

    # 1. TRANSLATE block: unprotected A1 hints are suppressed and do NOT pollute history;
    #    non-empty `hint.note` and non-`vocabulary` difficulty kinds are protected.
    tr_block = _block(
        "c1",
        0,
        "He smiled at the hamburger while reading cold banana signs and odd park rules.",
    )
    # 2. GLOSSARY_ONLY block with A1 + B1 -> A1 dropped, B1 kept.
    gl_with_b1 = _block("c1", 1, "She smiled while holding the cloak.")
    # 3. GLOSSARY_ONLY block with A1 + repeated B1 (within 20 blocks -> rescued!) ->
    #    A1 dropped BEFORE history/rescue, repeated B1 rescued as compact!
    gl_rescue_b1 = _block("c1", 2, "He ate a hamburger under the cloak.")
    # 4. GLOSSARY_ONLY block with ONLY A1 hints -> keeps 1 fallback, drops extra.
    gl_only_a1 = _block("c1", 3, "He smiled at the hamburger.")
    # 5. Another GLOSSARY_ONLY block within 20 blocks with the same sole A1 hint ->
    #    still keeps 1 fallback via rescue.
    gl_only_a1_repeat = _block("c1", 4, "He smiled again.")
    # 6. Mixed block [TRANSLATE (B1 wand + A1 hamburger), GLOSSARY_ONLY (ONLY A1 smiled + telescopes)]:
    #    TRANSLATE drops A1 & keeps B1 wand; GLOSSARY_ONLY segment keeps 1 A1 fallback (smiled) and drops telescopes!
    mix_1_text = "He held a wand and ate a hamburger. She smiled at telescopes."
    mix_1_cut = mix_1_text.index(" She smiled")
    mix_1 = _block(
        "c1",
        5,
        mix_1_text,
        segments=(
            BlockSegment(0, mix_1_cut, 0),
            BlockSegment(mix_1_cut, len(mix_1_text), 1),
        ),
    )
    # 7. Mixed block [TRANSLATE (A1 airport), GLOSSARY_ONLY (A1 hamburger + repeated B1 cloak -> rescued!)]:
    #    TRANSLATE drops airport; GLOSSARY_ONLY drops hamburger and rescues cloak!
    mix_2_text = "They walked to the airport. The cloak hid her."
    mix_2_cut = mix_2_text.index(" The cloak")
    mix_2 = _block(
        "c1",
        6,
        mix_2_text,
        segments=(
            BlockSegment(0, mix_2_cut, 0),
            BlockSegment(mix_2_cut, len(mix_2_text), 1),
        ),
    )
    # 8. Mixed block with 2 GLOSSARY_ONLY segments:
    #    seg 0 GLOSSARY_ONLY (only A1 smiled) -> keeps smiled fallback;
    #    seg 1 TRANSLATE (A1 computer) -> drops computer;
    #    seg 2 GLOSSARY_ONLY (A1 hotel + B1 lantern) -> drops hotel, keeps lantern!
    mix_3_text = "She smiled. He used a computer. He saw a hotel lantern."
    mix_3_c1 = mix_3_text.index(" He used")
    mix_3_c2 = mix_3_text.index(" He saw")
    mix_3 = _block(
        "c1",
        7,
        mix_3_text,
        segments=(
            BlockSegment(0, mix_3_c1, 0),
            BlockSegment(mix_3_c1, mix_3_c2, 1),
            BlockSegment(mix_3_c2, len(mix_3_text), 2),
        ),
    )
    # 9. GLOSSARY_ONLY block verifying all 8 non-vocabulary difficulty kinds protect A1 hints,
    #    while `vocabulary` kind does NOT protect an A1 hint when a B1 hint exists.
    prot_block = _block(
        "c1",
        8,
        "A doctor, bank, park, film, music, sport, taxi, coffee, and telephone stood by a cauldron.",
    )
    # 10. Later GLOSSARY_ONLY block in ch2 (> 20 blocks later) with `hamburger`:
    #     verify `hamburger` was NEVER added to history by earlier filtered blocks -> shown full!
    filler_blocks = [_block("c1", idx, f"Filler {idx}.") for idx in range(9, 32)]
    ch2_block = _block("c2", 0, "A hamburger sat on a desk.")

    non_vocab_kinds = (
        ("doctor", "idiom"),
        ("bank", "false_friend"),
        ("park", "cultural_reference"),
        ("film", "wordplay"),
        ("music", "parody"),
        ("sport", "speech_style"),
        ("taxi", "grammar"),
        ("coffee", "semantic_relation"),
    )
    tr_cold_diff = _difficulty(tr_block, "cold", "wordplay", "d-tr-wordplay", help_kind="lexical_hint")
    tr_park_diff = _difficulty(tr_block, "park", "vocabulary", "d-tr-vocab", help_kind="lexical_hint")
    prot_diffs = tuple(
        _difficulty(prot_block, word, kind, f"d-prot-{kind}", help_kind="lexical_hint")
        for word, kind in non_vocab_kinds
    )
    prot_vocab_diff = _difficulty(
        prot_block, "telephone", "vocabulary", "d-prot-vocab", help_kind="lexical_hint"
    )
    plan = _plan(tr_cold_diff, tr_park_diff, *prot_diffs, prot_vocab_diff)

    tr_h_smile = GlossaryHint("smiled", "улыбнулся")
    tr_h_burger = GlossaryHint("hamburger", "гамбургер")
    tr_h_banana_note = GlossaryHint("banana", "банан", "Здесь: жаргонное прозвище.")
    tr_h_cold_wordplay = GlossaryHint("cold", "холодный")
    tr_h_park_vocab = GlossaryHint("park", "парк")

    prot_hints = tuple(GlossaryHint(word, f"пер-{word}") for word, _ in non_vocab_kinds)
    prot_tel_vocab = GlossaryHint("telephone", "телефон")
    prot_cauldron = GlossaryHint("cauldron", "котёл", "Большой котёл.")

    results = {
        tr_block.id: BlockProcessingResult(
            tr_block.id,
            BlockTreatment.TRANSLATE,
            translated_text="Он улыбнулся гамбургеру.",
            glossary_hints=(
                tr_h_smile,
                tr_h_burger,
                tr_h_banana_note,
                tr_h_cold_wordplay,
                tr_h_park_vocab,
            ),
            content_evidence=ContentEvidence(
                plan.sha256,
                (
                    _hint_ev(tr_cold_diff, tr_block.id, tr_h_cold_wordplay),
                    _hint_ev(tr_park_diff, tr_block.id, tr_h_park_vocab),
                ),
            ),
        ),
        gl_with_b1.id: BlockProcessingResult(
            gl_with_b1.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint("smiled", "улыбнулась"),
                GlossaryHint("cloak", "плащ", "Длинная накидка."),
            ),
        ),
        gl_rescue_b1.id: BlockProcessingResult(
            gl_rescue_b1.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint("hamburger", "гамбургер"),
                GlossaryHint("cloak", "плащ", "Повтор плаща в окне."),
            ),
        ),
        gl_only_a1.id: BlockProcessingResult(
            gl_only_a1.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(
                GlossaryHint("smiled", "улыбнулся"),
                GlossaryHint("hamburger", "гамбургер"),
            ),
        ),
        gl_only_a1_repeat.id: BlockProcessingResult(
            gl_only_a1_repeat.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint("smiled", "улыбнулся"),),
        ),
        mix_1.id: BlockProcessingResult(
            mix_1.id,
            BlockTreatment.TRANSLATE,
            translated_text="Он держал палочку и ел гамбургер.",
            segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
            segment_translations=("Он держал палочку и ел гамбургер.", None),
            glossary_hints=(
                GlossaryHint("wand", "палочка", "Волшебная палочка."),
                GlossaryHint("hamburger", "гамбургер"),
                GlossaryHint("smiled", "улыбнулась"),
                GlossaryHint("telescopes", "телескопы"),
            ),
        ),
        mix_2.id: BlockProcessingResult(
            mix_2.id,
            BlockTreatment.TRANSLATE,
            translated_text="Они пошли в аэропорт.",
            segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
            segment_translations=("Они пошли в аэропорт.", None),
            glossary_hints=(
                GlossaryHint("airport", "аэропорт"),
                GlossaryHint("hamburger", "гамбургер"),
                GlossaryHint("cloak", "плащ", "Снова плащ."),
            ),
        ),
        mix_3.id: BlockProcessingResult(
            mix_3.id,
            BlockTreatment.TRANSLATE,
            translated_text="Он пользовался компьютером.",
            segment_treatments=(
                BlockTreatment.GLOSSARY_ONLY,
                BlockTreatment.TRANSLATE,
                BlockTreatment.GLOSSARY_ONLY,
            ),
            segment_translations=(None, "Он пользовался компьютером.", None),
            glossary_hints=(
                GlossaryHint("smiled", "улыбнулась"),
                GlossaryHint("computer", "компьютер"),
                GlossaryHint("hotel", "отель"),
                GlossaryHint("lantern", "фонарь", "Переносной фонарь."),
            ),
        ),
        prot_block.id: BlockProcessingResult(
            prot_block.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(*prot_hints, prot_tel_vocab, prot_cauldron),
            content_evidence=ContentEvidence(
                plan.sha256,
                (
                    *(
                        _hint_ev(diff, prot_block.id, hint)
                        for diff, hint in zip(prot_diffs, prot_hints, strict=True)
                    ),
                    _hint_ev(prot_vocab_diff, prot_block.id, prot_tel_vocab),
                ),
            ),
        ),
        ch2_block.id: BlockProcessingResult(
            ch2_block.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(GlossaryHint("hamburger", "гамбургер", "Первое появление в истории."),),
        ),
    }

    sections = (
        _section(
            "c1",
            0,
            "Chapter 1",
            (
                tr_block,
                gl_with_b1,
                gl_rescue_b1,
                gl_only_a1,
                gl_only_a1_repeat,
                mix_1,
                mix_2,
                mix_3,
                prot_block,
                *filler_blocks,
            ),
        ),
        _section("c2", 1, "Chapter 2", (ch2_block,)),
    )
    doc = _document(sections)
    analyses = (_analysis("a-c1", "c1", plan=plan),)

    a2_layout = build_book_layout(_inputs(doc, results, analyses, reader_level="a2"))

    # 1. TRANSLATE keeps only wordplay-protected `cold` and note-protected `banana` (sorted by marker)
    assert [n.term for n in _notes(a2_layout, tr_block.id)] == ["cold", "banana"]
    # 2. GLOSSARY_ONLY drops `smiled`, keeps `cloak`
    assert [n.term for n in _notes(a2_layout, gl_with_b1.id)] == ["cloak"]
    # 3. GLOSSARY_ONLY drops `hamburger` BEFORE rescue, rescues `cloak` as compact
    assert [(n.term, n.body) for n in _notes(a2_layout, gl_rescue_b1.id)] == [
        ("cloak", ("плащ",)),
    ]
    # 4 & 5. GLOSSARY_ONLY with ONLY A1 keeps 1 fallback (`smiled` -> `to smile`), drops `hamburger`
    assert [n.term for n in _notes(a2_layout, gl_only_a1.id)] == ["to smile"]
    assert [n.term for n in _notes(a2_layout, gl_only_a1_repeat.id)] == ["to smile"]
    # 6. Mixed block: TRANSLATE keeps B1 `wand` & drops `hamburger`;
    #    GLOSSARY_ONLY seg has ONLY A1 -> keeps `to smile` fallback, drops `telescopes`!
    assert [n.term for n in _notes(a2_layout, mix_1.id)] == ["wand", "to smile"]
    # 7. Mixed block: TRANSLATE drops `airport`, GLOSSARY_ONLY drops `hamburger` and rescues `cloak`!
    assert [(n.term, n.body) for n in _notes(a2_layout, mix_2.id)] == [
        ("cloak", ("плащ",)),
    ]
    # 8. Mixed block with 2 GLOSSARY_ONLY segments: seg 0 keeps `to smile` fallback, seg 1 drops `computer`,
    #    seg 2 drops `hotel` and keeps `lantern`!
    assert [n.term for n in _notes(a2_layout, mix_3.id)] == ["to smile", "lantern"]
    # 9. All 8 non-vocabulary kinds survive alongside B1 `cauldron`, while `vocabulary` `telephone` is dropped
    assert [n.term for n in _notes(a2_layout, prot_block.id)] == [
        *(word for word, _ in non_vocab_kinds),
        "cauldron",
    ]
    # 10. Filtered `hamburger` did NOT consume occurrence 1 in history, so ch2 shows full note!
    assert [(n.term, n.body) for n in _notes(a2_layout, ch2_block.id)] == [
        ("hamburger", ("гамбургер", "Первое появление в истории.")),
    ]

    # Non-A2 reader_level does NOT strip A1 hints!
    b2_layout = build_book_layout(_inputs(doc, results, analyses, reader_level="b2"))
    assert [n.term for n in _notes(b2_layout, tr_block.id)] == [
        "to smile",
        "hamburger",
        "cold",
        "banana",
        "park",
    ]
    b2_single = build_book_layout(
        _inputs(_document((_section("c1", 0, "Chapter 1", (gl_with_b1,)),)), {gl_with_b1.id: results[gl_with_b1.id]}, reader_level="b2")
    )
    assert [n.term for n in _notes(b2_single, gl_with_b1.id)] == ["to smile", "cloak"]


def test_a2_dedup_mixed_read_as_is_and_glossary_only_segments_do_not_collapse_buckets() -> None:
    """Mixed `(READ_AS_IS, GLOSSARY_ONLY, GLOSSARY_ONLY)` keeps per-segment buckets despite `treatment == GLOSSARY_ONLY`."""
    text = "A taxi waited. She smiled. He held a lantern in the hotel."
    c1 = text.index(" She smiled")
    c2 = text.index(" He held")
    block = _block(
        "c1",
        0,
        text,
        segments=(
            BlockSegment(0, c1, 0),
            BlockSegment(c1, c2, 1),
            BlockSegment(c2, len(text), 2),
        ),
    )
    result = BlockProcessingResult(
        block.id,
        BlockTreatment.GLOSSARY_ONLY,
        segment_treatments=(
            BlockTreatment.READ_AS_IS,
            BlockTreatment.GLOSSARY_ONLY,
            BlockTreatment.GLOSSARY_ONLY,
        ),
        segment_translations=(None, None, None),
        glossary_hints=(
            GlossaryHint("taxi", "такси"),
            GlossaryHint("smiled", "улыбнулась"),
            GlossaryHint("hotel", "отель"),
            GlossaryHint("lantern", "фонарь", "Переносной фонарь."),
        ),
    )
    doc = _document((_section("c1", 0, "Chapter 1", (block,)),))
    layout = build_book_layout(_inputs(doc, {block.id: result}, reader_level="a2"))

    assert [n.term for n in _notes(layout, block.id)] == ["to smile", "lantern"]


def test_book_layout_deduplicates_cultural_reference_and_short_term_notes_without_collapsing_sentences() -> None:
    """`cultural_reference` and short term notes (<=4 words) collapse to cap=1 via vocabulary_dedup_key; long sentences stay."""
    from linguary.application.layout.book import _deduplicate_grammar_notes
    from linguary.domain import LanguageNote
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    def _note_ev(diff: Difficulty, note: LanguageNote) -> HelpEvidence:
        key = artifact_key(
            "language_note.note",
            note.block_id,
            {
                "original_excerpt": note.original_excerpt,
                "note": note.note,
                "translated_excerpt": note.translated_excerpt,
            },
        )
        return HelpEvidence(diff.id, diff.required_points[0].id, "language_note.note", note.block_id, key, note.note)

    # 4 cultural_reference notes across 4 chapters: Muggles / Muggle / Muggles / Muggles -> only 1st stays
    b1 = _block("c1", 0, "They didn't want to mix with Muggles.")
    b2 = _block("c2", 0, "He is just a Muggle, she thought.")
    b3 = _block("c3", 0, "Even Muggles noticed the strange owls.")
    b4 = _block("c4", 0, "Never tell Muggles about magic.")

    # Long (>4 words) cultural_reference appearing twice -> collapsed by Difficulty.kind == "cultural_reference"
    b1_long_cult = _block("c1", 1, "He received the Order of Merlin, First Class.")
    b3_long_cult = _block("c3", 1, "Mentioning the Order of Merlin, First Class again.")

    # Unrelated long full-sentence notes (>4 words, not cultural_reference) -> NEVER collapsed!
    long_sentence = "She had never seen anything quite like this before."
    b1_sent = _block("c1", 2, long_sentence)
    b2_sent = _block("c2", 1, long_sentence)

    d1 = _difficulty(b1, "Muggles", "cultural_reference", "d-m1")
    d2 = _difficulty(b2, "Muggle", "cultural_reference", "d-m2")
    d3 = _difficulty(b3, "Muggles", "cultural_reference", "d-m3")
    d4 = _difficulty(b4, "Muggles", "cultural_reference", "d-m4")
    d1_lc = _difficulty(b1_long_cult, "Order of Merlin, First Class", "cultural_reference", "d-om1")
    d3_lc = _difficulty(b3_long_cult, "Order of Merlin, First Class", "cultural_reference", "d-om3")
    d1_sent = _difficulty(b1_sent, long_sentence, "grammar", "d-s1")
    d2_sent = _difficulty(b2_sent, long_sentence, "grammar", "d-s2")

    p1 = _plan(d1, d1_lc, d1_sent)
    p2 = _plan(d2, d2_sent)
    p3 = _plan(d3, d3_lc)
    p4 = _plan(d4)

    n1 = LanguageNote("ln-1", b1.id, "Muggles", "Маглы — люди без магических способностей.")
    n2 = LanguageNote("ln-2", b2.id, "Muggle", "Магл — человек без волшебных способностей.")
    n3 = LanguageNote("ln-3", b3.id, "Muggles", "Снова о маглах.")
    n4 = LanguageNote("ln-4", b4.id, "Muggles", "Ещё одно упоминание маглов.")
    n1_lc = LanguageNote("ln-5", b1_long_cult.id, "Order of Merlin, First Class", "Высшая награда волшебного мира.")
    n3_lc = LanguageNote("ln-6", b3_long_cult.id, "Order of Merlin, First Class", "Повтор про орден Мерлина.")
    n1_sent = LanguageNote("ln-7", b1_sent.id, long_sentence, "Первая длинная грамматическая заметка.")
    n2_sent = LanguageNote("ln-8", b2_sent.id, long_sentence, "Вторая длинная грамматическая заметка.")

    results = {
        b1.id: BlockProcessingResult(
            b1.id, BlockTreatment.READ_AS_IS, language_notes=(n1,),
            content_evidence=ContentEvidence(p1.sha256, (_note_ev(d1, n1),)),
        ),
        b2.id: BlockProcessingResult(
            b2.id, BlockTreatment.READ_AS_IS, language_notes=(n2,),
            content_evidence=ContentEvidence(p2.sha256, (_note_ev(d2, n2),)),
        ),
        b3.id: BlockProcessingResult(
            b3.id, BlockTreatment.READ_AS_IS, language_notes=(n3,),
            content_evidence=ContentEvidence(p3.sha256, (_note_ev(d3, n3),)),
        ),
        b4.id: BlockProcessingResult(
            b4.id, BlockTreatment.READ_AS_IS, language_notes=(n4,),
            content_evidence=ContentEvidence(p4.sha256, (_note_ev(d4, n4),)),
        ),
        b1_long_cult.id: BlockProcessingResult(
            b1_long_cult.id, BlockTreatment.READ_AS_IS, language_notes=(n1_lc,),
            content_evidence=ContentEvidence(p1.sha256, (_note_ev(d1_lc, n1_lc),)),
        ),
        b3_long_cult.id: BlockProcessingResult(
            b3_long_cult.id, BlockTreatment.READ_AS_IS, language_notes=(n3_lc,),
            content_evidence=ContentEvidence(p3.sha256, (_note_ev(d3_lc, n3_lc),)),
        ),
        b1_sent.id: BlockProcessingResult(
            b1_sent.id, BlockTreatment.READ_AS_IS, language_notes=(n1_sent,),
            content_evidence=ContentEvidence(p1.sha256, (_note_ev(d1_sent, n1_sent),)),
        ),
        b2_sent.id: BlockProcessingResult(
            b2_sent.id, BlockTreatment.READ_AS_IS, language_notes=(n2_sent,),
            content_evidence=ContentEvidence(p2.sha256, (_note_ev(d2_sent, n2_sent),)),
        ),
    }

    sections = (
        _section("c1", 0, "Chapter 1", (b1, b1_long_cult, b1_sent)),
        _section("c2", 1, "Chapter 2", (b2, b2_sent)),
        _section("c3", 2, "Chapter 3", (b3, b3_long_cult)),
        _section("c4", 3, "Chapter 4", (b4,)),
    )
    analyses = (
        _analysis("a1", "c1", plan=p1),
        _analysis("a2", "c2", plan=p2),
        _analysis("a3", "c3", plan=p3),
        _analysis("a4", "c4", plan=p4),
    )

    layout = build_book_layout(_inputs(_document(sections), results, analyses))

    assert len(_notes(layout, b1.id)) == 1
    assert _notes(layout, b2.id) == ()
    assert _notes(layout, b3.id) == ()
    assert _notes(layout, b4.id) == ()
    assert len(_notes(layout, b1_long_cult.id)) == 1
    assert _notes(layout, b3_long_cult.id) == ()
    assert len(_notes(layout, b1_sent.id)) == 1
    assert len(_notes(layout, b2_sent.id)) == 1

    # Direct helper check with plan parameter, plus short-term (<=4 words) dedup with plan=None
    history: dict = {}
    assert _deduplicate_grammar_notes(
        results[b1.id], dropped_difficulty_ids=frozenset(), chapter_num=1, history=history, plan=p1
    ) == frozenset()
    assert _deduplicate_grammar_notes(
        results[b2.id], dropped_difficulty_ids=frozenset(), chapter_num=2, history=history, plan=p2
    ) == frozenset({0})

    short_term_1 = BlockProcessingResult(
        b1.id, BlockTreatment.READ_AS_IS,
        language_notes=(LanguageNote("ln-st1", b1.id, "Privet Drive", "Улица в пригороде."),),
    )
    short_term_2 = BlockProcessingResult(
        b2.id, BlockTreatment.READ_AS_IS,
        language_notes=(LanguageNote("ln-st2", b2.id, "Privet Drive", "Снова про улицу."),),
    )
    assert _deduplicate_grammar_notes(
        short_term_1, dropped_difficulty_ids=frozenset(), chapter_num=1, history=history, plan=None
    ) == frozenset()
    assert _deduplicate_grammar_notes(
        short_term_2, dropped_difficulty_ids=frozenset(), chapter_num=2, history=history, plan=None
    ) == frozenset({0})
    assert _deduplicate_grammar_notes(
        results[b1_sent.id], dropped_difficulty_ids=frozenset(), chapter_num=1, history=history, plan=None
    ) == frozenset()
    assert _deduplicate_grammar_notes(
        results[b2_sent.id], dropped_difficulty_ids=frozenset(), chapter_num=2, history=history, plan=None
    ) == frozenset()


def test_mixed_segment_hint_anchor_consistent_between_dedup_and_layout_units() -> None:
    """`_hint_segment_index`, `_hint_is_glossary_only`, `_glossary_only_bucket`, and `block_notes` agree."""
    from linguary.application.layout.book import (
        _glossary_only_bucket,
        _hint_is_glossary_only,
        _hint_segment_index,
    )
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    def _hint_ev(diff: Difficulty, block_id: str, hint: GlossaryHint) -> HelpEvidence:
        key = artifact_key(
            "glossary_hint.translation",
            block_id,
            {"original": hint.original, "translation": hint.translation, "note": hint.note},
        )
        return HelpEvidence(
            diff.id, diff.required_points[0].id, "glossary_hint.translation", block_id, key, hint.translation
        )

    # Prior block puts `cliff` into local dedup window so rescue depends on GLOSSARY_ONLY classification.
    prior_block = _block("c1", 0, "They saw a cliff ahead.")
    seg0 = "They fell from the base of the cliff. "
    seg1 = "Now they lay at the base of the cliff."
    mixed_text = seg0 + seg1
    mixed_block = _block(
        "c1",
        1,
        mixed_text,
        segments=(
            BlockSegment(0, len(seg0), 0),
            BlockSegment(len(seg0), len(mixed_text), 1),
        ),
    )

    # Unanchored hint `cliff` must bind to GLOSSARY_ONLY segment 1 (and be rescued compactly).
    cliff_hint = GlossaryHint("cliff", "обрыв", "Крутой склон.")
    mixed_res = BlockProcessingResult(
        mixed_block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Они упали со скалы.",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        segment_translations=("Они упали со скалы.", None),
        glossary_hints=(cliff_hint,),
    )
    prior_res = BlockProcessingResult(
        prior_block.id,
        BlockTreatment.GLOSSARY_ONLY,
        glossary_hints=(GlossaryHint("cliff", "обрыв", "Первое пояснение."),),
    )

    assert _hint_segment_index(mixed_block, mixed_res, cliff_hint, plan=None) == 1
    assert _hint_is_glossary_only(mixed_block, mixed_res, cliff_hint, requires_hints=True, plan=None) is True

    doc = _document((_section("c1", 0, "Chapter 1", (prior_block, mixed_block)),))
    layout = build_book_layout(
        _inputs(doc, {prior_block.id: prior_res, mixed_block.id: mixed_res}, reader_level="a2")
    )
    units = layout.chapters["c1"].units[mixed_block.id]
    assert len(units) == 2
    assert units[0].notes == ()
    assert len(units[1].notes) == 1
    assert units[1].notes[0].note.term == "cliff"
    assert units[1].notes[0].note.body == ("обрыв",)
    assert units[1].notes[0].note.marker_end == mixed_text.rfind("cliff") + len("cliff")

    # Explicit Difficulty anchor in segment 0 keeps hint on segment 0 (TRANSLATE) in both dedup and notes.
    seg0_anchor = SourceAnchor(
        mixed_block.id,
        mixed_text.index("base of the cliff"),
        mixed_text.index("base of the cliff") + len("base of the cliff"),
        "base of the cliff",
    )
    diff_seg0 = Difficulty(
        "d-cliff-0",
        (seg0_anchor,),
        mixed_block.id,
        "vocabulary",
        "useful",
        "lexical_hint",
        (RequiredPoint("p-cliff-0", "Explain cliff"),),
    )
    plan_seg0 = _plan(diff_seg0)
    anchored_hint = GlossaryHint("cliff", "обрыв", "Важный образ.")
    anchored_res = BlockProcessingResult(
        mixed_block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Они упали со скалы.",
        segment_treatments=(BlockTreatment.TRANSLATE, BlockTreatment.GLOSSARY_ONLY),
        segment_translations=("Они упали со скалы.", None),
        glossary_hints=(anchored_hint,),
        content_evidence=ContentEvidence(
            plan_seg0.sha256,
            (_hint_ev(diff_seg0, mixed_block.id, anchored_hint),),
        ),
    )
    assert _hint_segment_index(mixed_block, anchored_res, anchored_hint, plan=plan_seg0) == 0
    assert (
        _hint_is_glossary_only(mixed_block, anchored_res, anchored_hint, requires_hints=True, plan=plan_seg0)
        is False
    )
    anchored_doc = _document((_section("c1", 0, "Chapter 1", (mixed_block,)),))
    anchored_layout = build_book_layout(
        _inputs(
            anchored_doc,
            {mixed_block.id: anchored_res},
            (_analysis("a-c1", "c1", plan=plan_seg0),),
            reader_level="a2",
        )
    )
    anchored_units = anchored_layout.chapters["c1"].units[mixed_block.id]
    assert len(anchored_units[0].notes) == 1
    assert anchored_units[0].notes[0].note.marker_end == mixed_text.index("cliff") + len("cliff")
    assert anchored_units[1].notes == ()

    # Multi-GLOSSARY_ONLY bucket honors explicit anchor via `_glossary_only_bucket(..., plan=...)`.
    s0 = "He stared at the cliff. "
    s1 = "The cliff rose high. "
    s2 = "The cliff felt cold."
    tri_text = s0 + s1 + s2
    tri_block = _block(
        "c1",
        2,
        tri_text,
        segments=(
            BlockSegment(0, len(s0), 0),
            BlockSegment(len(s0), len(s0) + len(s1), 1),
            BlockSegment(len(s0) + len(s1), len(tri_text), 2),
        ),
    )
    s2_start = tri_text.rfind("cliff")
    diff_s2 = Difficulty(
        "d-cliff-2",
        (SourceAnchor(tri_block.id, s2_start, s2_start + len("cliff"), "cliff"),),
        tri_block.id,
        "vocabulary",
        "useful",
        "lexical_hint",
        (RequiredPoint("p-cliff-2", "Explain cliff"),),
    )
    plan_s2 = _plan(diff_s2)
    tri_hint = GlossaryHint("cliff", "обрыв")
    tri_res = BlockProcessingResult(
        tri_block.id,
        BlockTreatment.TRANSLATE,
        translated_text="Он смотрел на обрыв.",
        segment_treatments=(
            BlockTreatment.TRANSLATE,
            BlockTreatment.GLOSSARY_ONLY,
            BlockTreatment.GLOSSARY_ONLY,
        ),
        segment_translations=("Он смотрел на обрыв.", None, None),
        glossary_hints=(tri_hint,),
        content_evidence=ContentEvidence(
            plan_s2.sha256,
            (_hint_ev(diff_s2, tri_block.id, tri_hint),),
        ),
    )
    assert _glossary_only_bucket(tri_block, tri_res, tri_hint, plan=plan_s2) == 2


def test_common_short_speaker_name_non_spoiler_and_preserves_titled_names() -> None:
    """Common short names switch only once dominant in `<= section.order`; titled names never shorten."""
    from linguary.application.layout.book import _extract_short_name_pair

    ch1_nar = _block("c1", 0, "Alexandra looked around. Alexandra sighed. Alexandra stood up.")
    ch1_dlg = _block("c1", 1, "“Wait for me.”")
    ch2_nar = _block("c2", 0, "Allie ran fast. Allie jumped. Allie laughed. Allie waved. Allie stopped.")
    ch2_dlg = _block("c2", 1, "“Over here!”")
    ch2_vernon = _block(
        "c2",
        2,
        "Vernon shouted. Vernon frowned. Vernon paced. Vernon yelled. Vernon glared.",
    )
    ch2_vernon_dlg = _block("c2", 3, "“Get inside!”")

    f1 = _fact("f-ch1", ch1_dlg, "“Wait for me.”", "Alexandra", "inferred")
    f2 = _fact("f-ch2", ch2_dlg, "“Over here!”", "Alexandra", "inferred")
    f3 = _fact("f-vernon", ch2_vernon_dlg, "“Get inside!”", "Uncle Vernon", "inferred")

    alex_concept = TermConcept(
        id="c-alex",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Alexandra",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-alex", "ru", "Александра", TermStatus.APPROVED),),
        note="Краткое обиходное имя: Allie (Элли).",
    )
    vernon_concept = TermConcept(
        id="c-vernon",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Uncle Vernon",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-vernon", "ru", "дядя Вернон", TermStatus.APPROVED),),
        note="Кратко: Vernon (Вернон).",
    )

    sections = (
        _section("c1", 0, "Chapter 1", (ch1_nar, ch1_dlg)),
        _section("c2", 1, "Chapter 2", (ch2_nar, ch2_dlg, ch2_vernon, ch2_vernon_dlg)),
    )
    doc = _document(sections)
    analyses = (
        _analysis("a1", "c1", plan=_plan(facts=(f1,))),
        _analysis("a2", "c2", plan=_plan(facts=(f2, f3))),
    )
    layout = build_book_layout(_inputs(doc, {}, analyses, (alex_concept, vernon_concept)))

    # Chapter 1: Allie has NOT appeared yet -> no spoiler! Must show Александра.
    ch1_unit = layout.chapters["c1"].units[ch1_dlg.id][0]
    assert ch1_unit.speaker == "Александра"
    assert ch1_unit.notes[0].note.body == ("Говорит Александра",)

    # Chapter 2: cumulative count(Allie)=5 > count(Alexandra)=3 -> switches to Элли.
    ch2_unit = layout.chapters["c2"].units[ch2_dlg.id][0]
    assert ch2_unit.speaker == "Элли"
    assert ch2_unit.notes[0].note.body == ("Говорит Элли",)

    # Titled canonical form `Uncle Vernon` is NEVER shortened even when `Vernon` is more frequent.
    vernon_unit = layout.chapters["c2"].units[ch2_vernon_dlg.id][0]
    assert vernon_unit.speaker == "дядя Вернон"
    assert vernon_unit.notes[0].note.body == ("Говорит дядя Вернон",)
    assert _extract_short_name_pair(vernon_concept, analyses, "ru") is None

    # Verify extraction from (a) parentheses in canonical_source_text and (c) candidate match in note.
    paren_concept = TermConcept(
        id="c-paren",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Alexandra (Allie)",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-paren", "ru", "Александра (Элли)", TermStatus.APPROVED),),
    )
    assert _extract_short_name_pair(paren_concept, (), "ru") == ("Allie", "Элли")

    cand_concept = TermConcept(
        id="c-cand",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Alexandra",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-cand", "ru", "Александра", TermStatus.APPROVED),),
        note="В тексте обычно называют Allie.",
    )
    cand_analyses = (
        _analysis("a-cand", "c1", candidates=(_candidate("a-cand", "cand-allie", "Allie", "Элли"),)),
    )
    assert _extract_short_name_pair(cand_concept, cand_analyses, "ru") == ("Allie", "Элли")

    abbr_concept = TermConcept(
        id="c-abbr",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Alexandra",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-abbr", "ru", "Александра", TermStatus.APPROVED),),
        note="Allie (сокращ. Элли).",
    )
    assert _extract_short_name_pair(abbr_concept, (), "ru") == ("Allie", "Элли")

    # Direct `fact.value == "Allie"` also resolves via the parent concept's short-name pair.
    from linguary.application.layout.book import _speaker_name

    assert _speaker_name("Allie", (alex_concept,), (), "ru", document=doc, section_order=1) == "Элли"

    # Multi-word canonical_source_text counts properly (re.escape must not break spaces!).
    mw_ch1 = _block(
        "mw1",
        0,
        "Alexandra Johnson walked in. Alexandra Johnson paused. Alexandra Johnson smiled. Allie nodded.",
    )
    mw_ch1_dlg = _block("mw1", 1, "“Hello.”")
    mw_ch2 = _block("mw2", 0, "Allie ran. Allie jumped. Allie called out.")
    mw_ch2_dlg = _block("mw2", 1, "“Look!”")
    mw_doc = _document(
        (
            _section("mw1", 0, "Chapter 1", (mw_ch1, mw_ch1_dlg)),
            _section("mw2", 1, "Chapter 2", (mw_ch2, mw_ch2_dlg)),
        )
    )
    mw_concept = TermConcept(
        id="c-alex-full",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Alexandra Johnson",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-alex-full", "ru", "Александра Джонсон", TermStatus.APPROVED),),
        note="Краткое имя: Allie (Элли).",
    )
    # Chapter 1: Alexandra Johnson=3, Allie=1 -> stays Александра Джонсон.
    assert (
        _speaker_name(
            "Alexandra Johnson", (mw_concept,), (), "ru", document=mw_doc, section_order=0
        )
        == "Александра Джонсон"
    )
    # Chapter 2: cumulative Allie=4 > Alexandra Johnson=3 -> switches to Элли.
    assert (
        _speaker_name(
            "Alexandra Johnson", (mw_concept,), (), "ru", document=mw_doc, section_order=1
        )
        == "Элли"
    )

    # Explicit TermConcept("Allie") appearing AFTER TermConcept("Alexandra", note="...Allie (Элли)")
    # keeps its explicit user translation regardless of concept order.
    explicit_allie = TermConcept(
        id="c-allie-explicit",
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text="Allie",
        status=TermStatus.APPROVED,
        variants=(TermVariant("c-allie-explicit", "ru", "Алли", TermStatus.APPROVED),),
    )
    assert (
        _speaker_name(
            "Allie", (alex_concept, explicit_allie), (), "ru", document=doc, section_order=1
        )
        == "Алли"
    )
    assert (
        _speaker_name(
            "Allie", (explicit_allie, alex_concept), (), "ru", document=doc, section_order=1
        )
        == "Алли"
    )


def test_self_identifies_speaker_suppresses_intro_line_but_keeps_generic_phrase() -> None:
    """Self-introduction lines suppress «Говорит …», including short names; generic nouns never suppress."""
    from linguary.application.layout.book import _part_self_identifies_speaker

    assert _part_self_identifies_speaker("“My name’s Nick.”", ("Nick",)) is True
    assert _part_self_identifies_speaker("“My name's Allie.”", ("Alexandra", "Allie")) is True
    assert _part_self_identifies_speaker("“Call me Allie.”", ("Alexandra", "Allie")) is True
    assert _part_self_identifies_speaker("“I'm the boy who stayed behind.”", ("the", "boy", "Nick")) is False
    assert _part_self_identifies_speaker("“I'm the boy who stayed behind.”", ("the boy",)) is False

    nick_intro = _block("c1", 0, "“My name’s Nick.”")
    allie_intro = _block("c1", 1, "“My name’s Allie.”")
    generic_line = _block("c1", 2, "“I'm the boy who stayed behind.”")
    nick_other = _block("c1", 3, "“Let’s go.”")

    facts = (
        _fact("f-nick-intro", nick_intro, "“My name’s Nick.”", "Nick", "inferred"),
        _fact("f-allie-intro", allie_intro, "“My name’s Allie.”", "Alexandra", "inferred"),
        _fact("f-generic", generic_line, "“I'm the boy who stayed behind.”", "Nick (the boy)", "inferred"),
        _fact("f-nick-other", nick_other, "“Let’s go.”", "Nick", "inferred"),
    )
    concepts = (
        _concept("c-nick", "Nick", "Ник", TermStatus.APPROVED),
        TermConcept(
            id="c-alex",
            book_id=DOCUMENT_ID,
            type=ConceptType.PERSON,
            canonical_source_text="Alexandra",
            status=TermStatus.APPROVED,
            variants=(TermVariant("c-alex", "ru", "Александра", TermStatus.APPROVED),),
            note="Краткое имя: Allie (Элли).",
        ),
    )
    section = _section("c1", 0, "Chapter 1", (nick_intro, allie_intro, generic_line, nick_other))
    doc = _document((section,))
    layout = build_book_layout(
        _inputs(doc, {}, (_analysis("a1", "c1", plan=_plan(facts=facts)),), concepts)
    )
    units = layout.chapters["c1"].units

    assert units[nick_intro.id][0].speaker is None
    assert units[nick_intro.id][0].notes == ()
    assert units[allie_intro.id][0].speaker is None
    assert units[allie_intro.id][0].notes == ()
    assert units[generic_line.id][0].speaker == "Ник"
    assert units[generic_line.id][0].notes[0].note.body == ("Говорит Ник",)
    assert units[nick_other.id][0].speaker == "Ник"
    assert units[nick_other.id][0].notes[0].note.body == ("Говорит Ник",)


def test_speech_style_subject_anchor_block_fallback_in_pattern_and_book_cap() -> None:
    """WP7: speaker matching uses subject_anchor + single-speaker block fallback, normalizes -in', and caps at 4 per book."""
    from linguary.application.layout.book import _speech_style_lexemes
    from linguary.domain.content_plan import Scene

    # 1. `-in'` normalization keeps companion lexemes (`yer`).
    probe_block = _block("probe", 0, "‘Yer deliverin’ that parcel, flyin’ high.’")
    d_probe = _difficulty(probe_block, "Yer deliverin’", "speech_style", "d-probe")
    assert _speech_style_lexemes(d_probe) == frozenset({"yer", "-in'"})

    # 2. Binding via `subject_anchor` (even when `fact.anchors` points at reporting clause),
    #    block-level single-speaker fallback (when anchor is outside both fact spans),
    #    `-in'` deduplication (`flyin'` vs `payin'`), and `_SPEECH_STYLE_BOOK_CAP = 4`.
    b1 = _block("c1", 0, "‘Yeh’re a wizard,’ Hagrid said softly.")
    b2 = _block("c1", 1, "‘Flyin’ over Bristol,’ Hagrid added.")
    b3 = _block("c1", 2, "‘Payin’ for it meself.’ He grinned. ‘Blimey!’")
    b4 = _block("c2", 0, "‘Codswallop!’ Hagrid barked.")
    b5 = _block("c2", 1, "‘Crikey, look at that!’ Hagrid laughed.")
    b6 = _block("c3", 0, "‘Gorgons alive!’ Hagrid roared.")

    d1 = _difficulty(b1, "Yeh’re", "speech_style", "d1")
    d2 = _difficulty(b2, "Flyin’", "speech_style", "d2")
    # `Payin'` is in b3, repeats `-in'` from d2; fact only anchors `‘Blimey!’` in b3 (block fallback!).
    d3_in = _difficulty(b3, "Payin’", "speech_style", "d3-in")
    # `Blimey` is unique (#3 kept).
    d3_unique = _difficulty(b3, "Blimey", "speech_style", "d3-unique")
    # `Codswallop` is unique (#4 kept — hits `_SPEECH_STYLE_BOOK_CAP = 4`).
    d4 = _difficulty(b4, "Codswallop", "speech_style", "d4")
    # `Crikey` and `Gorgons` are 5th & 6th unique lexemes -> dropped by `_SPEECH_STYLE_BOOK_CAP = 4`!
    d5 = _difficulty(b5, "Crikey", "speech_style", "d5")
    d6 = _difficulty(b6, "Gorgons", "speech_style", "d6")

    # Non-overlapping `fact.anchors` (reporting clause) vs `fact.subject_anchor` (the quote itself).
    f1 = ContextFact(
        "f1",
        "speaker",
        (_anchor(b1, "Hagrid said softly."),),
        _anchor(b1, "‘Yeh’re a wizard,’"),
        "Hagrid",
        "explicit",
    )
    f2 = ContextFact(
        "f2",
        "speaker",
        (_anchor(b2, "Hagrid added."),),
        _anchor(b2, "‘Flyin’ over Bristol,’"),
        "Hagrid",
        "explicit",
    )
    # Block-level fallback: `subject_anchor` is only `‘Blimey!’`, so `Payin’` does not overlap it.
    f3 = ContextFact(
        "f3",
        "speaker",
        (_anchor(b3, "He grinned."),),
        _anchor(b3, "‘Blimey!’"),
        "Hagrid",
        "inferred",
    )
    f4 = _fact("f4", b4, b4.text, "Hagrid", "explicit")
    f5 = _fact("f5", b5, b5.text, "Hagrid", "explicit")
    f6 = _fact("f6", b6, b6.text, "Hagrid", "explicit")

    p1 = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(d1, d2, d3_in, d3_unique),
        context_facts=(f1, f2, f3),
        scenes=(Scene("sc-1", 0, b1.id, b3.id, "Scene 1"),),
    )
    p2 = _plan(d4, d5, facts=(f4, f5))
    p3 = _plan(d6, facts=(f6,))

    s1 = _section("c1", 0, "Chapter 1", (b1, b2, b3))
    s2 = _section("c2", 1, "Chapter 2", (b4, b5))
    s3 = _section("c3", 2, "Chapter 3", (b6,))

    results = {
        b1.id: BlockProcessingResult(
            b1.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(d1, "пояснение d1", "sentence_rephrase"),),
        ),
        b2.id: BlockProcessingResult(
            b2.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(d2, "пояснение d2", "sentence_rephrase"),),
        ),
        b3.id: BlockProcessingResult(
            b3.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(
                _assistance(d3_in, "пояснение d3-in", "sentence_rephrase"),
                _assistance(d3_unique, "пояснение d3-unique", "sentence_rephrase"),
            ),
        ),
        b4.id: BlockProcessingResult(
            b4.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(d4, "пояснение d4", "sentence_rephrase"),),
        ),
        b5.id: BlockProcessingResult(
            b5.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(d5, "пояснение d5", "sentence_rephrase"),),
        ),
        b6.id: BlockProcessingResult(
            b6.id,
            BlockTreatment.READ_AS_IS,
            fragment_assistances=(_assistance(d6, "пояснение d6", "sentence_rephrase"),),
        ),
    }
    layout = build_book_layout(
        _inputs(
            _document((s1, s2, s3)),
            results,
            (
                _analysis("a1", s1.id, plan=p1),
                _analysis("a2", s2.id, plan=p2),
                _analysis("a3", s3.id, plan=p3),
            ),
        )
    )

    assert _labels(layout, b1.id) == ("Речь персонажа.",)
    assert _labels(layout, b2.id) == ("Речь персонажа.",)
    b3_bodies = [n.body for n in _notes(layout, b3.id) if n.origin != "speaker"]
    assert b3_bodies == [("пояснение d3-unique",)]
    assert _labels(layout, b4.id) == ("Речь персонажа.",)
    assert [n for n in _notes(layout, b5.id) if n.origin != "speaker"] == []
    assert [n for n in _notes(layout, b6.id) if n.origin != "speaker"] == []


def test_speech_style_fallback_does_not_ignore_multispeaker_owner_block() -> None:
    """If owner block has 2 speakers and a sibling anchor block has 1 speaker, fallback stays ambiguous."""
    from linguary.application.layout.book import _repeated_speech_style_ids
    from linguary.domain.content_plan import Scene

    b0 = _block("c1", 0, "First Harry line.")
    b1 = _block("c1", 1, "Second Harry line.")
    b2 = _block("c1", 2, "Owner line with two speakers.")
    b3 = _block("c1", 3, "Sibling line with one speaker.")
    a0 = SourceAnchor(b0.id, 0, 5, "First")
    a1 = SourceAnchor(b1.id, 0, 6, "Second")
    a2a = SourceAnchor(b2.id, 0, 5, "Owner")
    a2b = SourceAnchor(b2.id, 6, 10, "line")
    a3 = SourceAnchor(b3.id, 0, 7, "Sibling")
    f0 = ContextFact("f0", "speaker", (a0,), a0, "Harry", "explicit")
    f1 = ContextFact("f1", "speaker", (a1,), a1, "Harry", "explicit")
    f2a = ContextFact("f2a", "speaker", (a2a,), a2a, "Hagrid", "explicit")
    f2b = ContextFact("f2b", "speaker", (a2b,), a2b, "Harry", "explicit")
    f3 = ContextFact("f3", "speaker", (a3,), a3, "Harry", "explicit")
    d0 = Difficulty(
        "d-h0",
        (SourceAnchor(b0.id, 6, 11, "Harry"),),
        b0.id,
        "speech_style",
        "useful",
        "explanation",
        (RequiredPoint("p-h0", "Explain"),),
    )
    d1 = Difficulty(
        "d-h1",
        (SourceAnchor(b1.id, 7, 12, "Harry"),),
        b1.id,
        "speech_style",
        "useful",
        "explanation",
        (RequiredPoint("p-h1", "Explain"),),
    )
    # 3rd difficulty in the same scene: anchored on b2 (2 speakers) + b3 (1 speaker Harry).
    # Without the fix, b2 is skipped and d-ambig is misattributed to Harry -> dropped as 3rd in scene 1!
    d_ambig = Difficulty(
        "d-ambig",
        (SourceAnchor(b2.id, 11, 15, "with"), SourceAnchor(b3.id, 8, 12, "line")),
        b2.id,
        "speech_style",
        "useful",
        "explanation",
        (RequiredPoint("p-ambig", "Explain"),),
    )
    plan = ContentPlan(
        version=3,
        profile="content-plan-v3",
        difficulties=(d0, d1, d_ambig),
        context_facts=(f0, f1, f2a, f2b, f3),
        scenes=(Scene("sc-1", 0, b0.id, b3.id, "Scene 1"),),
    )
    sec = _section("c1", 0, "Ch 1", (b0, b1, b2, b3))
    dropped = _repeated_speech_style_ids(
        _document((sec,)),
        (_analysis("a1", sec.id, plan=plan),),
        plan.context_facts,
    )
    assert "d-ambig" not in dropped.get(sec.id, frozenset())


def test_same_chapter_hint_and_unlinked_note_dedup_beyond_20_blocks() -> None:
    """A4.4: same-chapter repeats beyond 20 blocks suppress without burning count; unlinked notes don't duplicate hints."""
    from linguary.domain import LanguageNote
    from linguary.domain.content_plan import ContentEvidence, HelpEvidence, artifact_key

    # Chapter 1:
    # - b0: 1st appearance of `the reptile house` and `ink` (FULL).
    # - b1: unlinked short-term LanguageNote for `reptile house` (same chapter -> suppressed!)
    #       AND cultural_reference LanguageNote for `ink` (linked cultural_reference -> kept!).
    # - b2..b24: 23 filler blocks (> 20-block local window).
    # - b25 (dist 25 > 20, SAME chapter 1): `reptile house` + `wand` in GLOSSARY_ONLY ->
    #       `reptile house` suppressed (no count burn!), `wand` shown FULL.
    # - b26 (dist 26 > 20, SAME chapter 1): sole `ink` in GLOSSARY_ONLY ->
    #       rescued compact (translation only) without burning count!
    # - b27 (dist 27 > 20, SAME chapter 1): `reptile house` in TRANSLATE -> suppressed.
    # - b28..b50: 23 filler blocks so chapter 2 is > 20 blocks away from rescued `ink`.
    # Chapter 2:
    # - c2_0: `reptile house` in GLOSSARY_ONLY -> 2nd cross-chapter occurrence, shown compact (stripped note).
    # - c2_1: `ink` in GLOSSARY_ONLY -> 2nd cross-chapter occurrence, shown compact (stripped note).
    # - c2_2: unlinked short-term LanguageNote for `wand` (hint was in ch1, NOT ch2 -> kept!).
    c1_b0 = _block("c1", 0, "They entered the reptile house and spilled ink.")
    c1_b1 = _block("c1", 1, "Inside the reptile house, dark ink stained the sign.")
    c1_fillers_1 = [_block("c1", idx, f"Filler one {idx}.") for idx in range(2, 25)]
    c1_b25 = _block("c1", 25, "The reptile house hid a wand.")
    c1_b26 = _block("c1", 26, "Drops of ink remained.")
    c1_b27 = _block("c1", 27, "He left the reptile house.")
    c1_fillers_2 = [_block("c1", idx, f"Filler two {idx}.") for idx in range(28, 51)]

    c2_b0 = _block("c2", 0, "He remembered the reptile house.")
    c2_b1 = _block("c2", 1, "He bought fresh ink.")
    c2_b2 = _block("c2", 2, "He picked up the wand.")

    reptile_h1 = GlossaryHint("the reptile house", "террариум", "Павильон с рептилиями.")
    reptile_h2 = GlossaryHint("reptile house", "террариум", "Повтор в той же главе.")
    reptile_h3 = GlossaryHint("reptile house", "террариум", "Повтор в следующей главе.")
    ink_h1 = GlossaryHint("ink", "чернила", "Жидкость для письма пером.")
    ink_h2 = GlossaryHint("ink", "чернила", "Повтор чернил в той же главе.")
    ink_h3 = GlossaryHint("ink", "чернила", "Повтор чернил во второй главе.")
    wand_h = GlossaryHint("wand", "палочка", "Волшебная палочка.")

    unlinked_reptile_note = LanguageNote("ln-rep", c1_b1.id, "reptile house", "Террариум в зоопарке.")
    cultural_ink_note = LanguageNote("ln-ink-cult", c1_b1.id, "ink", "В волшебном мире пишут перьями и чернилами.")
    c2_wand_note = LanguageNote("ln-wand-c2", c2_b2.id, "wand", "Палочка выбирает волшебника.")

    ink_cult_diff = _difficulty(c1_b1, "ink", "cultural_reference", "d-ink-cult")
    c1_plan = _plan(ink_cult_diff)
    ink_note_key = artifact_key(
        "language_note.note",
        c1_b1.id,
        {
            "original_excerpt": cultural_ink_note.original_excerpt,
            "note": cultural_ink_note.note,
            "translated_excerpt": cultural_ink_note.translated_excerpt,
        },
    )

    results = {
        c1_b0.id: BlockProcessingResult(
            c1_b0.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(reptile_h1, ink_h1),
        ),
        c1_b1.id: BlockProcessingResult(
            c1_b1.id,
            BlockTreatment.READ_AS_IS,
            language_notes=(unlinked_reptile_note, cultural_ink_note),
            content_evidence=ContentEvidence(
                c1_plan.sha256,
                (
                    HelpEvidence(
                        ink_cult_diff.id,
                        ink_cult_diff.required_points[0].id,
                        "language_note.note",
                        c1_b1.id,
                        ink_note_key,
                        cultural_ink_note.note,
                    ),
                ),
            ),
        ),
        c1_b25.id: BlockProcessingResult(
            c1_b25.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(reptile_h2, wand_h),
        ),
        c1_b26.id: BlockProcessingResult(
            c1_b26.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(ink_h2,),
        ),
        c1_b27.id: BlockProcessingResult(
            c1_b27.id,
            BlockTreatment.TRANSLATE,
            translated_text="Он вышел из террариума.",
            glossary_hints=(reptile_h2,),
        ),
        c2_b0.id: BlockProcessingResult(
            c2_b0.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(reptile_h3,),
        ),
        c2_b1.id: BlockProcessingResult(
            c2_b1.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=(ink_h3,),
        ),
        c2_b2.id: BlockProcessingResult(
            c2_b2.id,
            BlockTreatment.READ_AS_IS,
            language_notes=(c2_wand_note,),
        ),
    }

    sections = (
        _section("c1", 0, "Chapter 1", (c1_b0, c1_b1, *c1_fillers_1, c1_b25, c1_b26, c1_b27, *c1_fillers_2)),
        _section("c2", 1, "Chapter 2", (c2_b0, c2_b1, c2_b2)),
    )
    layout = build_book_layout(
        _inputs(_document(sections), results, (_analysis("a-c1", "c1", plan=c1_plan),))
    )

    # c1_b0: first appearances shown FULL.
    assert [(n.term, n.body) for n in _notes(layout, c1_b0.id)] == [
        ("the reptile house", ("террариум", "Павильон с рептилиями.")),
        ("ink", ("чернила", "Жидкость для письма пером.")),
    ]
    # c1_b1: unlinked short-term `reptile house` suppressed by same-chapter hint; cultural `ink` kept.
    assert [(n.origin, n.body) for n in _notes(layout, c1_b1.id)] == [
        ("language_note", ("В волшебном мире пишут перьями и чернилами.",)),
    ]
    # c1_b25 (> 20 blocks later, SAME chapter): `reptile house` suppressed, `wand` shown FULL.
    assert [(n.term, n.body) for n in _notes(layout, c1_b25.id)] == [
        ("wand", ("палочка", "Волшебная палочка.")),
    ]
    # c1_b26 (> 20 blocks later, SAME chapter, sole GLOSSARY_ONLY hint): rescued compact.
    assert [(n.term, n.body) for n in _notes(layout, c1_b26.id)] == [
        ("ink", ("чернила",)),
    ]
    # c1_b27 (> 20 blocks later, SAME chapter, TRANSLATE): suppressed.
    assert _notes(layout, c1_b27.id) == ()
    # c2_b0 & c2_b1 (NEXT chapter): 2nd compact occurrence still shown (same-chapter repeats didn't burn count).
    assert [(n.term, n.body) for n in _notes(layout, c2_b0.id)] == [
        ("reptile house", ("террариум",)),
    ]
    assert [(n.term, n.body) for n in _notes(layout, c2_b1.id)] == [
        ("ink", ("чернила",)),
    ]
    # c2_b2 (NEXT chapter): unlinked short-term `wand` note not suppressed by ch1 hint.
    assert [(n.origin, n.body) for n in _notes(layout, c2_b2.id)] == [
        ("language_note", ("Палочка выбирает волшебника.",)),
    ]

    # Same-block complementary GlossaryHint + unlinked LanguageNote on the same word are BOTH kept.
    same_blk = _block("c1", 99, "Alice was very tired.")
    same_doc = _document((_section("c1", 0, "Chapter 1", (same_blk,)),))
    same_res = BlockProcessingResult(
        same_blk.id,
        BlockTreatment.TRANSLATE,
        translated_text="Алиса очень устала.",
        glossary_hints=(GlossaryHint("tired", "уставать"),),
        language_notes=(
            LanguageNote("n-same", same_blk.id, "tired", "Разговорная заметка."),
        ),
    )
    same_layout = build_book_layout(_inputs(same_doc, {same_blk.id: same_res}))
    assert [n.origin for n in _notes(same_layout, same_blk.id)] == [
        "glossary_hint",
        "language_note",
    ]




