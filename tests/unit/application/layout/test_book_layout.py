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
) -> ChapterAnalysis:
    return ChapterAnalysis(
        id=analysis_id,
        section_id=section_id,
        coverage_status=CoverageStatus.INCOMPLETE,
        treatment_plan=TreatmentPlan(section_id),
        translation_instructions="Держи имена.",
        chapter_summary="Кратко.",
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
    assert layout.features == LayoutFeatures(True, False, frozenset(), False, False, False, frozenset())
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
    `previously` is absent. Before You Read counts only as a section key.
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
        sections=frozenset({"before_you_read"}),
    )
    assert "previously" not in layout.features.sections
    assert "review" not in layout.features.sections
    assert layout.legend[0] == "Абзац оригинала всегда идёт первым."
    assert "что было раньше" not in "\n".join(layout.legend)
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
    assert "previously" not in layout.features.sections
    assert "who_is_who" not in layout.features.sections
    assert "review" not in layout.features.sections
    assert layout.features.sections == frozenset()


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

    first = _block("one", 0, "‘We was late,’ Hagrid said.")
    second = _block("two", 0, "‘Yeh’ll see,’ Hagrid said.")
    early = _difficulty(first, "We was", "speech_style", "d-early")
    late = _difficulty(second, "Yeh’ll", "speech_style", "d-late")
    kept_grammar = _difficulty(second, "see", "grammar", "d-gram")
    first_fact = _fact("f1", first, first.text, "Hagrid", "explicit")
    second_fact = _fact("f2", second, second.text, "Hagrid", "explicit")
    first_plan = _plan(early, facts=(first_fact,))
    second_plan = _plan(late, kept_grammar, facts=(second_fact,))
    first_section = _section("one", 0, "Chapter 1", (first,))
    second_section = _section("two", 1, "Chapter 2", (second,))

    early_note = LanguageNote("n1", first.id, "We was", "Просторечное согласование.")
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

    first = _block("one", 0, "‘We was late,’ Hagrid said.")
    second = _block("two", 0, "‘Yeh’ll see,’ Hagrid said.")
    third = _block("two", 1, "‘An’ don’t forget,’ Hagrid said.")
    early = _difficulty(first, "We was", "speech_style", "d-early")
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
    raw_text = "Сказал: „Привет“ и „Ещё” – потом „обрыв и „Точка“ и требовать\u00adся."
    result = BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, translated_text=raw_text)

    ru_layout = build_book_layout(
        _inputs(_document((section,), target_language="ru"), {block.id: result})
    )
    ru_translation = ru_layout.chapters[section.id].units[block.id][0].translation
    assert ru_translation == "Сказал: «Привет» и «Ещё» — потом „обрыв и «Точка» и требоваться."

    de_text = "Er sagte: „Hallo“ – und „Welt” – und tren\u00adnen."
    de_result = BlockProcessingResult(block.id, BlockTreatment.TRANSLATE, translated_text=de_text)
    de_layout = build_book_layout(
        _inputs(_document((section,), target_language="de"), {block.id: de_result})
    )
    de_translation = de_layout.chapters[section.id].units[block.id][0].translation
    assert de_translation == "Er sagte: „Hallo“ – und „Welt” – und trennen."
