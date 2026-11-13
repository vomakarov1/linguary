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
from linguary.application.layout.review import ReviewEntry
from linguary.application.layout.who_is_who import WhoEntry

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


def _document(sections: Sequence[DocumentSection]) -> CanonicalDocument:
    return CanonicalDocument(
        id=DOCUMENT_ID,
        metadata=BookMetadata(title="Wonderland", authors=("Carroll",)),
        source_language="en",
        target_language="ru",
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

    Empty `segment_treatments` repeats the block treatment once per span, so a
    long translated block still splits. `translated_text` is not copied onto
    either part, and neither block grows notes.
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
    split = chapter.units[repeated.id]
    assert [(unit.start, unit.end) for unit in split] == [(0, 300), (300, 600)]
    assert [unit.translation for unit in split] == [None, None]
    assert [unit.notes for unit in split] == [(), ()]
    assert (split[0].continues, split[0].ends_paragraph) == (False, False)
    assert (split[1].continues, split[1].ends_paragraph) == (True, True)
    assert layout.features == LayoutFeatures(False, False, frozenset(), False, False, False, frozenset())
    assert layout.legend == ()
    assert layout.ipa_key == ()
    assert layout.who_is_who == ()


def test_repeated_speech_style_same_speaker_dropped() -> None:
    """The later speech style of one speaker leaves no note. The plan stays.

    `a-drawl` sorts before `z-drawl`, and the second fact is only inferred.
    Reading order still keeps the earlier explicit occurrence, label and body.
    The stored plans are the same objects and still contain both difficulties.
    """
    first = _block("one", 0, "Alice drawled softly.")
    second = _block("two", 0, "Alice drawled softly.")
    early = _difficulty(first, "drawled", "speech_style", "z-drawl")
    late = _difficulty(second, "drawled", "speech_style", "a-drawl")
    early_fact = _fact("f-explicit", first, "Alice drawled softly.", "Alice", "explicit")
    late_fact = _fact("f-inferred", second, "Alice drawled softly.", "Alice", "inferred")
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
    assert tuple(note.body for note in _notes(layout, second.id)) == (("Говорит Alice",),)
    assert layout.chapters[first_section.id].units[first.id][0].speaker == "Alice"
    assert layout.chapters[second_section.id].units[second.id][0].speaker == "Alice"


def test_speech_style_without_speaker_not_dropped() -> None:
    """Speech styles that overlap no speaker fact both stay, in either order."""
    rough = _block("one", 0, "The sailor spoke roughly today.")
    again = _block("one", 1, "The sailor spoke roughly again.")
    named = _block("one", 2, "Alice nodded once.")
    first = _difficulty(rough, "roughly", "speech_style", "z-rough")
    second = _difficulty(again, "roughly", "speech_style", "a-rough")
    fact = _fact("f-alice", named, "Alice", "Alice", "explicit")
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
        (_analysis("a-one", section.id, plan=plan),),
    ))

    assert _labels(layout, rough.id) == ("Речь персонажа.",)
    assert _labels(layout, again.id) == ("Речь персонажа.",)
    assert layout.chapters[section.id].units[named.id][0].speaker == "Alice"
    assert layout.chapters[section.id].units[rough.id][0].speaker is None
    assert plan.difficulties[0].id == "a-rough"


def test_unmatched_speaker_not_shown() -> None:
    """Unknown, gender, and a fact anchored elsewhere are not speakers.

    A user decision replaces the source name. A suggested variant does not:
    the source value is shown instead.
    """
    alice = _block("one", 0, "Alice waited here.")
    hatter = _block("one", 1, "Hatter waited here.")
    unknown = _block("one", 2, "Someone waited.")
    alice_fact = _fact("f-alice", alice, "Alice", "Alice", "explicit")
    hatter_fact = _fact("f-hatter", hatter, "Hatter", "Hatter", "inferred")
    unknown_anchor = _anchor(unknown, "Someone waited.")
    unknown_fact = ContextFact(
        "f-unknown",
        "speaker",
        (unknown_anchor,),
        unknown_anchor,
        value=None,
        certainty="unknown",
    )
    gender = _fact("f-gender", unknown, "Someone waited.", "female", "explicit", kind="gender")
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
    assert units[hatter.id][0].speaker == "Hatter"
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


def test_features_collected_for_legend() -> None:
    """Flags, labels, IPA, and section keys come from the finished projection.

    `english_help` hides the lexical-hint label, so it is not a legend label.
    `previously` is absent. Before You Read counts only as a section key.
    """
    poem = _block("one", 0, "Twinkle, twinkle, little bat.", kind=BlockType.POEM)
    grammar_block = _block("one", 1, "She had a curious dream.")
    wordplay_block = _block("one", 2, "The mouse's tale was long.")
    pronounced = _block("one", 3, "The cat sat.")
    speaker_block = _block("one", 4, "Alice said hello.")
    hinted = _block("one", 5, "Past prose.")
    current = _block("two", 0, "She saw a lantern.")
    grammar = _difficulty(grammar_block, "curious", "grammar", "d-grammar")
    wordplay = _difficulty(
        wordplay_block, "tale", "wordplay", "d-wordplay", help_kind="lexical_hint"
    )
    speaker = _fact("f-alice", speaker_block, "Alice said hello.", "Alice", "inferred")
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

    assert layout.chapters[first.id].units[poem.id][0].lines is True
    assert layout.chapters[first.id].units[poem.id][0].translation == translated
    assert layout.chapters[first.id].units[speaker_block.id][0].speaker == "Алиса"
    assert _labels(layout, grammar_block.id) == ("Грамматика.",)
    assert _labels(layout, wordplay_block.id) == ()
    assert any(note.ipa == "kæt" for note in _notes(layout, pronounced.id))
    assert layout.features == LayoutFeatures(
        has_translation=True,
        has_notes=True,
        labels=frozenset({"Грамматика."}),
        has_ipa=True,
        has_poems=True,
        has_speakers=True,
        sections=frozenset({"who_is_who", "review", "before_you_read"}),
    )
    assert "previously" not in layout.features.sections
    assert layout.legend[0] == "Абзац оригинала всегда идёт первым."
    assert "что было раньше" not in "\n".join(layout.legend)
    assert layout.ipa_key == ipa_key(("kæt",))
    assert layout.legend == legend_lines(layout.features)
    assert layout.who_is_who == (
        WhoEntry("Alice", "Алиса", ConceptType.PERSON, "девочка", None, 1),
    )
    assert layout.chapters[second.id].review == (ReviewEntry("lantern", "фонарь", 1),)
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
    assert layout.chapters[first.id].who_is_who == (
        WhoEntry("Alice", "Элис", ConceptType.PERSON, "девочка", None, 1),
    )
    assert layout.chapters[second.id].review == (ReviewEntry("lantern", "фонарь", 1),)
    assert layout.who_is_who == (
        WhoEntry("Alice", "Элис", ConceptType.PERSON, "девочка", None, 1),
    )
    front_unit = layout.chapters[front.id].units[front_block.id][0]
    note_unit = layout.chapters[notes.id].units[note.id][0]
    assert (front_unit.start, front_unit.end, front_unit.notes) == (0, len(front_block.text), ())
    assert (note_unit.start, note_unit.end, note_unit.notes) == (0, len(note.text), ())
    assert "previously" not in layout.features.sections
    assert layout.features.sections == frozenset({"who_is_who", "review"})


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
