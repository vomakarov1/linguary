"""Who-is-who and review lists projected from stored chapters.

Chapter numbers in these tests are content numbers: a front-matter section and
a notes-only section do not count. The lists themselves carry no headings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from linguary.application.layout.review import (
    REVIEW_DISTANCES,
    REVIEW_LIMITS,
    ReviewEntry,
    chapter_review,
)
from linguary.application.layout.who_is_who import (
    NAME_METHOD_LABELS,
    WhoEntry,
    book_who_is_who,
    chapter_who_is_who,
)
from linguary.domain import (
    BlockTreatment,
    BlockType,
    BookMetadata,
    CanonicalDocument,
    ChapterAnalysis,
    ConceptType,
    CoverageStatus,
    DocumentBlock,
    DocumentSection,
    GlossaryCandidate,
    GlossaryHint,
    LanguageNote,
    SourceReference,
    TermConcept,
    TermStatus,
    TermVariant,
    TreatmentPlan,
)
from linguary.domain.name_translation import (
    NameTranslationAssessment,
    NameTranslationEvidence,
    NameTranslationMethod,
    NameTranslationObservation,
    NameTranslationStrategy,
)
from linguary.domain.processing import BlockProcessingResult

DOCUMENT_ID = "book-1"


def _block(
    section_id: str,
    resource: str,
    order: int,
    text: str,
    *,
    kind: BlockType = BlockType.PARAGRAPH,
) -> DocumentBlock:
    """One block. The structural path keeps its identifier unique in the section."""
    return DocumentBlock.create(
        document_id=DOCUMENT_ID,
        section_id=section_id,
        type=kind,
        order=order,
        text=text,
        source_reference=SourceReference(resource, f"/body/p[{order + 1}]"),
    )


def _section(
    section_id: str,
    order: int,
    title: str,
    resource: str,
    blocks: Sequence[DocumentBlock],
) -> DocumentSection:
    return DocumentSection(
        id=section_id,
        document_id=DOCUMENT_ID,
        title=title,
        order=order,
        source_resource_path=resource,
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


def _front(order: int, section_id: str = "front") -> DocumentSection:
    """A contents page. The title classifies it as front matter."""
    resource = f"{section_id}.xhtml"
    return _section(
        section_id,
        order,
        "Contents",
        resource,
        (_block(section_id, resource, 0, "Table of contents."),),
    )


def _notes(order: int, section_id: str = "notes") -> DocumentSection:
    """A page of footnotes. It is not a content chapter."""
    resource = f"{section_id}.xhtml"
    return _section(
        section_id,
        order,
        "Notes",
        resource,
        (_block(section_id, resource, 0, "A footnote.", kind=BlockType.FOOTNOTE),),
    )


def _content(
    section_id: str,
    order: int,
    blocks: Sequence[DocumentBlock] | None = None,
) -> DocumentSection:
    """A story chapter. The title keeps it out of front matter."""
    resource = f"{section_id}.xhtml"
    chosen = blocks if blocks is not None else (_block(section_id, resource, 0, "Chapter text."),)
    return _section(section_id, order, f"Chapter {order}", resource, chosen)


def _candidate(
    analysis_id: str,
    candidate_id: str,
    source: str,
    concept_type: ConceptType,
    translation: str,
    *,
    description: str = "",
    method: NameTranslationMethod | None = None,
) -> GlossaryCandidate:
    """One glossary candidate. `method` attaches a name-translation assessment."""
    evidence = None
    if method is not None:
        evidence = NameTranslationEvidence(
            1,
            "conditions-1",
            1,
            NameTranslationStrategy.MEANING,
            (
                NameTranslationObservation(
                    translation,
                    0.9,
                    False,
                    NameTranslationAssessment(method, True, (), "Оценка способа"),
                ),
            ),
        )
    return GlossaryCandidate(
        id=candidate_id,
        analysis_id=analysis_id,
        source_text=source,
        type=concept_type,
        suggested_translation=translation,
        confidence=0.9,
        model_flagged_review=False,
        requires_review=concept_type is not ConceptType.TERM,
        description=description,
        name_translation_evidence=evidence,
    )


def _analysis(
    analysis_id: str,
    section_id: str,
    candidates: Sequence[GlossaryCandidate],
) -> ChapterAnalysis:
    return ChapterAnalysis(
        id=analysis_id,
        section_id=section_id,
        coverage_status=CoverageStatus.COMPLETE,
        treatment_plan=TreatmentPlan(section_id),
        translation_instructions="Держи имена.",
        chapter_summary="Кратко.",
        glossary_candidates=tuple(candidates),
    )


def _concept(concept_id: str, source: str, translation: str, status: TermStatus) -> TermConcept:
    """A concept whose Russian variant has `status`."""
    return TermConcept(
        id=concept_id,
        book_id=DOCUMENT_ID,
        type=ConceptType.PERSON,
        canonical_source_text=source,
        status=status,
        variants=(TermVariant(concept_id, "ru", translation, status),),
    )


def _result(
    block: DocumentBlock,
    hints: Sequence[tuple[str, str]] = (),
    excerpts: Sequence[str] = (),
) -> BlockProcessingResult:
    """A block result. Hints require glossary-only; a note alone is read-as-is."""
    notes = tuple(
        LanguageNote(f"note-{block.order}-{index}", block.id, excerpt, "Уже пояснено.")
        for index, excerpt in enumerate(excerpts, start=1)
    )
    if hints:
        return BlockProcessingResult(
            block.id,
            BlockTreatment.GLOSSARY_ONLY,
            glossary_hints=tuple(GlossaryHint(original, translation) for original, translation in hints),
            language_notes=notes,
        )
    return BlockProcessingResult(block.id, BlockTreatment.READ_AS_IS, language_notes=notes)


def _chapters(
    specs: Sequence[Sequence[Sequence[tuple[str, str]]]],
) -> tuple[CanonicalDocument, dict[str, BlockProcessingResult], tuple[str, ...]]:
    """Content chapters only. `specs[chapter][block]` is that block's hints.

    Returns the document, results for blocks that have hints, and section ids
    in content order.
    """
    sections: list[DocumentSection] = []
    results: dict[str, BlockProcessingResult] = {}
    section_ids: list[str] = []
    for index, blocks in enumerate(specs):
        section_id = f"ch-{index + 1}"
        resource = f"{section_id}.xhtml"
        built = []
        for block_order, hints in enumerate(blocks):
            block = _block(section_id, resource, block_order, "Chapter prose.")
            built.append(block)
            if hints:
                results[block.id] = _result(block, hints)
        sections.append(_content(section_id, index, built))
        section_ids.append(section_id)
    return _document(sections), results, tuple(section_ids)


def test_chapter_lists_only_first_mentioned_people_and_places() -> None:
    """People, then places, in candidate order. Later chapters and other types are absent.

    A key first seen in front matter or on a notes page is not a content mention.
    The same key later in the book is not listed again. The tuple has no headings
    and no «Упоминаются» group.
    """
    front = _front(0)
    first = _content("ch-1", 1)
    notes = _notes(2)
    second = _content("ch-2", 3)
    document = _document((front, first, notes, second))
    analyses = (
        _analysis("front-analysis", front.id, (
            _candidate("front-analysis", "printer", "Printer", ConceptType.PERSON, "Печатник"),
            _candidate("front-analysis", "front-alice", "Alice", ConceptType.PERSON, "Титульная", description="титул"),
        )),
        _analysis("ch-1-analysis", first.id, (
            _candidate("ch-1-analysis", "london", "London", ConceptType.PLACE, "Лондон"),
            _candidate("ch-1-analysis", "alice", "Alice", ConceptType.PERSON, "Элис"),
            _candidate("ch-1-analysis", "ministry", "The Ministry", ConceptType.ORGANIZATION, "Министерство"),
            _candidate("ch-1-analysis", "bob", "Bob", ConceptType.PERSON, "Боб"),
            _candidate("ch-1-analysis", "wonka", "Mr. Wonka", ConceptType.PERSON, "Вонка"),
        )),
        _analysis("notes-analysis", notes.id, (
            _candidate("notes-analysis", "notes-paris", "Paris", ConceptType.PLACE, "Из сносок", description="сноска"),
            _candidate("notes-analysis", "notes-wonka", "Mr Wonka", ConceptType.PERSON, "Мистер Вонка"),
        )),
        _analysis("ch-2-analysis", second.id, (
            _candidate("ch-2-analysis", "again-alice", "Alice", ConceptType.PERSON, "Алисия"),
            _candidate("ch-2-analysis", "paris", "Paris", ConceptType.PLACE, "Париж", description="Город"),
            _candidate("ch-2-analysis", "tea", "tea", ConceptType.TERM, "чай"),
            _candidate("ch-2-analysis", "again-wonka", "Mr Wonka", ConceptType.PERSON, "Мистер Вонка"),
            _candidate("ch-2-analysis", "glass", "Looking-Glass", ConceptType.FICTIONAL_CONCEPT, "Зазеркалье"),
        )),
    )

    assert chapter_who_is_who(document, analyses, (), section_id=front.id) == ()
    assert chapter_who_is_who(document, analyses, (), section_id=notes.id) == ()
    assert chapter_who_is_who(document, analyses, (), section_id=first.id) == (
        WhoEntry("Alice", "Элис", ConceptType.PERSON, "", None, 1),
        WhoEntry("Bob", "Боб", ConceptType.PERSON, "", None, 1),
        WhoEntry("Mr. Wonka", "Вонка", ConceptType.PERSON, "", None, 1),
        WhoEntry("London", "Лондон", ConceptType.PLACE, "", None, 1),
    )
    assert chapter_who_is_who(document, analyses, (), section_id=second.id) == (
        WhoEntry("Paris", "Париж", ConceptType.PLACE, "Город", None, 2),
    )


def test_description_never_from_later_chapter() -> None:
    """The description is the first-mention candidate's, even when that text is empty."""
    first = _content("ch-1", 0)
    second = _content("ch-2", 1)
    document = _document((first, second))
    analyses = (
        _analysis("a1", first.id, (
            _candidate("a1", "alice", "Alice", ConceptType.PERSON, "Элис", description="Девочка в начале."),
            _candidate("a1", "bob", "Bob", ConceptType.PERSON, "Боб", description=""),
        )),
        _analysis("a2", second.id, (
            _candidate("a2", "alice-later", "Alice", ConceptType.PERSON, "Алиса", description="Королева."),
            _candidate("a2", "bob-later", "Bob", ConceptType.PERSON, "Боб", description="Спойлер."),
            _candidate("a2", "carol", "Carol", ConceptType.PERSON, "Кэрол", description="Новая."),
        )),
    )

    assert chapter_who_is_who(document, analyses, (), section_id=first.id) == (
        WhoEntry("Alice", "Элис", ConceptType.PERSON, "Девочка в начале.", None, 1),
        WhoEntry("Bob", "Боб", ConceptType.PERSON, "", None, 1),
    )
    assert chapter_who_is_who(document, analyses, (), section_id=second.id) == (
        WhoEntry("Carol", "Кэрол", ConceptType.PERSON, "Новая.", None, 2),
    )
    assert book_who_is_who(document, analyses, ()) == (
        WhoEntry("Alice", "Элис", ConceptType.PERSON, "Девочка в начале.", None, 1),
        WhoEntry("Bob", "Боб", ConceptType.PERSON, "", None, 1),
        WhoEntry("Carol", "Кэрол", ConceptType.PERSON, "Новая.", None, 2),
    )


def test_user_decision_translation_preferred() -> None:
    """An approved or locked variant wins. A suggested variant does not replace the model."""
    first = _content("ch-1", 0)
    second = _content("ch-2", 1)
    document = _document((first, second))
    analyses = (
        _analysis("a1", first.id, (
            _candidate("a1", "alice", "Alice", ConceptType.PERSON, "Элис"),
            _candidate("a1", "bob", "Bob", ConceptType.PERSON, "Боб"),
            _candidate("a1", "carol", "Carol", ConceptType.PERSON, "Кэрол"),
            _candidate("a1", "dave", "Dave", ConceptType.PERSON, "Дейв"),
        )),
        _analysis("a2", second.id, (
            _candidate("a2", "alice-later", "Alice", ConceptType.PERSON, "Алисия"),
        )),
    )
    concepts = (
        _concept("alice", "ALICE", "Алиса", TermStatus.APPROVED),
        _concept("bob", "Bob", "Роберт", TermStatus.SUGGESTED),
        _concept("dave", "Dave", "Дэйв", TermStatus.LOCKED),
    )

    assert chapter_who_is_who(document, analyses, concepts, section_id=first.id) == (
        WhoEntry("Alice", "Алиса", ConceptType.PERSON, "", None, 1),
        WhoEntry("Bob", "Боб", ConceptType.PERSON, "", None, 1),
        WhoEntry("Carol", "Кэрол", ConceptType.PERSON, "", None, 1),
        WhoEntry("Dave", "Дэйв", ConceptType.PERSON, "", None, 1),
    )
    assert book_who_is_who(document, analyses, concepts)[0] == WhoEntry(
        "Alice", "Алиса", ConceptType.PERSON, "", None, 1,
    )


def test_book_list_uses_first_appearance() -> None:
    """One entry per key, from the first content chapter, ignoring analysis order."""
    front = _front(0)
    first = _content("ch-1", 1)
    notes = _notes(2)
    second = _content("ch-2", 3)
    document = _document((front, first, notes, second))
    analyses = (
        _analysis("a2", second.id, (
            _candidate("a2", "alice-2", "Alice", ConceptType.PERSON, "Алисия", description="вторая"),
            _candidate("a2", "land-2", "Footnote Land", ConceptType.PLACE, "Страна сносок", description="из главы"),
            _candidate("a2", "bob", "Bob", ConceptType.PERSON, "Боб"),
            _candidate("a2", "paris", "Paris", ConceptType.PLACE, "Париж"),
        )),
        _analysis("notes", notes.id, (
            _candidate("notes", "land-1", "Footnote Land", ConceptType.PLACE, "Сноска", description="из сносок"),
        )),
        _analysis("a1", first.id, (
            _candidate("a1", "alice", "Alice", ConceptType.PERSON, "Элис", description="первая"),
            _candidate("a1", "london", "London", ConceptType.PLACE, "Лондон"),
        )),
        _analysis("front", front.id, (
            _candidate("front", "printer", "Printer", ConceptType.PERSON, "Печатник"),
        )),
    )

    assert book_who_is_who(document, analyses, ()) == (
        WhoEntry("Alice", "Элис", ConceptType.PERSON, "первая", None, 1),
        WhoEntry("Bob", "Боб", ConceptType.PERSON, "", None, 2),
        WhoEntry("London", "Лондон", ConceptType.PLACE, "", None, 1),
        WhoEntry("Footnote Land", "Страна сносок", ConceptType.PLACE, "из главы", None, 2),
        WhoEntry("Paris", "Париж", ConceptType.PLACE, "", None, 2),
    )


def test_review_uses_scheduled_chapters() -> None:
    """Distances 1, 3, and 7 only. A repeated expression keeps the earliest chapter.

    The repeated expression is ordered with the nearer chapter. Its text and
    `first_chapter` come from the earlier one.
    """
    hints: list[tuple[tuple[tuple[str, str], ...], ...]] = [((),) for _ in range(8)]
    hints[0] = ((("odd phrase", "странная фраза"), ("seven leagues", "семь лье")),)
    hints[1] = ((("six leagues", "шесть лье"),),)
    hints[3] = ((("four leagues", "четыре лье"),),)
    hints[4] = ((("three leagues", "три лье"),),)
    hints[5] = ((("two leagues", "два лье"),),)
    hints[6] = ((("one league", "одно лье"), ("odd phrase", "странная фраза")),)
    document, results, section_ids = _chapters(hints)

    assert chapter_review(document, results, section_id=section_ids[7], reader_level="a2") == (
        ReviewEntry("one league", "одно лье", 7),
        ReviewEntry("odd phrase", "странная фраза", 1),
        ReviewEntry("three leagues", "три лье", 5),
        ReviewEntry("seven leagues", "семь лье", 1),
    )


def test_review_skips_ambiguous_word() -> None:
    """A word with two translations anywhere in the book is left out."""
    document, results, section_ids = _chapters((
        ((("groan", "стон"),),),
        ((("groaned", "стонал"),),),
        ((("lantern", "фонарь"),),),
        ((),),
    ))

    entries = chapter_review(document, results, section_id=section_ids[3], reader_level="b1")

    assert entries == (ReviewEntry("lantern", "фонарь", 3),)
    assert all("groan" not in entry.expression for entry in entries)


def test_review_keeps_unambiguous_single_word() -> None:
    """One word with one translation is kept. Inflections are the same expression."""
    hints: list[tuple[tuple[tuple[str, str], ...], ...]] = [((),) for _ in range(8)]
    hints[0] = ((("lanterns", "фонарь"),),)
    hints[6] = ((("lantern", "фонарь"),),)
    document, results, section_ids = _chapters(hints)

    assert chapter_review(document, results, section_id=section_ids[7], reader_level="c1") == (
        ReviewEntry("lanterns", "фонарь", 1),
    )


def test_review_skips_expressions_noted_in_chapter() -> None:
    """A hint or a language note already in this chapter drops the expression.

    Word forms share a stem: `heaved to` notes `heave to`.
    """
    past = _block("ch-1", "ch-1.xhtml", 0, "Chapter prose.")
    current_hints = _block("ch-2", "ch-2.xhtml", 0, "More prose.")
    first = _content("ch-1", 0, (past,))
    second = _content("ch-2", 1, (current_hints,))
    document = _document((first, second))
    results = {
        past.id: _result(past, (
            ("heave to", "лечь в дрейф"),
            ("kid gloves", "лайковые перчатки"),
            ("white rabbit", "белый кролик"),
        )),
        current_hints.id: _result(current_hints, (("heaved to", "лечь в дрейф"),), ("Kid Gloves",)),
    }

    assert chapter_review(document, results, section_id=second.id, reader_level="a2") == (
        ReviewEntry("white rabbit", "белый кролик", 1),
    )


def test_review_limit_by_level() -> None:
    """Nearer chapters fill the cap before older ones. An unknown level is empty."""
    assert REVIEW_DISTANCES == (1, 3, 7)
    assert REVIEW_LIMITS == {"a2": 8, "b1": 6, "b2": 4, "c1": 3}
    near = (
        (("signal zebra", "значение"), ("signal yankee", "значение")),
        (("signal xray", "значение"), ("signal whiskey", "значение"), ("signal victor", "значение")),
    )
    mid = ((
        ("signal uniform", "значение"),
        ("signal tango", "значение"),
        ("signal sierra", "значение"),
        ("signal romeo", "значение"),
        ("signal quebec", "значение"),
    ),)
    far = ((("signal papa", "значение"), ("signal oscar", "значение")),)
    specs: list[tuple[tuple[tuple[str, str], ...], ...]] = [((),) for _ in range(8)]
    specs[0] = far
    specs[4] = mid
    specs[6] = near
    document, results, section_ids = _chapters(specs)
    current = section_ids[7]

    def expressions(level: str | None) -> tuple[str, ...]:
        return tuple(
            entry.expression
            for entry in chapter_review(document, results, section_id=current, reader_level=level)
        )

    assert expressions("a2") == (
        "signal zebra",
        "signal yankee",
        "signal xray",
        "signal whiskey",
        "signal victor",
        "signal uniform",
        "signal tango",
        "signal sierra",
    )
    assert expressions("b1") == expressions("a2")[:6]
    assert expressions("b2") == expressions("a2")[:4]
    assert expressions("c1") == expressions("a2")[:3]
    assert expressions(None) == ()
    assert expressions("") == ()
    assert expressions("native") == ()


def test_front_matter_does_not_shift_review_distance() -> None:
    """A contents page and a notes page between chapters do not change distances."""
    opening = _block("ch-1", "ch-1.xhtml", 0, "Opening prose.")
    middle = _block("ch-2", "ch-2.xhtml", 0, "Middle prose.")
    closing = _block("ch-3", "ch-3.xhtml", 0, "Closing prose.")
    first = _content("ch-1", 0, (opening,))
    front = _front(1, "contents")
    second = _content("ch-2", 2, (middle,))
    notes = _notes(3)
    third = _content("ch-3", 4, (closing,))
    document = _document((first, front, second, notes, third))
    results: Mapping[str, BlockProcessingResult] = {
        opening.id: _result(opening, (("opening phrase", "начало"),)),
        middle.id: _result(middle, (("middle phrase", "середина"),)),
    }

    assert chapter_review(document, results, section_id=second.id, reader_level="a2") == (
        ReviewEntry("opening phrase", "начало", 1),
    )
    assert chapter_review(document, results, section_id=third.id, reader_level="b2") == (
        ReviewEntry("middle phrase", "середина", 2),
    )


def test_name_method_labels_match_enum() -> None:
    """Label keys are the name-translation methods, including those not shown."""
    assert set(NAME_METHOD_LABELS) == {method.value for method in NameTranslationMethod}
    assert NAME_METHOD_LABELS == {
        "sound": "по звучанию",
        "meaning": "по смыслу",
        "mixed": "звучание и смысл",
        "creative": "вольный перевод",
        "uncertain": "способ неясен",
    }


def test_name_method_shown_only_for_mixed_and_creative() -> None:
    """Sound, meaning, and uncertain stay out of the row. Mixed and creative are labelled."""
    section = _content("ch-1", 0)
    document = _document((section,))
    methods = (
        ("sound-name", "Sound", NameTranslationMethod.SOUND),
        ("meaning-name", "Meaning", NameTranslationMethod.MEANING),
        ("mixed-name", "Mixed", NameTranslationMethod.MIXED),
        ("creative-name", "Creative", NameTranslationMethod.CREATIVE),
        ("uncertain-name", "Uncertain", NameTranslationMethod.UNCERTAIN),
    )
    analysis = _analysis("a1", section.id, tuple(
        _candidate("a1", candidate_id, source, ConceptType.PERSON, source, method=method)
        for candidate_id, source, method in methods
    ) + (
        _candidate("a1", "plain", "Plain", ConceptType.PERSON, "Простой"),
    ))

    listed = {
        entry.source: entry.name_method
        for entry in chapter_who_is_who(document, (analysis,), (), section_id=section.id)
    }

    assert listed == {
        "Sound": None,
        "Meaning": None,
        "Mixed": "звучание и смысл",
        "Creative": "вольный перевод",
        "Uncertain": None,
        "Plain": None,
    }
