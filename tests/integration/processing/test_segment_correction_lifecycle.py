"""Сегментные исправления и помощь внутри смешанного блока (P6.T3)."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from processing_scenario import (
    BOOK_ID,
    RUN_ID,
    make_process_chapter,
    make_validator,
    store_processing_book,
    translation_reply,
)

from linguary.adapters.db.unit_of_work import SqlAlchemyUnitOfWork
from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.llm.correction_payloads import ChapterCorrectionSchema
from linguary.application.correction import CorrectionInputs
from linguary.application.correction_dto import (
    CORRECTION_VERSIONS,
    ChapterCorrectionResponse,
    CorrectedBlockResponse,
    CorrectionSnapshot,
    UnchangedBlockResponse,
)
from linguary.application.llm_dto import (
    FragmentAssistanceResponse,
    SegmentTranslationResponse,
    TranslationResultResponse,
)
from linguary.application.retry_policy import ResultValidationError
from linguary.application.use_cases.correct_chapter import CorrectChapter
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.application.validators.content_plan import resolve_fragment_assistances
from linguary.composition.correction_runtime import CORRECTION_PROMPT, CORRECTION_PROMPT_V2
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    ChapterAnalysis,
    DocumentBlock,
    DocumentSection,
    GlossaryHint,
    GlossarySnapshot,
    LanguageNote,
    SourceReference,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.content_plan import (
    ContentPlan,
    Difficulty,
    RequiredPoint,
    SourceAnchor,
)
from linguary.domain.learning import FragmentAssistance

_FIRST = "Alpha went home at dawn. "
_SECOND = "Beta stayed quiet."
_SOURCE = _FIRST + _SECOND
_ALPHA = "Альфа ушла домой на рассвете."
_BETA = "Бета молчала."
_GAMMA = "Бета осталась тихой."


def _snapshot() -> CorrectionSnapshot:
    return CorrectionSnapshot(
        model="gpt-test",
        route={
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "proxy": "",
            "route_id": "r1",
        },
        parameters={
            "temperature": 0.0,
            "max_output_tokens": 1000,
            "timeout_seconds": 30,
            "max_attempts": 1,
            "max_unit_source_chars": 8000,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
        prompt="Correct.",
        versions=CORRECTION_VERSIONS,
    )


def _paragraph(order: int, text: str, segments: tuple[BlockSegment, ...] = ()) -> DocumentBlock:
    return DocumentBlock.create(
        document_id=BOOK_ID,
        section_id="section-1",
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
        source_reference=SourceReference(
            resource_path="section-1.xhtml",
            structural_path=f"/body/p[{order + 1}]",
        ),
        segments=segments,
    )


def _spans() -> tuple[BlockSegment, BlockSegment]:
    return (BlockSegment(0, len(_FIRST), 0), BlockSegment(len(_FIRST), len(_SOURCE), 1))


def _section(*blocks: DocumentBlock) -> DocumentSection:
    return DocumentSection(
        id="section-1",
        document_id=BOOK_ID,
        title="Chapter 1",
        order=0,
        source_resource_path="section-1.xhtml",
        blocks=blocks,
    )


def _analysis(section: DocumentSection, items: tuple[TreatmentPlanItem, ...]) -> ChapterAnalysis:
    return ChapterAnalysis.create(
        id="analysis-section-1",
        section_id=section.id,
        treatment_plan=TreatmentPlan(section.id, items),
        translation_instructions="Сохранять смысл сегментов.",
        chapter_summary="Две короткие фразы.",
        expected_block_ids=tuple(block.id for block in section.ordered_blocks),
    )


def _document(section: DocumentSection):
    from linguary.domain import BookMetadata, CanonicalDocument

    return CanonicalDocument(
        id=BOOK_ID,
        metadata=BookMetadata(title="Pride and Prejudice", authors=("Jane Austen",)),
        source_language="en",
        target_language="ru",
        sections=(section,),
    )


def _inputs(section: DocumentSection, analysis: ChapterAnalysis, results: dict, *, english_help: bool = False):
    document = _document(section)
    return CorrectionInputs(
        document=document,
        section=section,
        analysis=analysis,
        run=SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot()),
        conditions=SimpleNamespace(
            id="cond-1",
            payload={
                "schema_version": 1,
                "english_help": english_help,
                "versions": {"validators": 3},
            },
        ),
        snapshot=_snapshot(),
        blocks=section.blocks,
        results=results,
        bindings={block.id: None for block in section.blocks},
    )


def _handler(inputs: CorrectionInputs) -> CorrectChapter:
    return CorrectChapter(
        unit_of_work=None,
        llm_client=None,
        snapshot=inputs.snapshot,
        validator=make_validator(),
    )


def _result(block: DocumentBlock, slots: tuple[str | None, ...], treatments: tuple[BlockTreatment, ...], **kwargs):
    parts = [text for text in slots if text]
    return BlockProcessingResult(
        block_id=block.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=" ".join(parts) if parts else None,
        segment_treatments=treatments,
        segment_translations=slots,
        **kwargs,
    )


def test_correction_of_segmented_block_returns_segments() -> None:
    """Многосегментный блок уходит и возвращается как segments, целый блок — как прежде."""
    whole = _paragraph(0, "A whole sentence stays together.")
    mixed = _paragraph(1, _SOURCE, _spans())
    section = _section(whole, mixed)
    whole_result = BlockProcessingResult(
        block_id=whole.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Целое предложение остаётся вместе.",
    )
    mixed_result = _result(
        mixed,
        (_ALPHA, _BETA),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    analysis = _analysis(
        section,
        (
            TreatmentPlanItem(whole.id, BlockTreatment.TRANSLATE),
            TreatmentPlanItem(
                mixed.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
                sentence_indexes=(0, 1),
            ),
        ),
    )
    inputs = _inputs(
        section,
        analysis,
        {whole.id: whole_result, mixed.id: mixed_result},
    )
    request = _handler(inputs)._request(
        inputs,
        SimpleNamespace(book_id=BOOK_ID, id="task-1", scope_id=section.id),
        (whole.id, mixed.id),
    )
    sent = {
        item["block_id"]: item["current_result"]
        for item in json.loads(request.input_text)["blocks"]
    }
    whole_sent = sent[inputs.block_aliases[whole.id]]
    mixed_sent = sent[inputs.block_aliases[mixed.id]]
    assert "segments" not in whole_sent
    assert whole_sent["translated_text"] == "Целое предложение остаётся вместе."
    assert mixed_sent["translated_text"] is None
    assert mixed_sent["segments"] == [
        {"index": 1, "translated_text": _ALPHA},
        {"index": 2, "translated_text": _BETA},
    ]

    parsed = ChapterCorrectionSchema.model_validate(
        {
            "blocks": [
                {
                    "block_id": whole.id,
                    "status": "corrected",
                    "translated_text": "Новое целое предложение.",
                    "glossary_hints": [],
                    "language_notes": [],
                },
                {
                    "block_id": mixed.id,
                    "status": "corrected",
                    "translated_text": None,
                    "glossary_hints": [],
                    "language_notes": [],
                    "segments": [
                        {"index": 1, "translated_text": "Новая альфа."},
                        {"index": 2, "translated_text": "Новая бета."},
                    ],
                },
            ]
        }
    ).to_dto()
    corrected_whole, corrected_mixed = parsed.blocks
    assert corrected_whole.translated_text == "Новое целое предложение."
    assert corrected_whole.segments == ()
    assert corrected_mixed.translated_text is None
    assert corrected_mixed.segments == (
        SegmentTranslationResponse(1, "Новая альфа."),
        SegmentTranslationResponse(2, "Новая бета."),
    )
    assert "segments" in CORRECTION_PROMPT
    assert "segments" in CORRECTION_PROMPT_V2


def test_correction_keeps_untouched_segments() -> None:
    """Возвращённый индекс заменяет слот, пропущенный остаётся, склейка берётся из rebuild."""
    block = _paragraph(0, _SOURCE, _spans())
    section = _section(block)
    previous = _result(
        block,
        (_ALPHA, _BETA),
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
    )
    analysis = _analysis(
        section,
        (
            TreatmentPlanItem(
                block.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
                sentence_indexes=(0, 1),
            ),
        ),
    )
    inputs = _inputs(section, analysis, {block.id: previous})
    saved, = _handler(inputs)._candidates(
        None,
        inputs,
        ChapterCorrectionResponse(
            blocks=(
                CorrectedBlockResponse(
                    block.id,
                    None,
                    (),
                    (),
                    segments=(SegmentTranslationResponse(2, _GAMMA),),
                ),
            )
        ),
    )
    assert saved.segment_translations == (_ALPHA, _GAMMA)
    assert saved.translated_text == saved.rebuild_translation()
    assert saved.translated_text == f"{_ALPHA} {_GAMMA}"


def _assistance(block_id: str, start: int, end: int, quote: str) -> FragmentAssistance:
    anchor = SourceAnchor(block_id, start, end, quote)
    return FragmentAssistance(
        "d-span",
        ("p1",),
        "sentence_rephrase",
        "Короткое пояснение смысла.",
        (anchor,),
    )


def _mixed_block(second: BlockTreatment, **result_kwargs) -> tuple[DocumentBlock, BlockProcessingResult]:
    block = _paragraph(0, _SOURCE, _spans())
    hints = result_kwargs.pop("glossary_hints", ())
    if second is BlockTreatment.GLOSSARY_ONLY and not hints:
        hints = (GlossaryHint("lane", "переулок"),)
    result = _result(
        block,
        (_ALPHA, None),
        (BlockTreatment.TRANSLATE, second),
        glossary_hints=hints,
        **result_kwargs,
    )
    return block, result


def _candidates_for(second: BlockTreatment, start: int, end: int, quote: str, **result_kwargs):
    block, result = _mixed_block(second, **result_kwargs)
    result = replace(result, fragment_assistances=(_assistance(block.id, start, end, quote),))
    section = _section(block)
    analysis = _analysis(
        section,
        (
            TreatmentPlanItem(
                block.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.TRANSLATE, second),
                sentence_indexes=(0, 1),
            ),
        ),
    )
    inputs = _inputs(section, analysis, {block.id: result}, english_help=True)
    return _handler(inputs)._candidates(
        None,
        inputs,
        ChapterCorrectionResponse(blocks=(UnchangedBlockResponse(block.id),)),
    )


def test_assistance_allowed_in_untranslated_segment_of_mixed_block() -> None:
    """Помощь в непереведённом сегменте разрешена, внутри TRANSLATE — нет."""
    kept = _candidates_for(BlockTreatment.READ_AS_IS, len(_FIRST), len(_SOURCE), _SECOND)
    assert kept[0].fragment_assistances[0].anchors[0].quote == _SECOND

    with pytest.raises(ResultValidationError, match="дублирует защищённую помощь к фрагменту"):
        _candidates_for(BlockTreatment.READ_AS_IS, 0, len("Alpha"), "Alpha")

    glossary_kept = _candidates_for(BlockTreatment.GLOSSARY_ONLY, len(_FIRST), len(_SOURCE), _SECOND)
    assert glossary_kept[0].segment_translations == (_ALPHA, None)

    with pytest.raises(ResultValidationError, match="дублирует защищённую помощь к фрагменту"):
        _candidates_for(
            BlockTreatment.READ_AS_IS,
            len(_FIRST),
            len(_SOURCE),
            _SECOND,
            glossary_hints=(GlossaryHint(_SECOND, "Бета молчала"),),
        )

    noted_block, noted = _mixed_block(BlockTreatment.READ_AS_IS)
    noted = replace(
        noted,
        language_notes=(LanguageNote("ln-1", noted_block.id, _SECOND, "Заметка о тишине."),),
        fragment_assistances=(_assistance(noted_block.id, len(_FIRST), len(_SOURCE), _SECOND),),
    )
    noted_section = _section(noted_block)
    noted_inputs = _inputs(
        noted_section,
        _analysis(
            noted_section,
            (
                TreatmentPlanItem(
                    noted_block.id,
                    BlockTreatment.TRANSLATE,
                    (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
                    sentence_indexes=(0, 1),
                ),
            ),
        ),
        {noted_block.id: noted},
        english_help=True,
    )
    with pytest.raises(ResultValidationError, match="дублирует защищённую помощь к фрагменту"):
        _handler(noted_inputs)._candidates(
            None,
            noted_inputs,
            ChapterCorrectionResponse(blocks=(UnchangedBlockResponse(noted_block.id),)),
        )

    whole = _paragraph(0, "Second translated with help.")
    whole_section = _section(whole)
    whole_anchor = SourceAnchor(whole.id, 0, 6, "Second")
    whole_help = FragmentAssistance(
        "d-whole",
        ("p1",),
        "sentence_rephrase",
        "Пояснение к фрагменту.",
        (whole_anchor,),
    )
    whole_result = BlockProcessingResult(
        block_id=whole.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text="Второй старый перевод.",
        fragment_assistances=(whole_help,),
    )
    whole_inputs = _inputs(
        whole_section,
        _analysis(whole_section, (TreatmentPlanItem(whole.id, BlockTreatment.TRANSLATE),)),
        {whole.id: whole_result},
        english_help=True,
    )
    updated, = _handler(whole_inputs)._candidates(
        None,
        whole_inputs,
        ChapterCorrectionResponse(
            blocks=(
                CorrectedBlockResponse(
                    whole.id,
                    "Второй обновлённый перевод.",
                    (),
                    (),
                ),
            )
        ),
    )
    assert updated.translated_text == "Второй обновлённый перевод."

    def plan_for(block: DocumentBlock, start: int, end: int, quote: str) -> ContentPlan:
        return ContentPlan(
            difficulties=(
                Difficulty(
                    "d-span",
                    (SourceAnchor(block.id, start, end, quote),),
                    block.id,
                    "idiom",
                    "critical",
                    "explanation",
                    (RequiredPoint("p1", "Смысл фразы"),),
                ),
            )
        )

    def bind(block: DocumentBlock, result: BlockProcessingResult, start: int, end: int, quote: str, **kwargs):
        payload = TranslationResultResponse(
            fragment_assistances=(
                FragmentAssistanceResponse(
                    "d-span",
                    ("p1",),
                    "sentence_rephrase",
                    "Короткое пояснение смысла.",
                ),
            )
        )
        return resolve_fragment_assistances(
            plan_for(block, start, end, quote),
            payload,
            (result,),
            {block.id: block.text},
            owner_block_ids=(block.id,),
            blocks=(block,),
            **kwargs,
        )

    plain_block, plain_result = _mixed_block(BlockTreatment.READ_AS_IS)
    bound = bind(plain_block, plain_result, len(_FIRST), len(_SOURCE), _SECOND.strip())
    assert bound[plain_block.id][0].anchors[0].quote == _SECOND.strip()
    with pytest.raises(ResultValidationError, match="дублирует"):
        bind(plain_block, plain_result, 0, len("Alpha"), "Alpha")

    glossary_block, glossary_result = _mixed_block(BlockTreatment.GLOSSARY_ONLY)
    glossary_bound = bind(
        glossary_block,
        glossary_result,
        len(_FIRST),
        len(_SOURCE),
        _SECOND.strip(),
    )
    assert glossary_bound[glossary_block.id]

    with pytest.raises(ResultValidationError, match="дублирует"):
        bind(
            plain_block,
            plain_result,
            len(_FIRST),
            len(_SOURCE),
            _SECOND.strip(),
            before_you_read=(SimpleNamespace(expression=_SECOND.strip()),),
        )
    hinted_result = replace(
        plain_result,
        glossary_hints=(GlossaryHint(_SECOND.strip(), "Бета молчала"),),
    )
    with pytest.raises(ResultValidationError, match="дублирует"):
        bind(plain_block, hinted_result, len(_FIRST), len(_SOURCE), _SECOND.strip())


def _stored(unit_of_work: SqlAlchemyUnitOfWork):
    with unit_of_work.transaction() as tx:
        return tx.results.list_for_run(RUN_ID)


async def test_lifecycle_mixed_block_processing_correction_recovery(
    unit_of_work: SqlAlchemyUnitOfWork,
) -> None:
    """Смешанный блок проходит обработку, исправление сегмента и восстановление."""
    store_processing_book(unit_of_work, chapters=((_SOURCE,),))
    with unit_of_work.transaction() as tx:
        document = tx.documents.get(BOOK_ID)
        analysis = tx.analyses.get_for_section(BOOK_ID, "section-1")
    section = document.section("section-1")
    block = section.ordered_blocks[0]
    segmented = replace(block, segments=_spans())
    revised = replace(analysis, treatment_plan=TreatmentPlan(
        section.id,
        (
            TreatmentPlanItem(
                block.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
                sentence_indexes=(0, 1),
            ),
        ),
    ))
    with unit_of_work.transaction() as tx:
        tx.documents.save(replace(document, sections=(replace(section, blocks=(segmented,)),)))
        tx.analyses.save(BOOK_ID, revised)

    client = FakeLLMClient.replying([
        translation_reply([
            {
                "source_block_id": "b1",
                "treatment": "TRANSLATE",
                "translated_text": None,
                "glossary_hints": [],
                "segments": [{"index": 1, "translated_text": _ALPHA}],
            }
        ])
    ])
    process, _ = make_process_chapter(unit_of_work, client)
    await process(ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1"))
    processed, = _stored(unit_of_work)
    assert processed.segment_translations == (_ALPHA, None)

    corrected = "Альфа вернулась домой на рассвете."
    with unit_of_work.transaction() as tx:
        document = tx.documents.get(BOOK_ID)
        analysis = tx.analyses.get_for_section(BOOK_ID, "section-1")
        run = tx.runs.active_for_book(BOOK_ID)
    section = document.section("section-1")
    block = section.ordered_blocks[0]
    inputs = CorrectionInputs(
        document=document,
        section=section,
        analysis=analysis,
        run=run,
        conditions=SimpleNamespace(
            id="cond-1",
            payload={
                "schema_version": 1,
                "english_help": False,
                "versions": {"validators": 1},
            },
        ),
        snapshot=_snapshot(),
        blocks=section.blocks,
        results={block.id: processed},
        bindings={block.id: None},
    )
    saved, = _handler(inputs)._candidates(
        None,
        inputs,
        ChapterCorrectionResponse(
            blocks=(
                CorrectedBlockResponse(
                    block.id,
                    None,
                    (),
                    (),
                    segments=(SegmentTranslationResponse(1, corrected),),
                ),
            )
        ),
    )
    with unit_of_work.transaction() as tx:
        tx.results.replace_results(RUN_ID, (saved,))
    process.prepare(ProcessChapterCommand(book_id=BOOK_ID, section_id="section-1"))
    recovered, = _stored(unit_of_work)
    assert recovered.segment_translations == (corrected, None)
    assert recovered.translated_text == recovered.rebuild_translation()
    assert recovered.translated_text == corrected
