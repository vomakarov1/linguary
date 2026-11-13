"""Segment translations survive cleanup, saving, and historical recovery."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from processing_fakes import make_validator

from linguary.application.block_processing import TranslationCall
from linguary.application.llm_dto import (
    BlockResultResponse,
    SegmentTranslationResponse,
    TranslationResultResponse,
)
from linguary.application.ports.llm import LLMRequest, LLMResponse
from linguary.application.reader_text import ReaderTextError, ReaderTextPolicy
from linguary.application.recovery import Recovery
from linguary.application.retry_policy import RetryPolicy
from linguary.domain import (
    BlockProcessingResult,
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    GlossaryHint,
    SourceReference,
    TranslationUnit,
    TreatmentPlan,
    TreatmentPlanItem,
)
from linguary.domain.treatment import strongest_treatment

SECTION_ID = "section-1"


def _block(order: int, parts: tuple[str, ...]) -> DocumentBlock:
    offset = 0
    segments: list[BlockSegment] = []
    for index, part in enumerate(parts):
        segments.append(BlockSegment(offset, offset + len(part), index))
        offset += len(part)
    return DocumentBlock.create(
        document_id="book-1",
        section_id=SECTION_ID,
        type=BlockType.PARAGRAPH,
        order=order,
        text="".join(parts),
        source_reference=SourceReference(
            resource_path="section-1.xhtml",
            structural_path=f"/body/p[{order + 1}]",
        ),
        segments=tuple(segments),
    )


def _plan(blocks: tuple[DocumentBlock, ...], modes: tuple[tuple[BlockTreatment, ...], ...]) -> TreatmentPlan:
    items = []
    for block, treatments in zip(blocks, modes, strict=True):
        items.append(
            TreatmentPlanItem(
                block_id=block.id,
                treatment=strongest_treatment(treatments),
                segment_treatments=treatments,
                sentence_indexes=tuple(range(len(treatments))),
            )
        )
    return TreatmentPlan(section_id=SECTION_ID, items=tuple(items))


def test_prepare_results_updates_segment_translations_and_rebuilds_text(monkeypatch) -> None:
    """Each non-empty slot is cleaned; the block text is only the domain join."""
    policy = ReaderTextPolicy()
    marked = BlockProcessingResult(
        "block-marked",
        BlockTreatment.TRANSLATE,
        translated_text="Метка [b17] в переводе.",
        segment_treatments=(BlockTreatment.TRANSLATE,),
        segment_translations=("Метка [b17] в переводе.",),
    )
    with pytest.raises(ReaderTextError):
        policy.prepare_results((marked,))

    segmented = BlockProcessingResult(
        "block-1",
        BlockTreatment.TRANSLATE,
        translated_text="Альфа. Гамма.",
        glossary_hints=(GlossaryHint("Alpha", "Альфа"),),
        segment_treatments=(
            BlockTreatment.TRANSLATE,
            BlockTreatment.READ_AS_IS,
            BlockTreatment.TRANSLATE,
        ),
        segment_translations=("Альфа.", None, "Гамма."),
    )
    legacy = BlockProcessingResult(
        "block-2",
        BlockTreatment.TRANSLATE,
        translated_text="Целиком.",
    )
    prepare_calls: list[tuple] = []
    rebuild_calls: list[tuple] = []
    original_prepare = ReaderTextPolicy.prepare
    original_rebuild = BlockProcessingResult.rebuild_translation

    def spy_prepare(self, kind, body, protected_quotes=(), *, block_id=None):
        prepare_calls.append((kind, body, block_id, protected_quotes))
        return original_prepare(self, kind, body, protected_quotes, block_id=block_id)

    def spy_rebuild(self):
        slots = tuple(self.segment_translations)
        rebuild_calls.append(slots)
        if slots:
            return "СКЛЕЕНО"
        return original_rebuild(self)

    monkeypatch.setattr(ReaderTextPolicy, "prepare", spy_prepare)
    monkeypatch.setattr(BlockProcessingResult, "rebuild_translation", spy_rebuild)

    (prepared,) = policy.prepare_results((segmented,), protected_quotes=("Alpha",))

    assert prepared.segment_translations == ("Альфа.", None, "Гамма.")
    assert prepared.translated_text == "СКЛЕЕНО"
    assert prepared.translated_text == prepared.rebuild_translation()
    assert prepared.glossary_hints[0].translation == "Альфа"
    translation_calls = [call for call in prepare_calls if call[0] == "translation"]
    assert ("translation", "Альфа.", "block-1", ("Alpha",)) in prepare_calls
    assert ("translation", "Гамма.", "block-1", ("Alpha",)) in prepare_calls
    assert all(call[1] != "Альфа. Гамма." for call in translation_calls)
    assert all(call[1] is not None for call in translation_calls)
    assert ("Альфа.", None, "Гамма.") in rebuild_calls

    prepare_calls.clear()
    rebuild_calls.clear()
    (kept,) = policy.prepare_results((legacy,))
    assert kept.translated_text == "Целиком."
    assert kept.segment_translations == ()
    assert ("translation", "Целиком.", "block-2", ()) in prepare_calls
    assert rebuild_calls == []


def test_recovery_validate_historical_preserves_segments() -> None:
    """Historical validation rebuilds segment slots and accepts a saved uniform tuple."""
    uniform = _block(0, ("First sentence.", " Second sentence."))
    mixed = _block(1, ("Rare word.", " Clear sentence."))
    untouched = _block(2, ("Keep this.", " And this."))
    blocks = (uniform, mixed, untouched)
    modes = (
        (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE),
        (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
        (BlockTreatment.READ_AS_IS, BlockTreatment.READ_AS_IS),
    )
    plan = _plan(blocks, modes)
    unit = TranslationUnit.of_blocks(id="historical-coverage", blocks=blocks, plan=plan)
    assert unit.items[0].segment_treatments == ()
    assert unit.items[2].segment_treatments == ()

    stored = (
        BlockProcessingResult(
            uniform.id,
            BlockTreatment.TRANSLATE,
            translated_text="Первое. Второе.",
            segment_treatments=modes[0],
            segment_translations=("Первое.", "Второе."),
        ),
        BlockProcessingResult(
            mixed.id,
            BlockTreatment.TRANSLATE,
            translated_text="Ясное предложение.",
            glossary_hints=(GlossaryHint("Rare", "Редкое"),),
            segment_treatments=modes[1],
            segment_translations=(None, "Ясное предложение."),
        ),
        BlockProcessingResult(
            untouched.id,
            BlockTreatment.READ_AS_IS,
            segment_treatments=modes[2],
            segment_translations=(None, None),
        ),
    )
    seen: list = []
    real = make_validator()

    class _Validator:
        def validate(self, payload, **kwargs):
            seen.append((payload, kwargs["unit"]))
            return real.validate(payload, **kwargs)

    class _Results:
        def get(self, _run_id, block_id):
            return next(item for item in stored if item.block_id == block_id)

    class _Uow:
        @contextmanager
        def transaction(self):
            yield SimpleNamespace(results=_Results())

    Recovery(_Uow())._validate_historical(
        SimpleNamespace(results=_Results()),
        SimpleNamespace(
            run=SimpleNamespace(id="run-1"),
            analysis=SimpleNamespace(treatment_plan=plan, content_plan=None),
            target_language="ru",
        ),
        tuple(block.id for block in blocks),
        {block.id: block for block in blocks},
        _Validator(),
        SimpleNamespace(),
    )

    payload, checked = seen[0]
    assert checked.items[0].segment_treatments == modes[0]
    assert checked.items[1].segment_treatments == modes[1]
    assert checked.items[2].segment_treatments == modes[2]
    full, partial, empty = payload.blocks
    assert full.translated_text is None
    assert full.segments == (
        SegmentTranslationResponse(index=1, translated_text="Первое."),
        SegmentTranslationResponse(index=2, translated_text="Второе."),
    )
    assert partial.translated_text is None
    assert partial.segments == (
        SegmentTranslationResponse(index=2, translated_text="Ясное предложение."),
    )
    assert partial.glossary_hints[0].original == "Rare"
    assert empty.segments == ()
    assert empty.translated_text is None


class _MemoryResults:
    def __init__(self) -> None:
        self.by_id: dict[str, BlockProcessingResult] = {}
        self.saved: list[tuple[BlockProcessingResult, ...]] = []

    def get(self, _run_id, block_id):
        return self.by_id.get(block_id)

    def get_many(self, _run_id, block_ids):
        return {block_id: self.by_id[block_id] for block_id in block_ids if block_id in self.by_id}

    def replace_results(self, _run_id, items):
        stored = tuple(items)
        self.saved.append(stored)
        for item in stored:
            self.by_id[item.block_id] = item


class _MemoryUow:
    def __init__(self) -> None:
        self.results = _MemoryResults()
        self.work = SimpleNamespace(
            get_binding=lambda _kind, _entity_id: None,
            delete_correction_checks=lambda _run_id, block_ids=None: None,
            stale_reviews=lambda _book_id: None,
        )
        self.runs = SimpleNamespace(active_for_book=lambda _book_id: SimpleNamespace(id="run-1"))

    @contextmanager
    def transaction(self):
        yield self


async def test_block_processing_persists_segment_results() -> None:
    """A processed segment result is stored, and a later replace does not drop the slots."""
    block = _block(0, ("First sentence.", " Second sentence."))
    modes = (BlockTreatment.TRANSLATE, BlockTreatment.TRANSLATE)
    plan = _plan((block,), (modes,))
    unit = TranslationUnit.of_blocks(id="unit-1", blocks=(block,), plan=plan)
    payload = LLMResponse(
        payload=TranslationResultResponse(
            blocks=(
                BlockResultResponse(
                    source_block_id=block.id,
                    treatment=BlockTreatment.TRANSLATE,
                    translated_text=None,
                    segments=(
                        SegmentTranslationResponse(1, "Первое."),
                        SegmentTranslationResponse(2, "Второе."),
                    ),
                ),
            )
        ),
        provider="test",
        model="test",
    )

    class _Client:
        async def generate(self, _request):
            return payload

    uow = _MemoryUow()
    call = TranslationCall(
        unit_of_work=uow,  # type: ignore[arg-type]
        llm_client=_Client(),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        validator=make_validator(),
        sleep=None,  # type: ignore[arg-type]
    )
    context = SimpleNamespace(
        request=LLMRequest.for_translation(instructions="Переведи.", input_text=block.text),
        explained_expressions=(),
        explained_entries=(),
        content_plan=None,
        expectation=None,
        to_block_ids=lambda value: value,
        to_aliases=lambda text: text,
        block_ids_by_alias={},
    )
    produced = await call.request(
        context,  # type: ignore[arg-type]
        unit=unit,
        target_language="ru",
        note_ids=lambda: "ln-ch01-001",
    )
    (result,) = produced.results
    assert result.segment_treatments == modes
    assert result.segment_translations == ("Первое.", "Второе.")
    assert result.translated_text == result.rebuild_translation()

    assert call.save(book_id="book-1", run_id="run-1", results=(result,), reconcile=False)
    assert uow.results.by_id[block.id].segment_treatments == modes
    assert uow.results.by_id[block.id].segment_translations == ("Первое.", "Второе.")

    omitted = BlockProcessingResult(
        block_id=result.block_id,
        treatment=result.treatment,
        translated_text=result.translated_text,
        glossary_hints=result.glossary_hints,
        language_notes=result.language_notes,
        warnings=result.warnings,
        content_evidence=result.content_evidence,
        fragment_assistances=result.fragment_assistances,
        pronunciation_assistances=result.pronunciation_assistances,
    )
    assert omitted.segment_treatments == ()
    assert omitted.segment_translations == ()
    assert call.save(book_id="book-1", run_id="run-1", results=(omitted,), reconcile=False)
    kept = uow.results.by_id[block.id]
    assert kept.segment_treatments == modes
    assert kept.segment_translations == ("Первое.", "Второе.")
    assert kept.translated_text == "Первое. Второе."
