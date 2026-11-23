"""Segment translations on a processing response, and retry expectations that keep them."""

from __future__ import annotations

import copy
import json

import pytest
from llm_payload_factories import as_content, block_payload
from pydantic import ValidationError

from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.payloads import _block_result, parse_payload, schema_for
from linguary.adapters.llm.schemas import BlockResultSchema, SegmentTranslationSchema
from linguary.application.llm_dto import (
    BlockResultResponse,
    GlossaryHintResponse,
    LLMPayloadError,
    SegmentTranslationResponse,
)
from linguary.application.ports.llm import LLMResponseFormatError, LLMResponseKind
from linguary.application.retry_policy import ExpectedBlock, ResultExpectation
from linguary.application.validators.treatment import content_violation, treatment_violation
from linguary.domain import (
    BlockSegment,
    BlockTreatment,
    BlockType,
    DocumentBlock,
    SourceReference,
    TranslationUnit,
    TreatmentPlan,
    TreatmentPlanItem,
)


def _segment_block(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_block_id": "block-1",
        "treatment": "TRANSLATE",
        "translated_text": None,
        "glossary_hints": [],
        "segments": [
            {"index": 1, "translated_text": "Первая часть."},
            {"index": 2, "translated_text": "Вторая часть."},
        ],
    }
    payload.update(overrides)
    return payload


def test_block_result_accepts_segment_translations() -> None:
    """A block may carry one translation per segment, indexed from 1."""
    segments = (
        SegmentTranslationResponse(1, "Первая часть."),
        SegmentTranslationResponse(2, "Вторая часть."),
    )
    result = BlockResultResponse(
        "block-1",
        BlockTreatment.TRANSLATE,
        segments=segments,
    )
    assert result.translated_text is None
    assert result.segments == segments
    assert result.glossary_hints == ()

    one = BlockResultResponse(
        "block-1",
        BlockTreatment.GLOSSARY_ONLY,
        segments=(SegmentTranslationResponse(1, "Один сегмент."),),
    )
    assert len(one.segments) == 1
    assert one.glossary_hints == ()

    gap = BlockResultResponse(
        "block-1",
        BlockTreatment.READ_AS_IS,
        segments=(
            SegmentTranslationResponse(1, "А"),
            SegmentTranslationResponse(1, "Б"),
            SegmentTranslationResponse(3, "В"),
        ),
    )
    assert [item.index for item in gap.segments] == [1, 1, 3]

    for index in (True, False, 0, -1, 1.0, "1"):
        with pytest.raises(LLMPayloadError, match="индекс сегмента начинается с 1"):
            SegmentTranslationResponse(index, "Да")  # type: ignore[arg-type]
    with pytest.raises(LLMPayloadError, match="перевод сегмента 1 пуст"):
        SegmentTranslationResponse(1, "  ")
    with pytest.raises(LLMPayloadError, match="перевод сегмента 1 пуст"):
        SegmentTranslationResponse(1, "")

    parsed = BlockResultSchema.model_validate(_segment_block())
    assert parsed.translated_text is None
    assert [(item.index, item.translated_text) for item in parsed.segments] == [
        (1, "Первая часть."),
        (2, "Вторая часть."),
    ]
    single = BlockResultSchema.model_validate(
        _segment_block(
            treatment="GLOSSARY_ONLY",
            segments=[{"index": 1, "translated_text": "Один сегмент."}],
        )
    )
    assert len(single.segments) == 1
    from_null = BlockResultSchema.model_validate_json(
        json.dumps(_segment_block(), ensure_ascii=False)
    )
    assert from_null.translated_text is None

    index_field = SegmentTranslationSchema.model_fields["index"]
    assert any(getattr(item, "strict", None) is True for item in index_field.metadata)
    assert SegmentTranslationSchema.model_json_schema()["properties"]["index"]["minimum"] == 1
    for index in (0, True, 1.0, "1"):
        with pytest.raises(ValidationError):
            SegmentTranslationSchema.model_validate(
                {"index": index, "translated_text": "Да"}
            )
    with pytest.raises(ValidationError):
        SegmentTranslationSchema.model_validate({"index": 1, "translated_text": "  "})

    wire = strict_json_schema(schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=4))
    block = wire["$defs"]["BlockResultSchema"]
    assert "segments" in block["properties"]
    assert "segments" in block["required"]
    segment = wire["$defs"]["SegmentTranslationSchema"]
    assert set(segment["required"]) == {"index", "translated_text"}
    assert "minimum" not in repr(segment)

    dto = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT,
        as_content(
            {
                "blocks": [_segment_block()],
                "language_notes": [],
                "fragment_assistances": [],
            }
        ),
        contract_version=4,
    )
    assert dto.blocks[0].segments == segments
    assert dto.blocks[0].translated_text is None
    assert dto.blocks[0].treatment is BlockTreatment.TRANSLATE


def test_segments_with_translated_text_rejected() -> None:
    """Segment translations replace the block translation; both together are invalid."""
    segments = (SegmentTranslationResponse(1, "Часть."),)
    with pytest.raises(LLMPayloadError, match="переводами сегментов"):
        BlockResultResponse(
            "block-1",
            BlockTreatment.TRANSLATE,
            translated_text="Лишний перевод.",
            segments=segments,
        )
    with pytest.raises(LLMPayloadError, match="переводами сегментов"):
        BlockResultResponse(
            "block-1",
            BlockTreatment.READ_AS_IS,
            translated_text="   ",
            segments=segments,
        )

    with pytest.raises(ValidationError, match="переводами сегментов"):
        BlockResultSchema.model_validate(
            _segment_block(translated_text="Лишний перевод.")
        )
    with pytest.raises(ValidationError, match="переводами сегментов"):
        BlockResultSchema.model_validate(_segment_block(translated_text="   "))


def test_single_segment_block_keeps_translated_text() -> None:
    """No segment list keeps the block translation and today's composition rules."""
    kept = BlockResultResponse(
        "block-1",
        BlockTreatment.TRANSLATE,
        "Общепризнанная истина.",
    )
    assert kept.translated_text == "Общепризнанная истина."
    assert kept.segments == ()

    omitted = BlockResultSchema.model_validate(block_payload())
    empty = BlockResultSchema.model_validate(block_payload(segments=[]))
    assert omitted.segments == []
    assert empty.segments == []
    assert omitted.translated_text == "Общепризнанная истина гласит…"
    assert empty.translated_text == omitted.translated_text

    with pytest.raises(LLMPayloadError, match="обязателен полный перевод"):
        BlockResultResponse("block-1", BlockTreatment.TRANSLATE, segments=())
    with pytest.raises(LLMPayloadError, match="полный перевод недопустим"):
        BlockResultResponse(
            "block-1",
            BlockTreatment.READ_AS_IS,
            translated_text="Нельзя",
        )
    with pytest.raises(LLMPayloadError, match="обязательны словарные подсказки"):
        BlockResultResponse("block-1", BlockTreatment.GLOSSARY_ONLY)
    with pytest.raises(LLMPayloadError, match="словарные подсказки недопустимы"):
        BlockResultResponse(
            "block-1",
            BlockTreatment.READ_AS_IS,
            glossary_hints=(GlossaryHintResponse("sir", "сэр"),),
        )

    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(block_payload(translated_text=None, segments=[]))
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(
            block_payload(
                treatment="READ_AS_IS",
                translated_text="Нельзя",
                glossary_hints=[],
                segments=[],
            )
        )
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(
            block_payload(treatment="GLOSSARY_ONLY", translated_text=None, glossary_hints=[])
        )


def test_expected_block_preserves_segment_treatments() -> None:
    """A retry block keeps its segment treatments; empty means there is no per-segment requirement."""
    treatments = (BlockTreatment.READ_AS_IS, BlockTreatment.TRANSLATE)
    block = ExpectedBlock("block-1", BlockTreatment.TRANSLATE, False, treatments)
    assert block.block_id == "block-1"
    assert block.treatment is BlockTreatment.TRANSLATE
    assert block.context_only is False
    assert block.segment_treatments == treatments

    assert ExpectedBlock("block-1").segment_treatments == ()
    assert ExpectedBlock("block-1", BlockTreatment.GLOSSARY_ONLY).segment_treatments == ()
    contextual = ExpectedBlock("block-1", None, True)
    assert contextual.context_only is True
    assert contextual.segment_treatments == ()

    expectation = ResultExpectation(target_language="ru", blocks=(block,))
    assert expectation.blocks[0].segment_treatments == treatments


def test_mixed_block_strips_valid_empty_segments_and_passes_treatment_validation() -> None:
    """Valid empty/None segments on non-TRANSLATE spans are stripped before treatment validation."""
    text = "Rare word. Clear sentence. Plain end."
    c1 = text.index("Clear")
    c2 = text.index("Plain")
    block = DocumentBlock(
        id="block-1",
        section_id="sec-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference("ch1.xhtml", "/html/body/p[1]"),
        segments=(
            BlockSegment(0, c1, 0),
            BlockSegment(c1, c2, 1),
            BlockSegment(c2, len(text), 2),
        ),
    )
    plan = TreatmentPlan(
        "sec-1",
        (
            TreatmentPlanItem(
                block.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS),
                (0, 1, 2),
            ),
        ),
    )
    unit = TranslationUnit.of_blocks(id="unit-mixed", blocks=(block,), plan=plan)

    raw_block = _segment_block(
        translated_text="",
        glossary_hints=[{"original": "Rare word", "translation": "редкое слово", "note": None}],
        segments=[
            {"index": 1, "translated_text": ""},
            {"index": 2, "translated_text": "Перевод"},
            {"index": 3, "translated_text": None},
        ],
    )
    raw_snapshot = copy.deepcopy(raw_block)

    parsed = BlockResultSchema.model_validate(raw_block)
    assert raw_block == raw_snapshot
    assert parsed.translated_text is None
    assert [(item.index, item.translated_text) for item in parsed.segments] == [(2, "Перевод")]

    dto = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT,
        as_content({"blocks": [raw_block], "language_notes": []}),
        contract_version=3,
    )
    assert dto.blocks[0].translated_text is None
    assert dto.blocks[0].segments == (SegmentTranslationResponse(2, "Перевод"),)
    assert treatment_violation(dto, unit=unit) is None
    assert content_violation(dto, unit=unit) is None

    # Non-translatable blocks (GLOSSARY_ONLY / READ_AS_IS) with only null/empty segments pass too.
    glossary_only_raw = _segment_block(
        treatment="GLOSSARY_ONLY",
        translated_text="",
        glossary_hints=[{"original": "Rare word", "translation": "редкое слово", "note": None}],
        segments=[
            {"index": 1, "translated_text": None},
            {"index": 2, "translated_text": ""},
        ],
    )
    glossary_only_parsed = BlockResultSchema.model_validate(glossary_only_raw)
    assert glossary_only_parsed.translated_text is None
    assert glossary_only_parsed.segments == []

    read_as_is_raw = _segment_block(
        treatment="READ_AS_IS",
        translated_text="",
        glossary_hints=[],
        segments=[
            {"index": 1, "translated_text": None},
            {"index": 2, "translated_text": "   "},
        ],
    )
    read_as_is_parsed = BlockResultSchema.model_validate(read_as_is_raw)
    assert read_as_is_parsed.translated_text is None
    assert read_as_is_parsed.segments == []

    # SegmentTranslationSchema itself allows None (and strict JSON schema keeps it required).
    null_seg = SegmentTranslationSchema.model_validate({"index": 1, "translated_text": None})
    assert null_seg.translated_text is None
    from_schema_instance = BlockResultSchema.model_validate(
        _segment_block(
            segments=[null_seg, {"index": 2, "translated_text": "Перевод"}],
        )
    )
    assert [(item.index, item.translated_text) for item in from_schema_instance.segments] == [
        (2, "Перевод"),
    ]
    wire = strict_json_schema(schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=4))
    seg_schema = wire["$defs"]["SegmentTranslationSchema"]
    assert set(seg_schema["required"]) == {"index", "translated_text"}
    assert seg_schema["properties"]["translated_text"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    with pytest.raises(ValidationError):
        SegmentTranslationSchema.model_validate({"index": 1, "translated_text": ""})


def test_translate_block_with_only_empty_segments_raises_validation_error() -> None:
    """TRANSLATE block whose segments are all empty/None fails at schema level, not LLMPayloadError."""
    for empty_value in (None, "", "   "):
        with pytest.raises(ValidationError, match="обязателен полный перевод"):
            BlockResultSchema.model_validate(
                _segment_block(
                    treatment="TRANSLATE",
                    translated_text=None,
                    segments=[{"index": 1, "translated_text": empty_value}],
                )
            )

    with pytest.raises(LLMResponseFormatError) as exc_info:
        parse_payload(
            LLMResponseKind.TRANSLATION_RESULT,
            as_content(
                {
                    "blocks": [
                        _segment_block(
                            treatment="TRANSLATE",
                            translated_text=None,
                            segments=[{"index": 1, "translated_text": None}],
                        )
                    ],
                    "language_notes": [],
                }
            ),
            contract_version=3,
        )
    assert exc_info.value.violation is not None
    assert exc_info.value.violation.rule_code == "response_schema"

    # Defensive filter in _block_result guards against pre-constructed empty segments.
    constructed = BlockResultSchema.model_construct(
        source_block_id="block-1",
        treatment="TRANSLATE",
        translated_text=None,
        glossary_hints=[],
        segments=[
            SegmentTranslationSchema.model_construct(index=1, translated_text=None),
            SegmentTranslationSchema.model_construct(index=2, translated_text="Второй."),
        ],
    )
    dto = _block_result(constructed)
    assert dto.segments == (SegmentTranslationResponse(2, "Второй."),)


@pytest.mark.parametrize(
    "bad_segment",
    [
        pytest.param({"index": 1, "translated_text": 123}, id="non-string-text"),
        pytest.param({"index": 0, "translated_text": None}, id="zero-index"),
        pytest.param({"index": True, "translated_text": None}, id="bool-index"),
        pytest.param({"index": 1}, id="missing-translated-text"),
        pytest.param({"index": 1, "translated_text": None, "extra": 1}, id="extra-field"),
    ],
)
def test_malformed_segment_dicts_raise_validation_error(bad_segment: dict[str, object]) -> None:
    """Malformed segment dicts are never silently dropped by before-validation cleanup."""
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(_segment_block(segments=[bad_segment]))
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(
            _segment_block(
                segments=[{"index": 2, "translated_text": "Перевод"}, bad_segment],
            )
        )


def test_treatment_validator_clarifies_forbidden_segment_removal() -> None:
    """Returning a non-empty translation for a non-TRANSLATE segment produces actionable hint."""
    text = "Rare word. Clear sentence."
    cut = text.index("Clear")
    block = DocumentBlock(
        id="block-1",
        section_id="sec-1",
        type=BlockType.PARAGRAPH,
        order=0,
        text=text,
        source_reference=SourceReference("ch1.xhtml", "/html/body/p[1]"),
        segments=(BlockSegment(0, cut, 0), BlockSegment(cut, len(text), 1)),
    )
    plan = TreatmentPlan(
        "sec-1",
        (
            TreatmentPlanItem(
                block.id,
                BlockTreatment.TRANSLATE,
                (BlockTreatment.GLOSSARY_ONLY, BlockTreatment.TRANSLATE),
                (0, 1),
            ),
        ),
    )
    unit = TranslationUnit.of_blocks(id="unit-err", blocks=(block,), plan=plan)
    dto = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT,
        as_content(
            {
                "blocks": [
                    _segment_block(
                        glossary_hints=[{"original": "Rare word", "translation": "редкое слово", "note": None}],
                        segments=[
                            {"index": 1, "translated_text": "Редкое слово."},
                            {"index": 2, "translated_text": "Ясное предложение."},
                        ],
                    )
                ],
                "language_notes": [],
            }
        ),
        contract_version=3,
    )
    error = content_violation(dto, unit=unit)
    assert error == (
        "для сегмента 1 блока block-1 с режимом GLOSSARY_ONLY перевод недопустим: "
        "удали элемент с index=1 из segments (или передай null)"
    )
