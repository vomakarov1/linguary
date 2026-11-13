"""Segment translations on a processing response, and retry expectations that keep them."""

from __future__ import annotations

import json

import pytest
from llm_payload_factories import as_content, block_payload
from pydantic import ValidationError

from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.payloads import parse_payload, schema_for
from linguary.adapters.llm.schemas import BlockResultSchema, SegmentTranslationSchema
from linguary.application.llm_dto import (
    BlockResultResponse,
    GlossaryHintResponse,
    LLMPayloadError,
    SegmentTranslationResponse,
)
from linguary.application.ports.llm import LLMResponseKind
from linguary.application.retry_policy import ExpectedBlock, ResultExpectation
from linguary.domain import BlockTreatment


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
