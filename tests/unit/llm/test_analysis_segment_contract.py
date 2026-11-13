"""Segment treatments, candidate descriptions, and learner difficulty kinds."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.payloads import parse_payload
from linguary.adapters.llm.schemas import (
    DifficultySchema,
    GlossaryCandidateSchema,
    NarrativeAnalysisSchema,
    PedagogicalAnalysisSchema,
    TreatmentPlanItemSchema,
)
from linguary.application.analysis_pass_cache import decode_pass, encode_pass
from linguary.application.llm_dto import (
    PedagogicalAnalysisResponse,
    SegmentTreatmentResponse,
    TreatmentPlanItemResponse,
)
from linguary.application.ports.llm import LLMResponseKind
from linguary.domain import BlockTreatment, ConceptType


def _candidate(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_text": "actual",
        "type": "term",
        "suggested_translation": "фактический",
        "confidence": 0.8,
        "model_flagged_review": False,
        "note": None,
    }
    payload.update(overrides)
    return payload


def _difficulty(kind: str) -> dict[str, object]:
    return {
        "id": "d1",
        "anchors": [{"block_id": "b1", "start": 0, "end": 4, "quote": "echo"}],
        "owner_block_id": "b1",
        "kind": kind,
        "priority": "useful",
        "help_kind": "explanation",
        "required_points": [{"id": "p1", "description": "Пояснить трудность"}],
    }


def _pedagogical(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "treatment_plan": [{"source_block_id": "b1", "treatment": "TRANSLATE"}],
        "before_you_read": [],
        "difficulties": [],
        "before_you_read_evidence": [],
    }
    payload.update(overrides)
    return payload


def test_plan_item_schema_accepts_segment_treatments() -> None:
    """Segment treatments start at index 1 and are required in the strict schema."""
    raw = {
        "source_block_id": "b1",
        "treatment": "GLOSSARY_ONLY",
        "segments": [
            {"index": 1, "treatment": "READ_AS_IS"},
            {"index": 2, "treatment": "TRANSLATE"},
        ],
    }
    item = TreatmentPlanItemSchema.model_validate(raw)
    assert [(segment.index, segment.treatment) for segment in item.segments] == [
        (1, "READ_AS_IS"),
        (2, "TRANSLATE"),
    ]

    wire = strict_json_schema(PedagogicalAnalysisSchema)
    plan = wire["$defs"]["TreatmentPlanItemSchema"]
    assert "segments" in plan["required"]
    segment = wire["$defs"]["SegmentTreatmentSchema"]
    assert set(segment["required"]) == {"index", "treatment"}

    dto = parse_payload(
        LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL,
        json.dumps(_pedagogical(treatment_plan=[raw]), ensure_ascii=False),
    )
    assert dto.treatment_plan[0].treatment is BlockTreatment.GLOSSARY_ONLY
    assert dto.treatment_plan[0].segments == (
        SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
        SegmentTreatmentResponse(2, BlockTreatment.TRANSLATE),
    )

    with pytest.raises(ValidationError):
        TreatmentPlanItemSchema.model_validate(
            {**raw, "segments": [{"index": 0, "treatment": "READ_AS_IS"}]}
        )


def test_plan_item_without_segments_means_whole_block() -> None:
    """An empty segment list is required in the strict schema and means the whole block."""
    omitted = TreatmentPlanItemSchema.model_validate(
        {"source_block_id": "b1", "treatment": "READ_AS_IS"}
    )
    empty = TreatmentPlanItemSchema.model_validate(
        {"source_block_id": "b1", "treatment": "READ_AS_IS", "segments": []}
    )
    assert omitted.segments == []
    assert empty.segments == []

    wire = strict_json_schema(PedagogicalAnalysisSchema)
    plan = wire["$defs"]["TreatmentPlanItemSchema"]
    assert "segments" in plan["properties"]
    assert "segments" in plan["required"]

    for plan_item in (
        {"source_block_id": "b1", "treatment": "READ_AS_IS"},
        {"source_block_id": "b1", "treatment": "READ_AS_IS", "segments": []},
    ):
        dto = parse_payload(
            LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL,
            json.dumps(_pedagogical(treatment_plan=[plan_item]), ensure_ascii=False),
        )
        assert dto.treatment_plan[0].segments == ()
        assert dto.treatment_plan[0].treatment is BlockTreatment.READ_AS_IS


def test_candidate_description_parsed() -> None:
    """A candidate description defaults to an empty string and is parsed when present."""
    assert GlossaryCandidateSchema.model_validate(_candidate()).description == ""
    parsed = GlossaryCandidateSchema.model_validate(
        _candidate(description="ложный друг: actual — не «актуальный»")
    )
    assert parsed.description == "ложный друг: actual — не «актуальный»"

    wire = strict_json_schema(NarrativeAnalysisSchema)
    candidate = wire["$defs"]["GlossaryCandidateSchema"]
    assert "description" in candidate["properties"]
    assert "description" in candidate["required"]

    body = {
        "translation_instructions": "Сохраняй смысл.",
        "chapter_summary": "A note about a false friend.",
        "glossary_candidates": [
            _candidate(description="ложный друг: actual — не «актуальный»")
        ],
        "context_facts": [],
        "scenes": [],
    }
    dto = parse_payload(
        LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE,
        json.dumps(body, ensure_ascii=False),
    )
    assert dto.glossary_candidates[0].description == "ложный друг: actual — не «актуальный»"
    assert dto.glossary_candidates[0].type is ConceptType.TERM

    body["glossary_candidates"] = [_candidate()]
    omitted = parse_payload(
        LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE,
        json.dumps(body, ensure_ascii=False),
    )
    assert omitted.glossary_candidates[0].description == ""


def test_difficulty_schema_accepts_new_kinds() -> None:
    """The difficulty schema accepts false_friend, speech_style, and parody."""
    wire = strict_json_schema(DifficultySchema)
    assert set(wire["properties"]["kind"]["enum"]) >= {
        "false_friend",
        "speech_style",
        "parody",
        "wordplay",
        "vocabulary",
    }
    for kind in ("false_friend", "speech_style", "parody"):
        assert DifficultySchema.model_validate(_difficulty(kind)).kind == kind
        dto = parse_payload(
            LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL,
            json.dumps(
                _pedagogical(
                    difficulties=[
                        {
                            "id": "d1",
                            "anchors": [{"block_id": "b1", "quote": "echo"}],
                            "owner_block_id": "b1",
                            "kind": kind,
                            "priority": "critical",
                            "help_kind": "lexical_hint",
                            "required_points": [{"id": "p1", "description": "Пояснить"}],
                        }
                    ]
                ),
                ensure_ascii=False,
            ),
        )
        assert dto.difficulties[0].kind == kind
    with pytest.raises(ValidationError):
        DifficultySchema.model_validate(_difficulty("pun"))


def test_analysis_pass_cache_preserves_segments() -> None:
    """The pass cache returns the same segments, including after a JSON round trip."""
    item = TreatmentPlanItemResponse(
        "b1",
        BlockTreatment.TRANSLATE,
        (
            SegmentTreatmentResponse(1, BlockTreatment.READ_AS_IS),
            SegmentTreatmentResponse(2, BlockTreatment.TRANSLATE),
        ),
    )
    payload = PedagogicalAnalysisResponse(treatment_plan=(item,))
    restored = decode_pass(encode_pass(payload), narrative=False)
    assert restored.treatment_plan[0].segments == item.segments
    assert restored.treatment_plan[0].treatment is BlockTreatment.TRANSLATE

    raw = json.loads(json.dumps(encode_pass(payload)))
    from_json = decode_pass(raw, narrative=False)
    assert from_json.treatment_plan[0].segments == item.segments

    for plan_item in raw["response"]["treatment_plan"]:
        plan_item.pop("segments")
    legacy = decode_pass(raw, narrative=False)
    assert legacy.treatment_plan[0].segments == ()

    whole_block = PedagogicalAnalysisResponse(
        treatment_plan=(TreatmentPlanItemResponse("b1", BlockTreatment.READ_AS_IS),)
    )
    encoded = encode_pass(whole_block)
    assert "segments" not in encoded["response"]["treatment_plan"][0]
    assert decode_pass(encoded, narrative=False).treatment_plan[0].segments == ()
