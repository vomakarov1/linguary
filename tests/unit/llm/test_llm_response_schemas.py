"""Structured model response schemas: what the adapter boundary rejects."""

from __future__ import annotations

import pytest
from llm_payload_factories import analysis_payload, as_content, block_payload, translation_payload
from pydantic import ValidationError

from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.schemas import (
    BlockResultSchema,
    ChapterAnalysisSchema,
    TranslationResultSchema,
)
from linguary.domain import BlockTreatment


def test_correct_analysis_response_passes_schema() -> None:
    """Valid chapter analysis response passes schema validation."""
    schema = ChapterAnalysisSchema.model_validate_json(as_content(analysis_payload()))

    assert [item.source_block_id for item in schema.treatment_plan] == ["block-1", "block-2"]


def test_correct_translation_response_passes_schema() -> None:
    """Valid block group processing response passes schema validation."""
    schema = TranslationResultSchema.model_validate_json(as_content(translation_payload()))

    assert [block.source_block_id for block in schema.blocks] == ["block-1", "block-2"]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("не json вовсе", id="not-json"),
        pytest.param('{"blocks": [', id="truncated-json"),
        pytest.param("[]", id="json-array-instead-of-object"),
    ],
)
def test_invalid_json_is_rejected(content: str) -> None:
    """A response that is not a JSON object is rejected by the schema."""
    with pytest.raises(ValidationError):
        TranslationResultSchema.model_validate_json(content)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            block_payload(source_block_id=""), id="empty-block-id"
        ),
        pytest.param(
            {key: value for key, value in block_payload().items() if key != "source_block_id"},
            id="missing-block-id",
        ),
    ],
)
def test_block_result_without_block_id_is_rejected(payload: dict[str, object]) -> None:
    """A result without a block identifier cannot be stored and is rejected by the schema."""
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(payload)


@pytest.mark.parametrize(
    "treatment",
    [
        pytest.param("PRESERVE_ONLY", id="non-textual-treatment"),
        pytest.param("DROP_WITH_WARNING", id="dropped-treatment"),
        pytest.param("translate", id="wrong-case"),
        pytest.param("REWRITE", id="unknown-treatment"),
    ],
)
def test_unknown_treatment_is_rejected(treatment: str) -> None:
    """The model cannot choose a treatment outside the three textual ones."""
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(block_payload(treatment=treatment))


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(block_payload(translated_text=None), id="null-translation"),
        pytest.param(block_payload(translated_text="   "), id="blank-translation"),
    ],
)
def test_translate_without_translation_is_rejected(payload: dict[str, object]) -> None:
    """For 'TRANSLATE' treatment, a full translation is required (PRD § 15)."""
    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(payload)


def test_read_as_is_with_glossary_hints_is_rejected() -> None:
    """A 'READ_AS_IS' block receives neither translation nor glossary hints (PRD § 15)."""
    payload = block_payload(
        treatment="READ_AS_IS",
        translated_text=None,
        glossary_hints=[{"original": "sir", "translation": "сэр", "note": None}],
    )

    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(payload)


def test_read_as_is_without_content_passes_schema() -> None:
    """A 'READ_AS_IS' block without translation or hints is valid."""
    schema = BlockResultSchema.model_validate(
        block_payload(treatment="READ_AS_IS", translated_text=None, glossary_hints=[])
    )

    assert schema.translated_text is None


def test_glossary_only_without_hints_is_rejected() -> None:
    """For 'GLOSSARY_ONLY' treatment, glossary hints are required (PRD § 15)."""
    payload = block_payload(
        treatment="GLOSSARY_ONLY", translated_text=None, glossary_hints=[]
    )

    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(payload)


def test_glossary_only_with_full_translation_is_rejected() -> None:
    """Full translation is allowed only for 'TRANSLATE' (PRD § 15)."""
    payload = block_payload(
        treatment="GLOSSARY_ONLY",
        translated_text="Полный перевод",
        glossary_hints=[{"original": "entailed", "translation": "заповедный", "note": None}],
    )

    with pytest.raises(ValidationError):
        BlockResultSchema.model_validate(payload)


@pytest.mark.parametrize(
    "confidence",
    [pytest.param(-0.1, id="below-range"), pytest.param(1.5, id="above-range")],
)
def test_glossary_candidate_confidence_range_is_enforced(confidence: float) -> None:
    """Glossary candidate confidence must be in the 0..1 range (DTO-02)."""
    payload = analysis_payload()
    payload["glossary_candidates"][0]["confidence"] = confidence

    with pytest.raises(ValidationError):
        ChapterAnalysisSchema.model_validate(payload)


def test_unexpected_field_is_rejected() -> None:
    """An unexpected field indicates the model did not adhere to the schema."""
    with pytest.raises(ValidationError):
        ChapterAnalysisSchema.model_validate(analysis_payload(commentary="я старалась"))


def test_schema_treatments_match_textual_domain_treatments() -> None:
    """Schema treatments match textual domain treatments.

    A mismatch would indicate the model is offered a treatment missing from the domain,
    or that a new textual treatment was not added to the schema.
    """
    domain_treatments = {
        treatment.value for treatment in BlockTreatment if treatment.is_textual
    }

    assert _treatment_values() == domain_treatments


def _treatment_values() -> set[str]:
    """Treatment values declared in the block result schema."""
    schema = BlockResultSchema.model_json_schema()
    field = schema["properties"]["treatment"]
    if "$ref" in field:
        field = schema["$defs"][field["$ref"].rsplit("/", maxsplit=1)[-1]]
    return set(field["enum"])


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ChapterAnalysisSchema, id="chapter-analysis"),
        pytest.param(TranslationResultSchema, id="translation-result"),
    ],
)
def test_strict_json_schema_has_no_value_constraints(model: type) -> None:
    """The provider schema defines structure without value constraints.

    Support varies among compatible servers, and strict mode in some of them
    rejects such requests entirely.
    """
    rendered = repr(strict_json_schema(model))

    assert "minLength" not in rendered
    assert "minItems" not in rendered
    assert "minimum" not in rendered


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ChapterAnalysisSchema, id="chapter-analysis"),
        pytest.param(TranslationResultSchema, id="translation-result"),
    ],
)
def test_strict_json_schema_forbids_additional_properties(model: type) -> None:
    """Each schema object forbids additional properties, as required by strict output."""
    schema = strict_json_schema(model)
    objects = [
        node
        for node in _walk(schema)
        if isinstance(node, dict) and node.get("type") == "object"
    ]

    assert objects
    assert all(node.get("additionalProperties") is False for node in objects)
    assert all(set(node.get("required", [])) == set(node.get("properties", {})) for node in objects)


def test_treatment_plan_item_asks_only_for_block_and_treatment() -> None:
    """Treatment plan item contains only block and treatment, without justification (QA-13).

    Treatment reasons were not stored anywhere, while each added output tokens
    to the analysis response, directly impacting latency.
    """
    schema = strict_json_schema(ChapterAnalysisSchema)
    plan_items = [
        node
        for node in _walk(schema)
        if isinstance(node, dict)
        and set(node.get("properties", {})) >= {"source_block_id", "treatment"}
        and "translated_text" not in node.get("properties", {})
    ]

    assert plan_items
    assert all(
        set(node["properties"]) == {"source_block_id", "treatment"} for node in plan_items
    )


def test_treatment_plan_item_with_reason_is_rejected() -> None:
    """A response with the legacy reason field violates the schema and is rejected."""
    payload = analysis_payload()
    payload["treatment_plan"][0]["reason"] = "обычный повествовательный абзац"

    with pytest.raises(ValidationError):
        ChapterAnalysisSchema.model_validate_json(as_content(payload))


def _walk(node: object) -> list[object]:
    """All schema nodes, including nested definitions."""
    found = [node]
    if isinstance(node, dict):
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk(item))
    return found


def test_v1_schema_unchanged():
    import hashlib
    import json
    from linguary.adapters.llm.payloads import schema_for
    from linguary.application.ports.llm import LLMResponseKind
    expected = {'chapter_analysis': '9c354997070f86d64b9ec7fb9ef3d741b76f7f2a08c84bf6bb8eccf9c9622b88', 'translation_result': '85cdddc508275c54c92f92cc5e39b7f9a5cbb0a0388051c093d7ad5329ba25f9'}
    for kind in LLMResponseKind:
        wire = strict_json_schema(schema_for(kind))
        assert hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == expected[kind.value]


def test_v2_preserves_exact_source_substrings():
    from linguary.adapters.llm.schemas import SourceAnchorSchema, HelpEvidenceSchema
    anchor = SourceAnchorSchema(block_id='b1', start=0, end=5, quote='Echo ')
    assert anchor.quote == 'Echo '
    evidence = HelpEvidenceSchema(difficulty_id='d', point_id='p', source_block_id='b1', artifact_kind='language_note.note', artifact_index=0, body_excerpt=' two meanings ')
    assert evidence.body_excerpt == ' two meanings '
