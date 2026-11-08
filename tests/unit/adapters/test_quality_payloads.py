import json
import pytest
from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.adapters.llm.payloads import parse_payload, schema_for
from linguary.application.ports.llm import LLMRequest, LLMResponseKind, LLMResponseFormatError


async def test_fake_review_response_validation():
    payload = dict(findings=[], checked_artifacts=['a' * 64], missing_artifacts=[])
    client = FakeLLMClient([FakeReply.of_payload(payload)])
    result = await client.generate(LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW, instructions='review', input_text='{}'))
    assert result.payload.checked_artifacts == ('a' * 64,)
    assert schema_for(LLMResponseKind.QUALITY_REVIEW).model_json_schema()['additionalProperties'] is False


@pytest.mark.parametrize('payload', [dict(findings=[], checked_artifacts=[]),
    dict(findings=[], checked_artifacts=[], missing_artifacts=[], extra=True),
    dict(findings=[dict(issue='unbound')], checked_artifacts=[], missing_artifacts=[])])
def test_strict_schema_rejects_incomplete_or_unbound_findings(payload):
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.QUALITY_REVIEW, json.dumps(payload))


def test_quality_review_strict_schema_omits_checked_artifacts_sha_and_offsets():
    from linguary.adapters.llm.json_schema import strict_json_schema
    from linguary.adapters.llm.quality_payloads import QualityReviewSchema

    schema = strict_json_schema(QualityReviewSchema)
    assert "checked_artifacts" not in schema["properties"]
    assert "missing_artifacts" not in schema["properties"]
    assert "checked_artifacts" not in schema.get("required", [])
    assert "missing_artifacts" not in schema.get("required", [])

    finding_def = schema["$defs"]["ReviewFinding"]
    assert "artifact_sha" not in finding_def["properties"]
    assert "artifact_sha" not in finding_def.get("required", [])

    anchor_def = schema["$defs"]["ReviewAnchor"]
    assert "quote" in anchor_def["properties"]
    assert "block_id" in anchor_def["properties"]
    assert "start" not in anchor_def["properties"]
    assert "end" not in anchor_def["properties"]
    assert "start" not in anchor_def.get("required", [])
    assert "end" not in anchor_def.get("required", [])


def test_validate_response_populates_checked_artifacts_sha_and_anchor_offsets_from_manifest():
    from linguary.application.quality_context import QualityReviewResponse, validate_response
    from linguary.domain.content_plan import canonical_sha

    original_env = dict(artifact_kind="original", block_ids=["b1"], body=dict(text="The quick brown fox jumps."))
    original_sha = canonical_sha(original_env)
    analysis_env = dict(artifact_kind="analysis", block_ids=["b1", "b2"], body=dict(summary="fox chapter"))
    analysis_sha = canonical_sha(analysis_env)
    result_env = dict(artifact_kind="result", block_ids=["b1"], body=dict(translated_text="Быстрая лиса."))
    result_sha = canonical_sha(result_env)

    manifest = dict(
        artifacts=[
            {**original_env, "artifact_sha": original_sha},
            {**analysis_env, "artifact_sha": analysis_sha},
            {**result_env, "artifact_sha": result_sha},
        ]
    )

    model_findings = [
        dict(
            block_ids=["b1"],
            artifact_kind="original",
            severity="minor",
            issue="Vocabulary note",
            evidence_anchors=[dict(block_id="b1", quote="brown fox")],
            suggestion="Clarify tone",
            uncertainty="low",
        ),
        dict(
            block_ids=["b1"],
            artifact_kind="analysis",
            severity="major",
            issue="Analysis gap",
            evidence_anchors=[dict(block_id="b1", quote="quick")],
            suggestion="Update plan",
            uncertainty="medium",
        ),
    ]
    raw_response = QualityReviewResponse(
        findings=tuple(model_findings),
        checked_artifacts=(),
        missing_artifacts=(),
    )

    validated = validate_response(manifest, raw_response)

    assert set(validated.checked_artifacts) == {original_sha, analysis_sha, result_sha}
    assert validated.missing_artifacts == ()

    f1 = validated.findings[0]
    assert f1["artifact_sha"] == original_sha
    a1 = f1["evidence_anchors"][0]
    assert a1["start"] == 10
    assert a1["end"] == 19
    assert a1["quote"] == "brown fox"
    assert "The quick brown fox jumps."[a1["start"]:a1["end"]] == "brown fox"

    f2 = validated.findings[1]
    assert f2["artifact_sha"] == analysis_sha
    a2 = f2["evidence_anchors"][0]
    assert a2["start"] == 4
    assert a2["end"] == 9
    assert a2["quote"] == "quick"
    assert "The quick brown fox jumps."[a2["start"]:a2["end"]] == "quick"
