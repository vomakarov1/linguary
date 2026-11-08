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

    original_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-1"],
        body=dict(text="The quick brown fox’s jump."),
    )
    original_sha = canonical_sha(original_env)
    analysis_env = dict(
        artifact_kind="analysis",
        block_ids=["block-uuid-1", "block-uuid-2"],
        body=dict(summary="fox chapter"),
    )
    analysis_sha = canonical_sha(analysis_env)
    result_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-1"],
        body=dict(translated_text="Быстрая лиса."),
    )
    result_sha = canonical_sha(result_env)

    manifest = dict(
        scope=["block-uuid-1", "block-uuid-2"],
        artifacts=[
            {**original_env, "artifact_sha": original_sha},
            {**analysis_env, "artifact_sha": analysis_sha},
            {**result_env, "artifact_sha": result_sha},
        ],
    )

    model_findings = [
        dict(
            block_ids=["b1"],
            artifact_kind="original",
            severity="minor",
            issue="Vocabulary note",
            evidence_anchors=[dict(block_id="b1", quote="brown fox's")],
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
    assert f1["block_ids"] == ["block-uuid-1"]
    a1 = f1["evidence_anchors"][0]
    assert a1["block_id"] == "block-uuid-1"
    assert a1["start"] == 10
    assert a1["end"] == 21
    assert a1["quote"] == "brown fox’s"
    assert "The quick brown fox’s jump."[a1["start"]:a1["end"]] == "brown fox’s"

    f2 = validated.findings[1]
    assert f2["artifact_sha"] == analysis_sha
    assert f2["block_ids"] == ["block-uuid-1"]
    a2 = f2["evidence_anchors"][0]
    assert a2["block_id"] == "block-uuid-1"
    assert a2["start"] == 4
    assert a2["end"] == 9
    assert a2["quote"] == "quick"
    assert "The quick brown fox’s jump."[a2["start"]:a2["end"]] == "quick"


def test_request_text_strips_internal_evidence_and_warnings_while_preserving_manifest_shas():
    import copy
    from linguary.application.quality_context import (
        QualityReviewResponse, _manifest_block_aliases, request_text, validate_response,
    )
    from linguary.domain.content_plan import canonical_sha

    original_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-1"],
        body=dict(text="The quick brown fox jumps."),
    )
    original_sha = canonical_sha(original_env)

    analysis_env = dict(
        artifact_kind="analysis",
        block_ids=["block-uuid-1", "block-uuid-2"],
        body=dict(
            id="analysis-1",
            section_id="section-1",
            coverage_status="complete",
            summary="fox chapter",
            treatment_plan=dict(
                section_id="section-1",
                items=[
                    dict(block_id="block-uuid-1", treatment="TRANSLATE"),
                    dict(block_id="block-uuid-2", treatment="READ_AS_IS"),
                ],
            ),
            glossary_candidates=[dict(source_text="Fox", suggested_translation="Лис")],
            name_translation_evidence=[dict(source_text="Fox", decision="translate")],
            before_you_read_evidence=[dict(difficulty_id="d1", artifact_key="a" * 64)],
            content_plan=dict(
                version=3,
                profile="content-plan-v3",
                sha256="f" * 64,
                difficulties=[
                    dict(
                        id="d1",
                        owner_block_id="block-uuid-1",
                        anchors=[dict(block_id="block-uuid-1", start=4, end=9, quote="quick")],
                    )
                ],
                context_facts=[
                    dict(
                        id="cf1",
                        anchors=[dict(block_id="block-uuid-1", start=10, end=19, quote="brown fox")],
                        subject_anchor=dict(block_id="block-uuid-1", start=16, end=19, quote="fox"),
                    )
                ],
                before_you_read_evidence=[dict(difficulty_id="d1", artifact_key="a" * 64)],
            ),
        ),
    )
    analysis_sha = canonical_sha(analysis_env)

    result_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-1"],
        body=dict(
            block_id="block-uuid-1",
            translated_text="Быстрая рыжая лиса прыгает.",
            language_notes=[
                dict(id="ln-ch01-001", block_id="block-uuid-1", original_excerpt="quick", note="быстрая")
            ],
            fragment_assistances=[
                dict(
                    difficulty_id="d1",
                    anchors=[dict(block_id="block-uuid-1", start=4, end=9, quote="quick")],
                )
            ],
            content_evidence=dict(plan_sha256="b" * 64, evidence=[dict(artifact_key="c" * 64)]),
            pronunciation_assistances=[dict(ipa="fɑks")],
            warnings=[dict(kind="repetition", message="повтор")],
        ),
    )
    result_sha = canonical_sha(result_env)

    manifest = dict(
        schema_version=1,
        book_id="book-1",
        book_manifest_sha="d" * 64,
        manifest_sha="e" * 64,
        conditions_id="cond-1",
        conditions_sha="1" * 64,
        scope=["block-uuid-1", "block-uuid-2"],
        missing_results=["block-uuid-2"],
        glossary=dict(
            entries=[
                dict(source_text="Brown Fox", translation="Рыжий Лис", type="person", status="approved", note=""),
                dict(source_text="Unrelated Castle", translation="Замок", type="place", status="approved", note=""),
            ]
        ),
        artifacts=[
            {**original_env, "artifact_sha": original_sha},
            {**analysis_env, "artifact_sha": analysis_sha},
            {**result_env, "artifact_sha": result_sha},
        ],
    )
    manifest_before = copy.deepcopy(manifest)

    assert _manifest_block_aliases(manifest) == {
        "block-uuid-1": "b1",
        "block-uuid-2": "b2",
    }

    serialized = request_text(manifest)
    parsed = json.loads(serialized)

    assert manifest == manifest_before
    for stripped_key in (
        "schema_version",
        "book_id",
        "book_manifest_sha",
        "manifest_sha",
        "conditions_id",
        "conditions_sha",
    ):
        assert stripped_key not in parsed

    assert parsed["scope"] == ["b1", "b2"]
    assert parsed["missing_results"] == ["b2"]
    assert [e["source_text"] for e in parsed["glossary"]["entries"]] == ["Brown Fox"]
    assert all("artifact_sha" not in a for a in parsed["artifacts"])
    assert [a["block_ids"] for a in parsed["artifacts"]] == [
        ["b1"],
        ["b1", "b2"],
        ["b1"],
    ]

    llm_analysis_body = parsed["artifacts"][1]["body"]
    assert llm_analysis_body["summary"] == "fox chapter"
    assert "id" not in llm_analysis_body
    assert "section_id" not in llm_analysis_body
    assert "coverage_status" not in llm_analysis_body
    assert "section_id" not in llm_analysis_body["treatment_plan"]
    assert llm_analysis_body["treatment_plan"]["items"] == [
        dict(block_id="b1", treatment="TRANSLATE"),
        dict(block_id="b2", treatment="READ_AS_IS"),
    ]
    assert "glossary_candidates" not in llm_analysis_body
    assert "name_translation_evidence" not in llm_analysis_body
    assert "before_you_read_evidence" not in llm_analysis_body
    assert "before_you_read_evidence" not in llm_analysis_body["content_plan"]
    assert "version" not in llm_analysis_body["content_plan"]
    assert "profile" not in llm_analysis_body["content_plan"]
    assert "sha256" not in llm_analysis_body["content_plan"]
    assert llm_analysis_body["content_plan"]["difficulties"] == [
        dict(
            id="d1",
            owner_block_id="b1",
            anchors=[dict(block_id="b1", quote="quick")],
        )
    ]
    assert llm_analysis_body["content_plan"]["context_facts"] == [
        dict(
            id="cf1",
            anchors=[dict(block_id="b1", quote="brown fox")],
            subject_anchor=dict(block_id="b1", quote="fox"),
        )
    ]

    llm_result_body = parsed["artifacts"][2]["body"]
    assert llm_result_body["translated_text"] == "Быстрая рыжая лиса прыгает."
    assert "block_id" not in llm_result_body
    assert "content_evidence" not in llm_result_body
    assert "pronunciation_assistances" not in llm_result_body
    assert "warnings" not in llm_result_body
    assert llm_result_body["language_notes"] == [
        dict(block_id="b1", original_excerpt="quick", note="быстрая")
    ]
    assert llm_result_body["fragment_assistances"] == [
        dict(difficulty_id="d1", anchors=[dict(block_id="b1", quote="quick")])
    ]

    raw_response = QualityReviewResponse(
        findings=(
            dict(
                block_ids=["b1"],
                artifact_kind="result",
                severity="minor",
                issue="Tone check",
                evidence_anchors=[dict(block_id="b1", quote="brown fox")],
                suggestion="Keep",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    validated = validate_response(manifest, raw_response)
    assert validated.findings[0]["artifact_sha"] == result_sha
    assert validated.findings[0]["block_ids"] == ["block-uuid-1"]
    assert validated.findings[0]["evidence_anchors"][0]["block_id"] == "block-uuid-1"
    assert set(validated.checked_artifacts) == {original_sha, analysis_sha, result_sha}

