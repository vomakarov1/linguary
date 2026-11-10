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
    assert [a["artifact_kind"] for a in parsed["artifacts"]] == [
        "analysis",
        "original",
        "result",
    ]
    assert [a["block_ids"] for a in parsed["artifacts"]] == [
        ["b1", "b2"],
        ["b1"],
        ["b1"],
    ]

    llm_analysis_body = parsed["artifacts"][0]["body"]
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


def test_quality_review_supports_multi_block_findings_and_cleans_glossary():
    from linguary.application.quality_context import (
        QualityReviewResponse,
        request_text,
        validate_response,
    )
    from linguary.domain.book_conditions import WorkStateError
    from linguary.domain.content_plan import canonical_sha

    long_diff_id = "df-" + "1" * 64
    long_fact_id = "cf-" + "2" * 64

    orig1_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-1"],
        body=dict(text="The quick brown fox jumps."),
    )
    orig1_sha = canonical_sha(orig1_env)
    orig2_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-2"],
        body=dict(text="The lazy dog sleeps."),
    )
    orig2_sha = canonical_sha(orig2_env)

    res1_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-1"],
        body=dict(
            block_id="block-uuid-1",
            translated_text="Быстрая бурая лиса прыгает.",
            fragment_assistances=[
                dict(
                    difficulty_id=long_diff_id,
                    anchors=[dict(block_id="block-uuid-1", start=4, end=9, quote="quick")],
                )
            ],
        ),
    )
    res1_sha = canonical_sha(res1_env)
    res2_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-2"],
        body=dict(
            block_id="block-uuid-2",
            translated_text="Ленивая собака спит.",
        ),
    )
    res2_sha = canonical_sha(res2_env)

    analysis_env = dict(
        artifact_kind="analysis",
        block_ids=["block-uuid-1", "block-uuid-2"],
        body=dict(
            summary="fox and dog",
            content_plan=dict(
                version=3,
                profile="content-plan-v3",
                sha256="f" * 64,
                difficulties=[
                    dict(
                        id=long_diff_id,
                        owner_block_id="block-uuid-1",
                        anchors=[dict(block_id="block-uuid-1", start=4, end=9, quote="quick")],
                    )
                ],
                context_facts=[
                    dict(
                        id=long_fact_id,
                        anchors=[dict(block_id="block-uuid-2", start=4, end=12, quote="lazy dog")],
                        subject_anchor=dict(block_id="block-uuid-2", start=9, end=12, quote="dog"),
                    )
                ],
            ),
        ),
    )
    analysis_sha = canonical_sha(analysis_env)

    manifest = dict(
        schema_version=1,
        book_id="book-1",
        scope=["block-uuid-1", "block-uuid-2"],
        book_complete=True,
        missing_results=[],
        glossary=dict(
            id="glossary-db-id",
            book_id="book-1",
            entries=[
                dict(
                    id="entry-1",
                    book_id="book-1",
                    created_at="2026-01-01T00:00:00Z",
                    concept_id="concept-1",
                    canonical_source_text="brown fox",
                    source_text="Brown Fox",
                    translation="Бурая лиса",
                    type="person",
                    status="approved",
                    note="   ",
                ),
                dict(
                    id="entry-2",
                    book_id="book-1",
                    source_text="Lazy Dog",
                    translation="Ленивый пёс",
                    type="person",
                    status="locked",
                    note="Важное имя",
                ),
            ],
        ),
        artifacts=[
            {**orig1_env, "artifact_sha": orig1_sha},
            {**res1_env, "artifact_sha": res1_sha},
            {**orig2_env, "artifact_sha": orig2_sha},
            {**res2_env, "artifact_sha": res2_sha},
            {**analysis_env, "artifact_sha": analysis_sha},
        ],
    )

    serialized = request_text(manifest)
    parsed = json.loads(serialized)

    assert list(parsed.keys()) == [
        "scope",
        "book_complete",
        "missing_results",
        "glossary",
        "artifacts",
    ]
    assert serialized.index('"glossary":') < serialized.index('"artifacts":')
    assert parsed["glossary"] == {
        "entries": [
            {
                "source_text": "Brown Fox",
                "translation": "Бурая лиса",
                "type": "person",
                "status": "approved",
            },
            {
                "source_text": "Lazy Dog",
                "translation": "Ленивый пёс",
                "type": "person",
                "status": "locked",
                "note": "Важное имя",
            },
        ]
    }
    assert long_diff_id not in serialized
    assert long_fact_id not in serialized
    assert parsed["artifacts"][3]["body"]["fragment_assistances"][0]["difficulty_id"] == "d1"
    assert parsed["artifacts"][0]["body"]["content_plan"]["difficulties"][0]["id"] == "d1"
    assert parsed["artifacts"][0]["body"]["content_plan"]["context_facts"][0]["id"] == "f1"

    multi_block_response = QualityReviewResponse(
        findings=(
            dict(
                block_ids=["b1", "b2"],
                artifact_kind="result",
                severity="minor",
                issue="Cross-block consistency",
                evidence_anchors=[
                    dict(block_id="b1", quote="brown fox"),
                    dict(block_id="b2", quote="lazy dog"),
                ],
                suggestion="Align terminology",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    validated = validate_response(manifest, multi_block_response)
    finding, = validated.findings
    assert finding["artifact_sha"] == res1_sha
    assert finding["block_ids"] == ["block-uuid-1", "block-uuid-2"]
    assert [a["block_id"] for a in finding["evidence_anchors"]] == ["block-uuid-1", "block-uuid-2"]

    manifest_missing_res2 = {
        **manifest,
        "artifacts": [a for a in manifest["artifacts"] if a["artifact_sha"] != res2_sha],
    }
    with pytest.raises(WorkStateError, match="чужой артефакт замечания"):
        validate_response(manifest_missing_res2, multi_block_response)


def test_quality_context_validates_multi_block_artifact_sha_and_filters_scoped_analysis():
    from linguary.application.quality_context import (
        QualityReviewResponse,
        request_text,
        validate_response,
    )
    from linguary.domain.book_conditions import WorkStateError
    from linguary.domain.content_plan import canonical_sha

    orig1_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-1"],
        body=dict(text="First block text here."),
    )
    orig1_sha = canonical_sha(orig1_env)
    orig2_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-2"],
        body=dict(text="Second block text here."),
    )
    orig2_sha = canonical_sha(orig2_env)

    res1_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-1"],
        body=dict(block_id="block-uuid-1", translated_text="Первый блок."),
    )
    res1_sha = canonical_sha(res1_env)
    res2_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-2"],
        body=dict(block_id="block-uuid-2", translated_text="Второй блок."),
    )
    res2_sha = canonical_sha(res2_env)

    analysis_env = dict(
        artifact_kind="analysis",
        block_ids=["block-uuid-1"],
        body=dict(
            summary="scoped chapter",
            treatment_plan=dict(
                section_id="sec-1",
                items=[
                    dict(block_id="block-uuid-1", treatment="TRANSLATE"),
                    dict(block_id="block-uuid-2", treatment="READ_AS_IS"),
                ],
            ),
            content_plan=dict(
                version=3,
                profile="content-plan-v3",
                sha256="f" * 64,
                difficulties=[
                    dict(
                        id="df-" + "1" * 64,
                        owner_block_id="block-uuid-1",
                        span_block_ids=["block-uuid-1"],
                        anchors=[dict(block_id="block-uuid-1", start=0, end=5, quote="First")],
                    ),
                    dict(
                        id="df-" + "2" * 64,
                        owner_block_id="block-uuid-2",
                        span_block_ids=["block-uuid-2"],
                        anchors=[dict(block_id="block-uuid-2", start=0, end=6, quote="Second")],
                    ),
                ],
                context_facts=[
                    dict(
                        id="cf-" + "1" * 64,
                        applies_to_block_ids=["block-uuid-1"],
                        anchors=[dict(block_id="block-uuid-1", start=0, end=5, quote="First")],
                        subject_anchor=dict(block_id="block-uuid-1", start=0, end=5, quote="First"),
                    ),
                    dict(
                        id="cf-" + "2" * 64,
                        applies_to_block_ids=["block-uuid-2"],
                        anchors=[dict(block_id="block-uuid-2", start=0, end=6, quote="Second")],
                        subject_anchor=dict(block_id="block-uuid-2", start=0, end=6, quote="Second"),
                    ),
                ],
            ),
        ),
    )
    analysis_sha = canonical_sha(analysis_env)

    scoped_manifest = dict(
        schema_version=1,
        book_id="book-1",
        scope=["block-uuid-1"],
        book_complete=False,
        missing_results=[],
        artifacts=[
            {**orig1_env, "artifact_sha": orig1_sha},
            {**res1_env, "artifact_sha": res1_sha},
            {**analysis_env, "artifact_sha": analysis_sha},
        ],
    )
    serialized = request_text(scoped_manifest)
    assert '": ' not in serialized
    assert '", ' not in serialized
    parsed = json.loads(serialized)
    analysis_body = parsed["artifacts"][0]["body"]
    assert analysis_body["treatment_plan"]["items"] == [
        dict(block_id="b1", treatment="TRANSLATE"),
    ]
    assert [d["id"] for d in analysis_body["content_plan"]["difficulties"]] == ["d1"]
    assert [f["id"] for f in analysis_body["content_plan"]["context_facts"]] == ["f1"]

    orig3_env = dict(
        artifact_kind="original",
        block_ids=["block-uuid-3"],
        body=dict(text="Third block text here."),
    )
    orig3_sha = canonical_sha(orig3_env)
    res3_env = dict(
        artifact_kind="result",
        block_ids=["block-uuid-3"],
        body=dict(block_id="block-uuid-3", translated_text="Третий блок."),
    )
    res3_sha = canonical_sha(res3_env)

    full_manifest = dict(
        schema_version=1,
        book_id="book-1",
        scope=["block-uuid-1", "block-uuid-2", "block-uuid-3"],
        artifacts=[
            {**orig1_env, "artifact_sha": orig1_sha},
            {**res1_env, "artifact_sha": res1_sha},
            {**orig2_env, "artifact_sha": orig2_sha},
            {**res2_env, "artifact_sha": res2_sha},
            {**orig3_env, "artifact_sha": orig3_sha},
            {**res3_env, "artifact_sha": res3_sha},
        ],
    )
    second_block_sha_response = QualityReviewResponse(
        findings=(
            dict(
                artifact_sha=res2_sha,
                block_ids=["b1", "b2"],
                artifact_kind="result",
                severity="minor",
                issue="Second block sha allowed",
                evidence_anchors=[
                    dict(block_id="b1", quote="First"),
                    dict(block_id="b2", quote="Second"),
                ],
                suggestion="Fix",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    validated_second = validate_response(full_manifest, second_block_sha_response)
    assert validated_second.findings[0]["artifact_sha"] == res2_sha

    wrong_sha_response = QualityReviewResponse(
        findings=(
            dict(
                artifact_sha=res3_sha,
                block_ids=["b1", "b2"],
                artifact_kind="result",
                severity="minor",
                issue="Mismatch sha",
                evidence_anchors=[
                    dict(block_id="b1", quote="First"),
                    dict(block_id="b2", quote="Second"),
                ],
                suggestion="Fix",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    with pytest.raises(WorkStateError, match="чужой артефакт замечания"):
        validate_response(full_manifest, wrong_sha_response)

    missing_second_manifest = dict(
        schema_version=1,
        book_id="book-1",
        scope=["block-uuid-1", "block-uuid-2"],
        artifacts=[
            {**orig1_env, "artifact_sha": orig1_sha},
            {**res1_env, "artifact_sha": res1_sha},
            {**orig2_env, "artifact_sha": orig2_sha},
        ],
    )
    explicit_first_sha_response = QualityReviewResponse(
        findings=(
            dict(
                artifact_sha=res1_sha,
                block_ids=["b1", "b2"],
                artifact_kind="result",
                severity="minor",
                issue="Second result missing",
                evidence_anchors=[
                    dict(block_id="b1", quote="First"),
                    dict(block_id="b2", quote="Second"),
                ],
                suggestion="Fix",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    with pytest.raises(WorkStateError, match="чужой артефакт замечания"):
        validate_response(missing_second_manifest, explicit_first_sha_response)

    from types import SimpleNamespace
    from linguary.application.quality_budget import QualityBudget
    import linguary.application.quality_context as qc_mod

    cancelled: list[str] = []
    snapshot_calls = 0

    def fail_if_snapshot_called(*args, **kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        raise AssertionError("snapshot must not be called when review is already stale")

    fake_tx = SimpleNamespace(
        work=SimpleNamespace(
            get_attempt=lambda aid: SimpleNamespace(id=aid, review_id="rev-stale"),
            get_review=lambda rid: SimpleNamespace(
                id=rid,
                book_id="book-1",
                state="stale",
                manifest={"scope": ["block-uuid-1"], "manifest_sha": "a" * 64},
            ),
            cancel_review_unsent=lambda aid: cancelled.append(aid),
        )
    )
    from contextlib import nullcontext

    budget = QualityBudget(SimpleNamespace(transaction=lambda: nullcontext(fake_tx)))
    orig_snapshot = qc_mod.snapshot
    qc_mod.snapshot = fail_if_snapshot_called
    try:
        assert budget.mark_sent("att-stale") is False
        assert cancelled == ["att-stale"]
        assert snapshot_calls == 0
    finally:
        qc_mod.snapshot = orig_snapshot


def test_partition_manifest_keeps_single_chapter_order_and_splits_multi_chapter_scope():
    from linguary.application.quality_context import (
        MAX_QUALITY_CHUNK_BLOCKS,
        QualityReviewResponse,
        partition_manifest,
        request_text,
        validate_response,
    )
    from linguary.domain.book_conditions import WorkStateError
    from linguary.domain.content_plan import canonical_sha

    assert MAX_QUALITY_CHUNK_BLOCKS == 80

    def make_orig(bid: str, text: str) -> dict:
        env = dict(artifact_kind="original", block_ids=[bid], body=dict(text=text))
        return {**env, "artifact_sha": canonical_sha(env)}

    def make_res(bid: str, tr: str) -> dict:
        env = dict(
            artifact_kind="result",
            block_ids=[bid],
            body=dict(block_id=bid, translated_text=tr),
        )
        return {**env, "artifact_sha": canonical_sha(env)}

    def make_analysis(bids: list[str], summary: str) -> dict:
        env = dict(
            artifact_kind="analysis",
            block_ids=list(bids),
            body=dict(
                summary=summary,
                treatment_plan=dict(
                    section_id=f"sec-{summary}",
                    items=[dict(block_id=b, treatment="TRANSLATE") for b in bids],
                ),
                content_plan=dict(
                    version=3,
                    profile="content-plan-v3",
                    sha256="a" * 64,
                    difficulties=[
                        dict(
                            id=f"df-{idx:062d}",
                            owner_block_id=b,
                            anchors=[dict(block_id=b, start=0, end=5, quote="Block")],
                        )
                        for idx, b in enumerate(bids, start=1)
                    ],
                    context_facts=[
                        dict(
                            id=f"cf-{idx:062d}",
                            anchors=[dict(block_id=b, start=0, end=5, quote="Block")],
                            subject_anchor=dict(block_id=b, start=0, end=5, quote="Block"),
                        )
                        for idx, b in enumerate(bids, start=1)
                    ],
                ),
            ),
        )
        return {**env, "artifact_sha": canonical_sha(env)}

    # 1. Single-chapter manifest within max_blocks_per_chunk keeps exact positional order
    orig_1 = make_orig("ch1-b1", "Block one alpha text.")
    res_1 = make_res("ch1-b1", "Блок один.")
    orig_2 = make_orig("ch1-b2", "Block two beta text.")
    res_2 = make_res("ch1-b2", "Блок два.")
    analysis_ch1_short = make_analysis(["ch1-b1", "ch1-b2"], "ch1")

    single_manifest = dict(
        schema_version=1,
        book_id="book-1",
        scope=["ch1-b1", "ch1-b2"],
        missing_results=[],
        glossary=dict(
            entries=[
                dict(source_text="Alpha", translation="Альфа", type="term", status="approved"),
            ]
        ),
        artifacts=[orig_1, res_1, orig_2, res_2, analysis_ch1_short],
    )
    single_chunks = partition_manifest(single_manifest)
    assert len(single_chunks) == 1
    assert single_chunks[0] == single_manifest
    assert single_chunks[0] is not single_manifest

    single_parsed = json.loads(request_text(single_manifest))
    assert [a["artifact_kind"] for a in single_parsed["artifacts"]] == [
        "analysis",
        "original",
        "original",
        "result",
        "result",
    ]

    # 2. Multi-chapter manifest: grouped by chapter in request_text when >1 analysis,
    #    and partitioned by chapters + windows of max_blocks_per_chunk when scope > limit.
    orig_3 = make_orig("ch1-b3", "Block three gamma text.")
    analysis_ch1 = make_analysis(["ch1-b1", "ch1-b2", "ch1-b3"], "ch1-long")

    orig_4 = make_orig("ch2-b1", "Block four delta text.")
    res_4 = make_res("ch2-b1", "Блок четыре.")
    orig_5 = make_orig("ch2-b2", "Block five epsilon text.")
    res_5 = make_res("ch2-b2", "Блок пять.")
    analysis_ch2 = make_analysis(["ch2-b1", "ch2-b2"], "ch2")

    multi_manifest = dict(
        schema_version=1,
        book_id="book-1",
        scope=["ch1-b1", "ch1-b2", "ch1-b3", "ch2-b1", "ch2-b2"],
        missing_results=["ch1-b3"],
        glossary=dict(
            entries=[
                dict(source_text="Gamma", translation="Гамма", type="term", status="approved"),
                dict(source_text="Delta", translation="Дельта", type="term", status="approved"),
            ]
        ),
        artifacts=[
            orig_1,
            res_1,
            orig_2,
            res_2,
            orig_3,
            orig_4,
            res_4,
            orig_5,
            res_5,
            analysis_ch1,
            analysis_ch2,
        ],
    )

    multi_parsed = json.loads(request_text(multi_manifest))
    assert [a["artifact_kind"] for a in multi_parsed["artifacts"]] == [
        "analysis",
        "original",
        "original",
        "original",
        "result",
        "result",
        "analysis",
        "original",
        "original",
        "result",
        "result",
    ]

    chunks = partition_manifest(multi_manifest, max_blocks_per_chunk=2)
    assert len(chunks) == 3
    assert [c["scope"] for c in chunks] == [
        ["ch1-b1", "ch1-b2"],
        ["ch1-b3"],
        ["ch2-b1", "ch2-b2"],
    ]
    assert [c["missing_results"] for c in chunks] == [
        [],
        ["ch1-b3"],
        [],
    ]
    assert [a["artifact_sha"] for a in chunks[0]["artifacts"]] == [
        orig_1["artifact_sha"],
        res_1["artifact_sha"],
        orig_2["artifact_sha"],
        res_2["artifact_sha"],
        analysis_ch1["artifact_sha"],
    ]
    assert [a["artifact_sha"] for a in chunks[1]["artifacts"]] == [
        orig_3["artifact_sha"],
        analysis_ch1["artifact_sha"],
    ]
    assert [a["artifact_sha"] for a in chunks[2]["artifacts"]] == [
        orig_4["artifact_sha"],
        res_4["artifact_sha"],
        orig_5["artifact_sha"],
        res_5["artifact_sha"],
        analysis_ch2["artifact_sha"],
    ]

    # Window 2 of long chapter 1 serializes only its window slice of the content/treatment plan
    chunk2_parsed = json.loads(request_text(chunks[1]))
    assert chunk2_parsed["scope"] == ["b1"]
    assert chunk2_parsed["missing_results"] == ["b1"]
    assert [e["source_text"] for e in chunk2_parsed["glossary"]["entries"]] == ["Gamma"]
    chunk2_analysis = next(
        a["body"] for a in chunk2_parsed["artifacts"] if a["artifact_kind"] == "analysis"
    )
    assert chunk2_analysis["treatment_plan"]["items"] == [
        dict(block_id="b1", treatment="TRANSLATE"),
    ]
    assert len(chunk2_analysis["content_plan"]["difficulties"]) == 1
    assert chunk2_analysis["content_plan"]["difficulties"][0]["owner_block_id"] == "b1"
    assert len(chunk2_analysis["content_plan"]["context_facts"]) == 1
    assert chunk2_analysis["content_plan"]["context_facts"][0]["subject_anchor"]["block_id"] == "b1"

    # 3. Actionable error messages in validate_response preserve prefixes and include index, bN, quote
    bad_quote_response = QualityReviewResponse(
        findings=(
            dict(
                block_ids=["b1"],
                artifact_kind="result",
                severity="minor",
                issue="Valid first finding",
                evidence_anchors=[dict(block_id="b1", quote="Block one")],
                suggestion="Keep",
                uncertainty="low",
            ),
            dict(
                block_ids=["b2"],
                artifact_kind="result",
                severity="major",
                issue="Broken second finding",
                evidence_anchors=[dict(block_id="b2", quote="nonexistent phrase")],
                suggestion="Fix",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    with pytest.raises(WorkStateError) as exc_info:
        validate_response(single_manifest, bad_quote_response)
    msg = str(exc_info.value)
    assert msg.startswith("неверная привязка замечания к оригиналу (")
    assert "2" in msg
    assert "b2" in msg
    assert "nonexistent phrase" in msg

    foreign_artifact_response = QualityReviewResponse(
        findings=(
            dict(
                block_ids=["b9"],
                artifact_kind="result",
                severity="minor",
                issue="Unknown block",
                evidence_anchors=[dict(block_id="b9", quote="Block two")],
                suggestion="Fix",
                uncertainty="low",
            ),
        ),
        checked_artifacts=(),
        missing_artifacts=(),
    )
    with pytest.raises(WorkStateError) as foreign_exc:
        validate_response(single_manifest, foreign_artifact_response)
    foreign_msg = str(foreign_exc.value)
    assert foreign_msg.startswith("чужой артефакт замечания (")
    assert "1" in foreign_msg
    assert "b9" in foreign_msg
    assert "Block two" in foreign_msg


def test_single_chapter_request_text_places_analysis_before_blocks_preserving_manifest_sha():
    from types import SimpleNamespace
    from analysis_fakes import make_analysis_document, make_analysis_section
    from linguary.application.quality_context import (
        _order_artifacts_for_request,
        partition_manifest,
        request_text,
        snapshot,
    )
    from linguary.domain import (
        BlockProcessingResult,
        BlockTreatment,
        ChapterAnalysis,
        GlossarySnapshot,
        TreatmentPlan,
    )

    section = make_analysis_section(
        section_id="sec-1",
        order=0,
        texts=("First paragraph.", "Second paragraph."),
    )
    b1, b2 = section.blocks
    doc = make_analysis_document((section,))
    analysis = ChapterAnalysis.create(
        id="analysis-sec-1",
        section_id="sec-1",
        treatment_plan=TreatmentPlan.of_treatments(
            "sec-1",
            ((b1.id, BlockTreatment.TRANSLATE), (b2.id, BlockTreatment.TRANSLATE)),
        ),
        translation_instructions="Keep style.",
        chapter_summary="Single chapter summary.",
        expected_block_ids=(b1.id, b2.id),
    )
    r1 = BlockProcessingResult(
        block_id=b1.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=" Первый абзац.",
    )
    r2 = BlockProcessingResult(
        block_id=b2.id,
        treatment=BlockTreatment.TRANSLATE,
        translated_text=" Второй абзац.",
    )
    run = SimpleNamespace(id="run-1", glossary_snapshot=GlossarySnapshot())
    tx = SimpleNamespace(
        documents=SimpleNamespace(get=lambda _bid: doc),
        work=SimpleNamespace(
            latest_conditions=lambda _bid: SimpleNamespace(
                id="cond-1",
                payload={"schema_version": 1},
            )
        ),
        runs=SimpleNamespace(active_for_book=lambda _bid: run),
        results=SimpleNamespace(list_for_run=lambda _rid: (r1, r2)),
        analyses=SimpleNamespace(list_for_book=lambda _bid: (analysis,)),
        glossary=SimpleNamespace(list_for_book=lambda _bid: ()),
    )

    manifest = snapshot(tx, "book-1", "all")
    expected_manifest_sha = manifest["manifest_sha"]
    # Snapshot itself keeps canonical artifact order (originals/results then analysis) for manifest_sha stability
    assert [a["artifact_kind"] for a in manifest["artifacts"]] == [
        "original",
        "result",
        "original",
        "result",
        "analysis",
    ]

    # Single-chapter request_text places chapter analysis first before original and result blocks
    chunk, = partition_manifest(manifest)
    parsed = json.loads(request_text(chunk))
    assert manifest["manifest_sha"] == expected_manifest_sha
    assert chunk["manifest_sha"] == expected_manifest_sha
    assert [a["artifact_kind"] for a in parsed["artifacts"]] == [
        "analysis",
        "original",
        "original",
        "result",
        "result",
    ]
    assert [a["block_ids"] for a in parsed["artifacts"]] == [
        ["b1", "b2"],
        ["b1"],
        ["b2"],
        ["b1"],
        ["b2"],
    ]

    # Without analysis artifacts, _order_artifacts_for_request preserves raw order
    no_analysis = [a for a in manifest["artifacts"] if a["artifact_kind"] != "analysis"]
    assert _order_artifacts_for_request(no_analysis, manifest["scope"]) == no_analysis


