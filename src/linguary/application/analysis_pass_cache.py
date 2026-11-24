"""Versioned internal pass DTOs with stable source references for durable reuse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Literal, overload

from linguary.application.illustration_validation import ChapterIllustrationPlan
from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse,
    GlossaryCandidateResponse,
    HelpEvidenceResponse,
    NarrativeAnalysisResponse,
    PedagogicalAnalysisResponse,
    QuotedContextFactResponse,
    QuotedDifficultyResponse,
    QuotedSceneResponse,
    QuotedSceneVisualFactResponse,
    QuotedSourceAnchorResponse,
    SegmentTreatmentResponse,
    TreatmentPlanItemResponse,
)
from linguary.domain import (
    AnalysisPassKind,
    BlockTreatment,
    ChapterAnalysis,
    ConceptType,
    TermStatus,
)
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import RequiredPoint, artifact_key
from linguary.domain.name_translation import NameTranslationAssessment

_INPUT_CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "sequential_analysis",
        "parallel_batch",
        "seed_active",
        "user_decisions",
        "nonbinding_suggestions",
        "prompts_version",
    }
)
_USER_DECISION_KEYS = frozenset({"source_text", "type", "translation", "status", "note"})
_SUGGESTION_KEYS = frozenset({"source_text", "type", "translation", "note"})


def validate_narrative_input_context(raw: object) -> dict[str, Any]:
    """Strictly validate `input_context` (`schema_version: 1`) stored on full NARRATIVE passes."""
    if not isinstance(raw, Mapping):
        raise ValueError("input_context должен быть объектом")
    unfrozen = thaw(raw)
    if set(unfrozen.keys()) != _INPUT_CONTEXT_KEYS:
        raise ValueError("input_context содержит недопустимый набор ключей")
    if type(unfrozen["schema_version"]) is not int or unfrozen["schema_version"] != 1:
        raise ValueError("неподдерживаемая версия input_context")
    for bool_key in ("sequential_analysis", "parallel_batch", "seed_active"):
        if type(unfrozen[bool_key]) is not bool:
            raise ValueError(f"поле input_context.{bool_key} должно быть логическим")
    if type(unfrozen["prompts_version"]) is not int:
        raise ValueError("поле input_context.prompts_version должно быть целым числом")

    raw_decisions = unfrozen["user_decisions"]
    if not isinstance(raw_decisions, list):
        raise ValueError("поле input_context.user_decisions должно быть списком")
    decisions: list[dict[str, str]] = []
    for item in raw_decisions:
        if not isinstance(item, Mapping) or set(item.keys()) != _USER_DECISION_KEYS:
            raise ValueError("некорректная запись в input_context.user_decisions")
        if any(not isinstance(item[k], str) for k in _USER_DECISION_KEYS):
            raise ValueError("все поля user_decisions должны быть строками")
        ConceptType(item["type"])
        TermStatus(item["status"])
        decisions.append({k: item[k] for k in ("source_text", "type", "translation", "status", "note")})

    raw_suggestions = unfrozen["nonbinding_suggestions"]
    if not isinstance(raw_suggestions, list):
        raise ValueError("поле input_context.nonbinding_suggestions должно быть списком")
    suggestions: list[dict[str, str]] = []
    for item in raw_suggestions:
        if not isinstance(item, Mapping) or set(item.keys()) != _SUGGESTION_KEYS:
            raise ValueError("некорректная запись в input_context.nonbinding_suggestions")
        if any(not isinstance(item[k], str) for k in _SUGGESTION_KEYS):
            raise ValueError("все поля nonbinding_suggestions должны быть строками")
        ConceptType(item["type"])
        suggestions.append({k: item[k] for k in ("source_text", "type", "translation", "note")})

    return {
        "schema_version": 1,
        "sequential_analysis": unfrozen["sequential_analysis"],
        "parallel_batch": unfrozen["parallel_batch"],
        "seed_active": unfrozen["seed_active"],
        "user_decisions": decisions,
        "nonbinding_suggestions": suggestions,
        "prompts_version": unfrozen["prompts_version"],
    }


def _omit_absent_offsets(anchor: dict) -> None:
    if anchor.get("start") is None:
        anchor.pop("start", None)
    if anchor.get("end") is None:
        anchor.pop("end", None)


def _omit_absent_anchor_offsets(response: dict) -> None:
    for fact in response.get("context_facts") or ():
        for anchor in fact.get("anchors") or ():
            _omit_absent_offsets(anchor)
        subject = fact.get("subject_anchor")
        if subject:
            _omit_absent_offsets(subject)
    for scene in response.get("scenes") or ():
        for visual in scene.get("visual_facts") or ():
            anchor = visual.get("anchor")
            if anchor:
                _omit_absent_offsets(anchor)
    for difficulty in response.get("difficulties") or ():
        for anchor in difficulty.get("anchors") or ():
            _omit_absent_offsets(anchor)


def encode_pass(payload: NarrativeAnalysisResponse | PedagogicalAnalysisResponse) -> dict:
    response = asdict(payload)
    # Do not write empty new fields: a pass saved before them keeps the same checksum.
    for candidate in response.get("glossary_candidates", ()):
        if candidate.get("description") == "":
            candidate.pop("description", None)
    for item in response.get("treatment_plan", ()):
        if not item.get("segments"):
            item.pop("segments", None)
    _omit_absent_anchor_offsets(response)
    return {"schema_version": 1, "response": response}


def _cached_plan_item(raw: Mapping) -> TreatmentPlanItemResponse:
    return TreatmentPlanItemResponse(
        raw["source_block_id"],
        BlockTreatment(raw["treatment"]),
        tuple(
            SegmentTreatmentResponse(segment["index"], BlockTreatment(segment["treatment"]))
            for segment in raw.get("segments", ())
        ),
    )


@overload
def decode_pass(
    payload: Mapping, *, pass_kind: Literal[AnalysisPassKind.NARRATIVE]
) -> NarrativeAnalysisResponse: ...


@overload
def decode_pass(
    payload: Mapping, *, pass_kind: Literal[AnalysisPassKind.PEDAGOGICAL]
) -> PedagogicalAnalysisResponse: ...


@overload
def decode_pass(
    payload: Mapping, *, pass_kind: Literal[AnalysisPassKind.ILLUSTRATION_PREP]
) -> ChapterIllustrationPlan: ...


@overload
def decode_pass(
    payload: Mapping, *, pass_kind: AnalysisPassKind
) -> NarrativeAnalysisResponse | PedagogicalAnalysisResponse | ChapterIllustrationPlan: ...


def encode_illustration_pass(plan: ChapterIllustrationPlan) -> dict[str, Any]:
    """Encode a `ChapterIllustrationPlan` into the versioned `AnalysisPassResult` mirror format."""
    return {"schema_version": 1, "response": plan.to_dict()}


def decode_illustration_pass(payload: Mapping[str, Any]) -> ChapterIllustrationPlan:
    """Decode a persisted `ILLUSTRATION_PREP` analysis pass payload."""
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise ValueError("неизвестная версия сохранённого прохода иллюстраций")
    raw_response = payload.get("response")
    if not isinstance(raw_response, Mapping):
        raise ValueError("отсутствует тело прохода иллюстраций")
    return ChapterIllustrationPlan.from_dict(thaw(raw_response))


def decode_pass(
    payload: Mapping,
    *,
    pass_kind: AnalysisPassKind,
) -> NarrativeAnalysisResponse | PedagogicalAnalysisResponse | ChapterIllustrationPlan:
    kind = AnalysisPassKind(pass_kind)
    if kind is AnalysisPassKind.ILLUSTRATION_PREP:
        return decode_illustration_pass(payload)
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise ValueError("неизвестная версия сохранённого прохода анализа")
    body = thaw(payload["response"])

    def anchor(value: Mapping[str, Any]) -> QuotedSourceAnchorResponse:
        return QuotedSourceAnchorResponse(**value)

    def merged(base: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
        return {**base, **extra}

    if kind is AnalysisPassKind.NARRATIVE:
        candidates = []
        for raw in body["glossary_candidates"]:
            assessment = raw["name_translation_assessment"]
            candidates.append(
                GlossaryCandidateResponse(
                    **merged(
                        raw,
                        type=ConceptType(raw["type"]),
                        name_translation_assessment=(
                            NameTranslationAssessment(**assessment) if assessment else None
                        ),
                    )
                )
            )
        facts = tuple(
            QuotedContextFactResponse(
                **merged(
                    f,
                    anchors=tuple(anchor(a) for a in f["anchors"]),
                    subject_anchor=anchor(f["subject_anchor"]),
                )
            )
            for f in body["context_facts"]
        )
        scenes = tuple(
            QuotedSceneResponse(
                **merged(
                    s,
                    visual_facts=tuple(
                        QuotedSceneVisualFactResponse(**merged(f, anchor=anchor(f["anchor"])))
                        for f in s["visual_facts"]
                    ),
                )
            )
            for s in body["scenes"]
        )
        return NarrativeAnalysisResponse(
            body["translation_instructions"],
            body["chapter_summary"],
            tuple(candidates),
            facts,
            scenes,
        )
    return PedagogicalAnalysisResponse(
        treatment_plan=tuple(_cached_plan_item(i) for i in body["treatment_plan"]),
        before_you_read=tuple(BeforeYouReadEntryResponse(**e) for e in body["before_you_read"]),
        difficulties=tuple(
            QuotedDifficultyResponse(
                **merged(
                    d,
                    anchors=tuple(anchor(a) for a in d["anchors"]),
                    required_points=tuple(RequiredPoint(**p) for p in d["required_points"]),
                )
            )
            for d in body["difficulties"]
        ),
        before_you_read_evidence=tuple(
            HelpEvidenceResponse(**e) for e in body["before_you_read_evidence"]
        ),
        reader_chapter_summary=body["reader_chapter_summary"],
    )


def pedagogical_response(
    analysis: ChapterAnalysis, *, reader_level: str
) -> PedagogicalAnalysisResponse:
    """Keep existing fragment merge semantics, including deduplicated BYR evidence."""
    plan = analysis.content_plan
    if plan is None:
        raise ValueError("учебный проход требует содержательный план")
    evidence = []
    entries_with_index = [
        (i, {"expression": e.expression, "explanation": e.explanation})
        for i, e in enumerate(analysis.before_you_read)
    ]
    index_cache: dict[tuple[str, str], dict[str, int]] = {}
    for item in plan.before_you_read_evidence:
        by_key = index_cache.get((item.artifact_kind, item.block_id))
        if by_key is None:
            by_key = {
                artifact_key(item.artifact_kind, item.block_id, payload): i
                for i, payload in entries_with_index
            }
            index_cache[(item.artifact_kind, item.block_id)] = by_key
        index = by_key[item.artifact_key]
        evidence.append(
            HelpEvidenceResponse(
                item.difficulty_id,
                item.point_id,
                item.artifact_kind,
                item.block_id,
                index,
                item.body_excerpt,
            )
        )
    return PedagogicalAnalysisResponse(
        treatment_plan=tuple(
            TreatmentPlanItemResponse(
                i.block_id,
                i.treatment,
                tuple(
                    SegmentTreatmentResponse(index, treatment)
                    for index, treatment in enumerate(i.segment_treatments, start=1)
                ),
            )
            for i in analysis.treatment_plan.items
        ),
        before_you_read=tuple(
            BeforeYouReadEntryResponse(e.expression, e.explanation)
            for e in analysis.before_you_read
        ),
        difficulties=tuple(
            QuotedDifficultyResponse(
                d.id,
                tuple(
                    QuotedSourceAnchorResponse(a.block_id, a.quote, a.start, a.end)
                    for a in d.anchors
                ),
                d.owner_block_id,
                d.kind,
                d.priority,
                d.help_kind,
                d.required_points,
            )
            for d in plan.difficulties
        ),
        before_you_read_evidence=tuple(evidence),
        reader_chapter_summary=analysis.chapter_summary if reader_level == "a2" else None,
    )
