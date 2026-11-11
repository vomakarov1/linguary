"""Versioned internal pass DTOs with stable source references for durable reuse."""
from collections.abc import Mapping
from dataclasses import asdict

from linguary.application.llm_dto import (
    BeforeYouReadEntryResponse, GlossaryCandidateResponse, HelpEvidenceResponse,
    NarrativeAnalysisResponse, PedagogicalAnalysisResponse, QuotedContextFactResponse,
    QuotedDifficultyResponse, QuotedSceneResponse, QuotedSceneVisualFactResponse,
    QuotedSourceAnchorResponse, TreatmentPlanItemResponse,
)
from linguary.domain import BlockTreatment, ChapterAnalysis, ConceptType
from linguary.domain.book_conditions import thaw
from linguary.domain.content_plan import RequiredPoint, artifact_key
from linguary.domain.name_translation import NameTranslationAssessment


def encode_pass(payload: NarrativeAnalysisResponse | PedagogicalAnalysisResponse) -> dict:
    return {'schema_version': 1, 'response': asdict(payload)}


def decode_pass(payload: Mapping, *, narrative: bool) -> NarrativeAnalysisResponse | PedagogicalAnalysisResponse:
    if type(payload.get('schema_version')) is not int or payload.get('schema_version') != 1:
        raise ValueError('неизвестная версия сохранённого прохода анализа')
    body = thaw(payload['response'])
    def anchor(value):
        return QuotedSourceAnchorResponse(**value)
    if narrative:
        candidates = []
        for raw in body['glossary_candidates']:
            assessment = raw['name_translation_assessment']
            candidates.append(GlossaryCandidateResponse(**{**raw, 'type': ConceptType(raw['type']),
                'name_translation_assessment': NameTranslationAssessment(**assessment) if assessment else None}))
        facts = tuple(QuotedContextFactResponse(**{**f,
            'anchors': tuple(anchor(a) for a in f['anchors']), 'subject_anchor': anchor(f['subject_anchor'])})
            for f in body['context_facts'])
        scenes = tuple(QuotedSceneResponse(**{**s, 'visual_facts': tuple(
            QuotedSceneVisualFactResponse(**{**f, 'anchor': anchor(f['anchor'])}) for f in s['visual_facts'])})
            for s in body['scenes'])
        return NarrativeAnalysisResponse(body['translation_instructions'], body['chapter_summary'],
            tuple(candidates), facts, scenes)
    return PedagogicalAnalysisResponse(
        treatment_plan=tuple(TreatmentPlanItemResponse(i['source_block_id'], BlockTreatment(i['treatment']))
                             for i in body['treatment_plan']),
        before_you_read=tuple(BeforeYouReadEntryResponse(**e) for e in body['before_you_read']),
        difficulties=tuple(QuotedDifficultyResponse(**{**d,
            'anchors': tuple(anchor(a) for a in d['anchors']),
            'required_points': tuple(RequiredPoint(**p) for p in d['required_points'])}) for d in body['difficulties']),
        before_you_read_evidence=tuple(HelpEvidenceResponse(**e) for e in body['before_you_read_evidence']),
        reader_chapter_summary=body['reader_chapter_summary'])


def pedagogical_response(analysis: ChapterAnalysis, *, reader_level: str) -> PedagogicalAnalysisResponse:
    """Keep existing fragment merge semantics, including deduplicated BYR evidence."""
    plan = analysis.content_plan
    if plan is None:
        raise ValueError('учебный проход требует содержательный план')
    evidence = []
    for item in plan.before_you_read_evidence:
        index = next(i for i, e in enumerate(analysis.before_you_read)
            if artifact_key(item.artifact_kind, item.block_id,
                {'expression': e.expression, 'explanation': e.explanation}) == item.artifact_key)
        evidence.append(HelpEvidenceResponse(item.difficulty_id, item.point_id, item.artifact_kind,
            item.block_id, index, item.body_excerpt))
    return PedagogicalAnalysisResponse(
        treatment_plan=tuple(TreatmentPlanItemResponse(i.block_id, i.treatment) for i in analysis.treatment_plan.items),
        before_you_read=tuple(BeforeYouReadEntryResponse(e.expression, e.explanation) for e in analysis.before_you_read),
        difficulties=tuple(QuotedDifficultyResponse(d.id,
            tuple(QuotedSourceAnchorResponse(a.block_id, a.quote) for a in d.anchors),
            d.owner_block_id, d.kind, d.priority, d.help_kind, d.required_points) for d in plan.difficulties),
        before_you_read_evidence=tuple(evidence),
        reader_chapter_summary=analysis.chapter_summary if reader_level == 'a2' else None)
