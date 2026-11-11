"""Model response parsing and mapping to internal application DTOs."""

from __future__ import annotations

import pytest
from llm_payload_factories import analysis_payload, as_content, block_payload, translation_payload

from linguary.adapters.llm.payloads import parse_payload, schema_for
from pydantic import ValidationError
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import LLMResponseFormatError, LLMResponseKind
from linguary.domain import BlockTreatment, ConceptType
from linguary.domain.name_translation import NameTranslationMethod


def pronunciation_candidate():
    return dict(anchor=dict(block_id='b1', start=0, end=6, quote=' zorp '), form=' zorp ',
        context=dict(meaning_in_context=None, part_of_speech='verb', grammatical_form='present', is_proper_name=False),
        selection_reason='unpredictable_spelling')


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_pronunciation_candidate_schema_and_dto_roundtrip(version):
    data = dict(blocks=[dict(source_block_id='b1', treatment='READ_AS_IS', translated_text=None, glossary_hints=[])], language_notes=[], pronunciation_candidates=[pronunciation_candidate()])
    if version in (2, 3, 4):
        data['content_evidence'] = []
    if version == 4:
        data['fragment_assistances'] = []
    result = parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data),
                           contract_version=version, pronunciation_candidates_version=1)
    candidate, = result.pronunciation_candidates
    assert candidate.form == candidate.anchor.quote == ' zorp '
    assert candidate.context.meaning_in_context is None
    assert candidate.context.part_of_speech == 'verb'
    assert candidate.selection_reason == 'unpredictable_spelling'


@pytest.mark.parametrize('field', ['ipa', 'source', 'url', 'hint_id', 'variant'])
def test_pronunciation_candidate_schema_rejects_ipa_source_and_url(field):
    candidate = pronunciation_candidate() | {field: 'must not be accepted'}
    with pytest.raises(ValidationError):
        schema_for(LLMResponseKind.TRANSLATION_RESULT, pronunciation_candidates_version=1).model_validate_json(
            as_content(dict(blocks=[dict(source_block_id='b1', treatment='READ_AS_IS', translated_text=None, glossary_hints=[])], language_notes=[], pronunciation_candidates=[candidate])))


def test_pronunciation_disabled_contract_unchanged():
    data = dict(blocks=[dict(source_block_id='b1', treatment='READ_AS_IS', translated_text=None, glossary_hints=[])], language_notes=[])
    assert parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data)).pronunciation_candidates == ()
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), pronunciation_candidates_version=1)
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data | {'pronunciation_candidates': []}))


@pytest.mark.parametrize('version', [True, False, 0, 2, '1'])
def test_pronunciation_candidate_mode_rejects_unknown_version(version):
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, '{}', pronunciation_candidates_version=version)


@pytest.mark.parametrize('kind', [LLMResponseKind.CHAPTER_ANALYSIS, LLMResponseKind.QUALITY_REVIEW])
def test_pronunciation_candidate_mode_rejects_other_stages(kind):
    from linguary.application.ports.llm import LLMRequest, LLMError
    with pytest.raises(LLMResponseFormatError):
        parse_payload(kind, '{}', pronunciation_candidates_version=1)
    with pytest.raises(LLMError):
        LLMRequest(kind=kind, instructions='instruction', input_text='input', pronunciation_candidates_version=1)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_pronunciation_candidate_schema_for_each_base_version(version):
    from linguary.adapters.llm.payloads import schema_for
    from linguary.adapters.llm.json_schema import strict_json_schema
    original = strict_json_schema(schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=version))
    extended = strict_json_schema(schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=version,
                                             pronunciation_candidates_version=1))
    assert set(extended['properties']) == set(original['properties']) | {'pronunciation_candidates'}
    assert set(extended['required']) == set(original['required']) | {'pronunciation_candidates'}
    assert extended['additionalProperties'] is False


def test_pronunciation_empty_candidates_and_nullable_context():
    data = dict(blocks=[dict(source_block_id='b1', treatment='READ_AS_IS', translated_text=None, glossary_hints=[])],
                language_notes=[], pronunciation_candidates=[])
    assert parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), pronunciation_candidates_version=1).pronunciation_candidates == ()
    data['pronunciation_candidates'] = [pronunciation_candidate() | {'context': dict(
        meaning_in_context=None, part_of_speech=None, grammatical_form=None, is_proper_name=False)}]
    assert parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), pronunciation_candidates_version=1).pronunciation_candidates[0].context.grammatical_form is None


def test_missing_or_null_assessment_falls_back() -> None:
    data = analysis_payload()
    for value in ("missing", None):
        if value is not None:
            data["glossary_candidates"][0]["name_translation_assessment"] = value
        else:
            data["glossary_candidates"][0]["name_translation_assessment"] = None
        if value == "missing":
            data["glossary_candidates"][0].pop("name_translation_assessment", None)
        result = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)
        assert result.glossary_candidates[0].name_translation_assessment is None


def test_malformed_assessment_rejected() -> None:
    data = analysis_payload()
    data["glossary_candidates"][0]["name_translation_assessment"] = {
        "applied_method": "sound", "eligible_without_review": "true", "review_reasons": [], "rationale": "Звучание", "opaque_name": False
    }
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)
    data["glossary_candidates"][0]["name_translation_assessment"]["eligible_without_review"] = True
    result = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)
    assert result.glossary_candidates[0].name_translation_assessment.applied_method is NameTranslationMethod.SOUND
    data["glossary_candidates"][0]["name_translation_assessment"]["opaque_name"] = "true"
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), name_translation_policy_version=1)


def test_chapter_analysis_is_mapped_to_dto() -> None:
    """Chapter analysis response maps completely to application DTO."""
    payload = parse_payload(
        LLMResponseKind.CHAPTER_ANALYSIS, as_content(analysis_payload())
    )

    assert isinstance(payload, ChapterAnalysisResponse)
    assert payload.chapter_summary == "Mr. Bennet visits the new neighbour."
    assert payload.translation_instructions == "Сохранять иронию рассказчика."
    assert payload.planned_block_ids == ("block-1", "block-2")
    assert payload.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert payload.glossary_candidates[0].type is ConceptType.PERSON
    assert payload.glossary_candidates[0].model_flagged_review is False
    assert payload.before_you_read[0].expression == "entailed estate"


def test_translation_result_is_mapped_to_dto() -> None:
    """Block group processing response maps completely to application DTO."""
    payload = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT, as_content(translation_payload())
    )

    assert isinstance(payload, TranslationResultResponse)
    assert payload.block_ids == ("block-1", "block-2")
    assert payload.blocks[0].treatment is BlockTreatment.TRANSLATE
    assert payload.blocks[1].treatment is BlockTreatment.GLOSSARY_ONLY
    assert payload.blocks[1].translated_text is None
    assert payload.blocks[1].glossary_hints[0].original == "entailed"
    assert payload.notes_for("block-1")[0].note == "Ироническая формула зачина."
    assert payload.notes_for("block-2") == ()


def test_read_as_is_block_is_mapped_without_content() -> None:
    """A 'READ_AS_IS' block is mapped into DTO without translation or hints."""
    payload = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT,
        as_content(
            translation_payload(
                blocks=[
                    block_payload(
                        treatment="READ_AS_IS", translated_text=None, glossary_hints=[]
                    )
                ],
                language_notes=[],
            )
        ),
    )

    assert isinstance(payload, TranslationResultResponse)
    assert payload.blocks[0].translated_text is None
    assert payload.blocks[0].glossary_hints == ()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("сейчас переведу, минутку", id="not-json"),
        pytest.param('{"blocks": [{"source_block_id": "block-1"', id="truncated-json"),
        pytest.param("{}", id="missing-required-fields"),
    ],
)
def test_invalid_response_gives_format_error(content: str) -> None:
    """An unparseable response yields a format error rather than a third-party library error."""
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, content)


def test_truncated_response_is_reported_as_truncated() -> None:
    """Token limit truncation is reflected in parse error as the cause."""
    with pytest.raises(LLMResponseFormatError) as error:
        parse_payload(
            LLMResponseKind.TRANSLATION_RESULT, '{"blocks": [{"source', truncated=True
        )

    assert error.value.truncated is True
    assert "лимиту токенов" in str(error.value)


def test_format_error_does_not_expose_full_response() -> None:
    """Error message does not leak the entire response which may contain book text."""
    content = "не json, " + "текст книги " * 200

    with pytest.raises(LLMResponseFormatError) as error:
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, content)

    assert len(str(error.value)) < len(content)


def test_response_of_wrong_kind_is_rejected() -> None:
    """An analysis response to a translation request is rejected as invalid."""
    with pytest.raises(LLMResponseFormatError):
        parse_payload(
            LLMResponseKind.TRANSLATION_RESULT, as_content(analysis_payload())
        )


def test_v2_round_trip_plan_and_evidence():
    data = analysis_payload()
    data['content_plan'] = {'version': 2, 'profile': 'content-plan-v2', 'difficulties': [], 'before_you_read_evidence': []}
    result = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, as_content(data), contract_version=2)
    assert result.content_plan.version == 2
    data = translation_payload()
    data['content_evidence'] = []
    assert parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=2).content_evidence == ()


@pytest.mark.parametrize('kind,data', [(LLMResponseKind.CHAPTER_ANALYSIS, analysis_payload())])
def test_v2_missing_fields_rejected(kind, data):
    with pytest.raises(LLMResponseFormatError):
        parse_payload(kind, as_content(data), contract_version=2)


def test_unknown_contract_version_rejected():
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, '{}', contract_version=3)


@pytest.mark.parametrize('kind,text', [('lexical_hint', 'only'), ('sentence_rephrase', 'She saw only the storm.'), ('russian_explanation', 'Она видела только бурю.'), ('unresolved', None)])
def test_v4_maps_typed_fragment_assistance(kind, text):
    data = translation_payload()
    data.update(content_evidence=[], fragment_assistances=[dict(difficulty_id='d', point_ids=['p'], kind=kind, text=text)])
    payload = parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=4)
    assert payload.fragment_assistances[0].point_ids == ('p',)
    assert payload.fragment_assistances[0].text == text
    for version in (1, 2, 3):
        with pytest.raises(LLMResponseFormatError):
            parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=version)


@pytest.mark.parametrize('item', [dict(kind='unknown', text='x', point_ids=['p']), dict(kind='unresolved', text='x', point_ids=['p']), dict(kind='lexical_hint', text='', point_ids=['p']), dict(kind='lexical_hint', text='only', point_ids=['p', 'p'])])
def test_v4_rejects_invalid_fragment_assistance(item):
    data = translation_payload()
    data.update(content_evidence=[], fragment_assistances=[dict(difficulty_id='d', **item)])
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data), contract_version=4)


def pronunciation_payload(version, candidates):
    data = translation_payload(pronunciation_candidates=candidates)
    if version >= 2:
        data['content_evidence'] = []
    if version == 4:
        data['fragment_assistances'] = []
    return data


def malformed_pronunciation_candidates():
    item = pronunciation_candidate()
    return [None, 'zorp', 7, [], {}, item | {'form': 123},
            item | {'anchor': item['anchor'] | {'start': True}},
            item | {'anchor': item['anchor'] | {'end': '6'}},
            item | {'context': item['context'] | {'is_proper_name': 'false'}},
            item | {'selection_reason': 'rare'},
            {key: value for key, value in item.items() if key != 'context'}]


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_pronunciation_mixed_malformed_candidates_keep_valid_translation_and_candidate(version):
    first = pronunciation_candidate()
    last = first | {'selection_reason': 'context_dependent'}
    result = parse_payload(LLMResponseKind.TRANSLATION_RESULT,
        as_content(pronunciation_payload(version, [first, *malformed_pronunciation_candidates(), last])),
        contract_version=version, pronunciation_candidates_version=1)
    assert result.blocks[0].translated_text == translation_payload()['blocks'][0]['translated_text']
    assert result.blocks[1].glossary_hints
    assert result.language_notes
    assert [item.selection_reason for item in result.pronunciation_candidates] == ['unpredictable_spelling', 'context_dependent']


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_pronunciation_all_malformed_candidates_keep_translation(version):
    result = parse_payload(LLMResponseKind.TRANSLATION_RESULT,
        as_content(pronunciation_payload(version, malformed_pronunciation_candidates())),
        contract_version=version, pronunciation_candidates_version=1)
    assert result.blocks[0].translated_text
    assert result.pronunciation_candidates == ()


@pytest.mark.parametrize('version', [1, 2, 3, 4])
@pytest.mark.parametrize('level', ['candidate', 'anchor', 'context'])
@pytest.mark.parametrize('field', ['ipa', 'source', 'url', 'hint_id', 'variant', 'extra'])
def test_pronunciation_forbidden_fields_skip_whole_candidate(version, level, field):
    item = pronunciation_candidate()
    if level == 'candidate':
        item[field] = 'FORBIDDEN_PRIVATE'
    else:
        item[level][field] = 'FORBIDDEN_PRIVATE'
    result = parse_payload(LLMResponseKind.TRANSLATION_RESULT,
        as_content(pronunciation_payload(version, [item, pronunciation_candidate()])),
        contract_version=version, pronunciation_candidates_version=1)
    assert len(result.pronunciation_candidates) == 1
    assert 'FORBIDDEN_PRIVATE' not in repr(result)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
@pytest.mark.parametrize('bad', ['translation', 'notes', 'extra', 'evidence', 'fragments'])
def test_pronunciation_invalid_main_response_still_rejected(version, bad):
    data = pronunciation_payload(version, [pronunciation_candidate()])
    if bad == 'translation':
        data['blocks'][0]['translated_text'] = None
    elif bad == 'notes':
        data['language_notes'] = [{}]
    elif bad == 'extra':
        data['source'] = 'forbidden'
    elif bad == 'evidence':
        data['content_evidence'] = [{}]
    else:
        data['fragment_assistances'] = [{}]
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data),
            contract_version=version, pronunciation_candidates_version=1)


@pytest.mark.parametrize('content', ['[]', 'null', '{', '"text"'])
def test_pronunciation_candidate_envelope_still_required(content):
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, content, pronunciation_candidates_version=1)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
@pytest.mark.parametrize('value', [None, {}, 'zorp', 1, True, 'missing'])
def test_pronunciation_candidate_list_still_required(version, value):
    data = pronunciation_payload(version, value)
    if value == 'missing':
        del data['pronunciation_candidates']
    with pytest.raises(LLMResponseFormatError):
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(data),
            contract_version=version, pronunciation_candidates_version=1)


def test_pronunciation_candidate_dto_error_is_isolated(monkeypatch):
    from linguary.adapters.llm import payloads
    from linguary.application.llm_dto import LLMPayloadError
    original = payloads.PronunciationCandidateResponse
    def checked(**values):
        if values['selection_reason'] == 'context_dependent':
            raise LLMPayloadError('invalid candidate DTO')
        return original(**values)
    monkeypatch.setattr(payloads, 'PronunciationCandidateResponse', checked)
    result = parse_payload(LLMResponseKind.TRANSLATION_RESULT, as_content(pronunciation_payload(1,
        [pronunciation_candidate() | {'selection_reason': 'context_dependent'}, pronunciation_candidate()])),
        pronunciation_candidates_version=1)
    assert result.blocks[0].translated_text
    assert len(result.pronunciation_candidates) == 1


def test_parse_payload_validates_parsed_dict_without_reserializing(monkeypatch):
    from linguary.adapters.llm import payloads

    translation_json = as_content(
        pronunciation_payload(2, [pronunciation_candidate()])
        | {
            "content_evidence": [
                {
                    "difficulty_id": "d-1",
                    "point_id": "p-1",
                    "artifact_kind": "language_note.note",
                    "source_block_id": "b1",
                    "artifact_index": 0,
                    "body_excerpt": "Explanation excerpt",
                }
            ]
        }
    )
    analysis_data = analysis_payload() | {
        "content_plan": {
            "version": 3,
            "profile": "content-plan-v3",
            "difficulties": [],
            "before_you_read_evidence": [
                {
                    "difficulty_id": "d-1",
                    "point_id": "p-1",
                    "artifact_kind": "before_you_read.explanation",
                    "source_block_id": "b1",
                    "artifact_index": 0,
                    "body_excerpt": "BYR excerpt",
                }
            ],
            "context_facts": [],
        }
    }
    analysis_json = as_content(analysis_data)

    real_loads = payloads.json.loads
    loads_calls = 0

    def counting_loads(*args, **kwargs):
        nonlocal loads_calls
        loads_calls += 1
        return real_loads(*args, **kwargs)

    def forbidden_dumps(*args, **kwargs):
        raise AssertionError("json.dumps must not be called in parse_payload")

    monkeypatch.setattr(payloads.json, "loads", counting_loads)
    monkeypatch.setattr(payloads.json, "dumps", forbidden_dumps)

    tr_res = parse_payload(
        LLMResponseKind.TRANSLATION_RESULT,
        translation_json,
        contract_version=2,
        pronunciation_candidates_version=1,
    )
    assert loads_calls == 1
    assert len(tr_res.pronunciation_candidates) == 1
    assert len(tr_res.content_evidence) == 1

    loads_calls = 0
    an_res = parse_payload(
        LLMResponseKind.CHAPTER_ANALYSIS,
        analysis_json,
        contract_version=3,
        name_translation_policy_version=1,
    )
    assert loads_calls == 1
    assert len(an_res.content_plan.before_you_read_evidence) == 1


@pytest.mark.parametrize(
    "fence_header",
    ["```json", "```JSON", "```"],
)
def test_parse_payload_strips_markdown_code_fences(fence_header: str) -> None:
    raw_json = as_content(translation_payload())
    fenced = f"  \n{fence_header}\n{raw_json}\n```\n  "

    result = parse_payload(LLMResponseKind.TRANSLATION_RESULT, fenced)

    assert len(result.blocks) == 2
    assert result.blocks[0].source_block_id == "block-1"

    with pytest.raises(LLMResponseFormatError) as caught:
        parse_payload(LLMResponseKind.TRANSLATION_RESULT, f"{fence_header}\n{{broken\n```")
    assert caught.value.violation is not None
    assert caught.value.violation.rule_code == "response_parse"




def narrative_payload():
    return {
        'translation_instructions': 'Сохраняй сухой тон рассказчика.',
        'chapter_summary': 'Holmes reads the letter.',
        'glossary_candidates': [{'source_text': 'Baker Street', 'type': 'place',
            'suggested_translation': 'Бейкер-стрит', 'confidence': 0.95,
            'model_flagged_review': False, 'note': None}],
        'context_facts': [{'kind': 'speaker',
            'anchors': [{'block_id': 'b1', 'quote': 'Look at this'}],
            'subject_anchor': {'block_id': 'b1', 'quote': 'Look at this'},
            'value': 'Holmes', 'certainty': 'explicit'}],
        'scenes': [{'start_block_id': 'b1', 'end_block_id': 'b2',
            'summary': 'Обсуждение письма.', 'location': '221B Baker Street',
            'participants': ['Holmes', 'Watson'],
            'visual_facts': [{'anchor': {'block_id': 'b1', 'quote': ' blue envelope '},
                'subject': 'Holmes', 'detail': 'держит синий конверт'}],
            'visual_assumptions': ['утренний свет'], 'continues_previous_scene': True}],
    }


def pedagogical_payload():
    return {
        'treatment_plan': [{'source_block_id': 'b1', 'treatment': 'TRANSLATE'}],
        'before_you_read': [{'expression': 'by Jove', 'explanation': 'восклицание'}],
        'difficulties': [{'id': 'd1', 'anchors': [{'block_id': 'b1', 'quote': 'by Jove'}],
            'owner_block_id': 'b1', 'kind': 'idiom', 'priority': 'useful',
            'help_kind': 'explanation', 'required_points': [{'id': 'p1', 'description': 'восклицание'}]}],
        'before_you_read_evidence': [{'difficulty_id': 'd1', 'point_id': 'p1',
            'artifact_kind': 'before_you_read.explanation', 'source_block_id': 'b1',
            'artifact_index': 0, 'body_excerpt': 'восклицание'}],
    }


def test_parse_narrative_analysis_payload_maps_scenes_facts_and_candidates():
    from linguary.application.llm_dto import NarrativeAnalysisResponse, QuotedSceneResponse
    dto = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE, as_content(narrative_payload()))
    assert isinstance(dto, NarrativeAnalysisResponse)
    assert dto.translation_instructions == 'Сохраняй сухой тон рассказчика.'
    assert dto.chapter_summary == 'Holmes reads the letter.'
    assert dto.glossary_candidates[0].type is ConceptType.PLACE
    assert dto.glossary_candidates[0].note == ''
    assert dto.context_facts[0].id == 'cf-1'
    assert dto.context_facts[0].subject_anchor.quote == 'Look at this'
    value, = dto.scenes
    assert isinstance(value, QuotedSceneResponse)
    assert (value.start_block_id, value.end_block_id) == ('b1', 'b2')
    assert value.location == '221B Baker Street'
    assert value.participants == ('Holmes', 'Watson')
    assert value.visual_assumptions == ('утренний свет',)
    assert value.continues_previous_scene is True
    assert value.visual_facts[0].anchor.quote == ' blue envelope '
    assert value.visual_facts[0].subject == 'Holmes'
    assert value.visual_facts[0].detail == 'держит синий конверт'


def test_parse_narrative_analysis_preserves_name_assessment_and_missing_fallback():
    data = narrative_payload()
    data['glossary_candidates'][0]['name_translation_assessment'] = {
        'applied_method': 'sound', 'eligible_without_review': True,
        'review_reasons': [], 'rationale': 'Звучание', 'opaque_name': False,
    }
    dto = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE, as_content(data), name_translation_policy_version=1)
    assert dto.glossary_candidates[0].name_translation_assessment.applied_method is NameTranslationMethod.SOUND
    data['glossary_candidates'][0].pop('name_translation_assessment')
    dto = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS_NARRATIVE, as_content(data), name_translation_policy_version=1)
    assert dto.glossary_candidates[0].name_translation_assessment is None


def test_parse_pedagogical_analysis_payload_with_and_without_a2_summary():
    from linguary.application.llm_dto import PedagogicalAnalysisResponse
    data = pedagogical_payload()
    dto = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL, as_content(data))
    assert isinstance(dto, PedagogicalAnalysisResponse)
    assert dto.reader_chapter_summary is None
    assert dto.treatment_plan[0].treatment is BlockTreatment.TRANSLATE
    assert dto.before_you_read[0].expression == 'by Jove'
    assert dto.difficulties[0].anchors[0].quote == 'by Jove'
    assert dto.difficulties[0].required_points[0].description == 'восклицание'
    assert dto.before_you_read_evidence[0].body_excerpt == 'восклицание'
    summary = 'Holmes reads a letter.\n\nХолмс читает письмо.'
    dto_a2 = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS_PEDAGOGICAL,
        as_content(data | {'reader_chapter_summary': summary}), contract_version=2)
    assert dto_a2.reader_chapter_summary == summary


def test_quoted_scene_and_staged_dtos_normalize_sequences():
    from linguary.application.llm_dto import (
        QuotedSceneResponse, QuotedContentPlanResponse, NarrativeAnalysisResponse, PedagogicalAnalysisResponse,
    )
    value = QuotedSceneResponse('b1', 'b2', 'A scene.', participants=['Holmes'],
                               visual_facts=[], visual_assumptions=['daylight'])
    assert value.participants == ('Holmes',)
    assert value.visual_facts == ()
    assert value.visual_assumptions == ('daylight',)
    assert value.continues_previous_scene is False
    assert QuotedContentPlanResponse(scenes=[value]).scenes == (value,)
    assert NarrativeAnalysisResponse('Tone.', 'Summary.', scenes=[value]).scenes == (value,)
    assert PedagogicalAnalysisResponse(difficulties=[]).difficulties == ()
