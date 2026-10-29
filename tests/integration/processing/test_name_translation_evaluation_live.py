"""Opt-in repeated model evaluation; its report requires a human review."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from urllib.request import getproxies

import httpx
import pytest

from linguary.adapters.config import load_settings
from linguary.adapters.config.settings import proxy_secrets
from linguary.adapters.llm.openai_compatible import _body, safe_detail
from linguary.adapters.llm.payloads import parse_payload
from linguary.application.context_builder import snapshot_prompt_templates
from linguary.application.glossary_review import build_glossary_review
from linguary.application.ports.llm import LLMRequest, LLMResponseKind
from linguary.domain import ChapterAnalysis, ConceptType, GlossaryCandidate, TermConcept, TermStatus, TermVariant, TreatmentPlan, glossary_key
from linguary.domain.name_translation import NameTranslationAssessment, NameTranslationEvidence, NameTranslationObservation, NameTranslationPolicy, NameTranslationStrategy


FIXTURE = Path(__file__).parents[2] / 'fixtures/llm/name-translation-strategy-v1.json'
CONDITIONS_ID = 'current'


def _review(case, payloads, fixed):
    mode = NameTranslationStrategy(case['mode'])
    analyses = []
    for number, payload in enumerate(payloads):
        source = next((item for item in payload.glossary_candidates
                       if glossary_key(item.source_text) == glossary_key(case['source'])), None)
        if source is None:
            raise AssertionError(f"model omitted evaluated name in chapter {number + 1}")
        form = case.get('chapter_forms', [source.suggested_translation] * case['chapters'])[number]
        missing = case.get('assessment') == 'missing' or case.get(
            'chapter_assessments', ['present'] * case['chapters'])[number] == 'missing'
        observations = tuple(NameTranslationObservation(
            item.get('form', form), item.get('confidence', source.confidence),
            case.get('flagged', False) or source.model_flagged_review,
            None if missing else source.name_translation_assessment,
        ) for item in case.get('observations', [{}]))
        analysis_id = f"analysis-{number}"
        candidate = GlossaryCandidate.create(
            id=f"candidate-{number}", analysis_id=analysis_id, source_text=case['source'],
            type=source.type, suggested_translation=form, confidence=source.confidence,
            model_flagged_review=case.get('flagged', False) or source.model_flagged_review,
            confidence_threshold=fixed['confidence_threshold'], note=source.note,
        )
        candidate = replace(candidate, name_translation_evidence=NameTranslationEvidence(
            1, case.get('chapter_conditions_ids', [case.get('conditions_id', CONDITIONS_ID)] * case['chapters'])[number],
            1, mode, observations,
        ))
        analyses.append(ChapterAnalysis.create(
            id=analysis_id, section_id=f"section-{number}", treatment_plan=TreatmentPlan(section_id=f"section-{number}"),
            translation_instructions='', chapter_summary='', expected_block_ids=(), glossary_candidates=(candidate,),
        ))
    decisions = ()
    if approved := case.get('approved_form'):
        decisions = (TermConcept(
            id='user-decision', book_id='book-eval', type=analyses[0].glossary_candidates[0].type,
            canonical_source_text=case['source'], status=TermStatus.APPROVED,
            variants=(TermVariant('user-decision', case['languages'][1], approved, TermStatus.APPROVED),),
        ),)
    review = build_glossary_review(
        book_id='book-eval', analyses=analyses, confidence_threshold=fixed['confidence_threshold'],
        count_mentions=lambda _: case['mentions'], min_mentions=fixed['min_mentions'],
        min_chapters=fixed['min_chapters'], decisions=decisions,
        name_translation_policy=NameTranslationPolicy(1, mode), conditions_id=CONDITIONS_ID,
    )
    return review.item_for(case['source'])


def _report_secrets(settings):
    urls = (settings.proxy_url,) if settings.proxy_mode == 'custom' else (
        tuple(value for key, value in getproxies().items() if key in ('http', 'https', 'all'))
        if settings.proxy_mode == 'auto' else ()
    )
    return tuple(secret for url in urls for secret in proxy_secrets(url))


def _report_text(report, settings, secrets):
    return safe_detail(json.dumps(report, ensure_ascii=False, indent=2), api_key=settings.api_key, secrets=secrets)


@pytest.mark.parametrize('case_id', [
    'contradictory-chapters', 'low-fragment', 'fragment-conflict',
    'one-stale-assessment', 'one-missing-assessment', 'user-decision',
])
def test_live_review_uses_all_fixture_observations(case_id):
    fixed = json.loads(FIXTURE.read_text())
    case = next(item for item in fixed['cases'] if item['id'] == case_id)
    assessment = NameTranslationAssessment('sound', True, (), 'Однозначное имя.')
    form = case.get('proposed_form', case['expected_forms'][0])
    response = SimpleNamespace(source_text=case['source'], suggested_translation=form,
                               type=ConceptType.PERSON,
                               confidence=0.9, model_flagged_review=False, note='',
                               name_translation_assessment=assessment)
    payloads = [SimpleNamespace(glossary_candidates=(response,)) for _ in range(case['chapters'])]
    item = _review(case, payloads, fixed)
    assert item.requires_review is case['expected_requires_review']
    if case_id == 'user-decision':
        assert item.decision.variant_for('ru').text == case['approved_form']


def test_live_report_redacts_key_and_proxy_credentials():
    settings = SimpleNamespace(api_key='sk-evaluation-secret', proxy_mode='custom',
                               proxy_url='http://proxy-user:proxy%40password@proxy.example:8080')
    report = {'raw_response': 'sk-evaluation-secret proxy-user proxy@password proxy%40password',
              'error': 'http://proxy-user:proxy%40password@proxy.example:8080'}
    output = _report_text(report, settings, _report_secrets(settings))
    for secret in ('sk-evaluation-secret', 'proxy-user', 'proxy@password', 'proxy%40password'):
        assert secret not in output
    assert '[REDACTED]' in output


@pytest.mark.slow
async def test_live_fixed_name_strategy_set(tmp_path):
    if os.getenv('LINGUARY_RUN_NAME_STRATEGY_EVAL') != '1':
        pytest.skip('set LINGUARY_RUN_NAME_STRATEGY_EVAL=1 for the live evaluation')

    fixed = json.loads(FIXTURE.read_text())
    assert fixed['version'] == 1
    settings = load_settings().llm
    settings.ensure_ready()
    analysis_model = settings.effective_analysis_model
    request_settings = replace(settings, model=analysis_model)
    secrets = _report_secrets(settings)
    report_path = Path(os.environ.get('LINGUARY_NAME_STRATEGY_EVAL_REPORT', str(tmp_path / 'name-translation-evaluation.json')))
    report = {'fixture_version': fixed['version'], 'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
              'policy_version': 1, 'provider': settings.provider, 'analysis_model': analysis_model,
              'parameters': {'temperature': settings.temperature, 'max_output_tokens': settings.max_output_tokens},
              'human_review_completed': False, 'runs': []}
    failures = []
    try:
        async with httpx.AsyncClient(
            timeout=settings.timeout_seconds, trust_env=settings.proxy_mode == 'auto',
            **({'proxy': settings.proxy_url} if settings.proxy_mode == 'custom' else {}),
        ) as client:
            for repetition in range(1, 4):
                for case in fixed['cases']:
                    mode = NameTranslationStrategy(case['mode'])
                    templates = snapshot_prompt_templates(
                        source_language=case['languages'][0], target_language=case['languages'][1],
                        reader_level='b2', name_translation_policy={'version': 1, 'strategy': mode.value},
                    )
                    row = {'repetition': repetition, 'id': case['id'], 'strategy': mode.value,
                           'policy_version': 1, 'expected_forms': case['expected_forms'],
                           'expected_requires_review': case['expected_requires_review'], 'chapters': []}
                    payloads = []
                    try:
                        for number in range(case['chapters']):
                            user_text = (
                                f"Оцени главу {number + 1} из {case['chapters']} и включи кандидата {case['source']!r} в глоссарий. "
                                f"Частота в книге: {case['mentions']}; главы с кандидатом: {case['chapters']}. "
                                f"Контекст: {case['context']}\nТекстовый блок b1: {case['context']}\n"
                                + (f"Принятое решение пользователя: {case['source']} → {case['approved_form']}."
                                   if case.get('approved_form') else 'Принятых решений пользователя нет.')
                            )
                            request = LLMRequest(
                                kind=LLMResponseKind.CHAPTER_ANALYSIS, instructions=templates['analysis'],
                                input_text=user_text, name_translation_policy_version=1,
                            )
                            body = _body(request, settings=request_settings)
                            prompt_bytes = json.dumps(body['messages'], ensure_ascii=False, sort_keys=True).encode()
                            chapter = {'number': number + 1, 'prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
                                       'exact_input': body}
                            row['chapters'].append(chapter)
                            response = await client.post(
                                settings.base_url.rstrip('/') + '/chat/completions', json=body,
                                headers={'Authorization': f'Bearer {settings.api_key}'},
                            )
                            chapter['http_status'] = response.status_code
                            chapter['raw_response'] = safe_detail(response.text, api_key=settings.api_key, secrets=secrets)
                            response.raise_for_status()
                            document = response.json()
                            chapter['actual_model'] = document.get('model')
                            raw_content = document['choices'][0]['message']['content']
                            chapter['raw_content'] = safe_detail(raw_content, api_key=settings.api_key, secrets=secrets)
                            payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, raw_content,
                                                    name_translation_policy_version=1)
                            payloads.append(payload)
                            model_candidate = next((item for item in payload.glossary_candidates
                                                    if glossary_key(item.source_text) == glossary_key(case['source'])), None)
                            chapter['model_form'] = None if model_candidate is None else model_candidate.suggested_translation
                            chapter['model_assessment'] = None if model_candidate is None or model_candidate.name_translation_assessment is None else asdict(model_candidate.name_translation_assessment)
                        item = _review(case, payloads, fixed)
                        decision_variant = item.decision.variant_for(case['languages'][1]) if item.decision else None
                        actual_form = decision_variant.text if decision_variant else item.suggested_translation
                        row['actual'] = {'form': actual_form, 'model_suggested_form': item.suggested_translation,
                                         'requires_review': item.requires_review,
                                         'has_translation_conflict': item.has_translation_conflict,
                                         'confidence': item.confidence,
                                         'decision': None if decision_variant is None else decision_variant.text}
                        row['passed'] = (actual_form in case['expected_forms']
                                         and item.requires_review is case['expected_requires_review'])
                    except (httpx.HTTPError, ValueError, KeyError, IndexError, AssertionError) as error:
                        row['error'] = safe_detail(f'{type(error).__name__}: {error}', api_key=settings.api_key, secrets=secrets)
                        row['passed'] = False
                    report['runs'].append(row)
                    if not row['passed']:
                        failures.append(f"{repetition}:{case['id']}")
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_report_text(report, settings, secrets))

    assert not failures, f"Live evaluation failed for {', '.join(failures)}; inspect {report_path} and review it manually"
