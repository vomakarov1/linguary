"""Opt-in repeated model evaluation; its report requires a human review."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from linguary.adapters.config import load_settings
from linguary.adapters.llm.openai_compatible import _body
from linguary.adapters.llm.payloads import parse_payload
from linguary.application.context_builder import snapshot_prompt_templates
from linguary.application.ports.llm import LLMRequest, LLMResponseKind
from linguary.domain import ConceptType, glossary_key
from linguary.domain.glossary import compute_requires_review
from linguary.domain.name_translation import NameTranslationStrategy


FIXTURE = Path(__file__).parents[2] / 'fixtures/llm/name-translation-strategy-v1.json'


@pytest.mark.slow
async def test_live_fixed_name_strategy_set(tmp_path):
    if os.getenv('LINGUARY_RUN_NAME_STRATEGY_EVAL') != '1':
        pytest.skip('set LINGUARY_RUN_NAME_STRATEGY_EVAL=1 for the live evaluation')

    fixed = json.loads(FIXTURE.read_text())
    assert fixed['version'] == 1
    settings = load_settings().llm
    settings.ensure_ready()
    report_path = Path(os.environ.get('LINGUARY_NAME_STRATEGY_EVAL_REPORT', str(tmp_path / 'name-translation-evaluation.json')))
    report = {'fixture_version': fixed['version'], 'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
              'policy_version': 1, 'provider': settings.provider, 'configured_model': settings.model,
              'parameters': {'temperature': settings.temperature, 'max_output_tokens': settings.max_output_tokens},
              'human_review_completed': False, 'runs': []}
    failures = []
    try:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds, trust_env=settings.proxy_mode == 'auto') as client:
            for repetition in range(1, 4):
                for case in fixed['cases']:
                    mode = NameTranslationStrategy(case['mode'])
                    templates = snapshot_prompt_templates(
                        source_language=case['languages'][0], target_language=case['languages'][1],
                        reader_level='b2', name_translation_policy={'version': 1, 'strategy': mode.value},
                    )
                    user_text = (
                        f"Оцени главу и включи кандидата {case['source']!r} в глоссарий. "
                        f"Частота в книге: {case['mentions']}; главы с кандидатом: {case['chapters']}. "
                        f"Контекст: {case['context']}\n"
                        f"Текстовый блок b1: {case['context']}\n"
                        + (f"Принятое решение пользователя: {case['source']} → {case['approved_form']}."
                           if case.get('approved_form') else 'Принятых решений пользователя нет.')
                    )
                    request = LLMRequest(
                        kind=LLMResponseKind.CHAPTER_ANALYSIS, instructions=templates['analysis'],
                        input_text=user_text, name_translation_policy_version=1,
                    )
                    body = _body(request, settings=settings)
                    prompt_bytes = json.dumps(body['messages'], ensure_ascii=False, sort_keys=True).encode()
                    row = {
                        'repetition': repetition, 'id': case['id'], 'strategy': mode.value,
                        'policy_version': 1, 'prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
                        'exact_input': body, 'expected_forms': case['expected_forms'],
                        'expected_requires_review': case['expected_requires_review'],
                    }
                    try:
                        response = await client.post(
                            settings.base_url.rstrip('/') + '/chat/completions', json=body,
                            headers={'Authorization': f'Bearer {settings.api_key}'},
                        )
                        row['http_status'] = response.status_code
                        row['raw_response'] = response.text
                        response.raise_for_status()
                        document = response.json()
                        row['actual_model'] = document.get('model')
                        raw_content = document['choices'][0]['message']['content']
                        row['raw_content'] = raw_content
                        payload = parse_payload(LLMResponseKind.CHAPTER_ANALYSIS, raw_content,
                                                name_translation_policy_version=1)
                        candidate = next((item for item in payload.glossary_candidates
                                          if glossary_key(item.source_text) == glossary_key(case['source'])), None)
                        if candidate is None:
                            raise AssertionError('model omitted evaluated name')
                        assessment = candidate.name_translation_assessment
                        eligible = bool(assessment and assessment.supports(mode) and
                                        candidate.confidence >= fixed['confidence_threshold'] and
                                        not candidate.model_flagged_review)
                        actual_review = compute_requires_review(
                            type=ConceptType.PERSON, confidence=candidate.confidence,
                            model_flagged_review=candidate.model_flagged_review or bool(assessment and assessment.review_reasons),
                            confidence_threshold=fixed['confidence_threshold'],
                            is_frequent=case['mentions'] >= fixed['min_mentions'] or case['chapters'] >= fixed['min_chapters'],
                            exempt_type_review=eligible,
                        )
                        row['actual'] = {'form': candidate.suggested_translation,
                                         'requires_review': actual_review,
                                         'assessment': None if assessment is None else {
                                             'applied_method': assessment.applied_method.value,
                                             'eligible_without_review': assessment.eligible_without_review,
                                             'review_reasons': [reason.value for reason in assessment.review_reasons],
                                             'rationale': assessment.rationale}}
                        row['passed'] = (candidate.suggested_translation in case['expected_forms']
                                         and actual_review is case['expected_requires_review'])
                    except (httpx.HTTPError, ValueError, KeyError, IndexError, AssertionError) as error:
                        row['error'] = f'{type(error).__name__}: {error}'
                        row['passed'] = False
                    report['runs'].append(row)
                    if not row['passed']:
                        failures.append(f"{repetition}:{case['id']}")
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    assert not failures, f"Live evaluation failed for {', '.join(failures)}; inspect {report_path} and review it manually"
