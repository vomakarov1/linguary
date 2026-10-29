"""Fixed independent expectations for the name policy's review boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from analysis_fakes import make_analysis_candidate, make_chapter_analysis

from linguary.application.glossary_review import build_glossary_review, glossary_snapshot
from linguary.domain import TermConcept, TermStatus, TermVariant
from linguary.domain.name_translation import (
    NameTranslationAssessment,
    NameTranslationEvidence,
    NameTranslationObservation,
    NameTranslationPolicy,
    NameTranslationStrategy,
)


DATA = json.loads((Path(__file__).parents[2] / 'fixtures/llm/name-translation-strategy-v1.json').read_text())
CONDITIONS_ID = 'current'


def _review(case, *, legacy=False):
    mode = NameTranslationStrategy(case['mode'])
    proposed = case.get('proposed_form', case['expected_forms'][0])
    analyses = []
    for number in range(case['chapters']):
        analysis_id = f"analysis-{number}"
        chapter_form = case.get('chapter_forms', [proposed] * case['chapters'])[number]
        observations = []
        for item in case.get('observations', [{}]):
            reason = case.get('reason')
            missing = case.get('assessment') == 'missing' or case.get('chapter_assessments', ['present'] * case['chapters'])[number] == 'missing'
            assessment = None if missing else NameTranslationAssessment(
                'sound' if mode is NameTranslationStrategy.SOUND or case['id'] in ('alice-meaning', 'bennet-meaning', 'opaque-meaning') else 'meaning',
                reason is None,
                (reason,) if reason else (),
                'Оценка для фиксированного сценария.',
            )
            observations.append(NameTranslationObservation(
                item.get('form', chapter_form), item.get('confidence', 0.9), case.get('flagged', False), assessment,
            ))
        candidate = make_analysis_candidate(
            source_text=case['source'], suggested_translation=chapter_form,
            analysis_id=analysis_id, confidence=0.9,
            model_flagged_review=case.get('flagged', False),
        )
        if not legacy:
            candidate = replace(candidate, name_translation_evidence=NameTranslationEvidence(
                1, case.get('chapter_conditions_ids', [case.get('conditions_id', CONDITIONS_ID)] * case['chapters'])[number], 1, mode, tuple(observations),
            ))
        analyses.append(make_chapter_analysis(
            id=analysis_id, section_id=f"section-{number}", candidates=(candidate,),
        ))
    decisions = ()
    if approved := case.get('approved_form'):
        decisions = (TermConcept(
            id='user-decision', book_id='book-eval', type=candidate.type,
            canonical_source_text=case['source'], status=TermStatus.APPROVED,
            variants=(TermVariant('user-decision', 'ru', approved, TermStatus.APPROVED),),
        ),)
    return build_glossary_review(
        book_id='book-eval', analyses=analyses, confidence_threshold=DATA['confidence_threshold'],
        count_mentions=lambda _: case['mentions'], min_mentions=DATA['min_mentions'],
        min_chapters=DATA['min_chapters'], decisions=decisions,
        name_translation_policy=None if legacy else NameTranslationPolicy(1, mode),
        conditions_id=None if legacy else CONDITIONS_ID,
    )


def test_clear_cases_reduce_type_review():
    assert DATA['version'] == 1
    for mode in ('sound', 'meaning'):
        clear = [case for case in DATA['cases'] if case['mode'] == mode and not case['expected_requires_review'] and case['mentions'] >= DATA['min_mentions'] and case['id'] != 'user-decision']
        assert clear
        assert sum(_review(case, legacy=True).items[0].requires_review for case in clear) > sum(
            _review(case).items[0].requires_review for case in clear
        )
    for case in DATA['cases']:
        item = _review(case).items[0]
        assert item.requires_review is case['expected_requires_review'], case['id']
        if approved := case.get('approved_form'):
            assert glossary_snapshot(_review(case), language='ru').entries[0].translation == approved


def test_ambiguous_wordplay_anagram_never_exempt():
    ambiguous = [case for case in DATA['cases'] if case.get('reason') in {
        'ambiguous', 'wordplay', 'anagram', 'uncertain_etymology', 'personal_name_substitution',
    }]
    assert {case['reason'] for case in ambiguous} >= {'ambiguous', 'wordplay', 'anagram', 'personal_name_substitution'}
    assert all(_review(case).items[0].requires_review for case in ambiguous)


def test_rare_doubt_rule_unchanged():
    case = next(case for case in DATA['cases'] if case['id'] == 'rare-doubt')
    assert _review(case).items[0].model_doubts
    assert not _review(case).items[0].requires_review
