"""Структурное покрытие по конечным видимым артефактам, без оценки смысла."""
from dataclasses import replace

import pytest

from linguary.application.validators.content_plan import (
    artifact_catalog, resolve_response_evidence, validate_content_coverage,
)
from linguary.application.retry_policy import ResultValidationError
from linguary.application.llm_dto import (
    HelpEvidenceResponse, TranslationResultResponse, LanguageNoteResponse,
)
from linguary.domain import BeforeYouReadEntry, BlockProcessingResult, BlockTreatment, LanguageNote, GlossaryHint
from linguary.domain.content_plan import ContentEvidence, ContentPlan, Difficulty, SourceAnchor, RequiredPoint, HelpEvidence


def plan(*, owner='block', help_kind='explanation'):
    return ContentPlan(difficulties=(Difficulty('d', (SourceAnchor(owner, 0, 4, 'Echo'),), owner, 'wordplay', 'critical', help_kind, (RequiredPoint('p', 'Explain the relation'),)),))


def result(body='Слово связывает два самостоятельных значения.', *, owner='block'):
    return BlockProcessingResult(owner, BlockTreatment.READ_AS_IS, language_notes=(LanguageNote('n', owner, 'Echo', body),))


def evidence(p, r, *, kind='language_note.note', excerpt=None):
    a = next(a for a in artifact_catalog((r,), (), (r.block_id,)) if a.kind == kind)
    return ContentEvidence(p.sha256, (HelpEvidence('d', 'p', kind, r.block_id, a.key, excerpt or a.body),))


def check(p, r, e, byr=()):
    return validate_content_coverage(p, (r,), byr, e, owner_block_ids=(r.block_id,))


def test_missing_point_rejected():
    p, r = plan(), result()
    with pytest.raises(ResultValidationError):
        check(p, r, ContentEvidence(p.sha256))


@pytest.mark.parametrize('body', ['Echo', '«Echo»', '[^1]', ' «Echo» [^1] '])
def test_quote_only_is_not_explanation(body):
    p, r = plan(), result(body)
    with pytest.raises(ResultValidationError):
        check(p, r, evidence(p, r))


def test_wrong_block_or_stale_plan_rejected():
    p, r = plan(), result()
    e = evidence(p, r)
    for invalid in [replace(e, plan_sha256='0' * 64), replace(e, evidence=(replace(e.evidence[0], block_id='foreign'),)), replace(e, evidence=(replace(e.evidence[0], point_id='foreign'),))]:
        with pytest.raises(ResultValidationError):
            check(p, r, invalid)


def test_deleted_note_does_not_count():
    p, r = plan(), result()
    with pytest.raises(ResultValidationError):
        check(p, replace(r, language_notes=()), evidence(p, r))


def test_real_byr_body_can_cover_point():
    p, r = plan(), result()
    byr = (BeforeYouReadEntry('Echo', 'Одинаковое слово связывает два значения.'),)
    a = next(a for a in artifact_catalog((r,), byr, ('block',)) if a.kind == 'before_you_read.explanation')
    e = ContentEvidence(p.sha256, (HelpEvidence('d', 'p', a.kind, 'block', a.key, a.body),))
    assert check(p, replace(r, language_notes=()), e, byr) == e.evidence
    with pytest.raises(ResultValidationError):
        check(p, r, e, ())


def test_plausible_wrong_meaning_is_not_certified():
    p, r = plan(), result('Здесь автор говорит о движении небесных тел вокруг луны.')
    e = evidence(p, r)
    assert check(p, r, e) == e.evidence
    assert e.structural_coverage == 'complete'
    assert e.semantic_correctness == 'not_evaluated'


def test_translation_is_only_allowed_for_lexical_hint():
    r = BlockProcessingResult('block', BlockTreatment.GLOSSARY_ONLY, glossary_hints=(GlossaryHint('Echo', 'эхо', 'Здесь слово связывает два значения.'),))
    p = plan()
    with pytest.raises(ResultValidationError):
        check(p, r, evidence(p, r, kind='glossary_hint.translation'))
    p = plan(help_kind='lexical_hint')
    assert check(p, r, evidence(p, r, kind='glossary_hint.translation'))
    r = replace(r, glossary_hints=(GlossaryHint('Echo', 'эхо', 'эхо'),))
    with pytest.raises(ResultValidationError):
        check(plan(), r, evidence(plan(), r, kind='glossary_hint.note'))


def test_global_note_index_is_bound_before_dedup_and_after_marker_normalization():
    p = plan(owner='second')
    first = result('Первое объяснение.', owner='first')
    second = result('Второе объяснение.', owner='second')
    payload = TranslationResultResponse(language_notes=(LanguageNoteResponse('first', 'Echo', 'Первое [^1] объяснение.'), LanguageNoteResponse('second', 'Echo', 'Второе [^1] объяснение.')), content_evidence=(HelpEvidenceResponse('d', 'p', 'language_note.note', 'second', 1, 'Второе [^1] объяснение.'),))
    e = resolve_response_evidence(p, payload, (first, second), (), owner_block_ids=('first', 'second'))
    assert e.evidence[0].body_excerpt == 'Второе объяснение.'
    assert validate_content_coverage(p, (replace(first, language_notes=()), second), (), e, owner_block_ids=('first', 'second'))


def test_empty_plan_still_checks_sha_and_result_owners():
    p = ContentPlan()
    r = result()
    assert check(p, r, ContentEvidence(p.sha256)) == ()
    with pytest.raises(ResultValidationError):
        check(p, r, ContentEvidence('0' * 64))
    with pytest.raises(ResultValidationError):
        validate_content_coverage(p, (), (), ContentEvidence(p.sha256), owner_block_ids=('block',))


async def test_post_dedup_failure_retries_and_records_rejection(monkeypatch):
    from types import SimpleNamespace
    from linguary.application.block_processing import TranslationCall
    from linguary.application.retry_policy import RetryPolicy
    from linguary.application.ports.llm import LLMRequest
    p, r = plan(), result()
    payload = TranslationResultResponse(language_notes=(LanguageNoteResponse('block', 'Echo', r.language_notes[0].note),), content_evidence=())
    request = LLMRequest.for_translation(contract_version='v2', instructions='test', input_text='test')
    context = SimpleNamespace(request=request, content_plan=p, before_you_read=(), explained_expressions=(), explained_entries=(), expectation=None, to_block_ids=lambda value:value, to_aliases=lambda text:text)
    calls, rejected = [], []
    async def generate(request):
        calls.append(request)
        return SimpleNamespace(payload=payload, truncated=False, usage=None)
    monkeypatch.setattr('linguary.application.block_processing.record_rejection', lambda *args:rejected.append(args))
    processor = TranslationCall(unit_of_work=None, llm_client=SimpleNamespace(generate=generate), retry_policy=RetryPolicy(max_attempts=2), validator=SimpleNamespace(validate=lambda *args, **kwargs:(r,)), sleep=None)
    with pytest.raises(ResultValidationError):
        await processor.request(context, unit=None, target_language='ru', note_ids=lambda:'n')
    assert len(calls) == len(rejected) == 2


def test_duplicate_byr_survives_as_real_evidence():
    from linguary.application.explained_expressions import ExplainedExpressions
    p, r = plan(), result()
    body = r.language_notes[0].note
    byr = (BeforeYouReadEntry('Echo', body),)
    a = next(a for a in artifact_catalog((r,), byr, ('block',)) if a.kind == 'before_you_read.explanation')
    e = ContentEvidence(p.sha256, (HelpEvidence('d', 'p', a.kind, 'block', a.key, body),))
    final = ExplainedExpressions(('Echo',), explanations=(('Echo', body),)).without_repeats_v2(r)
    assert final.language_notes == ()
    assert check(p, final, e, byr) == e.evidence
    with pytest.raises(ResultValidationError):
        check(p, final, evidence(p, r), byr)


def test_hint_index_bound_before_removal_of_first_hint():
    p = plan(help_kind='lexical_hint')
    r = BlockProcessingResult('block', BlockTreatment.GLOSSARY_ONLY, glossary_hints=(GlossaryHint('First', 'первый'), GlossaryHint('Echo', 'эхо')))
    payload = TranslationResultResponse(content_evidence=(HelpEvidenceResponse('d', 'p', 'glossary_hint.translation', 'block', 1, 'эхо'),))
    e = resolve_response_evidence(p, payload, (r,), (), owner_block_ids=('block',))
    assert check(p, replace(r, glossary_hints=r.glossary_hints[1:]), e)
    with pytest.raises(ResultValidationError):
        check(p, replace(r, glossary_hints=r.glossary_hints[:1]), e)
