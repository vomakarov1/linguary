from uuid import uuid4
import pytest
from linguary.domain.llm_log import CallContext, Violation
from linguary.application.retry_policy import violation_of
from linguary.application.ports.llm import LLMResponseFormatError


def context(**values):
    return CallContext(conditions_id=str(uuid4()), stage='chapter_analysis', scope_id='chapter-1',
        ordered_block_ids=('block-1',), attempt_id=str(uuid4()), generation=0,
        assigned_model='fake', prompt_hash='a' * 64, **values)


def test_parse_error_has_unknown_scope():
    violation = violation_of(LLMResponseFormatError('arbitrary secret source fragment'))
    assert violation.scope == 'unknown'
    assert violation.rule_code == 'response_format'
    assert violation.block_ids == ()


def test_analysis_has_no_fabricated_run():
    assert context().run_id is None


def test_no_source_in_error_structure():
    error = ValueError('private source and provider secret')
    assert 'private' not in str(violation_of(error).to_dict())
    with pytest.raises(ValueError):
        Violation('quote_balance', 'translation_unit', 'local', (), (), 'Нарушена целостность перевода')


def test_violation_must_reference_verified_request_blocks():
    violation = Violation('quote_balance', 'translation_unit', 'local', ('foreign',), ('translated_text',), 'Нарушены кавычки')
    assert violation.verified(context()).scope == 'unknown'


def test_quote_failure_has_exact_block():
    from processing_fakes import make_block, make_unit, make_block_result, make_response, make_validator, note_ids
    from linguary.application.retry_policy import ResultValidationError
    block = make_block(order=0, text='She returned home.')
    unit = make_unit((block,))
    payload = make_response((make_block_result(block, translated_text='«Она вернулась домой.'),))
    with pytest.raises(ResultValidationError) as caught:
        make_validator().validate(payload, unit=unit, target_language='ru', note_id_factory=note_ids())
    violation = caught.value.violation
    assert violation.rule_code == 'quote_balance'
    assert violation.scope == 'local' and violation.block_ids == (block.id,)
    assert 'She returned' not in str(violation.to_dict())
