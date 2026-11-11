"""Analysis totals include old calls, both staged passes, retries, and unknown amounts."""
from datetime import UTC, datetime
from decimal import Decimal

from linguary.application.book_overview import _analysis_call_statistics
from linguary.domain import ErrorType, LLMCallLog, LLMCallStage, TaskError, TokenUsage


def call(identity, stage, *, success=True, usage=None, cost=None):
    return LLMCallLog(identity, 'book', stage, 'provider', 'model', datetime.now(UTC),
        100, 1, success, usage=usage, cost=cost,
        error=None if success else TaskError(ErrorType.TRANSIENT, 'Request failed'))


def test_analysis_totals_include_failed_attempts_and_exclude_processing():
    stats = _analysis_call_statistics([
        call('old', LLMCallStage.CHAPTER_ANALYSIS, usage=TokenUsage(10, 5), cost=Decimal('.01')),
        call('narrative', LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
             usage=TokenUsage(20, 7), cost=Decimal('.02')),
        call('failed-pedagogy', LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL, success=False),
        call('pedagogy', LLMCallStage.CHAPTER_ANALYSIS_PEDAGOGICAL,
             usage=TokenUsage(30, 9), cost=Decimal('.03')),
        call('processing', LLMCallStage.TRANSLATION_UNIT,
             usage=TokenUsage(1000, 1000), cost=Decimal('1')),
    ])
    assert stats['calls'] == 4
    assert stats['failed_calls'] == 1
    assert stats['cost_usd'] is None
    assert stats['known_spent_usd'] == '0.06'
    assert stats['unknown_calls'] == 1
    assert stats['prompt_tokens'] is stats['completion_tokens'] is stats['total_tokens'] is None
    assert stats['passes']['single']['prompt_tokens'] == 10
    assert stats['passes']['narrative']['total_tokens'] == 27
    assert stats['passes']['pedagogical']['calls'] == 2
    assert stats['passes']['pedagogical']['failed_calls'] == 1
    assert stats['passes']['pedagogical']['cost_usd'] is None


def test_analysis_totals_preserve_known_zero_price_and_empty_passes():
    stats = _analysis_call_statistics([
        call('free', LLMCallStage.CHAPTER_ANALYSIS_NARRATIVE,
             usage=TokenUsage(2, 3), cost=Decimal('0'))])
    assert stats['cost_usd'] == '0'
    assert stats['total_tokens'] == 5
    assert stats['passes']['pedagogical']['calls'] == 0
    assert stats['passes']['pedagogical']['unknown_calls'] == 0
