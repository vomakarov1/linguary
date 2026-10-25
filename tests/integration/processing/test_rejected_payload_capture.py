from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json

import httpx
import pytest
from linguary.adapters.config import LLMSettings
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.adapters.llm.openai_compatible import OpenAICompatibleLLMClient
from linguary.application.diagnostics import Diagnostics, DiagnosticsPolicy
from linguary.application.llm_rejections import record_rejection, finalize_response
from linguary.application.ports.llm import LLMRequest, LLMResponseFormatError, LLMConfigurationError
from linguary.application.retry_policy import ResultValidationError
from linguary.domain import LLMCallStage, TaskStatus
from processing_scenario import store_processing_book, translated_reply
from test_book_conditions import make_app, store_imported, reply


def document(content):
    return {'model': 'actual-model', 'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 34, 'cost': 0.007}}


def request():
    return LLMRequest.for_chapter_analysis(instructions='Analyse synthetic story.', input_text='Synthetic source.').with_logging(
        book_id='book-1', stage=LLMCallStage.CHAPTER_ANALYSIS)


@pytest.mark.parametrize('content,rule', [('{broken private-key', 'response_parse'), ('{}', 'response_schema')])
async def test_json_and_schema_failure_capture_with_usage_cost(unit_of_work, content, rule):
    store_processing_book(unit_of_work)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document(content)))) as http:
        inner = OpenAICompatibleLLMClient(settings=LLMSettings(model='assigned', api_key='private-key'), http_client=http)
        client = LoggingLLMClient(inner, unit_of_work, 'provider', 'assigned')
        with pytest.raises(LLMResponseFormatError) as raised:
            await client.generate(request())
    error = raised.value
    assert error.usage.total_tokens == 46 and error.cost == Decimal('0.007')
    assert error.violation.rule_code == rule and error.violation.scope == 'unknown'
    assert 'private-key' not in str(error)
    assert error.__cause__ is None
    with unit_of_work.transaction() as transaction:
        call, = transaction.llm_calls.list_for_book('book-1')
        record = transaction.llm_calls.diagnostic_record(call_id=call.id)
        assert call.cost == Decimal('0.007') and call.usage.total_tokens == 46
        assert record['metadata']['decision'] == 'rejected'
        assert record['body'] is not None and 'private-key' not in record['body']


async def test_domain_rejection_after_parse_and_success_without_debug(unit_of_work):
    store_processing_book(unit_of_work)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document(reply().content)))) as http:
        inner = OpenAICompatibleLLMClient(settings=LLMSettings(model='assigned', api_key='private-key'), http_client=http)
        client = LoggingLLMClient(inner, unit_of_work, 'provider', 'assigned')
        response = await client.generate(request())
        with unit_of_work.transaction() as transaction:
            assert transaction.llm_calls.response_text_for(response.call_id) is None
        record_rejection(unit_of_work, response, ResultValidationError('Нарушено покрытие'))
        accepted = await client.generate(request())
        finalize_response(unit_of_work, accepted, 'accepted')
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.response_text_for(response.call_id) == reply().content
        assert transaction.llm_calls.response_text_for(accepted.call_id) is None
        assert transaction.llm_calls.diagnostic_record(call_id=accepted.call_id)['metadata']['decision'] == 'accepted'


async def test_truncated_and_secret_redacted(unit_of_work):
    store_processing_book(unit_of_work)
    content = 'private-key Authorization: Bearer hidden https://user:pass@host/path ' + 'ж' * 100000
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document(content)))) as http:
        inner = OpenAICompatibleLLMClient(settings=LLMSettings(model='assigned', api_key='private-key'), http_client=http)
        client = LoggingLLMClient(inner, unit_of_work, 'provider', 'assigned')
        with pytest.raises(LLMResponseFormatError) as raised:
            await client.generate(request())
    assert len(raised.value.raw_content.encode()) <= 65536
    assert not any(secret in raised.value.raw_content for secret in ('private-key', 'hidden', 'user:pass'))
    with unit_of_work.transaction() as transaction:
        record = transaction.llm_calls.diagnostic_record(call_id=raised.value.call_id)
        assert record['metadata']['body_meta']['truncated']
        assert record['metadata']['body_meta']['redacted']
        assert len(record['body'].encode()) <= 65536


async def test_missing_content_marks_unavailable(unit_of_work):
    store_processing_book(unit_of_work)
    body = document(None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as http:
        client = LoggingLLMClient(OpenAICompatibleLLMClient(settings=LLMSettings(model='assigned', api_key='private-key'), http_client=http), unit_of_work, 'provider', 'assigned')
        with pytest.raises(LLMResponseFormatError) as raised:
            await client.generate(request())
    assert raised.value.cost == Decimal('0.007')
    with unit_of_work.transaction() as transaction:
        record = transaction.llm_calls.diagnostic_record(call_id=raised.value.call_id)
        assert record['body'] is None
        assert record['metadata']['body_meta']['body_status'] == 'unavailable'


async def test_result_correlates_to_accepted_attempt(tmp_path):
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    fake = FakeLLMClient([reply(), FakeReply.of_payload({'blocks': [{'source_block_id': 'b1', 'treatment': 'READ_AS_IS', 'translated_text': None, 'glossary_hints': []}], 'language_notes': []})])
    app = make_app(tmp_path, fake)
    from linguary.composition import drain_tasks
    try:
        doc = store_imported(app)
        await app.analysis.start('book-1')
        assert all(task.status == TaskStatus.SUCCEEDED for task in await drain_tasks(app))
        run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        completed = await drain_tasks(app)
        assert all(task.status == TaskStatus.SUCCEEDED for task in completed), [(task.status, task.last_error) for task in completed]
        with app.unit_of_work.transaction() as transaction:
            binding = transaction.work.get_binding('result', f'{run.id}:{doc.sections[0].blocks[0].id}')
            assert transaction.work.get_attempt(binding['attempt_id']).state == 'accepted'
            record = transaction.llm_calls.diagnostic_record(attempt_id=binding['attempt_id'])
            assert record['metadata']['decision'] == 'accepted'
            assert record['body'] is None
            assert record['metadata']['context']['run_id'] == run.id
    finally:
        await app.aclose()


async def test_idle_queue_runs_hourly_maintenance(tmp_path):
    app = make_app(tmp_path)
    calls = []
    app.runner._maintenance = lambda: calls.append(True)
    try:
        assert await app.runner.run_next() is None
        assert calls == [True]
    finally:
        await app.aclose()


async def test_domain_to_dto_failure_keeps_usage_cost(unit_of_work, monkeypatch):
    from linguary.adapters.llm import payloads
    from linguary.application.llm_dto import LLMPayloadError
    def reject_dto(_):
        raise LLMPayloadError('private-key original source must not escape')
    monkeypatch.setattr(payloads, '_to_dto', reject_dto)
    store_processing_book(unit_of_work)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document(reply().content)))) as http:
        client = LoggingLLMClient(OpenAICompatibleLLMClient(settings=LLMSettings(model='assigned', api_key='private-key'), http_client=http), unit_of_work, 'provider', 'assigned')
        with pytest.raises(LLMResponseFormatError) as raised:
            await client.generate(request())
    assert raised.value.violation.rule_code == 'response_domain'
    assert raised.value.cost == Decimal('0.007') and raised.value.usage.total_tokens == 46
    assert 'original source' not in str(raised.value) and raised.value.__cause__ is None


async def test_metadata_failure_blocks_following_generations(unit_of_work, monkeypatch):
    from linguary.adapters.db.repositories import SqlAlchemyLLMCallLogRepository
    from linguary.application.ports.llm import LLMResponse
    store_processing_book(unit_of_work)
    calls = []
    class Inner:
        async def generate(self, request):
            calls.append(request)
            return LLMResponse(object(), 'actual-provider', 'actual-model', cost=Decimal('0.007'))
    def fail(*args, **kwargs):
        raise OSError('metadata write failed')
    monkeypatch.setattr(SqlAlchemyLLMCallLogRepository, 'update_diagnostics', fail)
    for _ in range(2):
        logger = LoggingLLMClient(Inner(), unit_of_work, 'provider', 'assigned')
        with pytest.raises(LLMConfigurationError):
            await logger.generate(request())
    assert len(calls) == 1
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.list_for_book('book-1')[0].cost == Decimal('0.007')


async def test_body_failure_honestly_marks_unavailable(unit_of_work, monkeypatch):
    from linguary.adapters.db.repositories import SqlAlchemyLLMCallLogRepository
    store_processing_book(unit_of_work)
    def fail(*args, **kwargs):
        raise OSError('body write failed')
    monkeypatch.setattr(SqlAlchemyLLMCallLogRepository, 'store_diagnostic_body', fail)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=document('{broken')))) as http:
        client = LoggingLLMClient(OpenAICompatibleLLMClient(settings=LLMSettings(model='assigned', api_key='private-key'), http_client=http), unit_of_work, 'provider', 'assigned')
        with pytest.raises(LLMResponseFormatError) as raised:
            await client.generate(request())
    with unit_of_work.transaction() as transaction:
        record = transaction.llm_calls.diagnostic_record(call_id=raised.value.call_id)
        assert record['metadata']['decision'] == 'rejected'
        assert record['metadata']['body_meta']['body_status'] == 'unavailable'
        assert transaction.llm_calls.response_text_for(raised.value.call_id) is None
        assert transaction.llm_calls.list_for_book('book-1')[0].cost == Decimal('0.007')


async def test_late_result_does_not_replace_accepted_attempt(tmp_path):
    import asyncio
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    from linguary.application.ports.llm import LLMResponseKind
    from linguary.application.use_cases.approve_glossary import StartProcessingCommand
    from linguary.composition import drain_tasks
    entered, release = asyncio.Event(), asyncio.Event()
    analysis = FakeLLMClient([reply()])
    processing = FakeLLMClient.always(FakeReply.of_payload({'blocks': [{'source_block_id': 'b1', 'treatment': 'READ_AS_IS', 'translated_text': None, 'glossary_hints': []}], 'language_notes': []}))
    count = 0
    class Delayed:
        async def generate(self, request):
            nonlocal count
            if request.kind == LLMResponseKind.CHAPTER_ANALYSIS:
                return await analysis.generate(request)
            count += 1
            if count == 1:
                entered.set()
                await release.wait()
            return await processing.generate(request)
    app = make_app(tmp_path, Delayed())
    try:
        doc = store_imported(app)
        await app.analysis.start('book-1')
        await drain_tasks(app)
        run = app.start_processing()(StartProcessingCommand('book-1'))
        task, = await app.processing.start('book-1')
        late = asyncio.create_task(app.processing.run_task(task))
        await asyncio.wait_for(entered.wait(), 3)
        await app.processing.run_task(task)
        entity_id = f'{run.id}:{doc.sections[0].blocks[0].id}'
        with app.unit_of_work.transaction() as transaction:
            winner = transaction.work.get_binding('result', entity_id)['attempt_id']
        release.set()
        await late
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_binding('result', entity_id)['attempt_id'] == winner
            calls = [call for call in transaction.llm_calls.list_for_book('book-1') if call.stage == LLMCallStage.TRANSLATION_UNIT]
            assert sorted(call.diagnostic_meta['decision'] for call in calls) == ['accepted', 'stale']
            assert transaction.work.get_attempt(winner).state == 'accepted'
    finally:
        release.set()
        await app.aclose()
