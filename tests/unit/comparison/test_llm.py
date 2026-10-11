from decimal import Decimal

import pytest

from linguary.comparison.llm import BudgetGuard, ComparisonStopped, ObservationLedger
from linguary.application.ports.llm import LLMRequest, LLMResponseFormatError


def test_budget_reserves_inflight_calls():
    guard = BudgetGuard(Decimal('1'))
    reservation = guard.reserve(Decimal('.6'))
    with pytest.raises(ComparisonStopped):
        guard.reserve(Decimal('.6'))
    guard.settle(reservation, Decimal('.2'))
    assert guard.reserve(Decimal('.6'))


def test_network_failure_then_first_response():
    ledger = ObservationLedger()
    request = LLMRequest.for_translation(instructions='translate', input_text='private')
    first = ledger.begin(request, model='model', route='route')
    ledger.finish(first, RuntimeError('offline'))
    second = ledger.begin(request, model='model', route='route')
    ledger.finish(second, LLMResponseFormatError('invalid', raw_content='private', truncated=True))
    assert first.operation_id == second.operation_id
    assert not first.first_response
    assert second.first_response and second.truncated
    assert ledger.first_responses == 1


def test_only_rejected_raw_retained_then_deleted(tmp_path):
    ledger = ObservationLedger(diagnostic_root=tmp_path, retain_rejected=True, retention_seconds=10, secrets=('SECRET',), clock=lambda: 0)
    request = LLMRequest.for_translation(instructions='secret prompt', input_text='secret input')
    item = ledger.begin(request, model='model', route='route')
    ledger.finish(item, LLMResponseFormatError('invalid', raw_content='SECRET rejected response'))
    ledger.finalize_operation(item.operation_id, accepted=False)
    path = tmp_path / f'{item.call_uid}.json'
    assert 'SECRET' not in path.read_text()
    assert 'secret prompt' not in path.read_text()
    ledger.cleanup(now=11)
    assert not path.exists()


async def test_receipt_survives_payload_parse_failure():
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'model': 'model', 'provider': 'route', 'id': 'generation', 'full_response_cache': False, 'usage': {'cost': .2}, 'choices': [{'message': {'content': '{'}, 'finish_reason': 'length'}]}))) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', api_key='SECRET'), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    item, = ledger.observations
    assert item.cost == Decimal('.2') and item.generation_id == 'generation'
    assert item.first_response and item.truncated and item.call_id is None


@pytest.mark.parametrize('changes,reason', [({'usage': {}}, 'unknown_price'), ({'full_response_cache': True}, 'unknown_or_full_response_cache'), ({'provider': 'fallback'}, 'unknown_or_changed_route')])
async def test_unknown_price_and_cache_hit_stop(changes, reason):
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    calls = []
    document = {'model': 'model', 'provider': 'route', 'id': 'generation', 'full_response_cache': False, 'usage': {'cost': .2}, 'choices': [{'message': {'content': '{'}}]}
    document.update(changes)
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=document)
    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', api_key='SECRET'), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
        request = LLMRequest.for_translation(instructions='translate', input_text='private')
        with pytest.raises(LLMResponseFormatError):
            await client.generate(request)
        with pytest.raises(ComparisonStopped):
            await client.generate(request)
    assert len(calls) == 1 and reason in ledger.violations


async def test_route_is_fixed_per_stage():
    from unittest.mock import AsyncMock
    from linguary.comparison.llm import StageLLMClient
    analysis, processing = AsyncMock(), AsyncMock()
    client = StageLLMClient(analysis=analysis, processing=processing)
    await client.generate(LLMRequest.for_translation(instructions='translate', input_text='text'))
    await client.generate(LLMRequest.for_chapter_analysis(instructions='analyze', input_text='text'))
    assert analysis.generate.await_count == processing.generate.await_count == 1


def test_successful_or_unknown_raw_is_never_saved(tmp_path):
    from linguary.application.ports.llm import LLMResponse
    ledger = ObservationLedger(diagnostic_root=tmp_path, retain_rejected=True, retention_seconds=10)
    request = LLMRequest.for_translation(instructions='translate', input_text='private')
    item = ledger.begin(request, model='model', route='route')
    ledger.finish(item, LLMResponse(payload=object(), model='model', provider='route', raw_content='successful private response'))
    ledger.finalize_operation(item.operation_id, accepted=None)
    assert item._raw is None and not list(tmp_path.iterdir())


@pytest.mark.parametrize('truncated', [False, True])
def test_invalid_and_truncated_first_response_count(truncated):
    ledger = ObservationLedger()
    request = LLMRequest.for_translation(instructions='translate', input_text='private')
    for index in range(2):
        item = ledger.begin(request, model='model', route='route')
        ledger.finish(item, LLMResponseFormatError('invalid', truncated=truncated))
        assert item.first_response == (index == 0)
    assert ledger.first_responses == 1


async def test_receipt_survives_log_repository_failure(tmp_path):
    from contextlib import contextmanager
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.adapters.llm.logging import LoggingLLMClient
    from linguary.application.ports.llm import LLMConfigurationError
    from linguary.comparison.llm import ObservedLLMClient
    from linguary.domain import LLMCallStage
    ledger = ObservationLedger(receipt_root=tmp_path)
    class BrokenUow:
        @contextmanager
        def transaction(self):
            raise RuntimeError('repository failed')
            yield
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'model': 'model', 'provider': 'route', 'id': 'paid-call', 'full_response_cache': False, 'usage': {'cost': .2}, 'choices': [{'message': {'content': '{'}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', api_key='SECRET'), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
        logger = LoggingLLMClient(inner=client, unit_of_work=BrokenUow(), provider='route', model='model', on_logged=ledger.on_logged)
        request = LLMRequest.for_translation(instructions='secret prompt', input_text='secret input').with_logging(book_id='book', stage=LLMCallStage.TRANSLATION_UNIT)
        for _ in range(2):
            with pytest.raises(LLMConfigurationError):
                await logger.generate(request)
    receipt, = tmp_path.glob('*.json')
    assert 'paid-call' in receipt.read_text()
    assert 'secret prompt' not in receipt.read_text() and 'SECRET' not in receipt.read_text()
    assert len(calls) == 1 and ledger.observations[0].cost == Decimal('.2')


def test_diagnostic_cleanup_survives_restart(tmp_path):
    ledger = ObservationLedger(diagnostic_root=tmp_path, retain_rejected=True, retention_seconds=10, clock=lambda: 0)
    item = ledger.begin(LLMRequest.for_translation(instructions='translate', input_text='private'), model='model', route='route')
    ledger.finish(item, LLMResponseFormatError('invalid', raw_content='rejected private'))
    ledger.finalize_operation(item.operation_id, accepted=False)
    ObservationLedger(diagnostic_root=tmp_path).cleanup(now=11)
    assert not list(tmp_path.iterdir())


async def test_changed_generation_parameters_block_before_network():
    from dataclasses import replace
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    from linguary.comparison.runner import _stage_client
    calls = []
    settings = LLMSettings(model='model', api_key='SECRET', temperature=.5, max_output_tokens=1000)
    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request))) as http:
        client = ObservedLLMClient(settings=replace(settings, temperature=.9), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
        with pytest.raises(ComparisonStopped):
            _stage_client(client, ledger, 'model', 'route', False, settings, client.budget, Decimal('.5'))
    assert not calls and 'generation_parameters_mismatch' in ledger.violations


def test_group_identity_survives_repair_and_changed_previous_findings():
    from dataclasses import replace
    from linguary.domain import LLMCallStage
    ledger = ObservationLedger(variant_id='variant', repeat=1)
    request = LLMRequest.for_chapter_analysis(instructions='analyze', input_text='previous findings one')
    ledger.register_context(request, book_id='book', chapter_id='chapter', block_ids=('source-block-1', 'source-block-2'))
    request = request.with_logging(book_id='book', stage=LLMCallStage.CHAPTER_ANALYSIS, task_id='task')
    first = ledger.begin(request, model='model', route='route')
    rerun = replace(request, instructions='repair', input_text='previous findings changed')
    ledger.register_context(rerun, book_id='book', chapter_id='chapter', block_ids=('source-block-1', 'source-block-2'))
    second = ledger.begin(rerun, model='model', route='route')
    assert first.operation_id == second.operation_id
    assert first.group_id == second.group_id
    assert first.chapter_id == second.chapter_id == 'chapter'


def test_observation_write_failure_clears_raw_on_finalization(tmp_path):
    from linguary.application.ports.llm import LLMResponse
    receipt_root = tmp_path / 'blocked-receipts'
    receipt_root.write_text('not a directory')
    ledger = ObservationLedger(receipt_root=receipt_root, diagnostic_root=tmp_path / 'diagnostics', retain_rejected=True, retention_seconds=10)
    item = ledger.begin(LLMRequest.for_translation(instructions='translate', input_text='private'), model='model', route='route')
    item.cost = Decimal('.2')
    with pytest.raises(ComparisonStopped):
        ledger.finish(item, LLMResponse(payload=object(), model='model', provider='route', raw_content='successful private response'))
    with pytest.raises(ComparisonStopped):
        ledger.finalize_operation(item.operation_id, accepted=None)
    assert item._raw is None and item.cost == Decimal('.2')
    assert not (tmp_path / 'diagnostics').exists()
    with pytest.raises(ComparisonStopped):
        ledger.begin(LLMRequest.for_translation(instructions='translate', input_text='next'), model='model', route='route')


@pytest.mark.parametrize('field', ['model', 'provider', 'id'])
async def test_external_receipt_metadata_redacts_secrets(tmp_path, field):
    import json
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    secret = 'synthetic-api-secret'
    document = {'model': 'model', 'provider': 'route', 'id': 'normal-id', 'full_response_cache': False, 'usage': {'cost': .2}, 'choices': [{'message': {'content': '{'}}]}
    document[field] = 'echo-' + secret
    ledger = ObservationLedger(receipt_root=tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=document))) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', api_key=secret), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    item, = ledger.observations
    assert item.received_response
    assert secret not in json.dumps(item.public())
    assert secret not in next(tmp_path.glob('*.json')).read_text()
    if field != 'id':
        assert item.generation_id == 'normal-id'
        assert 'unknown_or_changed_route' in ledger.violations


@pytest.mark.parametrize('change', ['guard', 'maximum'])
async def test_stage_client_enforces_shared_budget_before_http(change):
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    from linguary.comparison.runner import _stage_client
    ledger = ObservationLedger()
    guard = BudgetGuard(Decimal('1'))
    calls = []
    settings = LLMSettings(model='model', api_key='synthetic-key')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request))) as http:
        client = ObservedLLMClient(settings=settings, route='route', http_client=http, ledger=ledger, budget=guard if change == 'maximum' else BudgetGuard(Decimal('1')), max_call_cost=Decimal('.9') if change == 'maximum' else Decimal('.5'))
        with pytest.raises(ComparisonStopped):
            _stage_client(client, ledger, 'model', 'route', False, settings, guard, Decimal('.5'))
    assert not calls and 'budget_binding_mismatch' in ledger.violations


@pytest.mark.parametrize('policy,present', [('set', True), ('omit', False)])
async def test_explicit_temperature_policy_is_applied_before_http(policy, present):
    import json
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'model': 'model', 'provider': 'route', 'id': 'generation', 'full_response_cache': False, 'usage': {'cost': .2}, 'choices': [{'message': {'content': '{'}}]})
    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', api_key='test-only', temperature=.5, max_output_tokens=1234), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'), temperature_policy=policy)
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    assert len(requests) == 1 and ('temperature' in requests[0]) is present
    assert requests[0]['max_tokens'] == 1234
