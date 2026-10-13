from decimal import Decimal

import pytest

from linguary.comparison.llm import BudgetGuard, ComparisonStopped, ObservationLedger
from linguary.application.ports.llm import LLMRequest, LLMResponseFormatError
from linguary.comparison.protocol import CostBounds, RouteMetadata

BOUNDS = CostBounds(max_request_bytes=100000, max_input_tokens=100000, input_price_per_mtok=1, output_price_per_mtok=1, output_includes_reasoning=True, input_bound_method="synthetic", evidence="MockTransport only")
METADATA = RouteMetadata(provider_names=("route",), model_versions=("model",))


@pytest.mark.parametrize('status,compressed', [(400, True), (403, True), (400, False), (429, True)])
async def test_safe_http_error_preserves_status_and_stops_contract_rejection(tmp_path, status, compressed):
    import gzip
    import json
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.application.ports.llm import LLMConfigurationError, LLMProviderError, LLMTransportError
    from linguary.comparison.llm import ObservedLLMClient

    calls = []
    private_body = b'PRIVATE provider error SECRET'
    content = gzip.compress(private_body) if compressed else private_body
    headers = {'Content-Length': str(len(content)), 'Retry-After': '7'}
    if compressed:
        headers['Content-Encoding'] = 'gzip'

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=content, headers=headers)

    ledger = ObservationLedger(receipt_root=tmp_path)
    guard = BudgetGuard(Decimal('1'))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', max_output_tokens=8000, api_key='SECRET'),
            route='route', http_client=http, ledger=ledger, budget=guard, max_call_cost=Decimal('.5'),
            cost_bounds=BOUNDS, route_metadata=METADATA)
        request = LLMRequest.for_translation(instructions='translate', input_text='private')
        error_type = {400: LLMProviderError, 403: LLMConfigurationError, 429: LLMTransportError}[status]
        with pytest.raises(error_type, match=str(status)) as caught:
            await client.generate(request)
        assert 'PRIVATE' not in str(caught.value) and 'SECRET' not in str(caught.value)
        assert 'provider request failed' in str(caught.value)
        if status == 429:
            assert caught.value.retry_after_seconds == 7
            assert ledger.violations == []
        else:
            assert ledger.violations == ['provider_rejected_contract']
            with pytest.raises(ComparisonStopped):
                await client.generate(request)

    assert len(calls) == guard.attempts == 1
    assert guard.spent == 0 and sum(guard.uncertain.values()) == Decimal('.5')
    stored = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert stored['submitted'] is True and stored['received_response'] is False
    assert stored['failure_category'] == 'no_response' and stored['cost'] is None
    assert 'PRIVATE' not in json.dumps(stored) and 'SECRET' not in json.dumps(stored)


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
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'model': 'model', 'provider': 'route', 'id': 'generation', 'full_response_cache': False, 'usage': {'cost': .2, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'message': {'content': '{'}, 'finish_reason': 'length'}]}))) as http:
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=LLMSettings(max_output_tokens=8000, model='model', api_key='SECRET'), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    item, = ledger.observations
    assert item.cost == Decimal('.2') and item.generation_id == 'generation'
    assert item.first_response and item.truncated and item.call_id is None


@pytest.mark.parametrize('changes,reason', [({'usage': {}}, 'unknown_price'), ({'cache_hit': True}, 'unknown_or_full_response_cache'), ({'provider': 'fallback'}, 'unknown_or_changed_route')])
async def test_unknown_price_and_cache_hit_stop(changes, reason):
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    calls = []
    document = {'model': 'model', 'provider': 'route', 'id': 'generation', 'full_response_cache': False, 'usage': {'cost': .2, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'message': {'content': '{'}}]}
    document.update(changes)
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=document, headers={'X-OpenRouter-Cache-Status': 'HIT' if changes.get('cache_hit') else 'MISS'})
    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=LLMSettings(max_output_tokens=8000, model='model', api_key='SECRET'), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
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
        return httpx.Response(200, json={'model': 'model', 'provider': 'route', 'id': 'paid-call', 'full_response_cache': False, 'usage': {'cost': .2, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'message': {'content': '{'}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=LLMSettings(max_output_tokens=8000, model='model', api_key='SECRET'), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
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
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=replace(settings, temperature=.9), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
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
    document = {'model': 'model', 'provider': 'route', 'id': 'normal-id', 'full_response_cache': False, 'usage': {'cost': .2, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'message': {'content': '{'}}]}
    document[field] = 'echo-' + secret
    ledger = ObservationLedger(receipt_root=tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=document))) as http:
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=LLMSettings(max_output_tokens=8000, model='model', api_key=secret), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'))
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
    settings = LLMSettings(max_output_tokens=8000, model='model', api_key='synthetic-key')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request))) as http:
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=settings, route='route', http_client=http, ledger=ledger, budget=guard if change == 'maximum' else BudgetGuard(Decimal('1')), max_call_cost=Decimal('.9') if change == 'maximum' else Decimal('.5'))
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
        return httpx.Response(200, json={'model': 'model', 'provider': 'route', 'id': 'generation', 'full_response_cache': False, 'usage': {'cost': .2, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'message': {'content': '{'}}]})
    ledger = ObservationLedger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(cost_bounds=BOUNDS, route_metadata=METADATA, settings=LLMSettings(model='model', api_key='test-only', temperature=.5, max_output_tokens=1234), route='route', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'), temperature_policy=policy)
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    assert len(requests) == 1 and ('temperature' in requests[0]) is present
    assert requests[0]['max_tokens'] == 1234


@pytest.mark.parametrize('change', ['bytes', 'tariff', 'unverified', 'missing'])
async def test_unproven_or_excessive_cost_bounds_send_zero_requests(change):
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    bounds = {'bytes': BOUNDS.model_copy(update={'max_request_bytes': 1}),
              'tariff': BOUNDS.model_copy(update={'input_price_per_mtok': 100}),
              'unverified': BOUNDS.model_copy(update={'input_bound_method': 'unverified'}),
              'missing': None}[change]
    calls = []
    guard = BudgetGuard(Decimal('1'))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request))) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', max_output_tokens=8000, api_key='synthetic'), route='route', http_client=http, ledger=ObservationLedger(), budget=guard, max_call_cost=Decimal('.5'), cost_bounds=bounds, route_metadata=METADATA)
        with pytest.raises(ComparisonStopped):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    assert not calls and not guard.uncertain and guard.spent == 0


async def test_documented_receipt_provider_display_version_and_cache():
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers['X-OpenRouter-Cache'] == 'false'
        return httpx.Response(200, headers={'X-OpenRouter-Cache-Status': 'MISS'}, json={
            'provider': 'Display Provider', 'model': 'model-2026-10', 'id': 'gen',
            'usage': {'cost': .2, 'prompt_tokens': 100, 'completion_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 20}},
            'choices': [{'message': {'content': '{'}}]})
    ledger = ObservationLedger()
    metadata = RouteMetadata(provider_names=('Display Provider',), model_versions=('model-2026-10',))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', max_output_tokens=8000, api_key='synthetic'), route='slug', http_client=http, ledger=ledger, budget=BudgetGuard(Decimal('1')), max_call_cost=Decimal('.5'), cost_bounds=BOUNDS, route_metadata=metadata)
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    item, = ledger.observations
    assert not ledger.violations and item.context_cache is True and item.full_response_cache is False
    assert item.actual_model == 'model-2026-10' and item.actual_provider == 'Display Provider'


def test_uncertain_attempt_reservation_is_not_reused():
    guard = BudgetGuard(Decimal('1'))
    reservation = guard.reserve(Decimal('.6'))
    guard.settle(reservation, None)
    assert guard.uncertain[reservation] == Decimal('.6')
    with pytest.raises(ComparisonStopped):
        guard.reserve(Decimal('.5'))


def test_global_attempt_limit_includes_failed_attempts():
    guard = BudgetGuard(Decimal('10'), max_total_attempts=2)
    guard.settle(guard.reserve(Decimal('.5')), None)
    guard.settle(guard.reserve(Decimal('.5')), Decimal('.1'))
    with pytest.raises(ComparisonStopped, match='попыток'):
        guard.reserve(Decimal('.5'))
    assert guard.attempts == 2 and guard.spent == Decimal('.1') and sum(guard.uncertain.values()) == Decimal('.5')


async def test_context_envelope_price_ceiling_is_sent_before_mock_generation(tmp_path):
    import json
    import httpx
    from linguary.comparison.protocol import CostBounds, EndpointSnapshot, file_hash
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    path = tmp_path / 'endpoints.json'
    path.write_text(json.dumps({'data': {'id': 'model', 'endpoints': [{
        'tag': 'slug', 'provider_name': 'Display Provider', 'status': 0, 'context_length': 1000000,
        'max_prompt_tokens': None, 'max_completion_tokens': 128000,
        'pricing': {'prompt': '.000003', 'completion': '.000015', 'input_cache_write_1h': '.000006'}}]}}))
    bounds = CostBounds(max_request_bytes=1048576, max_input_tokens=1000000, input_price_per_mtok=6, output_price_per_mtok=15,
                        output_includes_reasoning=True, input_bound_method='provider_context_limit', input_price_multiplier=2,
                        endpoint_snapshots={'model': EndpointSnapshot(path=str(path), sha256=file_hash(path))}, evidence='Synthetic quota proof')
    metadata = RouteMetadata(provider_names=('Display Provider',), model_versions=('model-version',))
    calls = []
    def handler(request):
        calls.append(request)
        assert json.loads(request.content)['provider']['max_price'] == {'prompt': 3, 'completion': 15, 'request': 0}
        return httpx.Response(200, json={'model': 'model-version', 'provider': 'Display Provider', 'id': 'gen',
            'usage': {'cost': .01, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'message': {'content': '{'}}]})
    guard = BudgetGuard(Decimal('20'))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', max_output_tokens=8000, api_key='synthetic'), route='slug',
            http_client=http, ledger=ObservationLedger(), budget=guard, max_call_cost=Decimal('6.12'), cost_bounds=bounds, route_metadata=metadata)
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='translate', input_text='private'))
    assert len(calls) == 1 and guard.spent == Decimal('.01')


@pytest.mark.parametrize('details', [None, {}, {'cached_tokens': None}, {'cached_tokens': '0'}, {'cached_tokens': False}, {'cached_tokens': -1}, []])
async def test_unknown_context_cache_stops_followup_and_preserves_paid_receipt(tmp_path, details):
    import json
    import httpx
    from linguary.adapters.config import LLMSettings
    from linguary.comparison.llm import ObservedLLMClient
    calls = []
    def handler(request):
        calls.append(request)
        usage = {'cost': .2, 'prompt_tokens': 100, 'completion_tokens': 10}
        if details is not None:
            usage['prompt_tokens_details'] = details
        return httpx.Response(200, headers={'X-OpenRouter-Cache-Status': 'MISS'}, json={
            'model': 'model', 'provider': 'route', 'id': 'paid-generation', 'usage': usage,
            'choices': [{'message': {'content': '{'}}]})
    ledger = ObservationLedger(receipt_root=tmp_path)
    guard = BudgetGuard(Decimal('1'))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ObservedLLMClient(settings=LLMSettings(model='model', max_output_tokens=8000, api_key='synthetic'), route='route',
            http_client=http, ledger=ledger, budget=guard, max_call_cost=Decimal('.5'), cost_bounds=BOUNDS, route_metadata=METADATA)
        request = LLMRequest.for_translation(instructions='translate', input_text='private')
        with pytest.raises(LLMResponseFormatError):
            await client.generate(request)
        with pytest.raises(ComparisonStopped):
            await client.generate(request)
    assert len(calls) == 1 and ledger.violations == ['unknown_context_cache']
    item, = ledger.observations
    assert item.context_cache is None and item.cost == guard.spent == Decimal('.2')
    stored = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert stored['cost'] == '0.2' and stored['context_cache'] is None and stored['generation_id'] == 'paid-generation'
