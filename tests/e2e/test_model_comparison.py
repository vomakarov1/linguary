import json
from dataclasses import replace
from pathlib import Path

import pytest

from linguary.adapters.epub import EbookLibEpubReader
from linguary.application.ports.epub import EpubReadRequest
from linguary.application.ports.export_validation import ExportValidationReport
from linguary.adapters.validation import EpubCheckValidator
from linguary.application.context_builder import textual_blocks
from linguary.comparison.protocol import file_hash
from linguary.adapters.llm import FakeLLMClient
from linguary.comparison.protocol import draft_protocol
from linguary.comparison.runner import run_comparison
from linguary.comparison.workspace import prepare_workspace
from pipeline_support import SYNTHETIC_EPUB, llm_replies, exported_text


@pytest.fixture(autouse=True)
def no_external_validation(monkeypatch):
    monkeypatch.setattr(EpubCheckValidator, "validate", lambda *args: ExportValidationReport(available=False))


class Factory:
    simulated = True
    def __init__(self):
        self.clients = []
    def __call__(self, *, model, stage, **kwargs):
        document = EbookLibEpubReader().read(EpubReadRequest(book_id='fixture', source_path=str(SYNTHETIC_EPUB), source_language='en', target_language='ru')).document
        document = replace(document, sections=document.ordered_sections[:1])
        replies = llm_replies(document)
        selected = replies[:1] if stage == 'analysis' else replies[1:]
        if stage == "processing":
            payloads = [json.loads(reply.content) for reply in selected]
            for payload in payloads:
                for block in payload["blocks"]:
                    block["translated_text"] += " Рен Колдер."
            selected = tuple(replace(reply, content=json.dumps(payload, ensure_ascii=False)) for reply, payload in zip(selected, payloads))
        client = FakeLLMClient.replying([replace(reply, model=model) for reply in selected])
        self.clients.append(client)
        return client


async def test_blocked_next_analysis_explains_unlinked_log_without_post(tmp_path):
    import sqlite3
    from decimal import Decimal
    from linguary.adapters.config import LLMSettings
    from linguary.adapters.llm.logging import LoggingLLMClient
    from linguary.application.ports.llm import LLMConfigurationError, LLMRequest
    from linguary.application.use_cases.import_book import ImportBookCommand
    from linguary.comparison.llm import BudgetGuard, ObservationLedger, ObservedLLMClient
    from linguary.domain import LLMCallStage

    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    application = workspace.build_application(llm_client=FakeLLMClient.replying(()))
    imported = application.import_book(ImportBookCommand(book_id='book',source_epub_path=str(SYNTHETIC_EPUB),source_language='en',target_language='ru'))
    ledger = ObservationLedger()
    ledger.stop('unknown_or_changed_route')
    budget = BudgetGuard(Decimal('10'))
    class NoHTTP:
        async def post(self,*args,**kwargs):
            pytest.fail('the blocked attempt must never reach HTTP')
    inner = ObservedLLMClient(settings=LLMSettings(model='assigned',max_output_tokens=8000,api_key='synthetic-test-only'),route='assigned-route',http_client=NoHTTP(),
                              ledger=ledger,budget=budget,max_call_cost=Decimal('.1'))
    logger = LoggingLLMClient(inner=inner,unit_of_work=application.unit_of_work,provider='assigned-route',model='assigned',on_logged=ledger.on_logged)
    request = LLMRequest.for_chapter_analysis(instructions='Analyze',input_text='chapter').with_logging(book_id=imported.book.id,stage=LLMCallStage.CHAPTER_ANALYSIS,task_id='blocked-task')
    try:
        with pytest.raises(LLMConfigurationError,match='Запись наблюдения недоступна'):
            await logger.generate(request)
        with pytest.raises(LLMConfigurationError,match='Запись наблюдения недоступна'):
            await logger.generate(request)
        with sqlite3.connect(workspace.database_path) as db:
            logs=db.execute('SELECT success,provider,model,error_type,error_message,error_detail,prompt_tokens,completion_tokens,cost,response_text FROM llm_call_logs').fetchall()
        assert logs==[(0,'assigned-route','assigned','ConfigurationError','Нарушены условия сравнения; новые вызовы остановлены','linguary.comparison.llm.ComparisonStopped',None,None,None,None)]
        assert ledger.violations==['unknown_or_changed_route','unlinked_log']
        assert ledger.observations==[] and budget.attempts==0 and budget.spent==0
    finally:
        application.dispose()


def protocol():
    draft = draft_protocol()
    return draft.model_copy(update={'chapters': tuple(c.model_copy(update={'source': str(SYNTHETIC_EPUB), 'chapter_id': '1'}) for c in draft.chapters), 'repeats': 1, 'generation': {'temperature': .5, 'max_output_tokens': 2000, 'max_attempts': 1}, 'splitting': {'max_fragment_blocks': 1000, 'max_unit_source_chars': 100000}, 'previous_context': {c.id: 'FIXED PREVIOUS CONTEXT' for c in draft.chapters}, 'fixed_glossary': {'Wren Calder': {'decision': 'accept', 'translation': 'Рен Колдер', 'note': 'женский род'}}})


async def test_four_variants_reuse_two_analysis_artifacts(tmp_path):
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    factory = Factory()
    result = await run_comparison(protocol(), workspace, factory)
    assert len(result.analyses) == 2
    assert len(result.variants) == 4
    assert all(v.completed for v in result.variants), [v.failures for v in result.variants]
    assert len({v.analysis_hash for v in result.variants}) == 2
    assert sum(len(client.requests) for client in factory.clients[:]) == 6
    assert result.simulated


async def test_selected_one_of_three_preserves_original_and_fixed_context(tmp_path):
    import sqlite3
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    before = file_hash(SYNTHETIC_EPUB)
    original = EbookLibEpubReader().read(EpubReadRequest(book_id='fixture', source_path=str(SYNTHETIC_EPUB), source_language='en', target_language='ru')).document
    selected_blocks = textual_blocks(original.ordered_sections[0])
    factory = Factory()
    result = await run_comparison(protocol(), workspace, factory)
    assert file_hash(SYNTHETIC_EPUB) == before
    assert all(v.completed for v in result.variants)
    for variant in result.variants:
        path, = variant.epubs.values()
        text = exported_text(path)
        assert 'Русский перевод блока' in text
        assert all(block.text in text for block in selected_blocks)
        with sqlite3.connect(variant.workspace.database_path) as db:
            assert db.execute('SELECT COUNT(*) FROM sections').fetchone()[0] == 3
    analysis_requests = [client.requests[0].input_text for client in factory.clients if client.requests and client.requests[0].kind.value == 'chapter_analysis']
    assert len(analysis_requests) == 2 and analysis_requests[0] == analysis_requests[1]
    assert 'FIXED PREVIOUS CONTEXT' in analysis_requests[0]
    processing = [client.requests[0].input_text for client in factory.clients if client.requests and client.requests[0].kind.value == 'translation_result']
    assert len(processing) == 4 and len(set(processing)) == 1
    assert all(value.input_hashes['control'] == before for value in result.manifests.values())


async def test_analysis_receipts_not_duplicated_by_forks(tmp_path):
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    result = await run_comparison(protocol(), workspace, Factory())
    analyses = [item for item in result.observations if item.stage == 'chapter_analysis']
    assert len(analyses) == 2 and len(result.observations) == 6
    assert {item.call_id for item in analyses} == {call for artifact in result.analyses for call in artifact.provenance['analysis_call_ids']}
    assert len({item.call_uid for item in result.observations}) == 6
    assert all(item.accepted is True for item in result.observations)


class RemainingMatrixFactory:
    simulated = True

    def __init__(self, stop_after=None):
        self.clients = []
        self.stop_after = stop_after

    def __call__(self, *, model, stage, ledger, budget, **kwargs):
        import re
        from linguary.adapters.llm import FakeReply
        from pipeline_support import read_fixture

        owner = self

        class DynamicFake(FakeLLMClient):
            async def generate(self, request):
                if stage == 'analysis' and owner.stop_after is not None and len(self.requests) == owner.stop_after:
                    ledger.stop('unknown_or_changed_route')
                    ledger.ensure_open()
                aliases = tuple(dict.fromkeys(re.findall(r'\[(b\d+)\]', request.input_text)))
                if stage == 'analysis':
                    payload = read_fixture('analysis-template.json')
                    payload['treatment_plan'] = [{'source_block_id': alias, 'treatment': 'TRANSLATE'} for alias in aliases]
                else:
                    payload = {'blocks': [{'source_block_id': alias, 'treatment': 'TRANSLATE',
                        'translated_text': f'Русский перевод учебного блока сохраняет смысл и события главы. Рен Колдер. {alias}.',
                        'glossary_hints': []} for alias in aliases], 'language_notes': []}
                self.default = replace(FakeReply.of_payload(payload), model=model)
                return await super().generate(request)

        client = DynamicFake()
        self.clients.append((stage, model, client, ledger, budget))
        return client


@pytest.fixture
async def partial_analysis_continuation(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    from dataclasses import asdict
    from decimal import Decimal
    from linguary.comparison import runner
    from linguary.comparison.protocol import stable_hash
    from linguary.comparison.workspace import ContinuationState, PartialAnalysisSeed, fork_processing

    corpus = tmp_path / 'six-chapters.epub'
    with zipfile.ZipFile(SYNTHETIC_EPUB) as original, zipfile.ZipFile(corpus, 'w') as target:
        first = original.read('OEBPS/chapter1.xhtml')
        third = original.read('OEBPS/chapter3.xhtml')
        for member in original.infolist():
            data = original.read(member)
            if member.filename == 'OEBPS/content.opf':
                additions = b''.join(f'<item id="chapter{i}" href="chapter{i}.xhtml" media-type="application/xhtml+xml"/>'.encode() for i in range(4, 7))
                spine = b''.join(f'<itemref idref="chapter{i}"/>'.encode() for i in range(4, 7))
                data = data.replace(b'</manifest>', additions + b'</manifest>').replace(b'</spine>', spine + b'</spine>')
            if member.filename == 'OEBPS/chapter2.xhtml':
                data = first.replace(b'chapter1', b'chapter2')
            target.writestr(member, data)
        for index in range(4, 7):
            target.writestr(f'OEBPS/chapter{index}.xhtml', third.replace(b'chapter3', f'chapter{index}'.encode()))
    draft = protocol()
    value = draft.model_copy(update={'chapters': tuple(draft.chapters[min(i - 1, 3)].model_copy(update={'id': f'ch-{i}', 'source': str(corpus), 'chapter_id': str(i)}) for i in range(1, 7)),
        'previous_context': {f'ch-{i}': f'FIXED CONTEXT {i}' for i in range(1, 7)},
        'splitting': dict(draft.splitting, max_unit_source_chars=1250), 'budget': 9.888884480, 'max_total_attempts': 67})
    origin = prepare_workspace(tmp_path / 'historical', [tmp_path / 'production'], value)
    sources = {file_hash(corpus): 'book-' + file_hash(corpus)[:16]}
    baseline = runner.ComparisonResult(value.content_hash, True)
    factory = RemainingMatrixFactory()
    luna, sonnet = value.variants[0].analysis_model, value.variants[-1].analysis_model
    luna_artifact, selection = await runner._run_analysis(value, origin, factory, baseline, None, True, sources, 1, luna)
    sonnet_artifact, _ = await runner._run_analysis(value, origin, factory, baseline, None, True, sources, 1, sonnet)
    inspection = fork_processing(sonnet_artifact, 'seed-inspection')
    application = inspection.build_application(llm_client=FakeLLMClient.replying(()))
    book_id, = sources.values()
    with application.unit_of_work.transaction() as transaction:
        document = transaction.documents.get(book_id)
        section = document.ordered_sections[0]
        analysis = transaction.analyses.get_for_section(book_id, section.id)
        observation = next(item for item in baseline.observations if item.assigned_model == sonnet and item.chapter_id == section.id)
        log = next(item for item in transaction.llm_calls.list_for_book(book_id) if item.id == observation.call_id)
    await application.aclose()
    observation = replace(observation, cost=Decimal('.064236'), duration_ms=355243, generation_id='historical-sonnet-generation',
        model_identity_source='generation_metadata', metadata_status_code=200, metadata_check_reason='verified')
    log = replace(log, cost=observation.cost, duration_ms=observation.duration_ms)
    context = runner._FixedContext({section.id: 'FIXED CONTEXT 1'}, runner.ObservationLedger()).for_chapter_analysis(document=document, section=section, accumulated_candidates=())
    fragment = {'analysis_id': analysis.id, 'book_id': book_id, 'section_id': section.id, 'call_uid': observation.call_uid,
        'call_id': log.id, 'operation_id': observation.operation_id, 'generation_id': observation.generation_id,
        'block_ids': list(analysis.treatment_plan.block_ids), 'typed_analysis_sha256': stable_hash(asdict(analysis)),
        'request_input_sha256': hashlib.sha256(context.request.input_text.encode()).hexdigest(),
        'request_instructions_sha256': hashlib.sha256(context.request.instructions.encode()).hexdigest()}
    imported = [replace(item, cost=Decimal('.002'), generation_id=f'historical-luna-{i}') for i, item in enumerate(baseline.observations[:6])]
    imported.extend(replace(imported[0], call_uid=f'old-ll-{i}', call_id=f'old-log-{i}', generation_id=f'old-generation-{i}',
        operation_id=f'old-operation-{i}', stage='translation_result', variant_id=value.variants[0].id,
        accepted=False if i == 9 else True, sequence=i + 1) for i in range(10))
    imported[-1] = replace(imported[-1], cost=Decimal('.008675320'))
    provenance = {'kind': 'late_sonnet_canterville', 'historical_violations': {'sonnet': ['unknown_or_changed_route', 'unlinked_log']},
        'excluded_local_logs': [{'call_id': 'forbidden-local-log', 'new_post': False}],
        'accounting': {'spent_posts': 21, 'spent_cost': '.111115520', 'remaining_posts': 67, 'remaining_cost': '9.888884480'}}
    seed = PartialAnalysisSeed(analysis, log, observation, fragment, provenance)
    stored = {'variant_id': value.variants[0].id, 'repeat': 1, 'analysis_hash': luna_artifact.content_hash,
        'failures': ['chapter_failed:historical', 'branch_incomplete'], 'completed_chapters': [], 'epub_chapters': {}}
    stage = f'analysis:1:{sonnet}'
    state = ContinuationState(luna_artifact, stored, [*imported, observation], selection, provenance, {stage: 359.376}, seed)
    monkeypatch.setattr(runner, 'load_continuation', lambda workspace: state)
    return value, state


async def test_remaining_matrix_has_29_logical_operations(tmp_path, partial_analysis_continuation):
    from decimal import Decimal
    value, state = partial_analysis_continuation
    workspace = prepare_workspace(tmp_path / 'remaining', [tmp_path / 'production'], value)
    factory = RemainingMatrixFactory()
    result = await run_comparison(value, workspace, factory)
    assert all(branch.completed for branch in result.variants[1:]), result.violations
    assert result.variants[0].failures == state.variant['failures'] and not result.variants[0].completed
    luna, sonnet = value.variants[0].analysis_model, value.variants[-1].analysis_model
    assert [(stage, model) for stage, model, *_ in factory.clients] == [('processing', sonnet), ('analysis', sonnet), ('processing', luna), ('processing', sonnet)]
    analysis_calls = [request for stage, _, client, *_ in factory.clients if stage == 'analysis' for request in client.requests]
    assert len(analysis_calls) == 5
    assert all('FIXED CONTEXT 1\n' not in request.input_text for request in analysis_calls)
    assert sum(len(client.requests) for _, _, client, *_ in factory.clients) == 29
    assert len(result.analyses) == 2
    assert result.variants[2].analysis_hash == result.variants[3].analysis_hash == result.analyses[1].content_hash
    assert len(result.observations) == 46
    for field in ('call_uid', 'call_id'):
        assert len({getattr(item, field) for item in result.observations}) == 46
    assert len({item.operation_id for item in result.observations if item.first_response}) == 46
    assert sum(item.cost for item in result.observations) == Decimal('.102911320')
    assert result.observations[:17] == state.observations
    assert all(budget.spent == 0 and budget.attempts == 0 and budget.max_total_attempts == 67 for *_, budget in factory.clients)
    seed = state.partial_seed
    resumed = result.variants[2].workspace.build_application(llm_client=FakeLLMClient.replying(()))
    try:
        with resumed.unit_of_work.transaction() as transaction:
            assert transaction.analyses.get_for_section(seed.log.book_id, seed.analysis.section_id) == seed.analysis
            logs = transaction.llm_calls.list_for_book(seed.log.book_id)
            assert next(log for log in logs if log.id == seed.log.id) == seed.log
            assert not any(log.id == 'forbidden-local-log' for log in logs)
            assert not any(task.id == seed.log.task_id for task in transaction.tasks.list_for_book(seed.log.book_id))
    finally:
        await resumed.aclose()
    stage = f'analysis:1:{sonnet}'
    segments = result.stage_wall_segments[stage]
    assert segments[0] == {'origin': 'imported', 'seconds': 359.376}
    assert segments[1]['origin'] == 'current' and segments[1]['seconds'] > 0
    assert result.stage_wall_seconds[stage] == sum(segment['seconds'] for segment in segments)
    assert seed.observation.duration_ms == 355243


@pytest.mark.parametrize('changed', ['typed', 'context', 'instructions', 'blocks'])
async def test_seed_data_or_context_change_stops_before_analysis_call(tmp_path, partial_analysis_continuation, changed):
    value, state = partial_analysis_continuation
    fragment = state.partial_seed.fragment
    key = {'typed': 'typed_analysis_sha256', 'context': 'request_input_sha256', 'instructions': 'request_instructions_sha256', 'blocks': 'block_ids'}[changed]
    fragment[key] = [] if changed == 'blocks' else 'changed'
    workspace = prepare_workspace(tmp_path / 'remaining', [tmp_path / 'production'], value)
    factory = RemainingMatrixFactory()
    result = await run_comparison(value, workspace, factory)
    assert not any(client.requests for stage, _, client, *_ in factory.clients if stage == 'analysis')
    assert len(result.analyses) == 1
    assert 'partial_seed_mismatch' in result.violations


async def test_remaining_analysis_stops_on_future_metadata_failure(tmp_path, partial_analysis_continuation):
    value, _ = partial_analysis_continuation
    workspace = prepare_workspace(tmp_path / 'remaining', [tmp_path / 'production'], value)
    factory = RemainingMatrixFactory(stop_after=1)
    result = await run_comparison(value, workspace, factory)
    assert len(factory.clients) == 2
    assert len(factory.clients[1][2].requests) == 1
    assert len(result.analyses) == 1
    assert len(result.variants) == 2
    assert 'unknown_or_changed_route' in result.violations


async def test_future_missing_metadata_stops_followup_post(tmp_path, partial_analysis_continuation, monkeypatch):
    import re
    import httpx
    from decimal import Decimal
    from linguary.comparison import llm, runner
    from linguary.comparison.protocol import CostBounds, RouteMetadata
    from pipeline_support import read_fixture

    value, state = partial_analysis_continuation
    seed = state.partial_seed
    model = seed.observation.assigned_model
    value = value.model_copy(update={'routes': {model: 'mock-route'}, 'max_call_cost': .2})
    workspace = prepare_workspace(tmp_path / 'remaining', [tmp_path / 'production'], value)
    monkeypatch.setattr(llm, '_METADATA_MAX_ATTEMPTS', 2)
    monkeypatch.setattr(llm, '_METADATA_RETRY_DELAYS', (0,))
    methods = []

    def handler(request):
        methods.append(request.method)
        if request.method == 'GET':
            return httpx.Response(404, json={'error': 'metadata unavailable'})
        body = json.loads(request.content)
        aliases = tuple(dict.fromkeys(re.findall(r'\[(b\d+)\]', body['messages'][-1]['content'])))
        payload = read_fixture('analysis-template.json')
        payload['treatment_plan'] = [{'source_block_id': alias, 'treatment': 'TRANSLATE'} for alias in aliases]
        return httpx.Response(200, json={'id': 'new-generation', 'model': model, 'provider': 'mock-route',
            'usage': {'cost': .01, 'prompt_tokens': 100, 'completion_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 0}},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(payload, ensure_ascii=False)}}]})

    budget = llm.BudgetGuard(Decimal('9.888884480'), 67)

    def factory(*, settings, ledger, budget, **kwargs):
        assert budget.attempts == 0 and budget.spent == 0
        return llm.ObservedLLMClient(settings=replace(settings, api_key='synthetic-test-only'), route='mock-route',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), ledger=ledger, budget=budget, max_call_cost=Decimal('.2'),
            cost_bounds=CostBounds(max_request_bytes=1000000, max_input_tokens=100000, input_price_per_mtok=1, output_price_per_mtok=1,
                output_includes_reasoning=True, input_bound_method='synthetic', evidence='MockTransport only'),
            route_metadata=RouteMetadata(provider_names=('mock-route',), model_versions=(model + '-snapshot',)))

    result = runner.ComparisonResult(value.content_hash, False, observations=list(state.observations))
    source = Path(value.chapters[0].source)
    prepared = await runner._run_analysis(value, workspace, factory, result, budget, False,
        {file_hash(source): seed.log.book_id}, 1, model, seed=seed)
    assert prepared is None and result.analyses == []
    assert methods == ['POST', 'GET', 'GET']
    assert 'unknown_or_changed_route' in result.violations
    assert budget.attempts == 1 and budget.spent == Decimal('.01')
    assert len(result.observations) == 18 and result.observations[-1].actual_model is None
    assert result.observations[-1].metadata_check_reason == 'attempts_exhausted'
    assert seed.observation.duration_ms == 355243


async def test_stage_client_and_snapshot_match(tmp_path):
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    result = await run_comparison(protocol(), workspace, Factory())
    for variant in result.variants:
        snapshot, = variant.llm_snapshots.values()
        observation, = variant.observations
        assert snapshot.model == observation.assigned_model == observation.actual_model
        assert observation.call_id is not None
        assert observation.stage == 'translation_result'


async def test_final_epub_preserves_original_and_glossary(tmp_path):
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    result = await run_comparison(protocol(), workspace, Factory())
    for variant in result.variants:
        path, = variant.epubs.values()
        text = exported_text(path)
        assert 'Wren Calder' in text and 'Рен Колдер' in text
        assert 'Причастие humming' in text


@pytest.mark.parametrize('failure', ['analysis', 'glossary'])
async def test_incomplete_analysis_or_glossary_blocks_processing(tmp_path, failure):
    from linguary.adapters.llm import FakeReply
    prepared_protocol = protocol()
    if failure == 'glossary':
        prepared_protocol = prepared_protocol.model_copy(update={'fixed_glossary': {'Wren Calder': {'decision': 'accept', 'translation': 'Несогласованный вариант'}}})
    class BadAnalysisFactory(Factory):
        def __call__(self, *, model, stage, **kwargs):
            client = super().__call__(model=model, stage=stage, **kwargs)
            if failure == 'analysis' and stage == 'analysis':
                client.replies = (FakeReply(content='{'),)
            return client
    factory = BadAnalysisFactory()
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], prepared_protocol)
    result = await run_comparison(prepared_protocol, workspace, factory)
    assert not result.analyses and all(not variant.completed for variant in result.variants)
    assert len(factory.clients) == 2
    assert all(client.requests[0].kind.value == 'chapter_analysis' for client in factory.clients)


async def test_failed_chapter_is_reported(tmp_path):
    from linguary.adapters.llm import FakeReply
    class FailedChapterFactory(Factory):
        def __call__(self, *, model, stage, **kwargs):
            client = super().__call__(model=model, stage=stage, **kwargs)
            if stage == 'processing':
                client.replies = (FakeReply(content='{'),)
            return client
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    result = await run_comparison(protocol(), workspace, FailedChapterFactory())
    assert len(result.analyses) == 2 and len(result.variants) == 4
    assert all(not branch.completed and not branch.epubs and any(failure.startswith('chapter_failed:') for failure in branch.failures) for branch in result.variants)
    assert all(item.accepted is False for branch in result.variants for item in branch.observations)


async def test_protocol_schedule_is_applied_exactly(tmp_path):
    prepared_protocol = protocol()
    schedule = tuple(reversed([v.id for v in prepared_protocol.variants]))
    prepared_protocol = prepared_protocol.model_copy(update={'schedule': schedule})
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], prepared_protocol)
    result = await run_comparison(prepared_protocol, workspace, Factory())
    assert tuple(branch.variant_id for branch in result.variants) == schedule
    assert len(result.analyses) == 2 and all(branch.completed for branch in result.variants)


async def test_unapproved_live_never_constructs_client(tmp_path):
    from linguary.comparison.llm import ComparisonStopped
    calls = []
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], protocol())
    def factory(**kwargs):
        calls.append(kwargs)
        raise AssertionError('must not construct a real client')
    with pytest.raises(ComparisonStopped):
        await run_comparison(protocol(), workspace, factory)
    assert not calls


async def test_chapter_and_group_refs_with_two_chapters_and_retries(tmp_path):
    import re
    from collections import defaultdict
    from linguary.adapters.llm import FakeReply
    from linguary.application.ports.llm import LLMResponseKind
    from pipeline_support import read_fixture

    class DynamicFake(FakeLLMClient):
        async def generate(self, request):
            aliases = tuple(dict.fromkeys(re.findall(r'\[(b\d+)\]', request.input_text)))
            if request.attempt == 1:
                self.default = FakeReply(content='{', model=self.assigned_model)
            elif request.kind == LLMResponseKind.CHAPTER_ANALYSIS:
                payload = read_fixture('analysis-template.json')
                payload['treatment_plan'] = [{'source_block_id': alias, 'treatment': 'TRANSLATE'} for alias in aliases]
                self.default = replace(FakeReply.of_payload(payload), model=self.assigned_model)
            else:
                payload = {'blocks': [{'source_block_id': alias, 'treatment': 'TRANSLATE', 'translated_text': f'Русский перевод учебного блока сохраняет смысл и события главы. Рен Колдер. {alias}.', 'glossary_hints': []} for alias in aliases], 'language_notes': []}
                self.default = replace(FakeReply.of_payload(payload), model=self.assigned_model)
            return await super().generate(request)

    class DynamicFactory:
        simulated = True
        def __call__(self, *, model, **kwargs):
            client = DynamicFake()
            client.assigned_model = model
            return client

    prepared_protocol = protocol()
    prepared_protocol = prepared_protocol.model_copy(update={
        'chapters': tuple(chapter.model_copy(update={'chapter_id': str(index % 2 + 1)}) for index, chapter in enumerate(prepared_protocol.chapters)),
        'generation': dict(prepared_protocol.generation, max_attempts=2),
        'concurrency': 2,
        'splitting': dict(prepared_protocol.splitting, max_fragment_blocks=2, max_unit_source_chars=180),
    })
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], prepared_protocol)
    result = await run_comparison(prepared_protocol, workspace, DynamicFactory())
    assert len(result.variants) == 4 and all(branch.completed for branch in result.variants), [branch.failures for branch in result.variants]
    chapter_ids = {item.chapter_id for item in result.observations}
    assert None not in chapter_ids and len(chapter_ids) == 2
    operations = defaultdict(list)
    for item in result.observations:
        operations[item.operation_id].append(item)
    assert all(len(items) == 2 and items[0].group_id == items[1].group_id and items[0].chapter_id == items[1].chapter_id and items[0].first_response and not items[1].first_response for items in operations.values())
    for branch in result.variants:
        assert len({item.chapter_id for item in branch.observations}) == 2
        assert len({item.group_id for item in branch.observations}) > 2


def live_protocol():
    from linguary.comparison.protocol import ComparisonProtocol
    from linguary.adapters.config import CheckSettings
    from dataclasses import asdict
    value = protocol()
    return ComparisonProtocol.model_validate(value.model_dump() | {
        'temperature_policy': 'set', 'rubric': {k: 3 for k in value.rubric}, 'scale_max': 5, 'sufficient_gain': 1, 'comparability_margin': .2,
        'minimum_first_responses': {'analysis': 20, 'processing': 20}, 'schedule': tuple(v.id for v in value.variants),
        'critical_places': [{'chapter_id': 'control', 'location': 'wordplay', 'expected_explanation': 'Explain the semantic relation'}],
        'max_total_attempts': 88, 'checks': asdict(CheckSettings()), 'requests_per_minute': 1000, 'concurrency': 1, 'budget': 10, 'max_call_cost': 1,
        'stop_conditions': ['budget'], 'reference_book': 'synthetic reference', 'projection_method': 'fixed source character ratio',
        'max_book_cost': 10, 'max_book_seconds': 3600, 'routes': {v.analysis_model: 'mock-route' for v in value.variants},
        'context_cache_policy': 'record', 'retain_rejected': False, 'diagnostic_retention_seconds': 0, 'diagnostic_access': 'local',
        'code_version': 'test', 'instruction_version': 'test', 'schema_version': 'test', 'checks_version': 'test',
    })


def four_chapter_protocol(tmp_path):
    value = live_protocol()
    import zipfile
    corpus = tmp_path / 'four-chapters.epub'
    with zipfile.ZipFile(SYNTHETIC_EPUB) as original, zipfile.ZipFile(corpus, 'w') as target:
        fourth = original.read('OEBPS/chapter1.xhtml').replace(b'chapter1', b'chapter4').replace(b'Duskmere', b'Moonwater')
        for member in original.infolist():
            data = original.read(member)
            if member.filename == 'OEBPS/content.opf':
                data = data.replace(b'</manifest>', b'<item id="chapter4" href="chapter4.xhtml" media-type="application/xhtml+xml"/></manifest>').replace(b'</spine>', b'<itemref idref="chapter4"/></spine>')
            target.writestr(member, data)
        target.writestr('OEBPS/chapter4.xhtml', fourth)
    value = value.model_copy(update={'chapters': tuple(c.model_copy(update={'source': str(corpus), 'chapter_id': str(index + 1)}) for index, c in enumerate(value.chapters))})
    from linguary.comparison.protocol import CostBounds, RouteMetadata
    return value.model_copy(update={'version_hashes': {key: {str(Path(__file__).resolve()): file_hash(Path(__file__))} for key in ('code_version', 'instruction_version', 'schema_version', 'checks_version')}, 'source_hashes': {c.id: file_hash(Path(c.source)) for c in value.chapters},
        'route_metadata': {m: RouteMetadata(provider_names=(r,), model_versions=(m,)) for m, r in value.routes.items()},
        'cost_bounds': CostBounds(max_request_bytes=1000000, max_input_tokens=100000, input_price_per_mtok=1, output_price_per_mtok=1, output_includes_reasoning=True, input_bound_method='synthetic', evidence='MockTransport only')})


def cli_args(tmp_path, value, mode, approval=None):
    from linguary.comparison.__main__ import parser
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(value.model_dump_json())
    command = [mode, '--protocol', str(protocol_path), '--root', str(tmp_path / 'comparison'), '--protected-root', str(tmp_path / 'production')]
    if approval:
        command += ['--approval', str(approval)]
    return parser().parse_args(command)


async def test_prepare_without_api_key_or_network(tmp_path, monkeypatch):
    from linguary.comparison.__main__ import execute
    monkeypatch.delenv('COMPARISON_API_KEY', raising=False)
    calls = []
    assert await execute(cli_args(tmp_path, protocol(), '--prepare'), http_factory=lambda **kwargs: calls.append(kwargs)) == 0
    assert not calls and (tmp_path / 'comparison' / 'manifest.json').exists()


@pytest.fixture
async def continuation_origin(tmp_path, monkeypatch, request):
    import httpx
    import re
    from linguary.comparison import runner
    from linguary.comparison.__main__ import execute
    from linguary.comparison.protocol import ProtocolApproval, ContinuationBinding
    from linguary.comparison.workspace import _continuation_inventory
    from pipeline_support import read_fixture

    value = four_chapter_protocol(tmp_path)
    native_provider = getattr(request, 'param', 'OpenAI')
    value = value.model_copy(update={'routes': {model: 'openai' for model in value.routes},
        'route_metadata': {model: route.model_copy(update={'provider_names': (native_provider,)}) for model, route in value.route_metadata.items()}})
    value = value.model_copy(update={'generation': dict(value.generation, max_output_tokens=8000),
        'splitting': dict(value.splitting, max_unit_source_chars=700), 'max_call_cost': .2})
    processing_calls, requests = 0, []
    def handler(request):
        nonlocal processing_calls
        assert request.method == 'POST'
        body = json.loads(request.content)
        requests.append(body)
        aliases = tuple(dict.fromkeys(re.findall(r'\[(b\d+)\]', body['messages'][-1]['content'])))
        if 'analysis' in body['response_format']['json_schema']['name'].lower():
            payload = read_fixture('analysis-template.json')
            payload['treatment_plan'] = [{'source_block_id': alias, 'treatment': 'TRANSLATE'} for alias in aliases]
        else:
            processing_calls += 1
            payload = {'blocks': [{'source_block_id': alias, 'treatment': 'TRANSLATE',
                'translated_text': f'Русский перевод учебного блока сохраняет смысл и события главы. Рен Колдер. {alias}.',
                'glossary_hints': []} for alias in aliases], 'language_notes': []}
            if processing_calls == 2:
                payload = {'invalid': 'schema'}
        return httpx.Response(200, headers={'X-OpenRouter-Cache-Status': 'MISS'}, json={
            'id': f'gen-{len(requests)}', 'model': body['model'], 'provider': native_provider,
            'usage': {'cost': .01, 'prompt_tokens': 100, 'completion_tokens': 10,
                      'prompt_tokens_details': {'cached_tokens': 32}},
            'choices': [{'message': {'content': json.dumps(payload, ensure_ascii=False)}, 'finish_reason': 'stop'}]})
    http_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    await execute(cli_args(tmp_path, value, '--prepare'))
    approval = tmp_path / 'approval.json'
    approval.write_text(ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-14').model_dump_json())
    monkeypatch.setenv('COMPARISON_API_KEY', 'synthetic-key')
    original_finalize = runner._finalize
    def legacy_finalize(application, ledger, book_ids):
        statuses = {}
        with application.unit_of_work.transaction() as transaction:
            for book_id in book_ids:
                tasks = {task.id: task for task in transaction.tasks.list_for_book(book_id)}
                for item in transaction.llm_calls.list_for_book(book_id):
                    task = tasks.get(item.task_id)
                    statuses[item.id] = item.success if not item.success or (task is not None and task.status.value == 'succeeded') else None
        for operation in {item.operation_id for item in ledger.observations}:
            ledger.finalize_operation(operation, accepted=None, call_statuses=statuses)
    monkeypatch.setattr(runner, '_finalize', legacy_finalize)
    await execute(cli_args(tmp_path, value, '--live', approval), http_factory=http_factory)
    monkeypatch.setattr(runner, '_finalize', original_finalize)
    root = tmp_path / 'comparison'
    result = json.loads((root / 'result.json').read_text())
    assert len(result['analyses']) == len(result['variants']) == 1
    assert result['variants'][0]['completed'] is False and 'unknown_final_status' in result['violations']
    frozen = root / result['analyses'][0]['path']
    manifest_files = json.loads((frozen / 'artifact.json').read_text())['files']
    assert 'data/linguary.db-wal' not in manifest_files and 'data/linguary.db-shm' not in manifest_files
    (frozen / 'data' / 'linguary.db-wal').write_bytes(b'')
    (frozen / 'data' / 'linguary.db-shm').write_bytes(bytes(32768))
    recipe = {'source_root': str(root), 'source_protocol': value.model_dump(mode='json'),
        'file_hashes': _continuation_inventory(root), 'analysis_path': result['analyses'][0]['path'],
        'variant_path': str(next(root.glob('processing/*')).relative_to(root))}
    binding_path = tmp_path / 'continuation.json'
    binding_path.write_text(json.dumps(recipe))
    following = value.model_copy(update={'budget': 9.8, 'max_total_attempts': 68,
        'continuation': ContinuationBinding(path=str(binding_path), sha256=file_hash(binding_path))})
    return following, root, binding_path, result, http_factory, requests


@pytest.mark.parametrize('continuation_origin', ['OpenAI', 'Anthropic'], indirect=True)
async def test_continuation_keeps_failed_control_and_runs_only_remaining_variants(tmp_path, continuation_origin):
    from decimal import Decimal
    from linguary.comparison.__main__ import execute, _restore, result_document, LiveClientFactory
    from linguary.comparison.runner import run_comparison
    from linguary.comparison.protocol import ProtocolApproval, validate_for_live
    from linguary.comparison.workspace import _continuation_inventory

    value, old_root, _, original, http_factory, requests = continuation_origin
    before, prior_requests = _continuation_inventory(old_root), len(requests)
    args = cli_args(tmp_path, value, '--prepare')
    args.root = str(tmp_path / 'continued')
    await execute(args)
    workspace, _ = _restore(args.root, args.protected_root, value)
    clients = []
    class RecordingFactory(LiveClientFactory):
        def __call__(self, **kwargs):
            if not clients:
                assert kwargs['budget'].spent == 0 and kwargs['budget'].attempts == 0
            clients.append((kwargs['stage'], kwargs['model']))
            return super().__call__(**kwargs)
    approval = ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-14')
    factory = RecordingFactory(value, 'synthetic-key', http_factory)
    result = await run_comparison(validate_for_live(value, approval), workspace, factory)
    luna, sonnet = value.variants[0].analysis_model, value.variants[-1].analysis_model
    assert clients == [('processing', sonnet), ('analysis', sonnet), ('processing', luna), ('processing', sonnet)]
    assert [branch.variant_id for branch in result.variants] == list(value.schedule)
    assert result.variants[0].completed is False and result.variants[0].failures == original['variants'][0]['failures']
    assert all(branch.completed for branch in result.variants[1:]), result.violations
    assert result.variants[0].analysis_hash == result.variants[1].analysis_hash == original['analyses'][0]['content_hash']
    assert result.analyses[0].sidecar_hashes.keys() == {'data/linguary.db-wal', 'data/linguary.db-shm'}
    assert len({item.call_uid for item in result.observations}) == len(result.observations)
    old_receipts = [json.loads(path.read_text()) for path in (old_root / 'receipts').glob('*.json')]
    assert sum(item.cost for item in result.observations) == sum(Decimal(item['cost']) for item in old_receipts) + Decimal('.01') * (len(requests) - prior_requests)
    imported = {item.call_uid: item for item in result.observations if item.call_uid in {r['call_uid'] for r in old_receipts}}
    assert all(imported[item['call_uid']].first_response == item['first_response'] for item in old_receipts)
    assert all(imported[item['call_uid']].accepted is False for item in old_receipts if item['accepted'] is False)
    assert result.continuation_provenance['verdict_repairs']
    assert all(item.context_cache is True for item in imported.values())
    assert all(branch.workspace.root.is_relative_to(workspace.root) for branch in result.variants[1:])
    assert _continuation_inventory(old_root) == before
    assert result_document(result, value, workspace.root)['continuation_provenance']['origin_protocol_hash'] == original['protocol_hash']


@pytest.mark.parametrize('change', ['inventory', 'unknown_cost', 'unknown_route', 'wrong_logged_adapter', 'wrong_native_provider', 'cache_hit', 'unknown_context_cache', 'missing_log', 'committed_wal_missing_log',
    'symlink', 'escape', 'copied_artifact', 'budget_extension', 'attempt_extension',
    'nonempty_frozen_wal', 'unbound_frozen_sidecar', 'foreign_frozen_extra', 'missing_frozen_file', 'modified_frozen_file'])
async def test_continuation_admission_rejects_unproven_origin_before_http(tmp_path, continuation_origin, change):
    import sqlite3
    from contextlib import closing
    from linguary.comparison.__main__ import execute
    from linguary.comparison.protocol import ProtocolApproval, stable_hash
    from linguary.comparison.workspace import _continuation_inventory, WorkspaceError

    value, origin, binding_path, _, _, _ = continuation_origin
    args = cli_args(tmp_path, value, '--prepare')
    args.root = str(tmp_path / 'continued')
    await execute(args)
    fresh = Path(args.root)
    wal_writer = None
    recipe = json.loads(binding_path.read_text())
    if change == 'inventory':
        (origin / 'result.json').write_text('{}')
    elif change == 'copied_artifact':
        source = next((fresh / 'imported-analysis' / 'inputs').glob('*.epub'))
        source.write_bytes(b'changed')
    elif change == 'symlink':
        target = tmp_path / 'recipe-copy.json'
        target.write_bytes(binding_path.read_bytes())
        binding_path.unlink()
        binding_path.symlink_to(target)
    else:
        receipt = next((origin / 'receipts').glob('*.json'))
        data = json.loads(receipt.read_text())
        if change == 'unknown_cost':
            data['cost'] = None
        if change == 'unknown_route':
            data['actual_model'] = 'unapproved'
        if change == 'wrong_native_provider':
            data['actual_provider'] = 'openai'
        if change == 'wrong_logged_adapter':
            processing = next(json.loads(p.read_text()) for p in (origin / 'receipts').glob('*.json')
                              if json.loads(p.read_text())['stage'] == 'translation_result')
            with closing(sqlite3.connect(origin / recipe['variant_path'] / 'data' / 'linguary.db')) as db, db:
                db.execute('UPDATE llm_call_logs SET provider = ? WHERE id = ?', ('OpenAI', processing['call_id']))
        if change == 'cache_hit':
            data['full_response_cache'] = True
        if change == 'unknown_context_cache':
            data['context_cache'] = None
        if change in ('unknown_cost', 'unknown_route', 'wrong_native_provider', 'cache_hit', 'unknown_context_cache'):
            receipt.write_text(json.dumps(data))
        if change in ('missing_log', 'committed_wal_missing_log'):
            branch_db = origin / recipe['variant_path'] / 'data' / 'linguary.db'
            processing = next(json.loads(p.read_text()) for p in (origin / 'receipts').glob('*.json')
                              if json.loads(p.read_text())['stage'] == 'translation_result')
            if change == 'committed_wal_missing_log':
                wal_writer = sqlite3.connect(branch_db)
                wal_writer.execute('PRAGMA journal_mode=WAL')
                wal_writer.execute('DELETE FROM llm_call_logs WHERE id = ?', (processing['call_id'],))
                wal_writer.commit()
                assert branch_db.with_name(branch_db.name + '-wal').is_file()
            else:
                with closing(sqlite3.connect(branch_db)) as db, db:
                    db.execute('DELETE FROM llm_call_logs WHERE id = ?', (processing['call_id'],))
        if change == 'escape':
            recipe['analysis_path'] = '../outside'
        frozen = origin / json.loads(binding_path.read_text())['analysis_path']
        if change == 'nonempty_frozen_wal':
            (frozen / 'data' / 'linguary.db-wal').write_bytes(b'committed state outside the frozen digest')
        if change == 'foreign_frozen_extra':
            (frozen / 'data' / 'unexpected.bin').write_bytes(b'extra')
        if change == 'missing_frozen_file':
            next((frozen / 'inputs').glob('*.epub')).unlink()
        if change == 'modified_frozen_file':
            next((frozen / 'inputs').glob('*.epub')).write_bytes(b'changed')
        recipe['file_hashes'] = _continuation_inventory(origin)
        if change == 'unbound_frozen_sidecar':
            recipe['file_hashes'].pop(str((frozen / 'data' / 'linguary.db-wal').relative_to(origin)))
        binding_path.write_text(json.dumps(recipe))
        updates = {'continuation': value.continuation.model_copy(update={'sha256': file_hash(binding_path)})}
        if change == 'budget_extension':
            updates['budget'] = 10
        if change == 'attempt_extension':
            updates['max_total_attempts'] = 88
        value = value.model_copy(update=updates)
        (fresh / 'continuation.json').write_bytes(binding_path.read_bytes())
        manifest = json.loads((fresh / 'manifest.json').read_text())
        manifest['protocol_hash'] = value.content_hash
        manifest['artifact_hashes']['continuation.json'] = file_hash(fresh / 'continuation.json')
        (fresh / 'manifest.json').write_text(json.dumps(manifest))
        (fresh / 'prepared.json').write_text(json.dumps({'protocol_hash': value.content_hash, 'manifest_hash': stable_hash(manifest)}))
    approval_path = tmp_path / 'next-approval.json'
    approval_path.write_text(ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-14').model_dump_json())
    args = cli_args(tmp_path, value, '--live', approval_path)
    args.root = str(fresh)
    clients = []
    try:
        with pytest.raises((WorkspaceError, ValueError)):
            await execute(args, http_factory=lambda **kwargs: clients.append(kwargs))
    finally:
        if wal_writer is not None:
            wal_writer.close()
    assert not clients


def test_legacy_admission_hash_is_preserved_without_continuation(tmp_path):
    from linguary.comparison.protocol import ComparisonProtocol, ProtocolApproval, stable_hash, validate_for_live
    document = four_chapter_protocol(tmp_path).model_dump(mode='json')
    document.pop('continuation')
    old_hash = stable_hash(document)
    restored = ComparisonProtocol.model_validate(document)
    assert restored.content_hash == old_hash
    approval = ProtocolApproval(protocol_hash=old_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-14')
    assert validate_for_live(restored, approval).protocol_hash == old_hash


@pytest.mark.parametrize('proxy_mode', ['auto', 'direct'])
async def test_cli_factory_uses_prepared_proxy_mode_without_persisting_credentials(tmp_path, monkeypatch, proxy_mode):
    import httpx
    from decimal import Decimal
    from linguary.comparison.__main__ import LiveClientFactory, _restore, prepare
    from linguary.comparison.protocol import ProtocolError
    from linguary.comparison.llm import BudgetGuard, ObservationLedger

    for key in ('HTTP_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
        monkeypatch.delenv(key, raising=False)
    proxy_url = 'http://synthetic-user:synthetic-password@proxy.invalid:8080'
    monkeypatch.setenv('HTTPS_PROXY', proxy_url)
    value = four_chapter_protocol(tmp_path).model_copy(update={'proxy_mode': proxy_mode})
    workspace = prepare(tmp_path / 'comparison', [tmp_path / 'production'], value)
    restored, manifest = _restore(workspace.root, workspace.protected_roots, value)
    assert restored.settings.llm.proxy_mode == manifest.effective_settings['llm']['proxy_mode'] == proxy_mode
    for name in ('settings.json', 'manifest.json'):
        persisted = (workspace.root / name).read_text()
        assert proxy_url not in persisted and 'synthetic-password' not in persisted
        assert 'proxy_url' not in persisted

    clients = []
    def http_factory(**kwargs):
        assert kwargs['trust_env'] is (proxy_mode == 'auto')
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail('no HTTP expected')), **kwargs)
        clients.append(client)
        return client

    model = value.variants[0].analysis_model
    factory = LiveClientFactory(value, 'synthetic-key', http_factory)
    client = factory(model=model, settings=replace(restored.settings.llm, model=model),
        ledger=ObservationLedger(), budget=BudgetGuard(Decimal('10')))
    assert client.inner._settings.proxy_mode == proxy_mode
    assert len(clients) == 1
    await client.aclose()
    other_mode = 'direct' if proxy_mode == 'auto' else 'auto'
    assert value.content_hash != value.model_copy(update={'proxy_mode': other_mode}).content_hash
    with pytest.raises(ProtocolError, match='прокси'):
        factory(model=model, settings=replace(restored.settings.llm, model=model, proxy_mode=other_mode),
            ledger=ObservationLedger(), budget=BudgetGuard(Decimal('10')))
    assert len(clients) == 1


def test_restore_rejects_proxy_mode_not_bound_to_protocol(tmp_path):
    from linguary.comparison.__main__ import _restore, prepare
    from linguary.comparison.protocol import ProtocolError, stable_hash

    value = four_chapter_protocol(tmp_path)
    workspace = prepare(tmp_path / 'comparison', [tmp_path / 'production'], value)
    data = json.loads((workspace.root / 'settings.json').read_text())
    data['llm']['proxy_mode'] = 'direct'
    (workspace.root / 'settings.json').write_text(json.dumps(data))
    manifest = json.loads((workspace.root / 'manifest.json').read_text())
    manifest['effective_settings']['llm']['proxy_mode'] = 'direct'
    manifest['artifact_hashes']['settings.json'] = file_hash(workspace.root / 'settings.json')
    (workspace.root / 'manifest.json').write_text(json.dumps(manifest))
    (workspace.root / 'prepared.json').write_text(json.dumps({'manifest_hash': stable_hash(manifest), 'protocol_hash': value.content_hash}))
    with pytest.raises(ProtocolError, match='прокси'):
        _restore(workspace.root, workspace.protected_roots, value)


async def test_live_without_approval_makes_zero_requests(tmp_path):
    from linguary.comparison.__main__ import execute
    from linguary.comparison.protocol import ProtocolError
    calls = []
    with pytest.raises(ProtocolError):
        await execute(cli_args(tmp_path, live_protocol(), '--live'), http_factory=lambda **kwargs: calls.append(kwargs))
    assert not calls


async def test_changed_manifest_blocks_live(tmp_path, monkeypatch):
    from linguary.comparison.__main__ import execute
    from linguary.comparison.protocol import ProtocolApproval, ProtocolError
    value = four_chapter_protocol(tmp_path)
    await execute(cli_args(tmp_path, value, '--prepare'))
    manifest = tmp_path / 'comparison' / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['code_version'] = 'changed'
    manifest.write_text(json.dumps(data))
    approval = tmp_path / 'approval.json'
    approval.write_text(ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-11').model_dump_json())
    calls = []
    monkeypatch.setenv('COMPARISON_API_KEY', 'synthetic-key')
    with pytest.raises(ProtocolError):
        await execute(cli_args(tmp_path, value, '--live', approval), http_factory=lambda **kwargs: calls.append(kwargs))
    assert not calls


async def test_negative_run_keeps_production_data_root_unchanged(tmp_path):
    from linguary.comparison.__main__ import execute, parser
    from linguary.comparison.workspace import WorkspaceError
    protected = tmp_path / 'production'
    protected.mkdir()
    database = protected / 'database.sqlite'
    database.write_bytes(b'protected')
    before = file_hash(database)
    args = cli_args(tmp_path, protocol(), '--prepare')
    args.root = str(protected / 'comparison')
    with pytest.raises(WorkspaceError):
        await execute(args)
    assert file_hash(database) == before and list(protected.iterdir()) == [database]


async def test_complete_fake_comparison_to_report(tmp_path):
    from linguary.comparison.__main__ import prepare, save_result, local_report
    value = protocol()
    workspace = prepare(tmp_path / 'comparison', [tmp_path / 'production'], value)
    result = await run_comparison(value, workspace, Factory())
    from linguary.comparison.report import build_report
    direct = build_report(json.loads((workspace.root / 'manifest.json').read_text()), result.observations, result, {}, {})
    assert direct.total['unique_calls'] == 6
    assert all(details['analysis']['first_received'] == 1 for details in direct.variants.values())
    save_result(result, value, workspace.root, 10)
    report = local_report(workspace.root, workspace.protected_roots, value)
    assert report.simulated and report.decision.status == 'insufficient_data'
    assert report.total['unique_calls'] == 6
    assert (workspace.root / 'report.json').exists() and (workspace.root / 'report.md').exists()
    assert len(report.variants) == 4
    assert len(result.stage_wall_seconds) == 6
    # End-to-end wall duration includes local work; fake receipts have unknown duration.
    assert report.total['wall_seconds'] == 10 and report.total['request_seconds'] is None
    from linguary.comparison.__main__ import execute
    import zipfile
    assert await execute(cli_args(tmp_path, value, '--report'), http_factory=lambda **kwargs: pytest.fail('no report HTTP')) == 0
    mapping = json.loads((workspace.root / 'review-mapping.json').read_text())
    assert len(mapping) == 4
    for label, origin in mapping.items():
        branch = next(branch for branch in result.variants if branch.variant_id == origin['variant_id'])
        with zipfile.ZipFile(branch.epubs[origin['book_id']]) as original, zipfile.ZipFile(workspace.root / 'review-materials' / f'{label}.epub') as anonymous:
            assert original.namelist() == anonymous.namelist()
            assert all(original.read(name) == anonymous.read(name) for name in original.namelist())
    assert set(json.loads((workspace.root / 'review-materials' / 'review-sheet.json').read_text())) == set(mapping)


async def test_anonymous_epubs_exclude_model_labels(tmp_path):
    import zipfile
    from linguary.comparison.report import anonymize_epubs
    source = tmp_path / 'named.epub'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('mimetype', 'application/epub+zip')
        archive.writestr('content.opf', '<metadata><meta name="model">openai/gpt-6-luna anthropic/claude-sonnet-4.6 luna sonnet</meta></metadata>')
    outcomes = {'variants': [{'variant_id': 'luna/sonnet', 'repeat': 1, 'epubs': {'book': str(source)}}]}
    mapping = anonymize_epubs(outcomes, tmp_path)
    assert len(mapping) == 1
    for path in (tmp_path / 'review-materials').glob('*.epub'):
        with zipfile.ZipFile(path) as archive:
            contents = b''.join(archive.read(name) for name in archive.namelist())
            assert b'luna' not in contents and b'sonnet' not in contents
    assert 'luna/sonnet' not in (tmp_path / 'review-materials' / 'review-sheet.json').read_text()


@pytest.mark.parametrize("temperature_policy", ["set", "omit"])
async def test_live_admission_and_run_with_mock_transport(tmp_path, monkeypatch, temperature_policy):
    import httpx
    from linguary.comparison.__main__ import execute
    from linguary.comparison.protocol import ProtocolApproval
    value = live_protocol()
    value = four_chapter_protocol(tmp_path)
    value = value.model_copy(update={'temperature_policy': temperature_policy, 'generation': dict(value.generation, max_output_tokens=8000, temperature=None if temperature_policy == 'omit' else .5)})
    protected = tmp_path / 'production'
    protected.mkdir()
    protected_file = protected / 'database'
    protected_file.write_bytes(b'never change')
    before = file_hash(protected_file)
    await execute(cli_args(tmp_path, value, '--prepare'))
    approval = tmp_path / 'approval.json'
    approval.write_text(ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-11').model_dump_json())
    document = EbookLibEpubReader().read(EpubReadRequest(book_id='fixture', source_path=str(SYNTHETIC_EPUB), source_language='en', target_language='ru')).document
    document = replace(document, sections=document.ordered_sections[:1])
    replies = llm_replies(document)
    processing = json.loads(replies[1].content)
    for block in processing['blocks']:
        block['translated_text'] += ' Рен Колдер.'
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert ('temperature' in body) is (temperature_policy == 'set')
        assert body['max_tokens'] == value.generation['max_output_tokens']
        assert body['provider'] == {'order': ['mock-route'], 'allow_fallbacks': False, 'require_parameters': True, 'max_price': {'prompt': 1, 'completion': 1}}
        assert request.headers['X-OpenRouter-Cache'] == 'false'
        kind = body['response_format']['json_schema']['name']
        import re
        from pipeline_support import read_fixture
        aliases = tuple(dict.fromkeys(re.findall(r'\[(b\d+)\]', body['messages'][-1]['content'])))
        if 'analysis' in kind.lower():
            payload = read_fixture('analysis-template.json')
            payload['treatment_plan'] = [{'source_block_id': alias, 'treatment': 'TRANSLATE'} for alias in aliases]
        else:
            payload = {'blocks': [{'source_block_id': alias, 'treatment': 'TRANSLATE', 'translated_text': f'Русский перевод учебного блока сохраняет смысл и события главы. Рен Колдер. {alias}.', 'glossary_hints': []} for alias in aliases], 'language_notes': []}
        content = json.dumps(payload, ensure_ascii=False)
        return httpx.Response(200, json={'id': f'gen-{len(calls)}', 'model': body['model'], 'provider': 'mock-route', 'full_response_cache': False, 'context_cache': False,
                                      'usage': {'cost': .01, 'prompt_tokens': 100, 'completion_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 0}}, 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
    monkeypatch.setenv('COMPARISON_API_KEY', 'synthetic-key')
    assert await execute(cli_args(tmp_path, value, '--live', approval), http_factory=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)) == 0
    root = tmp_path / 'comparison'
    result = json.loads((root / 'result.json').read_text())
    assert len(calls) == 24 and len(result['variants']) == 4
    assert all(branch['completed'] for branch in result['variants']), result
    assert result['simulated'] is False
    assert all(manifest['temperature_policy'] == temperature_policy for manifest in result['manifests'].values())
    if temperature_policy == 'omit':
        assert all(manifest['effective_settings']['llm']['temperature'] is None for manifest in result['manifests'].values())
    assert file_hash(protected_file) == before
    # A separate report invocation reconstructs every receipt and outcome from disk.
    assert await execute(cli_args(tmp_path, value, '--report'), http_factory=lambda **kwargs: pytest.fail('no report HTTP')) == 0
    report = json.loads((root / 'report.json').read_text())
    assert report['total']['unique_calls'] == 24 and report['decision']['status'] == 'insufficient_data'
    assert 'synthetic-key' not in (root / 'report.json').read_text()


async def test_stop_keeps_receipts_and_partial_report(tmp_path):
    from linguary.adapters.llm import FakeReply
    from linguary.comparison.__main__ import prepare, save_result, local_report
    class Failure(Factory):
        def __call__(self, *, model, stage, **kwargs):
            client = super().__call__(model=model, stage=stage, **kwargs)
            if stage == 'processing':
                client.replies = (FakeReply(content='{'),)
            return client
    value = protocol()
    workspace = prepare(tmp_path / 'comparison', [tmp_path / 'production'], value)
    result = await run_comparison(value, workspace, Failure())
    save_result(result, value, workspace.root, 5)
    report = local_report(workspace.root, workspace.protected_roots, value)
    assert report.total['unique_calls'] == 6
    assert all(not repeat['completed'] and repeat['failures'] for details in report.variants.values() for repeat in details['repeats'])
    assert report.decision.status != 'candidate'


async def test_local_cleanup_removes_rejected_diagnostics_before_publication(tmp_path):
    from linguary.comparison.__main__ import prepare, cleanup_diagnostics, local_report
    value = protocol()
    workspace = prepare(tmp_path / 'comparison', [tmp_path / 'production'], value)
    diagnostics = workspace.root / 'analysis' / 'diagnostics'
    diagnostics.mkdir(parents=True)
    (diagnostics / 'receipt.json').write_text(json.dumps({'call_uid': 'receipt', 'expires_at': 999999999999, 'response': 'REJECTED PRIVATE RESPONSE'}))
    local_report(workspace.root, workspace.protected_roots, value)
    assert not list(diagnostics.iterdir())
    assert 'REJECTED PRIVATE RESPONSE' not in (workspace.root / 'report.json').read_text()


async def test_duplicate_physical_chapters_block_cli_before_http(tmp_path):
    from linguary.comparison.__main__ import execute
    from linguary.comparison.protocol import ProtocolApproval, ProtocolError
    value = live_protocol()
    await execute(cli_args(tmp_path, value, '--prepare'))
    approval = tmp_path / 'approval.json'
    approval.write_text(ProtocolApproval(protocol_hash=value.content_hash, accepted=True, accepted_by='TEST ONLY', accepted_at='2026-10-11').model_dump_json())
    calls = []
    with pytest.raises(ProtocolError):
        await execute(cli_args(tmp_path, value, '--live', approval), http_factory=lambda **kwargs: calls.append(kwargs))
    assert not calls


async def test_absent_fixed_concept_is_not_invented(tmp_path):
    value = protocol()
    value = value.model_copy(update={'fixed_glossary': dict(value.fixed_glossary, **{'Absent synthetic concept': {'decision': 'correct', 'translation': 'Отсутствующее понятие'}})})
    workspace = prepare_workspace(tmp_path / 'comparison', [tmp_path / 'production'], value)
    result = await run_comparison(value, workspace, Factory())
    assert all(branch.completed for branch in result.variants)
