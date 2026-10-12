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
