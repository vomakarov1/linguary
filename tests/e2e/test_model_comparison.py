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
