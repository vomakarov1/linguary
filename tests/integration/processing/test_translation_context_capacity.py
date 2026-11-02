"""Capacity recovery keeps one durable operation and all mandatory target content."""
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from linguary.adapters.llm import FakeLLMClient
from linguary.adapters.llm.logging import LoggingLLMClient
from linguary.application.block_processing import TranslationCall
from linguary.application.context_builder import ContextBuilder
from linguary.application.ports.llm import LLMContextCapacityError
from linguary.application.recovery import RecoveryError, RecoveryPolicy, root_manifest
from linguary.application.retry_policy import RetryPolicy
from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
from linguary.composition.book_runtime import ProvenanceClient
from linguary.domain import BlockTreatment, LLMCallStage, TranslationUnit
from linguary.domain.llm_log import CallContext
from linguary.domain.recovery import RecoveryRoot, RecoveryScope
from processing_scenario import make_validator, store_processing_book, translated_reply, translation_reply, block_reply
from test_book_conditions import make_app


def capacity_call(app, fake, *, neighbours=True, max_calls=12, mixed=False):
    treatments = (BlockTreatment.TRANSLATE, BlockTreatment.READ_AS_IS, BlockTreatment.GLOSSARY_ONLY) if mixed else (BlockTreatment.TRANSLATE,)
    targets = ('He returned home.', 'The cat sat.', 'She waited.') if mixed else ('He returned home.',)
    texts = ('Optional scene before.', *targets, 'Optional scene after.') if neighbours else targets
    modes = ((BlockTreatment.TRANSLATE, *treatments, BlockTreatment.TRANSLATE) if neighbours else treatments)
    scenario = store_processing_book(app.unit_of_work, chapters=(texts,), treatments={'section-1': modes})
    section = scenario.section()
    blocks = section.blocks[1:-1] if neighbours else section.blocks
    source = app.paths.source_epub('book-1')
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'isolated synthetic source')
    with app.unit_of_work.transaction() as tx:
        tx.books.save(replace(tx.books.get('book-1'), source_epub_path=str(source)))
        conditions = app.book_conditions.create(tx, tx.books.get('book-1'), 0, continue_legacy=True)
        analysis = tx.analyses.get_for_section('book-1', section.id)
        run = tx.runs.active_for_book('book-1')
        unit = TranslationUnit.of_blocks(id='capacity-unit', blocks=blocks, plan=analysis.treatment_plan)
        manifest = root_manifest(book_id='book-1', conditions_id=conditions.id,
            glossary_sha='a' * 64, analysis_sha='b' * 64, plan_sha=None,
            ordered_blocks=unit.block_ids, treatments=tuple(t.value for t in treatments),
            boundaries=(unit.block_ids,), atomic_intervals=tuple((b,) for b in unit.block_ids),
            max_source_chars=1000, policy=RecoveryPolicy(max_calls=max_calls))
        root = RecoveryRoot(str(uuid4()), 'book-1', conditions.id, str(uuid4()),
            {**manifest, 'active_run_id': run.id, 'historical_coverage': []}, policy_version=CURRENT_TRANSLATION_POLICY)
        scope = RecoveryScope(str(uuid4()), root.id, unit.block_ids)
        tx.work.add_root(root)
        tx.work.add_scope(scope)
    context = ContextBuilder().for_translation_unit(document=scenario.document, section=section,
        unit=unit, analysis=analysis, glossary=run.glossary_snapshot, policy_version=root.policy_version)
    context = replace(context, request=context.request.with_logging(book_id='book-1', stage=LLMCallStage.TRANSLATION_UNIT))
    logger = LoggingLLMClient(fake, app.unit_of_work, 'fake', 'pinned-model')
    client = ProvenanceClient(logger, app.book_conditions, conditions.id)
    call = TranslationCall(unit_of_work=app.unit_of_work, llm_client=client,
        retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0), validator=make_validator(), sleep=no_sleep)
    return SimpleNamespace(call=call, context=context, root=root, scope=scope, unit=unit, run=run,
        document=scenario.document, request_args=dict(unit=unit, target_language='ru', note_ids=lambda: 'note', recovery=(root.id, scope.id)))


async def no_sleep(seconds):
    pass


async def test_capacity_retry_drops_only_optional_context(tmp_path):
    fake = FakeLLMClient([LLMContextCapacityError('too large'), translated_reply(('b1',))])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake)
        result = await work.call.request(work.context, **work.request_args)
        assert result.attempts == 2
        first, second = fake.requests
        assert first.instructions == second.instructions == work.context.request.instructions
        assert second.input_text == work.context.without_optional_neighbours().request.input_text
        assert 'Optional scene' in first.input_text and 'Optional scene' not in second.input_text
        assert work.unit.items[0].block.text in second.input_text
        assert first.ordered_block_ids == second.ordered_block_ids == work.unit.block_ids
        assert first.call_context.assigned_model == second.call_context.assigned_model == 'pinned-model'
        assert first.call_context.attempt_id != second.call_context.attempt_id
        assert first.call_context.root_id == second.call_context.root_id == work.root.id
        assert work.call.save(book_id='book-1', run_id=work.run.id, results=result.results)
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(work.root.id).calls_used == 2
            assert len(tx.work.list_scopes(work.root.id)) == 1
    finally:
        await app.aclose()


@pytest.mark.parametrize('neighbours,max_calls,calls', [(False, 12, 1), (True, 12, 2), (True, 1, 1)])
async def test_mandatory_only_capacity_failure_saves_nothing(tmp_path, neighbours, max_calls, calls):
    fake = FakeLLMClient(default=LLMContextCapacityError('too large'))
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake, neighbours=neighbours, max_calls=max_calls)
        with pytest.raises(RecoveryError, match='context_capacity'):
            await work.call.request(work.context, **work.request_args)
        with app.unit_of_work.transaction() as tx:
            root = tx.work.get_root(work.root.id)
            assert (root.calls_used, root.status, root.reason) == (calls, 'stopped', 'context_capacity')
            assert tx.results.list_for_run(work.run.id) == ()
            assert all(tx.work.get_binding('result', f'{work.run.id}:{b}') is None for b in work.unit.block_ids)
            assert len(tx.work.list_scopes(root.id)) == 1
        assert fake.call_count == calls
    finally:
        await app.aclose()


async def test_capacity_failure_never_splits_targets_or_switches_model(tmp_path):
    fake = FakeLLMClient(default=LLMContextCapacityError('too large'))
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake, mixed=True)
        with pytest.raises(RecoveryError, match='context_capacity'):
            await work.call.request(work.context, **work.request_args)
        assert len(fake.requests) == 2
        for request in fake.requests:
            assert request.ordered_block_ids == work.unit.block_ids
            assert request.call_context.assigned_model == 'pinned-model'
            for item in work.unit.items:
                assert item.block.text in request.input_text
                assert item.treatment.value in request.input_text
        with app.unit_of_work.transaction() as tx:
            assert len(tx.work.list_scopes(work.root.id)) == 1
            assert tx.results.list_for_run(work.run.id) == ()
    finally:
        await app.aclose()


async def test_success_preserves_all_target_modes_ids_and_order(tmp_path):
    fake = FakeLLMClient([translation_reply([
        block_reply('b1'), block_reply('b2', treatment=BlockTreatment.READ_AS_IS, translated_text=None),
        block_reply('b3', treatment=BlockTreatment.GLOSSARY_ONLY, translated_text=None,
                    hints=(('waited', 'ждала'),)),
    ])])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake, mixed=True)
        result = await work.call.request(work.context, **work.request_args)
        assert tuple(r.block_id for r in result.results) == work.unit.block_ids
        assert tuple(r.treatment for r in result.results) == tuple(item.treatment for item in work.unit.items)
        assert work.call.save(book_id='book-1', run_id=work.run.id, results=result.results)
        with app.unit_of_work.transaction() as tx:
            assert {r.block_id for r in tx.results.list_for_run(work.run.id)} == set(work.unit.block_ids)
    finally:
        await app.aclose()


@pytest.mark.parametrize('policy_version', [None, 'translation-context-future'])
async def test_root_without_supported_policy_never_sends(tmp_path, policy_version):
    fake = FakeLLMClient([translated_reply(('b1',))])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake)
        root = replace(work.root, id=str(uuid4()), root_key=str(uuid4()), policy_version=policy_version)
        scope = replace(work.scope, id=str(uuid4()), root_id=root.id)
        with app.unit_of_work.transaction() as tx:
            tx.work.add_root(root)
            tx.work.add_scope(scope)
        with pytest.raises(RecoveryError, match='conditions_unavailable'):
            await work.call.request(work.context, **{**work.request_args, 'recovery': (root.id, scope.id)})
        assert fake.call_count == 0
        with app.unit_of_work.transaction() as tx:
            assert tx.work.get_root(root.id).calls_used == 0
    finally:
        await app.aclose()


async def test_explicit_policy_without_recovery_root_never_sends(tmp_path):
    fake = FakeLLMClient([translated_reply(('b1',))])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake)
        explicit = CallContext(work.root.conditions_id, LLMCallStage.TRANSLATION_UNIT.value,
            work.scope.id, work.unit.block_ids, str(uuid4()), 0, 'pinned-model', 'a' * 64,
            root_id=work.root.id, policy_version=work.root.policy_version)
        context = replace(work.context, request=replace(work.context.request, call_context=explicit))
        with pytest.raises(RecoveryError, match='conditions_unavailable'):
            await work.call.request(context, **{**work.request_args, 'recovery': None})
        assert fake.call_count == 0
    finally:
        await app.aclose()


async def test_unrooted_legacy_interface_keeps_unknown_provenance(tmp_path):
    fake = FakeLLMClient([translated_reply(('b1',))])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake, neighbours=False)
        with app.unit_of_work.transaction() as tx:
            analysis = tx.analyses.get_for_section('book-1', work.context.section_id)
        context = ContextBuilder().for_translation_unit(document=work.document,
            section=work.document.sections[0], unit=work.unit, analysis=analysis,
            glossary=work.run.glossary_snapshot)
        context = replace(context, request=context.request.with_logging(
            book_id='book-1', stage=LLMCallStage.TRANSLATION_UNIT))
        result = await work.call.request(context, **{**work.request_args, 'recovery': None})
        assert work.call.save(book_id='book-1', run_id=work.run.id, results=result.results)
        assert fake.requests[0].call_context.root_id is None
        assert fake.requests[0].call_context.policy_version is None
        with app.unit_of_work.transaction() as tx:
            binding = tx.work.get_binding('result', f'{work.run.id}:{work.unit.block_ids[0]}')
            assert binding['payload_json'] == {'schema_version': 1}
            assert 'provenance' not in tx.work.get_artifact(binding['artifact_revision_id'])['body']
    finally:
        await app.aclose()
