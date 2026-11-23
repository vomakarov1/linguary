from dataclasses import replace
import asyncio
import pytest
from linguary.adapters.llm import FakeReply
from tests.correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from linguary.application.recovery import RecoveryError
from linguary.application.use_cases.approve_glossary import StartProcessingCommand
from linguary.application.use_cases.reset_processing import ResetProcessingCommand
from linguary.application.use_cases.process_chapter import ProcessChapterCommand
from linguary.application.translation_units import plan_translation_units
from linguary.composition import drain_tasks
from linguary.domain import TaskStatus, LLMCallStage
from test_book_conditions import make_app
from test_recovery_commit import analysis_reply, translation_reply, prepare


async def test_restart_and_manual_retry_keep_scope_limit(tmp_path):
    from linguary.application.ports.llm import LLMTransportError

    fake = FakeLLMClient([
        analysis_reply(2),
        LLMTransportError('HTTP 503'),
        LLMTransportError('HTTP 503'),
        LLMTransportError('HTTP 503'),
    ])
    app = make_app(tmp_path, fake)
    _doc, run, task = await prepare(app)
    completed = await drain_tasks(app)
    assert completed[0].status == TaskStatus.FAILED
    with app.unit_of_work.transaction() as transaction:
        root, = transaction.work.list_roots('book-1', run_id=run.id)
        assert root.calls_used == 3 and root.reason == 'scope_limit'
        root_id, root_key = root.id, root.root_key
    await app.aclose()
    later_fake = FakeLLMClient([translation_reply(2)])
    restarted = make_app(tmp_path, later_fake)
    try:
        with restarted.unit_of_work.transaction() as transaction:
            transaction.tasks.save(transaction.tasks.get(task.id).retry())
        completed = await drain_tasks(restarted)
        assert completed[0].status == TaskStatus.FAILED
        assert later_fake.requests == []
        with restarted.unit_of_work.transaction() as transaction:
            persisted = transaction.work.get_root(root_id)
            assert persisted.root_key == root_key and persisted.calls_used == 3
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('policy_changed', [False, True])
async def test_restart_keeps_root_manifest_and_rejects_incompatible_notes_policy(tmp_path, monkeypatch, policy_changed):
    from linguary.application import recovery
    from linguary.application.ports.llm import LLMTransportError

    fake = FakeLLMClient([
        analysis_reply(2),
        LLMTransportError('HTTP 503'),
        LLMTransportError('HTTP 503'),
        LLMTransportError('HTTP 503'),
    ])
    app = make_app(tmp_path, fake)
    _doc, run, task = await prepare(app)
    assert (await drain_tasks(app))[0].status == TaskStatus.FAILED
    with app.unit_of_work.transaction() as transaction:
        before, = transaction.work.list_roots('book-1', run_id=run.id)
        assert before.calls_used == 3
        assert before.manifest['notes_policy'] == recovery.NOTES_POLICY
    await app.aclose()

    if policy_changed:
        monkeypatch.setattr(recovery, 'NOTES_POLICY', 'unsupported-notes-policy')
    later_fake = FakeLLMClient([translation_reply(2)])
    restarted = make_app(tmp_path, later_fake)
    try:
        with restarted.unit_of_work.transaction() as transaction:
            transaction.tasks.save(transaction.tasks.get(task.id).retry())
        completed = await drain_tasks(restarted)
        assert completed[0].status == TaskStatus.FAILED
        if policy_changed:
            assert 'conditions_unavailable' in completed[0].last_error.message
        assert later_fake.requests == []
        with restarted.unit_of_work.transaction() as transaction:
            after, = transaction.work.list_roots('book-1', run_id=run.id)
            assert after.id == before.id
            assert after.root_key == before.root_key
            assert after.calls_used == before.calls_used
            assert after.manifest == before.manifest
    finally:
        await restarted.aclose()


async def test_reset_preserves_historical_counter_and_creates_new_operation(tmp_path):
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2, bad=True), translation_reply(2, bad=True),
                          translation_reply(1), translation_reply(1), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, _task = await prepare(app)
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
        with app.unit_of_work.transaction() as transaction:
            before, = transaction.work.list_roots('book-1', run_id=run.id)
        app.reset_processing(ResetProcessingCommand('book-1'))
        with app.unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(before.id)
            assert root.calls_used == 4 and root.generation > before.generation
            assert not transaction.work.root_coverage_complete(root.id)
        new_run = app.start_processing()(StartProcessingCommand('book-1'))
        await app.processing.start('book-1')
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=new_run.id)
            assert root.id != before.id and root.root_key != before.root_key and root.calls_used == 1
            assert transaction.work.get_root(before.id).calls_used == 4
            assert len(transaction.results.list_for_run(new_run.id)) == 2
    finally:
        await app.aclose()


async def prepared_root(app, run, task):
    conditions = app.book_conditions.require('book-1')
    async with app.book_runtime.lease(conditions.id, 'processing') as generation:
        processor = generation.processing._process_chapter
        work = processor._load(ProcessChapterCommand('book-1', task.scope_id, task.id))
        units = plan_translation_units(work.section.ordered_blocks, work.analysis.treatment_plan,
            id_factory=lambda: 'unstable-unit', max_source_chars=processor._max_unit_source_chars)
        roots = processor._recovery.prepare_roots(work, units, max_source_chars=processor._max_unit_source_chars,
                                                 validator=processor._validator)
    with app.unit_of_work.transaction() as transaction:
        scope, = transaction.work.list_scopes(roots[0])
        context = transaction.work.begin_recovery_attempt(roots[0], scope.id, task_id=task.id,
            run_id=run.id, stage='translation_unit', assigned_model='pinned-model', prompt_hash='a' * 64,
            policy_version=transaction.work.get_root(roots[0]).policy_version)
    return roots[0], context


@pytest.mark.parametrize('state', ['created', 'sent', 'validated'])
async def test_crash_boundaries_do_not_reset_counter_or_repeat_unknown(tmp_path, state):
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, task = await prepare(app)
        root_id, context = await prepared_root(app, run, task)
        with app.unit_of_work.transaction() as transaction:
            if state in {'sent', 'validated'}:
                root = transaction.work.get_root(root_id)
                transaction.work.mark_attempt_sent(
                    context.attempt_id, expected_root_generation=root.generation)
            if state == 'validated':
                transaction.work.transition_attempt(context.attempt_id, 'validated', expected_state='sent')
        await app.runner.recover_interrupted()
        completed = await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(root_id)
            if state == 'created':
                assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
                assert root.calls_used == 1
                assert transaction.work.get_attempt(context.attempt_id).state == 'cancelled'
            else:
                assert completed[0].status == TaskStatus.FAILED
                assert root.reason == 'interrupted' and root.calls_used == 1
                assert len(fake.scripted_requests) == 1
                assert transaction.results.list_for_run(run.id) == ()
    finally:
        await app.aclose()


async def test_crash_after_commit_before_task_succeed_does_not_send_again(tmp_path, monkeypatch):
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, task = await prepare(app)
        original = app.task_queue.save
        async def crash(task):
            if task.task_type.value == 'process_translation_unit' and task.status == TaskStatus.SUCCEEDED:
                raise RuntimeError('crash after commit')
            await original(task)
        monkeypatch.setattr(app.task_queue, 'save', crash)
        # The chapter may mark itself succeeded in its short completion transaction;
        # the runner's final acknowledgement is the deliberately interrupted boundary.
        with pytest.raises(RuntimeError):
            await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert root.status == 'complete' and root.calls_used == 1
            assert len(transaction.results.list_for_run(run.id)) == 2
            stored_task = transaction.tasks.get(task.id)
            if stored_task.status == TaskStatus.SUCCEEDED:
                transaction.tasks.save(replace(stored_task, status=TaskStatus.PENDING))
        monkeypatch.setattr(app.task_queue, 'save', original)
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
        assert len(fake.scripted_requests) == 2
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(root.id).calls_used == 1
    finally:
        await app.aclose()


async def test_global_root_limit_stops_finite_tree(tmp_path):
    class Dynamic(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.CHAPTER_ANALYSIS:
                self.default = analysis_reply(8)
            elif len(request.ordered_block_ids) > 1:
                self.default = FakeReply(content='{')
            else:
                self.default = translation_reply(1, bad=False)
            return await super().generate(request)
    fake = Dynamic()
    app = make_app(tmp_path, fake)
    try:
        _doc, run, _task = await prepare(app, blocks=8)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.FAILED
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert root.reason == 'limit' and root.calls_used == 12
            scopes = transaction.work.list_scopes(root.id)
            assert sum(scope.status != 'split' for scope in scopes) <= 8
            assert len([call for call in transaction.llm_calls.list_for_book('book-1') if call.stage == LLMCallStage.TRANSLATION_UNIT]) == 12
    finally:
        await app.aclose()


async def test_resume_only_unresolved_child_after_saved_child_crash(tmp_path, monkeypatch):
    from linguary.application.block_processing import TranslationCall
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2, bad=True), translation_reply(2, bad=True), translation_reply(1)])
    app = make_app(tmp_path, fake)
    _doc, run, task = await prepare(app)
    original = TranslationCall.save
    crashed = False
    def crash_after_first_child(self, **kwargs):
        nonlocal crashed
        saved = original(self, **kwargs)
        if saved and len(kwargs['results']) == 1 and not crashed:
            crashed = True
            raise RuntimeError('crash after accepted child')
        return saved
    monkeypatch.setattr(TranslationCall, 'save', crash_after_first_child)
    assert (await drain_tasks(app))[0].status == TaskStatus.FAILED
    with app.unit_of_work.transaction() as transaction:
        root, = transaction.work.list_roots('book-1', run_id=run.id)
        assert root.calls_used == 3
        accepted, = transaction.results.list_for_run(run.id)
        previous_binding = transaction.work.get_binding('result', f'{run.id}:{accepted.block_id}')
    await app.aclose()
    monkeypatch.setattr(TranslationCall, 'save', original)
    remaining = FakeLLMClient([translation_reply(1)])
    restarted = make_app(tmp_path, remaining)
    try:
        with restarted.unit_of_work.transaction() as transaction:
            transaction.tasks.save(transaction.tasks.get(task.id).retry())
        completed = await drain_tasks(restarted)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        assert len(remaining.scripted_requests) == 1
        assert accepted.block_id not in remaining.requests[0].ordered_block_ids
        with restarted.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(root.id).calls_used == 4
            assert transaction.work.get_binding('result', f'{run.id}:{accepted.block_id}') == previous_binding
    finally:
        await restarted.aclose()


async def test_late_same_run_revoked_attempt_is_audit_only(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    analysis = FakeLLMClient([analysis_reply(2)])
    processing = FakeLLMClient.always(translation_reply(2))
    count = 0
    class Delayed:
        async def generate(self, request):
            nonlocal count
            if request.stage == LLMCallStage.CHAPTER_ANALYSIS:
                return await analysis.generate(request)
            count += 1
            if count == 1:
                entered.set()
                await release.wait()
            return await processing.generate(request)
    app = make_app(tmp_path, Delayed())
    app.rate_limiter.set_max_concurrency(2)
    try:
        doc, run, task = await prepare(app)
        late = asyncio.create_task(app.processing.run_task(task))
        await asyncio.wait_for(entered.wait(), 3)
        with app.unit_of_work.transaction() as transaction:
            transaction.work.invalidate_run_coverage('book-1', run.id)
        await app.processing.run_task(task)
        with app.unit_of_work.transaction() as transaction:
            binding = transaction.work.get_binding('result', f'{run.id}:{doc.sections[0].blocks[0].id}')
        release.set()
        with pytest.raises(RecoveryError, match='interrupted'):
            await late
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_binding('result', f'{run.id}:{doc.sections[0].blocks[0].id}') == binding
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            assert root.calls_used == 2 and root.status == 'complete'
            calls = [call for call in transaction.llm_calls.list_for_book('book-1') if call.stage == LLMCallStage.TRANSLATION_UNIT]
            assert sorted(call.diagnostic_meta['decision'] for call in calls) == ['accepted', 'stale']
    finally:
        release.set()
        await app.aclose()


async def test_complete_requires_union_not_task_success(tmp_path):
    from linguary.application.block_processing import complete_processing
    from linguary.domain import BookStatus, TaskType
    app = make_app(tmp_path, FakeLLMClient([analysis_reply(2)]))
    try:
        _doc, run, task = await prepare(app)
        root_id, _context = await prepared_root(app, run, task)
        with app.unit_of_work.transaction() as transaction:
            transaction.tasks.save(replace(transaction.tasks.get(task.id), status=TaskStatus.SUCCEEDED))
        with pytest.raises(RecoveryError):
            complete_processing(app.unit_of_work, book_id='book-1', finished_task_id=task.id,
                                task_types=(TaskType.PROCESS_TRANSLATION_UNIT,))
        with app.unit_of_work.transaction() as transaction:
            assert transaction.books.get('book-1').status == BookStatus.PROCESSING
            assert not transaction.work.root_coverage_complete(root_id)
    finally:
        await app.aclose()


async def test_export_still_409_and_regenerate_guard_for_incomplete_root(tmp_path):
    from linguary.domain import BookStatus
    from linguary.application.use_cases.regenerate_block import RegenerateBlockError
    from linguary.web import create_app
    from fastapi.testclient import TestClient
    app = make_app(tmp_path, FakeLLMClient([analysis_reply(2)]))
    try:
        doc, run, task = await prepare(app)
        await prepared_root(app, run, task)
        with app.unit_of_work.transaction() as transaction:
            transaction.tasks.save(replace(transaction.tasks.get(task.id), status=TaskStatus.SUCCEEDED))
            transaction.books.save(replace(transaction.books.get('book-1'), status=BookStatus.READY_TO_EXPORT))
        with TestClient(create_app(application=app, run_background_worker=False), base_url=app.settings.server.base_url) as client:
            assert client.post('/books/book-1/export').status_code == 409
        with pytest.raises(RegenerateBlockError):
            await app.regeneration.start('book-1', doc.sections[0].blocks[0].id)
    finally:
        await app.aclose()


async def test_regeneration_stales_review_and_preserves_root_coverage(tmp_path):
    from uuid import uuid4
    from linguary.domain.quality_review import QualityReview
    changed = FakeReply.of_payload(dict(blocks=[dict(source_block_id='b1', treatment='TRANSLATE',
        translated_text='Он возвратился домой.', glossary_hints=[])], language_notes=[]))
    app = make_app(tmp_path, FakeLLMClient([analysis_reply(2), translation_reply(2), changed]))
    try:
        doc, run, _task = await prepare(app)
        assert (await drain_tasks(app))[0].status == TaskStatus.SUCCEEDED
        conditions = app.book_conditions.require('book-1')
        review_id = str(uuid4())
        with app.unit_of_work.transaction() as transaction:
            transaction.work.add_review(QualityReview(review_id, 'book-1', 'explicit-review-fixture',
                {'schema_version': 1}, {'schema_version': 1}, 10, conditions_id=conditions.id, state='completed'))
        await app.regeneration.start('book-1', doc.sections[0].blocks[0].id)
        completed = await drain_tasks(app)
        assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_review(review_id).state == 'stale'
            root, = (root for root in transaction.work.list_roots('book-1', run_id=run.id)
                     if root.operation_kind == 'translation')
            assert root.calls_used == 1
            assert transaction.work.root_coverage_complete(root.id)
            assert transaction.results.get(run.id, doc.sections[0].blocks[0].id).translated_text == 'Он возвратился домой.'
    finally:
        await app.aclose()


@pytest.mark.parametrize('invalid_historical', [False, True])
async def test_legacy_partial_coverage_keeps_unknown_attempt_and_calls_only_unresolved(tmp_path, invalid_historical):
    from processing_scenario import store_processing_book, SOURCE_TEXT, TRANSLATED_TEXT
    from test_book_conditions import choose_legacy_fixture
    from linguary.domain import BlockProcessingResult, BlockTreatment
    fake = FakeLLMClient([translation_reply(1)])
    app = make_app(tmp_path, fake)
    try:
        scenario = store_processing_book(app.unit_of_work, chapters=((SOURCE_TEXT, SOURCE_TEXT),))
        first, second = scenario.block_ids()
        historical = BlockProcessingResult(first, BlockTreatment.TRANSLATE,
            translated_text='«Он вернулся домой.' if invalid_historical else TRANSLATED_TEXT)
        with app.unit_of_work.transaction() as transaction:
            transaction.results.replace_results(scenario.run_id, (historical,))
        choose_legacy_fixture(app)
        if invalid_historical:
            from linguary.application.retry_policy import ResultValidationError
            with pytest.raises(ResultValidationError):
                await app.processing.start('book-1')
        else:
            await app.processing.start('book-1')
        completed = await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_binding('result', f'{scenario.run_id}:{first}') is None
            assert transaction.results.get(scenario.run_id, first) == historical
            conditions = transaction.work.latest_conditions('book-1')
            assert conditions.reproducibility == 'legacy_partial'
            if invalid_historical:
                assert completed == ()
                assert transaction.tasks.list_for_book('book-1') == ()
                assert fake.requests == []
                assert transaction.results.get(scenario.run_id, second) is None
            else:
                assert completed[0].status == TaskStatus.SUCCEEDED, completed[0].last_error
                assert len(fake.scripted_requests) == 1 and fake.requests[0].ordered_block_ids == (second,)
                root, = transaction.work.list_roots('book-1', run_id=scenario.run_id)
                assert tuple(root.manifest['historical_coverage']) == (first,)
                assert root.calls_used == 1 and transaction.work.root_coverage_complete(root.id)
                assert transaction.work.get_binding('result', f'{scenario.run_id}:{second}')['attempt_id'] is not None
    finally:
        await app.aclose()


_AUDIT_FIELDS = {
    'schema_version', 'predecessor_root_id', 'observed_generation', 'previous_reason',
    'had_unknown_outcome', 'acknowledged_duplicate_billing_risk', 'restarted_at',
}
_UNKNOWN_RESTART = 'явное подтверждение перезапуска после неизвестного исхода'


def _stop_as(app, root, reason):
    with app.unit_of_work.transaction() as transaction:
        transaction.work.stop_root(root.id, reason, expected_generation=root.generation)
        return transaction.work.get_root(root.id)


def _restart(app, root, *, acknowledge=False):
    from linguary.application.use_cases.restart_translation import RestartTranslationCommand

    return app.restart_translation(RestartTranslationCommand(
        root.book_id, root.id, root.generation, acknowledge_unknown_outcome=acknowledge,
    ))


def _plant_unknown_attempt(app, root, run):
    from uuid import uuid4

    from linguary.domain.recovery import CallAttempt

    with app.unit_of_work.transaction() as transaction:
        scope = next(item for item in transaction.work.list_scopes(root.id) if item.status != 'split')
        ordinal = max((item.ordinal for item in transaction.work.list_root_attempts(root.id)), default=0) + 1
        attempt = CallAttempt(
            str(uuid4()), root.book_id, state='unknown', generation=scope.generation, ordinal=ordinal,
            conditions_id=root.conditions_id, run_id=run.id, root_id=root.id, scope_id=scope.id,
        )
        transaction.work.add_attempt(attempt)
        return attempt.id


def _assert_restart_audit(successor, *, predecessor, generation, reason, had_unknown, acknowledged):
    from datetime import datetime

    audit = successor.compatibility
    assert set(audit) == _AUDIT_FIELDS
    assert audit['schema_version'] == 1
    assert audit['predecessor_root_id'] == predecessor
    assert audit['observed_generation'] == generation
    assert audit['previous_reason'] == reason
    assert audit['had_unknown_outcome'] is had_unknown
    assert audit['acknowledged_duplicate_billing_risk'] is acknowledged
    restarted_at = datetime.fromisoformat(audit['restarted_at'])
    assert restarted_at.tzinfo is not None and restarted_at.utcoffset().total_seconds() == 0


async def test_restart_translation_allows_unknown_outcome_with_acknowledgement_and_audit(tmp_path):
    from test_restart_translation import make_partial_root

    app, _, _, run, _, root, _, _ = await make_partial_root(tmp_path)
    try:
        unknown = _stop_as(app, root, 'unknown_outcome')
        result = _restart(app, unknown, acknowledge=True)
        with app.unit_of_work.transaction() as transaction:
            successor = transaction.work.get_root(result.successor_root_id)
            _assert_restart_audit(
                successor, predecessor=unknown.id, generation=unknown.generation,
                reason='unknown_outcome', had_unknown=True, acknowledged=True,
            )
            assert transaction.work.get_root(unknown.id).superseded_by == successor.id
            assert (successor.status, successor.reason) == ('pending', None)
        limited = _stop_as(app, successor, 'scope_limit')
        _plant_unknown_attempt(app, limited, run)
        acknowledged = _restart(app, limited, acknowledge=True)
        with app.unit_of_work.transaction() as transaction:
            followed = transaction.work.get_root(acknowledged.successor_root_id)
            _assert_restart_audit(
                followed, predecessor=limited.id, generation=limited.generation,
                reason='scope_limit', had_unknown=True, acknowledged=True,
            )
    finally:
        await app.aclose()


async def test_restart_translation_rejects_unknown_without_acknowledgement(tmp_path):
    from linguary.application.use_cases.restart_translation import RestartTranslationError
    from test_restart_translation import make_partial_root

    app, _, _, run, _, root, _, _ = await make_partial_root(tmp_path)
    try:
        with app.unit_of_work.transaction() as transaction:
            assert (root.status, root.reason) == ('stopped', 'scope_limit')
            assert not any(item.state == 'unknown' for item in transaction.work.list_root_attempts(root.id))
        _plant_unknown_attempt(app, root, run)
        with pytest.raises(RestartTranslationError, match=_UNKNOWN_RESTART):
            _restart(app, root)
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(root.id).superseded_by is None
            attempt_id = next(
                item.id for item in transaction.work.list_root_attempts(root.id) if item.state == 'unknown'
            )
            transaction.work.transition_attempt(attempt_id, 'failed', expected_state='unknown')
        interrupted = _stop_as(app, root, 'interrupted')
        with pytest.raises(RestartTranslationError, match=_UNKNOWN_RESTART):
            _restart(app, interrupted)
        with app.unit_of_work.transaction() as transaction:
            stored = transaction.work.get_root(root.id)
            assert stored.superseded_by is None and stored.reason == 'interrupted'
            assert not any(item.state == 'unknown' for item in transaction.work.list_root_attempts(root.id))
    finally:
        await app.aclose()


async def test_restart_translation_allows_unsafe_partition_and_output_truncated(tmp_path):
    from linguary.application.use_cases.restart_translation import RestartTranslationError
    from test_restart_translation import make_partial_root

    app, _, _, _, _, root, _, _ = await make_partial_root(tmp_path)
    try:
        assert (root.status, root.reason) == ('stopped', 'scope_limit')
        opened = _restart(app, root)
        with app.unit_of_work.transaction() as transaction:
            successor = transaction.work.get_root(opened.successor_root_id)
            _assert_restart_audit(
                successor, predecessor=root.id, generation=root.generation,
                reason='scope_limit', had_unknown=False, acknowledged=False,
            )
        refused = _stop_as(app, successor, 'provider_error')
        with pytest.raises(RestartTranslationError, match='нельзя безопасно'):
            _restart(app, refused)
        current = refused
        for reason in ('unsafe_partition', 'output_truncated', 'context_capacity', 'provider_refusal', 'limit'):
            current = _stop_as(app, current, reason)
            restarted = _restart(app, current)
            with app.unit_of_work.transaction() as transaction:
                successor = transaction.work.get_root(restarted.successor_root_id)
                _assert_restart_audit(
                    successor, predecessor=current.id, generation=current.generation,
                    reason=reason, had_unknown=False, acknowledged=False,
                )
                assert transaction.work.get_root(current.id).superseded_by == successor.id
                current = successor
    finally:
        await app.aclose()


async def test_extend_recovery_limit_use_case_enforces_ownership_and_reason_codes(tmp_path):
    from datetime import UTC, datetime
    from uuid import uuid4

    from linguary.application.ports.persistence import BookRecord
    from linguary.application.use_cases.extend_recovery_limit import (
        ExtendRecoveryLimitCommand,
        ExtendRecoveryLimitConflict,
        ExtendRecoveryLimitError,
        ExtendRecoveryLimitNotFoundError,
    )
    from linguary.domain import BookMetadata, BookStatus, ProcessingRunStatus

    fake = FakeLLMClient([analysis_reply(2), FakeReply(content='{')])
    app = make_app(tmp_path, fake)
    settings = replace(app.settings, recovery=replace(app.settings.recovery, max_calls=1))
    app.runtime_settings.publish(app.runtime_settings.prepare(settings), 1)
    try:
        _, run, _ = await prepare(app)
        assert (await drain_tasks(app))[0].status == TaskStatus.FAILED
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            transaction.books.add(BookRecord(
                'book-2', BookMetadata('Other'), 'en', 'ru', BookStatus.PROCESSING,
                str(app.paths.source_epub('book-1')), datetime.now(UTC),
            ))
        assert (root.status, root.reason) == ('stopped', 'limit')

        def extend(book_id, root_id, generation, calls):
            return app.extend_recovery_limit(ExtendRecoveryLimitCommand(
                book_id=book_id, root_id=root_id, additional_calls=calls, expected_generation=generation,
            ))

        with pytest.raises(ExtendRecoveryLimitNotFoundError) as missing_book:
            extend('missing-book', root.id, root.generation, 1)
        assert missing_book.value.status_code == 404 and missing_book.value.reason_code == 'not_found'
        with pytest.raises(ExtendRecoveryLimitNotFoundError) as missing_root:
            extend('book-1', str(uuid4()), root.generation, 1)
        assert missing_root.value.reason_code == 'not_found'
        with pytest.raises(ExtendRecoveryLimitNotFoundError) as foreign:
            extend('book-2', root.id, root.generation, 1)
        assert foreign.value.reason_code == 'not_found'

        with pytest.raises(ExtendRecoveryLimitError) as invalid:
            extend('book-1', root.id, root.generation, 0)
        assert type(invalid.value) is ExtendRecoveryLimitError and invalid.value.reason_code == 'invalid_limit'
        with pytest.raises(ExtendRecoveryLimitConflict) as stale:
            extend('book-1', root.id, root.generation + 1, 1)
        assert stale.value.reason_code == 'generation_conflict'

        _plant_unknown_attempt(app, root, run)
        with pytest.raises(ExtendRecoveryLimitConflict) as unknown:
            extend('book-1', root.id, root.generation, 1)
        assert unknown.value.reason_code == 'unknown'
        with app.unit_of_work.transaction() as transaction:
            stored = transaction.work.get_root(root.id)
            assert (stored.generation, stored.extra_calls, stored.status, stored.reason) == (
                root.generation, root.extra_calls, 'stopped', 'limit',
            )
            attempt_id = next(
                item.id for item in transaction.work.list_root_attempts(root.id) if item.state == 'unknown'
            )
            transaction.work.transition_attempt(attempt_id, 'failed', expected_state='unknown')
            transaction.runs.save(replace(transaction.runs.get(run.id), status=ProcessingRunStatus.CANCELLED))
        with pytest.raises(ExtendRecoveryLimitError) as unavailable:
            extend('book-1', root.id, root.generation, 1)
        assert type(unavailable.value) is ExtendRecoveryLimitError
        assert unavailable.value.reason_code == 'conditions_unavailable'
        with app.unit_of_work.transaction() as transaction:
            transaction.runs.save(replace(transaction.runs.get(run.id), status=ProcessingRunStatus.ACTIVE))
        scoped = _stop_as(app, root, 'scope_limit')
        with pytest.raises(ExtendRecoveryLimitConflict) as conflict:
            extend('book-1', scoped.id, scoped.generation, 1)
        assert conflict.value.reason_code == 'state_conflict'
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(scoped.id).reason == 'scope_limit'
        limited = _stop_as(app, scoped, 'limit')
        result = extend('book-1', limited.id, limited.generation, 2)
        assert result.root_id == limited.id
        assert result.generation == limited.generation + 1
        assert result.extra_calls == limited.extra_calls + 2
        assert result.calls_used == limited.calls_used
        assert result.limit == limited.manifest['policy']['max_calls'] + result.extra_calls
        with app.unit_of_work.transaction() as transaction:
            updated = transaction.work.get_root(limited.id)
            assert (updated.status, updated.reason, updated.extra_calls, updated.generation) == (
                'pending', None, result.extra_calls, result.generation,
            )
        with pytest.raises(ExtendRecoveryLimitConflict) as raced:
            extend('book-1', limited.id, limited.generation, 1)
        assert raced.value.reason_code == 'generation_conflict'
        with pytest.raises(ExtendRecoveryLimitConflict) as pending:
            extend('book-1', limited.id, result.generation, 1)
        assert pending.value.reason_code == 'state_conflict'
    finally:
        await app.aclose()
