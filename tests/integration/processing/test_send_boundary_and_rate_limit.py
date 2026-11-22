"""Send boundary sits after the rate limiter, and drain recovers only when alone."""
import asyncio
from dataclasses import replace

from linguary.adapters.runtime import ExecutorLock
from linguary.application.ports.llm import (
    LLMConfigurationError, LLMContextCapacityError, LLMProviderError, LLMResponseFormatError,
    LLMTransportError,
)
from linguary.application.use_cases.restart_translation import RestartTranslationCommand
from linguary.composition import drain_tasks
from linguary.domain import LLMCallStage, ProcessingTask, TaskStatus, TaskType
from tests.correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from test_book_conditions import make_app, store_imported
from test_recovery_commit import analysis_reply, prepare, translation_reply


def _one_attempt(app):
    tuned = replace(app.settings, llm=replace(app.settings.llm, max_attempts=1))
    app.runtime_settings.publish(app.runtime_settings.prepare(tuned), app.runtime_settings.revision + 1)


def _processing_client(app):
    for entry in app.book_runtime._cache.values():
        return entry.generation.processing._process_chapter._call._llm_client
    raise AssertionError('processing client was not built')


def _arm_rate_limit(app):
    entered, release = asyncio.Event(), asyncio.Event()

    async def _sleep(_delay):
        entered.set()
        await release.wait()

    app.rate_limiter._sleep = _sleep
    app.rate_limiter.hold(30)
    return entered, release


def _open_gate(app, release):
    with app.rate_limiter._lock:
        app.rate_limiter._provider_blocked_until = None
        app.rate_limiter._last_admitted = None
    release.set()


def _translation_requests(fake, attempt_id=None):
    requests = [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT]
    if attempt_id is None:
        return requests
    return [request for request in requests
            if request.call_context is not None and request.call_context.attempt_id == attempt_id]


async def _running_attempt(app, run_id):
    with app.unit_of_work.transaction() as transaction:
        root, = transaction.work.list_roots('book-1', run_id=run_id)
        attempt, = transaction.work.list_root_attempts(root.id)
        return root, attempt


async def test_cancel_while_waiting_in_rate_limiter_stays_not_sent_without_budget_burn(tmp_path):
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, _task = await prepare(app)
        entered, _release = _arm_rate_limit(app)
        worker = asyncio.create_task(drain_tasks(app))
        await asyncio.wait_for(entered.wait(), 3)
        _root, attempt = await _running_attempt(app, run.id)
        assert attempt.state == 'created'
        assert _root.calls_used == 0
        worker.cancel()
        await asyncio.wait({worker})
        with app.unit_of_work.transaction() as transaction:
            saved = transaction.work.get_attempt(attempt.id)
            root = transaction.work.get_root(_root.id)
            scope = transaction.work.get_scope(saved.scope_id)
        assert saved.state == 'cancelled'
        assert scope.status == 'pending'
        assert root.calls_used == 0
        assert _translation_requests(fake) == []
    finally:
        await app.aclose()


async def test_restarted_root_waiting_in_rate_limiter_never_sends_http(tmp_path):
    fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, _task = await prepare(app)
        entered, release = _arm_rate_limit(app)
        worker = asyncio.create_task(drain_tasks(app))
        await asyncio.wait_for(entered.wait(), 3)
        root, attempt = await _running_attempt(app, run.id)
        assert attempt.state == 'created'
        assert root.calls_used == 0
        app.restart_translation(RestartTranslationCommand(
            book_id='book-1', root_id=root.id, expected_generation=root.generation,
            acknowledge_unknown_outcome=True))
        _open_gate(app, release)
        await asyncio.wait_for(worker, 5)
        assert _translation_requests(fake, attempt.id) == []
        with app.unit_of_work.transaction() as transaction:
            assert transaction.work.get_root(root.id).calls_used == 0
            assert transaction.work.get_attempt(attempt.id).state != 'sent'
            assert transaction.work.get_attempt(attempt.id).state != 'unknown'
    finally:
        await app.aclose()


def _known_send_errors():
    return (
        LLMResponseFormatError('format'),
        LLMContextCapacityError('capacity'),
        LLMConfigurationError('config'),
        LLMProviderError('provider'),
        LLMTransportError('HTTP 429', outcome='known_transient'),
        LLMTransportError('HTTP 503', outcome='known_transient'),
    )


async def test_known_format_and_429_errors_do_not_turn_attempt_into_unknown(tmp_path):
    for index, error in enumerate(_known_send_errors()):
        case = tmp_path / f'case-{index}'
        case.mkdir()
        fake = FakeLLMClient([analysis_reply(2), error, translation_reply(2)])
        app = make_app(case, fake)
        _one_attempt(app)
        seen = []
        try:
            _doc, run, _task = await prepare(app)
            client = _processing_client(app)
            original = client.generate

            async def generate(request, _original=original, _seen=seen, _app=app):
                try:
                    return await _original(request)
                except Exception:
                    attempt_id = request.call_context.attempt_id
                    with _app.unit_of_work.transaction() as transaction:
                        _seen.append(transaction.work.get_attempt(attempt_id).state)
                    raise

            client.generate = generate
            completed = await drain_tasks(app)
            assert completed[0].status == TaskStatus.FAILED, error
            # Logging finalizes a format error to rejected before the exception leaves the client.
            expected = 'rejected' if isinstance(error, LLMResponseFormatError) else 'sent'
            assert seen == [expected], (type(error).__name__, seen)
            with app.unit_of_work.transaction() as transaction:
                root, = transaction.work.list_roots('book-1', run_id=run.id)
                attempts = transaction.work.list_root_attempts(root.id)
            assert attempts and all(attempt.state != 'unknown' for attempt in attempts), type(error).__name__
            assert root.reason != 'unknown_outcome', type(error).__name__
        finally:
            await app.aclose()


async def test_drain_tasks_recovers_interrupted_when_alone_under_executor_lock(tmp_path):
    app = make_app(tmp_path, FakeLLMClient())
    try:
        store_imported(app)
        with app.unit_of_work.transaction() as transaction:
            transaction.tasks.add(ProcessingTask(
                id='task-interrupted', book_id='book-1', task_type=TaskType.ANALYZE_CHAPTER,
                scope_id='section-1').start())
        await drain_tasks(app)
        with app.unit_of_work.transaction() as transaction:
            recovered = transaction.tasks.get('task-interrupted')
        assert recovered.status is TaskStatus.FAILED
        assert recovered.last_error is not None and recovered.last_error.detail == 'Interrupted'

        with app.unit_of_work.transaction() as transaction:
            transaction.tasks.add(ProcessingTask(
                id='task-live', book_id='book-1', task_type=TaskType.EXPORT_BOOK,
                scope_id='export').start())
        other = ExecutorLock(app.paths.executor_lock)
        assert other.join() is True
        other.share()
        try:
            await drain_tasks(app)
        finally:
            other.leave()
        with app.unit_of_work.transaction() as transaction:
            assert transaction.tasks.get('task-live').status is TaskStatus.RUNNING
    finally:
        await app.aclose()


async def test_provenance_send_gate_commits_cancel_and_stop_on_rejection(tmp_path):
    """Send-gate rejections by limit, scope_limit, conditions conflict, and interrupted commit cancellation/stop."""
    from linguary.adapters.db.models import RecoveryRootRow
    from linguary.domain.recovery import CallAttempt
    from uuid import uuid4

    for reason in ('limit', 'scope_limit', 'conditions_revision_conflict', 'interrupted'):
        case = tmp_path / reason
        case.mkdir()
        fake = FakeLLMClient([analysis_reply(2), translation_reply(2)])
        app = make_app(case, fake)
        try:
            _doc, run, _task = await prepare(app)
            entered, release = _arm_rate_limit(app)
            worker = asyncio.create_task(drain_tasks(app))
            await asyncio.wait_for(entered.wait(), 3)
            root, attempt = await _running_attempt(app, run.id)
            scope_id = attempt.scope_id
            with app.unit_of_work.transaction() as transaction:
                if reason == 'limit':
                    row = transaction.work._session.get(RecoveryRootRow, root.id)
                    row.calls_used = row.manifest_json['policy']['max_calls'] + row.extra_calls
                elif reason == 'scope_limit':
                    policy = root.manifest['policy']
                    for idx in range(policy['max_scope_calls']):
                        transaction.work.add_attempt(CallAttempt(
                            str(uuid4()), 'book-1', state='rejected',
                            generation=attempt.generation, ordinal=100 + idx,
                            conditions_id=attempt.conditions_id, task_id=attempt.task_id,
                            run_id=run.id, root_id=root.id, scope_id=scope_id,
                            context={'schema_version': 1, 'root_generation': root.generation},
                        ))
                elif reason == 'conditions_revision_conflict':
                    current_cond = transaction.work.latest_conditions('book-1')
                    from linguary.domain.book_conditions import BookConditions
                    from datetime import UTC, datetime
                    transaction.work.add_conditions(BookConditions(
                        str(uuid4()), 'book-1', current_cond.revision + 1,
                        dict(current_cond.payload), datetime.now(UTC),
                    ))
                elif reason == 'interrupted':
                    transaction.work.pause_root(root.id, 'user_pause', expected_generation=root.generation)
            _open_gate(app, release)
            await asyncio.wait_for(worker, 5)
            assert _translation_requests(fake, attempt.id) == []
            with app.unit_of_work.transaction() as transaction:
                saved_attempt = transaction.work.get_attempt(attempt.id)
                saved_scope = transaction.work.get_scope(scope_id)
                saved_root = transaction.work.get_root(root.id)
            assert saved_attempt.state == 'cancelled', reason
            assert saved_scope.status == 'pending', reason
            if reason in {'limit', 'scope_limit'}:
                assert saved_root.status == 'stopped'
                assert saved_root.reason == reason
        finally:
            await app.aclose()


async def test_full_chain_not_sent_rollback_cancel_in_http_and_poisoned_diagnostics(tmp_path):
    """Full RateLimited -> Provenance -> Logging chain: not_sent rolls back, HTTP cancel becomes unknown, poisoned diagnostics cancels created."""
    from linguary.application.diagnostics import diagnostics_for
    from linguary.application.ports.persistence import PersistenceError

    # 1. not_sent rolls back calls_used and scope -> pending
    case1 = tmp_path / 'not_sent'
    case1.mkdir()
    fake1 = FakeLLMClient([analysis_reply(2), LLMTransportError('connect', outcome='not_sent'), translation_reply(2)])
    app1 = make_app(case1, fake1)
    _one_attempt(app1)
    try:
        _doc, run, _task = await prepare(app1)
        completed = await drain_tasks(app1)
        assert completed[0].status == TaskStatus.FAILED
        with app1.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            attempts = transaction.work.list_root_attempts(root.id)
            scopes = transaction.work.list_scopes(root.id)
        assert root.calls_used == 0
        assert [a.state for a in attempts] == ['cancelled']
        assert all(s.status == 'pending' for s in scopes)
    finally:
        await app1.aclose()

    # 2. Cancellation AFTER mark_attempt_sent during HTTP with _log failing still yields attempt=unknown & root=unknown_outcome
    case2 = tmp_path / 'cancel_http'
    case2.mkdir()
    entered_http = asyncio.Event()

    class _HangingFake(FakeLLMClient):
        async def generate(self, request):
            if request.stage == LLMCallStage.TRANSLATION_UNIT:
                entered_http.set()
                await asyncio.Event().wait()
            return await super().generate(request)

    fake2 = _HangingFake([analysis_reply(2)])
    app2 = make_app(case2, fake2)
    try:
        _doc, run, _task = await prepare(app2)
        worker = asyncio.create_task(drain_tasks(app2))
        await asyncio.wait_for(entered_http.wait(), 3)
        root, attempt = await _running_attempt(app2, run.id)
        assert attempt.state == 'sent'
        # Make LoggingLLMClient._log fail on cancellation to prove CancelledError still propagates cleanly.
        proc_client = _processing_client(app2)
        logging_client = proc_client._inner.inner
        async def _broken_log(*_a, **_kw):
            raise PersistenceError('hard failure')
        logging_client._log = _broken_log
        worker.cancel()
        await asyncio.wait({worker})
        with app2.unit_of_work.transaction() as transaction:
            saved_attempt = transaction.work.get_attempt(attempt.id)
            saved_root = transaction.work.get_root(root.id)
        assert saved_attempt.state == 'unknown'
        assert saved_root.status == 'stopped'
        assert saved_root.reason == 'unknown_outcome'
    finally:
        await app2.aclose()

    # 3. Permanently poisoned diagnostics cancels created attempt without burning calls_used
    case3 = tmp_path / 'poisoned'
    case3.mkdir()
    fake3 = FakeLLMClient([analysis_reply(2), translation_reply(2)])
    app3 = make_app(case3, fake3)
    try:
        _doc, run, _task = await prepare(app3)
        entered, release = _arm_rate_limit(app3)
        worker = asyncio.create_task(drain_tasks(app3))
        await asyncio.wait_for(entered.wait(), 3)
        root, attempt = await _running_attempt(app3, run.id)
        assert attempt.state == 'created'
        diagnostics_for(app3.unit_of_work)._failed = True
        _open_gate(app3, release)
        try:
            await asyncio.wait_for(worker, 5)
        except LLMConfigurationError:
            pass
        with app3.unit_of_work.transaction() as transaction:
            saved_attempt = transaction.work.get_attempt(attempt.id)
            saved_root = transaction.work.get_root(root.id)
            saved_scope = transaction.work.get_scope(attempt.scope_id)
        assert saved_attempt.state == 'cancelled'
        assert saved_scope.status == 'pending'
        assert saved_root.calls_used == 0
        assert _translation_requests(fake3) == []
    finally:
        await app3.aclose()
