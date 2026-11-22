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
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
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
        assert saved.state in {'created', 'cancelled'}
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
