"""Attempt budget, unknown sends, indivisible truncation, and save-time note ids."""
from dataclasses import replace

import pytest

from linguary.application.ports.llm import LLMTransportError
from linguary.application.recovery import RecoveryError
from linguary.composition import drain_tasks
from linguary.domain import LLMCallStage, TaskStatus
from correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
from processing_scenario import note_reply, translated_reply
from test_book_conditions import make_app
from test_recovery_commit import analysis_reply, prepare, translation_reply
from test_translation_context_capacity import capacity_call


def _one_attempt(app):
    tuned = replace(app.settings, llm=replace(app.settings.llm, max_attempts=1))
    app.runtime_settings.publish(app.runtime_settings.prepare(tuned), app.runtime_settings.revision + 1)


def _translation_requests(fake):
    return [request for request in fake.requests if request.stage == LLMCallStage.TRANSLATION_UNIT]


async def test_max_attempts_1_transient_error_preserves_budget_for_manual_retry(tmp_path):
    fake = FakeLLMClient([
        analysis_reply(2),
        LLMTransportError('HTTP 503', outcome='known_transient'),
        translation_reply(2),
    ])
    app = make_app(tmp_path, fake)
    _one_attempt(app)
    try:
        _doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[0].status is TaskStatus.FAILED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            attempts = transaction.work.list_root_attempts(root.id)
            scopes = transaction.work.list_scopes(root.id)
            assert (root.status, root.reason) == ('pending', 'transient_error')
            assert root.reason != 'unsafe_partition'
            assert root.calls_used == 1
            assert [attempt.state for attempt in attempts] == ['failed']
            assert all(scope.status == 'pending' for scope in scopes)
            root_id = root.id
        assert len(_translation_requests(fake)) == 1
        with app.unit_of_work.transaction() as transaction:
            transaction.tasks.save(transaction.tasks.get(task.id).retry())
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.SUCCEEDED, completed[-1].last_error
        with app.unit_of_work.transaction() as transaction:
            continued = transaction.work.get_root(root_id)
            assert continued.id == root_id
            assert continued.status == 'complete'
            assert continued.calls_used == 2
            assert len(transaction.results.list_for_run(run.id)) == 2
        assert len(_translation_requests(fake)) == 2
    finally:
        await app.aclose()


async def test_unknown_transport_outcome_stops_without_auto_retry(tmp_path):
    fake = FakeLLMClient([
        analysis_reply(2),
        LLMTransportError('read timeout', outcome='unknown'),
        translation_reply(2),
    ])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, task = await prepare(app)
        completed = await drain_tasks(app)
        assert completed[0].status is TaskStatus.FAILED, completed[0].last_error
        with app.unit_of_work.transaction() as transaction:
            root, = transaction.work.list_roots('book-1', run_id=run.id)
            attempts = transaction.work.list_root_attempts(root.id)
            assert (root.status, root.reason) == ('stopped', 'unknown_outcome')
            assert root.calls_used == 1
            assert [attempt.state for attempt in attempts] == ['unknown']
            root_id = root.id
        assert len(_translation_requests(fake)) == 1
        with app.unit_of_work.transaction() as transaction:
            transaction.tasks.save(transaction.tasks.get(task.id).retry())
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.FAILED
        assert len(_translation_requests(fake)) == 1
        with app.unit_of_work.transaction() as transaction:
            persisted = transaction.work.get_root(root_id)
            assert (persisted.status, persisted.reason, persisted.calls_used) == (
                'stopped', 'unknown_outcome', 1,
            )
    finally:
        await app.aclose()


async def test_indivisible_truncation_stops_with_output_truncated_without_repeat(tmp_path):
    fake = FakeLLMClient([
        translated_reply(('b1',), truncated=True),
        translated_reply(('b1',)),
    ])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake, neighbours=False)
        with pytest.raises(RecoveryError, match='output_truncated') as caught:
            await work.call.request(work.context, **work.request_args)
        assert caught.value.reason == 'output_truncated'
        assert fake.call_count == 1
        with app.unit_of_work.transaction() as transaction:
            root = transaction.work.get_root(work.root.id)
            scopes = transaction.work.list_scopes(root.id)
            assert (root.status, root.reason) == ('stopped', 'output_truncated')
            assert root.calls_used == 1
            assert len(scopes) == 1
            assert scopes[0].status != 'split'
            assert transaction.results.list_for_run(work.run.id) == ()
    finally:
        await app.aclose()


def _temporary_note_ids():
    issued = 0

    def next_id() -> str:
        nonlocal issued
        issued += 1
        return f'temp-{issued}'

    return next_id


async def test_save_allocates_db_note_ids_atomically_and_skips_stale(tmp_path):
    notes = (
        note_reply('b1', original_excerpt='He', translated_excerpt=None,
                   note='Местоимение третьего лица обозначает мужчину.'),
        note_reply('b1', original_excerpt='returned', translated_excerpt=None,
                   note='Прошедшее время: возвращение уже произошло.'),
    )
    fake = FakeLLMClient([translated_reply(('b1',), notes=notes)])
    app = make_app(tmp_path)
    try:
        work = capacity_call(app, fake, neighbours=False)
        result = await work.call.request(
            work.context, **{**work.request_args, 'note_ids': _temporary_note_ids()},
        )
        temporary = [note.id for item in result.results for note in item.language_notes]
        assert temporary == ['temp-1', 'temp-2']
        with pytest.raises(RuntimeError, match='rollback allocation'):
            with app.unit_of_work.transaction() as transaction:
                assert work.call.save(
                    book_id='book-1', run_id=work.run.id, results=result.results,
                    chapter_num=1, transaction=transaction,
                )
                raise RuntimeError('rollback allocation')
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.list_for_run(work.run.id) == ()
        assert work.call.save(
            book_id='book-1', run_id=work.run.id, results=result.results, chapter_num=1,
        )
        with app.unit_of_work.transaction() as transaction:
            stored = transaction.results.list_for_run(work.run.id)
            assert [note.id for item in stored for note in item.language_notes] == [
                'ln-ch01-001', 'ln-ch01-002',
            ]
        assert work.call.save(
            book_id='book-1', run_id=work.run.id, results=result.results, chapter_num=1,
        ) is False
        with app.unit_of_work.transaction() as transaction:
            assert transaction.results.list_for_run(work.run.id)[0].language_notes[0].id == 'ln-ch01-001'
            assert transaction.results.allocate_note_ids(work.run.id, 1, 1) == ('ln-ch01-003',)
    finally:
        await app.aclose()
