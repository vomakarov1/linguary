"""Attempt budget, unknown sends, indivisible truncation, and save-time note ids."""
from dataclasses import replace

import pytest

from linguary.application.ports.llm import LLMTransportError
from linguary.application.recovery import RecoveryError
from linguary.composition import drain_tasks
from linguary.domain import LLMCallStage, TaskStatus
from tests.correction_fakes import CorrectionAwareFakeLLMClient as FakeLLMClient
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


async def test_scope_limit_reported_when_max_calls_exceeds_max_scope_calls(tmp_path):
    """When begin_recovery_attempt hits max_scope_calls (< max_calls), RecoveryError.reason is scope_limit and persisted."""
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.adapters.config.settings import RecoverySettings
    from linguary.composition import build_application

    fake = FakeLLMClient([
        analysis_reply(1),
        LLMTransportError('HTTP 503', outcome='known_transient'),
        LLMTransportError('HTTP 503', outcome='known_transient'),
        LLMTransportError('HTTP 503', outcome='known_transient'),
        LLMTransportError('HTTP 503', outcome='known_transient'),
    ])
    app = build_application(
        settings=AppSettings(
            llm=LLMSettings(
                model='pinned-model',
                api_key='synthetic-secret',
                proxy_mode='direct',
                max_attempts=5,
            ),
            recovery=RecoverySettings(max_calls=10, max_scope_calls=2),
            storage=StorageSettings(data_root=str(tmp_path / 'data')),
        ),
        llm_client=fake,
        content_plan_profile='legacy-v1',
    )
    try:
        _doc, run, _task = await prepare(app, blocks=1)
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.FAILED
        with app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots('book-1', run_id=run.id)
            assert root.status == 'stopped'
            assert root.reason == 'scope_limit'
            assert root.calls_used == 2
        assert len(_translation_requests(fake)) == 2
    finally:
        await app.aclose()


@pytest.mark.parametrize('rule_code', ['provider_refusal', 'provider_content_filter'])
async def test_provider_refusal_and_content_filter_stop_without_retry_or_split(tmp_path, rule_code):
    """provider_refusal and provider_content_filter stop immediately with reason='provider_refusal' without retries or splits."""
    from linguary.application.ports.llm import LLMResponseFormatError
    from linguary.domain.llm_log import Violation

    refusal_err = LLMResponseFormatError(
        'отказ провайдера',
        truncated=False,
        violation=Violation(rule_code, 'translation_unit', safe_message='Отказ'),
    )
    fake = FakeLLMClient([
        analysis_reply(2),
        refusal_err,
        translation_reply(2),
    ])
    app = make_app(tmp_path, fake)
    try:
        _doc, run, _task = await prepare(app, blocks=2)
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.FAILED
        with app.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots('book-1', run_id=run.id)
            scopes = tx.work.list_scopes(root.id)
            assert (root.status, root.reason) == ('stopped', 'provider_refusal')
            assert root.calls_used == 1
            assert len(scopes) == 1
            assert scopes[0].status != 'split'
        assert len(_translation_requests(fake)) == 1
    finally:
        await app.aclose()


async def test_safe_auto_split_on_response_schema_and_mixed_failures_never_on_429(tmp_path):
    """2x response_schema (failed) and alternating response_schema <-> content_violation (with transient in between) split; 2x 429 never split."""
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.adapters.config.settings import RecoverySettings
    from linguary.adapters.llm import FakeReply
    from linguary.composition import build_application

    # 1. 2x non-local response_schema (state="failed") -> triggers fallback split and isolates recovery
    schema_bad = FakeReply.of_payload({
        "blocks": [
            {"source_block_id": "b1", "treatment": "TRANSLATE", "translated_text": "Один.", "glossary_hints": []},
            {"source_block_id": "b2", "treatment": "TRANSLATE", "translated_text": "Два.", "glossary_hints": []},
        ],
        "language_notes": [{"source_block_id": "b1", "original_excerpt": "He", "translated_excerpt": None, "note": ""}],
    })
    fake_schema = FakeLLMClient([
        analysis_reply(2),
        schema_bad,
        schema_bad,
        translation_reply(1),
        translation_reply(1),
    ])
    app_schema = make_app(tmp_path / "schema", fake_schema)
    try:
        _doc, run, _task = await prepare(app_schema, blocks=2)
        completed = await drain_tasks(app_schema)
        assert completed[-1].status is TaskStatus.SUCCEEDED, completed[-1].last_error
        with app_schema.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            scopes = tx.work.list_scopes(root.id)
            assert sorted(s.status for s in scopes) == ["accepted", "accepted", "split"]
            attempts = tx.work.list_root_attempts(root.id)
            assert len(attempts) == 4
            assert all(a.state in {"rejected", "failed"} for a in attempts[:2])
    finally:
        await app_schema.aclose()

    # 2. Alternating response_schema <-> content_violation with max_scope_calls=4 and an intervening HTTP 429 transient failure
    local_schema_bad_b2 = FakeReply.of_payload({
        "blocks": [
            {"source_block_id": "b1", "treatment": "TRANSLATE", "translated_text": "Один.", "glossary_hints": []},
            {"source_block_id": "b2", "treatment": "TRANSLATE", "translated_text": None, "glossary_hints": []},
            {"source_block_id": "b3", "treatment": "TRANSLATE", "translated_text": "Три.", "glossary_hints": []},
        ],
        "language_notes": [],
    })
    content_bad_b2 = FakeReply.of_payload({
        "blocks": [
            {"source_block_id": "b1", "treatment": "TRANSLATE", "translated_text": "Один.", "glossary_hints": []},
            {"source_block_id": "b2", "treatment": "TRANSLATE", "translated_text": "   ", "glossary_hints": []},
            {"source_block_id": "b3", "treatment": "TRANSLATE", "translated_text": "Три.", "glossary_hints": []},
        ],
        "language_notes": [],
    })
    fake_alt = FakeLLMClient([
        analysis_reply(3),
        local_schema_bad_b2,
        LLMTransportError("HTTP 429", outcome="known_transient"),
        content_bad_b2,
        local_schema_bad_b2,
        translation_reply(1),
        translation_reply(1),
        translation_reply(1),
    ])
    app_alt = build_application(
        settings=AppSettings(
            llm=LLMSettings(
                model="pinned-model",
                api_key="synthetic-secret",
                proxy_mode="direct",
                max_attempts=10,
            ),
            recovery=RecoverySettings(max_calls=12, max_scope_calls=4),
            storage=StorageSettings(data_root=str(tmp_path / "alt" / "data")),
        ),
        llm_client=fake_alt,
        content_plan_profile="legacy-v1",
    )
    try:
        doc, run, _task = await prepare(app_alt, blocks=3)
        completed = await drain_tasks(app_alt)
        assert completed[-1].status is TaskStatus.SUCCEEDED, completed[-1].last_error
        with app_alt.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            scopes = tx.work.list_scopes(root.id)
            child_blocks = [s.ordered_blocks for s in scopes if s.status == "accepted"]
            b1_id, b2_id, b3_id = (b.id for b in doc.sections[0].blocks)
            assert set(child_blocks) == {(b1_id,), (b2_id,), (b3_id,)}
    finally:
        await app_alt.aclose()

    # 3. 2x HTTP 429 never trigger split even when max_scope_calls=3
    fake_429 = FakeLLMClient([
        analysis_reply(2),
        LLMTransportError("HTTP 429", outcome="known_transient"),
        LLMTransportError("HTTP 429", outcome="known_transient"),
        translation_reply(2),
    ])
    app_429 = make_app(tmp_path / "rate429", fake_429)
    try:
        _doc, run, _task = await prepare(app_429, blocks=2)
        completed = await drain_tasks(app_429)
        assert completed[-1].status is TaskStatus.SUCCEEDED, completed[-1].last_error
        with app_429.unit_of_work.transaction() as tx:
            root, = tx.work.list_roots("book-1", run_id=run.id)
            scopes = tx.work.list_scopes(root.id)
            assert len(scopes) == 1
            assert scopes[0].status == "accepted"
            assert root.calls_used == 3
    finally:
        await app_429.aclose()


def test_work_state_splittable_rules_and_guards(unit_of_work) -> None:
    """stage='reader_text' splits; 401/402/403/429/5xx/refusal/unknown/None violation never split."""
    from datetime import UTC, datetime
    from uuid import uuid4
    from linguary.application.ports.persistence import BookRecord
    from linguary.application.recovery import root_manifest
    from linguary.application.translation_policy import CURRENT_TRANSLATION_POLICY
    from linguary.domain import BookMetadata, BookStatus
    from linguary.domain.book_conditions import BookConditions
    from linguary.domain.llm_log import Violation
    from linguary.domain.recovery import CallAttempt, RecoveryRoot, RecoveryScope

    with unit_of_work.transaction() as tx:
        tx.books.add(
            BookRecord(
                "book-ws", BookMetadata("Book", ("Author",)), "en", "ru",
                BookStatus.PROCESSING, "book.epub", datetime.now(UTC),
            )
        )
        cond = BookConditions(
            "00000000-0000-0000-0000-000000000099", "book-ws", 1,
            {"schema_version": 1, "models": {"processing": "m"}}, datetime.now(UTC),
        )
        tx.work.add_conditions(cond)

        def check_sequence(violations: tuple[Violation | None, ...], *, max_scope_calls: int = 3) -> bool:
            rid = str(uuid4())
            sid = str(uuid4())
            manifest = root_manifest(
                book_id="book-ws", conditions_id=cond.id, glossary_sha="g", analysis_sha="a",
                plan_sha=None, ordered_blocks=("b1", "b2"), treatments=("TRANSLATE", "TRANSLATE"),
                boundaries=(("b1", "b2"),), atomic_intervals=(("b1",), ("b2",)), max_source_chars=200,
            )
            manifest["policy"]["max_scope_calls"] = max_scope_calls
            tx.work.add_root(
                RecoveryRoot(rid, "book-ws", cond.id, f"rk-{rid}", manifest, policy_version=CURRENT_TRANSLATION_POLICY)
            )
            tx.work.add_scope(RecoveryScope(sid, rid, ("b1", "b2")))
            last_result = False
            for idx, viol in enumerate(violations, start=1):
                scope = tx.work.get_scope(sid)
                claimed = tx.work.claim_scope(sid, expected_generation=scope.generation)
                aid = str(uuid4())
                tx.work.add_attempt(
                    CallAttempt(
                        aid, "book-ws", state="sent", generation=claimed.generation, ordinal=idx,
                        conditions_id=cond.id, root_id=rid, scope_id=sid,
                        context={"schema_version": 1, "root_generation": 0},
                    )
                )
                last_result = tx.work.record_recovery_failure(aid, viol, outcome="known")
            return last_result

        # 0. Direct work_state test: 2x response_schema with state="failed" (no LoggingLLMClient finalize) splits
        assert check_sequence((
            Violation("response_schema", "translation_result", scope="unknown"),
            Violation("response_schema", "translation_result", scope="unknown"),
        )) is True
        root_failed = tx.work.list_roots("book-ws")[-1]
        assert [a.state for a in tx.work.list_root_attempts(root_failed.id)] == ["failed", "failed"]

        # 1. stage="reader_text" splits after 2 attempts even with different rule_codes
        assert check_sequence((
            Violation("a2_grammar_leak", "reader_text", scope="unknown"),
            Violation("a2_summary_length", "reader_text", scope="unknown"),
        )) is True

        # 2. Never split on transport_*, provider_refusal, provider_content_filter, unknown, None
        for bad_viol in (
            Violation("transport_transient", "translation_unit", scope="unknown"),
            Violation("transport_not_sent", "translation_unit", scope="unknown"),
            Violation("provider_refusal", "translation_result", scope="unknown"),
            Violation("provider_content_filter", "translation_result", scope="unknown"),
            Violation("unknown", "translation_unit", scope="unknown"),
            None,
        ):
            assert check_sequence((bad_viol, bad_viol, bad_viol), max_scope_calls=3) is False

        # 3. Even if prior attempts were splittable, a non-splittable current error must NOT split
        assert check_sequence((
            Violation("response_schema", "translation_result", scope="unknown"),
            Violation("content_violation", "translation_unit", scope="unknown"),
            Violation("transport_transient", "translation_unit", scope="unknown"),
        ), max_scope_calls=3) is False


async def test_scope_limit_preserves_last_error_and_recovers_original_error_from_attempts(tmp_path):
    """RecoveryError('scope_limit'|'limit') carries original_error from last_error or list_root_attempts after restart."""
    from uuid import uuid4
    from linguary.adapters.config import AppSettings, LLMSettings, StorageSettings
    from linguary.adapters.config.settings import RecoverySettings
    from linguary.application.retry_policy import classify_error
    from linguary.composition import build_application
    from linguary.domain import ErrorType
    from linguary.domain.llm_log import Violation
    from linguary.domain.recovery import CallAttempt

    # Case A: within same loop (last_error is populated)
    fake = FakeLLMClient([
        analysis_reply(1),
        LLMTransportError("HTTP 503", outcome="known_transient"),
        LLMTransportError("HTTP 503", outcome="known_transient"),
        LLMTransportError("HTTP 503", outcome="known_transient"),
    ])
    app = build_application(
        settings=AppSettings(
            llm=LLMSettings(model="pinned-model", api_key="synthetic-secret", proxy_mode="direct", max_attempts=5),
            recovery=RecoverySettings(max_calls=10, max_scope_calls=2),
            storage=StorageSettings(data_root=str(tmp_path / "same_loop")),
        ),
        llm_client=fake,
        content_plan_profile="legacy-v1",
    )
    try:
        _doc, _run, _task = await prepare(app, blocks=1)
        completed = await drain_tasks(app)
        assert completed[-1].status is TaskStatus.FAILED
        assert completed[-1].last_error is not None
        assert completed[-1].last_error.type is ErrorType.TRANSIENT
    finally:
        await app.aclose()

    # Case B: post-restart (last_error is None, read from list_root_attempts)
    app_b = make_app(tmp_path / "post_restart", FakeLLMClient([]))
    try:
        work = capacity_call(app_b, FakeLLMClient([]), neighbours=False)
        with app_b.unit_of_work.transaction() as tx:
            scope, = tx.work.list_scopes(work.root.id)
            claimed = tx.work.claim_scope(scope.id, expected_generation=scope.generation)
            # Exhaust max_scope_calls so begin_recovery_attempt returns None on the very first loop iteration.
            for idx in range(1, 4):
                aid = str(uuid4())
                tx.work.add_attempt(
                    CallAttempt(
                        aid, "book-1", state="sent", generation=claimed.generation, ordinal=idx,
                        conditions_id=work.root.conditions_id, root_id=work.root.id, scope_id=scope.id,
                        context={"schema_version": 1, "root_generation": work.root.generation},
                    )
                )
                tx.work.record_recovery_failure(
                    aid,
                    Violation("paragraph_count_mismatch", "translation_unit", scope="local", block_ids=(work.unit.block_ids[0],)),
                    outcome="known",
                )
                scope = tx.work.get_scope(scope.id)
                if idx < 3:
                    claimed = tx.work.claim_scope(scope.id, expected_generation=scope.generation)
        with pytest.raises(RecoveryError, match="scope_limit") as caught:
            await work.call.request(work.context, **work.request_args)
        assert caught.value.original_error is not None
        assert classify_error(caught.value).type is ErrorType.VALIDATION
    finally:
        await app_b.aclose()
