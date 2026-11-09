from dataclasses import replace
from datetime import UTC, datetime, timedelta
from linguary.application.diagnostics import Diagnostics, DiagnosticsPolicy
from persistence_factories import make_book, make_llm_call

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)


def seed(unit_of_work, *, count=1):
    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        for index in range(count):
            entry = replace(make_llm_call(call_id=f'call-{index}'), started_at=NOW + timedelta(seconds=index))
            transaction.llm_calls.add(entry, response_text='0123456789')


def test_disabled_body_keeps_metadata(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, policy=DiagnosticsPolicy(bodies_enabled=False), clock=lambda: NOW)
    diagnostics.purge_expired()
    with unit_of_work.transaction() as transaction:
        record = transaction.llm_calls.diagnostic_record(call_id='call-0')
        assert record['body'] is None
        assert record['metadata']['body_meta']['body_status'] == 'disabled'
        assert transaction.llm_calls.list_for_book('book-1')[0].cost is not None


def test_ttl_on_read_and_startup(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, policy=DiagnosticsPolicy(retention_seconds=10), clock=lambda: NOW + timedelta(seconds=11))
    diagnostics.purge_expired()
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.response_text_for('call-0') is None
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['metadata']['body_meta']['body_status'] == 'expired'


def test_per_book_quota(unit_of_work):
    seed(unit_of_work, count=3)
    diagnostics = Diagnostics(unit_of_work, policy=DiagnosticsPolicy(max_book_bytes=20), clock=lambda: NOW)
    diagnostics.purge_expired()
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['body'] is None
        assert transaction.llm_calls.diagnostic_record(call_id='call-1')['body'] == '0123456789'
        assert transaction.llm_calls.diagnostic_record(call_id='call-2')['body'] == '0123456789'


def test_delete_book_and_explicit_purge(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, clock=lambda: NOW)
    assert diagnostics.purge('book-1', call_id='call-0') == 1
    assert diagnostics.purge('book-1', call_id='call-0') == 0
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['metadata']['body_meta']['body_status'] == 'deleted'


def test_accept_reject_updates_one_call(unit_of_work):
    seed(unit_of_work)
    diagnostics = Diagnostics(unit_of_work, clock=lambda: NOW)
    diagnostics.finalize_attempt(None, 'rejected', body='bounded rejected', call_id='call-0')
    diagnostics.finalize_attempt(None, 'rejected', body='must not replace', call_id='call-0')
    with unit_of_work.transaction() as transaction:
        assert len(transaction.llm_calls.list_for_book('book-1')) == 1
        assert transaction.llm_calls.diagnostic_record(call_id='call-0')['body'] == 'bounded rejected'


def test_periodic_purge_throttles_repeated_calls_while_purge_expired_runs_immediately(
    unit_of_work, monkeypatch
):
    seed(unit_of_work)
    current_time = NOW
    diagnostics = Diagnostics(
        unit_of_work,
        policy=DiagnosticsPolicy(retention_seconds=10),
        clock=lambda: current_time,
    )
    purge_calls: list[datetime | None] = []
    original_purge = diagnostics.purge_expired

    def tracked_purge(now=None):
        purge_calls.append(now)
        return original_purge(now)

    monkeypatch.setattr(diagnostics, 'purge_expired', tracked_purge)

    assert diagnostics.periodic_purge() is True
    assert len(purge_calls) == 1

    current_time = NOW + timedelta(minutes=15)
    assert diagnostics.periodic_purge() is False
    diagnostics.ensure_available()
    assert len(purge_calls) == 1

    diagnostics.purge_expired()
    assert len(purge_calls) == 2
    with unit_of_work.transaction() as transaction:
        assert transaction.llm_calls.response_text_for('call-0') is None

    current_time = NOW + timedelta(hours=1, seconds=1)
    assert diagnostics.periodic_purge() is True
    assert len(purge_calls) == 3


def test_enforce_diagnostics_policy_evicts_by_byte_length_without_loading_untouched_bodies(
    unit_of_work,
):
    from linguary.adapters.db.models import LLMCallLogRow
    from linguary.domain.book_conditions import BookConditions
    from persistence_factories import (
        BLOCK_ID,
        SECOND_BLOCK_ID,
        make_document,
        make_glossary_only_result,
        make_run,
        make_translated_result,
    )

    policy = DiagnosticsPolicy(
        bodies_enabled=True,
        max_body_bytes=64,
        max_book_bytes=40,
        retention_seconds=3600,
    )
    text_input_only = 'привет мир'  # 10 chars, 19 UTF-8 bytes
    text_one = 'абвгдеёжзи'  # 10 chars, 20 UTF-8 bytes
    text_two = 'клмнопрсту'  # 10 chars, 20 UTF-8 bytes

    with unit_of_work.transaction() as transaction:
        transaction.books.add(make_book())
        entry0 = replace(
            make_llm_call(call_id='call-0'),
            started_at=NOW,
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'unknown',
                'input_text': text_input_only,
                'body_meta': {
                    'body_status': 'unavailable',
                    'stored_bytes': 0,
                    'expires_at': (NOW + timedelta(seconds=3600)).isoformat(),
                },
            },
        )
        entry1 = replace(
            make_llm_call(call_id='call-1'),
            started_at=NOW + timedelta(seconds=1),
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'rejected',
                'body_meta': {
                    'body_status': 'available',
                    'stored_bytes': len(text_one.encode('utf-8')),
                    'expires_at': (NOW + timedelta(seconds=3601)).isoformat(),
                },
            },
        )
        entry2 = replace(
            make_llm_call(call_id='call-2'),
            started_at=NOW + timedelta(seconds=2),
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'rejected',
                'body_meta': {
                    'body_status': 'available',
                    'stored_bytes': len(text_two.encode('utf-8')),
                    'expires_at': (NOW + timedelta(seconds=3602)).isoformat(),
                },
            },
        )
        transaction.llm_calls.add(entry0, response_text=None)
        transaction.llm_calls.add(entry1, response_text=text_one)
        transaction.llm_calls.add(entry2, response_text=text_two)

    from sqlalchemy import event

    executed_sql: list[tuple[str, object]] = []
    with unit_of_work.transaction() as transaction:
        engine = transaction._active_session().get_bind()

        def _capture_sql(conn, cursor, statement, parameters, context, executemany):
            executed_sql.append((statement, parameters))

        event.listen(engine, 'before_cursor_execute', _capture_sql)
        try:
            transaction.llm_calls.enforce_diagnostics_policy(policy, NOW, book_id='book-1')
        finally:
            event.remove(engine, 'before_cursor_execute', _capture_sql)

    select_statements = [
        (sql, params)
        for sql, params in executed_sql
        if sql.lstrip().upper().startswith('SELECT')
    ]
    assert len(select_statements) == 2
    projection_sql, _ = select_statements[0]
    assert 'CAST(' in projection_sql.upper()
    row_load_sql, row_load_params = select_statements[1]
    assert 'call-0' in repr(row_load_params)
    assert 'call-1' not in repr(row_load_params)
    assert 'call-2' not in repr(row_load_params)

    with unit_of_work.transaction() as transaction:
        rec0 = transaction.llm_calls.diagnostic_record(call_id='call-0')
        rec1 = transaction.llm_calls.diagnostic_record(call_id='call-1')
        rec2 = transaction.llm_calls.diagnostic_record(call_id='call-2')
        assert rec0['body'] is None
        assert 'input_text' not in rec0['metadata']
        assert rec0['metadata']['body_meta']['body_status'] == 'deleted'
        assert rec1['body'] == text_one
        assert rec2['body'] == text_two

        entry_ttl = replace(
            make_llm_call(call_id='call-ttl'),
            started_at=NOW + timedelta(seconds=3),
            diagnostic_meta={
                'schema_version': 1,
                'decision': 'unknown',
                'input_text': 'истекающий запрос',
                'body_meta': {
                    'body_status': 'unavailable',
                    'stored_bytes': 0,
                    'expires_at': (NOW + timedelta(seconds=3603)).isoformat(),
                },
            },
        )
        transaction.llm_calls.add(entry_ttl, response_text=None)

    with unit_of_work.transaction() as transaction:
        transaction.llm_calls.enforce_diagnostics_policy(
            policy, NOW + timedelta(seconds=3700), book_id='book-1'
        )
        rec_ttl = transaction.llm_calls.diagnostic_record(call_id='call-ttl')
        assert 'input_text' not in rec_ttl['metadata']
        assert rec_ttl['metadata']['body_meta']['body_status'] == 'expired'

        from uuid import uuid4

        conditions = BookConditions(
            id=str(uuid4()),
            book_id='book-1',
            revision=1,
            created_at=NOW,
            reproducibility='complete',
            payload={
                'schema_version': 1,
                'source_epub_sha256': 'a' * 64,
                'settings_revision': 1,
                'reader_level': 'b2',
                'english_help': False,
                'models': {'analysis': 'm', 'processing': 'm', 'correction': 'm'},
                'checks': {'correction_passes': 1},
                'versions': {
                    'prompts': 1,
                    'schemas': 1,
                    'validators': 1,
                    'translation_policy': 1,
                    'correction_policy': 1,
                    'quality_policy': 1,
                },
            },
        )
        transaction.work.add_conditions(conditions)
        assert transaction.work.is_book_paused('book-1') is False
        transaction.work.set_book_paused('book-1', conditions.id, True)
        assert transaction.work.is_book_paused('book-1') is True
        transaction.work.set_book_paused('book-1', conditions.id, False)
        assert transaction.work.is_book_paused('book-1') is False

        transaction.documents.save(make_document())
        run = make_run(run_id='run-many-1', book_id='book-1')
        transaction.runs.add(run)
        r1 = make_translated_result(block_id=BLOCK_ID)
        r2 = make_glossary_only_result(block_id=SECOND_BLOCK_ID)
        transaction.results.replace_results(run.id, (r1, r2))
        fetched = transaction.results.get_many(
            run.id, (SECOND_BLOCK_ID, 'sec-1:missing', BLOCK_ID)
        )
        assert fetched == {BLOCK_ID: r1, SECOND_BLOCK_ID: r2}
        assert transaction.results.get_many(run.id, ()) == {}


