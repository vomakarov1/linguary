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
