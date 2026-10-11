from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from linguary.adapters.db.system_settings import (
    SystemSettingsStore, SettingsStoreError, SettingsRevisionConflict,
)

OVERRIDES = dict(provider='openai_compatible', model='test', base_url='http://localhost:8080/v1', requests_per_minute=None, concurrency=2)


def test_empty_database_has_no_overrides(engine):
    assert SystemSettingsStore(engine).load() is None


def test_round_trip_keeps_key_out_of_repr(engine):
    store = SystemSettingsStore(engine)
    saved = store.save(OVERRIDES, 'private-test-key', expected_revision=0)
    assert store.load() == saved
    assert 'private-test-key' not in repr(saved)
    assert 'api_key' not in saved.overrides


def test_save_increments_revision(engine):
    store = SystemSettingsStore(engine)
    store.save(OVERRIDES, None, expected_revision=0)
    assert store.save(OVERRIDES, None, expected_revision=1).revision == 2


def test_reset_retains_monotonic_revision(engine):
    store = SystemSettingsStore(engine)
    store.save(OVERRIDES, 'test-key', expected_revision=0)
    reset = store.reset(expected_revision=1)
    assert reset.revision == 2
    assert reset.overrides is None and reset.api_key is None
    assert store.save(OVERRIDES, None, expected_revision=2).revision == 3


@pytest.mark.parametrize('existing', [False, True])
def test_concurrent_writes_reject_stale_revision(engine, existing):
    store = SystemSettingsStore(engine)
    revision = store.save(OVERRIDES, None, expected_revision=0).revision if existing else 0
    barrier = Barrier(2)
    def write():
        barrier.wait()
        try:
            return store.save(OVERRIDES, None, expected_revision=revision).revision
        except SettingsRevisionConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: write(), range(2)))
    assert results.count('conflict') == 1
    assert results.count(revision + 1) == 1


def test_failed_commit_preserves_previous_row(engine):
    store = SystemSettingsStore(engine)
    previous = store.save(OVERRIDES, 'private-test-key', expected_revision=0)
    def fail(connection):
        raise OperationalError('secret SQL', {}, Exception('private-test-key'))
    event.listen(engine, 'commit', fail)
    try:
        with pytest.raises(SettingsStoreError) as caught:
            store.save(OVERRIDES, 'new-test-key', expected_revision=1)
        assert 'private-test-key' not in str(caught.value)
        assert caught.value.__cause__ is None
    finally:
        event.remove(engine, 'commit', fail)
    assert store.load() == previous


def test_save_rejects_secret_inside_overrides(engine):
    store = SystemSettingsStore(engine)
    with pytest.raises(SettingsStoreError):
        store.save({**OVERRIDES, 'api_key': 'private-test-key'}, None, expected_revision=0)
    assert store.load() is None


def test_reset_empty_database_creates_revision_without_overrides(engine):
    store = SystemSettingsStore(engine)
    reset = store.reset(expected_revision=0)
    assert reset.revision == 1
    assert reset.overrides is None and reset.api_key is None
    assert store.load() == reset


def test_proxy_url_is_separate_from_overrides_json(engine):
    from sqlalchemy import text
    store = SystemSettingsStore(engine)
    secret = 'http://alice:p%40ss@localhost:9'
    saved = store.save({**OVERRIDES, 'proxy_mode': 'custom'}, 'key', expected_revision=0, proxy_url=secret)
    assert store.load() == saved
    assert secret not in repr(saved)
    with engine.connect() as connection:
        row = connection.execute(text('SELECT overrides_json, proxy_url FROM system_settings')).one()
        assert secret not in row.overrides_json
        assert row.proxy_url == secret
    reset = store.reset(expected_revision=1)
    assert reset.proxy_url is None and reset.revision == 2


def test_proxy_failed_commit_preserves_previous_row(engine):
    store = SystemSettingsStore(engine)
    previous = store.save({**OVERRIDES, 'proxy_mode': 'custom'}, 'key', expected_revision=0, proxy_url='http://old:password@localhost:9')
    def fail(connection):
        raise OperationalError('secret SQL', {}, Exception('new-password'))
    event.listen(engine, 'commit', fail)
    try:
        with pytest.raises(SettingsStoreError):
            store.save({**OVERRIDES, 'proxy_mode': 'direct'}, 'key', expected_revision=1)
    finally:
        event.remove(engine, 'commit', fail)
    assert store.load() == previous
