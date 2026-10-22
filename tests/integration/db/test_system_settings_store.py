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


def test_system_settings_store_saves_extended_fields(engine):
    store = SystemSettingsStore(engine)
    extended = {
        **OVERRIDES,
        'analysis_model': 'analysis-model',
        'processing_model': 'processing-model',
        'temperature': 0.7,
        'timeout_seconds': 120.0,
    }
    saved = store.save(extended, 'test-key', expected_revision=0)
    assert saved.overrides == extended
    loaded = store.load()
    assert loaded == saved
    assert loaded.overrides['analysis_model'] == 'analysis-model'
    assert loaded.overrides['processing_model'] == 'processing-model'
    assert loaded.overrides['temperature'] == 0.7
    assert loaded.overrides['timeout_seconds'] == 120.0


def test_overlay_settings_accepts_legacy_and_extended_records():
    from linguary.adapters.config import AppSettings, LLMSettings
    from linguary.adapters.config.system_settings import overlay_settings
    from linguary.adapters.db.system_settings import StoredSystemSettings

    initial = AppSettings(
        llm=LLMSettings(
            model="default-model",
            analysis_model="init-analysis",
            processing_model="init-proc",
            temperature=0.2,
            timeout_seconds=72.0,
        )
    )

    legacy = StoredSystemSettings(
        revision=1,
        overrides=dict(
            provider="openai_compatible",
            model="legacy-model",
            base_url="http://localhost:8080/v1",
            requests_per_minute=10,
            concurrency=2,
        ),
        api_key="legacy-key",
    )
    result_legacy = overlay_settings(initial, legacy)
    assert result_legacy.llm.model == "legacy-model"
    assert result_legacy.llm.analysis_model == "init-analysis"
    assert result_legacy.llm.processing_model == "init-proc"
    assert result_legacy.llm.temperature == 0.2
    assert result_legacy.llm.timeout_seconds == 72.0

    extended = StoredSystemSettings(
        revision=2,
        overrides=dict(
            provider="openai_compatible",
            model="ext-model",
            base_url="http://localhost:8080/v1",
            requests_per_minute=20,
            concurrency=3,
            analysis_model="ext-analysis",
            processing_model="ext-proc",
            temperature=0.7,
            timeout_seconds=120.0,
        ),
        api_key="ext-key",
    )
    result_extended = overlay_settings(initial, extended)
    assert result_extended.llm.model == "ext-model"
    assert result_extended.llm.analysis_model == "ext-analysis"
    assert result_extended.llm.processing_model == "ext-proc"
    assert result_extended.llm.temperature == 0.7
    assert result_extended.llm.timeout_seconds == 120.0


def test_candidate_settings_propagates_split_models_and_inference_params():
    from linguary.adapters.config import AppSettings, LLMSettings
    from linguary.adapters.config.system_settings import SystemSettingsUpdate, candidate_settings

    current = AppSettings(
        llm=LLMSettings(
            model="base",
            analysis_model="old-analysis",
            processing_model="old-proc",
            temperature=0.5,
            timeout_seconds=60.0,
        )
    )
    update = SystemSettingsUpdate(
        provider="openai_compatible",
        model="new-base",
        base_url="http://localhost:8080/v1",
        requests_per_minute=15,
        concurrency=4,
        analysis_model="new-analysis",
        processing_model="new-proc",
        temperature=0.9,
        timeout_seconds=180.0,
    )
    candidate = candidate_settings(current, update)
    assert candidate.llm.model == "new-base"
    assert candidate.llm.analysis_model == "new-analysis"
    assert candidate.llm.processing_model == "new-proc"
    assert candidate.llm.temperature == 0.9
    assert candidate.llm.timeout_seconds == 180.0

