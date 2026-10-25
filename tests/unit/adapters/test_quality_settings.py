from dataclasses import replace
from datetime import UTC, datetime
import pytest
from linguary.adapters.config import load_settings, AppSettings, StorageSettings
from linguary.adapters.config.settings import DiagnosticsSettings, RecoverySettings, SettingsError
from linguary.adapters.config.system_settings import overlay_settings, candidate_settings, SystemSettingsUpdate
from linguary.adapters.db.system_settings import StoredSystemSettings
from linguary.adapters.llm import FakeLLMClient
from linguary.composition import build_application
from linguary.application.ports.persistence import BookRecord
from linguary.domain import BookMetadata, BookStatus, LLMCallLog, LLMCallStage


@pytest.mark.parametrize('cls', [DiagnosticsSettings, RecoverySettings])
@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True])
def test_config_rejects_zero_negative_infinite_limits(cls, value):
    name = 'max_calls' if cls is RecoverySettings else 'retention_seconds'
    with pytest.raises(SettingsError):
        cls(**{name: value})


def test_env_overrides_local_file(tmp_path):
    path = tmp_path / 'settings.toml'
    path.write_text('[diagnostics]\nbodies_enabled=true\nretention_seconds=60\n[recovery]\nmax_calls=4\nmax_scope_calls=2\n')
    settings = load_settings(config_path=path, environ={'LINGUARY_DIAGNOSTICS_BODIES_ENABLED': 'false',
        'LINGUARY_DIAGNOSTICS_RETENTION_SECONDS': '30', 'LINGUARY_RECOVERY_MAX_CALLS': '7'})
    assert settings.diagnostics.bodies_enabled is False and settings.diagnostics.retention_seconds == 30
    assert settings.recovery.max_calls == 7 and settings.recovery.max_scope_calls == 2
    assert settings.recovery.max_depth == 3 and settings.recovery.max_leaves == 8


def test_settings_overlay_preserves_local_limits():
    settings = AppSettings(diagnostics=DiagnosticsSettings(retention_seconds=20), recovery=RecoverySettings(max_calls=4))
    update = SystemSettingsUpdate(provider=settings.llm.provider, model='other', base_url=settings.llm.base_url,
        requests_per_minute=None, concurrency=1)
    candidate = candidate_settings(settings, update)
    assert candidate.diagnostics == settings.diagnostics and candidate.recovery == settings.recovery
    stored = StoredSystemSettings(revision=1, overrides=dict(provider=settings.llm.provider, model='other',
        base_url=settings.llm.base_url, requests_per_minute=None, concurrency=1), api_key=None, proxy_url=None)
    effective = overlay_settings(settings, stored)
    assert effective.diagnostics == settings.diagnostics and effective.recovery == settings.recovery


def test_body_off_applies_to_existing_reads(tmp_path):
    settings = AppSettings(storage=StorageSettings(data_root=str(tmp_path / 'data')))
    app = build_application(settings=settings, llm_client=FakeLLMClient())
    with app.unit_of_work.transaction() as tx:
        tx.books.add(BookRecord('book', BookMetadata('Synthetic'), 'en', 'ru', BookStatus.IMPORTED, 'synthetic', datetime.now(UTC)))
        tx.llm_calls.add(LLMCallLog('call', 'book', LLMCallStage.CHAPTER_ANALYSIS, 'fake', 'fake', datetime.now(UTC), 1, 1, True),
            response_text='private old body')
    app.dispose()
    off = build_application(settings=replace(settings, diagnostics=DiagnosticsSettings(bodies_enabled=False)), llm_client=FakeLLMClient())
    try:
        value = off.diagnostics.read('book', 'call')
        assert value['body'] is None and value['body_status'] == 'disabled'
        with off.unit_of_work.transaction() as tx:
            assert tx.llm_calls.response_text_for('call') is None
    finally:
        off.dispose()


def test_lower_retention_applies_to_previous_body_on_first_read(tmp_path):
    from datetime import timedelta
    settings = AppSettings(storage=StorageSettings(data_root=str(tmp_path / 'data')))
    app = build_application(settings=settings, llm_client=FakeLLMClient())
    with app.unit_of_work.transaction() as tx:
        tx.books.add(BookRecord('book', BookMetadata('Synthetic'), 'en', 'ru', BookStatus.IMPORTED, 'synthetic', datetime.now(UTC)))
        tx.llm_calls.add(LLMCallLog('call', 'book', LLMCallStage.CHAPTER_ANALYSIS, 'fake', 'fake',
            datetime.now(UTC) - timedelta(seconds=10), 1, 1, True), response_text='previous body')
    assert app.diagnostics.read('book', 'call')['body'] == 'previous body'
    app.dispose()
    reduced = build_application(settings=replace(settings, diagnostics=DiagnosticsSettings(retention_seconds=1)), llm_client=FakeLLMClient())
    try:
        assert reduced.diagnostics.read('book', 'call')['body_status'] == 'expired'
        assert reduced.diagnostics.read('book', 'call')['body'] is None
    finally:
        reduced.dispose()
