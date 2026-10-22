import pytest

from linguary.adapters.config.settings import AppSettings, LLMSettings
from linguary.adapters.config.system_settings import SystemSettingsUpdate, SystemSettingsValidationError, candidate_settings, overlay_settings
from linguary.adapters.db.system_settings import StoredSystemSettings


def update(**values):
    fields = dict(provider='openai_compatible', model='model', base_url='http://localhost:8080/v1', requests_per_minute=None, concurrency=2)
    fields.update(values)
    return SystemSettingsUpdate(**fields)


@pytest.mark.parametrize('key', [None, '', '   '])
def test_blank_key_preserves_effective_key(key):
    current = AppSettings(llm=LLMSettings(api_key='private-test-key'))
    assert candidate_settings(current, update(api_key=key)).llm.api_key == current.llm.api_key


def test_nonempty_key_replaces_key():
    assert candidate_settings(AppSettings(), update(api_key=' new-key ')).llm.api_key == 'new-key'


def test_blank_model_is_valid_but_not_ready():
    assert not candidate_settings(AppSettings(), update(model='   ', api_key='key')).llm.is_ready


def test_provider_rejects_unsupported_value():
    with pytest.raises(SystemSettingsValidationError):
        update(provider='unsupported')


@pytest.mark.parametrize('url', ['relative/path', 'https://user:password@example.com', 'https://example.com?q=x', 'https://example.com#x', 'http://local\n/'])
def test_base_url_rejects_relative_and_credentials(url):
    with pytest.raises(SystemSettingsValidationError):
        update(base_url=url)


def test_base_url_allows_local_http():
    assert update(base_url=' http://localhost:123/v1 ').base_url == 'http://localhost:123/v1'


@pytest.mark.parametrize('value', [True, False, 1.0, '1', 0, -1])
@pytest.mark.parametrize('name', ['concurrency', 'requests_per_minute'])
def test_limits_reject_bool_noninteger_and_nonpositive(value, name):
    with pytest.raises(SystemSettingsValidationError):
        update(**{name: value})


def test_overlay_preserves_uneditable_values():
    initial = AppSettings(llm=LLMSettings(timeout_seconds=72, api_key='baseline'))
    stored = StoredSystemSettings(1, dict(provider='openai_compatible', model='new', base_url='http://localhost', requests_per_minute=4, concurrency=3), 'saved')
    result = overlay_settings(initial, stored)
    assert result.llm.timeout_seconds == 72
    assert result.storage is initial.storage and result.server is initial.server and result.checks is initial.checks
    assert result.llm.api_key == 'saved' and result.worker.concurrency == 3
    assert overlay_settings(initial, StoredSystemSettings(2, None, None)) == initial


def test_errors_and_repr_do_not_contain_secret():
    secret = 'private-test-key'
    assert secret not in repr(update(api_key=secret))
    with pytest.raises(SystemSettingsValidationError) as caught:
        update(provider=secret, api_key=secret)
    assert secret not in str(caught.value) and secret not in repr(caught.value)


def test_invalid_stored_fields_fail_safely():
    secret = 'private-test-key'
    with pytest.raises(SystemSettingsValidationError) as caught:
        overlay_settings(AppSettings(), StoredSystemSettings(1, {secret: secret}, secret))
    assert secret not in str(caught.value)


@pytest.mark.parametrize('mode', ['auto', 'direct', 'custom'])
def test_proxy_modes_and_http_https_urls_validate(mode):
    url = 'https://alice:p%40ss@localhost:8080' if mode == 'custom' else None
    result = candidate_settings(AppSettings(), update(proxy_mode=mode, proxy_url=url))
    assert result.llm.proxy_mode == mode
    assert result.llm.proxy_url == url
    assert 'p%40ss' not in repr(result)


@pytest.mark.parametrize('url', ['socks5://localhost:9', 'http://localhost:bad', 'http://localhost:99999', 'http://localhost/x', 'http://localhost?q=1', 'http://localhost#x', 'http://local\nhost', 'http://localhost /'])
def test_proxy_url_rejects_socks_invalid_port_controls_path_query_fragment(url):
    with pytest.raises(SystemSettingsValidationError) as caught:
        update(proxy_mode='custom', proxy_url=url)
    assert caught.value.field == 'proxy_url'
    assert url not in str(caught.value)


def test_custom_blank_preserves_previous_url():
    current = AppSettings(llm=LLMSettings(proxy_mode='custom', proxy_url='http://alice:password@localhost:9'))
    assert candidate_settings(current, update(proxy_mode='custom', proxy_url='  ')).llm.proxy_url == current.llm.proxy_url
    assert candidate_settings(current, update()).llm.proxy_url == current.llm.proxy_url
    for mode in ('auto', 'direct'):
        assert candidate_settings(current, update(proxy_mode=mode)).llm.proxy_url is None


def test_custom_without_url_rejects():
    with pytest.raises(SystemSettingsValidationError) as caught:
        candidate_settings(AppSettings(), update(proxy_mode='custom'))
    assert caught.value.field == 'proxy_url'


def test_url_without_mode_rejects():
    with pytest.raises(SystemSettingsValidationError):
        update(proxy_url='http://localhost:9')


@pytest.mark.parametrize('url', ['\nhttp://localhost', 'http://localhost\n', ' http://localhost '])
def test_proxy_url_rejects_outer_controls_and_whitespace(url):
    with pytest.raises(SystemSettingsValidationError) as caught:
        update(proxy_mode='custom', proxy_url=url)
    assert caught.value.field == 'proxy_url'


def test_proxy_secrets_include_httpx_basic_credentials():
    from linguary.adapters.config.settings import proxy_secrets
    assert 'YWxpY2U6cEBzcw==' in proxy_secrets('http://alice:p%40ss@localhost:9')


@pytest.mark.parametrize('timeout', [0, -1, -0.5, 600.1, 1000])
def test_timeout_seconds_rejects_nonpositive_and_above_limit(timeout):
    with pytest.raises(SystemSettingsValidationError) as caught:
        update(timeout_seconds=timeout)
    assert caught.value.field == 'timeout_seconds'


@pytest.mark.parametrize('timeout', [0.1, 1, 60, 600, 600.0])
def test_timeout_seconds_accepts_valid_range(timeout):
    assert update(timeout_seconds=timeout).timeout_seconds == float(timeout)


def test_candidate_settings_temperature_reset_and_preserve():
    current = AppSettings(llm=LLMSettings(temperature=0.7))
    assert candidate_settings(current, update()).llm.temperature == 0.7
    assert candidate_settings(current, update(temperature=None)).llm.temperature is None
    assert candidate_settings(current, update(temperature=0.3)).llm.temperature == 0.3
