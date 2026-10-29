"""Manual parsing of settings does not expose the raw request or key."""
import json
from dataclasses import asdict
from urllib.parse import urlencode

import pytest
from starlette.requests import Request
from pydantic import ValidationError

from linguary.adapters.config import AppSettings, LLMSettings
from linguary.web.schemas import SettingsErrorResponse
from linguary.web.system_settings_input import SettingsInputError, read_settings_update, read_settings_reset


SECRET = 'private-test-key'
CURRENT = AppSettings(llm=LLMSettings(api_key=SECRET))


def payload(**values):
    fields = dict(provider='openai_compatible', model='new-model', base_url='http://localhost:8080/v1', requests_per_minute=None, concurrency=3, expected_revision=0)
    fields.update(values)
    return fields


def request(body, content_type='application/json'):
    if isinstance(body, dict):
        body = json.dumps(body).encode() if content_type == 'application/json' else urlencode(body).encode()
    elif isinstance(body, str):
        body = body.encode()
    async def receive():
        return {'type': 'http.request', 'body': body, 'more_body': False}
    return Request({'type': 'http', 'method': 'POST', 'path': '/system/config', 'headers': [(b'content-type', content_type.encode())]}, receive)


async def test_form_and_json_have_same_semantics():
    json_update, json_revision = await read_settings_update(request(payload(requests_per_minute=30)), current_settings=CURRENT)
    form_update, form_revision = await read_settings_update(request(payload(requests_per_minute='30'), 'application/x-www-form-urlencoded'), current_settings=CURRENT)
    assert json_update == form_update and json_revision == form_revision == 0


async def test_save_strategy_html_json():
    for content_type in ('application/json', 'application/x-www-form-urlencoded'):
        rate = '' if content_type != 'application/json' else None
        update, _ = await read_settings_update(request(payload(name_translation_strategy='meaning', requests_per_minute=rate), content_type), current_settings=CURRENT)
        assert update.name_translation_strategy == 'meaning'


@pytest.mark.parametrize('body', [payload(name_translation_strategy=None), payload(name_translation_strategy='other'), json.dumps(payload())[:-1] + ',"name_translation_strategy":"sound","name_translation_strategy":"meaning"}'])
async def test_strategy_invalid_null_duplicate_rejected(body):
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(body), current_settings=CURRENT)
    assert caught.value.status_code == 422


async def test_omitted_strategy_preserved():
    update, _ = await read_settings_update(request(payload()), current_settings=CURRENT)
    assert update.name_translation_strategy is None


async def test_blank_form_rate_maps_to_none():
    update, _ = await read_settings_update(request(payload(requests_per_minute=''), 'application/x-www-form-urlencoded'), current_settings=CURRENT)
    assert update.requests_per_minute is None


async def test_missing_revision_is_rejected():
    values = payload()
    del values['expected_revision']
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(values), current_settings=CURRENT)
    assert 'expected_revision' in caught.value.field_errors
    assert caught.value.form_values.expected_revision is None


@pytest.mark.parametrize('name', ['host', 'data_root', 'max_attempts', SECRET])
async def test_unknown_and_uneditable_fields_are_rejected(name):
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(payload(**{name: SECRET})), current_settings=CURRENT)
    assert set(caught.value.field_errors) == {'form'}
    assert SECRET not in repr(caught.value) + repr(caught.value.form_values)


async def test_invalid_json_returns_safe_error():
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request('{"api_key":"' + SECRET), current_settings=CURRENT)
    assert caught.value.code == 'invalid_settings'
    assert SECRET not in repr(caught.value) + str(caught.value)


@pytest.mark.parametrize('values', [dict(concurrency=SECRET), dict(provider=SECRET), dict(model=SECRET, concurrency=0), dict(base_url='https://user:password@host', concurrency=0), dict(model='Bearer abcdefghijklmnop', concurrency=0)])
async def test_secret_is_not_echoed_on_validation_error(values):
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(payload(api_key=SECRET, **values)), current_settings=CURRENT)
    projection = asdict(caught.value.form_values)
    assert 'api_key' not in projection
    assert SECRET not in str(projection)
    assert 'Bearer' not in str(projection) and 'password' not in str(projection)
    assert SECRET not in str(caught.value.field_errors)


async def test_duplicate_form_fields_are_rejected():
    encoded = urlencode(payload(requests_per_minute='')) + '&model=' + SECRET
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(encoded, 'application/x-www-form-urlencoded'), current_settings=CURRENT)
    assert 'form' in caught.value.field_errors
    assert caught.value.form_values.model == ''


async def test_duplicate_json_fields_are_rejected():
    encoded = json.dumps(payload())[:-1] + ',"model":"' + SECRET + '"}'
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(encoded), current_settings=CURRENT)
    assert 'form' in caught.value.field_errors
    assert SECRET not in repr(caught.value.form_values)


async def test_safe_form_projection_keeps_structural_numbers_and_revision():
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(payload(concurrency=0, requests_per_minute='bad-value', expected_revision=7, model=' edited ')), current_settings=CURRENT)
    values = caught.value.form_values
    assert values.concurrency == 0 and values.requests_per_minute is None
    assert values.expected_revision == 7 and values.model == 'edited'
    assert 'bad-value' not in repr(values)


@pytest.mark.parametrize('name,value', [('concurrency', True), ('concurrency', 1.5), ('concurrency', '3'), ('model', {}), ('provider', []), ('api_key', 42), ('expected_revision', False), ('expected_revision', -1)])
async def test_json_rejects_wrong_types(name, value):
    with pytest.raises(SettingsInputError):
        await read_settings_update(request(payload(**{name: value})), current_settings=CURRENT)


async def test_error_schema_has_only_code_message_allowed_field_errors():
    error = SettingsErrorResponse(code='invalid_settings', message='Ошибка', field_errors={'model': 'Укажите модель'})
    assert set(error.model_dump()) == {'code', 'message', 'field_errors'}
    with pytest.raises(ValidationError):
        SettingsErrorResponse(code='invalid_settings', message='Ошибка', field_errors={SECRET: SECRET})


async def test_unsupported_content_type_is_rejected():
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(SECRET, 'text/plain'), current_settings=CURRENT)
    assert caught.value.status_code == 415


async def test_reset_accepts_only_revision():
    assert await read_settings_reset(request({'expected_revision': 0}), current_settings=CURRENT) == 0
    with pytest.raises(SettingsInputError):
        await read_settings_reset(request({'expected_revision': 0, 'model': 'ignored'}), current_settings=CURRENT)


async def test_duplicate_key_values_cannot_be_reflected_in_other_fields():
    values = payload(model='shadow-key', api_key='shadow-key')
    encoded = json.dumps(values)[:-1] + ',"api_key":"different-key"}'
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(encoded), current_settings=CURRENT)
    assert caught.value.form_values.model == ''
    assert 'shadow-key' not in str(asdict(caught.value.form_values))


async def test_malformed_key_object_is_not_reflected_through_model():
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(payload(model='object-secret-key', api_key={'nested': ['object-secret-key']})), current_settings=CURRENT)
    assert caught.value.form_values.model == ''
    assert 'object-secret-key' not in str(asdict(caught.value.form_values))


@pytest.mark.parametrize('url', ['\nhttp://localhost', 'http://localhost\n', 'http://localhost?token=hidden', 'http://localhost#hidden'])
async def test_unsafe_url_is_rejected_and_not_projected(url):
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_update(request(payload(base_url=url)), current_settings=CURRENT)
    assert caught.value.form_values.base_url == ''


async def test_unknown_reset_values_are_not_projected():
    with pytest.raises(SettingsInputError) as caught:
        await read_settings_reset(request({'expected_revision': 0, 'model': 'unknown-reset-value'}), current_settings=CURRENT)
    assert caught.value.form_values.model == CURRENT.llm.model
    assert 'unknown-reset-value' not in str(asdict(caught.value.form_values))


def test_projection_is_scrubbed_again_when_current_key_changes():
    from linguary.web.system_settings_input import safe_form_values, sanitize_form_values
    from linguary.adapters.config.system_settings import SystemSettingsUpdate
    update = SystemSettingsUpdate(provider='openai_compatible', model='next-secret-key', base_url='http://localhost', requests_per_minute=None, concurrency=2)
    values = safe_form_values(CURRENT, update=update, expected_revision=7)
    newer = AppSettings(llm=LLMSettings(api_key='next-secret-key'))
    clean = sanitize_form_values(values, newer)
    assert clean.model == '' and clean.expected_revision == 7


def test_public_config_fields_never_contain_the_effective_key():
    from linguary.web.schemas import LLMConfigResponse
    settings = AppSettings(llm=LLMSettings(model=SECRET, api_key=SECRET, base_url='https://example.com/' + SECRET))
    response = LLMConfigResponse.of_settings(settings)
    assert SECRET not in response.model_dump_json()


async def test_system_settings_input_parses_split_models_and_inference():
    values = payload(
        analysis_model="analysis-model",
        processing_model="proc-model",
        temperature=0.7,
        timeout_seconds=90.0,
    )
    json_update, json_revision = await read_settings_update(request(values), current_settings=CURRENT)
    assert json_update.analysis_model == "analysis-model"
    assert json_update.processing_model == "proc-model"
    assert json_update.temperature == 0.7
    assert json_update.timeout_seconds == 90.0

    form_values = payload(
        analysis_model="analysis-model",
        processing_model="proc-model",
        temperature="0.7",
        timeout_seconds="90.0",
        requests_per_minute="",
    )
    form_update, form_revision = await read_settings_update(
        request(form_values, 'application/x-www-form-urlencoded'),
        current_settings=CURRENT,
    )
    assert form_update.analysis_model == "analysis-model"
    assert form_update.processing_model == "proc-model"
    assert form_update.temperature == 0.7
    assert form_update.timeout_seconds == 90.0


def test_safe_form_values_includes_split_models_without_secrets():
    from linguary.web.system_settings_input import safe_form_values
    from linguary.adapters.config.system_settings import SystemSettingsUpdate

    update = SystemSettingsUpdate(
        provider='openai_compatible',
        model='base-model',
        base_url='http://localhost:8080/v1',
        requests_per_minute=None,
        concurrency=2,
        analysis_model='analysis-secret-key',
        processing_model='clean-proc',
        temperature=0.4,
        timeout_seconds=45.0,
        api_key=SECRET,
    )
    settings_with_secret = AppSettings(llm=LLMSettings(api_key='analysis-secret-key'))
    values = safe_form_values(settings_with_secret, update=update, expected_revision=3)
    assert values.analysis_model == ''
    assert values.processing_model == 'clean-proc'
    assert values.temperature == 0.4
    assert values.timeout_seconds == 45.0
    assert values.expected_revision == 3
    assert 'analysis-secret-key' not in str(asdict(values))

