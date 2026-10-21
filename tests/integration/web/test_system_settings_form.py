"""Браузерная форма настроек и безопасные ручные повторные отправки."""
from html.parser import HTMLParser

import pytest

HTML = {'Accept': 'text/html'}


def payload(**changes):
    values = dict(
        provider='openai_compatible', model='new-model',
        base_url='http://localhost:8080/v1', requests_per_minute=30,
        concurrency=3, expected_revision=0,
    )
    values.update(changes)
    return values


class FormPage(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.forms = {}
        self.current = None
        self.select = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.current = {'attrs': attrs, 'inputs': {}, 'buttons': [], 'selects': {}}
            self.forms[attrs['action']] = self.current
        elif self.current is not None:
            if tag == 'input':
                self.current['inputs'][attrs['name']] = attrs
            elif tag == 'button':
                self.current['buttons'].append(attrs)
            elif tag == 'select':
                self.current['selects'][attrs['name']] = attrs
                self.select = attrs
            elif tag == 'option' and self.select is not None:
                if 'value' not in self.select or 'selected' in attrs:
                    self.select['value'] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.current = None
        elif tag == 'select':
            self.select = None

    def values(self):
        return {name: attrs.get('value', '') for name, attrs in self.forms['/system/config']['inputs'].items()} | {name: attrs.get('value', '') for name, attrs in self.forms['/system/config']['selects'].items()}


def page(client):
    return client.get('/system/config', headers=HTML)


def post(client, **changes):
    return client.post('/system/config', data={k: '' if v is None else v for k, v in payload(**changes).items()}, headers=HTML, follow_redirects=False)


def test_config_form_displays_current_nonsecret_values(configured_client):
    form = FormPage(page(configured_client).text).forms['/system/config']
    assert form['inputs']['model']['value'] == 'fake-model'
    assert 'provider' in form['selects']
    assert 'base_url' in form['inputs'] and 'requests_per_minute' in form['inputs']
    assert int(form['inputs']['concurrency']['value']) > 0


def test_key_input_is_empty_and_password(configured_client):
    field = FormPage(page(configured_client).text).forms['/system/config']['inputs']['api_key']
    assert field['type'] == 'password' and 'value' not in field


def test_saved_key_is_indicated_by_fixed_mask(configured_client):
    before = page(configured_client).text
    assert '••••••••' in before
    post(configured_client, api_key='much-longer-private-key')
    after = page(configured_client).text
    assert '••••••••' in after and 'much-longer-private-key' not in after and 'test-key' not in before


def test_reset_form_has_current_revision(configured_client):
    post(configured_client)
    assert FormPage(page(configured_client).text).forms['/system/config/reset']['inputs']['expected_revision']['value'] == '1'


@pytest.mark.parametrize('model,ready', [('model', True), ('', False)])
def test_save_redirect_confirms_saved_and_actual_readiness(client, model, ready):
    response = post(client, model=model, api_key='private-key')
    assert response.status_code == 303 and response.headers['location'] == '/system/config?result=saved'
    result = client.get(response.headers['location'], headers=HTML)
    assert 'Настройки сохранены' in result.text
    assert ('LLM настроена' in result.text) is ready
    assert 'private-key' not in result.text


@pytest.mark.parametrize('fixture,ready', [('client', False), ('configured_client', True)])
def test_reset_redirect_confirms_reset_and_actual_readiness(request, fixture, ready):
    client = request.getfixturevalue(fixture)
    post(client, api_key='replacement-key')
    response = client.post('/system/config/reset', data={'expected_revision': 1}, headers=HTML, follow_redirects=False)
    assert response.status_code == 303
    result = client.get(response.headers['location'], headers=HTML)
    assert 'Настройки сброшены' in result.text
    assert ('LLM настроена' in result.text) is ready


def test_html_422_without_referer_keeps_safe_values_and_field_errors(configured_client):
    response = post(configured_client, model='edited-model', concurrency=0, api_key='private-key')
    assert response.status_code == 422
    values = FormPage(response.text).values()
    assert values['model'] == 'edited-model' and values['concurrency'] == '0'
    assert values['api_key'] == '' and values['expected_revision'] == '0'
    assert 'aria-invalid="true"' in response.text and 'private-key' not in response.text


def test_html_422_manual_resubmit_succeeds_with_empty_key(configured_client, configured_application):
    response = post(configured_client, concurrency=0, api_key='private-key')
    values = FormPage(response.text).values() | {'concurrency': '2'}
    assert configured_client.post('/system/config', data=values, headers=HTML, follow_redirects=False).status_code == 303
    assert configured_application.settings.llm.api_key == 'test-key'


def test_html_409_requires_reload_before_manual_resubmit(configured_client):
    post(configured_client)
    conflict = post(configured_client, model='stale')
    assert conflict.status_code == 409
    parsed = FormPage(conflict.text)
    assert all('disabled' in button for form in parsed.forms.values() for button in form['buttons'])
    assert parsed.values()['expected_revision'] == '0'
    assert 'Загрузить актуальные настройки' in conflict.text
    values = FormPage(page(configured_client).text).values()
    assert values['expected_revision'] == '1'
    assert configured_client.post('/system/config', data=values, headers=HTML, follow_redirects=False).status_code == 303


def test_html_reset_conflict_does_not_overwrite(configured_client, configured_application):
    post(configured_client)
    response = configured_client.post('/system/config/reset', data={'expected_revision': 0}, headers=HTML)
    assert response.status_code == 409 and configured_application.settings.llm.model == 'new-model'
    assert all('disabled' in b for f in FormPage(response.text).forms.values() for b in f['buttons'])


def test_html_500_without_referer_has_retry_and_config_return(configured_client, configured_application, monkeypatch):
    original = configured_application.runtime_settings.prepare
    def fail(settings):
        raise RuntimeError('private-key')
    monkeypatch.setattr(configured_application.runtime_settings, 'prepare', fail)
    response = post(configured_client, api_key='private-key')
    assert response.status_code == 500 and 'Изменения не применены' in response.text
    assert 'href="/system/config"' in response.text and 'private-key' not in response.text
    form = FormPage(response.text)
    assert all('disabled' not in b for f in form.forms.values() for b in f['buttons'])
    monkeypatch.setattr(configured_application.runtime_settings, 'prepare', original)
    assert configured_client.post('/system/config', data=form.values(), headers=HTML, follow_redirects=False).status_code == 303


def test_repeated_renders_never_include_key(configured_client):
    for url in ['/system/config', '/system/config?result=saved', '/system/config?result=reset']:
        text = configured_client.get(url, headers=HTML).text
        assert 'test-key' not in text
        assert 'value' not in FormPage(text).forms['/system/config']['inputs']['api_key']


def test_unknown_result_marker_is_not_reflected(client):
    assert 'private-marker' not in client.get('/system/config?result=private-marker', headers=HTML).text


def test_first_run_and_incomplete_settings_remain_editable(client):
    assert all('disabled' not in b for f in FormPage(page(client).text).forms.values() for b in f['buttons'])
    post(client, model='')
    assert all('disabled' not in b for f in FormPage(page(client).text).forms.values() for b in f['buttons'])


def test_existing_monitoring_panels_remain_visible(client):
    text = page(client).text
    for title in ['Внешний провайдер LLM', 'Хранилище', 'EPUBCheck', 'Сервер']:
        assert title in text


def test_missing_revision_disables_both_forms(client):
    values = payload()
    values.pop('expected_revision')
    response = client.post('/system/config', json=values, headers=HTML)
    assert response.status_code == 422
    parsed = FormPage(response.text)
    assert all('disabled' in b for f in parsed.forms.values() for b in f['buttons'])
    assert all('expected_revision' not in f['inputs'] for f in parsed.forms.values())
    assert 'Загрузить актуальные настройки' in response.text


def test_html_500_retry_after_external_change_conflicts(configured_client, configured_application, monkeypatch):
    original = configured_application.runtime_settings.prepare
    def fail(settings):
        raise RuntimeError('preparation failed')
    monkeypatch.setattr(configured_application.runtime_settings, 'prepare', fail)
    failed = post(configured_client)
    assert failed.status_code == 500
    values = FormPage(failed.text).values()
    monkeypatch.setattr(configured_application.runtime_settings, 'prepare', original)
    assert post(configured_client).status_code == 303
    response = configured_client.post('/system/config', data=values, headers=HTML)
    assert response.status_code == 409
    assert FormPage(response.text).values()['expected_revision'] == '0'


def test_config_html_renders_logical_groups_and_inline_errors(configured_client):
    res = page(configured_client)
    assert res.status_code == 200
    assert '<fieldset' in res.text
    assert '<legend>' in res.text
    assert 'Провайдер и модели' in res.text
    assert 'Параметры инференса' in res.text
    assert 'Сетевое подключение и прокси' in res.text
    assert 'Очередь и фоновая работа' in res.text

    err_res = post(configured_client, concurrency=0)
    assert err_res.status_code == 422
    assert '<span id="settings-error-concurrency"' in err_res.text
    assert 'role="alert"' in err_res.text
    assert 'aria-describedby="settings-error-concurrency"' in err_res.text


def test_config_html_has_reset_confirmation_attribute(configured_client):
    res = page(configured_client)
    form = FormPage(res.text).forms['/system/config/reset']
    assert 'data-confirm' in form['attrs']
    assert form['attrs']['data-confirm'] == 'Сбросить все настройки к исходным значениям? Это действие отменит все пользовательские изменения.'


def test_config_html_renders_test_connection_button(configured_client):
    res = page(configured_client)
    form = FormPage(res.text).forms['/system/config']
    test_btn = next((b for b in form['buttons'] if b.get('id') == 'settings-test-connection'), None)
    assert test_btn is not None
    assert test_btn.get('type') == 'button'
    assert 'Проверить подключение' in res.text
    assert 'id="settings-test-status"' in res.text


def test_config_html_includes_new_fields(configured_client):
    res = page(configured_client)
    inputs = FormPage(res.text).forms['/system/config']['inputs']
    assert 'analysis_model' in inputs
    assert 'processing_model' in inputs
    assert 'temperature' in inputs
    assert 'timeout_seconds' in inputs
    assert inputs['timeout_seconds']['value'] == '600.0'

    post(
        configured_client,
        analysis_model='claude-3-haiku',
        processing_model='claude-3-sonnet',
        temperature='0.7',
        timeout_seconds='45.0',
    )
    after = page(configured_client)
    after_inputs = FormPage(after.text).forms['/system/config']['inputs']
    assert after_inputs['analysis_model']['value'] == 'claude-3-haiku'
    assert after_inputs['processing_model']['value'] == 'claude-3-sonnet'
    assert after_inputs['temperature']['value'] == '0.7'
    assert after_inputs['timeout_seconds']['value'] == '45.0'

