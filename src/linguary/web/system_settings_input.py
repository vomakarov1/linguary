"""Явный разбор тела запроса и отдельная безопасная проекция для повторной формы."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from urllib.parse import urlsplit

from fastapi import Request

from linguary.adapters.config import AppSettings
from linguary.adapters.config.settings import proxy_secrets
from linguary.adapters.config.system_settings import SystemSettingsUpdate, SystemSettingsValidationError, UNSET
from linguary.web import labels
from linguary.web.schemas import SafeSettingsFormValues

_SAVE_FIELDS = frozenset({
    'provider', 'model', 'api_key', 'base_url', 'requests_per_minute',
    'concurrency', 'expected_revision', 'proxy_mode', 'proxy_url',
    'analysis_model', 'processing_model', 'temperature', 'timeout_seconds',
})
_OPTIONAL_FIELDS = {'api_key', 'proxy_mode', 'proxy_url', 'analysis_model', 'processing_model', 'temperature', 'timeout_seconds'}
_REQUIRED_FIELDS = _SAVE_FIELDS - _OPTIONAL_FIELDS
_INTEGER = re.compile(r'[+-]?[0-9]+\Z')
_FLOAT = re.compile(r'[+-]?[0-9]+(?:\.[0-9]+)?\Z')
_CREDENTIALS = re.compile(r'(?i)(bearer\s+|(?:api[_-]?key|token|password|secret|authorization)\s*[:=]|sk-[a-z0-9_-]{8,}|https?://[^/\s]+@)')


class SettingsInputError(Exception):
    def __init__(self, *, field_errors: dict[str, str], form_values: SafeSettingsFormValues, code: str = 'invalid_settings', status_code: int = 422) -> None:
        self.code = code
        self.status_code = status_code
        self.message = labels.SETTINGS_INVALID if status_code == 422 else labels.SETTINGS_UNSUPPORTED_MEDIA_TYPE
        self.field_errors = field_errors
        self.form_values = form_values
        super().__init__(self.message)


def _integer(value: object, *, form: bool) -> int | None:
    if type(value) is int:
        return value
    if form and isinstance(value, str) and _INTEGER.fullmatch(value.strip()):
        try:
            return int(value.strip())
        except ValueError:
            pass
    return None


def _float(value: object, *, form: bool) -> float | None:
    if type(value) in (int, float) and type(value) is not bool:
        return float(value)
    if form and isinstance(value, str) and _FLOAT.fullmatch(value.strip()):
        try:
            return float(value.strip())
        except ValueError:
            pass
    return None



def _secrets(current_settings: AppSettings, posted_key: object, posted_proxy: object = None) -> tuple[str, ...]:
    values = list(proxy_secrets(current_settings.llm.proxy_url))
    pending = [current_settings.llm.api_key, posted_key, posted_proxy]
    while pending:
        value = pending.pop()
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
            values.extend(proxy_secrets(value))
        elif isinstance(value, dict):
            pending.extend(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return tuple(values)


def _safe_text(value: object, secrets: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        return ''
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return ''
    text = value.strip()
    if any(secret in text for secret in secrets) or _CREDENTIALS.search(text):
        return ''
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return ''
    return text


def _safe_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username is not None or url.password is not None:
            return ''
        if '?' in value or '#' in value or any(char.isspace() for char in value):
            return ''
        url.port
    except ValueError:
        return ''
    return value


def _projection(values: Mapping[str, object], current_settings: AppSettings, *, form: bool, duplicate_fields: frozenset[str] = frozenset()) -> SafeSettingsFormValues:
    key = values.get('api_key')
    secrets = _secrets(current_settings, key, values.get('proxy_url'))
    fields = {name: value for name, value in values.items() if name in _SAVE_FIELDS and name not in duplicate_fields}
    provider = _safe_text(fields.get('provider'), secrets)
    revision = _integer(fields.get('expected_revision'), form=form)
    return SafeSettingsFormValues(
        provider=provider if provider == 'openai_compatible' else '',
        model='' if {'api_key', 'proxy_url'} & duplicate_fields else _safe_text(fields.get('model'), secrets),
        analysis_model='' if {'api_key', 'proxy_url'} & duplicate_fields else _safe_text(fields.get('analysis_model'), secrets),
        processing_model='' if {'api_key', 'proxy_url'} & duplicate_fields else _safe_text(fields.get('processing_model'), secrets),
        temperature=_float(fields.get('temperature'), form=form),
        timeout_seconds=_float(fields.get('timeout_seconds'), form=form),
        base_url='' if {'api_key', 'proxy_url'} & duplicate_fields else _safe_url(_safe_text(fields.get('base_url'), secrets)),
        requests_per_minute=_integer(fields.get('requests_per_minute'), form=form),
        concurrency=_integer(fields.get('concurrency'), form=form),
        proxy_mode=fields.get('proxy_mode') if fields.get('proxy_mode') in ('auto', 'direct', 'custom') else current_settings.llm.proxy_mode,
        expected_revision=revision if revision is not None and revision >= 0 else None,
    )


def safe_form_values(current_settings: AppSettings, *, update: SystemSettingsUpdate | None = None, expected_revision: int | None = None) -> SafeSettingsFormValues:
    source = current_settings.llm if update is None else update
    values = dict(
        provider=source.provider,
        model=source.model,
        analysis_model=source.analysis_model,
        processing_model=source.processing_model,
        temperature=source.temperature if (update is None or update.temperature is not UNSET) else current_settings.llm.temperature,
        timeout_seconds=source.timeout_seconds,
        base_url=source.base_url,
        requests_per_minute=source.requests_per_minute,
        concurrency=current_settings.worker.concurrency if update is None else update.concurrency,
        proxy_mode=source.proxy_mode,
        proxy_url=None if update is None else update.proxy_url,
        api_key=None if update is None else update.api_key,
        expected_revision=expected_revision,
    )
    return _projection(values, current_settings, form=False)


def sanitize_form_values(values: SafeSettingsFormValues, current_settings: AppSettings) -> SafeSettingsFormValues:
    """Перед рендером также исключает ключ, сменившийся за время ожидания операции."""
    return _projection(asdict(values), current_settings, form=False)


async def _read(request: Request, current_settings: AppSettings, *, reset: bool) -> tuple[dict[str, object], bool]:
    content_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
    form = content_type in ('application/x-www-form-urlencoded', 'multipart/form-data')
    if content_type != 'application/json' and not form:
        raise SettingsInputError(code='unsupported_media_type', status_code=415, field_errors={}, form_values=safe_form_values(current_settings))
    duplicates: set[str] = set()
    def pairs(items):
        values = {}
        for name, value in items:
            if name in values:
                duplicates.add(name)
            values[name] = value
        return values
    try:
        values = pairs((await request.form()).multi_items()) if form else json.loads(await request.body(), object_pairs_hook=pairs)
    except Exception:
        raise SettingsInputError(field_errors={'form': labels.SETTINGS_INVALID}, form_values=safe_form_values(current_settings)) from None
    if not isinstance(values, dict):
        raise SettingsInputError(field_errors={'form': labels.SETTINGS_INVALID}, form_values=safe_form_values(current_settings))
    allowed = {'expected_revision'} if reset else _SAVE_FIELDS
    if duplicates or set(values) - allowed:
        if reset:
            revision = None if 'expected_revision' in duplicates else _integer(values.get('expected_revision'), form=form)
            projection = safe_form_values(current_settings, expected_revision=revision if revision is not None and revision >= 0 else None)
        else:
            projection = _projection(values, current_settings, form=form, duplicate_fields=frozenset(duplicates))
        raise SettingsInputError(field_errors={'form': labels.SETTINGS_UNKNOWN_FIELDS}, form_values=projection)
    return values, form


def _revision(values: dict[str, object], *, form: bool, errors: dict[str, str]) -> int | None:
    revision = _integer(values.get('expected_revision'), form=form)
    if revision is None or revision < 0:
        errors['expected_revision'] = labels.SETTINGS_REVISION_FIELD
        return None
    return revision


async def read_settings_update(request: Request, *, current_settings: AppSettings) -> tuple[SystemSettingsUpdate, int]:
    values, form = await _read(request, current_settings, reset=False)
    errors = {name: labels.SETTINGS_REQUIRED_FIELD for name in _REQUIRED_FIELDS if name not in values}
    revision = _revision(values, form=form, errors=errors)
    strings = {}
    posted_key = values.get('api_key')
    secrets = _secrets(current_settings, posted_key, values.get('proxy_url'))
    for name in ('provider', 'model', 'base_url'):
        value = values.get(name)
        if not isinstance(value, str):
            errors[name] = labels.SETTINGS_STRING_FIELD
        else:
            strings[name] = value if name == 'base_url' else value.strip()
            if value.strip() and not _safe_text(value, secrets):
                errors[name] = labels.SETTINGS_SAFE_FIELD
    if 'api_key' in values and not isinstance(posted_key, str):
        errors['api_key'] = labels.SETTINGS_STRING_FIELD
    for name in ('proxy_mode', 'proxy_url'):
        if name in values and not isinstance(values[name], str):
            errors[name] = labels.SETTINGS_STRING_FIELD
    for name in ('analysis_model', 'processing_model'):
        if name in values:
            value = values[name]
            if not isinstance(value, str):
                errors[name] = labels.SETTINGS_STRING_FIELD
            else:
                strings[name] = value.strip()
                if value.strip() and not _safe_text(value, secrets):
                    errors[name] = labels.SETTINGS_SAFE_FIELD
    concurrency = _integer(values.get('concurrency'), form=form)
    if concurrency is None:
        errors['concurrency'] = labels.SETTINGS_INTEGER_FIELD
    raw_rate = values.get('requests_per_minute')
    rate = None if raw_rate is None or (form and isinstance(raw_rate, str) and not raw_rate.strip()) else _integer(raw_rate, form=form)
    if raw_rate is not None and not (form and isinstance(raw_rate, str) and not raw_rate.strip()) and rate is None:
        errors['requests_per_minute'] = labels.SETTINGS_INTEGER_FIELD
    temp: float | None | object = UNSET
    if 'temperature' in values:
        raw_temp = values['temperature']
        if raw_temp is None or (isinstance(raw_temp, str) and not raw_temp.strip()):
            temp = None
        else:
            temp = _float(raw_temp, form=form)
            if temp is None:
                errors['temperature'] = 'Ожидается число'
            elif not (0.0 <= temp <= 2.0):
                errors['temperature'] = 'Значение должно быть от 0.0 до 2.0'
    raw_timeout = values.get('timeout_seconds')
    timeout = None
    if raw_timeout is not None and not (form and isinstance(raw_timeout, str) and not raw_timeout.strip()):
        timeout = _float(raw_timeout, form=form)
        if timeout is None:
            errors['timeout_seconds'] = 'Ожидается число'
        elif timeout <= 0:
            errors['timeout_seconds'] = 'Ожидается положительное число'
        elif timeout > 600.0:
            errors['timeout_seconds'] = 'Значение не должно превышать 600 секунд'
    if not errors:
        try:
            update = SystemSettingsUpdate(
                **strings,
                requests_per_minute=rate,
                concurrency=concurrency,
                api_key=posted_key,
                proxy_mode=values.get('proxy_mode'),
                proxy_url=values.get('proxy_url'),
                temperature=temp,
                timeout_seconds=timeout,
            )
        except SystemSettingsValidationError as error:
            errors[error.field] = error.message
    if errors:
        raise SettingsInputError(field_errors=errors, form_values=_projection(values, current_settings, form=form))
    return update, revision


async def read_settings_reset(request: Request, *, current_settings: AppSettings) -> int:
    values, form = await _read(request, current_settings, reset=True)
    errors = {}
    revision = _revision(values, form=form, errors=errors)
    if errors:
        raise SettingsInputError(field_errors=errors, form_values=safe_form_values(current_settings))
    return revision
