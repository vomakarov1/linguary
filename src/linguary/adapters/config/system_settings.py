"""Проверка и применение редактируемых настроек без раскрытия исходного ввода."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from linguary.adapters.config.settings import AppSettings, OPENAI_COMPATIBLE_PROVIDER, SettingsError, validate_proxy

if TYPE_CHECKING:
    from linguary.adapters.db.system_settings import StoredSystemSettings


class SystemSettingsValidationError(SettingsError):
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(message)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SystemSettingsValidationError(name, 'Ожидается строка')
    return value.strip()


def _positive_integer(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise SystemSettingsValidationError(name, 'Ожидается положительное целое число')


BASE_OVERRIDE_FIELDS = frozenset({'provider', 'model', 'base_url', 'requests_per_minute', 'concurrency'})
OPTIONAL_OVERRIDE_FIELDS = frozenset({'proxy_mode', 'analysis_model', 'processing_model', 'temperature', 'timeout_seconds'})
ALLOWED_OVERRIDE_FIELDS = BASE_OVERRIDE_FIELDS | OPTIONAL_OVERRIDE_FIELDS


class _UnsetSentinel:
    def __repr__(self) -> str:
        return '<UNSET>'

    def __bool__(self) -> bool:
        return False


UNSET = _UnsetSentinel()


@dataclass(frozen=True, slots=True)
class SystemSettingsUpdate:
    provider: str = field(repr=False)
    model: str = field(repr=False)
    base_url: str = field(repr=False)
    requests_per_minute: int | None
    concurrency: int
    api_key: str | None = field(default=None, repr=False)

    proxy_mode: str | None = None
    proxy_url: str | None = field(default=None, repr=False)

    analysis_model: str | None = None
    processing_model: str | None = None
    temperature: float | None | object = UNSET
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        mode = self.proxy_mode
        if mode is None and self.proxy_url is not None:
            raise SystemSettingsValidationError('proxy_url', 'Укажите режим соединения для адреса прокси')
        if mode is not None and mode not in ('auto', 'direct', 'custom'):
            raise SystemSettingsValidationError('proxy_mode', 'Недопустимый режим соединения')
        if mode != 'custom':
            url = None
        else:
            url = None if self.proxy_url is None else _string(self.proxy_url, 'proxy_url') or None
            if url is not None:
                try:
                    validate_proxy(mode, self.proxy_url)
                except SettingsError as error:
                    raise SystemSettingsValidationError('proxy_url', str(error)) from None
        object.__setattr__(self, 'proxy_url', url)
        provider = _string(self.provider, 'provider')
        model = _string(self.model, 'model')
        base_url = _string(self.base_url, 'base_url')
        if provider != OPENAI_COMPATIBLE_PROVIDER:
            raise SystemSettingsValidationError('provider', 'Провайдер не поддерживается')
        if any(ord(char) < 32 or ord(char) == 127 for char in self.base_url):
            raise SystemSettingsValidationError('base_url', 'Недопустимый адрес провайдера')
        try:
            url = urlsplit(base_url)
            valid = (url.scheme in ('http', 'https') and bool(url.hostname)
                     and url.username is None and url.password is None
                     and '?' not in base_url and '#' not in base_url
                     and not any(char.isspace() for char in base_url))
            url.port
        except ValueError:
            valid = False
        if not valid:
            raise SystemSettingsValidationError('base_url', 'Ожидается абсолютный HTTP(S)-адрес без учётных данных, параметров и фрагмента')
        _positive_integer(self.concurrency, 'concurrency')
        if self.requests_per_minute is not None:
            _positive_integer(self.requests_per_minute, 'requests_per_minute')
        key = None if self.api_key is None else _string(self.api_key, 'api_key') or None
        for name, value in (('provider', provider), ('model', model), ('base_url', base_url), ('api_key', key)):
            object.__setattr__(self, name, value)

        analysis_model = None if self.analysis_model is None else _string(self.analysis_model, 'analysis_model')
        processing_model = None if self.processing_model is None else _string(self.processing_model, 'processing_model')
        if self.temperature is not None and self.temperature is not UNSET:
            if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
                raise SystemSettingsValidationError('temperature', 'Ожидается число')
            if not (0.0 <= float(self.temperature) <= 2.0):
                raise SystemSettingsValidationError('temperature', 'Значение должно быть от 0.0 до 2.0')
            object.__setattr__(self, 'temperature', float(self.temperature))
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
                raise SystemSettingsValidationError('timeout_seconds', 'Ожидается число')
            if float(self.timeout_seconds) <= 0:
                raise SystemSettingsValidationError('timeout_seconds', 'Ожидается положительное число')
            if float(self.timeout_seconds) > 600.0:
                raise SystemSettingsValidationError('timeout_seconds', 'Значение не должно превышать 600 секунд')
            object.__setattr__(self, 'timeout_seconds', float(self.timeout_seconds))
        if analysis_model is not None:
            object.__setattr__(self, 'analysis_model', analysis_model)
        if processing_model is not None:
            object.__setattr__(self, 'processing_model', processing_model)


def candidate_settings(current: AppSettings, update: SystemSettingsUpdate) -> AppSettings:
    mode = current.llm.proxy_mode if update.proxy_mode is None else update.proxy_mode
    if update.proxy_mode is None:
        url = current.llm.proxy_url
    elif mode == 'custom':
        url = update.proxy_url or current.llm.proxy_url
    else:
        url = None
    if mode == 'custom' and url is None:
        raise SystemSettingsValidationError('proxy_url', 'Укажите адрес прокси')

    analysis_model = (
        current.llm.analysis_model if update.analysis_model is None
        else update.analysis_model
    )
    processing_model = (
        current.llm.processing_model if update.processing_model is None
        else update.processing_model
    )
    temperature = (
        current.llm.temperature if update.temperature is UNSET
        else update.temperature
    )
    timeout_seconds = (
        current.llm.timeout_seconds if update.timeout_seconds is None
        else update.timeout_seconds
    )

    return replace(
        current,
        llm=replace(
            current.llm,
            proxy_mode=mode,
            proxy_url=url,
            provider=update.provider,
            model=update.model,
            analysis_model=analysis_model,
            processing_model=processing_model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            base_url=update.base_url,
            requests_per_minute=update.requests_per_minute,
            api_key=update.api_key if update.api_key is not None else current.llm.api_key,
        ),
        worker=replace(current.worker, concurrency=update.concurrency),
    )


def overlay_settings(initial: AppSettings, stored: StoredSystemSettings | None) -> AppSettings:
    if stored is None or stored.overrides is None:
        return initial
    if not isinstance(stored.overrides, dict) or not (BASE_OVERRIDE_FIELDS <= set(stored.overrides) <= ALLOWED_OVERRIDE_FIELDS):
        raise SystemSettingsValidationError('form', 'Сохранённые настройки имеют неверный формат')
    update = SystemSettingsUpdate(**(dict(proxy_mode='auto') | stored.overrides), api_key=stored.api_key, proxy_url=stored.proxy_url)
    return candidate_settings(initial, update)

