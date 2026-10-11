"""Клиент LLM поверх HTTP API, совместимого с OpenAI chat completions (ADR § 3).

Реализация порта `linguary.application.ports.llm.LLMClient` для первого провайдера MVP.
Выбран совместимый контракт, а не SDK конкретного вендора: тот же адаптер обслуживает
шлюзы и локальные серверы, отличаясь только базовым адресом и моделью в настройке, —
второй адаптер для этого не нужен.

Запрос всегда требует строгий структурированный ответ по схеме вида запроса: свободный
текст модели MVP не принимает (PRD § 23).

Ошибки приводятся к типам порта прямо здесь, потому что различить их можно только тут:
отказ по ключу — это настройка, ограничение частоты и сбой провайдера — временная
неудача, отвергнутый запрос — постоянная. Ключ провайдера не попадает ни в одно
сообщение об ошибке (PRD § 24).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from urllib.request import getproxies
from typing import Any

import httpx

from linguary.adapters.config.settings import LLMSettings, proxy_secrets
from linguary.adapters.llm.json_schema import strict_json_schema
from linguary.adapters.llm.payloads import parse_payload, schema_for
from linguary.application.ports.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMPayload,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.domain import TokenUsage

#: Путь метода генерации в совместимом API.
_COMPLETIONS_PATH = "chat/completions"

#: Коды, означающие, что провайдер не принял ключ или доступ к модели.
_AUTHORIZATION_STATUSES = frozenset({401, 403})

#: Код ограничения частоты запросов: временная неудача (PRD § 16).
_RATE_LIMIT_STATUS = 429

#: Код нехватки средств на счёте провайдера: повтор без пополнения ответит тем же.
_PAYMENT_REQUIRED_STATUS = 402

#: Причина завершения, которой провайдер сообщает об обрыве по лимиту токенов.
_LENGTH_FINISH_REASON = "length"

#: Сколько символов ответа провайдера попадает в текст ошибки.
_DETAIL_LIMIT = 300


class OpenAICompatibleLLMClient(LLMClient):
    """Вызов модели через совместимый с OpenAI HTTP API."""

    def __init__(
        self, *, settings: LLMSettings, http_client: httpx.AsyncClient | None = None
    ) -> None:
        """Создаёт клиент; ненастроенная LLM не даёт создать его вовсе.

        Проверка стоит в конструкторе, а не в вызове: сценарии анализа и обработки
        не должны стартовать без ключа и модели (PRD § 23), а собранное приложение
        либо имеет рабочий клиент, либо показывает состояние конфигурации.
        """
        settings.ensure_ready()
        self._settings = settings
        urls = ()
        if settings.proxy_mode == 'custom':
            urls = (settings.proxy_url,)
        elif settings.proxy_mode == 'auto':
            urls = tuple(value for key, value in getproxies().items() if key in ('http', 'https', 'all'))
        self._proxy_secrets = tuple(secret for url in urls for secret in proxy_secrets(url))
        try:
            self._http = http_client if http_client is not None else httpx.AsyncClient(
                timeout=settings.timeout_seconds, trust_env=settings.proxy_mode == 'auto',
                **({'proxy': settings.proxy_url} if settings.proxy_mode == 'custom' else {}),
            )
        except Exception:
            raise LLMConfigurationError('Не удалось подготовить соединение с провайдером') from None
        self._owns_http = http_client is None

    async def generate[PayloadT: LLMPayload](
        self, request: LLMRequest[PayloadT]
    ) -> LLMResponse[PayloadT]:
        """Выполняет запрос и возвращает разобранный структурированный ответ."""
        document = await self._post(_body(request, settings=self._settings))
        usage = _usage(document)
        cost = _cost(document)
        model = safe_detail(
            _model(document, fallback=self._settings.model), api_key=self._settings.api_key, secrets=self._proxy_secrets
        )
        try:
            choice = _choice(document)
            truncated = _finish_reason(choice) == _LENGTH_FINISH_REASON
            content = _content(choice)
            payload = parse_payload(request.kind, content, truncated=truncated)
        except LLMResponseFormatError as error:
            error.args = (safe_detail(str(error), api_key=self._settings.api_key, secrets=self._proxy_secrets),)
            if error.raw_content is not None:
                error.raw_content = safe_detail(
                    error.raw_content, api_key=self._settings.api_key, secrets=self._proxy_secrets
                )
            error.usage = usage
            error.cost = cost
            error.provider = self._settings.provider
            error.model = model
            if error.__cause__ is not None and self._safe(str(error.__cause__)) != str(error.__cause__):
                error.__cause__ = None
                error.__context__ = None
                error.__suppress_context__ = True
            raise
        return LLMResponse(
            payload=payload,  # type: ignore[arg-type]  # схему выбирает request.kind
            provider=self._settings.provider,
            model=model,
            truncated=truncated,
            usage=usage,
            cost=cost,
            raw_content=safe_detail(content, api_key=self._settings.api_key, secrets=self._proxy_secrets),
        )

    def _safe(self, text: str) -> str:
        return safe_detail(text, api_key=self._settings.api_key, secrets=self._proxy_secrets)

    async def aclose(self) -> None:
        """Закрывает HTTP-соединения, если клиент создавал их сам."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> OpenAICompatibleLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Отправляет запрос провайдеру и отдаёт разобранное тело успешного ответа."""
        try:
            response = await self._http.post(
                _endpoint(self._settings.base_url),
                json=body,
                headers=_headers(self._settings),
                timeout=self._settings.timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise LLMTransportError(
                f"провайдер {self._settings.provider} не ответил за "
                f"{self._settings.timeout_seconds:g} с"
            ) from None
        except httpx.HTTPError as error:
            raise LLMTransportError(
                safe_detail(
                    f"вызов провайдера {self._settings.provider} не удался: {self._safe(str(error))}",
                    api_key=self._settings.api_key, secrets=self._proxy_secrets,
                )
            ) from None
        _raise_for_status(
            response, provider=self._settings.provider, api_key=self._settings.api_key, secrets=self._proxy_secrets
        )
        return _document(response, api_key=self._settings.api_key, secrets=self._proxy_secrets)


def _endpoint(base_url: str) -> str:
    """Полный адрес метода генерации."""
    return f"{base_url.rstrip('/')}/{_COMPLETIONS_PATH}"


def _headers(settings: LLMSettings) -> dict[str, str]:
    """Заголовки запроса вместе с ключом провайдера."""
    return {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }


def _body(request: LLMRequest[Any], *, settings: LLMSettings) -> dict[str, Any]:
    """Тело запроса: сообщения и схема требуемого структурированного ответа."""
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": request.instructions},
            {"role": "user", "content": request.input_text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.kind.value,
                "strict": True,
                "schema": strict_json_schema(schema_for(request.kind)),
            },
        },
    }
    if settings.max_output_tokens is not None:
        body["max_tokens"] = settings.max_output_tokens
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    return body


def _raise_for_status(
    response: httpx.Response, *, provider: str, api_key: str | None, secrets: tuple[str, ...] = ()
) -> None:
    """Приводит неуспешный ответ провайдера к типу ошибки порта (PRD § 16)."""
    status = response.status_code
    if status < 400:
        return
    detail = f"{status}{_detail(response, api_key=api_key, secrets=secrets)}"
    if status in _AUTHORIZATION_STATUSES:
        raise LLMConfigurationError(
            f"провайдер {provider} не принял ключ или доступ к модели: {detail}"
        )
    if status == _PAYMENT_REQUIRED_STATUS:
        raise LLMProviderError(
            f"у провайдера {provider} не хватает средств на счёте: {detail}"
        )
    if status == _RATE_LIMIT_STATUS or status >= 500:
        raise LLMTransportError(
            f"провайдер {provider} временно недоступен: {detail}",
            retry_after_seconds=_retry_after(response),
        )
    raise LLMProviderError(f"провайдер {provider} отверг запрос: {detail}")


def _retry_after(response: httpx.Response) -> float | None:
    """Пауза из заголовка `Retry-After`: число секунд или момент в формате HTTP-даты.

    Нечитаемое или отрицательное значение считается несообщённым: провайдер ничего
    внятного не сказал, и паузу выбирает политика повторов сама.
    """
    raw = response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        seconds = _seconds_until(raw)
    if seconds is None or seconds != seconds or seconds < 0:
        return None
    return seconds


def _seconds_until(http_date: str) -> float | None:
    """Сколько секунд осталось до момента, заданного HTTP-датой."""
    try:
        moment = parsedate_to_datetime(http_date)
    except (TypeError, ValueError, IndexError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - _utc_now()).total_seconds())


def _utc_now() -> datetime:
    """Текущий момент в UTC."""
    return datetime.now(UTC)


def _document(response: httpx.Response, *, api_key: str | None, secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Тело успешного ответа провайдера."""
    try:
        document = response.json()
    except ValueError as error:
        raise LLMResponseFormatError(
            "ответ провайдера не является JSON",
            raw_content=safe_detail(response.text, api_key=api_key, secrets=secrets),
        ) from None
    if not isinstance(document, dict):
        raise LLMResponseFormatError(
            "ответ провайдера не похож на ответ генерации",
            raw_content=safe_detail(response.text, api_key=api_key, secrets=secrets),
        )
    return document


def _choice(document: dict[str, Any]) -> dict[str, Any]:
    """Первый вариант ответа модели."""
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseFormatError("ответ провайдера не содержит вариантов генерации")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseFormatError("вариант генерации в ответе провайдера повреждён")
    return first


def _content(choice: dict[str, Any]) -> str:
    """Текст структурированного ответа модели."""
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMResponseFormatError("модель вернула ответ без содержимого")
    return content


def _finish_reason(choice: dict[str, Any]) -> str | None:
    """Причина завершения генерации, если провайдер её сообщает."""
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else None


def _model(document: dict[str, Any], *, fallback: str) -> str:
    """Модель, которой провайдер выполнил запрос.

    Совместимые шлюзы подставляют собственное имя модели, и в журнал вызовов должно
    попасть именно оно (PRD § 24).
    """
    model = document.get("model")
    return model if isinstance(model, str) and model.strip() else fallback


def _usage(document: dict[str, Any]) -> TokenUsage | None:
    """Объём токенов вызова, если провайдер его сообщает (PRD § 24)."""
    usage = document.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    return TokenUsage(
        prompt_tokens=prompt if isinstance(prompt, int) else None,
        completion_tokens=completion if isinstance(completion, int) else None,
    )


def _cost(document: dict[str, Any]) -> Decimal | None:
    """Стоимость вызова по данным провайдера, если шлюз сообщил число."""
    usage = document.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("cost")
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        return None
    cost = Decimal(str(value))
    return cost if cost.is_finite() and cost >= 0 else None


def safe_detail(text: str, *, api_key: str | None, secrets: tuple[str, ...] = ()) -> str:
    """Убирает ключ этого клиента до усечения технических деталей."""
    for secret in sorted((*secrets, *((api_key,) if api_key else ())), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def _detail(response: httpx.Response, *, api_key: str | None, secrets: tuple[str, ...] = ()) -> str:
    """Хвост ответа провайдера для технической детали ошибки."""
    try:
        text = safe_detail(response.text, api_key=api_key, secrets=secrets).strip()
    except (UnicodeDecodeError, httpx.ResponseNotRead):
        return ""
    return f": {text[:_DETAIL_LIMIT]}" if text else ""


__all__ = ["OpenAICompatibleLLMClient"]
