"""LLM client over OpenAI-compatible chat completions HTTP API (ADR § 3).

Implementation of port `linguary.application.ports.llm.LLMClient` for the MVP's primary provider.
A compatible contract was chosen over vendor-specific SDKs: the same adapter serves gateways
and local inference servers, differing only in base URL and configured model — a second adapter
is not needed for this.

Requests always demand a strict structured output schema corresponding to the request kind:
free text is rejected by the MVP design (PRD § 23).

Errors are mapped to port exception types right here where distinguishing context is available:
authorization failure reflects configuration, rate limits and server errors are transient,
and rejected requests are permanent. The provider API key is never exposed in error messages (PRD § 24).
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
from linguary.application.diagnostics import bounded_body

#: Path for chat completion endpoint in compatible API.
_COMPLETIONS_PATH = "chat/completions"

#: HTTP status codes indicating rejected credentials or model access.
_AUTHORIZATION_STATUSES = frozenset({401, 403})

#: Rate limit status code: transient failure (PRD § 16).
_RATE_LIMIT_STATUS = 429

#: Insufficient credits status code: retrying without top-up produces identical result.
_PAYMENT_REQUIRED_STATUS = 402

#: Finish reason indicating generation cut off due to token limits.
_LENGTH_FINISH_REASON = "length"

#: Number of response characters included in error details.
_DETAIL_LIMIT = 300


class OpenAICompatibleLLMClient(LLMClient):
    """Model invocation via OpenAI-compatible HTTP API."""

    def __init__(
        self, *, settings: LLMSettings, http_client: httpx.AsyncClient | None = None
    ) -> None:
        """Creates the client; unconfigured LLM settings prevent creation altogether.

        Validation is placed in the constructor rather than invocation: analysis and processing
        workflows must not start without a key and model (PRD § 23), ensuring a built application
        either possesses a working client or surfaces its configuration state.
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
        """Executes request and returns parsed structured response."""
        document = await self._post(_body(request, settings=self._settings))
        usage = _usage(document)
        cost = _cost(document)
        model = safe_detail(
            _model(document, fallback=self._settings.model), api_key=self._settings.api_key, secrets=self._proxy_secrets
        )
        content = None
        truncated = False
        try:
            choice = _choice(document)
            truncated = _finish_reason(choice) == _LENGTH_FINISH_REASON
            content = _content(choice)
            payload = parse_payload(request.kind, content, contract_version=request.contract_version, name_translation_policy_version=request.name_translation_policy_version, truncated=truncated)
        except LLMResponseFormatError as error:
            safe_content, body_meta = bounded_body(content,
                secrets=(*self._proxy_secrets, self._settings.api_key))
            error.raw_content = safe_content
            error.body_meta = body_meta
            error.args = (self._safe(str(error))[:300],)
            error.usage, error.cost = usage, cost
            error.provider, error.model = self._settings.provider, model
            error.call_context = request.call_context
            error.__cause__ = None
            error.__context__ = None
            error.__suppress_context__ = True
            raise error from None
        safe_content, body_meta = bounded_body(content,
            secrets=(*self._proxy_secrets, self._settings.api_key))
        return LLMResponse(
            payload=payload,  # type: ignore[arg-type]  # schema selected by request.kind
            provider=self._settings.provider,
            model=model,
            truncated=truncated,
            usage=usage,
            cost=cost,
            raw_content=safe_content,
            body_meta=body_meta,
            call_context=request.call_context,
        )

    def _safe(self, text: str) -> str:
        return safe_detail(text, api_key=self._settings.api_key, secrets=self._proxy_secrets)

    async def aclose(self) -> None:
        """Closes underlying HTTP connections if created internally."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> OpenAICompatibleLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Sends request to provider and returns parsed JSON body on success."""
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
    """Full URL to completions endpoint."""
    return f"{base_url.rstrip('/')}/{_COMPLETIONS_PATH}"


def _headers(settings: LLMSettings) -> dict[str, str]:
    """Request headers including provider authorization token."""
    return {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }


def _body(request: LLMRequest[Any], *, settings: LLMSettings) -> dict[str, Any]:
    """Request payload: message list and required structured output schema."""
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
                "schema": strict_json_schema(schema_for(request.kind, contract_version=request.contract_version, name_translation_policy_version=request.name_translation_policy_version)),
            },
        },
    }
    if request.max_completion_tokens is not None:
        if type(request.max_completion_tokens) is not int or request.max_completion_tokens <= 0:
            raise LLMConfigurationError('неверный общий предел выходных токенов')
        body['max_completion_tokens'] = request.max_completion_tokens
    elif settings.max_output_tokens is not None:
        body["max_tokens"] = settings.max_output_tokens
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    return body


def _raise_for_status(
    response: httpx.Response, *, provider: str, api_key: str | None, secrets: tuple[str, ...] = ()
) -> None:
    """Maps provider error response status to port exception hierarchy (PRD § 16)."""
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
    """Retry-After delay in seconds or parsed HTTP-date timestamp.

    An unparseable or negative value is treated as absent: the provider communicated
    no usable value, leaving delay determination to the retry policy.
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
    """Seconds remaining until the target HTTP-date timestamp."""
    try:
        moment = parsedate_to_datetime(http_date)
    except (TypeError, ValueError, IndexError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - _utc_now()).total_seconds())


def _utc_now() -> datetime:
    """Current timestamp in UTC."""
    return datetime.now(UTC)


def _document(response: httpx.Response, *, api_key: str | None, secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Successful response JSON body."""
    try:
        document = response.json()
    except ValueError as error:
        raise LLMResponseFormatError(
            "ответ провайдера не является JSON",
            raw_content=None,
        ) from None
    if not isinstance(document, dict):
        raise LLMResponseFormatError(
            "ответ провайдера не похож на ответ генерации",
            raw_content=None,
        )
    return document


def _choice(document: dict[str, Any]) -> dict[str, Any]:
    """First generation choice from the model response."""
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseFormatError("ответ провайдера не содержит вариантов генерации")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseFormatError("вариант генерации в ответе провайдера повреждён")
    return first


def _content(choice: dict[str, Any]) -> str:
    """Structured response message text."""
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMResponseFormatError("модель вернула ответ без содержимого")
    return content


def _finish_reason(choice: dict[str, Any]) -> str | None:
    """Generation finish reason reported by provider, if present."""
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else None


def _model(document: dict[str, Any], *, fallback: str) -> str:
    """Model name reported by provider for this call.

    Compatible gateways often substitute actual model identifiers, which should
    be accurately recorded in call logs (PRD § 24).
    """
    model = document.get("model")
    return model if isinstance(model, str) and model.strip() else fallback


def _usage(document: dict[str, Any]) -> TokenUsage | None:
    """Token usage metrics reported by provider (PRD § 24)."""
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
    """Call cost reported by provider if supported by gateway."""
    usage = document.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("cost")
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        return None
    cost = Decimal(str(value))
    return cost if cost.is_finite() and cost >= 0 else None


def safe_detail(text: str, *, api_key: str | None, secrets: tuple[str, ...] = ()) -> str:
    """Redacts secrets and API keys prior to truncating technical details."""
    for secret in sorted((*secrets, *((api_key,) if api_key else ())), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def _detail(response: httpx.Response, *, api_key: str | None, secrets: tuple[str, ...] = ()) -> str:
    """Appended provider response snippet for technical error details."""
    try:
        text = safe_detail(response.text, api_key=api_key, secrets=secrets).strip()
    except (UnicodeDecodeError, httpx.ResponseNotRead):
        return ""
    return f": {text[:_DETAIL_LIMIT]}" if text else ""


__all__ = ["OpenAICompatibleLLMClient"]
