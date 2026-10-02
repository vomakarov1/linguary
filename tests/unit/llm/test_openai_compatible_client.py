"""Клиент LLM поверх совместимого HTTP API: запрос, разбор ответа и разбор неудач.

Сети в тестах нет: провайдер подменяется транспортом httpx, поэтому проверяется ровно
то, что делает адаптер, — тело запроса, чтение ответа и приведение неудач к типам порта.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest
from llm_payload_factories import analysis_payload, as_content, translation_payload
from pydantic import ValidationError

from linguary.adapters.config import LLMSettings
from linguary.adapters.llm import OpenAICompatibleLLMClient, openai_compatible
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRequest,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.domain import TokenUsage

API_KEY = "test-key-must-not-leak"


def settings(**overrides: Any) -> LLMSettings:
    """Настроенная LLM: модель и ключ заданы."""
    values: dict[str, Any] = {
        "base_url": "https://provider.test/v1",
        "model": "test-model",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return LLMSettings(**values)


def completion(content: str, **overrides: Any) -> dict[str, Any]:
    """Ответ совместимого API вокруг структурированного содержимого."""
    document: dict[str, Any] = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 340},
    }
    document.update(overrides)
    return document


def client_over(handler: Any, **settings_overrides: Any) -> OpenAICompatibleLLMClient:
    """Клиент поверх подменённого транспорта."""
    return OpenAICompatibleLLMClient(
        settings=settings(**settings_overrides),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def responding(document: dict[str, Any], *, status_code: int = 200) -> Any:
    """Транспорт, отвечающий одним и тем же документом."""
    return lambda request: httpx.Response(status_code, json=document)


def schema_error_content() -> str:
    """Минимальный ответ с пустым переводом TRANSLATE из воспроизведения дефекта."""
    return as_content({
        "blocks": [{
            "source_block_id": "b1", "treatment": "TRANSLATE",
            "translated_text": "", "glossary_hints": [],
        }],
        "language_notes": [],
    })


async def format_error_from(document: dict[str, Any]) -> LLMResponseFormatError:
    """Получает ошибку через настоящий HTTP-клиент без обращения к сети."""
    client = client_over(responding(document))
    with pytest.raises(LLMResponseFormatError) as caught:
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="b1: text")
        )
    return caught.value


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4.6", None])
async def test_schema_error_preserves_usage_and_provider_cost(model: str | None) -> None:
    content = schema_error_content()
    error = await format_error_from(completion(
        content, model=model,
        usage={"prompt_tokens": 24480, "completion_tokens": 1122, "cost": 0.09027},
    ))

    assert error.usage == TokenUsage(24480, 1122)
    assert error.cost == Decimal("0.09027")
    assert error.provider == settings().provider
    assert error.model == (model or settings().model)
    assert error.truncated is False
    assert error.raw_content == content
    assert isinstance(error.__cause__, ValidationError)
    assert error.__cause__.errors(include_input=False)[0]["type"] == "value_error"
    assert "не соответствует схеме" in str(error)
    for private in (content, "b1", "TRANSLATE", API_KEY):
        assert private not in str(error)


async def test_invalid_json_error_preserves_usage_and_provider_cost() -> None:
    content = '{"private": "secret book text"'
    error = await format_error_from(completion(
        content, usage={"prompt_tokens": 24480, "completion_tokens": 1122, "cost": 0.09027},
    ))

    assert error.usage == TokenUsage(24480, 1122)
    assert error.cost == Decimal("0.09027")
    assert error.raw_content == content
    assert isinstance(error.__cause__, ValidationError)
    assert error.__cause__.errors(include_input=False)[0]["type"] == "json_invalid"
    assert "secret book text" not in str(error)


async def test_truncated_format_error_preserves_usage_cost_and_flag() -> None:
    content = '{"private": "secret book text"'
    error = await format_error_from(completion(
        content,
        choices=[{"message": {"content": content}, "finish_reason": "length"}],
        usage={"prompt_tokens": 24480, "completion_tokens": 1122, "cost": 0.09027},
    ))

    assert error.usage == TokenUsage(24480, 1122)
    assert error.cost == Decimal("0.09027")
    assert error.truncated is True
    assert error.raw_content == content
    assert isinstance(error.__cause__, ValidationError)
    assert "обрыв ответа по лимиту токенов" in str(error)
    assert "secret book text" not in str(error)


@pytest.mark.parametrize("choices", [[], ["secret envelope text"], [{"message": {}}]])
async def test_generation_envelope_error_preserves_usage_and_cost(choices: list[Any]) -> None:
    error = await format_error_from({
        "model": "gateway/other-model", "choices": choices,
        "usage": {"prompt_tokens": 24480, "completion_tokens": 1122, "cost": 0.09027},
    })

    assert error.usage == TokenUsage(24480, 1122)
    assert error.cost == Decimal("0.09027")
    assert error.provider == settings().provider
    assert error.model == "gateway/other-model"
    assert error.raw_content is None
    assert error.__cause__ is None
    assert "secret envelope text" not in str(error)


async def test_format_error_without_usage_keeps_usage_and_cost_none() -> None:
    document = completion(schema_error_content())
    document.pop("usage")
    error = await format_error_from(document)

    assert error.usage is None
    assert error.cost is None
    assert error.provider == settings().provider
    assert error.model == settings().model


async def test_unreadable_http_document_keeps_metadata_none() -> None:
    client = client_over(lambda request: httpx.Response(200, text="private HTTP text"))
    with pytest.raises(LLMResponseFormatError) as caught:
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="b1: text")
        )

    assert caught.value.usage is None
    assert caught.value.cost is None
    assert caught.value.provider is None
    assert caught.value.model is None
    assert "private HTTP text" not in str(caught.value)


async def test_successful_call_returns_parsed_payload() -> None:
    """Успешный вызов даёт разобранный DTO вместе с данными вызова."""
    client = client_over(responding(completion(as_content(translation_payload()))))

    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    assert isinstance(response.payload, TranslationResultResponse)
    assert response.payload.block_ids == ("block-1", "block-2")
    assert response.model == "test-model"
    assert response.provider == settings().provider
    assert response.truncated is False
    assert response.usage is not None
    assert response.usage.total_tokens == 460


async def test_client_reads_provider_cost() -> None:
    document = completion(as_content(translation_payload()))
    document["usage"]["cost"] = 0.00123
    client = client_over(responding(document))
    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )
    assert response.cost == Decimal("0.00123")


async def test_client_without_provider_cost_leaves_none() -> None:
    client = client_over(responding(completion(as_content(translation_payload()))))
    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )
    assert response.cost is None


async def test_chapter_analysis_call_returns_analysis_payload() -> None:
    """Запрос анализа главы разбирается по схеме анализа."""
    client = client_over(responding(completion(as_content(analysis_payload()))))

    response = await client.generate(
        LLMRequest.for_chapter_analysis(instructions="Анализируй", input_text="block-1: text")
    )

    assert isinstance(response.payload, ChapterAnalysisResponse)
    assert response.payload.planned_block_ids == ("block-1", "block-2")


async def test_request_asks_provider_for_structured_output() -> None:
    """Запрос содержит ключ, инструкцию, содержимое и схему строгого ответа."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    client = client_over(handler, max_output_tokens=4096, temperature=0.2)

    await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    request = sent[0]
    body = _json(request)
    assert str(request.url) == "https://provider.test/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert body["model"] == "test-model"
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == "Переводи"
    assert body["messages"][1]["content"] == "block-1: text"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.2
    schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert schema["strict"] is True
    assert schema["name"] == "translation_result"
    assert set(schema["schema"]["properties"]) == {"blocks", "language_notes"}


async def test_optional_generation_settings_are_omitted_when_unset() -> None:
    """Незаданные лимит вывода и температура в запрос не попадают."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    client = client_over(handler)

    await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    body = _json(sent[0])
    assert "max_tokens" not in body
    assert "temperature" not in body


async def test_token_limit_stop_is_reported() -> None:
    """Обрыв ответа по лимиту токенов виден прикладному слою (PRD § 15)."""
    document = completion(
        as_content(translation_payload()),
        choices=[
            {
                "message": {"content": as_content(translation_payload())},
                "finish_reason": "length",
            }
        ],
    )
    client = client_over(responding(document))

    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    assert response.truncated is True


async def test_truncated_content_gives_format_error_with_token_limit_flag() -> None:
    """Оборванный по лимиту токенов ответ не разбирается и сообщает о причине."""
    document = completion(
        '{"blocks": [{"source',
        choices=[{"message": {"content": '{"blocks": [{"source'}, "finish_reason": "length"}],
    )
    client = client_over(responding(document))

    with pytest.raises(LLMResponseFormatError) as error:
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )

    assert error.value.truncated is True
    assert error.value.raw_content == '{"blocks": [{"source'
    assert '{"blocks"' not in str(error.value)


async def test_missing_usage_is_not_invented() -> None:
    """Провайдер без данных о токенах не даёт выдуманного объёма (PRD § 24)."""
    document = completion(as_content(translation_payload()))
    document.pop("usage")
    client = client_over(responding(document))

    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    assert response.usage is None


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        pytest.param(401, LLMConfigurationError, id="unauthorized"),
        pytest.param(403, LLMConfigurationError, id="forbidden"),
        pytest.param(429, LLMTransportError, id="rate-limited"),
        pytest.param(500, LLMTransportError, id="provider-failure"),
        pytest.param(503, LLMTransportError, id="provider-unavailable"),
        pytest.param(400, LLMProviderError, id="rejected-request"),
        pytest.param(404, LLMProviderError, id="unknown-model"),
        pytest.param(402, LLMProviderError, id="payment-required"),
    ],
)
async def test_provider_status_is_mapped_to_port_error(
    status_code: int, expected: type[Exception]
) -> None:
    """Каждый неуспешный код провайдера превращается в свой тип ошибки порта (PRD § 16)."""
    client = client_over(responding({"error": {"message": "нет"}}, status_code=status_code))

    with pytest.raises(expected):
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )


async def test_payment_required_names_the_balance() -> None:
    """Отказ из-за нехватки средств называет причину, а не ключ или модель (PRD § 16)."""
    client = client_over(responding({"error": {"message": "нет"}}, status_code=402))

    with pytest.raises(LLMProviderError, match="не хватает средств"):
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )


@pytest.mark.parametrize(
    ("status_code", "header", "expected"),
    [
        pytest.param(429, "7", 7.0, id="rate-limited-seconds"),
        pytest.param(503, "2.5", 2.5, id="unavailable-fractional-seconds"),
        pytest.param(
            429, "Wed, 21 Oct 2015 07:28:10 GMT", 10.0, id="rate-limited-http-date"
        ),
        pytest.param(429, "soon", None, id="unreadable"),
        pytest.param(429, "-3", None, id="negative"),
        pytest.param(429, None, None, id="absent"),
    ],
)
async def test_retry_after_is_passed_with_transient_error(
    status_code: int,
    header: str | None,
    expected: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пауза, которую провайдер просит заголовком `Retry-After`, доходит до политики повторов."""
    monkeypatch.setattr(
        openai_compatible, "_utc_now", lambda: datetime(2015, 10, 21, 7, 28, 0, tzinfo=UTC)
    )
    headers = {} if header is None else {"Retry-After": header}
    client = client_over(
        lambda request: httpx.Response(
            status_code, json={"error": "slow down"}, headers=headers
        )
    )

    with pytest.raises(LLMTransportError) as error:
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )

    assert error.value.retry_after_seconds == expected


async def test_provider_error_does_not_expose_api_key() -> None:
    """Ключ провайдера не попадает в текст ошибки (PRD § 24)."""
    client = client_over(responding({"error": "invalid api key"}, status_code=401))

    with pytest.raises(LLMConfigurationError) as error:
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )

    assert API_KEY not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.ConnectTimeout("таймаут"), id="timeout"),
        pytest.param(httpx.ConnectError("соединение закрыто"), id="connection-error"),
    ],
)
async def test_transport_failure_is_transient(failure: Exception) -> None:
    """Таймаут и обрыв связи — временная неудача, задачу можно повторить (PRD § 16)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    client = client_over(handler)

    with pytest.raises(LLMTransportError):
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"choices": []}, id="no-choices"),
        pytest.param({"choices": [{"message": {}}]}, id="no-content"),
        pytest.param({"model": "test-model"}, id="no-choices-key"),
    ],
)
async def test_response_without_generation_gives_format_error(
    document: dict[str, Any],
) -> None:
    """Ответ провайдера без сгенерированного содержимого — ошибка формата."""
    client = client_over(responding(document))

    with pytest.raises(LLMResponseFormatError):
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"model": ""}, id="model-not-set"),
        pytest.param({"api_key": None}, id="key-not-set"),
        pytest.param({"api_key": "   "}, id="key-blank"),
        pytest.param({"model": "", "api_key": None}, id="nothing-set"),
    ],
)
async def test_unconfigured_llm_does_not_start_calls(overrides: dict[str, Any]) -> None:
    """Без модели или ключа клиент не создаётся и запросов не отправляет (PRD § 23)."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    with pytest.raises(LLMConfigurationError):
        client_over(handler, **overrides)

    assert not calls


async def test_gateway_model_name_is_taken_from_response() -> None:
    """Имя модели берётся из ответа: совместимый шлюз подставляет своё (PRD § 24)."""
    document = completion(as_content(translation_payload()), model="gateway/other-model")
    client = client_over(responding(document))

    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    assert response.model == "gateway/other-model"


def _json(request: httpx.Request) -> dict[str, Any]:
    """Тело отправленного запроса."""
    return json.loads(request.content.decode("utf-8"))
