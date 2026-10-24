"""LLM client over compatible HTTP API: request, response parsing, and error mapping.

There is no actual network access in tests: the provider is mocked via httpx transport,
verifying exactly what the adapter does: request payload, response reading, and port error mapping.
"""

from __future__ import annotations

import asyncio
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
    """Configured LLM: model and key provided."""
    values: dict[str, Any] = {
        "base_url": "https://provider.test/v1",
        "model": "test-model",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return LLMSettings(**values)


def completion(content: str, **overrides: Any) -> dict[str, Any]:
    """Compatible API response wrapping structured content."""
    document: dict[str, Any] = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 340},
    }
    document.update(overrides)
    return document


def client_over(handler: Any, **settings_overrides: Any) -> OpenAICompatibleLLMClient:
    """Client over mocked transport."""
    return OpenAICompatibleLLMClient(
        settings=settings(**settings_overrides),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def responding(document: dict[str, Any], *, status_code: int = 200) -> Any:
    """Transport responding with the same document."""
    return lambda request: httpx.Response(status_code, json=document)


def schema_error_content() -> str:
    """Minimal response with empty TRANSLATE translation from bug reproduction."""
    return as_content({
        "blocks": [{
            "source_block_id": "b1", "treatment": "TRANSLATE",
            "translated_text": "", "glossary_hints": [],
        }],
        "language_notes": [],
    })


async def format_error_from(document: dict[str, Any]) -> LLMResponseFormatError:
    """Produces error via real HTTP client without network calls."""
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
    """A successful call returns parsed DTO alongside call metadata."""
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
    """Chapter analysis request is parsed using the analysis schema."""
    client = client_over(responding(completion(as_content(analysis_payload()))))

    response = await client.generate(
        LLMRequest.for_chapter_analysis(instructions="Анализируй", input_text="block-1: text")
    )

    assert isinstance(response.payload, ChapterAnalysisResponse)
    assert response.payload.planned_block_ids == ("block-1", "block-2")


async def test_request_asks_provider_for_structured_output() -> None:
    """Request contains key, instruction, content, and strict response schema."""
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
    """Unset output limit and temperature are omitted from the request."""
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
    """Token limit response truncation is exposed to the application layer (PRD § 15)."""
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
    """A response truncated by token limit fails parsing and reports the cause."""
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
    """A provider without token data does not invent usage metrics (PRD § 24)."""
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
    """Each failed provider HTTP status maps to its specific port error type (PRD § 16)."""
    client = client_over(responding({"error": {"message": "нет"}}, status_code=status_code))

    with pytest.raises(expected):
        await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )


async def test_payment_required_names_the_balance() -> None:
    """Rejection due to insufficient funds states the cause, not the key or model (PRD § 16)."""
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
    """Pause requested by provider via Retry-After header reaches retry policy."""
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
    """Provider key is not leaked in error message (PRD § 24)."""
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
    """Timeout and connection dropped are transient errors; task can be retried (PRD § 16)."""

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
    """Provider response without generated content is a format error."""
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
    """Without model or key, client is not created and sends no requests (PRD § 23)."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    with pytest.raises(LLMConfigurationError):
        client_over(handler, **overrides)

    assert not calls


async def test_gateway_model_name_is_taken_from_response() -> None:
    """Model name is taken from response: compatible gateway may substitute its own (PRD § 24)."""
    document = completion(as_content(translation_payload()), model="gateway/other-model")
    client = client_over(responding(document))

    response = await client.generate(
        LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
    )

    assert response.model == "gateway/other-model"


def _json(request: httpx.Request) -> dict[str, Any]:
    """Sent request body."""
    return json.loads(request.content.decode("utf-8"))


@pytest.mark.parametrize("status,error_type", [
    (401, LLMConfigurationError), (403, LLMConfigurationError),
    (402, LLMProviderError), (400, LLMProviderError),
    (429, LLMTransportError), (503, LLMTransportError),
])
async def test_provider_auth_error_redacts_echoed_key(status: int, error_type: type) -> None:
    client = client_over(lambda request: httpx.Response(
        status, text=f"rejected {API_KEY}", headers={"Retry-After": "17"},
    ))
    with pytest.raises(error_type) as caught:
        await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="text"))
    assert (API_KEY not in str(caught.value)) is True
    assert "[REDACTED]" in str(caught.value)
    if isinstance(caught.value, LLMTransportError):
        assert caught.value.retry_after_seconds == 17


async def test_redaction_happens_before_detail_truncation() -> None:
    client = client_over(lambda request: httpx.Response(401, text="x" * 290 + API_KEY))
    with pytest.raises(LLMConfigurationError) as caught:
        await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="text"))
    assert "[REDACTED]" in str(caught.value)
    assert (API_KEY[:10] not in str(caught.value)) is True


async def test_transport_error_redacts_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed with {API_KEY}")
    client = client_over(handler)
    with pytest.raises(LLMTransportError) as caught:
        await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="text"))
    assert (API_KEY not in str(caught.value)) is True
    assert "[REDACTED]" in str(caught.value)


@pytest.mark.parametrize("outer", [False, True])
async def test_format_error_diagnostic_redacts_key(outer: bool) -> None:
    content = f'{{"secret": "{API_KEY}"'
    client = client_over(lambda request: httpx.Response(
        200, text=content,
    )) if outer else client_over(responding(completion(
        content, model=f"gateway/{API_KEY}",
        usage={"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.007},
    )))
    with pytest.raises(LLMResponseFormatError) as caught:
        await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="text"))
    error = caught.value
    assert (API_KEY not in str(error)) is True
    assert error.raw_content == content.replace(API_KEY, "[REDACTED]")
    if not outer:
        assert error.model == "gateway/[REDACTED]"
        assert error.usage == TokenUsage(12, 3)
        assert error.cost == Decimal("0.007")


async def test_overlapping_clients_redact_their_own_generation_key() -> None:
    entered = 0
    both_entered = asyncio.Event()

    async def generate(key: str) -> str:
        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            return httpx.Response(401, text=f"rejected {key}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleLLMClient(settings=settings(api_key=key), http_client=http)
            with pytest.raises(LLMConfigurationError) as caught:
                await client.generate(LLMRequest.for_translation(
                    instructions="Переводи", input_text="text",
                ))
        assert (key not in str(caught.value)) is True
        return str(caught.value)

    messages = await asyncio.wait_for(asyncio.gather(
        generate("old-generation-key"), generate("new-generation-key"),
    ), timeout=2)
    assert messages[0] == messages[1]
    assert "[REDACTED]" in messages[0]


async def test_success_diagnostic_redacts_key_without_changing_payload() -> None:
    payload = translation_payload()
    payload["blocks"][0]["translated_text"] = API_KEY
    content = as_content(payload)
    client = client_over(responding(completion(content)))
    response = await client.generate(LLMRequest.for_translation(
        instructions="Переводи", input_text="text",
    ))
    assert response.payload.blocks[0].translated_text == API_KEY
    assert response.raw_content == content.replace(API_KEY, "[REDACTED]")


@pytest.mark.parametrize('kind', ['transport', 'provider', 'format', 'success'])
@pytest.mark.parametrize('mode', ['custom', 'auto'])
async def test_proxy_diagnostics_redact_encoded_decoded_credentials(monkeypatch, kind, mode):
    secret = 'http://alice:p%40ss@localhost:8123'
    text = f'{secret} http://alice:p@ss@localhost:8123 alice:p%40ss alice:p@ss p%40ss p@ss Basic YWxpY2U6cEBzcw=='
    if mode == 'auto':
        monkeypatch.setenv('HTTPS_PROXY', secret)
    def handler(request):
        if kind == 'transport':
            raise httpx.ConnectError(text)
        if kind == 'provider':
            return httpx.Response(403, text=text)
        if kind == 'format':
            return httpx.Response(200, json=completion(text))
        return httpx.Response(200, json=completion(as_content(translation_payload()), model=text))
    client = client_over(handler, proxy_mode=mode, proxy_url=secret if mode == 'custom' else None)
    if mode == 'auto':
        monkeypatch.setenv('HTTPS_PROXY', 'http://different:next-password@localhost:9')
    try:
        if kind == 'success':
            response = await client.generate(LLMRequest.for_translation(instructions='Переводи', input_text='text'))
            output = response.model + response.raw_content
        else:
            with pytest.raises((LLMTransportError, LLMConfigurationError, LLMResponseFormatError)) as caught:
                await client.generate(LLMRequest.for_translation(instructions='Переводи', input_text='text'))
            output = str(caught.value) + (getattr(caught.value, 'raw_content', '') or '')
            assert caught.value.__cause__ is None
        for value in (secret, 'alice', 'p%40ss', 'p@ss', 'YWxpY2U6cEBzcw=='):
            assert value not in output
    finally:
        await client._http.aclose()


def test_proxy_client_preparation_error_is_safe(monkeypatch):
    def fail(**kwargs):
        raise ValueError('http://alice:p%40ss@localhost alice p@ss')
    monkeypatch.setattr(httpx, 'AsyncClient', fail)
    with pytest.raises(LLMConfigurationError) as caught:
        OpenAICompatibleLLMClient(settings=settings(proxy_mode='custom', proxy_url='http://alice:p%40ss@localhost'))
    assert 'alice' not in str(caught.value) and caught.value.__cause__ is None


async def test_candidate_version_selects_schema_and_parser():
    seen = []
    data = analysis_payload()
    data['content_plan'] = {'version': 2, 'profile': 'content-plan-v2', 'difficulties': [], 'before_you_read_evidence': []}
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion(as_content(data)))
    client = client_over(handler)
    try:
        result = await client.generate(LLMRequest.for_chapter_analysis(instructions='Анализ', input_text='Текст', contract_version=2))
        assert result.payload.content_plan.version == 2
        schema = seen[0]['response_format']['json_schema']['schema']
        assert 'content_plan' in schema['required']
    finally:
        await client.aclose()


async def test_unknown_version_makes_zero_http_calls():
    calls = []
    client = client_over(lambda r: calls.append(r))
    try:
        with pytest.raises(LLMResponseFormatError):
            await client.generate(LLMRequest.for_translation(instructions='Перевод', input_text='Текст', contract_version=9))
        assert calls == []
    finally:
        await client.aclose()
