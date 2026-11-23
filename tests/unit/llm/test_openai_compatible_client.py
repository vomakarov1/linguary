"""LLM client over compatible HTTP API: request, response parsing, and error mapping.

There is no actual network access in tests: the provider is mocked via httpx transport,
verifying exactly what the adapter does: request payload, response reading, and port error mapping.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from dataclasses import replace
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
import pytest
from llm_payload_factories import analysis_payload, as_content, translation_payload

from linguary.adapters.config import LLMSettings
from linguary.adapters.llm import OpenAICompatibleLLMClient, openai_compatible
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import (
    LLMConfigurationError,
    LLMContextCapacityError,
    LLMProviderError,
    LLMRequest,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.domain import TokenUsage
from linguary.domain.llm_log import CallContext


async def test_pronunciation_candidate_clients_use_same_contract():
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    candidate = dict(anchor=dict(block_id='b1', start=0, end=4, quote='zorp'), form='zorp',
        context=dict(meaning_in_context='invented action', part_of_speech='verb', grammatical_form='present', is_proper_name=False),
        selection_reason='unpredictable_spelling')
    data = translation_payload(pronunciation_candidates=[candidate])
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(as_content(data)))
    real = client_over(handle)
    fake = FakeLLMClient([FakeReply.of_payload(data)])
    request = LLMRequest.for_translation(instructions='Предложи кандидатов.', input_text='b1: zorp',
                                         pronunciation_candidates_version=1)
    try:
        assert (await real.generate(request)).payload == (await fake.generate(request)).payload
        schema = bodies[0]['response_format']['json_schema']['schema']
        assert 'pronunciation_candidates' in schema['required']
        assert 'pronunciation_candidates' in schema['properties']
    finally:
        await real.aclose()

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
    assert error.__cause__ is None
    assert error.violation.rule_code == "response_schema"
    assert "value_error" in str(error)
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
    assert error.__cause__ is None
    assert error.violation.rule_code == "response_parse"
    assert "json_invalid" in str(error)
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
    assert error.__cause__ is None
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
    assert body["max_completion_tokens"] == 4096
    assert "max_tokens" not in body
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


async def test_context_length_code_maps_to_capacity_error() -> None:
    client = client_over(responding({"error": {"code": "context_length_exceeded", "message": "Too long"}}, status_code=400))
    context = CallContext(str(uuid4()), "translation_unit", "scope", ("target",),
                          str(uuid4()), 0, "test-model", "a" * 64, root_id=str(uuid4()),
                          policy_version="translation-context-v2")
    request = replace(LLMRequest.for_translation(instructions="Переводи", input_text="b1: text"), call_context=context)
    try:
        with pytest.raises(LLMContextCapacityError) as caught:
            await client.generate(request)
        assert isinstance(caught.value, LLMProviderError)
        assert caught.value.provider == settings().provider
        assert caught.value.model == "test-model"
        assert caught.value.call_context == context
        assert caught.value.__cause__ is None
    finally:
        await client._http.aclose()


@pytest.mark.parametrize("document", [
    {"error": {"message": "context_length_exceeded"}},
    {"error": {"message": "Unsupported parameter: max_completion_tokens"}},
    {"error": {"code": "invalid_request", "message": "context_length_exceeded"}},
    {"error": {"type": "context_length_exceeded"}},
    {"code": "context_length_exceeded"},
    {"error": "context_length_exceeded"},
    ["context_length_exceeded"],
])
async def test_unrelated_bad_request_is_not_capacity_error(document) -> None:
    client = client_over(responding(document, status_code=400))
    try:
        with pytest.raises(LLMProviderError) as caught:
            await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="b1: text"))
        assert not isinstance(caught.value, LLMContextCapacityError)
    finally:
        await client._http.aclose()


async def test_non_json_bad_request_is_not_capacity_error() -> None:
    client = client_over(lambda request: httpx.Response(400, text="context_length_exceeded"))
    try:
        with pytest.raises(LLMProviderError) as caught:
            await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="b1: text"))
        assert not isinstance(caught.value, LLMContextCapacityError)
    finally:
        await client._http.aclose()


async def test_capacity_error_redacts_credentials() -> None:
    proxy = "http://alice:p%40ss@localhost:8123"
    client = client_over(responding({"error": {
        "code": "context_length_exceeded",
        "message": f"{API_KEY} {proxy} alice:p@ss p%40ss Basic YWxpY2U6cEBzcw==",
    }}, status_code=400), proxy_mode="custom", proxy_url=proxy)
    try:
        with pytest.raises(LLMContextCapacityError) as caught:
            await client.generate(LLMRequest.for_translation(instructions="Переводи", input_text="b1: text"))
        for secret in (API_KEY, proxy, "alice", "p@ss", "p%40ss", "YWxpY2U6cEBzcw=="):
            assert secret not in str(caught.value)
        assert caught.value.__cause__ is None
    finally:
        await client._http.aclose()


async def test_translation_request_keeps_selected_model_and_response_schema() -> None:
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})

    client = client_over(handler, model="selected-model")
    request = LLMRequest.for_translation(instructions="Saved rules plus accuracy rules", input_text="[b1]\nSource")
    try:
        with pytest.raises(LLMProviderError):
            await client.generate(request)
        assert len(bodies) == 1
        assert bodies[0]["model"] == "selected-model"
        assert bodies[0]["messages"] == [
            {"role": "system", "content": request.instructions},
            {"role": "user", "content": request.input_text},
        ]
        response_format = bodies[0]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "translation_result"
        assert "blocks" in response_format["json_schema"]["schema"]["required"]
    finally:
        await client._http.aclose()


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
    assert error.raw_content == (None if outer else content.replace(API_KEY, "[REDACTED]"))
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
    foreign = 'http://fixture-foreign:not-a-real-password@127.0.0.1:9'
    monkeypatch.setenv('https_proxy', foreign)
    for key in list(os.environ):
        if 'proxy' in key.lower():
            monkeypatch.delenv(key, raising=False)
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
        for value in (secret, 'alice', 'p%40ss', 'p@ss', 'YWxpY2U6cEBzcw==', 'fixture-foreign', 'not-a-real-password'):
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


async def test_quality_total_completion_cap_is_sent_without_legacy_max_tokens():
    from linguary.application.ports.llm import LLMResponseKind
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion(json.dumps(dict(findings=[], checked_artifacts=[], missing_artifacts=[]))))
    client = client_over(handler, max_output_tokens=999)
    try:
        await client.generate(LLMRequest(kind=LLMResponseKind.QUALITY_REVIEW,
            instructions='Review', input_text='{}', max_completion_tokens=123))
        assert len(seen) == 1
        assert seen[0]['max_completion_tokens'] == 123 and 'max_tokens' not in seen[0]
        assert seen[0]['response_format']['json_schema']['strict'] is True
    finally:
        await client.aclose()


@pytest.mark.parametrize('version', [1, 2, 3, 4])
async def test_pronunciation_clients_share_item_isolation(version):
    from linguary.adapters.llm import FakeLLMClient, FakeReply
    from test_llm_payload_mapping import pronunciation_candidate, pronunciation_payload
    valid = pronunciation_candidate()
    data = pronunciation_payload(version, [None, 'invalid', valid | {'ipa': 'FORBIDDEN_PRIVATE'},
        valid | {'context': valid['context'] | {'source': 'FORBIDDEN_PRIVATE'}}, valid])
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(as_content(data)))
    real = client_over(handle)
    fake = FakeLLMClient([FakeReply.of_payload(data)])
    request = LLMRequest.for_translation(instructions='Переводи.', input_text='block-1: zorp',
        contract_version=version, pronunciation_candidates_version=1)
    try:
        response = await real.generate(request)
        assert response.payload == (await fake.generate(request)).payload
        assert len(response.payload.pronunciation_candidates) == 1
        assert response.payload.blocks[0].translated_text
        assert len(bodies) == fake.call_count == 1
        assert 'FORBIDDEN_PRIVATE' not in repr(bodies)
        schema = bodies[0]['response_format']['json_schema']['schema']
        assert schema['$defs']['PronunciationCandidateSchema']['additionalProperties'] is False
    finally:
        await real.aclose()


def test_repair_instructions_sent_as_separate_user_messages() -> None:
    """Repair instructions are sent as separate user messages following input_text, leaving system prompt intact (AC-007)."""
    settings = LLMSettings(
        provider="openai",
        api_key="secret",
        model="gpt-4o",
    )
    request = LLMRequest.for_translation(
        instructions="System instructions: translate accurately.",
        input_text="Source text of the chapter.",
    )
    request = LLMRequest(
        kind=request.kind,
        instructions=request.instructions,
        input_text=request.input_text,
        repair_instructions=("Fix block b1: keep English intact", "Add missing translation for b2"),
    )

    body = openai_compatible._body(request, settings=settings)

    assert len(body["messages"]) == 4
    assert body["messages"][0] == {
        "role": "system",
        "content": "System instructions: translate accurately.",
    }
    assert body["messages"][1] == {
        "role": "user",
        "content": "Source text of the chapter.",
    }
    assert body["messages"][2] == {
        "role": "user",
        "content": "Fix block b1: keep English intact",
    }
    assert body["messages"][3] == {
        "role": "user",
        "content": "Add missing translation for b2",
    }


def test_openai_client_sends_assistant_message_before_user_repair_instruction() -> None:
    """When repair turns include an assistant response, assistant message precedes user repair instruction (AC-003)."""
    settings = LLMSettings(
        provider="openai",
        api_key="secret",
        model="gpt-4o",
    )
    request = LLMRequest.for_translation(
        instructions="System instructions: translate accurately.",
        input_text="Source text of the chapter.",
    )
    request = replace(
        request,
        repair_turns=(
            ('{"broken": true}', "Please fix formatting"),
            (None, "And ensure all blocks are present"),
        ),
    )

    body = openai_compatible._body(request, settings=settings)

    assert body["messages"] == [
        {"role": "system", "content": "System instructions: translate accurately."},
        {"role": "user", "content": "Source text of the chapter."},
        {"role": "assistant", "content": '{"broken": true}'},
        {"role": "user", "content": "Please fix formatting"},
        {"role": "user", "content": "And ensure all blocks are present"},
    ]
    assert OpenAICompatibleLLMClient._build_messages(request) == body["messages"]


def test_global_max_output_tokens_sent_as_max_completion_tokens() -> None:
    cfg = LLMSettings(
        provider="openai",
        api_key="secret",
        model="gpt-4o",
        max_output_tokens=2048,
    )
    request = LLMRequest.for_translation(
        instructions="Translate.",
        input_text="b1: text",
    )

    body = openai_compatible._body(request, settings=cfg)

    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body

    capped_request = replace(request, max_completion_tokens=1024)
    capped_body = openai_compatible._body(capped_request, settings=cfg)

    assert capped_body["max_completion_tokens"] == 1024
    assert "max_tokens" not in capped_body


async def test_response_coverage_error_preserves_truncated_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from linguary.application.correction_dto import ChapterCorrectionResponse, UnchangedBlockResponse
    from linguary.application.ports.llm import LLMResponseKind

    sha = "a" * 64
    review_content = json.dumps({
        "candidates": [{
            "candidate_id": "c1",
            "verdict": "confirmed",
            "error_supported": True,
            "evidence_valid": True,
            "minimal": True,
            "no_semantic_loss": True,
            "reason": "Supported",
        }]
    })
    review_client = client_over(responding(completion(review_content)))
    try:
        review_res = await review_client.generate(
            LLMRequest(
                kind=LLMResponseKind.CHAPTER_CORRECTION_REVIEW,
                instructions="Review",
                input_text="c1",
                ordered_block_ids=("c1",),
                correction_candidate_refs=(("c1", sha),),
            )
        )
        assert review_res.payload.candidates[0].proposal_sha256 == sha
    finally:
        await review_client.aclose()

    monkeypatch.setattr(
        openai_compatible,
        "parse_payload",
        lambda *args, **kwargs: ChapterCorrectionResponse((UnchangedBlockResponse("b1"),)),
    )
    truncated_doc = completion(
        '{"blocks": [{"block_id": "b1", "status": "unchanged"}]}',
        choices=[{
            "message": {"content": '{"blocks": [{"block_id": "b1", "status": "unchanged"}]}'},
            "finish_reason": "length",
        }],
    )
    client = client_over(responding(truncated_doc))
    try:
        with pytest.raises(LLMResponseFormatError) as caught:
            await client.generate(
                LLMRequest(
                    kind=LLMResponseKind.CHAPTER_CORRECTION,
                    instructions="Correct",
                    input_text="b1, b2",
                    ordered_block_ids=("b1", "b2"),
                )
            )
        assert caught.value.truncated is True
        assert caught.value.violation is not None
        assert caught.value.violation.rule_code == "response_coverage"
    finally:
        await client.aclose()


async def test_strict_json_schema_is_cached_across_openai_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from linguary.adapters.llm import json_schema as json_schema_module
    from linguary.adapters.llm.json_schema import strict_json_schema
    from linguary.adapters.llm.payloads import schema_for
    from linguary.application.ports.llm import LLMResponseKind

    model_cls = schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=1)
    if hasattr(json_schema_module, "_cached_strict_json_schema"):
        json_schema_module._cached_strict_json_schema.cache_clear()

    build_count = 0
    orig_filter = json_schema_module._filter_omitted_properties

    def counting_filter(node: Any) -> None:
        nonlocal build_count
        if isinstance(node, dict) and "$defs" in node:
            build_count += 1
        orig_filter(node)

    monkeypatch.setattr(json_schema_module, "_filter_omitted_properties", counting_filter)

    client = client_over(responding(completion(as_content(translation_payload()))))
    try:
        req = LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        await client.generate(req)
        await client.generate(req)
        assert build_count == 1
    finally:
        await client.aclose()

    schema_copy = strict_json_schema(model_cls)
    schema_copy["properties"]["mutated_key"] = {"type": "string"}
    fresh_copy = strict_json_schema(model_cls)
    assert "mutated_key" not in fresh_copy["properties"]


@pytest.mark.parametrize(
    ("status_code", "response_kwargs"),
    [
        pytest.param(413, {"text": f"413 Payload Too Large ({API_KEY})"}, id="413-text"),
        pytest.param(413, {"json": {"error": {"message": "Request entity too large"}}}, id="413-json"),
        pytest.param(400, {"json": {"error": {"code": "context_overflow", "message": "Overflow"}}}, id="code-context-overflow"),
        pytest.param(400, {"json": {"error": {"code": "string_above_max_length", "message": "Too long"}}}, id="code-string-above-max-length"),
        pytest.param(
            400,
            {"json": {"error": {"message": "This model's Maximum Context Length is 128000 tokens."}}},
            id="msg-maximum-context-length",
        ),
        pytest.param(
            400,
            {"json": {"error": {"code": None, "message": "Prompt is too long: 210000 tokens > 200000 maximum"}}},
            id="msg-prompt-is-too-long",
        ),
        pytest.param(
            400,
            {"json": {"error": {"type": "invalid_request_error", "message": "Input tokens exceed the configured limit"}}},
            id="msg-input-tokens-exceed",
        ),
        pytest.param(
            400,
            {"json": {"error": {"message": f"Request exceeds model Context Window ({API_KEY})"}}},
            id="msg-context-window",
        ),
    ],
)
async def test_openai_client_maps_413_and_gateway_400_overflow_to_context_capacity_error(
    status_code: int, response_kwargs: dict[str, Any]
) -> None:
    context = CallContext(
        str(uuid4()),
        "translation_unit",
        "scope",
        ("target",),
        str(uuid4()),
        0,
        "test-model",
        "a" * 64,
        root_id=str(uuid4()),
        policy_version="translation-context-v2",
    )
    request = replace(
        LLMRequest.for_translation(instructions="Переводи", input_text="b1: text"),
        call_context=context,
    )
    client = client_over(lambda req: httpx.Response(status_code, **response_kwargs))
    try:
        with pytest.raises(LLMContextCapacityError) as caught:
            await client.generate(request)
        assert isinstance(caught.value, LLMProviderError)
        assert caught.value.provider == settings().provider
        assert caught.value.model == "test-model"
        assert caught.value.call_context == context
        assert caught.value.__cause__ is None
        assert API_KEY not in str(caught.value)
    finally:
        await client.aclose()


async def test_openai_client_extracts_cached_and_reasoning_tokens() -> None:
    from linguary.domain.errors import LLMCallLogError
    from linguary.domain.llm_log import TokenPricing

    for bad_kwargs in ({"cached_prompt_tokens": -1}, {"reasoning_tokens": -1}):
        with pytest.raises(LLMCallLogError):
            TokenUsage(prompt_tokens=100, completion_tokens=50, **bad_kwargs)

    zero_usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        cached_prompt_tokens=0,
        reasoning_tokens=0,
    )
    assert zero_usage.total_tokens == 150

    pricing = TokenPricing(input_per_mtok=Decimal("2.00"), output_per_mtok=Decimal("8.00"))
    nested_doc = completion(
        as_content(translation_payload()),
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 400,
            "prompt_tokens_details": {"cached_tokens": 600},
            "completion_tokens_details": {"reasoning_tokens": 250},
        },
    )
    client = client_over(responding(nested_doc))
    try:
        res = await client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )
        assert res.usage == TokenUsage(
            prompt_tokens=1000,
            completion_tokens=400,
            cached_prompt_tokens=600,
            reasoning_tokens=250,
        )
        assert res.usage.total_tokens == 1400
        assert pricing.cost_of(res.usage) == (
            Decimal("2.00") * 1000 + Decimal("8.00") * 400
        ) / Decimal(1_000_000)
    finally:
        await client.aclose()

    fallback_doc = completion(
        as_content(translation_payload()),
        usage={
            "prompt_tokens": 500,
            "completion_tokens": 120,
            "cached_tokens": 300,
            "reasoning_tokens": 45,
        },
    )
    fallback_client = client_over(responding(fallback_doc))
    try:
        fallback_res = await fallback_client.generate(
            LLMRequest.for_translation(instructions="Переводи", input_text="block-1: text")
        )
        assert fallback_res.usage == TokenUsage(
            prompt_tokens=500,
            completion_tokens=120,
            cached_prompt_tokens=300,
            reasoning_tokens=45,
        )
        assert fallback_res.usage.total_tokens == 620
    finally:
        await fallback_client.aclose()

    err = await format_error_from(
        completion(
            schema_error_content(),
            usage={
                "prompt_tokens": 800,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 400},
                "completion_tokens_details": {"reasoning_tokens": 90},
            },
        )
    )
    assert err.usage == TokenUsage(
        prompt_tokens=800,
        completion_tokens=200,
        cached_prompt_tokens=400,
        reasoning_tokens=90,
    )


@pytest.mark.parametrize(
    ("base_url", "expected_provider_routing"),
    [
        ("https://openrouter.ai/api/v1", {"require_parameters": True}),
        ("https://api.openrouter.ai/api/v1/", {"require_parameters": True}),
        ("https://api.openai.com/v1", None),
        ("http://localhost:8000/v1", None),
    ],
)
async def test_openrouter_base_url_includes_provider_require_parameters(
    base_url: str, expected_provider_routing: dict[str, Any] | None
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    client = client_over(
        handler,
        base_url=base_url,
        model="deepseek/deepseek-v4.1-flash-1017",
    )
    try:
        await client.generate(
            LLMRequest.for_translation(
                instructions="Translate.",
                input_text="b1: text",
            )
        )
    finally:
        await client.aclose()

    assert len(captured) == 1
    assert _json(captured[0]).get("provider") == expected_provider_routing


async def test_request_body_passes_reasoning_for_openrouter_and_openai() -> None:
    captured_or: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: (captured_or.append(req), httpx.Response(200, json=completion(as_content(translation_payload()))))[1]
        )
    ) as http_or:
        client_or = OpenAICompatibleLLMClient(
            settings=settings(base_url="https://openrouter.ai/api/v1", reasoning_effort="low"),
            http_client=http_or,
        )
        await client_or.generate(LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"))
    assert _json(captured_or[0]).get("reasoning") == {"effort": "low"}
    assert "reasoning_effort" not in _json(captured_or[0])

    captured_tokens: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: (captured_tokens.append(req), httpx.Response(200, json=completion(as_content(translation_payload()))))[1]
        )
    ) as http_tokens:
        client_tokens = OpenAICompatibleLLMClient(
            settings=settings(
                base_url="https://openrouter.ai/api/v1",
                max_output_tokens=16384,
                reasoning_max_tokens=4096,
            ),
            http_client=http_tokens,
        )
        await client_tokens.generate(LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"))
    assert _json(captured_tokens[0]).get("reasoning") == {"max_tokens": 4096}

    captured_oai: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: (captured_oai.append(req), httpx.Response(200, json=completion(as_content(translation_payload()))))[1]
        )
    ) as http_oai:
        client_oai = OpenAICompatibleLLMClient(
            settings=settings(base_url="https://api.openai.com/v1", reasoning_effort="medium"),
            http_client=http_oai,
        )
        await client_oai.generate(LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"))
    assert _json(captured_oai[0]).get("reasoning_effort") == "medium"
    assert "reasoning" not in _json(captured_oai[0])


async def test_reasoning_max_tokens_rejects_non_openrouter_and_exceeding_request_cap() -> None:
    from dataclasses import replace

    non_or_client = client_over(
        responding(completion(as_content(translation_payload()))),
        base_url="https://api.openai.com/v1",
        reasoning_max_tokens=4096,
    )
    with pytest.raises(LLMConfigurationError, match="OpenRouter"):
        await non_or_client.generate(LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"))

    or_client = client_over(
        responding(completion(as_content(translation_payload()))),
        base_url="https://openrouter.ai/api/v1",
        max_output_tokens=16384,
        reasoning_max_tokens=4096,
    )
    capped_request = replace(
        LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"),
        max_completion_tokens=4096,
    )
    with pytest.raises(LLMConfigurationError, match="меньше общего предела"):
        await or_client.generate(capped_request)


@pytest.mark.parametrize("empty_content", [None, "", "   "])
async def test_empty_content_with_length_and_stop_finish_reasons(empty_content: str | None) -> None:
    length_doc = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": empty_content}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 32768},
    }
    with pytest.raises(LLMResponseFormatError) as exc_len:
        await client_over(responding(length_doc)).generate(
            LLMRequest.for_translation(instructions="Translate.", input_text="b1: text")
        )
    assert exc_len.value.truncated is True
    assert exc_len.value.violation is not None
    assert exc_len.value.violation.rule_code == "provider_truncated"
    assert "finish_reason=length" in str(exc_len.value)

    stop_doc = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": empty_content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 800},
    }
    with pytest.raises(LLMResponseFormatError) as exc_stop:
        await client_over(responding(stop_doc)).generate(
            LLMRequest.for_translation(instructions="Translate.", input_text="b1: text")
        )
    assert exc_stop.value.truncated is False
    assert "finish_reason=stop" in str(exc_stop.value)


async def test_overall_asyncio_timeout_aborts_stalled_request_and_preserves_cancellation() -> None:
    class _SlowHttp:
        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(5)
            raise AssertionError("should have timed out")

    client = OpenAICompatibleLLMClient(
        settings=settings(timeout_seconds=0.05),
        http_client=_SlowHttp(),  # type: ignore[arg-type]
    )
    with pytest.raises(LLMTransportError, match="не ответил") as exc:
        await client.generate(LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"))
    assert exc.value.outcome == "unknown"

    slow_client = OpenAICompatibleLLMClient(
        settings=settings(timeout_seconds=5.0),
        http_client=_SlowHttp(),  # type: ignore[arg-type]
    )
    task = asyncio.create_task(
        slow_client.generate(LLMRequest.for_translation(instructions="Translate.", input_text="b1: text"))
    )
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _translation_request() -> LLMRequest:
    return LLMRequest.for_translation(instructions="Переводи", input_text="b1: text")


def _failing(failure: Exception) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure
    return handler


async def _transport_outcome(handler: Any, outcome: str, **overrides: Any) -> LLMTransportError:
    client = client_over(handler, **overrides)
    try:
        with pytest.raises(LLMTransportError) as caught:
            await client.generate(_translation_request())
    finally:
        await client.aclose()
    assert caught.value.outcome == outcome
    return caught.value


async def test_transport_outcome_classification_not_sent_known_unknown() -> None:
    """Connect failures are not_sent unless plain HTTP is proxied; ambiguous sends are unknown."""
    not_sent = (
        httpx.ConnectError("соединение закрыто"),
        httpx.ConnectTimeout("таймаут"),
        httpx.PoolTimeout("пул"),
    )
    for failure in not_sent:
        await _transport_outcome(_failing(failure), "not_sent")
        await _transport_outcome(
            _failing(failure), "not_sent", base_url="http://provider.test/v1", proxy_mode="direct",
        )
        await _transport_outcome(
            _failing(failure), "unknown", base_url="http://provider.test/v1", proxy_mode="auto",
        )
    await _transport_outcome(
        _failing(httpx.ConnectError("прокси")),
        "unknown",
        base_url="http://provider.test/v1",
        proxy_mode="custom",
        proxy_url="http://127.0.0.1:9",
    )
    await _transport_outcome(
        _failing(httpx.ConnectError("tls")),
        "not_sent",
        base_url="https://provider.test/v1",
        proxy_mode="custom",
        proxy_url="http://127.0.0.1:9",
    )
    for failure in (
        httpx.ReadTimeout("чтение"),
        httpx.WriteTimeout("запись"),
        httpx.RemoteProtocolError("протокол"),
        httpx.WriteError("обрыв записи"),
        httpx.ReadError("обрыв чтения"),
        httpx.CloseError("закрытие"),
        httpx.DecodingError("декодирование"),
    ):
        await _transport_outcome(_failing(failure), "unknown")
    for status in (429, 500, 503):
        await _transport_outcome(responding({"error": {"message": "нет"}}, status_code=status), "known_transient")
    for status in (502, 504, 524, 501):
        await _transport_outcome(responding({"error": {"message": "нет"}}, status_code=status), "unknown")


async def test_length_truncation_preserved_on_null_content_and_dto_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    null_content = await format_error_from(completion(
        "",
        choices=[{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
    ))
    assert null_content.truncated is True
    assert null_content.violation is not None
    assert null_content.violation.rule_code == "provider_truncated"

    empty_content = await format_error_from(completion(
        "",
        choices=[{"message": {"content": ""}, "finish_reason": "length"}],
    ))
    assert empty_content.truncated is True
    assert empty_content.violation is not None
    assert empty_content.violation.rule_code == "provider_truncated"

    bad_evidence = translation_payload(content_evidence=[{"difficulty_id": ""}])
    evidence_error = await format_error_from(completion(
        as_content(bad_evidence),
        choices=[{"message": {"content": as_content(bad_evidence)}, "finish_reason": "length"}],
    ))
    assert evidence_error.truncated is True
    assert evidence_error.violation is not None
    assert evidence_error.violation.rule_code == "provider_truncated"

    analysis = analysis_payload()
    analysis["content_plan"] = {
        "version": 2,
        "profile": "content-plan-v2",
        "difficulties": [],
        "before_you_read_evidence": [{"difficulty_id": ""}],
    }
    analysis_client = client_over(responding(completion(
        as_content(analysis),
        choices=[{"message": {"content": as_content(analysis)}, "finish_reason": "length"}],
    )))
    try:
        with pytest.raises(LLMResponseFormatError) as caught:
            await analysis_client.generate(LLMRequest.for_chapter_analysis(
                instructions="Анализ", input_text="Текст", contract_version=2,
            ))
    finally:
        await analysis_client.aclose()
    assert caught.value.truncated is True
    assert caught.value.violation is not None
    assert caught.value.violation.rule_code == "provider_truncated"

    def reject_dto(validated: Any) -> Any:
        raise ValueError("dto")

    monkeypatch.setattr("linguary.adapters.llm.payloads._to_dto", reject_dto)
    dto_error = await format_error_from(completion(
        as_content(translation_payload()),
        choices=[{"message": {"content": as_content(translation_payload())}, "finish_reason": "length"}],
    ))
    assert dto_error.truncated is True
    assert dto_error.violation is not None
    assert dto_error.violation.rule_code == "provider_truncated"

    non_string = await format_error_from(completion(
        "",
        choices=[{"message": {"role": "assistant", "content": ["part"]}, "finish_reason": "length"}],
    ))
    assert non_string.truncated is True
    assert non_string.violation is not None
    assert non_string.violation.rule_code == "provider_truncated"


async def test_message_refusal_and_content_filter_detected_for_translation() -> None:
    content = as_content(translation_payload())
    requests = (
        ("translation_result", _translation_request()),
        ("chapter_analysis", LLMRequest.for_chapter_analysis(instructions="Анализ", input_text="Текст")),
    )
    for stage, request in requests:
        refusal_client = client_over(responding(completion(
            content,
            choices=[{
                "message": {"role": "assistant", "content": content, "refusal": "I cannot complete this request."},
                "finish_reason": "length",
            }],
        )))
        try:
            with pytest.raises(LLMResponseFormatError) as refusal:
                await refusal_client.generate(request)
        finally:
            await refusal_client.aclose()
        assert refusal.value.truncated is False
        assert refusal.value.violation is not None
        assert refusal.value.violation.rule_code == "provider_refusal"
        assert refusal.value.violation.stage == stage

        filtered_client = client_over(responding(completion(
            content,
            choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "content_filter"}],
        )))
        try:
            with pytest.raises(LLMResponseFormatError) as filtered:
                await filtered_client.generate(request)
        finally:
            await filtered_client.aclose()
        assert filtered.value.truncated is False
        assert filtered.value.violation is not None
        assert filtered.value.violation.rule_code == "provider_content_filter"
        assert filtered.value.violation.stage == stage


async def test_openrouter_200_error_and_coded_context_overflow() -> None:
    context = CallContext(
        str(uuid4()), "translation_unit", "scope", ("target",), str(uuid4()), 0, "test-model", "a" * 64,
        root_id=str(uuid4()), policy_version="translation-context-v2",
    )
    request = replace(_translation_request(), call_context=context)

    async def captured(document: dict[str, Any], *, status_code: int = 200) -> Exception:
        client = client_over(
            responding(document, status_code=status_code),
            base_url="https://openrouter.ai/api/v1",
        )
        try:
            with pytest.raises(Exception) as caught:
                await client.generate(request)
        finally:
            await client.aclose()
        return caught.value

    bad_gateway = await captured({"choices": [], "error": {"code": 502, "message": "bad gateway"}})
    assert isinstance(bad_gateway, LLMTransportError)
    assert bad_gateway.outcome == "unknown"

    unavailable = await captured({"error": {"code": 503, "message": "overloaded"}})
    assert isinstance(unavailable, LLMTransportError)
    assert unavailable.outcome == "known_transient"

    limited = await captured({"error": {"code": 429, "message": "slow down"}})
    assert isinstance(limited, LLMTransportError)
    assert limited.outcome == "known_transient"

    internal = await captured({"error": {"code": 500, "message": "internal"}})
    assert isinstance(internal, LLMTransportError)
    assert internal.outcome == "known_transient"

    cloudflare = await captured({"choices": None, "error": {"code": 524, "message": "timeout"}})
    assert isinstance(cloudflare, LLMTransportError)
    assert cloudflare.outcome == "unknown"

    gateway_timeout = await captured({"error": {"code": "504", "message": "gateway timeout"}})
    assert isinstance(gateway_timeout, LLMTransportError)
    assert gateway_timeout.outcome == "unknown"

    rejected_key = await captured({"error": {"code": 401, "message": f"bad key {API_KEY}"}})
    assert isinstance(rejected_key, LLMConfigurationError)
    assert API_KEY not in str(rejected_key)

    coded = await captured(
        {"error": {"code": "invalid_request", "message": "context_length_exceeded: maximum context length exceeded"}},
        status_code=400,
    )
    assert isinstance(coded, LLMContextCapacityError)
    assert coded.provider == settings().provider
    assert coded.model == "test-model"
    assert coded.call_context == context
    assert coded.__cause__ is None

    embedded = await captured({
        "error": {"code": "invalid_request", "message": "This model's maximum context length is 128000 tokens."},
    })
    assert isinstance(embedded, LLMContextCapacityError)
    assert embedded.call_context == context
    assert embedded.provider == settings().provider
    assert embedded.model == "test-model"


@pytest.mark.parametrize("header", ["inf", "+inf", "Infinity", "nan", "1e309"])
async def test_retry_after_rejects_infinity(header: str) -> None:
    client = client_over(lambda request: httpx.Response(
        429, json={"error": "slow down"}, headers={"Retry-After": header},
    ))
    try:
        with pytest.raises(LLMTransportError) as caught:
            await client.generate(_translation_request())
    finally:
        await client.aclose()
    assert caught.value.retry_after_seconds is None
    assert caught.value.outcome == "known_transient"


async def test_real_httpx_pool_timeout_wins_over_hard_asyncio_timeout() -> None:
    """Real httpx.PoolTimeout fires before the outer asyncio.timeout and classifies as not_sent (or unknown via plain HTTP proxy)."""
    hold = asyncio.Event()
    accepted = asyncio.Event()

    async def handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.set()
        try:
            await reader.read(1024)
            await hold.wait()
        finally:
            writer.close()

    server = await asyncio.start_server(handle_conn, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1)) as http:
            occupier = asyncio.create_task(http.get(f"http://127.0.0.1:{port}/hold", timeout=5.0))
            await asyncio.wait_for(accepted.wait(), timeout=2.0)
            direct_client = OpenAICompatibleLLMClient(
                settings=settings(base_url=base_url, timeout_seconds=0.1, proxy_mode="direct"),
                http_client=http,
            )
            with pytest.raises(LLMTransportError) as direct_err:
                await direct_client.generate(_translation_request())
            assert direct_err.value.outcome == "not_sent"

            proxied_client = OpenAICompatibleLLMClient(
                settings=settings(
                    base_url=base_url,
                    timeout_seconds=0.1,
                    proxy_mode="custom",
                    proxy_url="http://127.0.0.1:9",
                ),
                http_client=http,
            )
            with pytest.raises(LLMTransportError) as proxied_err:
                await proxied_client.generate(_translation_request())
            assert proxied_err.value.outcome == "unknown"

            hold.set()
            occupier.cancel()
            await asyncio.gather(occupier, return_exceptions=True)
    finally:
        server.close()
        await server.wait_closed()


async def test_embedded_gateway_errors_with_choices_retry_after_and_non_ascii_code() -> None:
    """HTTP 200 with finish_reason='error', choices[0].error, Retry-After, or non-ASCII code is classified safely."""
    bad_gw_doc = {
        "choices": [{"finish_reason": "error", "message": {"content": ""}}],
        "error": {"code": 502, "message": "upstream bad gateway"},
    }
    client = client_over(responding(bad_gw_doc))
    try:
        with pytest.raises(LLMTransportError) as bad_gw:
            await client.generate(_translation_request())
        assert bad_gw.value.outcome == "unknown"
    finally:
        await client.aclose()

    finish_error_only = {
        "choices": [{"finish_reason": "error", "message": {"content": ""}}],
    }
    client = client_over(responding(finish_error_only))
    try:
        with pytest.raises(LLMTransportError) as finish_err:
            await client.generate(_translation_request())
        assert finish_err.value.outcome == "unknown"
    finally:
        await client.aclose()

    choice_overflow = {
        "error": None,
        "choices": [
            {
                "finish_reason": "error",
                "message": {"content": ""},
                "error": {"code": 400, "message": "This model's maximum context length is 128000"},
            }
        ],
    }
    client = client_over(responding(choice_overflow))
    try:
        with pytest.raises(LLMContextCapacityError):
            await client.generate(_translation_request())
    finally:
        await client.aclose()

    rate_limited_doc = {"error": {"code": 429, "message": "slow down"}}
    client = client_over(
        lambda req: httpx.Response(200, json=rate_limited_doc, headers={"Retry-After": "7"})
    )
    try:
        with pytest.raises(LLMTransportError) as limited:
            await client.generate(_translation_request())
        assert limited.value.outcome == "known_transient"
        assert limited.value.retry_after_seconds == 7.0
    finally:
        await client.aclose()

    unicode_code_doc = {"error": {"code": "²", "message": "weird code"}}
    client = client_over(responding(unicode_code_doc))
    try:
        with pytest.raises(LLMProviderError):
            await client.generate(_translation_request())
    finally:
        await client.aclose()


async def test_embedded_gateway_errors_on_gzip_compressed_200_responses() -> None:
    """Compressed HTTP 200 responses with embedded errors do not trigger httpx.DecodingError."""
    import gzip

    rate_limited_body = gzip.compress(json.dumps({"error": {"code": 429, "message": "slow down"}}).encode("utf-8"))
    client = client_over(
        lambda req: httpx.Response(
            200,
            content=rate_limited_body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Retry-After": "7",
            },
        )
    )
    try:
        with pytest.raises(LLMTransportError) as limited:
            await client.generate(_translation_request())
        assert limited.value.outcome == "known_transient"
        assert limited.value.retry_after_seconds == 7.0
    finally:
        await client.aclose()

    overflow_body = gzip.compress(
        json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "error",
                        "message": {"content": ""},
                        "error": {"code": 400, "message": "This model's maximum context length is 128000"},
                    }
                ]
            }
        ).encode("utf-8")
    )
    client = client_over(
        lambda req: httpx.Response(
            200,
            content=overflow_body,
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
        )
    )
    try:
        with pytest.raises(LLMContextCapacityError):
            await client.generate(_translation_request())
    finally:
        await client.aclose()


@pytest.mark.parametrize("model", ["anthropic/claude-haiku-5.5", "claude-sonnet-4.6"])
async def test_openrouter_claude_enables_explicit_prompt_cache_on_system_and_long_or_repair_user(
    model: str,
) -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    client = client_over(
        handler,
        base_url="https://openrouter.ai/api/v1",
        model=model,
    )
    try:
        short_req = LLMRequest.for_translation(instructions="Translate.", input_text="b1: short text")
        await client.generate(short_req)

        long_input = "x" * 1024
        long_req = LLMRequest.for_translation(instructions="Translate.", input_text=long_input)
        await client.generate(long_req)

        repair_req = replace(
            short_req,
            repair_turns=(('{"broken": true}', "Fix formatting"),),
        )
        await client.generate(repair_req)
    finally:
        await client.aclose()

    assert captured[0]["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Translate.", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": "b1: short text"},
    ]
    assert captured[1]["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Translate.", "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": long_input, "cache_control": {"type": "ephemeral"}}],
        },
    ]
    assert captured[2]["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Translate.", "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "b1: short text", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "assistant", "content": '{"broken": true}'},
        {"role": "user", "content": "Fix formatting"},
    ]


@pytest.mark.parametrize(
    ("base_url", "model", "allow_cache"),
    [
        ("https://openrouter.ai/api/v1", "deepseek/deepseek-v4.1-flash-1017", True),
        ("https://api.openai.com/v1", "anthropic/claude-haiku-5.5", True),
        ("https://api.anthropic.com/v1", "claude-sonnet-4.6", True),
        ("https://openrouter.ai/api/v1", "anthropic/claude-haiku-5.5", False),
    ],
)
async def test_explicit_prompt_cache_omitted_for_non_claude_non_openrouter_or_disabled(
    base_url: str, model: str, allow_cache: bool
) -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=completion(as_content(translation_payload())))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatibleLLMClient(
            settings=settings(base_url=base_url, model=model),
            http_client=http,
            allow_explicit_prompt_cache=allow_cache,
        )
        req = replace(
            LLMRequest.for_translation(instructions="Translate.", input_text="x" * 1024),
            repair_instructions=("Fix it",),
        )
        await client.generate(req)

    assert len(captured) == 1
    assert all(isinstance(msg["content"], str) for msg in captured[0]["messages"])





