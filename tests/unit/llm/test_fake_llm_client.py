"""LLM client for tests: determinism, scenarios, and request recording."""

from __future__ import annotations

import pytest
from llm_payload_factories import analysis_payload, block_payload, translation_payload

from linguary.adapters.llm import FakeLLMClient, FakeReply, FakeScenarioError
from linguary.application.llm_dto import ChapterAnalysisResponse, TranslationResultResponse
from linguary.application.ports.llm import (
    LLMRequest,
    LLMResponseFormatError,
    LLMTransportError,
)
from linguary.domain import TokenUsage


def translation_request() -> LLMRequest[TranslationResultResponse]:
    """Block group processing request."""
    return LLMRequest.for_translation(
        instructions="Переводи", input_text="block-1: text"
    )


async def test_prepared_reply_is_parsed_like_a_provider_answer() -> None:
    """A pre-arranged reply is parsed in the same manner as a provider response."""
    client = FakeLLMClient.always(FakeReply.of_payload(translation_payload()))

    response = await client.generate(translation_request())

    assert isinstance(response.payload, TranslationResultResponse)
    assert response.payload.block_ids == ("block-1", "block-2")
    assert response.provider == "fake"


async def test_replies_are_returned_in_order() -> None:
    """The scenario replies with the specified sequence rather than random responses."""
    client = FakeLLMClient.replying(
        [
            FakeReply.of_payload(translation_payload(language_notes=[])),
            FakeReply.of_payload(
                translation_payload(blocks=[block_payload()], language_notes=[])
            ),
        ]
    )

    first = await client.generate(translation_request())
    second = await client.generate(translation_request())

    assert isinstance(first.payload, TranslationResultResponse)
    assert isinstance(second.payload, TranslationResultResponse)
    assert first.payload.block_ids == ("block-1", "block-2")
    assert second.payload.block_ids == ("block-1",)


async def test_same_scenario_gives_the_same_result_twice() -> None:
    """The same scenario yields identical results: responses are deterministic."""
    payload = translation_payload()

    async def run() -> tuple[str, ...]:
        client = FakeLLMClient.always(FakeReply.of_payload(payload))
        response = await client.generate(translation_request())
        assert isinstance(response.payload, TranslationResultResponse)
        return response.payload.block_ids

    assert await run() == await run()


async def test_requests_are_recorded() -> None:
    """Outbound requests are recorded to verify context composition."""
    client = FakeLLMClient.always(FakeReply.of_payload(analysis_payload()))

    await client.generate(
        LLMRequest.for_chapter_analysis(instructions="Анализируй", input_text="глава")
    )

    assert client.call_count == 1
    assert client.last_request.instructions == "Анализируй"
    assert client.last_request.input_text == "глава"


async def test_analysis_scenario_returns_analysis_payload() -> None:
    """A chapter analysis scenario returns an analysis DTO."""
    client = FakeLLMClient.always(FakeReply.of_payload(analysis_payload()))

    response = await client.generate(
        LLMRequest.for_chapter_analysis(instructions="Анализируй", input_text="глава")
    )

    assert isinstance(response.payload, ChapterAnalysisResponse)
    assert response.payload.planned_block_ids == ("block-1", "block-2")


async def test_invalid_json_scenario_gives_format_error() -> None:
    """An invalid response scenario reproduces a format error."""
    client = FakeLLMClient.always(FakeReply(content="сейчас переведу"))

    with pytest.raises(LLMResponseFormatError):
        await client.generate(translation_request())


async def test_truncated_scenario_reports_token_limit_stop() -> None:
    """A token limit truncation scenario is reflected in the client response."""
    client = FakeLLMClient.always(
        FakeReply.of_payload(translation_payload(), truncated=True)
    )

    response = await client.generate(translation_request())

    assert response.truncated is True


async def test_mismatched_blocks_scenario_is_delivered_to_application() -> None:
    """The client passes responses for unexpected blocks as-is: the application rejects them."""
    client = FakeLLMClient.always(
        FakeReply.of_payload(
            translation_payload(
                blocks=[block_payload(source_block_id="block-99")], language_notes=[]
            )
        )
    )

    response = await client.generate(translation_request())

    assert isinstance(response.payload, TranslationResultResponse)
    assert response.payload.block_ids == ("block-99",)


async def test_failure_scenario_raises_prepared_error() -> None:
    """A failure scenario raises the configured port error."""
    client = FakeLLMClient.replying(
        [LLMTransportError("провайдер недоступен"), FakeReply.of_payload(translation_payload())]
    )

    with pytest.raises(LLMTransportError):
        await client.generate(translation_request())
    response = await client.generate(translation_request())

    assert response.truncated is False
    assert client.call_count == 2


async def test_usage_is_reported_when_scenario_sets_it() -> None:
    """Configured token usage appears in the client response (PRD § 24)."""
    client = FakeLLMClient.always(
        FakeReply.of_payload(
            translation_payload(), usage=TokenUsage(prompt_tokens=10, completion_tokens=20)
        )
    )

    response = await client.generate(translation_request())

    assert response.usage is not None
    assert response.usage.total_tokens == 30


async def test_exhausted_scenario_is_reported() -> None:
    """Exhausted responses are reported rather than returning arbitrary data."""
    client = FakeLLMClient.replying([FakeReply.of_payload(translation_payload())])

    await client.generate(translation_request())

    with pytest.raises(FakeScenarioError):
        await client.generate(translation_request())


async def test_client_without_requests_has_no_last_request() -> None:
    """Accessing the last request before any call is made is a scenario configuration error."""
    client = FakeLLMClient()

    with pytest.raises(FakeScenarioError):
        _ = client.last_request
